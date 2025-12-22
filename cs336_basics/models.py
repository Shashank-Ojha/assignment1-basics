import torch
import torch.nn as nn

import math
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        lo = -3 * std
        hi = 3 * std
        torch.nn.init.trunc_normal_(W, mean=0, std=std, a=lo, b=hi)

        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (..., in_features)
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        embedding_table = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embedding_table, mean=0, std=1, a=-3, b=3)
        self.embedding_table = nn.Parameter(embedding_table)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.randn(d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Shape (batch_size, seq_len, 1)
        sum_of_sq = torch.sum(torch.square(x), axis=-1, keepdim=True)

        # Shape (batch_size, seq_len, 1)
        rms = torch.sqrt(sum_of_sq / self.d_model + self.eps)

        result = x * self.gain / rms

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device)
        self.w2 = Linear(d_ff, d_model, device=device)
        self.w3 = Linear(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (..., d_model)
        lhs = self.w1(x)
        lhs = lhs * torch.sigmoid(lhs)

        rhs = self.w3(x)

        return self.w2(lhs * rhs)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FFSiLU(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, 4 * d_model, device=device)
        self.w2 = Linear(4 * d_model, d_model, device=device)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (..., d_model)
        return self.w2(self.silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # Shape (max_seq_len, )
        i = torch.arange(max_seq_len, device=device)

        # Shape (d_model / 2, )
        denominator = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        # Shape (max_seq_len, d_model / 2)
        angle = torch.outer(i, denominator)

        # Shape (max_seq_len, d_model / 2)
        # This basically takes every theta and represents it in the complex coordinate system as
        # cos(theta) + i sin(theta). The first parameter of torch.polar controls the absolute value:
        # abs⋅cos(angle)+abs⋅sin(angle)⋅i which we set to 1.
        rotary_complex_coordinates = torch.polar(torch.ones_like(angle, device=device), angle)

        # We do persistent = False so these matrices aren't stored in the state_dict when
        # saved/loaded. These matrices are cheap to compute and are fully deterministic (independent of training)
        # so there is no need to pollute the saved dict with all this extra memory.
        self.register_buffer("rotary_complex_coordinates", rotary_complex_coordinates, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., seq_len, d_k)
            token_positions (..., seq_len)

        Returns:
            out: shape (..., seq_len, d_k)
        """
        # Input tensor has shape (..., seq_len, d_k). Returns tensor of same shape.
        assert x.shape[-1] % 2 == 0, "d_k is not divisble by 2."

        # Shape = (..., seq_len, d_k / 2, 2)
        # We upcast to float (float32) for numerical stability.
        x_in_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

        # Shape = (..., seq_len, d_k / 2)
        x_in_complex = torch.view_as_complex(x_in_pairs)

        # Shape = (seq_len, d_k / 2)
        seq_len = x_in_complex.shape[-2]
        rotary_complex_coordinates = self.rotary_complex_coordinates[token_positions, :]

        # Applies rotation by doing element wise multiplication of complex numbers. Specifically the
        # math becomes:
        #
        #   (cos(theta) + i * sin(theta)) * (x + i * y)
        #    = x * cos(theta) + i * x * sin(theta) + y * cos(theta) + i^2 * y * sin(theta)
        #
        #    If we now substitute i^2 = -1 and collect the real and imaginary terms, we get
        #
        #           real = x * cos(theta) - y * sin(theta)
        #           imag = x * sin(theta) + y * cos(theta)
        #
        rotated = rotary_complex_coordinates * x_in_complex

        # Shape = (..., seq_len, d_k / 2, 2)
        # This is basically the inverse of view as complex which expands each complex number into the real and
        # imag parts in the last dimension
        rotated_real = torch.view_as_real(rotated)

        # Shape = (..., seq_len, d_k)
        rotated_real_original_shape = rotated_real.flatten(-2)

        return rotated_real_original_shape.type_as(x)


def softmax(tensor: torch.Tensor, dim):
    """Normalizes the tensor along the dim provided such that values in that dim following a distribution"""

    max_values, max_indices = torch.max(tensor, dim=dim, keepdims=True)
    shifted = tensor - max_values
    exp_shifted = torch.exp(shifted)

    sums = torch.sum(exp_shifted, dim=dim, keepdims=True)
    return exp_shifted / sums


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    d_k = q.shape[-1]

    attention_matrix = einsum(q, k, "b ... l_q d_k, b ... l_k d_k -> b ... l_q l_k") / math.sqrt(d_k)

    if mask is not None:
        attention_matrix = torch.where(
            mask, attention_matrix, float("-inf") * torch.ones_like(attention_matrix, device=q.device)
        )

    normalized_softmax = softmax(attention_matrix, -1)

    return einsum(normalized_softmax, v, "b ... l_q l_k, b ... l_k d_v -> b ... l_q d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None, device=None):
        super().__init__()

        assert d_model % num_heads == 0, "The number of heads must evenly divide d_model."

        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = Linear(d_model, d_model, device=device)
        self.Wk = Linear(d_model, d_model, device=device)
        self.Wv = Linear(d_model, d_model, device=device)

        self.Wo = Linear(d_model, d_model, device=device)

        self.rope = None
        if theta:
            d_k = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.full((max_seq_len, max_seq_len), True, dtype=torch.bool, device=device)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        # X has shape (..., seq_len, d_model)

        # (..., seq_len, d_model) --> (..., seq_len, num_heads, d_model // num_heads) --> (..., num_heads, seq_len, d_model // num_heads)
        q = rearrange(
            self.Wq(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )
        k = rearrange(
            self.Wk(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )
        v = rearrange(
            self.Wv(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads
        )

        if self.rope is not None:
            if token_positions is None:
                # (..., seq_len) + (seq_len,)
                token_positions = torch.zeros(q.shape[:-1], dtype=int, device=q.device) + torch.arange(
                    0, q.shape[-2], device=q.device
                )

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        attn = scaled_dot_product_attention(q, k, v, causal_mask)

        attn = rearrange(attn, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")

        out = self.Wo(attn)

        return out


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.eps = 1e-5

        self.rms_norm1 = RMSNorm(d_model, self.eps, device=device)
        # TODO: This can be more efficient if you pass in the RoPE layer as an argument and reuse.
        self.mh_attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)

        self.rms_norm2 = RMSNorm(d_model, self.eps, device=device)
        self.ff = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x):
        x = x + self.mh_attn(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.eps = 1e-5
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, d_model, device=device)

        self.blocks = torch.nn.ModuleList(
            [Block(d_model, num_heads, d_ff, context_length, rope_theta, device=device) for _ in range(num_layers)]
        )

        self.ln = RMSNorm(d_model, self.eps, device=device)
        self.proj = Linear(d_model, vocab_size, device=device)

    def forward(self, x):
        out = self.embedding(x)

        for i in range(self.num_layers):
            out = self.blocks[i](out)

        out = self.ln(out)
        out = self.proj(out)

        return out
