from dataclasses import dataclass


@dataclass
class LLM_Params:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: int


@dataclass
class Optimizer_Params:
    # Cosine Schedule params.
    min_lr: float
    max_lr: float
    warmup_iters: int
    total_iters: int

    # Optimizer params.
    betas: tuple[float, float]
    weight_decay: float
    eps: float

    # max norm for gradient clipping.
    max_norm: float
