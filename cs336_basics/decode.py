"""Full training logic for the LLM"""

import argparse
import torch


from cs336_basics.param_defs import LLM_Params
from cs336_basics.models import Transformer, softmax
from cs336_basics.loss import AdamW
from cs336_basics.utils import load_checkpoint, get_checkpoint_path
from cs336_basics.tokenization.tokenize_helpers import get_tokenizer
from cs336_basics.tokenization.constants import SPEICAL_TOKEN_EOT
from cs336_basics.experiments import EXPERIMENTAL_CONFIGS


LLM_MINI_PARAMS = LLM_Params(
    vocab_size=-1,  # set dynamically in the code since it varies based on the dataset
    context_length=64,
    num_layers=4,
    d_model=32,
    num_heads=4,
    d_ff=4 * 32,
    rope_theta=10000,
)

DUMMY_VALUE = 1.0


def top_p_sampling(probs, top_p):
    """
    Args:
        probs: (B, vocab_size)
        top_p: float

    Returns:
        probs: (B, vocab_size) -- retains only top tokens cumulatively within top_p probability
    """
    # Sort probabilities and corresponding indices
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    # Compute cumulative probability along sorted tokens
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for entries where cumulative probability exceeds top_p
    # Note: We want to keep the *minimal* set of tokens whose cumulative prob >= top_p
    # So we shift right by 1 and set the first token always to keep
    # This way, the first time cum_probs > top_p, we keep up to (and including) that token
    shifted_cum_probs = torch.cat([torch.zeros_like(cum_probs[:, :1]), cum_probs[:, :-1]], dim=-1)
    # Mask: True for tokens to keep (where previous cumulative < top_p)
    mask = shifted_cum_probs < top_p

    # Zero out probabilities not in the mask
    filtered_sorted_probs = sorted_probs * mask

    # Map back to original order
    # We need to scatter the filtered_sorted_probs to their original positions
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_probs)

    # Renormalize so sum to 1
    filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)
    return filtered_probs


def generate_text(
    initial_context: torch.Tensor,
    model,
    context_length: int,
    max_len: int,
    temperature: float,
    top_p: float,
    end_of_text_token_id: int,
):
    # (B, T)
    context = initial_context
    for i in range(max_len):
        sub_context = context[:, -context_length:]

        # (B, T, vocab_size)
        logits = model(sub_context)

        # (B, vocab_size)
        last_logits = logits[:, -1, :]
        probs = softmax(last_logits / temperature, dim=-1)

        if top_p < 1.0:
            probs = top_p_sampling(probs, top_p)

        # (B, 1)
        pred = torch.multinomial(probs, num_samples=1)

        # (B, T+1)
        context = torch.cat([context, pred], dim=1)

        if pred.item() == end_of_text_token_id:
            break

    return context


def load_model(llm_params, checkpoint_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        vocab_size=llm_params.vocab_size,
        context_length=llm_params.context_length,
        num_layers=llm_params.num_layers,
        d_model=llm_params.d_model,
        num_heads=llm_params.num_heads,
        d_ff=llm_params.d_ff,
        rope_theta=llm_params.rope_theta,
        device=device,
    )

    # Create dummy optimizer for checkpoint loading. We aren't trianing, so this doesn't really matter.
    opt = AdamW(model.parameters(), DUMMY_VALUE, (DUMMY_VALUE, DUMMY_VALUE), DUMMY_VALUE, DUMMY_VALUE)

    checkpoint_path = get_checkpoint_path(checkpoint_file)
    load_checkpoint(checkpoint_path, model, opt)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train LLM with given parameters.")
    parser.add_argument(
        "--tokenizer_version",
        type=str,
        help="Tokenizer (ts or owt)",
    )

    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="File to save checkpoints (.pt)",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="File to save checkpoints (.pt)",
    )

    parser.add_argument(
        "--initial_text",
        type=str,
        default="\n",
        help="Initial text to generate from",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for sampling",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config settings to use from experiments.py",
    )

    args = parser.parse_args()

    llm_params, _ = EXPERIMENTAL_CONFIGS[args.config]

    tokenizer = get_tokenizer(args.tokenizer_version)
    vocab_size = len(tokenizer.vocab)
    llm_params.vocab_size = vocab_size

    model = load_model(llm_params, args.checkpoint_file)

    end_of_text_token_id = tokenizer.encode(SPEICAL_TOKEN_EOT)
    assert len(end_of_text_token_id) == 1
    end_of_text_token_id = end_of_text_token_id[0]

    # Shape (B, T). In this case, just (1, 1)
    initial_text = torch.tensor(tokenizer.encode(args.initial_text)).unsqueeze(0)
    generated_tokens = generate_text(
        initial_text,
        model,
        llm_params.context_length,
        args.max_len,
        args.temperature,
        args.top_p,
        end_of_text_token_id,
    )
    text = tokenizer.decode(generated_tokens.flatten().numpy())

    print(text)


if __name__ == "__main__":
    main()
