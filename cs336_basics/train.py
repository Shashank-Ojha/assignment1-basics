"""Full training logic for the LLM"""

import argparse
import torch

from cs336_basics.param_defs import LLM_Params, Optimizer_Params
from cs336_basics.models import Transformer
from cs336_basics.loss import AdamW, cross_entropy_loss, clip_gradient, get_lr_cosine_schedule
from cs336_basics.utils import (
    get_batch,
    save_checkpoint,
    create_checkpoints_folder,
    get_checkpoint_path,
    CHECKPOINTS_FOLDER,
    load_data,
)

import wandb
import time

LLM_MINI_PARAMS = LLM_Params(
    vocab_size=-1,  # set dynamically in the code since it varies based on the dataset
    context_length=64,
    num_layers=4,
    d_model=32,
    num_heads=4,
    d_ff=4 * 32,
    rope_theta=10000,
)
OPT_PARAMS = Optimizer_Params(
    min_lr=1e-2,
    max_lr=1e-1,
    warmup_iters=1000,
    total_iters=10000,
    betas=(0.9, 0.95),
    weight_decay=0.9,
    eps=1e-8,
    max_norm=1e-2,
)


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_STEPS = 100

TINY_STORIES_VOCAB_SIZE = 10_000
OWT_VOCAB_SIZE = 32_000

TOKENIZED_DATA_PATH = "tokenized_data"

CHECKPOINTS_INTERAVAL_FRACTION = 0.1

# Validation loss logging interval
LOSS_LOG_INTERVAL_FRACTION = 0.01

# Number of batches to sample for validation loss
NUM_BATCHES_FOR_VALIDATION_LOSS = 5


@torch.no_grad()
def sample_validation_loss(model, valid_set, batch_size, llm_params, num_batches, device):
    """Sample the loss on the validation set."""
    total_loss = 0.0
    for i in range(num_batches):
        xb, yb = get_batch(
            valid_set,
            batch_size,
            llm_params.context_length,
            device,
        )

        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        total_loss += loss.cpu().item()
    return total_loss / num_batches


def train(run, start_time, train_set, valid_set, batch_size, llm_params, opt_params, num_steps, checkpoint_file):
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

    opt = AdamW(model.parameters(), opt_params.min_lr, opt_params.betas, opt_params.weight_decay, opt_params.eps)
    checkpoint_filepath = get_checkpoint_path(checkpoint_file)

    for step in range(num_steps):
        # Set the learning rate based on the optimizer schedule.
        lr = get_lr_cosine_schedule(
            step, opt_params.max_lr, opt_params.min_lr, opt_params.warmup_iters, opt_params.total_iters
        )
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        # (B, T)
        xb, yb = get_batch(
            train_set,
            batch_size,
            llm_params.context_length,
            device,
        )

        # (B, T, V)
        logits = model(xb)

        opt.zero_grad()

        loss = cross_entropy_loss(logits, yb)
        loss.backward()

        clip_gradient(model.parameters(), opt_params.max_norm)
        opt.step()

        if (step + 1) % (num_steps * LOSS_LOG_INTERVAL_FRACTION) == 0:
            val_loss = sample_validation_loss(
                model, valid_set, batch_size, llm_params, NUM_BATCHES_FOR_VALIDATION_LOSS, device
            )
            print(f"{step + 1}/{num_steps} -- Training loss = {loss.cpu().item()} -- Validation loss = {val_loss}")
            run.log(
                {
                    "step": step,
                    "train_loss": loss.cpu().item(),
                    "val_loss": val_loss,
                    "wallclock_time": time.time() - start_time,
                }
            )
        else:
            print(f"{step + 1}/{num_steps} -- Training loss = {loss.cpu().item()}")
            run.log(
                {
                    "step": step,
                    "train_loss": loss.cpu().item(),
                    "wallclock_time": time.time() - start_time,
                }
            )

        if (step + 1) % (num_steps * CHECKPOINTS_INTERAVAL_FRACTION) == 0:
            print(f"Saving checkpoint at step: {step + 1}")
            save_checkpoint(model, opt, step, checkpoint_filepath)

    # Save final checkpoint.
    print(f"Saving final checkpoint at: {checkpoint_filepath}")
    save_checkpoint(model, opt, num_steps - 1, checkpoint_filepath)


def main():
    parser = argparse.ArgumentParser(description="Train LLM with given parameters.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ts",
        help="Dataset (either ts or owt)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Num steps",
    )

    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="File to save checkpoints (.pt)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run",
    )

    args = parser.parse_args()

    train_set, valid_set, vocab_size = load_data(args.dataset)
    LLM_MINI_PARAMS.vocab_size = vocab_size

    # Start a new wandb run to track this script.
    run = wandb.init(
        name=args.run_name,
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="shashanko-meta",
        # Set the wandb project where this run will be logged.
        project="StandfordCS336Assignment1",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "LLM Debug",
            # General Params
            "dataset": "TinyStories" if args.dataset == "ts" else "OpenWebText",
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            # LLM Params
            "vocab_size": LLM_MINI_PARAMS.vocab_size,
            "context_length": LLM_MINI_PARAMS.context_length,
            "num_layers": LLM_MINI_PARAMS.num_layers,
            "d_model": LLM_MINI_PARAMS.d_model,
            "num_heads": LLM_MINI_PARAMS.num_heads,
            "d_ff": LLM_MINI_PARAMS.d_ff,
            "rope_theta": LLM_MINI_PARAMS.rope_theta,
            # Optimizer Params
            "min_lr": OPT_PARAMS.min_lr,
            "max_lr": OPT_PARAMS.max_lr,
            "warmup_iters": OPT_PARAMS.warmup_iters,
            "total_iters": OPT_PARAMS.total_iters,
            "betas": OPT_PARAMS.betas,
            "weight_decay": OPT_PARAMS.weight_decay,
            "eps": OPT_PARAMS.eps,
            "max_norm": OPT_PARAMS.max_norm,
        },
    )

    create_checkpoints_folder(CHECKPOINTS_FOLDER)
    start_time = time.time()
    train(
        run,
        start_time,
        train_set,
        valid_set,
        args.batch_size,
        LLM_MINI_PARAMS,
        OPT_PARAMS,
        args.num_steps,
        args.checkpoint_file,
    )
    run.finish()


if __name__ == "__main__":
    main()
