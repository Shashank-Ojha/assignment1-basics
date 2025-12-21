"""Full training logic for the LLM"""

import argparse
import torch

from cs336_basics.experiments import EXPERIMENTAL_CONFIGS
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

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_STEPS = 100

# Checkpointing interval
CHECKPOINTS_INTERAVAL_FRACTION = 0.1

# Validation loss logging interval
LOSS_LOG_INTERVAL_FRACTION = 0.01

# Number of batches to sample for validation loss
NUM_BATCHES_FOR_VALIDATION_LOSS = 5

# Warmup Iters Fraction
WARM_UP_ITERS_FRACTION = 0.1  # (This is the recommendation based on Google)

DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


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


def train(
    run, start_time, train_set, valid_set, batch_size, llm_params, opt_params, num_steps, checkpoint_file, device
):
    device = DEVICE_CUDA if (device == DEVICE_CUDA and torch.cuda.is_available()) else device
    if device == DEVICE_CUDA:
        torch.set_float32_matmul_precision("high")

    print(f"device = {device}")

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

    if device == DEVICE_MPS:
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

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

        if (step + 1) % int(num_steps * LOSS_LOG_INTERVAL_FRACTION) == 0:
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

        if (step + 1) % int(num_steps * CHECKPOINTS_INTERAVAL_FRACTION) == 0:
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
        "--tokens_to_process",
        type=int,
        required=True,
        help="Num tokens to process",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config settings to use from experiments.py",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE_CPU,
        help="Device to use (cpu, cuda, mps)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run. Will also save checkpoints to this filename (.pt)",
    )

    args = parser.parse_args()
    # assert that run_name doesn't contain spaces. It should connect multiple words with "-"/"_"
    assert args.run_name.replace("_", "").replace("-", "").isalnum(), "run_name shouldn't contain spaces."

    llm_params, opt_params = EXPERIMENTAL_CONFIGS[args.config]

    # Update LLM Params.
    train_set, valid_set, vocab_size = load_data(args.dataset)
    llm_params.vocab_size = vocab_size

    # Num Steps.
    num_steps = args.tokens_to_process // (args.batch_size * llm_params.context_length)
    print(f"Running for num steps {num_steps}")

    # Update OPT Params.
    opt_params.total_iters = num_steps
    opt_params.warmup_iters = int(WARM_UP_ITERS_FRACTION * num_steps)

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
            "tokens_to_process": args.tokens_to_process,
            "num_steps": num_steps,
            "batch_size": args.batch_size,
            # LLM Params
            "vocab_size": llm_params.vocab_size,
            "context_length": llm_params.context_length,
            "num_layers": llm_params.num_layers,
            "d_model": llm_params.d_model,
            "num_heads": llm_params.num_heads,
            "d_ff": llm_params.d_ff,
            "rope_theta": llm_params.rope_theta,
            # Optimizer Params
            "min_lr": opt_params.min_lr,
            "max_lr": opt_params.max_lr,
            "warmup_iters": opt_params.warmup_iters,
            "total_iters": opt_params.total_iters,
            "betas": opt_params.betas,
            "weight_decay": opt_params.weight_decay,
            "eps": opt_params.eps,
            "max_norm": opt_params.max_norm,
            "default_device": args.device,
            "config_name": args.config,
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
        llm_params,
        opt_params,
        num_steps,
        args.run_name,  # checkpoint file is just the run name
        args.device,
    )
    run.finish()


if __name__ == "__main__":
    main()
