import torch
import os
import pathlib
import typing
import numpy.typing as npt
import numpy as np


TOKENIZED_DATA_PATH = "tokenized_data"
CHECKPOINTS_FOLDER = "checkpoints/"

TINY_STORIES_VOCAB_SIZE = 10_000
OWT_VOCAB_SIZE = 32_000


def load_data(dataset):
    assert dataset in ["ts", "owt"]
    if dataset == "ts":
        train_set = np.memmap(f"{TOKENIZED_DATA_PATH}/TinyStoriesV2-GPT4-train.npy", dtype=np.uint16, mode="r")
        valid_set = np.memmap(f"{TOKENIZED_DATA_PATH}/TinyStoriesV2-GPT4-valid.npy", dtype=np.uint16, mode="r")
        vocab_size = TINY_STORIES_VOCAB_SIZE
    elif dataset == "owt":
        train_set = np.memmap(f"{TOKENIZED_DATA_PATH}/owt_train.npy", dtype=np.uint16, mode="r")
        valid_set = np.memmap(f"{TOKENIZED_DATA_PATH}/owt_valid.npy", dtype=np.uint16, mode="r")
        vocab_size = OWT_VOCAB_SIZE

    return train_set, valid_set, vocab_size


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    """Returns a tuple of tensors (x, y) where both x and y have shape
    (batch_size, context_length)
    """
    dataset_size = len(x)

    # max value sampled is dataset_size - context_length - 1
    # max arange then becomes dataset_size (exclusive)
    ix = torch.randint(0, dataset_size - context_length, size=(batch_size,))

    # Slice the numpy array FIRST (reads from disk), then convert to Tensor.
    # We use standard python slicing x[start:end] which is efficient for memmap.
    # We strictly cast to int64 because PyTorch embedding layers require Long/Int64.
    x_batch = torch.stack([torch.from_numpy((x[i : i + context_length]).astype(np.int64)) for i in ix])

    y_batch = torch.stack([torch.from_numpy((x[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in ix])

    if "cuda" in device:
        # pin_memory=True allows faster transfer to GPU
        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
        y_batch = y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

    return x_batch, y_batch


def create_checkpoints_folder(output_dir: str):
    """Creates the folder where checkpoints are saved if it doesn't already exist."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(checkpoint_file):
    return f"{CHECKPOINTS_FOLDER}{checkpoint_file}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    full_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(full_state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    full_state = torch.load(src)

    state_dict = full_state["model_state"]

    # Remove "_orig_mod." prefix if present. This is so you can deal with differences in naming when
    # you do model = torch.compile(model). We will load an uncompiled model.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    optimizer.load_state_dict(full_state["optimizer_state"])

    return full_state["iteration"]
