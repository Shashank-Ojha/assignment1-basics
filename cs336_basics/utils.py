import torch
import os
import typing
import numpy.typing as npt
import numpy as np


# Example of loading the data correctly
# data = np.memmap('train.bin', dtype=np.uint16, mode='r')

# Now pass 'data' into your function
# xb, yb = get_batch(data, batch_size=4, context_length=8, device='cuda')


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

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    return x_batch, y_batch


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

    model.load_state_dict(full_state["model_state"])
    optimizer.load_state_dict(full_state["optimizer_state"])

    return full_state["iteration"]
