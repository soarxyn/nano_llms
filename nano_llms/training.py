import os
import torch
import numpy as np
from typing import BinaryIO, IO


def sample_sequence(
    x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = x.shape[0] - context_length
    start_indices = np.random.randint(0, max_idx, size=batch_size)

    offsets = np.arange(context_length)
    idx = start_indices[:, None] + offsets[None, :]

    x_tensor = torch.from_numpy(x)

    sequences = x_tensor[idx]
    next_tokens = x_tensor[idx + 1]

    return sequences, next_tokens


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]
