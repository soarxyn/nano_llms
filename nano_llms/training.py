import os
import torch
import numpy as np
from typing import BinaryIO, IO

from torch.utils.data import IterableDataset


def sample_sequence(
    data: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = data.shape[0] - context_length
    indices = np.random.randint(0, max_idx, size=batch_size)

    x = [
        torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in indices
    ]
    y = [
        torch.from_numpy(data[i + 1 : i + context_length + 1].astype(np.int64))
        for i in indices
    ]

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)

    return x, y


class TinyStoriesDataset(IterableDataset):
    def __init__(
        self, data_path: str, batch_size: int, context_length: int, device: str
    ):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.max_id = self.data.shape[0] - context_length - 1

    def __iter__(self):
        while True:
            yield sample_sequence(
                self.data, self.batch_size, self.context_length, self.device
            )


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
