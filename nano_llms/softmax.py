import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val, _ = x.max(dim=dim, keepdim=True)
    z = (x - max_val).exp()

    return z / z.sum(dim=dim, keepdim=True)
