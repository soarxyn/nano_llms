from math import sqrt

import torch
from einops import einsum


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val, _ = x.max(dim=dim, keepdim=True)
    z = (x - max_val).exp()

    return z / z.sum(dim=dim, keepdim=True)


def scaled_dot_product_attn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    inv_d_k = 1.0 / sqrt(K.size(-1))

    scores = inv_d_k * einsum(Q, K, "... n dk, ... m dk -> ... n m")

    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))

    weights = softmax(scores, dim=-1)
    return einsum(weights, V, "... n m, ... m dv -> ... n dv")


def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    batch_size = logits.size(0)

    max_val, _ = logits.max(dim=-1, keepdim=True)
    x = logits - max_val

    target_logit = x.gather(1, target[:, None])
    log_sump_exp = x.exp().sum(dim=-1, keepdim=True).log()

    loss = -target_logit + log_sump_exp

    return loss.sum() / batch_size
