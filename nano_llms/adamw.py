from collections.abc import Callable
from math import cos, pi, sqrt
from typing import Iterable
import torch


def lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_steps):
    if step < warmup_steps:
        return step * max_lr / warmup_steps
    elif warmup_steps <= step <= cosine_steps:
        return min_lr + 0.5 * (
            1 + cos((step - warmup_steps) / (cosine_steps - warmup_steps) * pi)
        ) * (max_lr - min_lr)
    return min_lr


def clip_grad(
    params: Iterable[torch.nn.Parameter], max_grad_norm: float, eps: float = 1e-6
):
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = sqrt(sum([(g * g).sum() for g in grads]))

    if total_norm <= max_grad_norm:
        return

    for g in grads:
        g *= max_grad_norm / (total_norm + eps)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):  # type:ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data

                state["m"] = m = beta_1 * m + (1 - beta_1) * grad
                state["v"] = v = beta_2 * v + (1 - beta_2) * grad * grad

                lr_t = lr * sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1

        return loss
