from collections.abc import Callable
from math import cos, pi, sqrt
from typing import Iterable
import torch


def lr_cosine_schedule(
    step: int, max_lr: float, min_lr: float, warmup_steps: int, cosine_steps: int
):
    if step < warmup_steps:
        return step * max_lr / warmup_steps
    elif warmup_steps <= step <= cosine_steps:
        return min_lr + 0.5 * (
            1 + cos((step - warmup_steps) / (cosine_steps - warmup_steps) * pi)
        ) * (max_lr - min_lr)
    return min_lr


class Scheduler:
    def __init__(self, optim: torch.optim.Optimizer, lr_fun: Callable[[int], float]):
        self.optim = optim
        self.lr_fun = lr_fun

        self.last_lr = 0

    def step(self, step: int):
        current_lr = self.lr_fun(step)

        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr

        self.last_lr = current_lr

        return current_lr


def clip_grad(
    params: Iterable[torch.nn.Parameter], max_grad_norm: float, eps: float = 1e-6
):
    grads = [p.grad for p in params if p.grad is not None]

    if not grads:
        return

    total_norm = torch.stack([(g * g).sum() for g in grads]).sum().sqrt()

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


# === Muon taken from https://github.com/KellerJordan/modded-nanogpt/tree/master


@torch.compile
def ns5_zeropower(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.95, nesterov=True, backend_steps=5):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "backend_steps": backend_steps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buffer = state["momentum_buffer"]
                buffer.mul(momentum).add_(grad)

                if group["nesterov"]:
                    grad = grad.add(buffer, alpha=momentum)

                if grad.size(0) == 3 * grad.size(1):
                    grad = torch.cat(
                        [
                            ns5_zeropower(grad_, steps=group["backend_steps"])
                            for grad_ in grad.split(grad.size(1))
                        ]
                    )
                    scale = grad.size(1) ** 0.5
                else:
                    grad = ns5_zeropower(grad, steps=group["backend_steps"])
                    scale = max(grad.size(0), grad.size(1)) ** 0.5

                p.data.add_(grad, alpha=-lr * scale)

        return loss
