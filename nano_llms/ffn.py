import torch
import torch.nn as nn

from nano_llms.linear import Linear
from nano_llms.ops import swish


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swish(self.w1(x)) * self.w3(x))
