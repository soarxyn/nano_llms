import torch
import torch.nn as nn

from nano_llms.linear import Linear
from nano_llms.ops import swish


class ReLU2GLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        zero_init_projection: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

        if zero_init_projection:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.relu(self.w1(x).square()))


#        self.w1 = Linear(d_model, d_ff, device, dtype)
#        self.w3 = Linear(d_model, d_ff, device, dtype)
#        self.w2 = Linear(d_ff, d_model, device, dtype)
#
#        if zero_init_projection:
#            self.w2.weight.data.zero_()
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        return self.w2(relu_sqr(self.w1(x)) * self.w3(x))


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        zero_init_projection: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

        if zero_init_projection:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swish(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        zero_init_projection: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

        if zero_init_projection:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swish(self.w1(x)))
