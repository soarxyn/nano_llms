import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.gain: torch.Tensor = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            ),
        )

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype: torch.dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        result = x * self.gain * rms

        return result.to(in_dtype)
