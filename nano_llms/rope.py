import torch
import torch.nn as nn
from einops import rearrange


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        angles = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        inv_freq = 1.0 / (theta ** (angles / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cosines", freqs.cos(), persistent=False)
        self.register_buffer("sines", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.cosines, torch.Tensor)
        assert isinstance(self.sines, torch.Tensor)

        cos_i, sin_i = self.cosines[pos], self.sines[pos]

        x = rearrange(x, "... (d r) -> ... d r", r=2)  # [... D] -> [... D/2 2]

        x_rot = torch.stack(
            [
                x[..., 0] * cos_i - x[..., 1] * sin_i,
                x[..., 0] * sin_i + x[..., 1] * cos_i,
            ],
            dim=-1,
        )

        return rearrange(x_rot, "... d r -> ... (d r)", r=2)
