import torch
import torch.nn as nn
from einops import rearrange

from nano_llms.linear import Linear
from nano_llms.ops import scaled_dot_product_attn
from nano_llms.rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoder: RoPE | None = None,
    ) -> None:
        super().__init__()

        self.W_QKV = Linear(d_model, 3 * d_model)
        self.W_O = Linear(d_model, d_model)

        self.num_heads = num_heads

        self.rope = pos_encoder

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        QKV = self.W_QKV(x)
        Q, K, V = rearrange(QKV, "... (r h d) -> r h ... d", r=3, h=self.num_heads)

        if token_positions is not None and self.rope:
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        n, m = Q.size(-2), K.size(-2)

        mask = torch.tril(torch.ones(n, m, dtype=torch.bool))

        A = scaled_dot_product_attn(Q, K, V, mask)
        A = rearrange(A, "h ... d -> ... (h d)", h=self.num_heads)

        out = self.W_O(A)

        return out
