from math import sqrt

import torch
import torch.nn as nn
from einops import einsum, rearrange

from nano_llms.linear import Linear
from nano_llms.rope import RoPE
from nano_llms.softmax import softmax


def scaled_dot_product_attn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    inv_d_k = 1.0 / sqrt(K.size(-1))

    scores = inv_d_k * einsum(Q, K, "... n dk, ... m dk -> ... n m")

    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))

    weights = softmax(scores, dim=-1)
    return einsum(weights, V, "... n m, ... m dv -> ... n dv")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        assert (theta is None and max_seq_len is None) or (
            theta is not None and max_seq_len is not None
        )

        self.W_QKV = Linear(d_model, 3 * d_model)
        self.W_O = Linear(d_model, d_model)

        self.num_heads = num_heads

        self.rope = (
            RoPE(theta, d_model // num_heads, max_seq_len)
            if theta and max_seq_len
            else None
        )

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
