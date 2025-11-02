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

        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.output_proj = Linear(d_model, d_model)

        self.num_heads = num_heads

        self.rope = pos_encoder

        self._register_load_state_dict_pre_hook(self._load_qkv_hook)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        QKV = self.qkv_proj(x)
        Q, K, V = rearrange(QKV, "... (r h d) -> r h ... d", r=3, h=self.num_heads)

        if self.rope:
            if token_positions is None:
                seq_len = x.size(-2)
                token_positions = torch.arange(seq_len)

            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        n, m = Q.size(-2), K.size(-2)

        mask = torch.tril(torch.ones(n, m, dtype=torch.bool))

        A = scaled_dot_product_attn(Q, K, V, mask)
        A = rearrange(A, "h ... d -> ... (h d)", h=self.num_heads)

        out = self.output_proj(A)

        return out

    def _load_qkv_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if f"{prefix}q_proj.weight" in state_dict:
            q_weight = state_dict.pop(f"{prefix}q_proj.weight")
            k_weight = state_dict.pop(f"{prefix}k_proj.weight")
            v_weight = state_dict.pop(f"{prefix}v_proj.weight")

            state_dict[f"{prefix}qkv_proj.weight"] = torch.cat(
                [q_weight, k_weight, v_weight], dim=0
            )

            return state_dict
