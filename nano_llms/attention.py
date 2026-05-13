import torch
import torch.nn as nn
from einops import rearrange

from nano_llms.linear import Linear
from nano_llms.ops import scaled_dot_product_attn
from nano_llms.rmsnorm import RMSNorm
from nano_llms.rope import NoPE, RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoder: RoPE | NoPE | None = None,
        qk_norm: bool = False,
        zero_init_projection: bool = False,
    ) -> None:
        super().__init__()

        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.output_proj = Linear(d_model, d_model)

        self.qk_norm = qk_norm

        if self.qk_norm:
            self.q_norm = RMSNorm(d_model // num_heads)
            self.k_norm = RMSNorm(d_model // num_heads)

        self.num_heads = num_heads

        self.pos_encoder = pos_encoder

        if zero_init_projection:
            nn.init.zeros_(self.output_proj.weight)

        self._register_load_state_dict_pre_hook(self._load_qkv_hook)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        QKV = self.qkv_proj(x)
        Q, K, V = rearrange(QKV, "... (r h d) -> r h ... d", r=3, h=self.num_heads)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if self.pos_encoder:
            if token_positions is None:
                seq_len = x.size(-2)
                token_positions = torch.arange(seq_len, device=x.device)

            Q, K = (
                self.pos_encoder(Q, token_positions),
                self.pos_encoder(K, token_positions),
            )

        n, m = Q.size(-2), K.size(-2)

        mask = torch.tril(torch.ones(n, m, dtype=torch.bool, device=x.device))

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
