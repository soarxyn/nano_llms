import torch
import torch.nn as nn

from nano_llms.attention import MultiHeadSelfAttention
from nano_llms.embedding import Embedding
from nano_llms.ffn import SwiGLUFFN
from nano_llms.linear import Linear
from nano_llms.rmsnorm import RMSNorm
from nano_llms.rope import RoPE


class Block(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, pos_encoder: RoPE | None = None
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, pos_encoder)

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        pos_encoder = RoPE(theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList(
            [Block(d_model, num_heads, d_ff, pos_encoder) for _ in range(num_layers)]
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.ln_final(x))
