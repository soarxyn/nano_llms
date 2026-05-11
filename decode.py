import torch

from nano_llms.transformer import Transformer
from nano_llms.adamw import AdamW
from nano_llms.ops import softmax
from nano_llms.tokenizer import Tokenizer
from nano_llms.training import load_checkpoint

from jsonargparse import ArgumentParser


def decode(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    transformer = Transformer(**cfg.transformer.as_dict()).to(device)

    optimizer = AdamW(
        transformer.parameters(),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    tokenizer = Tokenizer.load(cfg.tokenizer_profile)
    transformer = torch.compile(transformer)

    load_checkpoint(cfg.checkpoint, transformer, optimizer)

    transformer.eval()

    prompt = tokenizer.encode(cfg.prompt)
    new_token = -1

    print(tokenizer.decode(prompt), end="")

    while (
        new_token != tokenizer.special_tokens["<|endoftext|>"]
        and len(prompt) < cfg.transformer.context_length
    ):
        x = torch.tensor(prompt, device=device).unsqueeze(0)
        logits = transformer(x)[0, -1]
        probs = softmax(logits, dim=-1, temperature=cfg.temperature)

        values, indices = probs.sort(dim=-1, descending=True)
        cumulative = torch.cumsum(values, dim=-1)

        probs[indices[cumulative - values > cfg.top_p]] = 0.0

        new_token = probs.multinomial(num_samples=1).item()
        prompt.append(new_token)
        print(tokenizer.decode([new_token]), end="")

    print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Transformer, "transformer")
    parser.add_argument("--tokenizer_profile")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)

    cfg = parser.parse_args()
    decode(cfg)
