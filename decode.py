import torch

from nano_llms.transformer import Transformer
from nano_llms.adamw import AdamW
from nano_llms.ops import softmax
from nano_llms.tokenizer import Tokenizer
from nano_llms.training import load_checkpoint


def decode():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = Transformer(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        theta=10000,
    ).to(device)

    optimizer = AdamW(
        transformer.parameters(),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    tokenizer = Tokenizer.load("")

    load_checkpoint("", transformer, optimizer)

    transformer = torch.compile(transformer)
    transformer.eval()

    prompt = tokenizer.encode("")
    new_token = -1

    while new_token != tokenizer.special_tokens["<|endoftext|>"] and len(prompt) < 256:
        x = torch.tensor(prompt, device=device).unsqueeze(0)
        logits = transformer(x)[0, -1]
        probs = softmax(logits, dim=-1)

        new_token = probs.argmax().item()
        prompt.append(new_token)

    final_story = tokenizer.decode(prompt)
    print(final_story)
