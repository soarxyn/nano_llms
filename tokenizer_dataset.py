from jsonargparse import auto_cli
from nano_llms.tokenizer import Tokenizer


def tokenize_dataset(dataset_path: str, tokenizer_path: str):
    tokenizer = Tokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(dataset_path)

    return tokens


if __name__ == "__main__":
    auto_cli(tokenize_dataset)
