from collections import defaultdict

import regex as re

# From: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/gpt4.py#L48C22-L48C138
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocabulary_size: int, *, pat: str | None = None):
        assert vocabulary_size > 256, (
            "Vocabulary size must be over the default 256 Unicode Bytes."
        )

        self.vocabulary_size = vocabulary_size
        self.pat = re.compile(pat or GPT4_PAT)

        self.merges: dict[tuple[int, int], int] = {}
        self.vocabulary = {token: bytes([token]) for token in range(256)}

    def get_pair_counts(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += 1

        return pair_counts

    def merge(
        self, tokens: list[int], pair_to_merge: tuple[int, int], new_token: int
    ) -> list[int]:
        new_tokens: list[int] = []

        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair_to_merge[0]
                and tokens[i + 1] == pair_to_merge[1]
            ):
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, train_dataset: str, *, verbose: bool = False):
        text_chunks: list[str] = self.pat.findall(train_dataset)
        token_chunks: list[list[int]] = [
            list(chunk.encode("utf-8", errors="ignore")) for chunk in text_chunks
        ]

        num_merges = self.vocabulary_size - 256

        for i in range(num_merges):
            per_chunk_max: dict = defaultdict(int)

            for chunk in token_chunks:
                if len(chunk) > 2:
                    chunk_pair_counts = self.get_pair_counts(chunk)
                    chunk_most_frequent = max(
                        chunk_pair_counts,
                        key=chunk_pair_counts.get,  # type: ignore
                    )

                    per_chunk_max[chunk_most_frequent] += 1

            pair_to_merge = max(per_chunk_max, key=per_chunk_max.get)  # type: ignore

            new_token = 256 + i
            new_chunks = []

            for chunk in token_chunks:
                new_chunks.append(self.merge(chunk, pair_to_merge, new_token))

            token_chunks = new_chunks

            self.merges[pair_to_merge] = new_token
            self.vocabulary[new_token] = (
                self.vocabulary[pair_to_merge[0]] + self.vocabulary[pair_to_merge[1]]
            )

            if verbose:
                print(
                    f"({i:03d}/{num_merges}) Merged {self.vocabulary[pair_to_merge[0]]}, {self.vocabulary[pair_to_merge[1]]} into {self.vocabulary[new_token]}"
                )
