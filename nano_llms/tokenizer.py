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
    ) -> tuple[list[int], dict[tuple[int, int], int]]:
        if len(tokens) < 2:
            return tokens, {}

        new_tokens: list[int] = []
        deltas: dict[tuple[int, int], int] = defaultdict(int)

        i = 0
        a, b = pair_to_merge

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                prev_token = new_tokens[-1] if new_tokens else None
                next_token = tokens[i + 2] if i + 2 < len(tokens) else None

                if prev_token is not None:
                    deltas[(prev_token, a)] -= 1
                    deltas[(prev_token, new_token)] += 1

                deltas[(a, b)] -= 1

                if next_token is not None:
                    deltas[(b, next_token)] -= 1
                    deltas[(new_token, next_token)] += 1

                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens, deltas

    def train(self, train_dataset: str, *, verbose: bool = False):
        text_chunks: list[str] = self.pat.findall(train_dataset)

        token_chunks: list[list[int]] = [
            list(chunk.encode("utf-8", errors="ignore")) for chunk in text_chunks
        ]

        num_merges = self.vocabulary_size - 256

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        # Tracks all chunks in which each pair occur.
        index: dict[tuple[int, int], set[int]] = defaultdict(set)

        for chunk_id, chunk in enumerate(token_chunks):
            if len(chunk) > 2:
                chunk_pair_counts = self.get_pair_counts(chunk)

                for pair, count in chunk_pair_counts.items():
                    pair_counts[pair] += count
                    index[pair].add(chunk_id)

        for i in range(num_merges):
            if not pair_counts:
                break

            pair_to_merge = max(pair_counts, key=pair_counts.get)  # type: ignore
            new_token = 256 + i

            for chunk_id in index[pair_to_merge]:
                new_chunk, pair_deltas = self.merge(
                    token_chunks[chunk_id], pair_to_merge, new_token
                )

                token_chunks[chunk_id] = new_chunk

                for pair, delta in pair_deltas.items():
                    pair_counts[pair] += delta

                    if delta > 0:
                        index[pair].add(chunk_id)

            del pair_counts[pair_to_merge]
            del index[pair_to_merge]

            self.merges[pair_to_merge] = new_token
            self.vocabulary[new_token] = (
                self.vocabulary[pair_to_merge[0]] + self.vocabulary[pair_to_merge[1]]
            )

            if verbose:
                print(
                    f"({i:03d}/{num_merges}) Merged {self.vocabulary[pair_to_merge[0]]}, {self.vocabulary[pair_to_merge[1]]} into {self.vocabulary[new_token]}"
                )

    def encode(self, text: str) -> list[int]:
        text_chunks: list[str] = self.pat.findall(text)

        tokens: list[int] = []

        for chunk in text_chunks:
            chunk_tokens: list[int] = list(chunk.encode("utf-8"))

            while len(chunk_tokens) >= 2:
                pair_to_merge: tuple[int, int] | None = None
                pair_position: int = -1
                pair_token: int = -1

                for i in range(len(chunk_tokens) - 1):
                    pair = (chunk_tokens[i], chunk_tokens[i + 1])

                    if merged_token := self.merges.get(pair):
                        if not pair_to_merge or merged_token < pair_token:
                            pair_to_merge = pair
                            pair_position = i
                            pair_token = merged_token

                if pair_to_merge:
                    chunk_tokens[pair_position] = pair_token
                    chunk_tokens.pop(pair_position + 1)
                else:
                    break
            tokens.extend(chunk_tokens)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        return b"".join([self.vocabulary[id] for id in tokens]).decode(
            "utf-8", errors="replace"
        )
