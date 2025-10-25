from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Self
from multiprocessing import Pool

import regex as re
from tqdm import trange

# From: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/gpt4.py#L48C22-L48C138
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


@dataclass
class Word:
    tokens: list[int]

    def pairs(self) -> Iterator[tuple[int, int]]:
        return zip(self.tokens, self.tokens[1:])

    def merge_pair(
        self, pair_to_merge: tuple[int, int], new_token: int
    ) -> dict[tuple[int, int], int]:
        if (n := len(self.tokens)) < 2:
            return {}

        new_tokens: list[int] = []
        deltas: dict[tuple[int, int], int] = defaultdict(int)

        i = 0
        a, b = pair_to_merge

        while i < n:
            if i < n - 1 and self.tokens[i] == a and self.tokens[i + 1] == b:
                prev_token = new_tokens[-1] if new_tokens else None
                next_token = self.tokens[i + 2] if i + 2 < n else None

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
                new_tokens.append(self.tokens[i])
                i += 1

        self.tokens = new_tokens

        return deltas


@dataclass
class MergeJob:
    pair_to_merge: tuple[int, int]
    count: int
    index: set[int]

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, MergeJob)
        return self.count == other.count and self.pair_to_merge == other.pair_to_merge

    def __lt__(self, other: Self) -> bool:
        if self.count != other.count:
            return self.count < other.count
        return self.pair_to_merge < other.pair_to_merge


class Tokenizer:
    def __init__(self, vocabulary_size: int, *, pat: str | None = None):
        assert vocabulary_size > 256, (
            "Vocabulary size must be over the default 256 Unicode Bytes."
        )

        self.vocabulary_size = vocabulary_size
        self.pat = re.compile(pat or GPT4_PAT)

        self.merges: dict[tuple[int, int], int] = {}
        self.vocabulary = {token: bytes([token]) for token in range(256)}

        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    # def get_pair_counts(self, tokens: list[int]) -> dict[tuple[int, int], int]:
    #     pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    #     for pair in zip(tokens, tokens[1:]):
    #         pair_counts[pair] += 1

    #     return pair_counts

    def get_pair_counts(self, words: list[Word], counts: list[int]):
        def map_count(
            idx: int, word: Word
        ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
            local_counts: dict[tuple[int, int], int] = defaultdict(int)
            local_index: dict[tuple[int, int], set[int]] = defaultdict(set)

            if len(word.tokens) >= 2 and counts[idx] != 0:
                for a, b in word.pairs():
                    local_counts[(a, b)] += counts[idx]
                    local_index[(a, b)].add(idx)

            return local_counts, local_index

        with Pool(processes=12) as pool:
            local_results = pool.starmap(map_count, (words, counts))

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        index: dict[tuple[int, int], set[int]] = defaultdict(set)

        for local_counts, local_indices in local_results:
            for pair, count in local_counts.items():
                pair_counts[pair] += count

            for pair, ind in local_indices.items():
                index[pair].update(ind)

        return pair_counts, index

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

        for i in trange(num_merges):
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

    def register_special_tokens(self, special_tokens: list):
        starting_id: int = len(self.vocabulary)

        self.special_tokens = {
            token: starting_id + idx for idx, token in enumerate(special_tokens)
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        special_pat = [re.escape(s) for s in special_tokens]
        special_pat = sorted(special_pat, key=len, reverse=True)
        self.special_pat = re.compile("(" + "|".join(special_pat) + ")")

    def encode_simple(self, text: str) -> list[int]:
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

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_simple(text)

        simple_chunks: list[str] = self.special_pat.split(text)

        tokens: list[int] = []
        for chunk in simple_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
            else:
                tokens.extend(self.encode_simple(chunk))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        word_bytes: list[bytes] = []

        for token in tokens:
            if token in self.vocabulary:
                word_bytes.append(self.vocabulary[token])
            elif token in self.inverse_special_tokens:
                word_bytes.append(self.inverse_special_tokens[token].encode("utf-8"))

        return b"".join(word_bytes).decode("utf-8", errors="replace")
