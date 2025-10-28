import os
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop_max, heappush_max
from multiprocessing import Pool
from typing import BinaryIO, Iterator, Self

import regex as re
from tqdm import trange

# From: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/gpt4.py#L48C22-L48C138
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# From: https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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
    def __init__(
        self, vocabulary_size: int, *, pat: str | None = None, num_workers: int = 12
    ):
        assert vocabulary_size > 256, (
            "Vocabulary size must be over the default 256 Unicode Bytes."
        )

        self.vocabulary_size = vocabulary_size
        self.pat = re.compile(pat or GPT4_PAT)

        self.eot_pat = re.compile(re.escape("<|endoftext|>"))

        self.merges: dict[tuple[int, int], int] = {}
        self.vocabulary = {token: bytes([token]) for token in range(256)}

        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

        self.num_workers = num_workers

    def map_count(
        self, idx: int, word: Word, counts: list[int]
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
        local_counts: dict[tuple[int, int], int] = defaultdict(int)
        local_index: dict[tuple[int, int], set[int]] = defaultdict(set)

        if len(word.tokens) >= 2 and counts[idx] != 0:
            for a, b in word.pairs():
                local_counts[(a, b)] += counts[idx]
                local_index[(a, b)].add(idx)

        return local_counts, local_index

    def get_pair_counts(self, words: list[Word], counts: list[int]):
        with Pool(processes=self.num_workers) as pool:
            local_results = pool.starmap(
                self.map_count, [(idx, word, counts) for idx, word in enumerate(words)]
            )

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

    def count_in_chunk(self, strings: str):
        local_counts: dict[str, int] = defaultdict(int)

        for atom in self.pat.findall(strings):
            local_counts[atom] += 1

        return local_counts

    def train(self, dataset_path: str, *, verbose: bool = False):
        counts: dict[str, int] = defaultdict(int)

        with open(dataset_path, mode="rb") as fp:
            boundaries = find_chunk_boundaries(fp, self.num_workers, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                fp.seek(start)
                chunk = fp.read(end - start).decode("utf-8", errors="ignore")

                strings = self.eot_pat.split(chunk)

                with Pool(processes=self.num_workers) as pool:
                    local_counts = pool.map(self.count_in_chunk, strings)

                for local_count in local_counts:
                    for word, count in local_count.items():
                        counts[word] += count

        words: list[Word] = []
        count_vector: list[int] = []

        for word, count in counts.items():
            words.append(Word(list(word.encode("utf-8", errors="ignore"))))
            count_vector.append(count)

        num_merges = self.vocabulary_size - 256

        pair_counts, index = self.get_pair_counts(words, count_vector)

        heap: list[MergeJob] = []

        for pair, position in index.items():
            count = pair_counts.get(pair, 0)

            if count > 0:
                heappush_max(heap, MergeJob(pair, count, position))

        for i in trange(num_merges):
            if not pair_counts:
                break

            top_pair = heappop_max(heap)

            current_count = pair_counts.get(top_pair.pair_to_merge, 0)

            if top_pair.count != current_count:
                top_pair.count = current_count

                if top_pair.count > 0:
                    heappush_max(heap, top_pair)
                continue

            if top_pair.count == 0:
                break

            new_token = 256 + i

            local_index_updates: dict[tuple[int, int], set[int]] = defaultdict(set)
            for word_id in top_pair.index:
                pair_deltas: dict[tuple[int, int], int] = words[word_id].merge_pair(
                    top_pair.pair_to_merge, new_token
                )

                for pair, delta in pair_deltas.items():
                    delta_total = delta * count_vector[word_id]

                    if delta_total != 0:
                        pair_counts[pair] += delta_total

                        if delta_total > 0:
                            local_index_updates[pair].add(word_id)

            for pair, position in local_index_updates.items():
                count = pair_counts[pair]

                if count > 0:
                    heappush_max(heap, MergeJob(pair, count, position))

            self.merges[top_pair.pair_to_merge] = new_token
            self.vocabulary[new_token] = (
                self.vocabulary[top_pair.pair_to_merge[0]]
                + self.vocabulary[top_pair.pair_to_merge[1]]
            )

            if verbose:
                print(
                    f"({i:03d}/{num_merges}) Merged {self.vocabulary[top_pair.pair_to_merge[0]]}, {self.vocabulary[top_pair.pair_to_merge[1]]} into {self.vocabulary[new_token]}"
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
