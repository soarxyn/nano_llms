from collections import defaultdict


class Tokenizer:
    def __init__(self, vocabulary_size: int):
        assert vocabulary_size > 256, (
            "Vocabulary size must be over the default 256 Unicode Bytes."
        )

        self.vocabulary_size = vocabulary_size
        self.merges: dict[tuple[int, int], int] = {}

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
        tokens = list(train_dataset.encode("utf-8", errors="ignore"))
        num_merges = self.vocabulary_size - 256

        for i in range(num_merges):
            pair_counts = self.get_pair_counts(tokens)

            pair_to_merge = max(pair_counts, key=pair_counts.get)  # type: ignore

            new_token = 256 + i
            tokens = self.merge(tokens, pair_to_merge, new_token)

            self.merges[pair_to_merge] = new_token

            if verbose:
                print(f"({i:03d}/{num_merges}) Merged {pair_to_merge} into {new_token}")
