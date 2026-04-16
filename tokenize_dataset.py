import multiprocessing as mp
import os

import numpy as np
from jsonargparse import auto_cli

from nano_llms.tokenizer import Tokenizer, find_chunk_boundaries


def process_chunk_to_tokens(args):
    dataset_path, start_byte, end_byte, tokenizer_path = args
    tokenizer = Tokenizer.load(tokenizer_path)

    with open(dataset_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    documents = chunk_text.split("<|endoftext|>")

    all_tokens = []
    for doc in documents:
        if doc.strip():
            tokens = tokenizer.encode(doc.strip())
            all_tokens.extend(tokens)

    return np.array(all_tokens, dtype=np.uint16)


def tokenize_dataset(
    dataset_path: str,
    tokenizer_path: str,
    output_path: str,
):
    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=mp.cpu_count() * 2,
            split_special_token=b"<|endoftext|>",
        )

    tasks = []
    for i in range(len(boundaries) - 1):
        tasks.append((dataset_path, boundaries[i], boundaries[i + 1], tokenizer_path))

    max_possible_tokens = os.path.getsize(dataset_path)
    mmap_array = np.memmap(
        output_path, dtype=np.uint16, mode="w+", shape=(max_possible_tokens,)
    )

    current_pos = 0
    with mp.Pool(mp.cpu_count()) as pool:
        for result_array in pool.imap(process_chunk_to_tokens, tasks):
            n = len(result_array)
            mmap_array[current_pos : current_pos + n] = result_array
            current_pos += n

    mmap_array.flush()
    del mmap_array

    os.truncate(output_path, current_pos * np.dtype(np.uint16).itemsize)

    print(f"Done! Saved {current_pos} tokens.")


if __name__ == "__main__":
    auto_cli(tokenize_dataset)
