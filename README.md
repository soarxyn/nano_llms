# Nano LLMs

This repository contains LLM architectures and techniques I am currently studying. They are deeply base on awesome well-known resources such as [Stanford's CS336](https://stanford-cs336.github.io/spring2025/) and some of [Karpathy's Youtube Lectures](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

As I add more content, these resources will be updated.

## Roadmap

1. Tokenization
    - [x] Add a simple BPE training implementation in Python.
    - [x] Add Regex Pretokenization.
    - [x] Upgrade BPE training to consider frequency deltas instead of recounting strategy.
    - [x] Add a simple BPE encoding/decoding function.
    - [x] Add support for special tokens.
    - [x] Train Tokenizer on TinyStories.
2. LLM Architecture

## Data

The following data sources have been used:

- `tokenizer_pages.txt`: A copy-and-paste from 🤗Hugging Face's [article on Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4).

- `TinyStoriesV2-GPT4-{train|valid}.txt`: The [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, obtained as per [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) instructions:

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## Notes

### On Tokenization

> This implementation is largely based on [MinBPE](https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py) and [NanoChat's RustBPE](https://github.com/karpathy/nanochat/blob/master/rustbpe/src/lib.rs).

- Initial implementation (`69d6a2a`) works well with very small files (`tokenizer_pages.txt`).
- But it fails for medium-sized datasets, such as TinyStories' validation split, which should be relatively easy (CS336 reports < 2 min training on consumer hardware).
- Root cause probably comes from:
  - Recalculation of frequency counts every step, easily bouncing complexity to $O(N^2)$.
  - Maybe maximum calculation as well, but it should be more subtle as it takes $O(V^2)$.
  - Ingestion of large whole text chunks into memory (won't scale for training split, as probably will flood RAM and make swap a bottleneck during Regex matching).
- The same assignment from CS336 suggests that switching from frequency recalculation at every step to frequency deltas should alleviate the quadratic complexity.
- Using RustBPE's implementation has solved all issues and lets us train on the TinyStories train split.
