# Nano LLMs

This repository contains LLM architectures and techniques I am currently studying. They are deeply base on awesome well-known resources such as [Stanford's CS336](https://stanford-cs336.github.io/spring2025/) and some of [Karpathy's Youtube Lectures](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

As I add more content, these resources will be updated.

## Roadmap

1. Tokenization
    - [x] Add a simple BPE training implementation in Python.
    - [x] Add Regex Pretokenization.
    - [ ] Upgrade BPE training to consider frequency deltas instead of recounting strategy.
    - [ ] Add a simple BPE encoding/decoding function.
    - [ ] Train Tokenizer on TinyStories.
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
