from nano_llms.tokenizer import Tokenizer


def main():
    test_tokenizer = Tokenizer(vocabulary_size=10000)

    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as fp:
        train_dataset: str = fp.read()

    test_tokenizer.train(train_dataset, verbose=False)


if __name__ == "__main__":
    main()
