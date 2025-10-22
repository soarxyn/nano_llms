from nano_llms.tokenizer import Tokenizer


def main():
    test_tokenizer = Tokenizer(vocabulary_size=512)

    with open("data/tokenizer_page.txt") as fp:
        fp.readline()

        train_dataset: str = fp.read()

    test_tokenizer.train(train_dataset, verbose=True)


if __name__ == "__main__":
    main()
