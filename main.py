from nano_llms.tokenizer import Tokenizer


def main():
    test_tokenizer = Tokenizer(vocabulary_size=1000)

    # This works well! 😊
    with open("data/tokenizer_page.txt", encoding="utf-8") as fp: 
        train_dataset: str = fp.read()

    # This doesn't at all...
    # with open("data/tokenizer_page.txt", encoding="utf-8") as fp: 
        # train_dataset: str = fp.read()

    test_tokenizer.train(train_dataset, verbose=True)


if __name__ == "__main__":
    main()
