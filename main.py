from nano_llms.tokenizer import Tokenizer


def main():
    test_tokenizer = Tokenizer(vocabulary_size=10000)

    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as fp:
        train_dataset: str = fp.read()

    test_tokenizer.train(train_dataset, verbose=False)
    test_tokenizer.register_special_tokens(["<|endoftext|>"])

    original_string = "Hello World 🌍! <|endoftext|> Goodbye World!"
    encoded_string = test_tokenizer.encode(original_string)
    decoded_string = test_tokenizer.decode(encoded_string)

    print(encoded_string, decoded_string, decoded_string == original_string)


if __name__ == "__main__":
    main()
