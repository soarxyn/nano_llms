from nano_llms.tokenizer import Tokenizer


def main():
    test_tokenizer = Tokenizer(vocabulary_size=10000)

    test_tokenizer.train("data/TinyStoriesV2-GPT4-train.txt", verbose=True)
    test_tokenizer.register_special_tokens(["<|endoftext|>"])

    # test_tokenizer = Tokenizer.load("data/tiny_storie.s")

    original_string = "Hello World 🌍! <|endoftext|> Goodbye World!"
    encoded_string = test_tokenizer.encode(original_string)
    decoded_string = test_tokenizer.decode(encoded_string)

    print(encoded_string, decoded_string, decoded_string == original_string)

    test_tokenizer.save("data/tiny_stories.json")


if __name__ == "__main__":
    main()
