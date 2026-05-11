from nano_llms.tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer(vocabulary_size=10000, num_workers=32)

    tokenizer.train("data/owt_train.txt", verbose=True)
    tokenizer.register_special_tokens(["<|endoftext|>"])

    original_string = "Hello World 🌍! <|endoftext|> Goodbye World!"
    encoded_string = tokenizer.encode(original_string)
    decoded_string = tokenizer.decode(encoded_string)

    print(encoded_string, decoded_string, decoded_string == original_string)
    tokenizer.save("data/owt.json")


if __name__ == "__main__":
    main()
