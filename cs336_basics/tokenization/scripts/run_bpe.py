import argparse
import pickle
import pathlib
from cs336_basics.tokenization.bpe_trainer import train_tokenizer
from cs336_basics.tokenization.constants import SPECIAL_TOKENS

import resource


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Size of the vocabulary (must be >= 256 + number of special tokens). Default: 10000",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file to be used for training.",
    )

    # This path is created relative to the current working directory from when the command to
    # run this script is called.
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save vocab and merges files (default: None, no saving)",
    )
    args = parser.parse_args()

    dataset = f"data/{args.input_path}"
    vocab, merges = train_tokenizer(dataset, args.vocab_size, SPECIAL_TOKENS)
    print(f"Training complete. Vocabulary size: {len(vocab)}")

    # Save vocab and merges
    if args.output_dir:
        output_path = pathlib.Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        vocab_filepath = f"{args.output_dir}/vocab.pkl"
        merges_filepath = f"{args.output_dir}/merges.pkl"

        with open(vocab_filepath, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to: {vocab_filepath}")

        with open(merges_filepath, "wb") as f:
            pickle.dump(merges, f)
        print(f"Merges saved to: {merges_filepath}")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # The measurement is provided in bytes so we convert it to MB
    print(f"Max memory (MB): {usage.ru_maxrss / (1024**2)}")


if __name__ == "__main__":
    main()
