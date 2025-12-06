import argparse
import pathlib


from cs336_basics.tokenization.tokenize_helpers import encode_texts_in_parallel

SAMPLE_SIZE = 10


def create_output_file(output_dir: str, dataset_name: str):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name.split(".")[0]
    return f"{output_dir}/{dataset_name}.npy"


def get_num_bytes(string: str):
    """Returns the number of bytes needed to represent the string"""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    return num_bytes


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)  # @inspect num_tokens
    return num_bytes / num_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize dataset given the tokenizer (either Tinystories or OpenWebText)."
    )
    parser.add_argument(
        "--tokenizer_version",
        type=str,
        default="ts",
        help="Tokenizer (ts or owt)",
    )
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default="TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file to tokenize.",
    )
    # This path is created relative to the current working directory from when the command to
    # run this script is called.
    parser.add_argument(
        "--output_dir",
        default="tokenized_data",
        type=str,
        help="Directory to save the tokenized data.",
    )

    parser.add_argument(
        "--should_sample",
        default=False,
        type=bool,
        help="If we should just sample 10 examples for testing purposes.",
    )

    args = parser.parse_args()

    dataset = f"data/{args.input_dataset_path}"

    print("Dataset:", dataset)

    if args.should_sample:
        # TODO: fix this after refactor
        pass
        # encoded = []
        # total_num_bytes = 0
        # for doc in itertools.islice(file_itr, SAMPLE_SIZE):
        #     encoded.extend(tokenizer.encode(doc))
        #     total_num_bytes += get_num_bytes(doc)

        # ttoal_num_tokens = len(encoded)
        # print("Compression Ratio:", total_num_bytes / ttoal_num_tokens)
    else:
        output_filename = create_output_file(args.output_dir, args.input_dataset_path)
        encode_texts_in_parallel(dataset, output_filename, args.tokenizer_version)

        print(f"Saved to file: {output_filename}")


if __name__ == "__main__":
    main()
