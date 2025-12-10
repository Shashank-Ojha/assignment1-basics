from cs336_basics.tokenization.tokenize_helpers import (
    get_tokenizer,
)
import numpy as np
import multiprocessing

from cs336_basics.utils import load_data


def verify_tokenizer_sizes():
    print("TinyStories tokenizer:")
    tokenizer = get_tokenizer("ts")
    print("Vocabulary size:", len(tokenizer.vocab))

    print("OpenWebText tokenizer:")
    tokenizer = get_tokenizer("owt")
    print("Vocabulary size:", len(tokenizer.vocab))


def max_of_slice(slice_):
    return np.max(slice_)


def compute_max_value(array):
    num_cpus = multiprocessing.cpu_count()

    chunk_size = min(10_000_000, len(array) // num_cpus + 1)
    slices = [array[i : i + chunk_size] for i in range(0, len(array), chunk_size)]
    with multiprocessing.Pool(num_cpus) as pool:
        max_values = pool.map(max_of_slice, slices)
    return max(max_values)


def verify_tokenized_data():
    train_set, valid_set, vocab_size = load_data("ts")
    print("TinyStories tokenized data:")
    max_train_value = compute_max_value(train_set)
    print("Max train set value:", max_train_value)
    max_valid_value = compute_max_value(valid_set)
    print("Max valid set value:", max_valid_value)

    train_set, valid_set, vocab_size = load_data("owt")
    print("OpenWebText tokenized data:")
    max_train_value = compute_max_value(train_set)
    print("Max train set value:", max_train_value)
    max_valid_value = compute_max_value(valid_set)
    print("Max valid set value:", max_valid_value)


if __name__ == "__main__":
    verify_tokenizer_sizes()
    verify_tokenized_data()
