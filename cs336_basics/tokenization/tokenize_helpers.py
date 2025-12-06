from cs336_basics.tokenization.constants import SPECIAL_TOKENS
from cs336_basics.tokenization.tokenizer import BPETokenizer
from cs336_basics.tokenization.bpe_trainer_helpers import (
    find_chunk_boundaries,
)
import multiprocessing
import os
import pickle

import numpy as np


OWT_TOKENIZER_FOLDER = "training_results_owt_final"
TS_TOKENIZER_FOLDER = "training_results_ts_final"


NUM_PROCESSES = os.cpu_count()
# This is a hyperparameter which can help speed up things.
# I didn't try playing with it too much, but 100 was slower.
NUM_TASKS_PER_PROCESS = 10


def read_tokenizer_files(tokenizer_folder):
    with open(f"{tokenizer_folder}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open(f"{tokenizer_folder}/merges.pkl", "rb") as f:
        merges = pickle.load(f)

    return vocab, merges


def get_tokenizer_folder(tokenizer_version: str):
    assert tokenizer_version in ["owt", "ts"], f"Tokenizer version {tokenizer_version} not supported."
    if tokenizer_version == "owt":
        return OWT_TOKENIZER_FOLDER
    return TS_TOKENIZER_FOLDER


def get_tokenizer(tokenizer_version: str):
    tokenizer_folder = get_tokenizer_folder(tokenizer_version)
    vocab, merges = read_tokenizer_files(tokenizer_folder)
    return BPETokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)


def encode_texts(args):
    (index, filename, start, end, tokenizer_version) = args
    tokenizer: BPETokenizer = get_tokenizer(tokenizer_version)
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return index, tokenizer.encode(chunk)


def write_to_file(f, chunk):
    np.asarray(chunk, dtype=np.uint16).tofile(f)


def encode_texts_in_parallel(input_filename, output_filename, tokenizer_version: str) -> dict[tuple[int, ...], int]:
    """
    Pretokenizes all the text in the file and returns the frequency of each
    pretoken.

    Note that each pretoken is represented as a tuple of code points.

    @arg special_tokens - Used to ensure we never split the special tokens.
    """
    with open(input_filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES * NUM_TASKS_PER_PROCESS, b"<|endoftext|>")

    pending = {}  # stores out-of-order results
    next_to_write = 0  # next required index

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    args = [
        (i, input_filename, start, end, tokenizer_version)
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    with open(output_filename, "wb") as f:
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            for idx, result in p.imap_unordered(encode_texts, args):
                pending[idx] = result

                # Flush all consecutive ready results
                while next_to_write in pending:
                    write_to_file(f, pending[next_to_write])
                    print(f"Wrote chunk {next_to_write + 1}/{len(args)}")
                    del pending[next_to_write]
                    next_to_write += 1
