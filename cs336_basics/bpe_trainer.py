import os
from typing import BinaryIO, Tuple, Dict
import regex as re
from collections import defaultdict
import multiprocessing

from cs336_basics.pretokenizer import read_chunk_and_pretokenize, SPECIAL_TOKENS

INITIAL_VOCAB_SIZE = 256
NUM_WORKERS = 20

def get_pretoken_counts(filename, special_tokens) -> dict[tuple[int, ...], int]:
    """ 
    Pretokenizes all the text in the file and returns the frequency of each
    pretoken.

    Note that each pretoken is represented as a tuple of code points.

    @arg special_tokens - Used to ensure we never split the special tokens.
    """
    pretoken_counts = defaultdict(int)
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_WORKERS, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    args = [(filename, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(NUM_WORKERS) as p:
        pretoken_counts_per_worker = p.starmap(read_chunk_and_pretokenize, args)

    # Merge all the counts
    for counts in pretoken_counts_per_worker:
        for code_points_tuple, count in counts.items():
            pretoken_counts[code_points_tuple] += count

    return pretoken_counts


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_pair_counts(pretokens_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """
    Get frequency of every code point pair. Only pairs within a pretoken are considered. Pairs between
    pretokens are not considered. Similarly, special tokens are also not considered.
    """
    pair_counts = defaultdict(int)  # (int, int) -> count

    for code_points_tuple, count in pretokens_counts.items():
        for b1, b2 in zip(code_points_tuple, code_points_tuple[1:]):
            pair_counts[(b1, b2)] += count

    return pair_counts


def merge_code_points_in_pretokens(pretokens_counts: dict[tuple[int, ...], int], to_merge_pair: tuple[int, int], new_code_point: int) -> dict[tuple[int, ...], int]:
    """
    Given pretoken counts (tuple of code points -> counts), returns a new
    version where the tuple of code points keys merge the to_merge_pair to the new_code_point.
    """
    new_pretoken_counts = defaultdict(int)
    for code_points_tuple, count in pretokens_counts.items():
        i = 0
        new_code_points_list = []
        while i < len(code_points_tuple):
            if (
                i + 1 < len(code_points_tuple)
                and code_points_tuple[i] == to_merge_pair[0]
                and code_points_tuple[i + 1] == to_merge_pair[1]
            ):
                new_code_points_list.append(new_code_point)
                i += 2
            else:
                new_code_points_list.append(code_points_tuple[i])
                i += 1

        new_pretoken_counts[tuple(new_code_points_list)] += count

    return new_pretoken_counts

def convert_to_byte_space(vocab, byte_pair):
    return (vocab[byte_pair[0]], vocab[byte_pair[1]])

def is_lexigraphically_greater(vocab, lht, rht):
    lht_byte_pair = convert_to_byte_space(vocab, lht)
    rht_byte_pair = convert_to_byte_space(vocab, rht)
    return lht_byte_pair > rht_byte_pair

def train_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, tuple[bytes, ...]], list[tuple[bytes, bytes]]]:
    """
    Returns tuple of:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    assert vocab_size >= INITIAL_VOCAB_SIZE, (
        f"Vocab size must be at least {INITIAL_VOCAB_SIZE} to account for all ascii characters"
    )

    # Make the first INITIAL_VOCAB_SIZE vocab entries
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(INITIAL_VOCAB_SIZE)}  # code point -> bytes

    pretokens_counts = get_pretoken_counts(input_path, special_tokens)  # tuple(int, ...) -> int

    merges = []
    num_merges = vocab_size - INITIAL_VOCAB_SIZE - len(special_tokens)
    for i in range(num_merges):
        # (1) Get counts
        counts = get_pair_counts(pretokens_counts)

        # (2) Find the highest frequency counts (resolve ties)
        max_count = 0
        to_merge_pair = None
        for code_point_pair, count in counts.items():
            if (to_merge_pair is None) or (count > max_count) or (count == max_count and is_lexigraphically_greater(vocab, code_point_pair, to_merge_pair)):
                to_merge_pair = code_point_pair
                max_count = count

        assert to_merge_pair is not None

        # (3) Add to vocab.
        new_bytes_token = vocab[to_merge_pair[0]] + vocab[to_merge_pair[1]]
        new_code_point = len(vocab)

        vocab[new_code_point] = new_bytes_token
        merges.append(convert_to_byte_space(vocab, to_merge_pair))

        pretokens_counts = merge_code_points_in_pretokens(pretokens_counts, to_merge_pair, new_code_point)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges

def main():
    # tiny_stores_dataset = "data/swift.txt"
    tiny_stores_dataset = "data/TinyStoriesV2-GPT4-valid.txt"
    VOCAB_SIZE = 1000
    vocab, merges = train_tokenizer(tiny_stores_dataset, VOCAB_SIZE, SPECIAL_TOKENS)

if __name__ == "__main__":
    main()
