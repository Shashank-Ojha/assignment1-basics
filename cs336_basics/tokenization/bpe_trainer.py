import os
from collections import defaultdict
import multiprocessing

from cs336_basics.tokenization.bpe_trainer_helpers import (
    read_chunk_and_pretokenize,
    SPECIAL_TOKENS,
    merge_code_points_in_pretokens_helper,
    find_chunk_boundaries,
)

INITIAL_VOCAB_SIZE = 256

NUM_PROCESSES = os.cpu_count()
NUM_THREADS = min(32, 4 * os.cpu_count())


def get_pretoken_counts(filename, special_tokens) -> dict[tuple[int, ...], int]:
    """
    Pretokenizes all the text in the file and returns the frequency of each
    pretoken.

    Note that each pretoken is represented as a tuple of code points.

    @arg special_tokens - Used to ensure we never split the special tokens.
    """
    pretoken_counts = defaultdict(int)
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    args = [(filename, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(NUM_PROCESSES) as p:
        pretoken_counts_per_worker = p.starmap(read_chunk_and_pretokenize, args)

    # Merge all the counts
    for counts in pretoken_counts_per_worker:
        for code_points_tuple, count in counts.items():
            pretoken_counts[code_points_tuple] += count

    return pretoken_counts


# @TODO, use threads to make this faster. It's a huge bottle neck in the code right now.
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


def merge_code_points_in_pretokens(
    pretokens_counts: dict[tuple[int, ...], int], to_merge_pair: tuple[int, int], new_code_point: int
) -> dict[tuple[int, ...], int]:
    """
    Given pretoken counts (tuple of code points -> counts), returns a new
    version where the tuple of code points keys merge the to_merge_pair to the new_code_point.
    """
    # Case on the size of pretokens. For a small amount, adding threads just adds too much overhead
    # which fails the test cases.
    if len(pretokens_counts) < 5000:
        num_threads = 1
    else:
        num_threads = NUM_THREADS

    new_pretoken_counts = defaultdict(int)

    args = [
        (code_points_tuple, count, to_merge_pair, new_code_point)
        for code_points_tuple, count in pretokens_counts.items()
    ]
    with multiprocessing.pool.ThreadPool(num_threads) as p:
        pretoken_counts_after_merge = p.starmap(merge_code_points_in_pretokens_helper, args)

    # Merge all the counts
    for new_code_points_tuple, count in pretoken_counts_after_merge:
        new_pretoken_counts[new_code_points_tuple] += count

    return new_pretoken_counts


def convert_to_byte_space(vocab, byte_pair):
    return (vocab[byte_pair[0]], vocab[byte_pair[1]])


def is_lexigraphically_greater(vocab, lht, rht):
    lht_byte_pair = convert_to_byte_space(vocab, lht)
    rht_byte_pair = convert_to_byte_space(vocab, rht)
    return lht_byte_pair > rht_byte_pair


def train_tokenizer(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, tuple[bytes, ...]], list[tuple[bytes, bytes]]]:
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
            if (
                (to_merge_pair is None)
                or (count > max_count)
                or (count == max_count and is_lexigraphically_greater(vocab, code_point_pair, to_merge_pair))
            ):
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
    tiny_stores_dataset = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    vocab, merges = train_tokenizer(tiny_stores_dataset, VOCAB_SIZE, SPECIAL_TOKENS)


if __name__ == "__main__":
    main()
