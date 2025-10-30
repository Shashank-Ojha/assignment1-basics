import os
from typing import BinaryIO, Tuple, Dict
import regex as re
from collections import defaultdict
import multiprocessing

SPECIAL_TOKENS = ["<|beginoftext|>", "<|endoftext|>"]
NUM_WORKERS = 20


def split_on_special_tokens(text, special_tokens) -> list[str]:
    # Escape each special token
    escaped = [re.escape(tok) for tok in special_tokens]

    # Join them with '|' to mean “match any of these”
    pattern = "|".join(escaped)

    return re.split(pattern, text)


def pretokenize(text, special_tokens):
    # Pretokenize pattern.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    split_texts = split_on_special_tokens(text, special_tokens)

    pretoken_counts = defaultdict(int)  # type: Dict[tuple[bytes], int]
    for split_text in split_texts:
        iterable = re.finditer(PAT, split_text)
        for match in iterable:
            bytes_tuple = tuple([bytes(ch, "utf-8") for ch in match.group()])
            pretoken_counts[bytes_tuple] += 1

    return pretoken_counts


def read_chunk_and_pretokenize(filename, start, end, special_tokens):
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens)


def get_pretoken_counts(filename, special_tokens) -> dict[tuple[bytes], int]:
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
        pretoken_counts.update(counts)

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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time (2^12)

    for bi in range(1, len(chunk_boundaries) - 1):  # if num_chunks = 4, this will loop through 1, 2, 3,
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


def get_counts(pretokens_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Get aggregate counts from all preotkens
    """
    bp_counts = defaultdict(int)  # (bytes, bytes) -> count

    for bytes_tuple, count in pretokens_counts.items():
        for b1, b2 in zip(bytes_tuple, bytes_tuple[1:]):
            bp_counts[(b1, b2)] += count

    return bp_counts


def merge_bytes_in_pretokens(pretokens_counts, to_merge_pair, new_bytes_token):
    new_pretoken_counts = {}
    for bytes_tuple, count in pretokens_counts.items():
        i = 0
        new_bytes_list = []
        while i < len(bytes_tuple):
            if (
                i + 1 < len(bytes_tuple)
                and bytes_tuple[i] == to_merge_pair[0]
                and bytes_tuple[i + 1] == to_merge_pair[1]
            ):
                new_bytes_list.append(new_bytes_token)
                i += 2
            else:
                new_bytes_list.append(bytes_tuple[i])
                i += 1

        new_pretoken_counts[tuple(new_bytes_list)] = count

    return new_pretoken_counts


def train_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Returns tuple of:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    DEFAULT_VOCAB_SIZE = 256

    assert vocab_size >= DEFAULT_VOCAB_SIZE, (
        f"Vocab size must be at least {DEFAULT_VOCAB_SIZE} to account for all ascii characters"
    )

    # Make the first DEFAULT_VOCAB_SIZE vocab entries
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(DEFAULT_VOCAB_SIZE)}  # index -> bytes
    merges = []

    num_special_tokens = len(special_tokens)
    num_merges = vocab_size - DEFAULT_VOCAB_SIZE - num_special_tokens

    pretokens_counts = get_pretoken_counts(input_path, special_tokens)  # tuple(bytes) -> int

    for i in range(num_merges):
        # (1) Get counts
        counts = get_counts(pretokens_counts)

        # (2) Find the highest frequency counts (resolve ties)
        max_count = 0
        to_merge_pair = None
        for byte_pair, count in counts.items():
            if (to_merge_pair is None) or (count > max_count) or (count == max_count and byte_pair > to_merge_pair):
                to_merge_pair = byte_pair
                max_count = count

        assert to_merge_pair is not None

        # (3) Add to vocab. Key is len(vocab) and value is bytes
        new_bytes_token = to_merge_pair[0] + to_merge_pair[1]

        new_index = len(vocab)
        vocab[new_index] = new_bytes_token
        merges.append(to_merge_pair)

        pretokens_counts = merge_bytes_in_pretokens(pretokens_counts, to_merge_pair, new_bytes_token)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges


if __name__ == "__main__":
    tiny_stores_dataset = "data/TinyStoriesV2-GPT4-valid.txt"
    VOCAB_SIZE = 260
    # read_file("data/TinyStoriesV2-GPT4-valid.txt")
    vocab, merges = train_tokenizer(tiny_stores_dataset, VOCAB_SIZE, SPECIAL_TOKENS)
