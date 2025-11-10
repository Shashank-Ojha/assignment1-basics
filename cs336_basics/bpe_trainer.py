import os
from typing import BinaryIO, Tuple, Dict
import regex as re
from collections import defaultdict
import multiprocessing

SPECIAL_TOKENS = ["<|beginoftext|>", "<|endoftext|>"]

INITIAL_VOCAB_SIZE = 256
NUM_WORKERS = 20

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Splits the text on any of the speical tokens passed in. Returns the list
    of strings after the split.
    """
    # Escape each special token
    escaped = [re.escape(tok) for tok in special_tokens]

    # Join them with '|' to mean “match any of these”
    pattern = "|".join(escaped)

    return re.split(pattern, text)


def to_utf8_bytes_tuple(text: str) -> tuple[bytes, ...]:
    """
    Converts the text to tuple of bytes (utf-8 encoding). We use utf-8
    because:
        (1) It ensures everything gets mapped to a sequence of bytes where
            each byte is between 0-255
        (2) It's the most compressed way to represent any character. utf-16 and 
            utf-32 take more bytes to represent the same character.
    """
    # Note that the original code below was wrong. 
    #   return tuple([bytes(ch, "utf-8") for ch in text])
    # The issue was that characters that were represented at multiple bytes would
    # turn into one merged element in the tuple, so we couldn't merge those subbytes
    # For example, the character '≈' is represented as bytes [\xe2,\x89,\x88], but
    # the above code would return the tuple(\xe2\x89\x88, ) meaning xe2 and x89 
    # could never be merged.
    return tuple([bytes([b]) for b in text.encode("utf-8")])


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Splits the text into pretokens, ensuring we never split a special token, 
    and then returns the frequency of each pretoken which is represents as a tuple
    of bytes.
    """
    # Pretokenize pattern.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    split_texts = split_on_special_tokens(text, special_tokens)

    pretoken_counts = defaultdict(int)  # type: Dict[tuple[bytes, ...], int]
    for split_text in split_texts:
        iterable = re.finditer(PAT, split_text)
        for match in iterable:
            pretoken = match.group()
            bytes_tuple = to_utf8_bytes_tuple(pretoken)
            pretoken_counts[bytes_tuple] += 1

    return pretoken_counts


def read_chunk_and_pretokenize(
    filename: str, 
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Reads the file between the start and end (exclusive) index and 
    returns the frequency of each pretoken in that chunk.
    """
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens)


def get_pretoken_counts(filename, special_tokens) -> dict[tuple[bytes, ...], int]:
    """ 
    Pretokenizes all the text in the file and returns the frequency of each
    pretoken.

    Note that each pretoken is represented as a tuple of bytes.

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
        for bytes_tuple, count in counts.items():
            pretoken_counts[bytes_tuple] += count

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


def get_pair_counts(pretokens_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Get frequency of every byte pair. Only pairs within a pretoken are considered. Pairs between
    pretokens are not considered. Similarly, special tokens are also not considered.
    """
    bp_counts = defaultdict(int)  # (bytes, bytes) -> count

    for bytes_tuple, count in pretokens_counts.items():
        for b1, b2 in zip(bytes_tuple, bytes_tuple[1:]):
            bp_counts[(b1, b2)] += count

    return bp_counts


def merge_bytes_in_pretokens(pretokens_counts: dict[tuple[bytes, ...], int], to_merge_pair: tuple[bytes, bytes], new_bytes_token: bytes) -> dict[tuple[bytes, ...], int]:
    """
    Given pretoken counts (tuple of bytes -> counts), returns a new
    version where the tuple of bytes keys merge the to_merge_pair to the new_bytes_token.
    """
    new_pretoken_counts = defaultdict(int)
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

        new_pretoken_counts[tuple(new_bytes_list)] += count

    return new_pretoken_counts


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
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(INITIAL_VOCAB_SIZE)}  # index -> bytes

    pretokens_counts = get_pretoken_counts(input_path, special_tokens)  # tuple(bytes) -> int

    merges = []
    num_merges = vocab_size - INITIAL_VOCAB_SIZE - len(special_tokens)
    for i in range(num_merges):
        # (1) Get counts
        counts = get_pair_counts(pretokens_counts)

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

        vocab[len(vocab)] = new_bytes_token
        merges.append(to_merge_pair)

        pretokens_counts = merge_bytes_in_pretokens(pretokens_counts, to_merge_pair, new_bytes_token)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges


if __name__ == "__main__":
    tiny_stores_dataset = "data/TinyStoriesV2-GPT4-valid.txt"
    VOCAB_SIZE = 260
    vocab, merges = train_tokenizer(tiny_stores_dataset, VOCAB_SIZE, SPECIAL_TOKENS)
