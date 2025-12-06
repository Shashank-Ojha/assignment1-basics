import regex as re
from typing import BinaryIO
from collections import defaultdict
import os

PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def to_utf8_code_points_tuple(text: str) -> tuple[int, ...]:
    """
    Converts the text to tuple of code points (utf-8 encoding). We use utf-8
    because:
        (1) It ensures everything gets mapped to a sequence of code points where
            each code point is between 0-255
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

    # Similarly we decided to represent the bytes as their integer code points since
    # it performed faster as evidenced by cProfile
    # return tuple([bytes([b]) for b in text.encode("utf-8")])
    return tuple(text.encode("utf-8"))


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """
    Splits the text into pretokens, ensuring we never split a special token,
    and then returns the frequency of each pretoken which is represents as a tuple
    of code points.
    """
    split_texts = split_on_special_tokens(text, special_tokens)

    pretoken_counts = defaultdict(int)  # type: Dict[tuple[int, ...], int]
    for split_text in split_texts:
        iterable = re.finditer(PRETOKEN_PAT, split_text)
        for match in iterable:
            pretoken = match.group()
            code_points_tuple = to_utf8_code_points_tuple(pretoken)
            pretoken_counts[code_points_tuple] += 1

    return pretoken_counts


def read_chunk_and_pretokenize(
    filename: str, start: int, end: int, special_tokens: list[str]
) -> dict[tuple[int, ...], int]:
    """
    Reads the file between the start and end (exclusive) index and
    returns the frequency of each pretoken in that chunk.
    """
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens)
