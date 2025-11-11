import regex as re
from collections import defaultdict

SPECIAL_TOKENS = ["<|beginoftext|>", "<|endoftext|>"]

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
    # Pretokenize pattern.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    split_texts = split_on_special_tokens(text, special_tokens)

    pretoken_counts = defaultdict(int)  # type: Dict[tuple[int, ...], int]
    for split_text in split_texts:
        iterable = re.finditer(PAT, split_text)
        for match in iterable:
            pretoken = match.group()
            code_points_tuple = to_utf8_code_points_tuple(pretoken)
            pretoken_counts[code_points_tuple] += 1

    return pretoken_counts


def read_chunk_and_pretokenize(
    filename: str, 
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[tuple[int, ...], int]:
    """
    Reads the file between the start and end (exclusive) index and 
    returns the frequency of each pretoken in that chunk.
    """
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens)


def merge_code_points_in_pretokens_helper(code_points_tuple, count, to_merge_pair, new_code_point):
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

    return tuple(new_code_points_list), count
