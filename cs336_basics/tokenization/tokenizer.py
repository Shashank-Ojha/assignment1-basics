import pickle
from typing import Iterable, Iterator
import regex as re

from cs336_basics.tokenization.bpe_trainer_helpers import (
    read_chunk_and_pretokenize, 
    split_on_special_tokens,
    find_chunk_boundaries,
    PRETOKEN_PAT
)

def merge_bytes_list_on_pair(bytes_list, pair):
    i = 0
    new_bytes_list = []

    new_byte = pair[0] + pair[1]

    while i < len(bytes_list):
        if (
            i + 1 < len(bytes_list)
            and bytes_list[i] == pair[0]
            and bytes_list[i + 1] == pair[1]
        ):
            new_bytes_list.append(new_byte)
            i += 2
        else:
            new_bytes_list.append(bytes_list[i])
            i += 1

    return new_bytes_list


def get_chunks_split_by_special_tokens(text: str, special_tokens: list[str] | None = None):
    if not special_tokens:
        return iter([]) # Return empty iterable to indicate no special token matches

    escaped = [re.escape(tok) for tok in special_tokens]

    # Join them with '|' to mean “match any of these”
    pattern = "|".join(escaped)

    return re.finditer(pattern, text)


class BPETokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab  # Dict of 270 -> b'abc'
        self.merges = merges # List of (b'a', b'c')

        self.special_tokens = special_tokens
        # Sort by length to handle overlapping special tokens. Longer ones should be first.
        if self.special_tokens:
            self.special_tokens.sort(key=lambda tok: len(tok), reverse=True)

        self.bytes_to_codepoint = {byte: code_point for code_point, byte in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)


        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    
    def encode_token(self, token: str) -> list[int]:
        bytes_list = list([bytes([b]) for b in token.encode("utf-8")])

        # TODO, we could probably make this faster, but okay for now.
        for byte_pair in self.merges:
            bytes_list = merge_bytes_list_on_pair(bytes_list, byte_pair)

        code_point_list = [self.bytes_to_codepoint[b] for b in bytes_list]
        return code_point_list


    def encode(self, text: str) -> list[int]:
        encoded = []

        token_iterable = get_chunks_split_by_special_tokens(text, self.special_tokens)

        start_idx = 0
        token_match = next(token_iterable, None)
        while token_match:
            end_idx = token_match.start()
            chunk = text[start_idx:end_idx]

            pretoken_iterable = re.finditer(PRETOKEN_PAT, chunk)
            for pretoken_match in pretoken_iterable:
                pretoken = pretoken_match.group()
                encoded.extend(self.encode_token(pretoken))

            current_special_token = token_match.group()
            encoded.append(self.bytes_to_codepoint[current_special_token.encode("utf-8")])

            start_idx = token_match.end()

            token_match = next(token_iterable, None)

        # Process last text section.
        chunk = text[start_idx:]
        if chunk:
            pretoken_iterable = re.finditer(PRETOKEN_PAT, chunk)
            for pretoken_match in pretoken_iterable:
                pretoken = pretoken_match.group()
                encoded.extend(self.encode_token(pretoken))


        return encoded        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        encoded = []

        subtext = next(iterable, None)
        while subtext:
            encoded_subtext = self.encode(subtext)
            encoded.extend(encoded_subtext)

            subtext = next(iterable, None)
        
        return encoded


    def decode(self, ids: list[int]) -> str:
        full_bytes = b''
        for code_point in ids:
            full_bytes += self.vocab[code_point]

        
        return bytes.decode(full_bytes, errors='replace')
