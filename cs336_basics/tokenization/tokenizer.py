import pickle
from collections.abc import Iterable, Iterator
import regex as re

from cs336_basics.tokenization.bpe_trainer_helpers import PRETOKEN_PAT


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab  # Dict of 270 -> b'abc'
        self.merges = merges  # List of (b'a', b'c')

        self.special_tokens = special_tokens
        # Sort by length to handle overlapping special tokens. Longer ones should be first.
        if self.special_tokens:
            self.special_tokens.sort(key=lambda tok: len(tok), reverse=True)
            escaped = [re.escape(tok) for tok in self.special_tokens]
            self.special_token_pattern = re.compile("|".join(escaped))

        self.bytes_to_codepoint = {byte: code_point for code_point, byte in self.vocab.items()}

        self.merge_priority = {}
        for i, merge in enumerate(self.merges):
            self.merge_priority[merge] = i

        self.pattern = re.compile(PRETOKEN_PAT)

        # Store pretoken encodings as you go to avoid encoding the same pretokens over and over.
        self.pretoken_2_encoding_cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        code_point_list = self.pretoken_2_encoding_cache.get(pretoken)
        if code_point_list:
            return code_point_list

        pretoken_bytes = pretoken.encode("utf-8")
        if pretoken_bytes in self.bytes_to_codepoint:
            # It's a known word! Return the ID immediately.
            ids = [self.bytes_to_codepoint[pretoken_bytes]]
            self.pretoken_2_encoding_cache[pretoken] = ids
            return ids

        bytes_list = list([bytes([b]) for b in pretoken_bytes])

        bytes_list = self._apply_merges(bytes_list)

        code_point_list = [self.bytes_to_codepoint[b] for b in bytes_list]

        self.pretoken_2_encoding_cache[pretoken] = code_point_list
        return code_point_list

    def encode(self, text: str) -> list[int]:
        encoded = []

        token_iterable = iter([]) if not self.special_tokens else self.special_token_pattern.finditer(text)

        start_idx = 0
        token_match = next(token_iterable, None)
        while token_match:
            end_idx = token_match.start()
            chunk = text[start_idx:end_idx]

            pretoken_iterable = self.pattern.finditer(chunk)
            for pretoken_match in pretoken_iterable:
                pretoken = pretoken_match.group()
                encoded.extend(self._encode_pretoken(pretoken))

            current_special_token = token_match.group()
            encoded.append(self.bytes_to_codepoint[current_special_token.encode("utf-8")])

            start_idx = token_match.end()

            token_match = next(token_iterable, None)

        # Process last text section.
        chunk = text[start_idx:]
        if chunk:
            pretoken_iterable = self.pattern.finditer(chunk)
            for pretoken_match in pretoken_iterable:
                pretoken = pretoken_match.group()
                encoded.extend(self._encode_pretoken(pretoken))

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for subtext in iterable:
            encoded_subtext = self.encode(subtext)
            # Yield from essentially only yield one element of the inner iterable from self.encode
            yield from encoded_subtext

    def decode(self, ids: list[int]) -> str:
        full_bytes = b""
        for code_point in ids:
            full_bytes += self.vocab[code_point]

        return bytes.decode(full_bytes, errors="replace")

    def _apply_merges(self, bytes_list: tuple[bytes, ...]) -> tuple[int, ...]:
        while len(bytes_list) > 1:
            # Find all pairs that can be merged
            pairs_to_merge = []
            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i + 1])
                if pair in self.merge_priority:
                    # Found a mergeable pair, record it with its priority
                    pairs_to_merge.append((self.merge_priority[pair], i, pair))

            if not pairs_to_merge:
                # No more merges apply
                break

            # Find the highest priority merge (lowest number = earliest in merge list)
            best_match = min(pairs_to_merge, key=lambda x: x[0])
            pair_to_merge = best_match[2]
            head, tail = pair_to_merge

            new_pre_token = []
            i = 0
            while i < len(bytes_list):
                if i < len(bytes_list) - 1 and bytes_list[i] == head and bytes_list[i + 1] == tail:
                    new_pre_token.append(pair_to_merge[0] + pair_to_merge[1])
                    i += 2
                else:
                    new_pre_token.append(bytes_list[i])
                    i += 1
            bytes_list = tuple(new_pre_token)

        return bytes_list
