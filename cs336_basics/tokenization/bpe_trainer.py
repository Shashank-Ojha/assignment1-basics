import os
from collections import defaultdict
import multiprocessing
import heapq
from cs336_basics.tokenization.bpe_trainer_helpers import (
    read_chunk_and_pretokenize,
    find_chunk_boundaries,
)

INITIAL_VOCAB_SIZE = 256

NUM_PROCESSES = os.cpu_count()


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


class PairInfo:
    def __init__(self):
        self.count = 0
        self.associated_code_point_tuples = defaultdict(int)  # code_points_tuple to int


def get_pair_counts(pretokens_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """
    Get frequency of every code point pair. Only pairs within a pretoken are considered. Pairs between
    pretokens are not considered. Similarly, special tokens are also not considered.
    """
    pair_infos = defaultdict(PairInfo)

    for code_points_tuple, count in pretokens_counts.items():
        for b1, b2 in zip(code_points_tuple, code_points_tuple[1:]):
            info = pair_infos[(b1, b2)]
            info.count += count
            info.associated_code_point_tuples[code_points_tuple] += 1

    return pair_infos


# This will break for cases like aaaaaa and to_merge_pair is aa
def merge_code_points_in_pretokens(
    pretoken_counts,
    pair_infos,
    to_merge_pair: tuple[int, int],
    new_code_point: int,
    vocab,
    heap,
    pair2node,
) -> dict[tuple[int, ...], int]:
    """
    Given pretoken counts (tuple of code points -> counts), returns a new
    version where the tuple of code points keys merge the to_merge_pair to the new_code_point.
    """
    affected_tuples = list(pair_infos[to_merge_pair].associated_code_point_tuples.keys())

    modified_pairs = set()

    for code_points_tuple in affected_tuples:
        original_tuple_count = pretoken_counts[code_points_tuple]
        del pretoken_counts[code_points_tuple]

        # (1) First go through original pairs and update pair infos
        for b1, b2 in zip(code_points_tuple, code_points_tuple[1:]):
            modified_pairs.add((b1, b2))
            info = pair_infos[(b1, b2)]
            info.count -= original_tuple_count
            if info.count == 0:
                # This should eventually delete the to_merge_pair as well.
                del pair_infos[(b1, b2)]
            else:
                info.associated_code_point_tuples[code_points_tuple] -= 1
                if info.associated_code_point_tuples[code_points_tuple] == 0:
                    del info.associated_code_point_tuples[code_points_tuple]

        # (2) Do the merge
        new_code_points_list = []
        i = 0
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

        # (3) Finally go through new pairs and update pair infos
        new_code_points_tuple = tuple(new_code_points_list)
        for b1, b2 in zip(new_code_points_tuple, new_code_points_tuple[1:]):
            modified_pairs.add((b1, b2))
            info = pair_infos[(b1, b2)]
            info.count += original_tuple_count
            info.associated_code_point_tuples[new_code_points_tuple] += 1

        # Update pretokencounts
        pretoken_counts[new_code_points_tuple] = original_tuple_count

    for b1, b2 in modified_pairs:
        # Mark old node as dead
        if (b1, b2) in pair2node:
            node = pair2node[(b1, b2)]
            node.is_dead = True

        # Add new node
        count = pair_infos[(b1, b2)].count
        node = Node(count, (b1, b2), convert_to_byte_space(vocab, (b1, b2)))
        heapq.heappush(heap, node)
        pair2node[(b1, b2)] = node

    # These datastructures get modified in place so this isn't necessary, but
    # returning them makes it clearer from the caller side that they've changed.
    return pretoken_counts, pair_infos


def convert_to_byte_space(vocab, byte_pair):
    return (vocab[byte_pair[0]], vocab[byte_pair[1]])


def is_lexigraphically_greater(vocab, lht, rht):
    lht_byte_pair = convert_to_byte_space(vocab, lht)
    rht_byte_pair = convert_to_byte_space(vocab, rht)
    return lht_byte_pair > rht_byte_pair


class Node:
    def __init__(self, pair_count, code_point_pair, byte_pair):
        self.pair_count = pair_count
        self.code_point_pair = code_point_pair
        self.byte_pair = byte_pair
        self.is_dead = False

    def __lt__(self, other):
        # This function defines the heap ordering. The heap orders from "smallest" to "largest".
        # In otherwords, this function should identify if self is higher priority.
        return (self.pair_count, self.byte_pair) > (other.pair_count, other.byte_pair)


def create_pair_ordering(vocab, pair_infos):
    heap = []
    pair2node = {}

    for code_point_pair, info in pair_infos.items():
        node = Node(info.count, code_point_pair, convert_to_byte_space(vocab, code_point_pair))
        heapq.heappush(heap, node)
        pair2node[code_point_pair] = node

    return heap, pair2node


def find_max_pair(heap):
    while heap:
        node = heapq.heappop(heap)

        if node.is_dead:
            continue

        break

    assert node is not None
    return node.code_point_pair


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

    # State. All these variables need to be maintained each loop iteration.
    pretoken_counts = get_pretoken_counts(input_path, special_tokens)  # tuple(int, ...) -> int
    pair_infos = get_pair_counts(pretoken_counts)
    heap, pair2node = create_pair_ordering(vocab, pair_infos)

    merges = []
    num_merges = vocab_size - INITIAL_VOCAB_SIZE - len(special_tokens)
    for i in range(num_merges):
        # (1) Find the highest frequency counts (resolve ties)
        to_merge_pair = find_max_pair(heap)

        # (2) Add to vocab.
        new_bytes_token = vocab[to_merge_pair[0]] + vocab[to_merge_pair[1]]
        new_code_point = len(vocab)

        vocab[new_code_point] = new_bytes_token
        merges.append(convert_to_byte_space(vocab, to_merge_pair))

        # (3) Merge and update data structures accordingly.
        pretoken_counts, pair_infos = merge_code_points_in_pretokens(
            pretoken_counts,
            pair_infos,
            to_merge_pair,
            new_code_point,
            vocab,
            heap,
            pair2node,
        )

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges
