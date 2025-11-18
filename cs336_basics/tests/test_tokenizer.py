import pytest

from cs336_basics.tokenizer import get_chunks_split_by_special_tokens

def test_split_on_special_tokens():
    text = "[Doc 1]<|endoftext|>[Doc 2]<|beginoftext|>[Doc 3]"
    special_tokens = ["<|beginoftext|>", "<|endoftext|>"]

    iterable = get_chunks_split_by_special_tokens(text, special_tokens)

    first_match = next(iterable)
    assert first_match.group() == "<|endoftext|>"
    assert first_match.start() == 7  # inclusive
    assert first_match.end() == 20 # exclusive
    assert text[7] == "<"
    assert text[20] == "["

    
    second_match = next(iterable)
    assert second_match.group() == "<|beginoftext|>"
    assert second_match.start() == 27 # inclusive
    assert second_match.end() == 42 # exclusive
    assert text[27] == "<"
    assert text[42] == "["

def test_overlapping_tokens():
    text = "abcabc"
    special_tokens = ["abcabc", "abc"] # ensure sorted order fixes things.
    iterable = get_chunks_split_by_special_tokens(text, special_tokens)

    first_match = next(iterable)
    assert first_match.group() == "abcabc"



if __name__ == "__main__":
    pytest.main([__file__])