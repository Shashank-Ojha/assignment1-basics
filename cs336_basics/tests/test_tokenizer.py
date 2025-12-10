import pytest

from cs336_basics.tokenization.bpe_trainer_helpers import split_on_special_tokens


def test_split_on_special_tokens():
    text = "[Doc 1]<|endoftext|>[Doc 2]<|beginoftext|>[Doc 3]"
    special_tokens = ["<|beginoftext|>", "<|endoftext|>"]

    matches = split_on_special_tokens(text, special_tokens)

    assert matches[0] == "[Doc 1]"
    assert matches[1] == "[Doc 2]"
    assert matches[2] == "[Doc 3]"


def test_overlapping_tokens():
    text = "abcabc"
    special_tokens = ["abcabc", "abc"]  # ensure sorted order fixes things.
    matches = split_on_special_tokens(text, special_tokens)

    assert matches[0] == ""


if __name__ == "__main__":
    pytest.main([__file__])
