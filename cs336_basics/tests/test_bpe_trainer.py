import pytest

from cs336_basics.tokenization.bpe_trainer_helpers import split_on_special_tokens

def test_split_on_special_tokens():
    text = "[Doc 1]<|endoftext|>[Doc 2]"
    special_tokens = ["<|beginoftext|>", "<|endoftext|>"]

    result = split_on_special_tokens(text, special_tokens)

    assert result == ["[Doc 1]", "[Doc 2]"]


if __name__ == "__main__":
    pytest.main([__file__])