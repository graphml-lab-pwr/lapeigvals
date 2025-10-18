import pytest
import torch
from torch import Tensor

from hallucinations.features.answers import remove_padding_tokens_from_generated_tokens


@pytest.mark.parametrize(
    "generated_tokens,pad_token_id,expected_result",
    [
        (
            torch.tensor([[1, 2, 3, 0, 0], [0, 0, 1, 2, 3], [1, 2, 0, 0, 3]]),
            0,
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 2, 0, 0, 3]),
            ],
        ),
        (
            torch.tensor([[5, 6, 7, 8, 8], [8, 5, 6, 7, 8]]),
            8,
            [
                torch.tensor([5, 6, 7]),
                torch.tensor([5, 6, 7]),
            ],
        ),
    ],
)
def test_remove_padding_tokens_from_generated_tokens(
    generated_tokens: Tensor,
    pad_token_id: int,
    expected_result: list[Tensor],
) -> None:
    """Test removing padding tokens from generated tokens."""
    result = remove_padding_tokens_from_generated_tokens(
        generated_tokens=generated_tokens,
        pad_token_id=pad_token_id,
    )

    assert len(result) == len(expected_result)
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp)
