import pytest
import torch
from torch import Tensor

from hallucinations.features.processing import (
    left_right_contiguous_padding_mask,
    remove_padding_from_intermediate_states,
)


def test_remove_padding_from_attention_matrix_when_batch_size_is_1() -> None:
    # input has dimension [#layers, [batch_size x num_heads x sequence_length x sequence_length]]
    # output is expected to have shape: [#examples, [#layers, [num_heads x sequence_length x sequence_length]]]
    # we set it to 2 layers, 1 in batch, 4 heads, 3 tokens in sequence
    attention_matrix_non_padded = [
        torch.tril(torch.randn(1, 4, 3, 3), 0),
        torch.tril(torch.randn(1, 4, 3, 3), 0),
    ]
    attention_matrix_non_padded = [_tril_softmax(attn) for attn in attention_matrix_non_padded]
    generated_tokens = torch.tensor([[1, 1, 1, 1]])

    pad_token_id = 0
    tokenizer = None
    result = remove_padding_from_intermediate_states(
        per_layer_batched_data=attention_matrix_non_padded,
        data_type="attn",
        generated_tokens=generated_tokens,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )
    assert len(result) == 1  # 1 example
    assert len(result[0]) == 2  # 2 layers

    assert torch.allclose(
        result[0][0],
        attention_matrix_non_padded[0][0, :],
    )  # first layer, 4 heads
    assert torch.allclose(
        result[0][1],
        attention_matrix_non_padded[1][0, :],
    )  # second layer, 4 heads


@pytest.mark.parametrize(
    "in_data,expected_mask",
    [
        ([0, 0, 1, 0, 0], [True, True, False, True, True]),
        ([0, 1, 0, 1, 0], [True, False, False, False, True]),
        ([1, 1, 0, 2, 0], [False, False, False, False, True]),
        ([0, 0, 1, 2, 3], [True, True, False, False, False]),
    ],
)
def test_left_right_contiguous_padding_mask(in_data: list[int], expected_mask: list[int]) -> None:
    assert torch.allclose(
        left_right_contiguous_padding_mask(torch.tensor(in_data), 0),
        torch.tensor(expected_mask),
    )


def _tril_softmax(tril_matrix: Tensor) -> Tensor:
    masked_matrix = tril_matrix + torch.triu(
        torch.full_like(tril_matrix, float("-inf")), diagonal=1
    )
    return torch.softmax(masked_matrix, dim=-1)
