import math

import pytest
import torch
from torch.nn import functional as F

from hallucinations.features.attention_weights import (
    laplacian_diagonal_from_attn,
    log_det_attnn_over_dataset,
    stack_attention_matrix,
)


def test_log_det_atnn() -> None:
    # input has dimension [#examples, #layers, [#heads x sequence_length x sequence_length]]
    attn_scores = [
        [
            torch.ones(4, 3, 3) * math.exp(1),
            torch.ones(4, 3, 3) * math.exp(1),
        ],
        [
            torch.ones(4, 3, 3) * math.exp(2),
            torch.ones(4, 3, 3) * math.exp(2),
        ],
        [
            torch.ones(4, 3, 3) * math.exp(3),
            torch.ones(4, 3, 3) * math.exp(3),
        ],
    ]
    result = log_det_attnn_over_dataset(attn_scores)
    assert torch.allclose(result[0], torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    assert torch.allclose(result[1], torch.tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]))
    assert torch.allclose(result[2], torch.tensor([[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]))


def test_stack_attention_matrix() -> None:
    # input has dimension (#num_gen_tokens, #num_layers, [batch_size x num_heads x sequence_length x sequence_length])
    attentions = (
        (torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8)),
        (torch.randn(3, 2, 1, 9), torch.randn(3, 2, 1, 9)),
        (torch.randn(3, 2, 1, 10), torch.randn(3, 2, 1, 10)),
    )
    result = stack_attention_matrix(attentions)
    assert len(result) == 2  # 2 layers

    target_layer_0 = torch.cat(
        [
            F.pad(attentions[0][0], (0, 2)),
            F.pad(attentions[1][0], (0, 1)),
            F.pad(attentions[2][0], (0, 0)),
        ],
        dim=-2,
    )

    target_layer_1 = torch.cat(
        [
            F.pad(attentions[0][1], (0, 2)),
            F.pad(attentions[1][1], (0, 1)),
            F.pad(attentions[2][1], (0, 0)),
        ],
        dim=-2,
    )
    assert torch.allclose(result[0], target_layer_0)
    assert torch.allclose(result[1], target_layer_1)


@pytest.mark.parametrize("vertical_edges", [True, False])
def test_laplacian_diagonal_from_attn(vertical_edges: bool) -> None:
    layer_0 = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
        ],
        dtype=torch.float32,
    )
    layer_1 = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
            ],
            [
                [10.0, 0.0, 0.0],
                [20.0, 10.0, 0.0],
                [30.0, 20.0, 10.0],
            ],
        ],
        dtype=torch.float32,
    )

    layer_2 = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
            ],
            [
                [100.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [100.0, 100.0, 100.0],
            ],
        ],
        dtype=torch.float32,
    )

    attention_scores = [layer_0, layer_1, layer_2]

    # Test without vertical edges
    result = laplacian_diagonal_from_attn(
        attention_scores,
        vertical_edges=vertical_edges,
        vertical_edge_weight=1.0,
    )

    if vertical_edges:
        l_0_h0 = [(3 / 3 - 1), (2 / 2 - 1), (1 / 1 - 1)]
        l_0_h1 = [(1 / 3 - 1), (2 / 2 - 2), (3 / 1 - 3)]
        l_1_h0 = [(7 / 4 - 1), (4 / 3 - 1), (2 / 2 - 1)]
        l_1_h1 = [(61 / 4 - 10), (31 / 3 - 10), (11 / 2 - 10)]
        l_2_h0 = [(7 / 4 - 1), (4 / 3 - 1), (2 / 2 - 1)]
        l_2_h1 = [(301 / 4 - 100), (201 / 3 - 100), (101 / 2 - 100)]
    else:
        l_0_h0 = [(3 / 3 - 1), (2 / 2 - 1), (1 / 1 - 1)]
        l_0_h1 = [(1 / 3 - 1), (2 / 2 - 2), (3 / 1 - 3)]
        l_1_h0 = [(6 / 3 - 1), (3 / 2 - 1), (1 / 1 - 1)]
        l_1_h1 = [(60 / 3 - 10), (30 / 2 - 10), (10 / 1 - 10)]
        l_2_h0 = [(6 / 3 - 1), (3 / 2 - 1), (1 / 1 - 1)]
        l_2_h1 = [(300 / 3 - 100), (200 / 2 - 100), (100 / 1 - 100)]
    expected_result = torch.tensor(
        [
            [l_0_h0, l_0_h1],
            [l_1_h0, l_1_h1],
            [l_2_h0, l_2_h1],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result, expected_result)
