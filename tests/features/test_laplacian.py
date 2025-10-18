from typing import List

import pytest
import torch

from hallucinations.features.laplacian import full_laplacian_from_attn, laplacian_diagonal_from_attn


@pytest.fixture
def item_attn_fixture() -> List[torch.Tensor]:
    return [
        torch.tensor(
            [
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
            ]
        ),
        torch.tensor(
            [
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                [
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
            ]
        ),
    ]


def test_full_laplacian_from_attn(item_attn_fixture: List[torch.Tensor]) -> None:
    laplacian = full_laplacian_from_attn(item_attn_fixture)

    target_laplacian = torch.tensor(
        [
            [
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, 0.0, 0.0, 0.0],
                [-0.50, (0.5 + 0.5) / 2 - 0.5, 0.0, 0.0],
                [-1.0, 0.0, 0.5, 0.0],
                [0.0, -0.5, -0.5, 0.0],
            ],
            [
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, 0.0, 0.0, 0.0],
                [-0.50, (0.5 + 0.5) / 2 - 0.5, 0.0, 0.0],
                [-1.0, 0.0, 0.5, 0.0],
                [0.0, -0.5, -0.5, 0.0],
            ],
            [
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, 0.0, 0.0, 0.0],
                [-0.50, (0.5 + 0.5) / 2 - 0.5, 0.0, 0.0],
                [-1.0, 0.0, 0.5, 0.0],
                [0.0, -0.5, -0.5, 0.0],
            ],
        ]
    )

    assert torch.allclose(laplacian, target_laplacian)


def test_laplacian_diagonal_from_attn(item_attn_fixture: List[torch.Tensor]) -> None:
    diagonal = laplacian_diagonal_from_attn(item_attn_fixture)
    target_diagonal = torch.tensor(
        [
            [
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, (0.5 + 0.5) / 2 - 0.5],
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, (0.5 + 0.5) / 2 - 0.5],
                [(1.0 + 1.0 + 0.5) / 3 - 1.0, (0.5 + 0.5) / 2 - 0.5],
            ],
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.5, 0.0],
            ],
        ]
    )

    assert torch.allclose(diagonal, target_diagonal)
