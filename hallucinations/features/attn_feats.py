import torch
from torch import Tensor

QUANTILES = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.95, 0.99])


def get_attn_log_det(
    attn_diags: list[Tensor],
    layer_idx: int | None,
) -> Tensor:
    """Computes log determinant of attention matrix.
    Returns
        - [#examples x (#layers * #heads)] if layer_idx is not None
        - [#examples x #heads] if layer_idx is None
    """
    if layer_idx is None:
        return torch.stack([a_diag.log().mean(dim=-1).flatten() for a_diag in attn_diags])
    else:
        return torch.stack([a_diag[layer_idx].log().mean(dim=-1) for a_diag in attn_diags])


def get_attn_eigvals_per_head_topk(
    attn_diags: list[Tensor],
    layer_idx: int | None,
    top_k: int,
) -> Tensor:
    """Computes top k eigenvalues of the attention matrix diagonal.
    Note attention matrix is always positive (no need to consider sign).
    Returns
        - [#examples x (#layers * #heads * top_k)] if layer_idx is None
        - [#examples x (#heads * top_k)] if layer_idx is not None
    """
    if layer_idx is None:
        return torch.stack(
            [
                eigvals.sort(dim=-1, descending=True).values[:, :, :top_k].flatten()
                for eigvals in attn_diags
            ]
        )
    else:
        return torch.stack(
            [
                eigvals.sort(dim=-1, descending=True).values[layer_idx, :, :top_k].flatten()
                for eigvals in attn_diags
            ]
        )


def get_laplacian_eigvals_per_head_topk(
    laplacian_diags: list[Tensor],
    layer_idx: int | None,
    top_k: int,
) -> Tensor:
    """Computes top k eigenvalues of the Laplacian.
    Returns [#examples x (#layers * #heads * top_k)]
    """
    if layer_idx is None:
        return torch.stack(
            [
                eigval.sort(dim=-1, descending=True).values[:, :, :top_k].flatten()
                for eigval in laplacian_diags
            ]
        )
    else:
        return torch.stack(
            [
                eigval.sort(dim=-1, descending=True).values[layer_idx, :, :top_k].flatten()
                for eigval in laplacian_diags
            ]
        )


def get_laplacian_eigvals_per_head_topk_stats(
    laplacian_diags: list[Tensor],
    layer_idx: int | None,
    top_k: int,
) -> Tensor:
    """Computes top k eigenvalues of the Laplacian.
    Returns [#examples x (#layers * #heads * top_k)]
    """
    if layer_idx is None:
        return torch.stack(
            [
                compute_stats_on_last_dim(eigval.sort(dim=-1, descending=True).values[:, :, :top_k])
                for eigval in laplacian_diags
            ]
        )
    else:
        raise NotImplementedError("Layer-specific stats not implemented")


def compute_stats_on_last_dim(data: Tensor) -> Tensor:
    assert data.dtype == torch.float32
    return torch.cat(
        [
            data.mean(dim=-1).unsqueeze(-1),
            data.std(dim=-1).unsqueeze(-1),
            torch.quantile(data, QUANTILES, dim=-1).permute(1, 2, 0),
        ],
        dim=-1,
    )
