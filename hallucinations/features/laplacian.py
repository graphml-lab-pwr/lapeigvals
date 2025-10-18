import torch
from torch import Tensor


def laplacian_diagonal_from_attn(
    item_attn: list[Tensor] | Tensor,
) -> Tensor:
    """Laplacian diagonal from attention.
    Input shape of item_attn is [#layers, [#heads x seq_length x seq_length]]
    Output shape is [#layers, #heads, #seq_length]
    """
    if len(item_attn) <= 1:
        raise ValueError(
            f"Laplacian is not defined for a number of layers {len(item_attn)} (<= 1)."
        )
    device = item_attn[0].device
    num_layers = len(item_attn)
    num_heads, num_tokens, _ = item_attn[0].size()

    num_out_edges = torch.flip(torch.arange(1, num_tokens + 1, device=device), dims=[-1])
    fst_layer_num_out_edges = num_out_edges + 1  # additional self loop on first layer
    last_layer_num_out_edges = num_out_edges - 1  # misses vertical upward edges
    diag_idx = torch.arange(num_tokens, device=device)

    laplacian_diag = torch.zeros(num_layers, num_heads, num_tokens, device=device)

    # Compute for first layer
    fst_layer_attn, sec_layer_attn, *_ = item_attn
    fst_layer_out_deg = fst_layer_attn.sum(dim=1) + sec_layer_attn[:, diag_idx, diag_idx]
    fst_layer_out_deg = fst_layer_out_deg / fst_layer_num_out_edges
    laplacian_diag[0] = (fst_layer_out_deg - fst_layer_attn[:, diag_idx, diag_idx]).clone()

    for layer_idx in range(1, num_layers - 1):
        layer_attn = item_attn[layer_idx]
        next_layer_attn = item_attn[layer_idx + 1]

        layer_attn[:, diag_idx, diag_idx] = 0.0
        layer_out_deg = layer_attn.sum(dim=1) + next_layer_attn[:, diag_idx, diag_idx]
        layer_out_deg = layer_out_deg / num_out_edges
        laplacian_diag[layer_idx] = layer_out_deg.clone()

    # Compute for last layer
    *_, last_layer_attn = item_attn
    last_layer_attn[:, diag_idx, diag_idx] = 0.0
    last_layer_out_deg = last_layer_attn.sum(dim=1)
    # as we zero-out the attention diagonal, we don't need to subtract it from the out-degree
    last_layer_out_deg = last_layer_out_deg / last_layer_num_out_edges

    # correct for zero division of last-layer token
    last_layer_out_deg[:, -1] = 0.0

    laplacian_diag[-1] = last_layer_out_deg.clone()

    return laplacian_diag


def full_laplacian_from_attn(
    item_attn: list[Tensor],
) -> Tensor:
    """Refined Laplacian.
    Here, we consider vertical edges to be weighted by self-loops (which indeed are compatible with horizontal edges).

    Input shape of item_attn is [#layers, [#heads x seq_length x seq_length]]
    Output shape is [#heads x (#layers * seq_length) x (#layers * seq_length)]
    """
    if len(item_attn) <= 1:
        raise NotImplementedError(
            f"Laplacian is not defined for a number of layers {len(item_attn)} (<= 1)."
        )

    device = item_attn[0].device
    num_layers = len(item_attn)
    num_heads, seq_length, _ = item_attn[0].size()

    block_diags = []
    for head_idx in range(num_heads):
        block_diags.append(
            torch.block_diag(*[item_attn[layer_idx][head_idx] for layer_idx in range(num_layers)])
        )
    attn_graph_adj = torch.stack(block_diags)

    diag_idx = torch.arange(seq_length, seq_length * num_layers, device=device)
    shifted_diag_idx = diag_idx - seq_length

    attn_graph_adj[:, diag_idx, shifted_diag_idx] = attn_graph_adj[:, diag_idx, diag_idx]
    attn_graph_adj[:, diag_idx, diag_idx] = 0

    normalized_out_degree = attn_graph_adj.sum(dim=1) / torch.count_nonzero(attn_graph_adj, dim=1)
    normalized_out_degree[torch.isnan(normalized_out_degree)] = 0.0

    laplacian = torch.diag_embed(normalized_out_degree) - attn_graph_adj

    return laplacian
