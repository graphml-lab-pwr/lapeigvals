import os
from pathlib import Path
from typing import Generator

import psutil
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from hallucinations.dirs import DatasetDir
from hallucinations.features.processing import pad_tensor, remove_padding_from_intermediate_states
from hallucinations.utils.misc import load_and_resolve_config


def attention_diagonal(item_attn: list[Tensor]) -> Tensor:
    """Computes attention diagonal for single example from dataset.
    Input shape of item_attn is [#layers, [#heads x seq_length x seq_length]]
    Output shape is [#heads x (#layers * seq_length)]
    """
    return torch.stack([torch.diagonal(layer_attn, dim1=1, dim2=2) for layer_attn in item_attn])


def laplacian_diagonal_from_attn(
    item_attn: list[Tensor],
    vertical_edges: bool,
    vertical_edge_weight: float | None = None,
) -> Tensor:
    """Computes laplacian diagonal for single example from dataset.
    Input shape of item_attn is [#layers, [#heads x seq_length x seq_length]]
    Output shape is [#heads x (#layers * seq_length)]
    """
    device = item_attn[0].device
    if vertical_edges:
        assert vertical_edge_weight is not None
    # we treat attention matrix as weighted adjacency matrix
    # to obtain the laplacian we need to substract diagonal degree matrix from the adjacency matrix
    # I guess, we can ignore self-loops and use diagonal degree matrix only
    # to account for vertical edges, we add one to layers from the second onwards
    fst_layer_attn = item_attn[0]
    fst_nom = fst_layer_attn.sum(dim=1)
    fst_denom = torch.arange(1, fst_layer_attn.size(1) + 1, device=device).flip(dims=[0])
    # D := weighted out-degree
    fst_weighted_degree = fst_nom / fst_denom
    # L := D - A
    fst_lap = fst_weighted_degree - torch.diagonal(fst_layer_attn, offset=0, dim1=1, dim2=2)

    per_layer_laplacian_diags = [fst_lap]
    for layer_attn in item_attn[1:]:
        # per-layer weighted out-degree
        # for vertical edges, we set weight to constant
        if vertical_edges:
            assert vertical_edge_weight is not None
            nom = layer_attn.sum(dim=1) + vertical_edge_weight
            denom = torch.arange(1, layer_attn.size(1) + 1, device=device).flip(dims=[0]) + 1
        else:
            nom = layer_attn.sum(dim=1)
            denom = torch.arange(1, layer_attn.size(1) + 1, device=device).flip(dims=[0])

        layer_weighted_degree = nom / denom
        layer_lap_diag = layer_weighted_degree - torch.diagonal(
            layer_attn, offset=0, dim1=1, dim2=2
        )
        per_layer_laplacian_diags.append(layer_lap_diag)

    laplacian_diags = torch.stack(per_layer_laplacian_diags)

    return laplacian_diags


def log_det_attnn_over_dataset(attn_scores: list[list[Tensor]]) -> list[Tensor]:
    """Computes Attention Score (non-aggregated) over the dataset.
    Dimensions of the input are [#examples, #layers, [#heads x sequence_length x sequence_length]].
    Dimensions of the output are [#examples, [#layers x #heads]].
    """
    log_dets = []
    for example_attn in tqdm(attn_scores, desc="log-det(atnn)", leave=False):
        log_dets.append(log_det_attn(example_attn))
    return log_dets


def log_det_attn(attn_scores: list[Tensor]) -> Tensor:
    """Computes log-det(attn) for a single example.

    AttnScore was proposed in  https://openreview.net/forum?id=LYx4w3CAgy,
    Implementation: https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection/blob/2f3bf9ea6db19e60a416090f77694816c92a9146/common_utils.py#L319

    Attention Score is defined as:
    log(det(A)) = mean(log(diag(A))), where A is the attention matrix.
    In the original paper, the scores were additionally summed over heads and one layer was used.

    Dimensions of the input are [#layers, [num_heads x sequence_length x sequence_length]].
    Dimensions of the output Tensor is [#layers x #heads].
    """
    per_example_log_dets = []
    for layer_attn in attn_scores:
        per_head_log_det = torch.diagonal(layer_attn, dim1=1, dim2=2).log().mean(dim=1)
        per_example_log_dets.append(per_head_log_det)
    return torch.stack(per_example_log_dets)


def yield_stacked_attentions(
    dataset_dir: Path | DatasetDir,
    attentions_dir: Path | None = None,
    remove_padding: bool = False,
) -> Generator[list[list[Tensor]] | list[Tensor], None, None]:
    """Yields stacked attention scores without padding for all shards in the dataset.
    Dimensions of each shard is [#examples, [#layers, [num_heads x sequence_length x sequence_length]]].
    """
    ds_dir = DatasetDir(dataset_dir) if isinstance(dataset_dir, Path) else dataset_dir

    if remove_padding:
        config = load_and_resolve_config(ds_dir.config_file)
        tokenizer = AutoTokenizer.from_pretrained(config["llm"]["name"])
    else:
        tokenizer = None

    if attentions_dir is None:
        data_shards = list(ds_dir.attentions_dir.glob("*.pt"))
    else:
        data_shards = list(attentions_dir.glob("*.pt"))

    return yield_stacked_attentions_from_shard_list(data_shards, remove_padding, tokenizer)


def yield_stacked_attentions_from_shard_list(
    data_shards: list[Path],
    remove_padding: bool = False,
    tokenizer: PreTrainedTokenizer | None = None,
) -> Generator[list[list[Tensor]] | list[Tensor], None, None]:
    process = psutil.Process(os.getpid())

    with tqdm(data_shards, desc="Loading attentions", total=len(data_shards)) as pbar:
        for shard_file in pbar:
            memory_info = process.memory_info()
            pbar.set_postfix(dict(memory=f"{memory_info.rss / 1024**3:.2f} GB"))

            stacked_attn_scores, generated_tokens = load_and_stack_attentions_shard(shard_file)
            if remove_padding:
                attn_scores_without_padding = remove_padding_from_intermediate_states(
                    per_layer_batched_data=stacked_attn_scores,
                    data_type="attn",
                    generated_tokens=generated_tokens,
                    tokenizer=tokenizer,
                )
                yield attn_scores_without_padding
            else:
                yield stacked_attn_scores


def load_and_stack_attentions_shard(shard_file: Path) -> tuple[list[Tensor], Tensor]:
    data = torch.load(shard_file, weights_only=True, mmap=True, map_location="cpu")
    return stack_attention_matrix(data["attentions"]), data["generated_tokens"]


def stack_attention_matrix(attentions: tuple[tuple[Tensor, ...], ...]) -> list[Tensor]:
    """Stacks attention scores for all tokens and layers of a single example into a single tensor.
    Dimensions of the input are (#num_gen_tokens, #num_layers, [batch_size x num_heads x sequence_length x sequence_length]).
    Dimensions of the output are (#num_layers, [batch_size x num_heads x sequence_length x sequence_length]).
    """
    stacked_attn_all_tokens_per_single_layer: list[Tensor] = []
    num_gen_tokens = len(attentions)
    num_layers = len(attentions[0])
    for layer_idx in range(num_layers):
        seq_length = attentions[-1][0].size(-1)
        attn_scores_all_tokens_single_layer = [
            pad_tensor(attentions[token_idx][layer_idx], seq_length)
            for token_idx in range(num_gen_tokens)
        ]
        attn = torch.cat(attn_scores_all_tokens_single_layer, dim=-2)
        stacked_attn_all_tokens_per_single_layer.append(attn)

    return stacked_attn_all_tokens_per_single_layer
