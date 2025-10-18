from pathlib import Path
from typing import Any

import torch
import typer
from loguru import logger
from tqdm import tqdm, trange

from hallucinations.dirs import DatasetDir, ResultsDir
from hallucinations.features.attention_weights import (
    attention_diagonal,
    laplacian_diagonal_from_attn,
    yield_stacked_attentions,
)
from hallucinations.features.attn_feats import (
    get_attn_eigvals_per_head_topk,
    get_attn_log_det,
    get_laplacian_eigvals_per_head_topk,
)
from hallucinations.probe_models.lr import train_logistic_regression

RANDOM_SEED = 42
TOP_K_EIGVALS = [5, 10, 25, 50, 100]


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    all_layers_only: bool = typer.Option(False, help="Whether to only train on all layers"),
    pca_dim: int | None = typer.Option(None, help="Dimension of PCA"),
    use_cuda: bool = typer.Option(False, help="Whether to use CUDA"),
) -> None:
    ds_dir = DatasetDir(dataset_dir)
    results_dir = ResultsDir.from_dataset_dir(ds_dir)
    attn_diags, laplacian_diags = load_and_prepare_data(ds_dir)
    labels = ds_dir.load_labels()["labels"]
    split = ds_dir.load_split()

    top_k_eigvals = [
        eigval
        for eigval in TOP_K_EIGVALS
        if eigval <= min(attn_diag.size(-1) for attn_diag in attn_diags)
    ]
    attn_results = train_on_attn(
        attn_diags,
        labels,
        split,
        top_k_eigvals=top_k_eigvals,
        all_layers_only=all_layers_only,
        pca_dim=pca_dim,
        use_cuda=use_cuda,
    )
    laplacian_results = train_on_laplacian(
        laplacian_diags,
        labels,
        split,
        top_k_eigvals=top_k_eigvals,
        all_layers_only=all_layers_only,
        pca_dim=pca_dim,
        use_cuda=use_cuda,
    )
    results = attn_results | laplacian_results

    results_file = results_dir.attn_vs_lap_file(pca_dim)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, results_file)


def load_and_prepare_data(
    ds_dir: DatasetDir,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    attn_diags = []
    laplacian_diags = []

    if ds_dir.attn_diags_file.exists() and ds_dir.laplacian_diags_file.exists():
        logger.info("Loading cached diags...")
        attn_diags = torch.load(ds_dir.attn_diags_file, weights_only=True)
        laplacian_diags = torch.load(ds_dir.laplacian_diags_file, weights_only=True)
    else:
        logger.info("Cached diags not found. Computing diags from attention matrices...")
        for attn_shard in yield_stacked_attentions(ds_dir, remove_padding=True):
            for attn_example in tqdm(attn_shard, desc="attn shards", leave=False):
                attn_diags.append(attention_diagonal(attn_example))
                laplacian_diags.append(
                    laplacian_diagonal_from_attn(attn_example, vertical_edges=False)
                )
        torch.save(attn_diags, ds_dir.attn_diags_file)
        torch.save(laplacian_diags, ds_dir.laplacian_diags_file)

    attn_diags = [diag.float() for diag in attn_diags]
    laplacian_diags = [diag.float() for diag in laplacian_diags]

    return attn_diags, laplacian_diags


def train_on_attn(
    attn_diags: list[torch.Tensor],
    labels: torch.Tensor,
    split: dict[str, torch.Tensor],
    top_k_eigvals: list[int],
    all_layers_only: bool = False,
    pca_dim: int | None = None,
    use_cuda: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    ### --------- PER LAYER --------- ###
    logger.info("[ATTN][PER LAYER] Training...")
    attn_log_det_per_layer_results = []
    attn_eigval_topk_per_layer_results = []

    if not all_layers_only:
        for layer_idx in trange(attn_diags[0].size(0), desc="[ATTN][PER LAYER]"):
            features = get_attn_log_det(
                attn_diags=attn_diags,
                layer_idx=layer_idx,
            )
            layer_log_det_results = train_logistic_regression(
                features=features,
                labels=labels,
                split=split,
                pca_dim=pca_dim,
                random_seed=RANDOM_SEED,
                use_cuda=use_cuda,
            )
            layer_log_det_results["metadata"] |= {"layer_idx": layer_idx}
            attn_log_det_per_layer_results.append(layer_log_det_results)

            for top_k in tqdm(top_k_eigvals, desc="[ATTN][PER LAYER][TOP-K]", leave=False):
                features = get_attn_eigvals_per_head_topk(
                    attn_diags=attn_diags,
                    layer_idx=layer_idx,
                    top_k=top_k,
                )

                layer_eig_results = train_logistic_regression(
                    features=features,
                    labels=labels,
                    split=split,
                    pca_dim=pca_dim,
                    random_seed=RANDOM_SEED,
                    use_cuda=use_cuda,
                )
                layer_eig_results["metadata"] |= {"layer_idx": layer_idx, "top_eigval": top_k}
                attn_eigval_topk_per_layer_results.append(layer_eig_results)

    ### --------- ALL LAYERS --------- ###
    logger.info("[ATTN][ALL LAYERS] Training on log det features...")
    features = get_attn_log_det(
        attn_diags=attn_diags,
        layer_idx=None,
    )
    attn_log_det_all_layers = [
        train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            random_seed=RANDOM_SEED,
            use_cuda=use_cuda,
        )
    ]

    logger.info("[ATTN][ALL LAYERS] Training on top eigvals features...")
    attn_eigval_topk_all_layers = []
    for top_k in tqdm(top_k_eigvals, desc="[ATTN][ALL LAYERS][TOP-K]", leave=False):
        features = get_attn_eigvals_per_head_topk(
            attn_diags=attn_diags,
            layer_idx=None,
            top_k=top_k,
        )
        eig_results = train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            random_seed=RANDOM_SEED,
            use_cuda=use_cuda,
        )
        eig_results["metadata"] |= {"top_eigval": top_k}
        attn_eigval_topk_all_layers.append(eig_results)

    return {
        "attn_log_det_per_layer": attn_log_det_per_layer_results,
        "attn_log_det_all_layers": attn_log_det_all_layers,
        "attn_eigval_topk_per_layer": attn_eigval_topk_per_layer_results,
        "attn_eigval_topk_all_layers": attn_eigval_topk_all_layers,
    }


def train_on_laplacian(
    laplacian_diags: list[torch.Tensor],
    labels: torch.Tensor,
    split: dict[str, torch.Tensor],
    top_k_eigvals: list[int],
    all_layers_only: bool = False,
    pca_dim: int | None = None,
    use_cuda: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    ### --------- PER LAYER --------- ###
    logger.info("[LAPLACIAN][PER LAYER] Training...")
    laplacian_eigvals_topk_results_per_layer = []

    if not all_layers_only:
        for layer_idx in trange(laplacian_diags[0].size(0), desc="[LAPLACIAN][PER LAYER]"):
            for top_k in tqdm(top_k_eigvals, desc="[LAPLACIAN][PER LAYER][TOP-K]", leave=False):
                features = get_laplacian_eigvals_per_head_topk(
                    laplacian_diags=laplacian_diags,
                    layer_idx=layer_idx,
                    top_k=top_k,
                )

                layer_top_eigvals_results = train_logistic_regression(
                    features=features,
                    labels=labels,
                    split=split,
                    pca_dim=pca_dim,
                    random_seed=RANDOM_SEED,
                    use_cuda=use_cuda,
                )
                layer_top_eigvals_results["metadata"] |= {
                    "layer_idx": layer_idx,
                    "top_eigval": top_k,
                }
                laplacian_eigvals_topk_results_per_layer.append(layer_top_eigvals_results)

    ### --------- ALL LAYERS --------- ###
    logger.info("[LAPLACIAN][ALL LAYERS] Training on top eigvals features...")
    laplacian_eigvals_topk_results = []
    for top_k in tqdm(top_k_eigvals, desc="[LAPLACIAN][ALL LAYERS][TOP-K]", leave=False):
        features = get_laplacian_eigvals_per_head_topk(
            laplacian_diags=laplacian_diags,
            layer_idx=None,
            top_k=top_k,
        )
        layer_top_eigvals_results = train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            random_seed=RANDOM_SEED,
            use_cuda=use_cuda,
        )
        layer_top_eigvals_results["metadata"] |= {"top_eigval": top_k}
        laplacian_eigvals_topk_results.append(layer_top_eigvals_results)

    return {
        "laplacian_eigval_topk_per_layer": laplacian_eigvals_topk_results_per_layer,
        "laplacian_eigval_topk_all_layers": laplacian_eigvals_topk_results,
    }


if __name__ == "__main__":
    typer.run(main)
