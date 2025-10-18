from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from hallucinations.dirs import DatasetDir
from hallucinations.features.attn_feats import (
    get_attn_eigvals_per_head_topk,
    get_attn_log_det,
    get_laplacian_eigvals_per_head_topk,
)
from hallucinations.probe_models.lr import train_logistic_regression

DEFAULT_DATASET = "trivia_qa"
DEFAULT_PROMPT = "prompt_qa_short_few_shot_sep"
DEFAULT_TEMPERATURES = [0.1, 0.5, 1.0, 2.0]
DEFAULT_SEED = "seed_42"
DEFAULT_TOP_EIGVAL = 100
DEFAULT_LLM = "llama_3.1_8b_instruct"


SAMPLING_SEEDS = [42, 123, 456, 789, 1000, 1234, 5678, 9101, 1122, 1345]
NUM_SAMPLES_PER_CLASS = 1_000


def main(
    dataset: str = typer.Option(DEFAULT_DATASET),
    llm: str = typer.Option(DEFAULT_LLM),
    prompt: str = typer.Option(DEFAULT_PROMPT),
    seed: str = typer.Option(DEFAULT_SEED),
    temperatures: list[float] = typer.Option(DEFAULT_TEMPERATURES, "--temperatures", "-t"),
    top_eigval: int = typer.Option(DEFAULT_TOP_EIGVAL),
    pca_dim: int | None = typer.Option(None),
    output_file: Path = typer.Option(...),
) -> None:
    attn_log_det_features = {}
    attn_eigval_features = {}
    laplacian_eigval_features = {}
    labels = {}

    metadata_to_include = {
        "dataset": dataset,
        "prompt": prompt,
        "seed": seed,
        "top_eigval": top_eigval,
        "llm": llm,
    }

    for temp in tqdm(temperatures, desc="Loading and computing features"):
        res_dir = Path(f"data/activations/{dataset}/{llm}/temp_{temp}__{prompt}__{seed}")
        ds_dir = DatasetDir(res_dir)
        attn_diags, laplacian_diags, labels[temp], _ = load_input_features(ds_dir)
        attn_log_det_features[temp], attn_eigval_features[temp], laplacian_eigval_features[temp] = (
            compute_features(attn_diags, laplacian_diags, top_eigval)
        )

    results = train_probes(
        attn_log_det_features=attn_log_det_features,
        attn_eigval_features=attn_eigval_features,
        laplacian_eigval_features=laplacian_eigval_features,
        labels=labels,
        pca_dim=pca_dim,
        metadata_to_include=metadata_to_include,
    )

    torch.save(results, output_file)


def load_input_features(
    ds_dir: DatasetDir,
) -> tuple[list[Tensor], list[Tensor], Tensor, dict[str, Tensor]]:
    attn_diags = [adiag.float() for adiag in torch.load(ds_dir.attn_diags_file, weights_only=True)]
    laplacian_diags = [
        ldiag.float() for ldiag in torch.load(ds_dir.laplacian_diags_file, weights_only=True)
    ]
    labels = ds_dir.load_labels()["labels"]
    split = ds_dir.load_split()
    return attn_diags, laplacian_diags, labels, split


def compute_features(
    attn_diags: list[Tensor],
    laplacian_diags: list[Tensor],
    top_eigvals: int,
) -> tuple[Tensor, Tensor, Tensor]:
    attn_log_det_all_layers = get_attn_log_det(
        attn_diags,
        layer_idx=None,
    )
    attn_eigval_topk_all_layers = get_attn_eigvals_per_head_topk(
        attn_diags,
        layer_idx=None,
        top_k=top_eigvals,
    )
    laplacian_eigval_topk_all_layers = get_laplacian_eigvals_per_head_topk(
        laplacian_diags,
        layer_idx=None,
        top_k=top_eigvals,
    )

    return (
        attn_log_det_all_layers,
        attn_eigval_topk_all_layers,
        laplacian_eigval_topk_all_layers,
    )


def train_probes(
    attn_log_det_features: dict[float, Tensor],
    attn_eigval_features: dict[float, Tensor],
    laplacian_eigval_features: dict[float, Tensor],
    labels: dict[float, Tensor],
    pca_dim: int | None = None,
    metadata_to_include: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if metadata_to_include is None:
        metadata_to_include = {}

    results: list[dict[str, Any]] = []
    for temp_value in tqdm(attn_log_det_features.keys(), desc="Training probes"):
        temp_labels = labels[temp_value]
        hallu_idx = torch.argwhere(temp_labels == 1).flatten()
        non_hallu_idx = torch.argwhere(temp_labels == 0).flatten()
        assert len(hallu_idx) > NUM_SAMPLES_PER_CLASS
        assert len(non_hallu_idx) > NUM_SAMPLES_PER_CLASS

        for i, seed in tqdm(
            enumerate(SAMPLING_SEEDS),
            total=len(SAMPLING_SEEDS),
            desc="Sampling seeds",
            leave=False,
        ):
            # Sample balanced number of indices per class
            gen = np.random.default_rng(seed)
            hallu_sampled_idx = gen.choice(hallu_idx, size=NUM_SAMPLES_PER_CLASS, replace=False)
            non_hallu_sampled_idx = gen.choice(
                non_hallu_idx,
                size=NUM_SAMPLES_PER_CLASS,
                replace=False,
            )

            # Create splits from sampled indices
            sampled_idx = torch.tensor(np.concatenate([hallu_sampled_idx, non_hallu_sampled_idx]))
            sampled_labels = torch.tensor(
                np.concatenate(
                    [np.ones(hallu_sampled_idx.shape[0]), np.zeros(non_hallu_sampled_idx.shape[0])]
                )
            )
            sampled_train_idx, sampled_test_idx = train_test_split(
                np.arange(NUM_SAMPLES_PER_CLASS * 2),
                stratify=sampled_labels,
                test_size=0.2,
                random_state=seed,
            )
            sampled_split = {
                "train_idx": torch.tensor(sampled_train_idx),
                "test_idx": torch.tensor(sampled_test_idx),
            }

            # ---- Attn log det ----
            sampled_attn_log_det = attn_log_det_features[temp_value][sampled_idx]
            attn_log_det_results = train_logistic_regression(
                features=sampled_attn_log_det,
                labels=sampled_labels,
                split=sampled_split,
                pca_dim=pca_dim,
            )
            attn_log_det_results["metadata"] = {
                "exp_repeat": i,
                "temperature": temp_value,
                "probe": "attn_log_det",
                **metadata_to_include,
                **attn_log_det_results["metadata"],
            }
            results.append(attn_log_det_results)

            # ---- Attn eigval ----
            sampled_attn_eigval = attn_eigval_features[temp_value][sampled_idx]
            attn_eigval_results = train_logistic_regression(
                features=sampled_attn_eigval,
                labels=sampled_labels,
                split=sampled_split,
                pca_dim=pca_dim,
            )
            attn_eigval_results["metadata"] = {
                "exp_repeat": i,
                "temperature": temp_value,
                "probe": "attn_eigval",
                **metadata_to_include,
                **attn_eigval_results["metadata"],
            }
            results.append(attn_eigval_results)

            # ---- Laplacian eigval ----
            sampled_laplacian_eigval = laplacian_eigval_features[temp_value][sampled_idx]
            laplacian_eigval_results = train_logistic_regression(
                features=sampled_laplacian_eigval,
                labels=sampled_labels,
                split=sampled_split,
                pca_dim=pca_dim,
            )
            laplacian_eigval_results["metadata"] = {
                "exp_repeat": i,
                "temperature": temp_value,
                "probe": "laplacian_eigval",
                **metadata_to_include,
                **laplacian_eigval_results["metadata"],
            }
            results.append(laplacian_eigval_results)

    return results


if __name__ == "__main__":
    typer.run(main)
