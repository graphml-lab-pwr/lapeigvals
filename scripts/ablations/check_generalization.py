from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from torch import Tensor
from tqdm import tqdm

from hallucinations.dirs import DatasetDir
from hallucinations.features.attn_feats import (
    get_attn_eigvals_per_head_topk,
    get_attn_log_det,
    get_laplacian_eigvals_per_head_topk,
)

DEFAULT_LLM = "llama_3.1_8b_instruct"
DEFAULT_TEMP = "temp_1.0"
DEFAULT_SEED = "seed_42"
DEFAULT_TOP_EIGVALS = 100

FEW_SHOT_PROMPT = "{temp}__prompt_qa_short_few_shot_sep__{seed}"
GSM8K_PROMPT = "{temp}__prompt_qa_gsm8k__{seed}"


def main(
    llm: str = typer.Option(DEFAULT_LLM),
    temp: str = typer.Option(DEFAULT_TEMP),
    seed: str = typer.Option(DEFAULT_SEED),
    top_eigvals: int = typer.Option(DEFAULT_TOP_EIGVALS),
    pca_dim: str | None = typer.Option(None),
    output_file: Path = typer.Option(...),
) -> None:
    _pca_dim: int | None
    if isinstance(pca_dim, str) and (pca_dim == "null" or pca_dim == "None"):
        _pca_dim = None
    elif isinstance(pca_dim, str):
        _pca_dim = int(pca_dim)
    elif pca_dim is None:
        _pca_dim = None
    else:
        raise ValueError(f"Invalid PCA dimension: {pca_dim}")

    few_shot_prompt = FEW_SHOT_PROMPT.format(temp=temp, seed=seed)
    gsm8k_prompt = GSM8K_PROMPT.format(temp=temp, seed=seed)
    datasets = {
        "nq_open": DatasetDir(Path("data/activations/nq_open/") / llm / few_shot_prompt),
        "squad_v2": DatasetDir(Path("data/activations/squad_v2/") / llm / few_shot_prompt),
        "trivia_qa": DatasetDir(Path("data/activations/trivia_qa/") / llm / few_shot_prompt),
        "halueval_qa": DatasetDir(Path("data/activations/halueval_qa/") / llm / few_shot_prompt),
        "coqa": DatasetDir(Path("data/activations/coqa/") / llm / few_shot_prompt),
        "truthful_qa": DatasetDir(Path("data/activations/truthful_qa/") / llm / few_shot_prompt),
        "gsm8k": DatasetDir(Path("data/activations/gsm8k/") / llm / gsm8k_prompt),
    }

    attn_diags, laplacian_diags, labels, splits = load_input_features(datasets)
    attn_log_det_all_layers, attn_eigval_topk_all_layers, laplacian_eigval_topk_all_layers = (
        compute_features(attn_diags, laplacian_diags, top_eigvals)
    )

    generalization_metrics: list[dict[str, Any]] = []
    for main_dataset in tqdm(datasets.keys(), desc="Generalization on attention eigvals"):
        res = test_generalization(
            main_dataset=main_dataset,
            datasets=datasets,
            features=attn_eigval_topk_all_layers,
            labels=labels,
            splits=splits,
            pca_dim=_pca_dim,
        )
        for gm in res:
            gm["features"] = "attn_eigval_topk_all_layers"
            generalization_metrics.append(gm)

    for main_dataset in tqdm(datasets.keys(), desc="Generalization on laplacian eigvals"):
        gen_metrics = test_generalization(
            main_dataset=main_dataset,
            datasets=datasets,
            features=laplacian_eigval_topk_all_layers,
            labels=labels,
            splits=splits,
            pca_dim=_pca_dim,
        )
        for gm in gen_metrics:
            gm["features"] = "laplacian_eigval_topk_all_layers"
            generalization_metrics.append(gm)

    for main_dataset in tqdm(datasets.keys(), desc="Generalization on attn log det"):
        gen_metrics = test_generalization(
            main_dataset=main_dataset,
            datasets=datasets,
            features=attn_log_det_all_layers,
            labels=labels,
            splits=splits,
            pca_dim=_pca_dim,
        )
        for gm in gen_metrics:
            gm["features"] = "attn_log_det_all_layers"
            generalization_metrics.append(gm)

    df = pd.DataFrame(generalization_metrics)
    df.to_json(output_file, index=False, orient="records", indent=4)


def load_input_features(
    datasets: dict[str, DatasetDir],
) -> tuple[
    dict[str, list[torch.Tensor]],
    dict[str, list[torch.Tensor]],
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
]:
    attn_diags = {}
    laplacian_diags = {}
    for dataset_name, dataset_dir in datasets.items():
        attn_diags[dataset_name] = torch.load(dataset_dir.attn_diags_file, weights_only=True)
        laplacian_diags[dataset_name] = torch.load(
            dataset_dir.laplacian_diags_file, weights_only=True
        )

    labels = {}
    for dataset_name, dataset_dir in datasets.items():
        labels[dataset_name] = dataset_dir.load_labels()["labels"]

    splits = {}
    for dataset_name, dataset_dir in datasets.items():
        splits[dataset_name] = dataset_dir.load_split()

    return attn_diags, laplacian_diags, labels, splits


def compute_features(
    attn_diags: dict[str, list[Tensor]],
    laplacian_diags: dict[str, list[Tensor]],
    top_eigvals: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    attn_log_det_all_layers = {}
    attn_eigval_topk_all_layers = {}
    laplacian_eigval_topk_all_layers = {}
    for dataset_name in tqdm(attn_diags.keys(), desc="Computing features"):
        attn_log_det_all_layers[dataset_name] = get_attn_log_det(
            attn_diags[dataset_name],
            layer_idx=None,
        )
        attn_eigval_topk_all_layers[dataset_name] = get_attn_eigvals_per_head_topk(
            attn_diags[dataset_name],
            layer_idx=None,
            top_k=top_eigvals,
        )
        laplacian_eigval_topk_all_layers[dataset_name] = get_laplacian_eigvals_per_head_topk(
            laplacian_diags[dataset_name],
            layer_idx=None,
            top_k=top_eigvals,
        )

    return (
        attn_log_det_all_layers,
        attn_eigval_topk_all_layers,
        laplacian_eigval_topk_all_layers,
    )


def test_generalization(
    main_dataset: str,
    datasets: dict[str, DatasetDir],
    features: dict[str, Tensor],
    labels: dict[str, Tensor],
    splits: dict[str, dict[str, Tensor]],
    pca_dim: int | None = None,
) -> list[dict[str, Any]]:
    x_train, y_train = (
        features[main_dataset][splits[main_dataset]["train_idx"]].float().numpy(),
        labels[main_dataset][splits[main_dataset]["train_idx"]].numpy(),
    )
    x_test, y_test = (
        features[main_dataset][splits[main_dataset]["test_idx"]].float().numpy(),
        labels[main_dataset][splits[main_dataset]["test_idx"]].numpy(),
    )

    model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)
    if pca_dim is not None:
        model = Pipeline(
            [
                ("pca", PCA(n_components=pca_dim, random_state=42)),
                ("model", model),
            ]
        )

    model = model.fit(x_train, y_train)
    test_proba = model.predict_proba(x_test)
    test_preds = np.argmax(test_proba, axis=1)

    res = [
        {
            "train_dataset": main_dataset,
            "test_dataset": main_dataset,
            "test_auc": roc_auc_score(y_test, test_proba[:, 1]),
            "test_ap": average_precision_score(y_test, test_proba[:, 1]),
            "test_precision": precision_score(y_test, test_preds),
            "test_recall": recall_score(y_test, test_preds),
            "test_f1": f1_score(y_test, test_preds),
        },
    ]

    for dataset_name in datasets.keys():
        if dataset_name == main_dataset:
            continue
        x_test, y_test = (
            features[dataset_name][splits[dataset_name]["test_idx"]].float().numpy(),
            labels[dataset_name][splits[dataset_name]["test_idx"]].numpy(),
        )
        test_proba = model.predict_proba(x_test)
        test_preds = np.argmax(test_proba, axis=1)

        res.append(
            {
                "train_dataset": main_dataset,
                "test_dataset": dataset_name,
                "test_auc": roc_auc_score(y_test, test_proba[:, 1]),
                "test_ap": average_precision_score(y_test, test_proba[:, 1]),
                "test_precision": precision_score(y_test, test_preds),
                "test_recall": recall_score(y_test, test_preds),
                "test_f1": f1_score(y_test, test_preds),
            }
        )

    return res


if __name__ == "__main__":
    typer.run(main)
