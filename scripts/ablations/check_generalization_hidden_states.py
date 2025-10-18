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
from hallucinations.features.hidden_states import HiddenStatesSelection, load_hidden_states

DEFAULT_LLM = "llama_3.1_8b_instruct"
DEFAULT_GEN_CONFIG = "temp_1.0__prompt_qa_short_few_shot_sep__seed_42"


def main(
    llm: str = typer.Option(DEFAULT_LLM),
    gen_config: str = typer.Option(DEFAULT_GEN_CONFIG),
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

    datasets = {
        "nq_open": DatasetDir(Path("data/activations/nq_open/") / llm / gen_config),
        "squad_v2": DatasetDir(Path("data/activations/squad_v2/") / llm / gen_config),
        "trivia_qa": DatasetDir(Path("data/activations/trivia_qa/") / llm / gen_config),
        "halueval_qa": DatasetDir(Path("data/activations/halueval_qa/") / llm / gen_config),
        "coqa": DatasetDir(Path("data/activations/coqa/") / llm / gen_config),
        "truthful_qa": DatasetDir(Path("data/activations/truthful_qa/") / llm / gen_config),
    }

    hidden_states, labels, splits = prepare_hidden_states_features(datasets)

    for ds_name in hidden_states.keys():
        hidden_states[ds_name]["hs_last_input_token"] = hidden_states[ds_name][
            "hs_last_input_token"
        ].flatten(start_dim=1)
        hidden_states[ds_name]["hs_last_generated_token"] = hidden_states[ds_name][
            "hs_last_generated_token"
        ].flatten(start_dim=1)

    generalization_metrics: list[dict[str, Any]] = []
    for main_dataset in tqdm(datasets.keys(), desc="Generalization on attention eigvals"):
        features = {
            ds_name: hidden_states[ds_name]["hs_last_input_token"] for ds_name in datasets.keys()
        }
        res = test_generalization(
            main_dataset=main_dataset,
            datasets=datasets,
            features=features,
            labels=labels,
            splits=splits,
            pca_dim=_pca_dim,
        )
        for gm in res:
            gm["features"] = "hidden_state_last_input_token_all_layers"
            generalization_metrics.append(gm)

    for main_dataset in tqdm(datasets.keys(), desc="Generalization on laplacian eigvals"):
        features = {
            ds_name: hidden_states[ds_name]["hs_last_generated_token"]
            for ds_name in datasets.keys()
        }
        gen_metrics = test_generalization(
            main_dataset=main_dataset,
            datasets=datasets,
            features=features,
            labels=labels,
            splits=splits,
            pca_dim=_pca_dim,
        )
        for gm in gen_metrics:
            gm["features"] = "hidden_state_last_generated_token_all_layers"
            generalization_metrics.append(gm)

    df = pd.DataFrame(generalization_metrics)
    df.to_json(output_file, index=False, orient="records", indent=4)


def prepare_hidden_states_features(
    datasets: dict[str, DatasetDir],
) -> tuple[
    dict[str, dict[str, Tensor]],
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
]:
    hidden_states = {}
    labels = {}
    splits = {}
    for dataset_name, dataset_dir in datasets.items():
        hidden_states[dataset_name] = _load_hidden_states_features(dataset_dir)
        labels[dataset_name] = dataset_dir.load_labels()["labels"]
        splits[dataset_name] = dataset_dir.load_split()

    return hidden_states, labels, splits


def _load_hidden_states_features(ds_dir: DatasetDir) -> dict[str, Tensor]:
    if ds_dir.hidden_states_for_last_input_last_gen_tokens_file.exists():
        hidden_states = torch.load(
            ds_dir.hidden_states_for_last_input_last_gen_tokens_file,
            weights_only=True,
            map_location="cpu",
        )
    else:
        hs_selection = HiddenStatesSelection(
            layer="all",
            hs_last_input_token=True,
            hs_last_generated_token=True,
        )
        hidden_states = load_hidden_states(ds_dir.hidden_states_dir, hs_selection)

        # eventually cast bfloat16 to regular float
        for layer_idx in range(len(hidden_states["hs_last_input_token"])):
            hidden_states["hs_last_input_token"][layer_idx] = hidden_states["hs_last_input_token"][
                layer_idx
            ].float()
            hidden_states["hs_last_generated_token"][layer_idx] = hidden_states[
                "hs_last_generated_token"
            ][layer_idx].float()

    return hidden_states


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
            "test_auc": roc_auc_score(y_test, test_proba[:, 1]).item(),
            "test_ap": average_precision_score(y_test, test_proba[:, 1]).item(),
            "test_precision": precision_score(y_test, test_preds).item(),
            "test_recall": recall_score(y_test, test_preds).item(),
            "test_f1": f1_score(y_test, test_preds).item(),
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
                "test_auc": roc_auc_score(y_test, test_proba[:, 1]).item(),
                "test_ap": average_precision_score(y_test, test_proba[:, 1]).item(),
                "test_precision": precision_score(y_test, test_preds).item(),
                "test_recall": recall_score(y_test, test_preds).item(),
                "test_f1": f1_score(y_test, test_preds).item(),
            }
        )

    return res


if __name__ == "__main__":
    typer.run(main)
