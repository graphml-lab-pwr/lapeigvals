from pathlib import Path
from typing import Any

import torch
import typer
from loguru import logger
from torch import Tensor
from tqdm import trange

from hallucinations.dirs import DatasetDir, ResultsDir
from hallucinations.features.hidden_states import HiddenStatesSelection, load_hidden_states
from hallucinations.probe_models.lr import train_logistic_regression

RANDOM_SEED = 42


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    pca_dim: int | None = typer.Option(None, help="Dimension of PCA"),
    use_cuda: bool = typer.Option(False, help="Use CUDA"),
) -> None:
    ds_dir = DatasetDir(dataset_dir)
    results_dir = ResultsDir.from_dataset_dir(ds_dir)

    labels = ds_dir.load_labels()["labels"]
    split = ds_dir.load_split()
    hidden_states = prepare_hidden_states_features(ds_dir)

    results = train_on_token_hidden_states(
        hidden_states=hidden_states,
        labels=labels,
        split=split,
        pca_dim=pca_dim,
        use_cuda=use_cuda,
    )

    results_dir.root_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, results_dir.hs_baseline_file(pca_dim))


def prepare_hidden_states_features(ds_dir: DatasetDir) -> dict[str, Tensor]:
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


def train_on_token_hidden_states(
    hidden_states: dict[str, Tensor],
    labels: Tensor,
    split: dict[str, Tensor],
    pca_dim: int | None = None,
    use_cuda: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    num_layers = hidden_states["hs_last_input_token"].size(1)

    ### --------- PER LAYER --------- ###
    logger.info("[HS][last_input_token][per_layer] Training...")
    last_input_token_per_layer_results = []
    for layer_idx in trange(num_layers, desc="[HS][last_input_token]"):
        features = hidden_states["hs_last_input_token"][:, layer_idx]
        results = train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            use_cuda=use_cuda,
            random_seed=RANDOM_SEED,
        )
        results["metadata"] |= {"layer_idx": layer_idx}
        last_input_token_per_layer_results.append(results)

    logger.info("[HS][last_generated_token][per_layer] Training...")
    last_generated_token_per_layer_results = []
    for layer_idx in trange(num_layers, desc="[HS][last_generated_token]"):
        features = hidden_states["hs_last_generated_token"][:, layer_idx]
        results = train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            use_cuda=use_cuda,
            random_seed=RANDOM_SEED,
        )
        results["metadata"] |= {"layer_idx": layer_idx}
        last_generated_token_per_layer_results.append(results)

    ### --------- ALL LAYERS --------- ###
    logger.info("[HS][last_input_token][all_layers] Training...")
    features = hidden_states["hs_last_input_token"].flatten(start_dim=1)
    last_input_token_all_layers_results = [
        train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            use_cuda=use_cuda,
            random_seed=RANDOM_SEED,
        )
    ]

    logger.info("[HS][last_generated_token][all_layers] Training...")
    features = hidden_states["hs_last_generated_token"].flatten(start_dim=1)
    last_generated_token_all_layers_results = [
        train_logistic_regression(
            features=features,
            labels=labels,
            split=split,
            pca_dim=pca_dim,
            use_cuda=use_cuda,
            random_seed=RANDOM_SEED,
        )
    ]
    return {
        "hidden_state_last_input_token_per_layer": last_input_token_per_layer_results,
        "hidden_state_last_generated_token_per_layer": last_generated_token_per_layer_results,
        "hidden_state_last_input_token_all_layers": last_input_token_all_layers_results,
        "hidden_state_last_generated_token_all_layers": last_generated_token_all_layers_results,
    }


if __name__ == "__main__":
    typer.run(main)
