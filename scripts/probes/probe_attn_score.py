from pathlib import Path

import torch
import typer
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

from hallucinations.dirs import DatasetDir, ResultsDir
from hallucinations.features.attention_weights import yield_stacked_attentions

ATTN_SCORE_OUTPUT_FILENAME = "attn_score.pkl"


def main(dataset_dir: Path = typer.Option(..., help="Path to the dataset directory")) -> None:
    ds_dir = DatasetDir(dataset_dir)
    results_dir = ResultsDir.from_dataset_dir(ds_dir)
    attn_diags = load_and_prepare_data(ds_dir)
    labels = ds_dir.load_labels()["labels"]
    split = ds_dir.load_split()

    # Attn log-det as proposed in the LLMCheck paper
    # input_shape: [examples, layers, heads, seq_len]
    # output_shape: [examples, layers]

    ### PER LAYER ATTN SCORE ###
    attn_score = -torch.stack(
        [attn_diag.float().log().mean(dim=-1).sum(dim=-1) for attn_diag in attn_diags]
    )
    num_layers = attn_score[0].size(-1)
    per_layer_results = []
    for layer_idx in trange(num_layers, desc="Computing attn scores"):
        attn_scores = attn_score[:, layer_idx]
        test_auc = roc_auc_score(labels[split["test_idx"]], attn_scores[split["test_idx"]])
        train_auc = roc_auc_score(labels[split["train_idx"]], attn_scores[split["train_idx"]])
        per_layer_results.append(
            {
                "metadata": {"trainable_params": 0, "layer_idx": layer_idx},
                "metrics": {
                    "test_auc": test_auc.item(),
                    "train_auc": train_auc.item(),
                },
            }
        )

    ### ALL LAYERS ATTN SCORE ###
    attn_score_all_layers = attn_score.mean(dim=-1)
    test_auc = roc_auc_score(labels[split["test_idx"]], attn_score_all_layers[split["test_idx"]])
    train_auc = roc_auc_score(labels[split["train_idx"]], attn_score_all_layers[split["train_idx"]])
    all_layers_results = [
        {
            "metadata": {"trainable_params": 0, "layer_idx": "all"},
            "metrics": {
                "test_auc": test_auc.item(),
                "train_auc": train_auc.item(),
            },
        }
    ]

    results = {
        "attn_score_per_layer": per_layer_results,
        "attn_score_all_layers": all_layers_results,
    }

    torch.save(results, results_dir.attn_score_file)


def load_and_prepare_data(
    ds_dir: DatasetDir,
) -> list[torch.Tensor]:
    attn_diags = []

    if ds_dir.attn_diags_file.exists():
        logger.info("Loading cached diagonals...")
        attn_diags = torch.load(ds_dir.attn_diags_file, weights_only=True)
    else:
        logger.info("Computing diagonals...")
        for attn_shard in yield_stacked_attentions(ds_dir, remove_padding=True):
            for attn_example in tqdm(attn_shard, desc="attn shards", leave=False):
                attn_diags.append(
                    torch.stack(
                        [torch.diagonal(layer_attn, dim1=1, dim2=2) for layer_attn in attn_example]
                    )
                )
        torch.save(attn_diags, ds_dir.attn_diags_file)

    return attn_diags


if __name__ == "__main__":
    typer.run(main)
