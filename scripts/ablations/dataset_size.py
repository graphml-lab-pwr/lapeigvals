from pathlib import Path

import pandas as pd
import torch
from lightning_fabric import seed_everything
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm.auto import tqdm

from hallucinations.dirs import DatasetDir
from hallucinations.features.attn_feats import (
    get_attn_eigvals_per_head_topk,
    get_laplacian_eigvals_per_head_topk,
)
from hallucinations.probe_models.lr import train_logistic_regression

DATASETS = [
    # "data/activations/trivia_qa/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    # "data/activations/squad_v2/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    # "data/activations/halueval_qa/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    # "data/activations/coqa/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    # "data/activations/nq_open/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    # "data/activations/truthful_qa/llama_3.1_8b_instruct/temp_1.0__prompt_qa_short_few_shot_sep__seed_42",
    "data/activations/gsm8k/llama_3.1_8b_instruct/temp_1.0__prompt_qa_gsm8k__seed_42",
]
FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 420]
TOP_K = 100
OUTPUT_DIR = "data/studies/dataset_size"


def main() -> None:
    ds_dirs = [DatasetDir(path) for path in DATASETS]

    all_results = []
    for ds_dir in tqdm(ds_dirs):
        try:
            results = run_ablation_for_dataset(ds_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing dataset {ds_dir.activations_dir}: {e}")
            continue

    if not all_results:
        print("No results obtained. Exiting.")
        return

    df = pd.DataFrame(all_results)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path / "dataset_size_ablation_results_gsm8k.json", orient="records", indent=4)

    print("\nSummary Statistics:")
    print("=" * 50)
    summary = (
        df.groupby(["dataset", "feature_type", "frac"])["test_auc"].agg(["mean", "std"]).round(4)
    )
    print(summary)

    print(f"\nResults saved to: {output_path}")


def run_ablation_for_dataset(
    ds_dir: DatasetDir,
) -> list[dict]:
    print(f"Processing dataset: {ds_dir.activations_dir}")

    labels = ds_dir.load_labels()["labels"]
    split = ds_dir.load_split()

    # Load both feature types
    laplacian_feats = torch.load(ds_dir.laplacian_diags_file)
    attn_feats = torch.load(ds_dir.attn_diags_file)

    # Extract features using both methods
    lap_eigvals = get_laplacian_eigvals_per_head_topk(laplacian_feats, layer_idx=None, top_k=TOP_K)
    attn_eigvals = get_attn_eigvals_per_head_topk(attn_feats, layer_idx=None, top_k=TOP_K)

    results = []

    # Run experiments for both feature types
    for feature_type, features in [("laplacian", lap_eigvals), ("attention", attn_eigvals)]:
        for frac in tqdm(FRACTIONS, desc=f"Fractions ({feature_type})", leave=False):
            for seed in tqdm(SEEDS, desc="Seeds", leave=False):
                if frac < 1.0:
                    seed_everything(seed, verbose=False)

                    sampled_train_idx = sample_fraction_of_data(split, labels, frac)
                    sampled_split = {
                        "train_idx": sampled_train_idx,
                        "test_idx": split["test_idx"],
                    }
                else:
                    sampled_split = split

                res = train_logistic_regression(
                    features=features, labels=labels, split=sampled_split, pca_dim=512
                )

                result = {
                    "dataset": str(ds_dir.root_dir),
                    "feature_type": feature_type,
                    "frac": frac,
                    "size": len(sampled_split["train_idx"]),
                    "seed": seed,
                    **res["metrics"],
                    **res["metadata"],
                }
                results.append(result)

    return results


def sample_fraction_of_data(split: dict[str, Tensor], labels: Tensor, frac: float) -> Tensor:
    train_labels = labels[split["train_idx"]]
    train_idx, _ = train_test_split(split["train_idx"], train_size=frac, stratify=train_labels)
    return train_idx.clone()


if __name__ == "__main__":
    main()
