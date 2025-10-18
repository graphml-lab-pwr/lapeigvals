import random
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import typer
from tqdm.auto import tqdm, trange

from hallucinations.dirs import DatasetDir
from hallucinations.features.attn_feats import get_laplacian_eigvals_per_head_topk
from hallucinations.probe_models.lr import train_logistic_regression

PERTURBATION_STD = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, None]
K_VALUES = [5, 10, 20, 50, 100]
NUM_REPETITIONS = 5

FILE_NAME = "perturbation_analysis_with_fraction_of_features.pt"


def main(
    root_dir: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(...),
) -> None:
    """Run the first perturbation study."""
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ds_dir = DatasetDir(root_dir)

    split = ds_dir.load_split()
    labels = ds_dir.load_labels()["labels"]

    data = torch.load(ds_dir.laplacian_diags_file)
    print(f"Data loaded. First tensor size: {data[0].size()}")

    torch.manual_seed(42)
    random.seed(42)

    print("Running first perturbation study...")
    res = []

    for k in tqdm(K_VALUES, desc="k values"):
        for std in tqdm(PERTURBATION_STD, leave=False, desc="std values"):
            for _ in trange(NUM_REPETITIONS, leave=False, desc="iterations"):
                if std is not None:
                    data_perturbed = []
                    perturbation_fractions = []

                    for item in data:
                        perturbation_fraction = random.uniform(0.5, 1.0)
                        perturbation_fractions.append(perturbation_fraction)

                        total_features = item.numel()
                        num_features_to_perturb = int(total_features * perturbation_fraction)

                        indices_to_perturb = torch.randperm(total_features)[
                            :num_features_to_perturb
                        ]

                        perturbed_flat = item.clone().flatten()
                        noise = torch.randn(num_features_to_perturb) * std
                        perturbed_flat[indices_to_perturb] += noise

                        perturbed_item = perturbed_flat.reshape(item.shape)
                        data_perturbed.append(perturbed_item)

                    avg_perturbation_fraction = (
                        sum(perturbation_fractions) / len(perturbation_fractions)
                        if perturbation_fractions
                        else 0.0
                    )
                else:
                    data_perturbed = [item.clone() for item in data]
                    avg_perturbation_fraction = 0.0

                avg_diff_l2_norm = torch.mean(
                    torch.tensor(
                        [
                            torch.linalg.vector_norm(item_perturbed - item)
                            for item_perturbed, item in zip(data_perturbed, data)
                        ]
                    )
                ).item()

                feats = get_laplacian_eigvals_per_head_topk(
                    data_perturbed,
                    layer_idx=None,
                    top_k=k,
                )

                results = train_logistic_regression(
                    features=feats,
                    labels=labels,
                    split=split,
                    pca_dim=512,
                )

                res.append(
                    {
                        "k": k,
                        "std": std,
                        "avg_diff_l2_norm": avg_diff_l2_norm,
                        "avg_perturbation_fraction": avg_perturbation_fraction,
                        **results["metrics"],
                    }
                )

    output_file = output_dir / FILE_NAME
    torch.save(res, output_file)
    print(f"Results saved to {output_file}")
    print(f"Total number of results: {len(res)}")

    if res:
        print("\nSample results:")
        df = pd.DataFrame(res)
        print(df.head())

        print("\nSummary:")
        print(f"k values tested: {sorted(df['k'].unique())}")
        print(
            f"std values tested: {sorted([x for x in df['std'].unique() if x is not None]) + [None]}"
        )
        print(f"Iterations per combination: {len(df) // (len(K_VALUES) * len(PERTURBATION_STD))}")

        perturbed_df = df[df["std"].notna()]
        if not perturbed_df.empty:
            print("\nPerturbation fraction statistics:")
            print(f"  Mean: {perturbed_df['avg_perturbation_fraction'].mean():.3f}")
            print(f"  Min:  {perturbed_df['avg_perturbation_fraction'].min():.3f}")
            print(f"  Max:  {perturbed_df['avg_perturbation_fraction'].max():.3f}")
            print(f"  Std:  {perturbed_df['avg_perturbation_fraction'].std():.3f}")


if __name__ == "__main__":
    typer.run(main)
