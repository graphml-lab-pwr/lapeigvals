from pathlib import Path

import torch
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch import Tensor

from hallucinations.dirs import DatasetDir


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    test_size: float = typer.Option(..., help="Size of the test set"),
    seed: int = typer.Option(..., help="Random seed"),
    overwrite: bool = typer.Option(False, help="Overwrite existing split file"),
) -> None:
    ds_dir = DatasetDir(dataset_dir)
    if not overwrite and (ds_dir.split_file).exists():
        raise ValueError(f"Split file already exists at {ds_dir.split_file}")

    assert ds_dir.labels_file.exists(), "Labels file not found"

    label_data = torch.load(ds_dir.labels_file, weights_only=True)
    valid_labeled_data_idx = torch.arange(label_data["labels"].numel())[
        label_data["valid_labels_mask"]
    ]
    valid_labels = label_data["labels"][valid_labeled_data_idx]

    # In split, only valid labeled data is used
    train_idx, test_idx = train_test_split(
        valid_labeled_data_idx,
        test_size=test_size,
        random_state=seed,
        stratify=valid_labels,
    )

    print_split_statistics(label_data["labels"][train_idx], label_data["labels"][test_idx])

    logger.info(f"Saving split to {dataset_dir / 'split.pt'}")
    torch.save(
        {
            "train_idx": train_idx,
            "test_idx": test_idx,
        },
        dataset_dir / "split.pt",
    )


def print_split_statistics(train_labels: Tensor, test_labels: Tensor) -> None:
    """Print statistics about the dataset split."""
    train_counts = torch.bincount(train_labels)
    test_counts = torch.bincount(test_labels)

    table = [
        ["Split", "No-Hallucination", "Hallucination", "Total", "Ratio (H/NH)"],
        [
            "Train",
            train_counts[0].item(),
            train_counts[1].item(),
            len(train_labels),
            f"{train_counts[1].item() / train_counts[0].item():.3f}",
        ],
        [
            "Test",
            test_counts[0].item(),
            test_counts[1].item(),
            len(test_labels),
            f"{test_counts[1].item() / test_counts[0].item():.3f}",
        ],
        [
            "Total",
            (train_counts[0] + test_counts[0]).item(),
            (train_counts[1] + test_counts[1]).item(),
            len(train_labels) + len(test_labels),
            f"{(train_counts[1] + test_counts[1]).item() / (train_counts[0] + test_counts[0]).item():.3f}",
        ],
    ]

    logger.info(f"\n{tabulate(table, headers='firstrow', tablefmt='grid')}")  # type: ignore


if __name__ == "__main__":
    typer.run(main)
