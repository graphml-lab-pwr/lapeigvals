from pathlib import Path

import torch
import typer

from hallucinations.dirs import DatasetDir
from hallucinations.features.labels import compute_labels
from hallucinations.utils.misc import load_json


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    llm_judge_prompt: str = typer.Option(..., help="LLM judge prompt"),
    llm_judge_llm: str = typer.Option(..., help="LLM judge LLM"),
) -> None:
    ds_dir = DatasetDir(dataset_dir)

    assert ds_dir.metrics_file.exists(), "Metrics file not found"
    assert ds_dir.llm_judge_file(llm_judge_prompt, llm_judge_llm).exists(), (
        "LLM as judge file not found"
    )
    assert not ds_dir.labels_file.exists(), "Labels file already exists"

    llm_as_judge = load_json(ds_dir.llm_judge_file(llm_judge_prompt, llm_judge_llm))
    metrics = load_json(ds_dir.metrics_file)
    labels, valid_labels_mask = compute_labels(llm_as_judge, metrics["all"])  # type: ignore[arg-type]

    torch.save(
        {"labels": torch.tensor(labels), "valid_labels_mask": torch.tensor(valid_labels_mask)},
        ds_dir.labels_file,
    )


if __name__ == "__main__":
    typer.run(main)
