import re
from pathlib import Path

import pandas as pd
import torch
import typer
from datasets import load_dataset
from loguru import logger

from hallucinations.dirs import DatasetDir

# Credit: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml
FINAL_ANSWER_REGEX = "The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"
IGNORE_ANSWER_PATTERNS = [
    r",",
    r"\$",
    r"(?s).*#### ",
    r"\.$",
]


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
) -> None:
    ds_dir = DatasetDir(dataset_dir)

    df = pd.read_json(ds_dir.answers_file)
    df["final_answer"] = df["prediction"].apply(extract_final_answer_from_prediction)

    logger.info(f"Number of missing final answers: {df['final_answer'].isnull().sum()}")

    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.map(extract_exact_answer_from_gold, batched=False)
    dataset = dataset.to_pandas()

    res = df.join(dataset[["answer", "exact_answer"]], how="right")

    assert (res["answer"] == res["gold"]).all(), "Gold answers and LLM answers have different order"
    res = res.drop(columns=["answer"])

    labels, valid_labels_mask = compute_gsm8k_labels(res)

    logger.info(f"Number of hallucinated answers: {labels[valid_labels_mask].sum()}")
    logger.info(f"Number of non-hallucinated answers: {(labels[valid_labels_mask] == 0).sum()}")
    logger.info(f"LLM accuracy: {(labels[valid_labels_mask] == 0).float().mean():0.3f}")

    torch.save(
        {"labels": labels, "valid_labels_mask": valid_labels_mask},
        ds_dir.labels_file,
    )


def extract_exact_answer_from_gold(item: dict[str, str]) -> dict[str, str]:
    return {"exact_answer": item["answer"].partition("####")[2].strip()}


def extract_final_answer_from_prediction(prediction: str) -> str | None:
    match = re.search(FINAL_ANSWER_REGEX, prediction)
    if match:
        final_answer = match.group(1)
        for ignore_pattern in IGNORE_ANSWER_PATTERNS:
            final_answer = re.sub(ignore_pattern, "", final_answer)
        return final_answer.strip()  # Add strip to remove whitespace
    else:
        return None


def compute_gsm8k_labels(res: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(res["final_answer"] != res["exact_answer"], dtype=torch.long)
    labels[res["final_answer"].isnull()] = -1
    valid_labels_mask = torch.tensor(res["final_answer"].notnull(), dtype=torch.bool)
    return labels, valid_labels_mask


if __name__ == "__main__":
    typer.run(main)
