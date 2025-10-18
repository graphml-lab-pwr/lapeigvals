import itertools
from pathlib import Path
from pprint import pprint
from statistics import mean, median

import typer
from transformers import AutoTokenizer

from hallucinations.config import QaDatasetConfig
from hallucinations.data.factory import get_dataset
from hallucinations.utils.misc import load_and_resolve_config


def main(
    dataset_config: Path = typer.Option(...),
    tokenizer_name: str | None = typer.Option(...),
    use_tiktoken: bool = typer.Option(False),
) -> None:
    config = QaDatasetConfig(**load_and_resolve_config(dataset_config))
    dataset = get_dataset(config, split="train")

    if use_tiktoken:
        raise NotImplementedError("TikToken is not implemented yet.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = tokenizer(
        list(itertools.chain.from_iterable(dataset["answer"])),
        padding=False,
        truncation=False,
        return_length=True,
    )["length"]

    print("Number of tokens".center(80, "="))
    pprint(
        {
            "max": max(lengths),
            "min": min(lengths),
            "mean": mean(lengths),
            "median": median(lengths),
        }
    )


if __name__ == "__main__":
    typer.run(main)
