from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from hallucinations.config import (
    DatasetConfig,
    PromptConfig,
    QaDatasetConfig,
    QaPromptConfig,
)
from hallucinations.data.coqa import load_and_prepare_coqa
from hallucinations.data.formatter import (
    HaluEvalQAFormatter,
    QaFormatter,
)
from hallucinations.data.squad_v2 import load_squad_v2_devset
from hallucinations.data.trivia_qa import load_trivia_qa_nocontext_dataset
from hallucinations.dirs import DatasetDir, dataset_dir_to_obj


def prepare_dataset(
    dataset_config: DatasetConfig,
    split: str | None,
    prompt_config: PromptConfig,
    use_output: bool,
    seed: int,
    return_raw: bool = False,
) -> Dataset | tuple[Dataset, Dataset]:
    dataset = get_dataset(config=dataset_config, split=split)

    if dataset_config.name == "google-research-datasets/nq_open":
        assert isinstance(prompt_config, QaPromptConfig)
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "gsm8k":
        assert isinstance(prompt_config, QaPromptConfig)
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "halueval_qa":
        assert isinstance(prompt_config, QaPromptConfig)
        formatted_ds = dataset.map(
            function=HaluEvalQAFormatter(prompt=prompt_config, use_context=False),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "trivia_qa":
        assert isinstance(prompt_config, QaPromptConfig)
        # trivia_qa has the same format as nq_open
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "coqa":
        assert isinstance(prompt_config, QaPromptConfig)
        # coqa has the same format as nq_open
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "squad_v2":
        assert isinstance(prompt_config, QaPromptConfig)
        # squad_v2 has the same format as nq_open
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "truthful_qa":
        assert isinstance(prompt_config, QaPromptConfig)
        # truthful_qa has the same format as nq_open
        formatted_ds = dataset.map(
            function=QaFormatter(prompt=prompt_config, use_output=use_output),
            batched=False,
            desc="Formatting dataset",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")

    if return_raw:
        return dataset, formatted_ds
    else:
        return formatted_ds


def get_dataset(config: DatasetConfig, split: str | None) -> Dataset | DatasetDict:
    if config.name == "google-research-datasets/nq_open":
        assert isinstance(config, QaDatasetConfig)
        return load_dataset(config.name, split=split)
    elif config.name == "gsm8k":
        assert isinstance(config, QaDatasetConfig)
        return load_dataset(config.name, "main", split=split)
    elif config.name == "halueval_qa":
        assert isinstance(config, QaDatasetConfig)
        return (
            Dataset.from_json(str(config.path))
            .select_columns(["question", "right_answer"])
            .rename_columns({"right_answer": "answer"})
        )
    elif config.name == "trivia_qa":
        assert isinstance(config, QaDatasetConfig)
        assert config.subset == "rc.nocontext"
        return load_trivia_qa_nocontext_dataset(split=split)
    elif config.name == "coqa":
        assert isinstance(config, QaDatasetConfig)
        assert split is None
        assert config.path is not None
        return load_and_prepare_coqa(config.path)
    elif config.name == "squad_v2":
        assert isinstance(config, QaDatasetConfig)
        assert split is None
        assert config.path is not None
        return load_squad_v2_devset(config.path)
    elif config.name == "truthful_qa":
        assert isinstance(config, QaDatasetConfig)
        assert config.subset == "generation"
        assert split == "validation"
        ds = load_dataset(config.name, config.subset, split=split)
        ds = ds.rename_column("correct_answers", "answer")
        return ds
    else:
        raise ValueError(f"Unknown dataset: {config.name}")


def get_dataset_from_dir(dataset_dir: str | Path | DatasetDir, split: str | None) -> Dataset:
    ds_dir = dataset_dir_to_obj(dataset_dir)
    config = ds_dir.load_dataset_config()

    if split is None:
        split = config.test_split_name

    return get_dataset(config=config, split=split)
