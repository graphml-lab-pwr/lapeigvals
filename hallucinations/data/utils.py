from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets

from hallucinations.config import GenerateActivationsConfig
from hallucinations.data.factory import get_dataset
from hallucinations.dirs import DatasetDir, dataset_dir_to_obj
from hallucinations.utils.misc import load_json


def load_qa_dataset_with_metrics(
    dataset_dir: Path | DatasetDir,
    llm_judge_prompt: str,
    llm_judge_llm: str,
) -> Dataset:
    ds_dir = dataset_dir_to_obj(dataset_dir)
    config = GenerateActivationsConfig(**ds_dir.load_raw_config())
    dataset = get_dataset(config=config.dataset, split=config.split)
    metrics = Dataset.from_pandas(load_qa_metrics(ds_dir, llm_judge_prompt, llm_judge_llm))

    metrics = metrics.remove_columns("answers")

    return concatenate_datasets([dataset, metrics])


def load_qa_metrics(
    dataset_dir: Path | DatasetDir,
    llm_judge_prompt: str | None = None,
    llm_judge_llm: str | None = None,
) -> pd.DataFrame:
    ds_dir = dataset_dir_to_obj(dataset_dir)
    answers = pd.read_json(ds_dir.answers_file)
    metrics = pd.DataFrame(load_json(ds_dir.metrics_file)["all"])
    metrics = pd.concat([answers, metrics], axis=1)

    if llm_judge_prompt is None and llm_judge_llm is None:
        judge_configs = ds_dir.list_llm_judge_configs()
    else:
        assert llm_judge_prompt is not None and llm_judge_llm is not None
        judge_configs = [
            {
                "llm_judge_prompt": llm_judge_prompt,
                "llm_judge_llm": llm_judge_llm,
            }
        ]

    judge_res = []
    for config in judge_configs:
        llm_judge_file = ds_dir.llm_judge_file(config["llm_judge_prompt"], config["llm_judge_llm"])
        llm_judge_results = pd.DataFrame(
            {
                f"llm_judge_{config['llm_judge_llm']}_{config['llm_judge_prompt']}": load_json(
                    llm_judge_file
                )
            }
        )
        judge_res.append(llm_judge_results)
    metrics = pd.concat([metrics, *judge_res], axis=1)
    return metrics
