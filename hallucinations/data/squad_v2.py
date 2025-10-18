import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_squad_v2_devset(devset_path: Path) -> Dataset:
    """Loads and filters SQuAD v2 dev set.
    Credit: https://github.com/alibaba/eigenscore/blob/main/dataeval/SQuAD.py
    Further taken from: https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py"""
    with devset_path.open() as file:
        data = json.load(file)["data"]

    dataset: dict[str, Any] = defaultdict(list)

    for item in data:
        paragraphs = item["paragraphs"]
        for sample_idx, sample in enumerate(paragraphs):
            context = sample["context"]
            questions = sample["qas"]
            for question_index, question in enumerate(questions):
                if question["is_impossible"]:
                    continue
                dataset["sample_idx"].append(sample_idx)
                dataset["question_idx"].append(question_index)
                dataset["id"].append(question["id"])
                dataset["context"].append(context)
                dataset["question"].append(question["question"])
                dataset["answer_info"].append(
                    {
                        "text": question["answers"][0]["text"],
                        "answer_start": question["answers"][0]["answer_start"],
                    }
                )
                additional_answers_list = list(set(ans["text"] for ans in question["answers"]))
                dataset["answer"].append(additional_answers_list)

    dataset = dict(dataset)
    return Dataset.from_dict(dataset)
