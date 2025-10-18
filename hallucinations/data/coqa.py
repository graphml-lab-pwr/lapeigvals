import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_and_prepare_coqa(path: Path) -> Dataset:
    """Loads and flattens CoQa dev set.

    Credit: https://github.com/alibaba/eigenscore/blob/main/dataeval/coqa.py
    Further taken from: https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
    """

    with path.open() as infile:
        data = json.load(infile)["data"]

    return flatten_coqa_conversations(data)


def flatten_coqa_conversations(raw_coqa_dataset: list[dict[str, Any]]) -> Dataset:
    dataset: dict[str, list[Any]] = defaultdict(list)
    for sample_idx, sample in enumerate(raw_coqa_dataset):
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        additional_answers = sample["additional_answers"]
        for question_index, question in enumerate(questions):
            dataset["sample_idx"].append(sample_idx)
            dataset["id"].append(sample["id"] + "_" + str(question_index))
            dataset["story"].append(story)
            dataset["question"].append(question["input_text"])
            dataset["answer"].append(
                {
                    "text": answers[question_index]["input_text"],
                    "answer_start": answers[question_index]["span_start"],
                }
            )
            additional_answers_list = []

            for i in range(3):
                additional_answers_list.append(
                    additional_answers[str(i)][question_index]["input_text"]
                )

            dataset["additional_answers"].append(additional_answers_list)

    ds = Dataset.from_dict(dataset)

    ds = ds.map(lambda x: {"answer": x["answer"]["text"]}, desc="Extracting answer text")
    ds = ds.map(
        lambda x: {"question": x["story"] + " Q: " + x["question"] + " A:"},
        desc="Extracting prompt",
    )

    return ds
