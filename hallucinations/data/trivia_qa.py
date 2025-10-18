from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


def load_trivia_qa_nocontext_dataset(split: str | None) -> Dataset | DatasetDict:
    """Loads trivia_qa dataset as was used by INSIDE paper (https://arxiv.org/abs/2402.03744)
    Credit: https://github.com/alibaba/eigenscore/blob/main/dataeval/triviaqa.py
    """
    data = load_dataset("trivia_qa", "rc.nocontext", split=split)
    data = _deduplicate_trivia_qa_dataset(data)
    data = data.select_columns(["question_id", "question", "answer"])
    data = data.map(_get_answer, batched=False, desc="Extracting answers")
    return data


def _deduplicate_trivia_qa_dataset(dataset: Dataset) -> Dataset:
    id_mem = set()

    def remove_dups(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if batch["question_id"][0] in id_mem:
            return {_: [] for _ in batch.keys()}
        id_mem.add(batch["question_id"][0])
        return batch

    return dataset.map(
        remove_dups,
        batch_size=1,
        batched=True,
        load_from_cache_file=False,
        desc="Removing duplicates from trivia_qa",
    )


def _get_answer(item: dict[str, Any]) -> dict[str, Any]:
    return {"answer": item["answer"]["aliases"]}
