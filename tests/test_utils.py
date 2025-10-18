from datasets import Dataset

from hallucinations.utils import sort_dataset_by_input_length


def test_dataset_sort() -> None:
    ds = Dataset.from_dict({"text": ["a", "bb", "ccc", "ddd", "eeee"], "label": [1, 2, 3, 4, 5]})

    ds, sort_idx = sort_dataset_by_input_length(ds, "text")
    assert ds["text"] == ["eeee", "ccc", "ddd", "bb", "a"]
    assert ds["label"] == [5, 3, 4, 2, 1]
    assert sort_idx == [4, 3, 1, 2, 0]


def test_dataset_inverse_sort() -> None:
    ds = Dataset.from_dict({"text": ["a", "bb", "ccc", "ddd", "eeee"], "label": [1, 2, 3, 4, 5]})

    ds, sort_idx = sort_dataset_by_input_length(ds, "text")
    ds = ds.select(sort_idx)
    assert ds["text"] == ["a", "bb", "ccc", "ddd", "eeee"]
    assert ds["label"] == [1, 2, 3, 4, 5]
