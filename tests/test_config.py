from typing import Any

import pytest
from pydantic import ValidationError

from hallucinations.config import (
    GenerateActivationsConfig,
    QaDatasetConfig,
)


@pytest.fixture
def raw_config() -> dict[str, Any]:
    return {
        "llm": {
            "name": "test-llm",
            "tokenizer_name": "test-tokenizer",
            "tokenizer_padding_side": "right",
            "context_size": 2048,
            "compile": False,
            "torch_dtype": "float16",
            "attn_implementation": "eager",
        },
        "dataset": {
            "cls_path": "hallucinations.config.QaDatasetConfig",
            "name": "test-dataset",
            "test_split_name": "test",
            "max_answer_tokens": 128,
            "target_column_name": "answer",
            "path": "/tmp/test",
        },
        "prompt": {
            "content": "test prompt",
            "question_key": "question",
        },
        "generation_config": {},
        "stored_features": "hidden_states",
        "split": "test",
        "batch_size": 1,
        "results_dir": "/tmp/results",
        "random_seed": 42,
    }


def test_generate_activations_config_create_dataset_class_validator_success(
    raw_config: dict[str, Any],
) -> None:
    config = GenerateActivationsConfig(**raw_config)
    assert isinstance(config.dataset, QaDatasetConfig)


def test_generate_activations_config_create_dataset_class_validator_missing_cls_path(
    raw_config: dict[str, Any],
) -> None:
    raw_config["dataset"]["cls_path"] = None
    with pytest.raises(ValueError, match="'cls_path' must be provided"):
        GenerateActivationsConfig(**raw_config)


def test_generate_activations_config_update_generation_config_hidden_states(
    raw_config: dict[str, Any],
) -> None:
    raw_config["stored_features"] = "hidden_states"
    config = GenerateActivationsConfig(**raw_config)

    assert config.generation_config["return_dict_in_generate"] is True
    assert config.generation_config["output_hidden_states"] is True
    assert config.generation_config["output_attentions"] is False


def test_generate_activations_config_update_generation_config_attentions(
    raw_config: dict[str, Any],
) -> None:
    raw_config["stored_features"] = "attentions"
    config = GenerateActivationsConfig(**raw_config)

    assert config.generation_config["return_dict_in_generate"] is True
    assert config.generation_config["output_hidden_states"] is False
    assert config.generation_config["output_attentions"] is True


def test_generate_activations_config_update_generation_config_attention_diags(
    raw_config: dict[str, Any],
) -> None:
    raw_config["stored_features"] = "attention_diags"
    config = GenerateActivationsConfig(**raw_config)

    assert config.generation_config["return_dict_in_generate"] is True
    assert config.generation_config["output_hidden_states"] is False
    assert config.generation_config["output_attentions"] is True


def test_generate_activations_config_update_generation_config_attention_with_hidden_states(
    raw_config: dict[str, Any],
) -> None:
    raw_config["stored_features"] = "attention_with_hidden_states"
    config = GenerateActivationsConfig(**raw_config)

    assert config.generation_config["return_dict_in_generate"] is True
    assert config.generation_config["output_hidden_states"] is True
    assert config.generation_config["output_attentions"] is True


def test_generate_activations_config_update_generation_config_hidden_states_for_last_input_last_gen_tokens(
    raw_config: dict[str, Any],
) -> None:
    raw_config["stored_features"] = "hidden_states_for_last_input_last_gen_tokens"
    config = GenerateActivationsConfig(**raw_config)

    assert config.generation_config["return_dict_in_generate"] is True
    assert config.generation_config["output_hidden_states"] is True
    assert config.generation_config["output_attentions"] is False


def test_generate_activations_config_update_generation_config_overrides_existing_values(
    raw_config: dict[str, Any],
) -> None:
    # Set pre-existing values that should be overridden
    raw_config["generation_config"] = {
        "return_dict_in_generate": False,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    raw_config["stored_features"] = "attention_with_hidden_states"

    # This should raise an assertion error because these keys shouldn't be in generation_config
    with pytest.raises(ValidationError):
        GenerateActivationsConfig(**raw_config)
