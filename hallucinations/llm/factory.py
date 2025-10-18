from dataclasses import dataclass
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from hallucinations.config import LlmConfig

LLAMA_3_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

PHI_35_MODELS = [
    "microsoft/Phi-3.5-mini-instruct",
]

MISTRAL_NEMO_MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]


@dataclass
class ModelForGeneration:
    llm: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    generate_kwargs: dict[str, Any]


def get_llm(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    if llm_config.name in LLAMA_3_MODELS:
        return get_llama_3(llm_config, **kwargs)
    elif llm_config.name in PHI_35_MODELS:
        return get_phi_35(llm_config, **kwargs)
    elif llm_config.name in MISTRAL_NEMO_MODELS:
        return get_mistral_nemo(llm_config, **kwargs)
    else:
        raise ValueError(f"Model {llm_config.name} not supported.")


def get_llama_3(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model = get_model(llm_config, **kwargs)
    tokenizer = get_tokenizer(llm_config)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )


def get_mistral_nemo(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model = get_model(llm_config, **kwargs)
    tokenizer = get_tokenizer(llm_config)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )


def get_phi_35(llm_config: LlmConfig, **kwargs: Any) -> ModelForGeneration:
    model = get_model(llm_config, **kwargs)
    tokenizer = get_tokenizer(llm_config)

    return ModelForGeneration(
        llm=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )


def get_model(
    llm_config: LlmConfig,
    **kwargs: Any,
) -> PreTrainedModel:
    if llm_config.quantization is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(**llm_config.quantization)

    model = AutoModelForCausalLM.from_pretrained(
        llm_config.name,
        torch_dtype=llm_config.torch_dtype,
        attn_implementation=llm_config.attn_implementation,
        **kwargs,
    )

    return model


def get_tokenizer(llm_config: LlmConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(llm_config.tokenizer_name)
    tokenizer.padding_side = llm_config.tokenizer_padding_side

    return tokenizer
