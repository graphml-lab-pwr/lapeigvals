import asyncio
import json
import logging
import os
from collections import Counter
from pprint import pformat
from typing import Any

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from hallucinations.config import LllmJudgeConfig
from hallucinations.data.factory import get_dataset
from hallucinations.evaluation.eval_with_api import LlmJudgeEvaluator
from hallucinations.utils import resolve_config, save_json, save_yaml

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "api_key_missing")
RATE_LIMIT = os.getenv("RATE_LIMIT", None)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../../config", config_name="llm_as_judge")
def main(cfg: DictConfig) -> None:
    config = LllmJudgeConfig(**resolve_config(cfg))
    logger.info(f"Config: {pformat(config.model_dump())}")

    config.evaluation_config_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(config.dataset, split=config.dataset.test_split_name)

    with config.answers_file.open("r") as file:
        answers = json.load(file)

    answers = [
        {
            "question": ds_item["question"],
            "prediction": ans_item["prediction"],
            "gold": ans_item["gold"],
        }
        for ds_item, ans_item in zip(dataset, answers, strict=True)
    ]

    eval_func = LlmJudgeEvaluator(
        config=config,
        openai_api_key=OPENAI_API_KEY,
        batch_size=config.llm_api.batch_size,
        rate_limit=int(RATE_LIMIT) if RATE_LIMIT is not None else None,
    )
    eval_results = asyncio.run(eval_func(answers))

    eval_assesments = [item["content"] for item in eval_results]
    eval_metadata = [
        {k: v for k, v in item.items() if k not in ["content"]} for item in eval_results
    ]

    eval_metadata_with_summary = {
        "summary": _compute_eval_metadata_summary(eval_metadata),
        "metadata": eval_metadata,
    }

    save_json(config.evaluation_file, eval_assesments)
    save_json(config.evaluation_metadata_file, eval_metadata_with_summary)
    save_yaml(config.evaluation_config_file, config.model_dump())


def _compute_eval_metadata_summary(eval_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    total_input_tokens = sum(item["usage_metadata"]["input_tokens"] for item in eval_metadata)
    total_output_tokens = sum(item["usage_metadata"]["output_tokens"] for item in eval_metadata)
    total_cached_input_tokens = sum(
        item["usage_metadata"]["input_token_details"]["cache_read"] for item in eval_metadata
    )
    finish_reason_counts = dict(
        Counter(item["response_metadata"]["finish_reason"] for item in eval_metadata)
    )

    return {
        "total_api_calls": len(eval_metadata),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_input_tokens": total_cached_input_tokens,
        "finish_reason_counts": finish_reason_counts,
    }


if __name__ == "__main__":
    main()
