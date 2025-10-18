from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from hallucinations.config import LllmJudgeConfig, LlmJudgePromptConfig
from hallucinations.utils.misc import list_or_single_to_list

MAX_API_CALL_RETRIES = 3
TEMPERATURE = 0.0
DEFAULT_CACHE_PATH = ".langchain_cache.db"


class LlmJudgeEvaluator:
    def __init__(
        self,
        config: LllmJudgeConfig,
        openai_api_key: str,
        batch_size: int,
        rate_limit: int | None = None,
        enable_cache: bool = True,
    ):
        self.config = config

        if rate_limit is not None:
            self.rate_limiter: InMemoryRateLimiter | None = InMemoryRateLimiter(
                requests_per_second=rate_limit,
                check_every_n_seconds=0.05,
                max_bucket_size=rate_limit,
            )
        else:
            logger.warning("No rate limit provided")
            self.rate_limiter = None

        self.client = ChatOpenAI(
            model=config.llm_api.version,
            base_url=config.llm_api.base_url,
            api_key=openai_api_key,  # type: ignore
            temperature=TEMPERATURE,
            max_retries=MAX_API_CALL_RETRIES,
            cache=SQLiteCache(database_path=DEFAULT_CACHE_PATH) if enable_cache else None,
            rate_limiter=self.rate_limiter,
        )
        self.batch_size = batch_size

    async def __call__(
        self,
        predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.config.prompt.separate_multi_answers:
            raise NotImplementedError("Separating multiple answers is not implemented")
            logger.info("LLM Judge: Separating multiple answers")
            eval_func = self.eval_pred_answer_separately
        else:
            logger.info("LLM Judge: Evaluating answer once")
            eval_func = self.eval_pred_answer_once

        eval_tasks = [eval_func(ans) for ans in predictions]
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating predictions")

        return results

    async def eval_pred_answer_separately(
        self,
        pred: dict[str, Any],
    ) -> str | None:
        assesments: list[str | None] = []
        pred_gold = list_or_single_to_list(pred["gold"])
        for gold_answer in pred_gold:
            res = await self.do_eval_answer(
                question=pred["question"],
                pred_answer=pred["prediction"],
                gold_answers=gold_answer,
            )
            assesments.append(res["content"])

        return self.combine_assesments(assesments)

    async def eval_pred_answer_once(
        self,
        pred: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.do_eval_answer(
            question=pred["question"],
            pred_answer=pred["prediction"],
            gold_answers=pred["gold"],
        )

    async def do_eval_answer(
        self,
        question: str,
        pred_answer: str,
        gold_answers: str,
    ) -> dict[str, Any]:
        messages = prepare_messages(
            prompt_cfg=self.config.prompt,
            question=question,
            pred_answer=pred_answer,
            gold_answer=gold_answers,
        )
        completion = await self.client.ainvoke(
            input=messages,
        )
        res = completion.content
        assert isinstance(res, str)

        return {
            "content": res,
            "response_metadata": completion.response_metadata,
            "usage_metadata": completion.usage_metadata,  # type: ignore
        }

    def combine_assesments(self, assesments: list[str | None]) -> str | None:
        """Combines multiple assesment results into a single result.
        Note that possible answers declared in prompt are in descending priority.

        For example:
        - if the possible answers are ["correct", "incorrect", "refuse"],
            and the assesment results are ["correct", "refuse", "incorrect"],
            the combined result would be "correct".
        - if the possible answers are ["refuse", "correct", "incorrect"],
            and the assesment results are ["correct", "incorrect", "refuse"],
            the combined result would be "refuse".
        """
        for answer in self.config.prompt.possible_answers:
            if answer in assesments:
                return answer
        return None


def prepare_messages(
    prompt_cfg: LlmJudgePromptConfig,
    question: str,
    pred_answer: str,
    gold_answer: str,
) -> list[dict[str, Any]]:
    prompt_str = prompt_cfg.format(
        question=question,
        pred_answer=pred_answer,
        gold_answer=gold_answer,
    )
    if prompt_cfg.system_prompt is not None:
        messages = [
            {"role": "system", "content": prompt_cfg.system_prompt},
            {"role": "user", "content": prompt_str},
        ]
    else:
        messages = [{"role": "user", "content": prompt_str}]
    return messages
