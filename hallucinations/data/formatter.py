from typing import Any

from hallucinations.config import PromptConfig, QaPromptConfig


class DatasetFormatter:
    def __init__(self, prompt: PromptConfig):
        self.prompt = prompt

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class QaFormatter(DatasetFormatter):
    def __init__(self, prompt: QaPromptConfig, use_output: bool):
        self.prompt: QaPromptConfig = prompt
        assert self.prompt.context_key is None
        self.use_output = use_output

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        content = self.prompt.content.format(
            **{self.prompt.question_key: item[self.prompt.question_key]}
        )
        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }
        if self.use_output:
            raise NotImplementedError("Need to determine which answer to use")

        return messages


class HaluEvalQAFormatter(DatasetFormatter):
    def __init__(self, prompt: QaPromptConfig, use_context: bool):
        self.prompt: QaPromptConfig = prompt
        self.use_context = use_context

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.use_context:
            raise NotImplementedError("Using context is not supported for HaluEvalQA")

        content = self.prompt.content.format(
            **{self.prompt.question_key: item[self.prompt.question_key]}
        )
        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }

        return messages
