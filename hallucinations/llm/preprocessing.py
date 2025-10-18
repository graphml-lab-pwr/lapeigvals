from typing import Any

import torch
from transformers import PreTrainedTokenizer


class SimpleEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, torch.Tensor]:
        try:
            final_input = self.tokenizer.apply_chat_template(
                batch["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
        except ValueError:
            assert all(len(item) == 1 for item in batch["messages"]), (
                f"Expected single message in batch, got {batch['messages']}"
            )
            final_input = [item[0]["content"].rstrip() for item in batch["messages"]]

        return self.tokenizer(final_input, return_tensors="pt", padding="longest", truncation=False)
