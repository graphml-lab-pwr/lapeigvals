from typing import Generator

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from hallucinations.dirs import DatasetDir
from hallucinations.features.processing import left_right_contiguous_padding_mask


def yield_generated_tokens(
    ds_dir: DatasetDir,
    remove_padding_tokens: bool = True,
    tokenizer: PreTrainedTokenizer | None = None,
    pad_token_id: int | None = None,
    verbose: bool = True,
) -> Generator[Tensor | list[Tensor], None, None]:
    attn_files = list(ds_dir.attentions_dir.glob("*.pt"))

    if len(attn_files) == 0:
        raise ValueError(f"No attention files found in {ds_dir.attentions_dir}")

    for attn_file in tqdm(
        attn_files,
        desc="Yielding generated tokens",
        disable=not verbose,
        leave=False,
    ):
        attn = torch.load(attn_file, weights_only=True)
        if remove_padding_tokens:
            yield remove_padding_tokens_from_generated_tokens(
                attn["generated_tokens"],
                _get_pad_token_id(tokenizer, pad_token_id),
            )
        else:
            yield attn["generated_tokens"]


def remove_padding_tokens_from_generated_tokens(
    generated_tokens: Tensor,
    pad_token_id: int,
) -> list[Tensor]:
    results: list[Tensor] = []
    for example_idx in range(generated_tokens.size(0)):
        example_pad_mask = left_right_contiguous_padding_mask(
            generated_tokens[example_idx],
            pad_token_id,
        )
        results.append(generated_tokens[example_idx][~example_pad_mask])
    return results


def _get_pad_token_id(tokenizer: PreTrainedTokenizer | None, pad_token_id: int | None) -> int:
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = tokenizer.pad_token_id
    assert pad_token_id is not None
    return pad_token_id
