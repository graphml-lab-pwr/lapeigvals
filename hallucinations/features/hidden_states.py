from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Literal

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from hallucinations.features.processing import remove_padding_from_intermediate_states
from hallucinations.utils.misc import load_and_resolve_config


@dataclass
class HiddenStatesSelection:
    layer: Literal["all", "last"] | int
    hs_last_input_token: bool
    hs_last_generated_token: bool


def load_hidden_states(
    hidden_states_dir: Path,
    hs_selection: HiddenStatesSelection,
) -> dict[str, list[Tensor]]:
    """Loads hidden states from a given directory.
    Dimension of the output tensor is: [#layers, [#examples, hidden_size]]
    """
    shard_acts = []
    shard_tokens = []
    shard_lengths = []

    for shard_features in yield_hidden_states(hidden_states_dir, hs_selection):
        shard_acts.append(
            {
                k: v
                for k, v in shard_features.items()
                if k not in ["generated_tokens", "input_length"]
            }
        )
        shard_tokens.extend(shard_features["generated_tokens"])
        shard_lengths.extend(shard_features["input_length"])

    dataset_features = concat_shard_features(shard_acts)
    dataset_features["generated_tokens"] = shard_tokens
    dataset_features["input_length"] = shard_lengths

    return dataset_features


def yield_hidden_states(
    hidden_states_dir: Path,
    hs_selection: HiddenStatesSelection | None = None,
    remove_padding: bool = False,
    device: str = "cpu",
) -> Generator[dict[str, Any], None, None]:
    """Yields features from hidden states shards."""
    if remove_padding and hs_selection is not None:
        raise NotImplementedError(
            "Removing padding is not yet `supported for hidden states selection."
        )
    shard_paths = list(hidden_states_dir.glob("*.pt"))
    if not shard_paths:
        raise ValueError(f"No hidden states found in {hidden_states_dir}")

    if remove_padding:
        config = load_and_resolve_config(hidden_states_dir.parent.parent / "config.yaml")
        tokenizer = AutoTokenizer.from_pretrained(config["llm"]["name"])

    for s_path in tqdm(shard_paths, desc="Loading hidden states"):
        shard: dict[str, Any] = torch.load(
            s_path,
            weights_only=True,
            mmap=True,
            map_location=device,
        )
        if remove_padding and hs_selection is None:
            shard["hidden_states"] = remove_padding_from_intermediate_states(
                per_layer_batched_data=shard["hidden_states"],
                data_type="hs",
                generated_tokens=shard["generated_tokens"],
                tokenizer=tokenizer,
            )
        if hs_selection is None:
            yield shard
        else:
            shard_features = select_hidden_states_from_layers(
                shard=shard,
                hs_selection=hs_selection,
            )
            shard_features["generated_tokens"] = shard["generated_tokens"].tolist()
            shard_features["input_length"] = [shard["input_length"]] * shard[
                "generated_tokens"
            ].size(0)
            yield shard_features


def select_hidden_states_from_layers(
    shard: dict[str, Any],
    hs_selection: HiddenStatesSelection,
) -> dict[str, list[torch.Tensor]]:
    """Extracts features from a given layer(s) of a single shard (creates a copy of the data)."""
    if hs_selection.layer == "all":
        layer_idx = list(range(len(shard["hidden_states"])))
    elif hs_selection.layer == "last":
        layer_idx = [-1]
    else:
        layer_idx = [hs_selection.layer]

    shard_layerwise_acts = defaultdict(list)
    for l_idx in layer_idx:
        layer_feats = select_hidden_states_from_single_layer(
            shard=shard,
            layer_idx=l_idx,
            hs_selection=hs_selection,
        )
        for feat_name, layerwise_data in layer_feats.items():
            shard_layerwise_acts[feat_name].append(layerwise_data.clone())

    return dict(shard_layerwise_acts)


def select_hidden_states_from_single_layer(
    shard: dict[str, Any],
    layer_idx: int,
    hs_selection: HiddenStatesSelection,
) -> dict[str, torch.Tensor]:
    """Extracts requested features from a given layer."""
    token_mask = torch.bitwise_not(
        torch.bitwise_or(shard["special_token_mask"], shard["decoder_token_mask"])
    )
    token_mask = token_mask[:, 1:]
    input_length = shard["input_length"] - 1
    layer_hidden_states = shard["hidden_states"][layer_idx]

    feats: dict[str, torch.Tensor] = {}
    if hs_selection.hs_last_input_token:
        input_token_mask = token_mask[:, :input_length]
        feats["hs_last_input_token"], feats["hs_last_input_token_idx"] = _get_last_masked_token(
            layer_hidden_states, input_token_mask
        )
    if hs_selection.hs_last_generated_token:
        gen_hs_layer = layer_hidden_states[:, input_length:]
        gen_token_mask = token_mask[:, input_length:]
        feats["hs_last_generated_token"], feats["hs_last_generated_token_idx"] = (
            _get_last_masked_token(gen_hs_layer, gen_token_mask)
        )
        feats["hs_last_generated_token_idx"] += input_length
    return feats


def _get_last_masked_token(data: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    last_token_idx = (mask.shape[1] - 1) - torch.argmax(
        mask.flip(dims=[1]),
        dim=1,
    )
    # by using gather we copy data thus memory is freed from the large original tensor
    last_token_data = torch.gather(
        input=data,
        dim=1,
        index=last_token_idx.view(-1, 1, 1).expand(-1, 1, data.size(2)),
    ).squeeze(1)
    return last_token_data, last_token_idx


def concat_shard_features(
    shard_acts: list[dict[str, list[torch.Tensor]]],
) -> dict[str, list[torch.Tensor]]:
    results = defaultdict(list)
    for feat_name in shard_acts[0].keys():
        num_layers = len(shard_acts[0][feat_name])
        for l_idx in range(num_layers):
            results[feat_name].append(
                torch.cat(
                    [current_shard_acts[feat_name][l_idx] for current_shard_acts in shard_acts],
                    dim=0,
                )
            )

    return dict(results)
