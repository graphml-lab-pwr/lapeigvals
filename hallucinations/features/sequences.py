from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizer

from hallucinations.config import GenerateActivationsConfig
from hallucinations.features.labels import CORRECT_ANSWER, INCORRECT_ANSWER
from hallucinations.llm.factory import get_tokenizer
from hallucinations.utils.misc import load_and_resolve_config, load_json

# todo: in future we may consider to merge this with hidden_states.py


@dataclass(kw_only=True, frozen=True)
class LayerData:
    layer_idx: int
    hidden_states: list[Tensor]
    gen_tokens: list[Tensor]
    last_input_token_idx: list[int]
    labels: list[int]
    metric_res: list[dict[str, float]]
    gen_answers: list[dict[str, Any]]
    config: dict[str, Any]
    llm_config: PretrainedConfig

    def __post_init__(self) -> None:
        assert (
            len(self.hidden_states)
            == len(self.labels)
            == len(self.metric_res)
            == len(self.gen_answers)
        )

    @property
    def num_samples(self) -> int:
        return len(self.hidden_states)

    def filter_data_with_invalid_labels(self) -> "LayerData":
        # TODO: change this to use valid_labels_mask saved with labels
        valid_indices = [
            i for i, label in enumerate(self.labels) if label in [CORRECT_ANSWER, INCORRECT_ANSWER]
        ]
        return LayerData(
            layer_idx=self.layer_idx,
            hidden_states=[self.hidden_states[i] for i in valid_indices],
            gen_tokens=[self.gen_tokens[i] for i in valid_indices],
            last_input_token_idx=[self.last_input_token_idx[i] for i in valid_indices],
            labels=[self.labels[i] for i in valid_indices],
            metric_res=[self.metric_res[i] for i in valid_indices],
            gen_answers=[self.gen_answers[i] for i in valid_indices],
            config=self.config,
            llm_config=self.llm_config,
        )

    def __repr__(self) -> str:
        return f"LayerData(layer_idx={self.layer_idx}, num_samples={self.num_samples}, llm_summary={get_llm_summary(self.config['results_dir'])})"


def load_hidden_states_dataset_for_single_layer(
    root_data_dir: Path,
    layer_idx: int,
    remove_padding: bool,
) -> LayerData:
    """Loads all activations datasets data found in the root_dir.
    Returns tuple of lists, each item in a list corresponds to a single dataset of activations.
    """
    answers_file = root_data_dir / "answers.json"
    llm_judge_file = root_data_dir / "llm_judge.json"
    metrics_file = root_data_dir / "metrics.json"
    config_file = root_data_dir / "config.yaml"
    labels_file = root_data_dir / "labels.pt"

    llm_judge_results: list[str] = load_json(llm_judge_file)  # type: ignore
    gen_answers: list[dict[str, str]] = load_json(answers_file)  # type: ignore
    config = load_and_resolve_config(config_file)

    metric_results: list[dict[str, Any]] = load_json(metrics_file)["all"]
    for i, m_res in enumerate(metric_results):
        m_res["llm_as_judge"] = llm_judge_results[i]

    hs_l, gen_tokens, last_input_token_idx = load_hidden_states_for_single_layer(
        root_data_dir,
        layer_idx,
        remove_padding,
    )
    labels = torch.load(labels_file)["labels"]

    return LayerData(
        layer_idx=layer_idx,
        hidden_states=hs_l,
        gen_tokens=gen_tokens,
        last_input_token_idx=last_input_token_idx,
        labels=labels,
        metric_res=metric_results,
        gen_answers=gen_answers,
        config=config,
        llm_config=get_llm_config(root_data_dir),
    )


def load_hidden_states_for_single_layer(
    root_dir: Path,
    layer_idx: int,
    remove_padding: bool,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """Returns hidden_states, generated_tokens and last_input_token_idx for a given layer.
    - The shape of hidden_states is (dataset_size, [sequence_length, hidden_size])
    - Due to padding to the longest example in batch during generation, sequences length may vary.
    """
    act_dir = root_dir / "activations"
    shard_files = list(act_dir.glob("batch_*.pt"))

    cfg = GenerateActivationsConfig(**load_and_resolve_config(root_dir / "config.yaml"))
    tokenizer = get_tokenizer(cfg.llm)

    hidden_states = []
    gen_tokens = []
    last_input_token_idx = []
    for data_shard in tqdm(shard_files, desc=f"Loading shards for layer {layer_idx}", leave=False):
        data = torch.load(
            data_shard,
            weights_only=False,
            mmap=True,
            map_location="cpu",
        )

        if remove_padding:
            hs_l, tokens, last_in_tok_idx = remove_padding_from_hidden_states(
                generated_tokens=data["generated_tokens"],
                hidden_states=data["hidden_states"][layer_idx],
                input_length=data["input_length"],
                tokenizer=tokenizer,
            )
        else:
            hs_l = data["hidden_states"][layer_idx]
            tokens = data["generated_tokens"]
            last_in_tok_idx = [data["input_length"] - 1] * hs_l.size(0)  # type: ignore

        hidden_states.extend(hs_l)
        gen_tokens.extend(tokens)
        last_input_token_idx.extend(last_in_tok_idx)

    return hidden_states, gen_tokens, last_input_token_idx


def remove_padding_from_hidden_states(
    generated_tokens: Tensor,
    hidden_states: Tensor,
    input_length: int,
    tokenizer: PreTrainedTokenizer,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generated_tokens = generated_tokens[:, 1:]
    padding_mask = generated_tokens == tokenizer.pad_token_id

    masked_hiddens: list[Tensor] = []
    masked_tokens: list[Tensor] = []
    last_input_token_idx: list[int] = []
    for hs_row, tokens_row, pad_mask in zip(hidden_states, generated_tokens, padding_mask):
        masked_hiddens.append(hs_row[~pad_mask].clone())
        masked_tokens.append(tokens_row[~pad_mask].clone())
        last_input_token_idx.append(input_length - pad_mask[:input_length].sum().item() - 1)

    return masked_hiddens, masked_tokens, last_input_token_idx


def get_llm_config(root_data_dir: Path) -> PretrainedConfig:
    config = load_and_resolve_config(root_data_dir / "config.yaml")
    llm_name = config["llm"]["name"]
    return AutoConfig.from_pretrained(llm_name)


def get_llm_summary(root_data_dir: Path) -> dict[str, Any]:
    config = get_llm_config(root_data_dir)
    return {
        "name": config.name_or_path,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
    }


if __name__ == "__main__":
    data_dir = Path(
        "data/activations/nq_open/llama_3.1_8b_instruct/sampling_high_temp_with_activations__prompt_short_few_shot_sep__seed_42"
    )
    data = load_hidden_states_dataset_for_single_layer(
        root_data_dir=data_dir, layer_idx=0, remove_padding=True
    )
    print(data.last_input_token_idx)
    print(f"Num samples: {data.num_samples}")
    print(f"Num samples (after filtering): {data.filter_data_with_invalid_labels().num_samples}")
