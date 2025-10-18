from collections import defaultdict
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers.generation import GenerateDecoderOnlyOutput

from hallucinations.dirs import DatasetDir
from hallucinations.features.attention_weights import (
    attention_diagonal,
    laplacian_diagonal_from_attn,
    stack_attention_matrix,
)
from hallucinations.features.hidden_states import (
    HiddenStatesSelection,
    select_hidden_states_from_layers,
)
from hallucinations.features.processing import remove_padding_from_intermediate_states
from hallucinations.llm.activation_storage import (
    ActivationStorage,
    AllActivationsStorage,
    get_sequences_by_layer,
)


class AttentionAndLaplacianDiagsFeatureStorage(AllActivationsStorage):
    """Saves minimal amount of intermediate data, and attention, laplacian diagonals."""

    def __init__(
        self,
        ds_dir: DatasetDir,
        max_save_workers: int,
        pad_token_id: int,
        verbose: bool = True,
    ):
        super().__init__(ds_dir.attentions_dir, max_save_workers, verbose)
        self.ds_dir = ds_dir
        # NOTE: attention won't be stored, only outputs
        self.ds_dir.attentions_dir.mkdir(parents=True, exist_ok=True)
        self.ds_dir.features_dir.mkdir(parents=True, exist_ok=True)

        self.pad_token_id = pad_token_id
        self.attention_diags: list[Tensor] = []
        self.laplacian_diags: list[Tensor] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(features_dir={self.ds_dir.features_dir}, max_save_workers={self.max_save_workers}, pad_token_id={self.pad_token_id}, verbose={self.verbose})"

    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        **kwargs: Any,
    ) -> None:
        batch_idx = kwargs["batch_idx"]
        assert outputs.attentions is not None

        generated_tokens = outputs.sequences.cpu()
        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "generated_tokens": generated_tokens,
        }
        self.extract_and_record_features_data(outputs.attentions, generated_tokens)
        self.save(intermediate_states, batch_idx)

    def extract_and_record_features_data(
        self,
        attentions: tuple[tuple[Tensor, ...], ...],
        generated_tokens: Tensor,
    ) -> None:
        attentions = _map_attentions_to_cpu(attentions)
        stacked_attn_scores = stack_attention_matrix(attentions)
        attn_shard_without_padding = remove_padding_from_intermediate_states(
            per_layer_batched_data=stacked_attn_scores,
            data_type="attn",
            generated_tokens=generated_tokens,
            pad_token_id=self.pad_token_id,
        )

        for attn_example in tqdm(
            attn_shard_without_padding,
            desc="Computing attention diagonals",
            leave=False,
        ):
            self.attention_diags.append(attention_diagonal(attn_example))
            self.laplacian_diags.append(
                laplacian_diagonal_from_attn(attn_example, vertical_edges=False)
            )

    def flush(self) -> None:
        super().flush()

        torch.save(self.attention_diags, self.ds_dir.attn_diags_file)
        torch.save(self.laplacian_diags, self.ds_dir.laplacian_diags_file)

        logger.info(
            f"Saved ({self.ds_dir.attn_diags_file.stat().st_size * 1e-9:0.1f}GB) "
            f"attention diagonals to {self.ds_dir.attn_diags_file}"
        )
        logger.info(
            f"Saved ({self.ds_dir.laplacian_diags_file.stat().st_size * 1e-9:0.1f}GB) "
            f"laplacian diagonals to {self.ds_dir.laplacian_diags_file}"
        )


class HiddenStatesFeatureStorage(AllActivationsStorage):
    """Saves hidden states features based on specified selection criteria."""

    def __init__(
        self,
        ds_dir: DatasetDir,
        max_save_workers: int,
        hs_selection: HiddenStatesSelection,
        verbose: bool = True,
    ):
        super().__init__(ds_dir.hidden_states_dir, max_save_workers, verbose)
        self.ds_dir = ds_dir
        self.ds_dir.hidden_states_dir.mkdir(parents=True, exist_ok=True)
        self.ds_dir.features_dir.mkdir(parents=True, exist_ok=True)

        self.hs_selection = hs_selection
        self.features: dict[str, list[Tensor | list[Tensor]]] = defaultdict(list)

    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        **kwargs: Any,
    ) -> None:
        batch_idx = kwargs["batch_idx"]
        assert outputs.hidden_states is not None

        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "generated_tokens": outputs.sequences.cpu(),
        }
        self.extract_and_record_features_data(outputs.hidden_states, intermediate_states)
        self.save(intermediate_states, batch_idx)

    def extract_and_record_features_data(
        self,
        raw_hidden_states: tuple[tuple[Tensor, ...], ...],
        intermediate_states: dict[str, Any],
    ) -> None:
        hidden_states = get_sequences_by_layer(raw_hidden_states, concat=True)
        hidden_states = _map_hidden_states_to_cpu(hidden_states)  # type: ignore

        shard = {
            "hidden_states": hidden_states,
            "special_token_mask": intermediate_states["special_token_mask"],
            "decoder_token_mask": intermediate_states["decoder_token_mask"],
            "input_length": intermediate_states["input_length"],
            "generated_tokens": intermediate_states["generated_tokens"],
        }

        selected_features = select_hidden_states_from_layers(
            shard=shard,
            hs_selection=self.hs_selection,
        )

        for feat_name, feat_data in selected_features.items():
            self.features[feat_name].append(feat_data)

    def flush(self) -> None:
        """Stores hidden states for selected tokens with dimension [#examples x #layers x hidden_dim]"""
        super().flush()
        features_file = self.ds_dir.hidden_states_for_last_input_last_gen_tokens_file

        features_t = {}
        for feat_name, feat_data in self.features.items():
            if all(isinstance(feat_data_item, list) for feat_data_item in feat_data):
                features_t[feat_name] = torch.stack(
                    [torch.stack(feat_data_item) for feat_data_item in feat_data]  # type: ignore
                ).squeeze()
            elif all(isinstance(feat_data_item, Tensor) for feat_data_item in feat_data):
                features_t[feat_name] = torch.stack(feat_data).squeeze()  # type: ignore
            else:
                raise ValueError(f"Unknown feature data type: {type(feat_data)}")

        torch.save(features_t, features_file)

        logger.info(
            f"Saved ({features_file.stat().st_size * 1e-9:0.1f}GB) "
            f"hidden states features to {features_file}"
        )


class AttentionAndHiddenStatesFeatureStorage(ActivationStorage):
    """Saves attention and hidden states features."""

    def __init__(
        self,
        attention_feature_storage: AttentionAndLaplacianDiagsFeatureStorage,
        hidden_states_feature_storage: HiddenStatesFeatureStorage,
    ):
        super().__init__()
        self.attention_feature_storage = attention_feature_storage
        self.hidden_states_feature_storage = hidden_states_feature_storage

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.attention_feature_storage.update(*args, **kwargs)
        self.hidden_states_feature_storage.update(*args, **kwargs)

    def flush(self) -> None:
        self.attention_feature_storage.flush()
        self.hidden_states_feature_storage.flush()


def _map_attentions_to_cpu(
    attentions: tuple[tuple[Tensor, ...], ...],
) -> tuple[tuple[Tensor, ...], ...]:
    return tuple(
        tuple(layer_attn.cpu() for layer_attn in gen_tok_attn) for gen_tok_attn in attentions
    )


def _map_hidden_states_to_cpu(
    hidden_states: list[Tensor],
) -> list[Tensor]:
    return [hs.cpu() for hs in hidden_states]
