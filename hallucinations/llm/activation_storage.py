from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from transformers.generation import GenerateDecoderOnlyOutput

from hallucinations.features.processing import get_sequences_by_layer

# hf-transformers return a tuple of tensors representing the hidden states/attentions of different layers
# dimensions are: (num_new_tokens, num_layers, tensor[batch_size, sequence_length, hidden_size])
# in particular, data for token 0 represents hidden states for the whole input sequence (thus sequence_length = input_length)
# and data for the rest of the tokens represents hidden states for the generated tokens (thus each one has sequence_length = 1)
# NOTE 1: Special tokens don't mark chat-template tokens
# NOTE 2: Even when mask chat-template tokens is given, not all might be covered, e.g., in <spec_tok> and <spec_tok> "and" won't be marked


class ActivationStorage(ABC):
    """Extract intermediate states of an LLM and save them to disk."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.update(*args, **kwargs)

    @abstractmethod
    def update(
        self,
        outputs: GenerateDecoderOnlyOutput,
        attention_mask: Tensor,
        special_token_mask: Tensor,
        decoder_added_token_mask: Tensor,
        input_length: int,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    def flush(self) -> None:
        pass


class AllActivationsStorage(ActivationStorage):
    """Saves all activations to disk.
    The saved hidden_states has shape: (num_layers, [batch_size, sequence_length, hidden_size])
    """

    def __init__(
        self,
        save_dir: Path,
        max_save_workers: int,
        verbose: bool = True,
    ):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._check_save_dir()

        self.verbose = verbose

        self.max_save_workers = max_save_workers
        self.save_executor = ThreadPoolExecutor(max_workers=max_save_workers)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(save_dir={self.save_dir}, max_save_workers={self.max_save_workers}, verbose={self.verbose})"

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
        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "generated_tokens": outputs.sequences.cpu(),
        }
        if "hidden_states" in outputs:
            assert outputs.hidden_states is not None
            intermediate_states["hidden_states"] = get_sequences_by_layer(
                outputs.hidden_states,
                concat=True,
            )
        if "attentions" in outputs:
            intermediate_states["attentions"] = outputs.attentions

        self.save(intermediate_states, batch_idx)

    def flush(self) -> None:
        self.save_executor.shutdown(wait=True)
        if self.verbose:
            files = list(self.save_dir.glob("*.pt"))
            size = sum(file.stat().st_size for file in files)
            logger.info(f"Stored total {size * 1e-9:0.1f}GB in {len(files)} files")

    def save(self, intermediate_states: dict[str, Any], batch_idx: int) -> None:
        self.save_executor.submit(
            self._do_save,
            intermediate_states=intermediate_states,
            batch_idx=batch_idx,
        )

    def _do_save(self, intermediate_states: dict[str, Any], batch_idx: int) -> None:
        save_file = self.save_dir / f"batch_{batch_idx}.pt"
        torch.save(intermediate_states, save_file)
        if self.verbose:
            logger.info(
                f"Saved ({save_file.stat().st_size * 1e-9:0.1f}GB) activations to {save_file}"
            )

    def _check_save_dir(self) -> None:
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        legacy_data = list(self.save_dir.glob("batch_*.pt"))
        if len(legacy_data) > 0:
            raise FileExistsError(
                f"Save directory {self.save_dir} already contain data, remove it first."
            )

        other_data = list(self.save_dir.iterdir())
        if len(other_data) > 0:
            logger.warning(
                f"Save directory {self.save_dir} contains {len(other_data)} files that are not .pt files."
            )


class AttentionsOnlyStorage(AllActivationsStorage):
    """Saves only attention matrices to disk."""

    def __init__(
        self,
        save_dir: Path,
        max_save_workers: int,
        verbose: bool = True,
    ):
        super().__init__(save_dir, max_save_workers, verbose)

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

        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "generated_tokens": outputs.sequences.cpu(),
            "attentions": outputs.attentions,
        }

        self.save(intermediate_states, batch_idx)


class HiddenStatesOnlyStorage(AllActivationsStorage):
    """Saves only hidden states to disk."""

    def __init__(
        self,
        save_dir: Path,
        max_save_workers: int,
        verbose: bool = True,
    ):
        super().__init__(save_dir, max_save_workers, verbose)

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

        hidden_states = get_sequences_by_layer(outputs.hidden_states, concat=True)
        intermediate_states = {
            "attention_mask": attention_mask.cpu(),
            "special_token_mask": special_token_mask.cpu(),
            "decoder_token_mask": decoder_added_token_mask.cpu(),
            "input_length": input_length,
            "generated_tokens": outputs.sequences.cpu(),
            "hidden_states": hidden_states,
        }

        self.save(intermediate_states, batch_idx)
