from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer


def remove_padding_from_intermediate_states(
    per_layer_batched_data: list[Tensor],
    data_type: Literal["attn", "hs"],
    generated_tokens: Tensor,
    tokenizer: PreTrainedTokenizer | None = None,
    pad_token_id: int | None = None,
) -> list[list[Tensor]]:
    """Takes stacked attention matrix (attentions per each layer) and removes padding tokens from it
    Dimensions of the input are [#layers, [batch_size x num_heads x sequence_length x sequence_length]].
    Returned results is of shape [#examples, [#layers, [num_heads x sequence_length x sequence_length]]].
    """
    assert tokenizer is not None or pad_token_id is not None
    num_layers = len(per_layer_batched_data)
    num_examples = per_layer_batched_data[0].size(0)

    generated_tokens = generated_tokens[:, :-1]

    if data_type == "attn":
        assert per_layer_batched_data[0].size(-1) == generated_tokens.size(-1)
        assert per_layer_batched_data[0].size(-2) == generated_tokens.size(-1)
    elif data_type == "hs":
        assert per_layer_batched_data[0].size(-2) == generated_tokens.size(-1)

    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = tokenizer.pad_token_id
    assert pad_token_id is not None

    results: list[list[Tensor]] = []
    for example_idx in range(num_examples):
        results.append([])
        example_pad_mask = left_right_contiguous_padding_mask(
            generated_tokens[example_idx], pad_token_id
        )
        for layer_idx in range(num_layers):
            masked_attn_scores = per_layer_batched_data[layer_idx][
                example_idx, :, ~example_pad_mask
            ][:, :, ~example_pad_mask].clone()
            results[-1].append(masked_attn_scores)

            summed_attn = masked_attn_scores.sum(dim=-1)
            assert torch.isclose(
                summed_attn,
                torch.tensor(1.0, dtype=summed_attn.dtype),
                atol=1e-2,  # due to unknown reasons, the margin is quite large
            ).all()
    return results


def left_right_contiguous_padding_mask(tokens: Tensor, pad_token_id: int) -> Tensor:
    """Prepares mask for 1D tensor which allows to strip padding from the left and right side of the tensor."""
    # Create a boolean mask where the target value is present
    is_pad = (tokens != pad_token_id).long()

    # Get the indices of the first and last occurrence of the target value along each row
    first_occurrence = torch.argmax(is_pad)
    last_occurrence = tokens.size(0) - 1 - torch.argmax(is_pad.flip(dims=[0]))

    # Create a range tensor to compare against
    range_tensor = torch.arange(tokens.size(0), device=tokens.device)
    # Generate the mask
    mask = (range_tensor < first_occurrence) | (range_tensor > last_occurrence)

    return mask.bool()


def pad_tensor(tensor: Tensor, max_length: int) -> Tensor:
    return torch.nn.functional.pad(tensor, (0, max_length - tensor.size(-1)), "constant", 0.0)


def get_sequences_by_layer(
    interm_state: tuple[tuple[Tensor, ...], ...],
    concat: bool,
) -> list[Tensor] | list[list[Tensor]]:
    """Return hidden states with layers as a first dimension.
    Shape of the output: (num_layers, [batch_size, sequence_length, hidden_size])
    """
    layerwise_states: list[list[Tensor]] = []
    for gen_token_data in interm_state:
        for layer_idx, layer_data in enumerate(gen_token_data):
            try:
                layerwise_states[layer_idx].append(layer_data.cpu())
            except IndexError:
                layerwise_states.append([layer_data.cpu()])

    if concat:
        return [torch.cat(layer_data, dim=-2) for layer_data in layerwise_states]
    else:
        return layerwise_states
