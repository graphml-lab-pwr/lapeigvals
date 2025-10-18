import time

import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig

from hallucinations.llm.activation_storage import ActivationStorage


@torch.inference_mode()
def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    generation_config: GenerationConfig,
    activation_storage: ActivationStorage | None,
    batch_size: int,
    num_proc: int,
) -> list[str]:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    model_outputs = []

    device = next(model.parameters()).device

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Generating predictions",
    ) as pbar:
        for i, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_length = input_ids.size(1)

            start_time = time.time()
            outputs = model.generate(  # type: ignore
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            duration = time.time() - start_time

            if isinstance(outputs, GenerateDecoderOnlyOutput):
                assert activation_storage is not None, (
                    "activation_storage must be provided for GenerateDecoderOnlyOutput"
                )
                outputs.sequences = outputs.sequences.cpu()  # type: ignore
                generated_ids = outputs.sequences
                token_masks = get_token_masks(outputs.sequences, tokenizer)
                activation_storage.update(
                    outputs=outputs,
                    attention_mask=attention_mask,
                    special_token_mask=token_masks["special_token_mask"],
                    decoder_added_token_mask=token_masks["decoder_added_token_mask"],
                    input_length=input_length,
                    batch_idx=i,
                )
            elif isinstance(outputs, Tensor):
                generated_ids = outputs.cpu()  # type: ignore
            else:
                raise ValueError(f"Unexpected generation output: {type(outputs)}")

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            model_outputs.extend(decoded)

            stats = {
                "input_size": input_length,
                "throughput": f"{generated_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

            del outputs, input_ids, attention_mask, generated_ids
            torch.cuda.empty_cache()

    return model_outputs


def get_token_masks(token_ids: Tensor, tokenizer: PreTrainedTokenizer) -> dict[str, Tensor]:
    special_token_masks = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(
                seq_tok_ids,
                already_has_special_tokens=True,
            )
            for seq_tok_ids in token_ids
        ]
    )

    decoder_added_token_mask = torch.tensor(
        [
            [tok_id in tokenizer.added_tokens_decoder.keys() for tok_id in seq_token_ids]
            for seq_token_ids in token_ids
        ]
    )

    return {
        "special_token_mask": special_token_masks,
        "decoder_added_token_mask": decoder_added_token_mask,
    }
