from pathlib import Path

import torch
import typer
from transformers import AutoTokenizer

from hallucinations.dirs import BATCH_ATTN_FILENAME, DatasetDir
from hallucinations.features.attention_weights import (
    attention_diagonal,
    laplacian_diagonal_from_attn,
    yield_stacked_attentions_from_shard_list,
)


def main(
    dataset_dir: Path = typer.Option(...),
) -> None:
    ds_dir = DatasetDir(dataset_dir)

    attn_diags = []
    lap_diags = []

    num_attn_files = len(
        list(ds_dir.full_attentions_dir.glob(BATCH_ATTN_FILENAME.format(batch_idx="*")))
    )
    attn_files = [
        ds_dir.full_attentions_dir / BATCH_ATTN_FILENAME.format(batch_idx=batch_idx)
        for batch_idx in range(num_attn_files)
    ]

    config = ds_dir.load_raw_config()
    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["name"])

    for batch_attns in yield_stacked_attentions_from_shard_list(
        attn_files,
        remove_padding=True,
        tokenizer=tokenizer,
    ):
        for item_attn in batch_attns:
            attn_diags.append(attention_diagonal(item_attn))  # type: ignore [arg-type]
            lap_diags.append(laplacian_diagonal_from_attn(item_attn, vertical_edges=False))  # type: ignore [arg-type]

    ds_dir.features_dir.mkdir(parents=True, exist_ok=True)
    torch.save(attn_diags, ds_dir.attn_diags_file)
    torch.save(lap_diags, ds_dir.laplacian_diags_file)


if __name__ == "__main__":
    typer.run(main)
