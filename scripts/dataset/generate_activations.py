import copy
import os
from pprint import pformat

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch_geometric import seed_everything

from hallucinations.config import GenerateActivationsConfig
from hallucinations.data.factory import prepare_dataset
from hallucinations.dirs import DatasetDir
from hallucinations.features.hidden_states import HiddenStatesSelection
from hallucinations.llm.activation_storage import (
    ActivationStorage,
    AllActivationsStorage,
    AttentionsOnlyStorage,
    HiddenStatesOnlyStorage,
)
from hallucinations.llm.factory import get_llm
from hallucinations.llm.feature_storage import (
    AttentionAndHiddenStatesFeatureStorage,
    AttentionAndLaplacianDiagsFeatureStorage,
    HiddenStatesFeatureStorage,
)
from hallucinations.llm.predict import predict_with_llm
from hallucinations.llm.preprocessing import SimpleEncoder
from hallucinations.utils import resolve_config, save_json, save_yaml

NUM_PROC = int(os.getenv("NUM_PROC", 1))
NUM_SAVE_WORKERS = int(os.getenv("NUM_SAVE_WORKERS", 4))

if torch.cuda.device_count() != 1:
    raise ValueError("This script requires a single CUDA device.")


@hydra.main(version_base="1.3", config_path="../../config", config_name="generate_activations")
def main(cfg: DictConfig) -> None:
    config = GenerateActivationsConfig(**resolve_config(cfg))
    dataset_dir = DatasetDir(config.results_dir)
    logger.info(f"Config: {pformat(config.model_dump())}")

    seed_everything(config.random_seed)

    raw_ds, dataset = prepare_dataset(
        dataset_config=config.dataset,
        split=config.split,
        prompt_config=config.prompt,
        use_output=False,
        return_raw=True,
        seed=config.random_seed,
    )
    # NOTE: Changing order of the dataset by length might cause ordering issues when saving activations

    model_pack = get_llm(
        config.llm,
        device_map="auto",  # loads model in a balanced mode on all available GPUs
    )

    if config.llm.compile:
        # NOTE: using built-in hf compile results in wall of warnings
        model_pack.llm = torch.compile(model_pack.llm)  # type: ignore

    dataset.set_transform(SimpleEncoder(model_pack.tokenizer))
    activation_storage: ActivationStorage
    if config.stored_features == "attentions":
        activation_storage = AttentionsOnlyStorage(
            save_dir=dataset_dir.full_attentions_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            verbose=True,
        )
    elif config.stored_features == "hidden_states":
        activation_storage = HiddenStatesOnlyStorage(
            save_dir=dataset_dir.hidden_states_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            verbose=True,
        )
    elif config.stored_features == "attention_diags":
        activation_storage = AttentionAndLaplacianDiagsFeatureStorage(
            ds_dir=dataset_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            pad_token_id=model_pack.tokenizer.pad_token_id,
            verbose=True,
        )
    elif config.stored_features == "hidden_states_for_last_input_last_gen_tokens":
        activation_storage = HiddenStatesFeatureStorage(
            ds_dir=dataset_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            hs_selection=HiddenStatesSelection(
                layer="all",
                hs_last_input_token=True,
                hs_last_generated_token=True,
            ),
            verbose=True,
        )
    elif (
        config.stored_features == "attention_diags_and_hidden_states_for_last_input_last_gen_tokens"
    ):
        hs_features_storage = HiddenStatesFeatureStorage(
            ds_dir=dataset_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            hs_selection=HiddenStatesSelection(
                layer="all",
                hs_last_input_token=True,
                hs_last_generated_token=True,
            ),
            verbose=True,
        )
        attn_features_storage = AttentionAndLaplacianDiagsFeatureStorage(
            ds_dir=dataset_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            pad_token_id=model_pack.tokenizer.pad_token_id,
            verbose=True,
        )
        activation_storage = AttentionAndHiddenStatesFeatureStorage(
            attn_features_storage,
            hs_features_storage,
        )
    else:
        activation_storage = AllActivationsStorage(
            save_dir=dataset_dir.activations_dir,
            max_save_workers=NUM_SAVE_WORKERS,
            verbose=True,
        )

    logger.info(f"Using activation storage: {activation_storage}")

    assert model_pack.llm.generation_config is not None
    assert not any(key in model_pack.generate_kwargs for key in config.generation_config)
    generation_config = copy.deepcopy(model_pack.llm.generation_config)
    generation_config.update(**(config.generation_config | model_pack.generate_kwargs))

    logger.info(f"Generation config: {generation_config}")

    preds = predict_with_llm(
        model=model_pack.llm,
        tokenizer=model_pack.tokenizer,
        dataset=dataset,
        generation_config=generation_config,
        activation_storage=activation_storage,
        batch_size=config.batch_size,
        num_proc=NUM_PROC,
    )

    golds = [ans for ans in raw_ds[config.dataset.target_column_name]]
    results = [{"prediction": ans, "gold": g} for ans, g in zip(preds, golds, strict=True)]

    activation_storage.flush()
    save_json(dataset_dir.answers_file, results)
    save_yaml(dataset_dir.config_file, config.model_dump())


if __name__ == "__main__":
    main()
