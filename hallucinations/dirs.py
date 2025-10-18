import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hallucinations.config import GenerateActivationsConfig, QaDatasetConfig, QaPromptConfig
from hallucinations.utils.misc import load_and_resolve_config

LAPLACIAN_DIAGS_FILENAME = "laplacian_diags.pt"
SORTED_LAPLACIAN_DIAGS_FILENAME = "sorted_laplacian_diags.pt"
ATTN_DIAGS_FILENAME = "attn_diags.pt"
LOG_DET_ATTN_FILENAME = "log_det_attn.pt"
BATCH_ATTN_FILENAME = "batch_{batch_idx}.pt"


class DatasetDir:
    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir)

    def __repr__(self) -> str:
        return f"DatasetDir({self.root_dir})"

    @property
    def activations_dir(self) -> Path:
        return self.root_dir / "activations"

    @property
    def attentions_dir(self) -> Path:
        return self.root_dir / "attentions"

    @property
    def full_attentions_dir(self) -> Path:
        """Directory with the whole attention matrices (not only the outputs corresponding to the diagonals)."""
        # todo: refactor dirs to differentiate between storing full data and features only
        return self.root_dir / "full_attentions"

    @property
    def hidden_states_dir(self) -> Path:
        return self.root_dir / "hidden_states"

    @property
    def llm_judge_dir(self) -> Path:
        return self.root_dir / "llm_judge"

    @property
    def config_file(self) -> Path:
        return self.root_dir / "config.yaml"

    @property
    def answers_file(self) -> Path:
        return self.root_dir / "answers.json"

    @property
    def split_file(self) -> Path:
        return self.root_dir / "split.pt"

    @property
    def labels_file(self) -> Path:
        return self.root_dir / "labels.pt"

    @property
    def metrics_file(self) -> Path:
        return self.root_dir / "metrics.json"

    @property
    def num_activations_shards(self) -> int:
        # todo: adjust for multiple dirs
        return len(list(self.activations_dir.glob("*.pt")))

    @property
    def features_dir(self) -> Path:
        return self.root_dir / "features"

    @property
    def log_det_attn_file(self) -> Path:
        return self.features_dir / LOG_DET_ATTN_FILENAME

    @property
    def attn_diags_file(self) -> Path:
        return self.features_dir / ATTN_DIAGS_FILENAME

    @property
    def laplacian_diags_file(self) -> Path:
        return self.features_dir / LAPLACIAN_DIAGS_FILENAME

    @property
    def sorted_laplacian_diags_file(self) -> Path:
        return self.features_dir / SORTED_LAPLACIAN_DIAGS_FILENAME

    @property
    def hidden_states_for_last_input_last_gen_tokens_file(self) -> Path:
        return self.features_dir / "hidden_states_for_last_input_last_gen_tokens.pt"

    def get_dataset_metadata(self) -> dict[str, Any]:
        cfg = self.load_raw_config()
        temp, prompt, seed = self.root_dir.name.split("__")

        match = re.search(r"-?\d+\.\d+|-?\d+", temp)  # Matches floats and integers
        if match:
            temp = f"{float(match.group()):.3f}"

        return {
            "dataset": cfg["dataset"]["name"],
            "llm": cfg["llm"]["name"],
            "prompt": prompt,
            "temp": temp,
            "seed": seed,
        }

    def llm_judge_file(self, llm_judge_prompt: str, llm_judge_llm: str) -> Path:
        return self.llm_judge_dir / f"llm_judge_results_{llm_judge_llm}_{llm_judge_prompt}.json"

    def list_llm_judge_configs(self) -> list[dict[str, str]]:
        configs = []
        for judge_cfg in self.llm_judge_dir.glob("*.yaml"):
            cfg = load_and_resolve_config(judge_cfg)
            # bakcward compatibility
            try:
                llm_name = cfg["llm_name"]
            except KeyError:
                llm_name = cfg["llm_api"]["name"]

            configs.append(
                {
                    "llm_judge_prompt": cfg["prompt"]["name"],
                    "llm_judge_llm": llm_name.replace("-", "_"),
                }
            )
        return configs

    def load_dataset_config(self) -> QaDatasetConfig:
        return self.load_config().dataset

    def load_prompt_config(self) -> QaPromptConfig:
        return self.load_config().prompt

    def load_config(self) -> GenerateActivationsConfig:
        raw_cfg = self.load_raw_config()

        # need to remove to avoid validation error
        # fields are automatically added by the config
        del raw_cfg["generation_config"]["return_dict_in_generate"]
        del raw_cfg["generation_config"]["output_attentions"]
        del raw_cfg["generation_config"]["output_hidden_states"]

        return GenerateActivationsConfig(**raw_cfg)

    def load_raw_config(self) -> dict[str, Any]:
        return load_and_resolve_config(self.config_file)

    def load_labels(self) -> dict[str, Tensor]:
        return torch.load(self.labels_file, weights_only=True)

    def load_split(self) -> dict[str, Tensor]:
        return torch.load(self.split_file, weights_only=True)

    @classmethod
    def from_results_dir(cls, results_dir: Path) -> "DatasetDir":
        *_, act_dir, data_dir, _ = results_dir.parents
        root_dir = data_dir / "activations" / results_dir.relative_to(act_dir)
        return cls(root_dir)


class ResultsDir:
    def __init__(self, root_dir: Path | str, dataset_dir: DatasetDir | Path | str | None = None):
        self.root_dir = Path(root_dir)
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir_to_obj(dataset_dir)
        else:
            self.dataset_dir = DatasetDir.from_results_dir(self.root_dir)

    def __hash__(self) -> int:
        return hash(self.root_dir)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ResultsDir):
            return False
        return self.root_dir == other.root_dir

    @property
    def attn_score_file(self) -> Path:
        return self.root_dir / "attn_score.pkl"

    @property
    def stats_file(self) -> Path:
        return self.root_dir / "dataset_stats.json"

    def attn_vs_lap_file(self, pca_dim: int | None = None) -> Path:
        if pca_dim is None:
            return self.root_dir / "attn_vs_laplacian_results.pkl"
        else:
            return self.root_dir / f"attn_vs_laplacian_pca_{pca_dim}_results.pkl"

    def hs_baseline_file(self, pca_dim: int | None = None) -> Path:
        if pca_dim is None:
            return self.root_dir / "hs_baselines_results.pkl"
        else:
            return self.root_dir / f"hs_baselines_pca_{pca_dim}_results.pkl"

    @classmethod
    def from_dataset_dir(cls, dataset_dir: DatasetDir | Path | str) -> "ResultsDir":
        ds_dir = dataset_dir_to_obj(dataset_dir)

        *_, act_dir, data_dir, _ = ds_dir.root_dir.parents
        root_dir = data_dir / "results" / ds_dir.root_dir.relative_to(act_dir)
        return cls(root_dir, ds_dir)


def dataset_dir_to_obj(dataset_dir: str | Path | DatasetDir) -> DatasetDir:
    if isinstance(dataset_dir, str):
        return DatasetDir(dataset_dir)
    elif isinstance(dataset_dir, Path):
        return DatasetDir(dataset_dir)
    elif isinstance(dataset_dir, DatasetDir):
        return dataset_dir
    else:
        raise ValueError(f"Unknown dataset dir type: {type(dataset_dir)}")
