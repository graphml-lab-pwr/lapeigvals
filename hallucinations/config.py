from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from jinja2 import Template
from pydantic import BaseModel, Field, model_validator

from hallucinations.utils import load_and_resolve_config
from hallucinations.utils.misc import import_cls_from_str


class LlmConfig(BaseModel, extra="forbid"):
    name: str | Path
    tokenizer_name: str | Path
    tokenizer_padding_side: Literal["left", "right"]
    context_size: int
    compile: bool
    torch_dtype: str
    attn_implementation: str
    quantization: dict[str, Any] | None = None


class DatasetConfig(BaseModel, extra="forbid"):
    cls_path: str
    name: str | Path
    test_split_name: str | None
    max_answer_tokens: int
    target_column_name: str


class QaDatasetConfig(DatasetConfig, extra="forbid"):
    path: Path | None = None
    subset: str | None = None


class PromptConfig(BaseModel, extra="forbid"):
    content: str


class QaPromptConfig(PromptConfig, extra="forbid"):
    question_key: str
    context_key: str | None = None
    num_few_shot_examples: int | None = None


class GenerateActivationsConfig(BaseModel, extra="forbid"):
    llm: LlmConfig
    dataset: QaDatasetConfig
    prompt: QaPromptConfig
    split: str | None
    batch_size: int
    generation_config: dict[str, Any]
    stored_features: Literal[
        "attentions",
        "attention_diags",
        "hidden_states",
        "attention_with_hidden_states",
        "hidden_states_for_last_input_last_gen_tokens",
        "attention_diags_and_hidden_states_for_last_input_last_gen_tokens",
    ]
    results_dir: Path
    random_seed: int

    @model_validator(mode="before")
    @classmethod
    def create_dataset_class(cls, data: Any) -> Any:
        dataset_config_cls_path = data["dataset"].get("cls_path")
        if dataset_config_cls_path is None:
            raise ValueError("'cls_path' must be provided.")

        dataset_config_cls = import_cls_from_str(dataset_config_cls_path)
        data["dataset"] = dataset_config_cls(**data["dataset"])
        return data

    @model_validator(mode="after")
    def update_generation_config(self) -> "GenerateActivationsConfig":
        assert "return_dict_in_generate" not in self.generation_config
        assert "output_attentions" not in self.generation_config
        assert "output_hidden_states" not in self.generation_config

        self.generation_config["return_dict_in_generate"] = True

        self.generation_config["output_attentions"] = self.stored_features in [
            "attentions",
            "attention_diags",
            "attention_with_hidden_states",
            "attention_diags_and_hidden_states_for_last_input_last_gen_tokens",
        ]
        self.generation_config["output_hidden_states"] = self.stored_features in [
            "hidden_states",
            "attention_with_hidden_states",
            "hidden_states_for_last_input_last_gen_tokens",
            "attention_diags_and_hidden_states_for_last_input_last_gen_tokens",
        ]

        return self

    @property
    def max_input_length(self) -> int:
        return self.llm.context_size - self.dataset.max_answer_tokens

    @property
    def answers_file(self) -> Path:
        return self.results_dir / "answers.json"

    @property
    def metrics_file(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def config_file(self) -> Path:
        return self.results_dir / "config.yaml"


class LlmJudgePromptConfig(BaseModel, extra="forbid"):
    name: str
    system_prompt: str | None = None
    content: str
    question_key: str
    predicted_answer_key: str
    gold_answer_key: str
    possible_answers: list[str]  # write them in descending priority (for combining)
    separate_multi_answers: bool

    @cached_property
    def prompt_template(self) -> Template:
        return Template(self.content)

    def format(self, question: str, pred_answer: str, gold_answer: list[str] | str) -> str:
        return self.prompt_template.render(
            **{
                self.question_key: question,
                self.predicted_answer_key: pred_answer,
                self.gold_answer_key: gold_answer,
            }
        )


class LlmApiConfig(BaseModel, extra="forbid"):
    name: str = Field(help="Project-wise simplified name of the LLM API.")  # type: ignore
    version: str = Field(help="Version of the LLM API to use.")  # type: ignore
    base_url: str | None
    batch_size: int


class LllmJudgeConfig(BaseModel, extra="forbid"):
    llm_api: LlmApiConfig
    prompt: LlmJudgePromptConfig
    answers_file: Path

    @property
    def dataset(self) -> QaDatasetConfig:
        return QaDatasetConfig(**load_and_resolve_config(self.config_file)["dataset"])

    @property
    def config_file(self) -> Path:
        return self.answers_file.with_name("config.yaml")

    @property
    def evaluation_file(self) -> Path:
        return (
            self.answers_file.parent
            / "llm_judge"
            / f"llm_judge_results_{self.llm_api.name}_{self.prompt.name}.json"
        )

    @property
    def evaluation_config_file(self) -> Path:
        return (
            self.answers_file.parent
            / "llm_judge"
            / f"llm_judge_config_{self.llm_api.name}_{self.prompt.name}.yaml"
        )

    @property
    def evaluation_metadata_file(self) -> Path:
        return (
            self.answers_file.parent
            / "llm_judge"
            / f"llm_judge_metadata_{self.llm_api.name}_{self.prompt.name}.json"
        )
