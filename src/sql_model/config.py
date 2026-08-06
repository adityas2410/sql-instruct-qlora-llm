"""Configuration for the Falcon QLoRA SQL generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FALCON_SQL_BASE_MODEL_ID = "tiiuae/falcon-11B"
DEFAULT_FALCON_TARGET_MODULES = (
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)


@dataclass(frozen=True)
class FalconLoraConfig:
    """LoRA adapter settings for Falcon SQL instruction tuning."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_FALCON_TARGET_MODULES
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class FalconQuantizationConfig:
    """4-bit quantization settings for QLoRA training."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class FalconTrainingConfig:
    """Training arguments used by the supervised fine-tuning wrapper."""

    output_dir: Path = Path("artifacts/sql_model/falcon_sql_sft")
    adapter_output_dir: Path = Path("models/sql_lora_adapter")
    max_seq_length: int = 2048
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FalconGenerationConfig:
    """Generation settings for SQL inference."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.05


class FalconSQLSettings(BaseSettings):
    """Environment-backed settings for Falcon SQL training and inference."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    base_model_id: str = Field(default=FALCON_SQL_BASE_MODEL_ID, alias="SQL_MODEL_BASE_ID")
    adapter_dir: Path = Field(default=Path("models/sql_lora_adapter"), alias="SQL_MODEL_ADAPTER_DIR")
    output_dir: Path = Field(default=Path("artifacts/sql_model"), alias="SQL_MODEL_OUTPUT_DIR")
    use_4bit: bool = Field(default=True, alias="SQL_MODEL_USE_4BIT")
    use_wandb: bool = Field(default=False, alias="SQL_MODEL_USE_WANDB")
    wandb_project: str = Field(default="insurance-sql-agent", alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, alias="WANDB_ENTITY")
    train_jsonl_path: Path = Field(
        default=Path("data/generated/sql_instruction_train.jsonl"),
        alias="SQL_TRAIN_JSONL_PATH",
    )
    eval_jsonl_path: Path = Field(
        default=Path("data/generated/sql_instruction_eval.jsonl"),
        alias="SQL_EVAL_JSONL_PATH",
    )

    @field_validator("base_model_id")
    @classmethod
    def require_falcon_11b(cls, value: str) -> str:
        """Keep this project pinned to the Falcon 11B SQL fine-tuning lineage."""
        if value != FALCON_SQL_BASE_MODEL_ID:
            raise ValueError(f"SQL_MODEL_BASE_ID must be {FALCON_SQL_BASE_MODEL_ID}")
        return value


def build_training_config(settings: FalconSQLSettings) -> FalconTrainingConfig:
    """Build training config from environment-backed settings."""
    report_to = ("wandb",) if settings.use_wandb else ()
    return FalconTrainingConfig(
        output_dir=settings.output_dir / "falcon_sql_sft",
        adapter_output_dir=settings.adapter_dir,
        report_to=report_to,
    )
