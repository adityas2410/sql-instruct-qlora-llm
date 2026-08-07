"""Falcon QLoRA supervised fine-tuning pipeline for SQL generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from sql_model.config import (
    FALCON_SQL_BASE_MODEL_ID,
    FalconLoraConfig,
    FalconQuantizationConfig,
    FalconSQLSettings,
    FalconTrainingConfig,
    build_training_config,
)
from sql_model.dataset import load_instruction_jsonl, to_huggingface_dataset
from sql_model.quantization import (
    attach_lora_adapter,
    build_bitsandbytes_config,
    prepare_model_for_qlora,
)
from sql_model.wandb_logging import WandbLogger, WandbRunSettings


class FalconSQLTrainer:
    """Train Falcon LoRA adapters for SQL instruction generation."""

    def __init__(
        self,
        settings: FalconSQLSettings | None = None,
        training_config: FalconTrainingConfig | None = None,
        lora_config: FalconLoraConfig | None = None,
        quantization_config: FalconQuantizationConfig | None = None,
    ) -> None:
        self.settings = settings or FalconSQLSettings()
        self.training_config = training_config or build_training_config(self.settings)
        self.lora_config = lora_config or FalconLoraConfig()
        self.quantization_config = quantization_config or FalconQuantizationConfig(
            load_in_4bit=self.settings.use_4bit
        )

    def train(
        self,
        train_jsonl_path: Path | None = None,
        eval_jsonl_path: Path | None = None,
    ) -> Path:
        """Run supervised fine-tuning and save Falcon LoRA adapters."""
        train_path = train_jsonl_path or self.settings.train_jsonl_path
        eval_path = eval_jsonl_path or self.settings.eval_jsonl_path
        train_examples = load_instruction_jsonl(train_path)
        eval_examples = load_instruction_jsonl(eval_path) if eval_path.exists() else []
        train_dataset = to_huggingface_dataset(train_examples)
        eval_dataset = to_huggingface_dataset(eval_examples) if eval_examples else None

        logger = WandbLogger(
            WandbRunSettings(
                enabled=self.settings.use_wandb,
                project=self.settings.wandb_project,
                entity=self.settings.wandb_entity,
                run_name="falcon-sql-qlora",
                tags=("falcon-11b", "qlora", "sql-generation"),
            )
        )
        logger.start(config=self._wandb_config(train_path=train_path, eval_path=eval_path))
        try:
            model, tokenizer = self.load_trainable_model()
            trainer = self.build_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()
            self.training_config.adapter_output_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(self.training_config.adapter_output_dir)
            tokenizer.save_pretrained(self.training_config.adapter_output_dir)
            return self.training_config.adapter_output_dir
        finally:
            logger.finish()

    def load_trainable_model(self) -> tuple[Any, Any]:
        """Load Falcon, configure QLoRA, and attach LoRA adapters."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FALCON_SQL_BASE_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            FALCON_SQL_BASE_MODEL_ID,
            quantization_config=build_bitsandbytes_config(self.quantization_config),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_qlora(model)
        model = attach_lora_adapter(model, self.lora_config)
        if self.training_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, tokenizer

    def build_trainer(
        self,
        *,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any | None,
    ) -> Any:
        """Build a TRL SFTTrainer for text-to-SQL instruction tuning."""
        from trl import SFTConfig, SFTTrainer

        args = SFTConfig(
            output_dir=str(self.training_config.output_dir),
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            report_to=list(self.training_config.report_to),
            eval_strategy="steps" if eval_dataset is not None else "no",
            dataset_text_field="text",
            max_length=self.training_config.max_seq_length,
            packing=False,
        )
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
        )

    def _wandb_config(self, *, train_path: Path, eval_path: Path) -> dict[str, Any]:
        """Build a serializable W&B config payload."""
        return {
            "base_model_id": FALCON_SQL_BASE_MODEL_ID,
            "train_jsonl_path": str(train_path),
            "eval_jsonl_path": str(eval_path),
            "training": _json_safe(asdict(self.training_config)),
            "lora": _json_safe(asdict(self.lora_config)),
            "quantization": _json_safe(asdict(self.quantization_config)),
        }


def _json_safe(value: Any) -> Any:
    """Convert paths and tuples in nested config values into JSON-friendly data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
