"""Falcon LoRA adapter inference for SQL generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database.schema import render_schema_for_prompt
from sql_model.config import FALCON_SQL_BASE_MODEL_ID, FalconGenerationConfig, FalconSQLSettings
from sql_model.prompts import render_sql_prompt


@dataclass(frozen=True)
class SQLGenerationResult:
    """Generated SQL and lightweight inference metadata."""

    sql: str
    base_model_id: str
    adapter_dir: str
    prompt_characters: int


class FalconSQLGenerator:
    """Generate PostgreSQL from investigator requests using Falcon adapters."""

    def __init__(
        self,
        settings: FalconSQLSettings | None = None,
        generation_config: FalconGenerationConfig | None = None,
    ) -> None:
        self.settings = settings or FalconSQLSettings()
        self.generation_config = generation_config or FalconGenerationConfig()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self, adapter_dir: Path | None = None) -> None:
        """Load Falcon and the fine-tuned LoRA adapter for inference."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_path = adapter_dir or self.settings.adapter_dir
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            FALCON_SQL_BASE_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base_model, adapter_path)
        self._model.eval()
        self._tokenizer = tokenizer

    def generate_sql(self, instruction: str, schema: str | None = None) -> SQLGenerationResult:
        """Generate one SQL statement for an investigator request."""
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt = render_sql_prompt(
            instruction=instruction,
            schema=schema or render_schema_for_prompt(),
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=self.generation_config.do_sample,
            repetition_penalty=self.generation_config.repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        sql = self._extract_sql(decoded, prompt)
        return SQLGenerationResult(
            sql=sql,
            base_model_id=FALCON_SQL_BASE_MODEL_ID,
            adapter_dir=str(self.settings.adapter_dir),
            prompt_characters=len(prompt),
        )

    @staticmethod
    def _extract_sql(decoded_text: str, prompt: str) -> str:
        """Extract SQL text after the prompt and strip extra formatting."""
        sql = decoded_text[len(prompt) :] if decoded_text.startswith(prompt) else decoded_text
        sql = sql.strip().strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
        return sql.rstrip(";") + ";" if sql else sql
