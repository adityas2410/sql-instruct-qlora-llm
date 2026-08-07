"""Falcon PEFT adapter wrapper for Hugging Face smolagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sql_model.config import FALCON_SQL_BASE_MODEL_ID, FalconGenerationConfig, FalconSQLSettings


@dataclass(frozen=True)
class FalconAgentMessage:
    """Fallback message shape for smolagents-compatible model responses."""

    role: str
    content: str


class FalconSmolagentsModel:
    """Lazy-loading Falcon model callable for smolagents agents.

    The wrapper keeps Falcon as the only reasoning model. It loads the same base
    model and PEFT adapter path used by the SQL fine-tuning pipeline, but exposes
    a generic chat-style callable for the agent framework.
    """

    def __init__(
        self,
        settings: FalconSQLSettings | None = None,
        generation_config: FalconGenerationConfig | None = None,
    ) -> None:
        self.settings = settings or FalconSQLSettings()
        self.generation_config = generation_config or FalconGenerationConfig()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """Load Falcon base weights and the configured PEFT adapter."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.settings.adapter_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            FALCON_SQL_BASE_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base_model, self.settings.adapter_dir)
        self._model.eval()
        self._tokenizer = tokenizer

    def __call__(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: list[str] | None = None,
        **_: Any,
    ) -> Any:
        """Generate an assistant response for smolagents."""
        text = self.generate_text(self._render_messages(messages), stop_sequences=stop_sequences)
        try:
            from smolagents import ChatMessage

            return ChatMessage(role="assistant", content=text)
        except Exception:
            return FalconAgentMessage(role="assistant", content=text)

    def generate_text(self, prompt: str, stop_sequences: list[str] | None = None) -> str:
        """Generate text from Falcon for the agent loop."""
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None

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
        response = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
        if stop_sequences:
            response = self._truncate_at_stop_sequence(response, stop_sequences)
        return response

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> str:
        """Render smolagents chat messages into a Falcon text prompt."""
        rendered: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"{role}: {content}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    @staticmethod
    def _truncate_at_stop_sequence(text: str, stop_sequences: list[str]) -> str:
        end = len(text)
        for stop_sequence in stop_sequences:
            index = text.find(stop_sequence)
            if index >= 0:
                end = min(end, index)
        return text[:end].strip()
