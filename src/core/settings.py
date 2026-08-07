"""Environment-backed application settings."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Shared application and agent settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="sql-instruct-qlora-llm", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    agent_framework: str = Field(default="huggingface_smolagents", alias="AGENT_FRAMEWORK")
    agent_model_provider: str = Field(default="local_falcon_adapter", alias="AGENT_MODEL_PROVIDER")
    agent_max_steps: int = Field(default=6, alias="AGENT_MAX_STEPS")
    agent_use_base_tools: bool = Field(default=False, alias="AGENT_USE_BASE_TOOLS")
    agent_require_grounded_evidence: bool = Field(
        default=True,
        alias="AGENT_REQUIRE_GROUNDED_EVIDENCE",
    )

    @field_validator("agent_framework")
    @classmethod
    def require_smolagents_framework(cls, value: str) -> str:
        """Keep the application wired to the Hugging Face agent framework."""
        if value != "huggingface_smolagents":
            raise ValueError("AGENT_FRAMEWORK must be huggingface_smolagents")
        return value
