"""Request and response schemas for the FastAPI application."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Application health payload."""

    status: str = "ok"
    app_name: str
    app_env: str
    agent_framework: str


class SQLGenerateRequest(BaseModel):
    """Request for Falcon SQL generation."""

    instruction: str = Field(min_length=1)
    schema_text: str | None = None


class SQLGenerateResponse(BaseModel):
    """Generated SQL response."""

    sql: str
    base_model_id: str
    adapter_dir: str
    prompt_characters: int


class SQLExecuteRequest(BaseModel):
    """Request for read-only SQL execution."""

    sql: str = Field(min_length=1)


class SQLExecuteResponse(BaseModel):
    """Read-only SQL execution response."""

    sql: str
    rows: list[dict[str, Any]]
    row_count: int


class SimilarClaimResponse(BaseModel):
    """One similar claim match returned by semantic retrieval."""

    claim_id: str
    similarity_score: float
    shared_tokens: list[str]
    shared_entities: list[str]
    historical_fraud_label: bool | None = None
    investigation_outcome: str | None = None


class SimilarClaimsResponse(BaseModel):
    """Semantic retrieval response for a source claim."""

    source_claim_id: str
    matches: list[SimilarClaimResponse]


class AgentQueryRequest(BaseModel):
    """Natural-language investigation request."""

    query: str = Field(min_length=1)


class AgentQueryResponse(BaseModel):
    """Agentic investigation response."""

    answer: str
    framework: str
    model_provider: str
    grounded: bool
