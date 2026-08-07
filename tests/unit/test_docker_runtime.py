"""Static checks for Docker runtime configuration."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_runs_fastapi_application() -> None:
    dockerfile = read_text("Dockerfile")

    assert "FROM python:3.11-slim" in dockerfile
    assert "PYTHONPATH=/app/src" in dockerfile
    assert '"uvicorn", "api.main:app"' in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile
    assert '"--port", "8000"' in dockerfile


def test_compose_declares_api_and_pgvector_postgres() -> None:
    compose = read_text("docker-compose.yml")

    assert "api:" in compose
    assert "postgres:" in compose
    assert "image: pgvector/pgvector:pg16" in compose
    assert "postgres_data:/var/lib/postgresql/data" in compose
    assert "condition: service_healthy" in compose


def test_compose_mounts_existing_schema_migration() -> None:
    compose = read_text("docker-compose.yml")

    assert "./src/database/migrations/001_create_insurance_schema.sql" in compose
    assert "/docker-entrypoint-initdb.d/001_create_insurance_schema.sql:ro" in compose


def test_compose_exposes_runtime_artifact_mounts() -> None:
    compose = read_text("docker-compose.yml")

    assert "./data:/app/data" in compose
    assert "./models:/app/models" in compose
    assert "./artifacts:/app/artifacts" in compose


def test_dockerignore_excludes_generated_outputs_and_model_artifacts() -> None:
    dockerignore = read_text(".dockerignore")

    assert "data/generated/" in dockerignore
    assert "artifacts/" in dockerignore
    assert "models/" in dockerignore
    assert "wandb/" in dockerignore
    assert "*.safetensors" in dockerignore
    assert "*.pt" in dockerignore
