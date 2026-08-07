"""Health-check route."""

from __future__ import annotations

from fastapi import APIRouter

from core.settings import AppSettings
from schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return application readiness metadata."""
    settings = AppSettings()
    return HealthResponse(
        app_name=settings.app_name,
        app_env=settings.app_env,
        agent_framework=settings.agent_framework,
    )
