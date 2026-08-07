"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import agent, health, semantic, sql
from core.settings import AppSettings


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or AppSettings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Insurance investigation API with Falcon SQL and semantic claim retrieval.",
    )
    application.include_router(health.router)
    application.include_router(sql.router)
    application.include_router(semantic.router)
    application.include_router(agent.router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
