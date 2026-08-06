"""Database engine and session configuration."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseSettings(BaseSettings):
    """Environment-backed database settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://insurance_user:insurance_password@localhost:5432/insurance_fraud",
        alias="DATABASE_URL",
    )
    sql_default_limit: int = Field(default=100, alias="SQL_DEFAULT_LIMIT")
    sql_max_limit: int = Field(default=1000, alias="SQL_MAX_LIMIT")
    pgvector_dimension: int = Field(default=128, alias="PGVECTOR_DIMENSION")
    echo_sql: bool = Field(default=False, alias="ECHO_SQL")


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Return cached database settings loaded from environment variables."""
    return DatabaseSettings()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the application SQLAlchemy engine.

    The engine is constructed lazily so importing modules never opens a database
    connection. Connections are established only when sessions execute work.
    """
    settings = get_database_settings()
    return create_engine(settings.database_url, echo=settings.echo_sql, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return a configured SQLAlchemy session factory."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope for scripts and services."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session for FastAPI dependency injection."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
