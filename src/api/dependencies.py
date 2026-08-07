"""FastAPI dependency helpers."""

from __future__ import annotations

from database.session import get_db_session

__all__ = ["get_db_session"]
