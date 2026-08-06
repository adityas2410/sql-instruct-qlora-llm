"""SQLAlchemy helpers for PostgreSQL pgvector columns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import UserDefinedType


class PgVector(UserDefinedType):
    """Represent a PostgreSQL pgvector column with a fixed dimension.

    The project stores vectors produced by the custom Skip-Gram embedding model.
    This type only describes serialization for SQLAlchemy; pgvector remains the
    storage and nearest-neighbour search layer.
    """

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **_: Any) -> str:
        """Return the PostgreSQL column definition for this vector."""
        return f"VECTOR({self.dimension})"

    @property
    def python_type(self) -> type[list[float]]:
        """Expose Python values as lists of floats."""
        return list

    def bind_processor(self, dialect: Dialect):
        """Serialize Python vector values for PostgreSQL."""

        def process(value: Sequence[float] | None) -> str | None:
            if value is None:
                return None
            if len(value) != self.dimension:
                raise ValueError(
                    f"Expected vector dimension {self.dimension}, received {len(value)}"
                )
            return "[" + ",".join(str(float(item)) for item in value) + "]"

        return process

    def result_processor(self, dialect: Dialect, coltype: object):
        """Deserialize pgvector values returned by psycopg."""

        def process(value: Any) -> list[float] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return [float(item) for item in value]
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            if not text:
                return []
            return [float(item) for item in text.split(",")]

        return process
