"""Read-only SQL validation and execution service."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlglot
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlglot import exp

from database.schema import APPROVED_SCHEMA, APPROVED_TABLE_NAMES

LIMIT_PATTERN = re.compile(r"\blimit\s+(\d+)\b", flags=re.IGNORECASE)


class SQLExecutionSettings(BaseSettings):
    """Environment-backed SQL execution limits."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    default_limit: int = Field(default=100, alias="SQL_EXECUTION_DEFAULT_LIMIT")
    max_limit: int = Field(default=1000, alias="SQL_EXECUTION_MAX_LIMIT")


class SQLSafetyError(ValueError):
    """Raised when SQL is unsafe or outside the approved schema."""


@dataclass(frozen=True)
class SQLExecutionResult:
    """Executed SQL text and materialized rows."""

    sql: str
    rows: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        """Return the number of rows materialized from the query."""
        return len(self.rows)


def validate_readonly_select(sql: str) -> exp.Select:
    """Parse and validate a single approved read-only SELECT statement."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError as exc:
        raise SQLSafetyError("SQL must parse as PostgreSQL") from exc

    if len(statements) != 1:
        raise SQLSafetyError("SQL must contain exactly one statement")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SQLSafetyError("Only read-only SELECT statements are allowed")

    for table in statement.find_all(exp.Table):
        table_name = table.name
        schema_name = table.db
        if schema_name and schema_name != APPROVED_SCHEMA:
            raise SQLSafetyError(f"Schema is not approved for SQL execution: {schema_name}")
        if table_name not in APPROVED_TABLE_NAMES:
            raise SQLSafetyError(f"Table is not approved for SQL execution: {table_name}")
    return statement


def apply_row_limit(sql: str, settings: SQLExecutionSettings | None = None) -> str:
    """Ensure SQL has a bounded LIMIT clause."""
    settings = settings or SQLExecutionSettings()
    stripped_sql = sql.strip().rstrip(";")
    match = LIMIT_PATTERN.search(stripped_sql)
    if match is None:
        return f"{stripped_sql} LIMIT {settings.default_limit}"

    requested_limit = int(match.group(1))
    applied_limit = min(requested_limit, settings.max_limit)
    if applied_limit == requested_limit:
        return stripped_sql
    return LIMIT_PATTERN.sub(f"LIMIT {applied_limit}", stripped_sql, count=1)


def execute_readonly_sql(
    session: Session,
    sql: str,
    settings: SQLExecutionSettings | None = None,
) -> SQLExecutionResult:
    """Validate, limit, execute, and materialize a read-only SQL statement."""
    validate_readonly_select(sql)
    bounded_sql = apply_row_limit(sql, settings=settings)
    rows = [
        _serialize_mapping(row)
        for row in session.execute(text(bounded_sql)).mappings().all()
    ]
    return SQLExecutionResult(sql=bounded_sql, rows=rows)


def _serialize_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert SQLAlchemy row values into JSON-safe scalars."""
    return {key: _serialize_value(value) for key, value in dict(row).items()}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
