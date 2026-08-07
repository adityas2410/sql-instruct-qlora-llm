"""Unit tests for read-only SQL validation and limit handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.sql_execution import SQLSafetyError, apply_row_limit, validate_readonly_select


def test_readonly_select_is_allowed() -> None:
    """Approved SELECT statements pass validation."""
    statement = validate_readonly_select("SELECT claim_id FROM claims LIMIT 10")
    assert statement is not None


def test_write_ddl_and_multi_statement_sql_are_rejected() -> None:
    """Unsafe SQL forms are blocked before execution."""
    with pytest.raises(SQLSafetyError):
        validate_readonly_select("DELETE FROM claims")
    with pytest.raises(SQLSafetyError):
        validate_readonly_select("DROP TABLE claims")
    with pytest.raises(SQLSafetyError):
        validate_readonly_select("SELECT claim_id FROM claims; SELECT customer_id FROM customers")


def test_unapproved_tables_and_schemas_are_rejected() -> None:
    """SQL execution stays inside the approved application schema."""
    with pytest.raises(SQLSafetyError):
        validate_readonly_select("SELECT * FROM pg_catalog.pg_tables")
    with pytest.raises(SQLSafetyError):
        validate_readonly_select("SELECT * FROM unknown_table")


def test_sql_limits_are_added_and_clamped() -> None:
    """Execution SQL always has a bounded row limit."""
    settings = SimpleNamespace(default_limit=25, max_limit=100)
    assert apply_row_limit("SELECT claim_id FROM claims", settings=settings).endswith("LIMIT 25")
    assert apply_row_limit("SELECT claim_id FROM claims LIMIT 500", settings=settings).endswith(
        "LIMIT 100"
    )
    assert apply_row_limit("SELECT claim_id FROM claims LIMIT 50", settings=settings).endswith(
        "LIMIT 50"
    )
