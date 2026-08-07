"""Unit tests for SQL instruction dataset utilities."""

from __future__ import annotations

from sql_model.config import FALCON_SQL_BASE_MODEL_ID, FalconSQLSettings
from sql_model.dataset import (
    build_insurance_sql_instruction_examples,
    split_examples,
    validate_instruction_record,
)
from sql_model.prompts import render_sql_prompt, render_training_text


def test_falcon_model_id_is_fixed() -> None:
    """The SQL model stays pinned to the Falcon adapter lineage."""
    assert FALCON_SQL_BASE_MODEL_ID == "tiiuae/falcon-11B"
    assert FalconSQLSettings().base_model_id == FALCON_SQL_BASE_MODEL_ID


def test_instruction_record_validation() -> None:
    """Instruction records normalize into typed examples."""
    example = validate_instruction_record(
        {
            "instruction": "Show London claims.",
            "schema": "claims(...)",
            "output": "SELECT claim_id FROM claims LIMIT 100",
            "metadata": {"query_type": "exact_sql"},
        }
    )
    assert example.instruction == "Show London claims."
    assert example.metadata["query_type"] == "exact_sql"


def test_prompt_contains_schema_instruction_and_sql_marker() -> None:
    """Prompts include the approved schema and SQL-only instruction."""
    prompt = render_sql_prompt("Show claims.", "claims(claim_id)")
    assert "Approved schema:" in prompt
    assert "claims(claim_id)" in prompt
    assert "Return SQL only" in prompt
    assert prompt.endswith("SQL:\n")


def test_training_text_appends_expected_sql() -> None:
    """Training text contains the prompt followed by target SQL."""
    text = render_training_text("Show claims.", "claims(claim_id)", "SELECT claim_id FROM claims")
    assert text.endswith("SELECT claim_id FROM claims")


def test_build_and_split_examples() -> None:
    """Deterministic examples cover multiple insurance query patterns."""
    examples = build_insurance_sql_instruction_examples(schema="claims(...)")
    train, eval_examples = split_examples(examples, eval_fraction=0.25)
    assert len(examples) >= 6
    assert train
    assert eval_examples
    assert all(example.schema == "claims(...)" for example in examples)
