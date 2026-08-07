"""Unit tests for SQL evaluation helpers."""

from __future__ import annotations

from sql_model.evaluation import (
    SQLPrediction,
    evaluate_sql_predictions,
    extract_column_names,
    extract_table_names,
    is_readonly_select,
    is_syntax_valid,
    normalize_sql,
)


def test_normalize_sql_ignores_case_spacing_and_semicolon() -> None:
    """Normalized SQL supports stable exact-match checks."""
    assert normalize_sql(" SELECT  *  FROM claims; ") == "select * from claims"


def test_parse_helpers_extract_tables_and_columns() -> None:
    """Evaluation can compare selected tables and columns."""
    sql = "SELECT c.claim_id, c.claim_amount FROM claims c JOIN incidents i ON c.incident_id = i.incident_id"
    assert extract_table_names(sql) == {"claims", "incidents"}
    assert {"claim_id", "claim_amount", "incident_id"}.issubset(extract_column_names(sql))


def test_readonly_select_detection_blocks_write_sql() -> None:
    """Generated write statements are counted as unsafe SQL."""
    assert is_syntax_valid("SELECT claim_id FROM claims")
    assert is_readonly_select("SELECT claim_id FROM claims")
    assert not is_readonly_select("DELETE FROM claims")


def test_evaluate_sql_predictions_returns_metrics() -> None:
    """Aggregate metrics are computed from static generated SQL strings."""
    result = evaluate_sql_predictions(
        [
            SQLPrediction(
                instruction="Show claims.",
                expected_sql="SELECT claim_id FROM claims LIMIT 100",
                generated_sql="SELECT claim_id FROM claims LIMIT 100",
            ),
            SQLPrediction(
                instruction="Delete claims.",
                expected_sql="SELECT claim_id FROM claims LIMIT 100",
                generated_sql="DELETE FROM claims",
            ),
        ]
    )
    metrics = result.to_metrics()
    assert metrics["sql_eval/total"] == 2
    assert metrics["sql_eval/exact_match"] == 0.5
    assert metrics["sql_eval/unsafe_sql_rate"] == 0.5
