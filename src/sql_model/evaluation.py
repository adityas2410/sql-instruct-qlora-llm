"""Evaluation helpers for generated SQL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class SQLPrediction:
    """Expected and generated SQL for one evaluation case."""

    instruction: str
    expected_sql: str
    generated_sql: str


@dataclass(frozen=True)
class SQLEvaluationResult:
    """Aggregate SQL evaluation metrics."""

    total: int
    syntax_validity: float
    exact_match: float
    table_selection_accuracy: float
    column_selection_accuracy: float
    unsafe_query_rejection_rate: float
    rows: list[dict[str, Any]]

    def to_metrics(self) -> dict[str, float | int]:
        """Return scalar metrics suitable for W&B logging."""
        return {
            "sql_eval/total": self.total,
            "sql_eval/syntax_validity": self.syntax_validity,
            "sql_eval/exact_match": self.exact_match,
            "sql_eval/table_selection_accuracy": self.table_selection_accuracy,
            "sql_eval/column_selection_accuracy": self.column_selection_accuracy,
            "sql_eval/unsafe_query_rejection_rate": self.unsafe_query_rejection_rate,
        }


def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact string comparison."""
    return " ".join(sql.strip().rstrip(";").lower().split())


def is_syntax_valid(sql: str) -> bool:
    """Return whether SQL parses as a single PostgreSQL statement."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return False
    return len(statements) == 1


def is_readonly_select(sql: str) -> bool:
    """Return whether SQL parses as a single read-only SELECT statement."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return False
    return len(statements) == 1 and isinstance(statements[0], exp.Select)


def extract_table_names(sql: str) -> set[str]:
    """Extract table names referenced by a SQL statement."""
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return set()
    return {table.name for table in parsed.find_all(exp.Table) if table.name}


def extract_column_names(sql: str) -> set[str]:
    """Extract column names referenced by a SQL statement."""
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return set()
    return {column.name for column in parsed.find_all(exp.Column) if column.name}


def evaluate_sql_predictions(predictions: Iterable[SQLPrediction]) -> SQLEvaluationResult:
    """Evaluate generated SQL against expected SQL strings and parsed structures."""
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        expected_tables = extract_table_names(prediction.expected_sql)
        generated_tables = extract_table_names(prediction.generated_sql)
        expected_columns = extract_column_names(prediction.expected_sql)
        generated_columns = extract_column_names(prediction.generated_sql)
        row = {
            "instruction": prediction.instruction,
            "expected_sql": prediction.expected_sql,
            "generated_sql": prediction.generated_sql,
            "syntax_valid": is_syntax_valid(prediction.generated_sql),
            "readonly_select": is_readonly_select(prediction.generated_sql),
            "exact_match": normalize_sql(prediction.expected_sql)
            == normalize_sql(prediction.generated_sql),
            "table_match": expected_tables == generated_tables,
            "column_match": expected_columns == generated_columns,
        }
        rows.append(row)

    total = len(rows)
    if total == 0:
        return SQLEvaluationResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    return SQLEvaluationResult(
        total=total,
        syntax_validity=_mean(row["syntax_valid"] for row in rows),
        exact_match=_mean(row["exact_match"] for row in rows),
        table_selection_accuracy=_mean(row["table_match"] for row in rows),
        column_selection_accuracy=_mean(row["column_match"] for row in rows),
        unsafe_query_rejection_rate=_mean(not row["readonly_select"] for row in rows),
        rows=rows,
    )


def _mean(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for item in items if item) / len(items) if items else 0.0
