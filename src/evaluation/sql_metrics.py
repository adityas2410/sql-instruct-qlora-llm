"""SQL generation and execution-result evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sql_model.evaluation import SQLPrediction, evaluate_sql_predictions


@dataclass(frozen=True)
class SQLExecutionEvaluationCase:
    """Generated SQL with optional execution-result comparison data."""

    instruction: str
    expected_sql: str
    generated_sql: str
    expected_rows: Sequence[Mapping[str, Any]] | None = None
    generated_rows: Sequence[Mapping[str, Any]] | None = None


def evaluate_sql_execution_cases(cases: Iterable[SQLExecutionEvaluationCase]) -> dict[str, Any]:
    """Evaluate static SQL quality and execution-result agreement."""
    case_list = list(cases)
    static_result = evaluate_sql_predictions(
        SQLPrediction(
            instruction=case.instruction,
            expected_sql=case.expected_sql,
            generated_sql=case.generated_sql,
        )
        for case in case_list
    )
    execution_rows = [_execution_match_row(case) for case in case_list]
    comparable_rows = [row for row in execution_rows if row["execution_comparable"]]
    execution_match_rate = (
        sum(1 for row in comparable_rows if row["execution_match"]) / len(comparable_rows)
        if comparable_rows
        else 0.0
    )
    metrics = dict(static_result.to_metrics())
    metrics.update(
        {
            "sql_eval/execution_match_rate": execution_match_rate,
            "sql_eval/execution_comparable": len(comparable_rows),
        }
    )
    return {
        "metrics": metrics,
        "rows": [dict(row, **execution_rows[index]) for index, row in enumerate(static_result.rows)],
    }


def _execution_match_row(case: SQLExecutionEvaluationCase) -> dict[str, Any]:
    expected = _normalize_rows(case.expected_rows)
    generated = _normalize_rows(case.generated_rows)
    comparable = case.expected_rows is not None and case.generated_rows is not None
    return {
        "execution_comparable": comparable,
        "execution_match": comparable and expected == generated,
    }


def _normalize_rows(rows: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if rows is None:
        return []
    normalized = [{str(key): _json_safe(value) for key, value in row.items()} for row in rows]
    return sorted(normalized, key=lambda row: repr(sorted(row.items())))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
