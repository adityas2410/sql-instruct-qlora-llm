"""Semantic retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

FRAUD_METADATA_FIELDS = {"historical_fraud_label", "investigation_outcome"}


@dataclass(frozen=True)
class SemanticRetrievalCase:
    """One semantic retrieval evaluation case."""

    query_claim_id: str
    relevant_claim_ids: set[str]
    retrieved_claim_ids: list[str]
    explanation_tokens: list[str] = field(default_factory=list)
    explanation_entities: list[str] = field(default_factory=list)


def evaluate_semantic_cases(
    cases: Iterable[SemanticRetrievalCase],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Compute retrieval and explanation metrics for semantic search output."""
    case_list = list(cases)
    metrics: dict[str, float | int] = {"semantic_eval/total": len(case_list)}
    for k in k_values:
        metrics[f"semantic_eval/recall_at_{k}"] = _mean(_recall_at_k(case, k) for case in case_list)
        metrics[f"semantic_eval/precision_at_{k}"] = _mean(
            _precision_at_k(case, k) for case in case_list
        )
    metrics["semantic_eval/mrr"] = _mean(_reciprocal_rank(case) for case in case_list)
    metrics["semantic_eval/explanation_coverage"] = _mean(
        _has_explanation(case) for case in case_list
    )
    metrics["semantic_eval/fraud_reason_leak_rate"] = _mean(
        _uses_fraud_metadata(case) for case in case_list
    )
    return {"metrics": metrics, "rows": [_case_row(case) for case in case_list]}


def _recall_at_k(case: SemanticRetrievalCase, k: int) -> float:
    if not case.relevant_claim_ids:
        return 0.0
    retrieved = set(case.retrieved_claim_ids[:k])
    return len(retrieved.intersection(case.relevant_claim_ids)) / len(case.relevant_claim_ids)


def _precision_at_k(case: SemanticRetrievalCase, k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = case.retrieved_claim_ids[:k]
    if not retrieved:
        return 0.0
    return len(set(retrieved).intersection(case.relevant_claim_ids)) / len(retrieved)


def _reciprocal_rank(case: SemanticRetrievalCase) -> float:
    for index, claim_id in enumerate(case.retrieved_claim_ids, start=1):
        if claim_id in case.relevant_claim_ids:
            return 1.0 / index
    return 0.0


def _has_explanation(case: SemanticRetrievalCase) -> bool:
    return bool(case.explanation_tokens or case.explanation_entities)


def _uses_fraud_metadata(case: SemanticRetrievalCase) -> bool:
    reasons = {reason.lower() for reason in case.explanation_tokens + case.explanation_entities}
    return any(field in reason for field in FRAUD_METADATA_FIELDS for reason in reasons)


def _case_row(case: SemanticRetrievalCase) -> dict[str, Any]:
    return {
        "query_claim_id": case.query_claim_id,
        "relevant_claim_ids": sorted(case.relevant_claim_ids),
        "retrieved_claim_ids": case.retrieved_claim_ids,
        "reciprocal_rank": _reciprocal_rank(case),
        "has_explanation": _has_explanation(case),
        "fraud_reason_leak": _uses_fraud_metadata(case),
    }


def _mean(values: Iterable[float | bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
