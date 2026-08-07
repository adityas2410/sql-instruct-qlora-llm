"""Agent route, tool-use, and groundedness evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentEvaluationCase:
    """One agent evaluation case with observed behavior."""

    prompt_id: str
    prompt_type: str
    expected_route: str
    actual_route: str
    expected_tools: Sequence[str]
    actual_tools: Sequence[str]
    grounded_answer: bool
    rejected_unsupported_conclusion: bool
    cited_claim_ids: set[str] = field(default_factory=set)
    evidence_claim_ids: set[str] = field(default_factory=set)
    cited_entities: set[str] = field(default_factory=set)
    evidence_entities: set[str] = field(default_factory=set)


def evaluate_agent_cases(cases: Iterable[AgentEvaluationCase]) -> dict[str, Any]:
    """Aggregate agent evaluation cases into scalar metrics and rows."""
    case_list = list(cases)
    rows = [_case_row(case) for case in case_list]
    metrics = {
        "agent_eval/total": len(case_list),
        "agent_eval/route_accuracy": _mean(row["route_correct"] for row in rows),
        "agent_eval/tool_correctness": _mean(row["tools_correct"] for row in rows),
        "agent_eval/grounded_answer_rate": _mean(row["grounded_answer"] for row in rows),
        "agent_eval/unsupported_conclusion_rejection_rate": _mean(
            row["rejected_unsupported_conclusion"] for row in rows
        ),
        "agent_eval/evidence_citation_coverage": _mean(
            row["evidence_citation_coverage"] for row in rows
        ),
    }
    return {"metrics": metrics, "rows": rows}


def _case_row(case: AgentEvaluationCase) -> dict[str, Any]:
    return {
        "prompt_id": case.prompt_id,
        "prompt_type": case.prompt_type,
        "expected_route": case.expected_route,
        "actual_route": case.actual_route,
        "route_correct": case.expected_route == case.actual_route,
        "expected_tools": list(case.expected_tools),
        "actual_tools": list(case.actual_tools),
        "tools_correct": list(case.expected_tools) == list(case.actual_tools),
        "grounded_answer": case.grounded_answer,
        "rejected_unsupported_conclusion": case.rejected_unsupported_conclusion,
        "evidence_citation_coverage": _citation_coverage(case),
    }


def _citation_coverage(case: AgentEvaluationCase) -> float:
    evidence_items = case.evidence_claim_ids.union(case.evidence_entities)
    cited_items = case.cited_claim_ids.union(case.cited_entities)
    if not evidence_items:
        return 0.0
    return len(evidence_items.intersection(cited_items)) / len(evidence_items)


def _mean(values: Iterable[float | bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
