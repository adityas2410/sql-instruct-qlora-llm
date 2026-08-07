"""Unit tests for evaluation metrics and report assets."""

from __future__ import annotations

from pathlib import Path

from evaluation.agent_metrics import AgentEvaluationCase, evaluate_agent_cases
from evaluation.reporting import load_evaluation_report, report_to_metrics
from evaluation.semantic_metrics import SemanticRetrievalCase, evaluate_semantic_cases
from evaluation.sql_metrics import SQLExecutionEvaluationCase, evaluate_sql_execution_cases


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sql_evaluation_tracks_static_and_execution_metrics() -> None:
    result = evaluate_sql_execution_cases(
        [
            SQLExecutionEvaluationCase(
                instruction="Show paid claims.",
                expected_sql="SELECT claim_id FROM claims WHERE claim_status = 'paid'",
                generated_sql="SELECT claim_id FROM claims WHERE claim_status = 'paid'",
                expected_rows=[{"claim_id": "CLM-0001"}],
                generated_rows=[{"claim_id": "CLM-0001"}],
            ),
            SQLExecutionEvaluationCase(
                instruction="Delete old claims.",
                expected_sql="SELECT claim_id FROM claims",
                generated_sql="DELETE FROM claims",
            ),
        ]
    )

    metrics = result["metrics"]
    assert metrics["sql_eval/total"] == 2
    assert metrics["sql_eval/execution_comparable"] == 1
    assert metrics["sql_eval/execution_match_rate"] == 1.0
    assert metrics["sql_eval/unsafe_sql_rate"] == 0.5


def test_semantic_metrics_compute_retrieval_at_k_and_block_fraud_reason_leakage() -> None:
    result = evaluate_semantic_cases(
        [
            SemanticRetrievalCase(
                query_claim_id="CLM-0001",
                relevant_claim_ids={"CLM-0002", "CLM-0003"},
                retrieved_claim_ids=["CLM-0002", "CLM-0009", "CLM-0003"],
                explanation_tokens=["repair_shop_id=rs_001"],
            ),
            SemanticRetrievalCase(
                query_claim_id="CLM-0010",
                relevant_claim_ids={"CLM-0011"},
                retrieved_claim_ids=["CLM-0019"],
                explanation_entities=["historical_fraud_label=true"],
            ),
        ],
        k_values=(1, 3),
    )

    metrics = result["metrics"]
    assert metrics["semantic_eval/recall_at_1"] == 0.25
    assert metrics["semantic_eval/precision_at_1"] == 0.5
    assert metrics["semantic_eval/fraud_reason_leak_rate"] == 0.5


def test_agent_metrics_score_routes_tools_grounding_and_citations() -> None:
    result = evaluate_agent_cases(
        [
            AgentEvaluationCase(
                prompt_id="AGT-001",
                prompt_type="combined",
                expected_route="combined",
                actual_route="combined",
                expected_tools=["find_similar_claims", "execute_sql"],
                actual_tools=["find_similar_claims", "execute_sql"],
                grounded_answer=True,
                rejected_unsupported_conclusion=True,
                cited_claim_ids={"CLM-0001"},
                evidence_claim_ids={"CLM-0001", "CLM-0002"},
                cited_entities={"repair_shop_id=rs_001"},
                evidence_entities={"repair_shop_id=rs_001"},
            )
        ]
    )

    metrics = result["metrics"]
    assert metrics["agent_eval/route_accuracy"] == 1.0
    assert metrics["agent_eval/tool_correctness"] == 1.0
    assert metrics["agent_eval/evidence_citation_coverage"] == 2 / 3


def test_evaluation_report_loads_and_flattens_metrics() -> None:
    report = load_evaluation_report(REPO_ROOT / "docs/evaluation/evaluation_summary.json")
    metrics = report_to_metrics(report)

    assert metrics["sql_eval/parse_validity"] == 0.964
    assert metrics["semantic_eval/recall_at_5"] == 0.823
    assert metrics["agent_eval/grounded_answer_rate"] == 0.935


def test_readme_referenced_evaluation_assets_exist() -> None:
    asset_paths = [
        "docs/assets/sql_eval_metrics.png",
        "docs/assets/semantic_retrieval_metrics.png",
        "docs/assets/agent_eval_metrics.png",
        "docs/assets/training_curves.png",
    ]

    for asset_path in asset_paths:
        assert (REPO_ROOT / asset_path).exists()
