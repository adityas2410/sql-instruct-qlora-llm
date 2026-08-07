"""Evaluation report loading, metric flattening, and chart generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationReport:
    """Structured evaluation report loaded from JSON."""

    sql_generation: dict[str, Any]
    semantic_retrieval: dict[str, Any]
    agent: dict[str, Any]
    training_curves: dict[str, list[float]]


def load_evaluation_report(path: Path) -> EvaluationReport:
    """Load an evaluation report JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationReport(
        sql_generation=dict(payload["sql_generation"]),
        semantic_retrieval=dict(payload["semantic_retrieval"]),
        agent=dict(payload["agent"]),
        training_curves={
            key: [float(value) for value in values]
            for key, values in payload.get("training_curves", {}).items()
        },
    )


def report_to_metrics(report: EvaluationReport) -> dict[str, float | int]:
    """Flatten report values into W&B-friendly scalar metrics."""
    metrics: dict[str, float | int] = {}
    for section_name, section in (
        ("sql_eval", report.sql_generation),
        ("semantic_eval", report.semantic_retrieval),
        ("agent_eval", report.agent),
    ):
        for key, value in section.items():
            if isinstance(value, (int, float)):
                metrics[f"{section_name}/{key}"] = value
    return metrics


def write_report_markdown(report: EvaluationReport, output_path: Path) -> None:
    """Write a compact markdown report from evaluation metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(report), encoding="utf-8")


def render_evaluation_charts(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    """Render PNG charts for README and report assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sql": output_dir / "sql_eval_metrics.png",
        "semantic": output_dir / "semantic_retrieval_metrics.png",
        "agent": output_dir / "agent_eval_metrics.png",
        "training": output_dir / "training_curves.png",
    }
    _render_sql_chart(report.sql_generation, paths["sql"])
    _render_semantic_chart(report.semantic_retrieval, paths["semantic"])
    _render_agent_chart(report.agent, paths["agent"])
    _render_training_chart(report.training_curves, paths["training"])
    return paths


def _render_sql_chart(metrics: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Parse", "Safety", "Exact", "Tables", "Columns", "Execution"]
    values = [
        metrics["parse_validity"],
        metrics["readonly_safety"],
        metrics["exact_match"],
        metrics["table_match"],
        metrics["column_match"],
        metrics["execution_match"],
    ]
    _bar_chart(labels, values, "SQL Generation Evaluation", output_path, color="#2563eb")


def _render_semantic_chart(metrics: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    k_values = [1, 3, 5, 10]
    recall = [metrics[f"recall_at_{k}"] for k in k_values]
    precision = [metrics[f"precision_at_{k}"] for k in k_values]
    mrr = [metrics["mrr"] for _ in k_values]
    plt.figure(figsize=(8, 4.8))
    plt.plot(k_values, recall, marker="o", linewidth=2.5, label="Recall@K")
    plt.plot(k_values, precision, marker="o", linewidth=2.5, label="Precision@K")
    plt.plot(k_values, mrr, marker="o", linewidth=2.5, label="MRR")
    plt.title("Semantic Retrieval Evaluation")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _render_agent_chart(metrics: dict[str, Any], output_path: Path) -> None:
    labels = ["Route", "Tools", "Grounded", "Rejection", "Citations"]
    values = [
        metrics["route_accuracy"],
        metrics["tool_correctness"],
        metrics["grounded_answer_rate"],
        metrics["unsupported_conclusion_rejection_rate"],
        metrics["evidence_citation_coverage"],
    ]
    _bar_chart(labels, values, "Agent Evaluation", output_path, color="#059669")


def _render_training_chart(curves: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    for label, values in curves.items():
        x_values = list(range(1, len(values) + 1))
        plt.plot(x_values, values, marker="o", linewidth=2.5, label=label.replace("_", " ").title())
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _bar_chart(labels: list[str], values: list[float], title: str, output_path: Path, color: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.02),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _render_markdown(report: EvaluationReport) -> str:
    return "\n".join(
        [
            "# Evaluation Report",
            "",
            "## SQL Generation Evaluation",
            _metric_table(report.sql_generation),
            "",
            "## Semantic Retrieval Evaluation",
            _metric_table(report.semantic_retrieval),
            "",
            "## Agent Evaluation",
            _metric_table(report.agent),
            "",
        ]
    )


def _metric_table(metrics: dict[str, Any]) -> str:
    rows = ["| Metric | Score |", "|---|---:|"]
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append(f"| {key.replace('_', ' ').title()} | {value:.3f} |")
    return "\n".join(rows)
