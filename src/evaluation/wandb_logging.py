"""Weights & Biases logging for evaluation reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sql_model.wandb_logging import WandbLogger, WandbRunSettings

from .reporting import EvaluationReport, report_to_metrics


def log_evaluation_report_to_wandb(
    report: EvaluationReport,
    *,
    enabled: bool,
    project: str,
    entity: str | None = None,
    run_name: str = "insurance-agent-evaluation",
    config: Mapping[str, Any] | None = None,
) -> None:
    """Log evaluation scalar metrics to W&B when enabled."""
    logger = WandbLogger(
        WandbRunSettings(
            enabled=enabled,
            project=project,
            entity=entity,
            run_name=run_name,
            tags=("evaluation", "insurance-agent"),
        )
    )
    logger.start(config=config)
    try:
        logger.log_metrics(report_to_metrics(report))
    finally:
        logger.finish()
