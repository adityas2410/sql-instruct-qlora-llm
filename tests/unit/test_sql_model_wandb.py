"""Unit tests for optional W&B logging wrapper."""

from __future__ import annotations

from sql_model.wandb_logging import WandbLogger, WandbRunSettings


def test_disabled_wandb_logger_is_noop() -> None:
    """Disabled W&B logger does not import or start W&B."""
    logger = WandbLogger(WandbRunSettings(enabled=False, project="insurance-sql-agent"))
    logger.start(config={"base_model_id": "tiiuae/falcon-11B"})
    logger.log_metrics({"sql_eval/exact_match": 1.0})
    logger.log_prediction_table([], table_name="predictions")
    logger.finish()
    assert not logger.enabled
