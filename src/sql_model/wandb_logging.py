"""Optional Weights & Biases logging for SQL model training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WandbRunSettings:
    """Settings required to initialize a W&B run."""

    enabled: bool
    project: str
    entity: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()


class WandbLogger:
    """Small adapter around W&B that is disabled by default."""

    def __init__(self, settings: WandbRunSettings) -> None:
        self.settings = settings
        self._run: Any | None = None

    @property
    def enabled(self) -> bool:
        """Return whether W&B logging is enabled."""
        return self.settings.enabled

    def start(self, config: Mapping[str, Any] | None = None) -> None:
        """Start a W&B run when logging is enabled."""
        if not self.enabled:
            return
        import wandb

        self._run = wandb.init(
            project=self.settings.project,
            entity=self.settings.entity or None,
            name=self.settings.run_name,
            tags=list(self.settings.tags),
            config=dict(config or {}),
        )

    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        """Log scalar metrics when W&B is enabled."""
        if not self.enabled:
            return
        import wandb

        wandb.log(dict(metrics), step=step)

    def log_prediction_table(self, rows: Sequence[Mapping[str, Any]], table_name: str) -> None:
        """Log generated SQL examples as a W&B table."""
        if not self.enabled or not rows:
            return
        import wandb

        columns = sorted({key for row in rows for key in row.keys()})
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*(row.get(column) for column in columns))
        wandb.log({table_name: table})

    def finish(self) -> None:
        """Finish the active W&B run when logging is enabled."""
        if not self.enabled:
            return
        import wandb

        wandb.finish()
        self._run = None
