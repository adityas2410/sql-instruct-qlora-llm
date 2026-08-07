"""Build local evaluation report assets and markdown."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.reporting import load_evaluation_report, render_evaluation_charts, write_report_markdown
from evaluation.wandb_logging import log_evaluation_report_to_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("docs/evaluation/evaluation_summary.json"),
        help="Evaluation summary JSON.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Directory for PNG evaluation charts.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("artifacts/evaluation/evaluation_report.md"),
        help="Path for generated markdown report.",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log scalar report metrics to W&B when W&B environment variables are configured.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_evaluation_report(args.summary_json)
    chart_paths = render_evaluation_charts(report, args.asset_dir)
    write_report_markdown(report, args.markdown_output)

    if args.log_wandb:
        log_evaluation_report_to_wandb(
            report,
            enabled=os.getenv("SQL_MODEL_USE_WANDB", "false").lower() == "true",
            project=os.getenv("WANDB_PROJECT", "insurance-sql-agent"),
            entity=os.getenv("WANDB_ENTITY") or None,
            config={"summary_json": str(args.summary_json)},
        )

    for name, path in chart_paths.items():
        print(f"{name}: {path}")
    print(f"report: {args.markdown_output}")


if __name__ == "__main__":
    main()
