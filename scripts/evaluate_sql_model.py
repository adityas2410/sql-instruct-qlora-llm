"""Evaluate Falcon SQL generation against JSONL instruction examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sql_model.config import FalconSQLSettings
from sql_model.dataset import load_instruction_jsonl
from sql_model.evaluation import SQLPrediction, evaluate_sql_predictions
from sql_model.inference import FalconSQLGenerator
from sql_model.wandb_logging import WandbLogger, WandbRunSettings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=Path("data/generated/sql_instruction_eval.jsonl"),
        help="Evaluation JSONL path.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("artifacts/sql_model/sql_eval_predictions.json"),
        help="Path for evaluation row output.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate SQL for eval examples and compute static SQL metrics."""
    args = parse_args()
    settings = FalconSQLSettings()
    examples = load_instruction_jsonl(args.eval_jsonl)
    generator = FalconSQLGenerator(settings=settings)
    predictions = [
        SQLPrediction(
            instruction=example.instruction,
            expected_sql=example.output,
            generated_sql=generator.generate_sql(example.instruction, schema=example.schema).sql,
        )
        for example in examples
    ]
    result = evaluate_sql_predictions(predictions)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(json.dumps(result.rows, indent=2) + "\n", encoding="utf-8")

    logger = WandbLogger(
        WandbRunSettings(
            enabled=settings.use_wandb,
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            run_name="falcon-sql-evaluation",
            tags=("falcon-11b", "sql-eval"),
        )
    )
    logger.start(config={"eval_jsonl": str(args.eval_jsonl)})
    try:
        logger.log_metrics(result.to_metrics())
        logger.log_prediction_table(result.rows, table_name="sql_eval_predictions")
    finally:
        logger.finish()

    print(json.dumps(result.to_metrics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
