"""Train Falcon LoRA adapters for SQL generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sql_model.config import FalconSQLSettings
from sql_model.training import FalconSQLTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Training JSONL path. Defaults to SQL_TRAIN_JSONL_PATH.",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Evaluation JSONL path. Defaults to SQL_EVAL_JSONL_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Falcon QLoRA supervised fine-tuning."""
    args = parse_args()
    trainer = FalconSQLTrainer(settings=FalconSQLSettings())
    adapter_path = trainer.train(train_jsonl_path=args.train_jsonl, eval_jsonl_path=args.eval_jsonl)
    print(f"Saved Falcon SQL LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
