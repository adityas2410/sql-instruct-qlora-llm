"""Prepare SQL instruction-tuning JSONL files for Falcon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sql_model.dataset import (
    build_insurance_sql_instruction_examples,
    split_examples,
    write_instruction_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/generated/sql_instruction_train.jsonl"),
        help="Path for training JSONL examples.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/generated/sql_instruction_eval.jsonl"),
        help="Path for evaluation JSONL examples.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Fraction of examples assigned to eval JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    """Build deterministic SQL instruction data files."""
    args = parse_args()
    examples = build_insurance_sql_instruction_examples()
    train_examples, eval_examples = split_examples(examples, eval_fraction=args.eval_fraction)
    write_instruction_jsonl(args.train_output, train_examples)
    write_instruction_jsonl(args.eval_output, eval_examples)
    print(
        f"Prepared Falcon SQL instruction data: "
        f"train={len(train_examples)} eval={len(eval_examples)}"
    )


if __name__ == "__main__":
    main()
