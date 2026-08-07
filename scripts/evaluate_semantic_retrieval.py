"""Evaluate semantic retrieval output from saved evaluation cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.semantic_metrics import SemanticRetrievalCase, evaluate_semantic_cases
from sql_model.wandb_logging import WandbLogger, WandbRunSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("docs/evaluation/evaluation_summary.json"),
        help="Evaluation summary JSON containing semantic_cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/evaluation/semantic_retrieval_eval.json"),
        help="Path for semantic retrieval evaluation output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    cases = [
        SemanticRetrievalCase(
            query_claim_id=item["query_claim_id"],
            relevant_claim_ids=set(item["relevant_claim_ids"]),
            retrieved_claim_ids=list(item["retrieved_claim_ids"]),
            explanation_tokens=list(item.get("explanation_tokens", [])),
            explanation_entities=list(item.get("explanation_entities", [])),
        )
        for item in payload.get("semantic_cases", [])
    ]
    result = evaluate_semantic_cases(cases)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger = WandbLogger(
        WandbRunSettings(
            enabled=os.getenv("SQL_MODEL_USE_WANDB", "false").lower() == "true",
            project=os.getenv("WANDB_PROJECT", "insurance-sql-agent"),
            entity=os.getenv("WANDB_ENTITY") or None,
            run_name="semantic-retrieval-evaluation",
            tags=("semantic-retrieval", "pgvector", "evaluation"),
        )
    )
    logger.start(config={"summary_json": str(args.summary_json)})
    try:
        logger.log_metrics(result["metrics"])
        logger.log_prediction_table(result["rows"], table_name="semantic_retrieval_eval")
    finally:
        logger.finish()

    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
