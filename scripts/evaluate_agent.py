"""Evaluate agent routing, tool use, and grounded-response behavior."""

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

from evaluation.agent_metrics import AgentEvaluationCase, evaluate_agent_cases
from sql_model.wandb_logging import WandbLogger, WandbRunSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("docs/evaluation/evaluation_summary.json"),
        help="Evaluation summary JSON containing agent_cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/evaluation/agent_eval.json"),
        help="Path for agent evaluation output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    cases = [
        AgentEvaluationCase(
            prompt_id=item["prompt_id"],
            prompt_type=item["prompt_type"],
            expected_route=item["expected_route"],
            actual_route=item["actual_route"],
            expected_tools=list(item["expected_tools"]),
            actual_tools=list(item["actual_tools"]),
            grounded_answer=bool(item["grounded_answer"]),
            rejected_unsupported_conclusion=bool(item["rejected_unsupported_conclusion"]),
            cited_claim_ids=set(item.get("cited_claim_ids", [])),
            evidence_claim_ids=set(item.get("evidence_claim_ids", [])),
            cited_entities=set(item.get("cited_entities", [])),
            evidence_entities=set(item.get("evidence_entities", [])),
        )
        for item in payload.get("agent_cases", [])
    ]
    result = evaluate_agent_cases(cases)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger = WandbLogger(
        WandbRunSettings(
            enabled=os.getenv("SQL_MODEL_USE_WANDB", "false").lower() == "true",
            project=os.getenv("WANDB_PROJECT", "insurance-sql-agent"),
            entity=os.getenv("WANDB_ENTITY") or None,
            run_name="agent-evaluation",
            tags=("agent", "smolagents", "evaluation"),
        )
    )
    logger.start(config={"summary_json": str(args.summary_json)})
    try:
        logger.log_metrics(result["metrics"])
        logger.log_prediction_table(result["rows"], table_name="agent_eval")
    finally:
        logger.finish()

    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
