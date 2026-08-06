"""Dataset utilities for Falcon SQL instruction tuning."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from database.schema import render_schema_for_prompt
from sql_model.prompts import render_training_text


@dataclass(frozen=True)
class SQLInstructionExample:
    """One supervised instruction-tuning example for SQL generation."""

    instruction: str
    schema: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        """Serialize the example to the JSONL record shape."""
        return asdict(self)

    def to_training_text(self) -> str:
        """Render the text consumed by supervised fine-tuning."""
        return render_training_text(
            instruction=self.instruction,
            schema=self.schema,
            sql=self.output,
        )


def validate_instruction_record(record: dict[str, Any]) -> SQLInstructionExample:
    """Validate and normalize an instruction JSON record."""
    missing = {"instruction", "schema", "output"} - set(record)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"SQL instruction record is missing required fields: {missing_list}")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("SQL instruction metadata must be an object when provided")
    return SQLInstructionExample(
        instruction=str(record["instruction"]).strip(),
        schema=str(record["schema"]).strip(),
        output=str(record["output"]).strip(),
        metadata=metadata,
    )


def load_instruction_jsonl(path: Path) -> list[SQLInstructionExample]:
    """Load SQL instruction examples from a JSONL file."""
    examples: list[SQLInstructionExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            examples.append(validate_instruction_record(record))
        except Exception as exc:
            raise ValueError(f"Invalid SQL instruction record at {path}:{line_number}") from exc
    return examples


def write_instruction_jsonl(path: Path, examples: Iterable[SQLInstructionExample]) -> None:
    """Write SQL instruction examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(example.to_json_record(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_huggingface_dataset(examples: Sequence[SQLInstructionExample]):
    """Convert examples to a Hugging Face Dataset.

    Importing datasets lazily keeps lightweight tooling usable without requiring
    the training dependencies at import time.
    """
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "instruction": example.instruction,
                "schema": example.schema,
                "output": example.output,
                "text": example.to_training_text(),
                "metadata": example.metadata,
            }
            for example in examples
        ]
    )


def build_insurance_sql_instruction_examples(schema: str | None = None) -> list[SQLInstructionExample]:
    """Build deterministic SQL instruction examples for the insurance schema."""
    approved_schema = schema or render_schema_for_prompt()
    examples = [
        SQLInstructionExample(
            instruction="Show vehicle-theft claims in London above 20000.",
            schema=approved_schema,
            output=(
                "SELECT c.claim_id, c.claim_date, c.claim_amount, i.incident_city, "
                "i.incident_type FROM claims c JOIN incidents i ON c.incident_id = i.incident_id "
                "WHERE i.incident_type = 'vehicle_theft' AND i.incident_city = 'London' "
                "AND c.claim_amount > 20000 ORDER BY c.claim_amount DESC LIMIT 100"
            ),
            metadata={"query_type": "exact_sql", "tables": ["claims", "incidents"]},
        ),
        SQLInstructionExample(
            instruction="List claims paid to the same bank account BANK-SHARED-001.",
            schema=approved_schema,
            output=(
                "SELECT c.claim_id, p.payment_id, p.bank_account_reference, p.payment_amount "
                "FROM claims c JOIN payments p ON c.claim_id = p.claim_id "
                "WHERE p.bank_account_reference = 'BANK-SHARED-001' "
                "ORDER BY p.payment_date DESC LIMIT 100"
            ),
            metadata={"query_type": "entity_link", "tables": ["claims", "payments"]},
        ),
        SQLInstructionExample(
            instruction="Find claims filed within 30 days of policy start.",
            schema=approved_schema,
            output=(
                "SELECT c.claim_id, c.claim_date, p.policy_start_date, c.claim_amount "
                "FROM claims c JOIN policies p ON c.policy_id = p.policy_id "
                "WHERE c.claim_date <= p.policy_start_date + INTERVAL '30 days' "
                "ORDER BY c.claim_date DESC LIMIT 100"
            ),
            metadata={"query_type": "temporal", "tables": ["claims", "policies"]},
        ),
        SQLInstructionExample(
            instruction="Show repair shops connected to more than five claims.",
            schema=approved_schema,
            output=(
                "SELECT rs.repair_shop_id, rs.name, rs.city, COUNT(c.claim_id) AS claim_count "
                "FROM repair_shops rs JOIN claims c ON rs.repair_shop_id = c.repair_shop_id "
                "GROUP BY rs.repair_shop_id, rs.name, rs.city HAVING COUNT(c.claim_id) > 5 "
                "ORDER BY claim_count DESC LIMIT 100"
            ),
            metadata={"query_type": "aggregation", "tables": ["repair_shops", "claims"]},
        ),
        SQLInstructionExample(
            instruction="Show historically fraudulent claims above 15000 with customer names.",
            schema=approved_schema,
            output=(
                "SELECT c.claim_id, cu.full_name, c.claim_amount, c.investigation_outcome "
                "FROM claims c JOIN policies p ON c.policy_id = p.policy_id "
                "JOIN customers cu ON p.customer_id = cu.customer_id "
                "WHERE c.historical_fraud_label = TRUE AND c.claim_amount > 15000 "
                "ORDER BY c.claim_amount DESC LIMIT 100"
            ),
            metadata={"query_type": "outcome_metadata", "tables": ["claims", "policies", "customers"]},
        ),
        SQLInstructionExample(
            instruction="Find participants who share a phone number across multiple claims.",
            schema=approved_schema,
            output=(
                "SELECT phone_number, COUNT(DISTINCT claim_id) AS claim_count "
                "FROM claim_participants WHERE phone_number IS NOT NULL "
                "GROUP BY phone_number HAVING COUNT(DISTINCT claim_id) > 1 "
                "ORDER BY claim_count DESC LIMIT 100"
            ),
            metadata={"query_type": "shared_identifier", "tables": ["claim_participants"]},
        ),
        SQLInstructionExample(
            instruction="Show claims with repair costs greater than 80 percent of the claim amount.",
            schema=approved_schema,
            output=(
                "SELECT claim_id, claim_amount, repair_cost, damage_type FROM claims "
                "WHERE repair_cost IS NOT NULL AND repair_cost > claim_amount * 0.8 "
                "ORDER BY repair_cost DESC LIMIT 100"
            ),
            metadata={"query_type": "amount_ratio", "tables": ["claims"]},
        ),
        SQLInstructionExample(
            instruction="Find London collision claims with injury reported.",
            schema=approved_schema,
            output=(
                "SELECT c.claim_id, c.claim_amount, i.incident_city, i.incident_type "
                "FROM claims c JOIN incidents i ON c.incident_id = i.incident_id "
                "WHERE i.incident_city = 'London' AND i.incident_type = 'collision' "
                "AND c.injury_reported = TRUE ORDER BY c.claim_date DESC LIMIT 100"
            ),
            metadata={"query_type": "exact_sql", "tables": ["claims", "incidents"]},
        ),
    ]
    return examples


def split_examples(
    examples: Sequence[SQLInstructionExample],
    eval_fraction: float = 0.2,
) -> tuple[list[SQLInstructionExample], list[SQLInstructionExample]]:
    """Split deterministic examples into train and eval partitions."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    eval_count = max(1, int(len(examples) * eval_fraction))
    return list(examples[:-eval_count]), list(examples[-eval_count:])
