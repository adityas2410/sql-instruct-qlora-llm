"""Load generated insurance JSON records into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import delete

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.models import Claim, ClaimParticipant, Customer, Incident, Payment, Policy, RepairShop, Vehicle
from database.session import session_scope

TABLE_LOAD_ORDER = (
    "customers",
    "policies",
    "vehicles",
    "repair_shops",
    "incidents",
    "claims",
    "claim_participants",
    "payments",
)

MODEL_BY_TABLE = {
    "customers": Customer,
    "policies": Policy,
    "vehicles": Vehicle,
    "repair_shops": RepairShop,
    "incidents": Incident,
    "claims": Claim,
    "claim_participants": ClaimParticipant,
    "payments": Payment,
}

DATE_FIELDS = {
    "date_of_birth",
    "policy_start_date",
    "policy_end_date",
    "registration_date",
    "incident_date",
    "claim_date",
    "payment_date",
}
DATETIME_FIELDS = {"account_created_at"}
DECIMAL_FIELDS = {
    "annual_income",
    "coverage_amount",
    "premium_amount",
    "deductible",
    "estimated_value",
    "claim_amount",
    "repair_cost",
    "payment_amount",
}


def parse_args() -> argparse.Namespace:
    """Parse loader command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory containing table JSON files from generate_insurance_data.py.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing insurance records before loading the JSON files.",
    )
    return parser.parse_args()


def load_table_records(input_dir: Path, table_name: str) -> list[dict[str, Any]]:
    """Load one table JSON file from disk."""
    path = input_dir / f"{table_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated table file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_record(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce JSON scalar values into Python types expected by SQLAlchemy models."""
    coerced = dict(record)
    for field_name, value in list(coerced.items()):
        if value is None:
            continue
        if field_name in DATE_FIELDS:
            coerced[field_name] = date.fromisoformat(value)
        elif field_name in DATETIME_FIELDS:
            coerced[field_name] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif field_name in DECIMAL_FIELDS:
            coerced[field_name] = Decimal(str(value))
    return coerced


def delete_existing_records() -> None:
    """Delete existing records in reverse foreign-key order."""
    with session_scope() as session:
        for table_name in reversed(TABLE_LOAD_ORDER):
            session.execute(delete(MODEL_BY_TABLE[table_name]))


def load_records(input_dir: Path) -> dict[str, int]:
    """Load generated records into the database and return row counts."""
    counts: dict[str, int] = {}
    with session_scope() as session:
        for table_name in TABLE_LOAD_ORDER:
            model = MODEL_BY_TABLE[table_name]
            records = load_table_records(input_dir, table_name)
            session.add_all(model(**coerce_record(record)) for record in records)
            counts[table_name] = len(records)
    return counts


def main() -> None:
    """Load generated insurance records into PostgreSQL."""
    args = parse_args()
    if args.replace:
        delete_existing_records()
    counts = load_records(args.input_dir)
    table_counts = ", ".join(f"{name}={count}" for name, count in counts.items())
    print(f"Loaded synthetic insurance data from {args.input_dir}: {table_counts}")


if __name__ == "__main__":
    main()
