"""Approved insurance database schema metadata.

This module is the application-level allowlist for future SQL generation and
validation. It intentionally contains only business tables and columns.
"""

from __future__ import annotations

from dataclasses import dataclass

CLAIM_VECTOR_DIMENSION = 128
APPROVED_SCHEMA = "public"


@dataclass(frozen=True)
class TableDefinition:
    """Describe an approved table and its public queryable columns."""

    name: str
    columns: tuple[str, ...]
    description: str


APPROVED_TABLES: dict[str, TableDefinition] = {
    "customers": TableDefinition(
        name="customers",
        description="Insurance customers and contact information.",
        columns=(
            "customer_id",
            "full_name",
            "date_of_birth",
            "occupation",
            "annual_income",
            "address",
            "city",
            "postcode",
            "phone_number",
            "email",
            "account_created_at",
        ),
    ),
    "policies": TableDefinition(
        name="policies",
        description="Customer insurance policies and coverage details.",
        columns=(
            "policy_id",
            "customer_id",
            "policy_type",
            "coverage_amount",
            "premium_amount",
            "deductible",
            "policy_start_date",
            "policy_end_date",
            "policy_status",
        ),
    ),
    "vehicles": TableDefinition(
        name="vehicles",
        description="Insured vehicles attached to customer policies and claims.",
        columns=(
            "vehicle_id",
            "customer_id",
            "make",
            "model",
            "manufacture_year",
            "vehicle_type",
            "estimated_value",
            "registration_region",
        ),
    ),
    "repair_shops": TableDefinition(
        name="repair_shops",
        description="Repair shops associated with vehicle claims and payments.",
        columns=(
            "repair_shop_id",
            "name",
            "city",
            "postcode",
            "owner_name",
            "bank_account_reference",
            "registration_date",
        ),
    ),
    "incidents": TableDefinition(
        name="incidents",
        description="Claim incidents, locations, dates, and report metadata.",
        columns=(
            "incident_id",
            "incident_type",
            "incident_date",
            "incident_city",
            "incident_address",
            "weather_condition",
            "police_report_reference",
            "witness_count",
        ),
    ),
    "claims": TableDefinition(
        name="claims",
        description="Insurance claims, outcomes, and stored claim vectors.",
        columns=(
            "claim_id",
            "policy_id",
            "vehicle_id",
            "incident_id",
            "repair_shop_id",
            "claim_date",
            "claim_amount",
            "repair_cost",
            "damage_type",
            "injury_reported",
            "police_report_available",
            "claim_description",
            "claim_status",
            "historical_fraud_label",
            "investigation_outcome",
            "claim_embedding",
        ),
    ),
    "claim_participants": TableDefinition(
        name="claim_participants",
        description="People or organizations linked to claims.",
        columns=(
            "participant_id",
            "claim_id",
            "participant_type",
            "full_name",
            "phone_number",
            "email",
            "address",
            "relationship_to_claim",
        ),
    ),
    "payments": TableDefinition(
        name="payments",
        description="Claim payments and recipient bank-account references.",
        columns=(
            "payment_id",
            "claim_id",
            "recipient_type",
            "recipient_id",
            "bank_account_reference",
            "payment_amount",
            "payment_date",
            "payment_status",
        ),
    ),
}

APPROVED_TABLE_NAMES = frozenset(APPROVED_TABLES)
APPROVED_COLUMNS_BY_TABLE = {
    table_name: frozenset(definition.columns)
    for table_name, definition in APPROVED_TABLES.items()
}


def render_schema_for_prompt() -> str:
    """Render the approved schema in a compact format for SQL generation prompts."""
    table_lines = []
    for table_name, definition in APPROVED_TABLES.items():
        columns = ", ".join(definition.columns)
        table_lines.append(f"{table_name}({columns})")
    return "\n".join(table_lines)
