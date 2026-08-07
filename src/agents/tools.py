"""Project tool factories for Hugging Face smolagents."""

from __future__ import annotations

import json
from typing import Callable

from sqlalchemy.orm import Session

from database.repositories import ClaimRepository
from services.semantic_search import find_similar_claims
from services.sql_execution import execute_readonly_sql
from sql_model.inference import FalconSQLGenerator


def build_agent_tools(
    session: Session,
    sql_generator: FalconSQLGenerator | None = None,
) -> list[Callable]:
    """Build smolagents tools bound to one database session."""
    from smolagents import tool

    generator = sql_generator or FalconSQLGenerator()

    @tool
    def generate_sql(instruction: str) -> str:
        """
        Generate read-only PostgreSQL for an insurance investigation request.

        Args:
            instruction: Natural-language request describing the records to retrieve.
        """
        return generator.generate_sql(instruction).sql

    @tool
    def execute_sql(sql: str) -> str:
        """
        Execute one validated read-only SELECT statement and return JSON rows.

        Args:
            sql: A single PostgreSQL SELECT statement over the approved insurance schema.
        """
        result = execute_readonly_sql(session, sql)
        return json.dumps({"sql": result.sql, "rows": result.rows, "row_count": result.row_count})

    @tool
    def find_similar_claims_tool(claim_id: str, top_k: int = 10) -> str:
        """
        Find claims with vectors similar to a source claim and return evidence signals.

        Args:
            claim_id: Source claim ID whose indexed claim_embedding should be searched.
            top_k: Maximum number of similar claims to return.
        """
        result = find_similar_claims(session, claim_id, top_k=top_k)
        return json.dumps(
            {
                "source_claim_id": result.source_claim_id,
                "matches": [
                    {
                        "claim_id": match.claim_id,
                        "similarity_score": match.similarity_score,
                        "shared_tokens": match.shared_tokens,
                        "shared_entities": match.shared_entities,
                        "historical_fraud_label": (
                            match.evidence.claim.historical_fraud_label if match.evidence else None
                        ),
                        "investigation_outcome": (
                            match.evidence.claim.investigation_outcome if match.evidence else None
                        ),
                    }
                    for match in result.matches
                ],
            }
        )

    @tool
    def get_claim_evidence(claim_id: str) -> str:
        """
        Retrieve complete relational evidence metadata for one claim.

        Args:
            claim_id: Claim ID to retrieve.
        """
        evidence = ClaimRepository(session).get_claim_evidence(claim_id)
        if evidence is None:
            return json.dumps({"claim_id": claim_id, "found": False})
        return json.dumps(
            {
                "claim_id": evidence.claim.claim_id,
                "claim_amount": float(evidence.claim.claim_amount),
                "claim_status": evidence.claim.claim_status,
                "incident_type": evidence.incident.incident_type,
                "incident_city": evidence.incident.incident_city,
                "vehicle_type": evidence.vehicle.vehicle_type,
                "repair_shop_id": evidence.repair_shop.repair_shop_id if evidence.repair_shop else None,
                "historical_fraud_label": evidence.claim.historical_fraud_label,
                "investigation_outcome": evidence.claim.investigation_outcome,
            }
        )

    return [generate_sql, execute_sql, find_similar_claims_tool, get_claim_evidence]
