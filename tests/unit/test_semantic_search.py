"""Unit tests for semantic claim search service helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from database.models import Claim, ClaimParticipant, Customer, Incident, Payment, Policy, RepairShop, Vehicle
from database.repositories import ClaimEvidence
from services import semantic_search
from services.semantic_search import (
    ClaimEmbeddingMissingError,
    ClaimNotFoundError,
    InvalidClaimVectorError,
    NoIndexedClaimsError,
    SimilarClaim,
    attach_evidence_and_explanations,
    build_vector_search_sql,
    explain_similarity,
    find_similar_claims,
    resolve_top_k,
    search_similar_to_vector,
    validate_query_vector,
    vector_to_pgvector_literal,
)


def _claim_evidence(
    claim_id: str,
    historical_fraud_label: bool | None,
    investigation_outcome: str | None,
    repair_shop_id: str = "RS-1",
    bank_account_reference: str = "BANK-1",
    phone_number: str = "555-0101",
    address: str = "10 Main Street",
) -> ClaimEvidence:
    customer = Customer(
        customer_id=f"CUST-{claim_id}",
        full_name="Alex Reed",
        date_of_birth=date(1988, 5, 4),
        occupation="Engineer",
        annual_income=Decimal("82000.00"),
        address=address,
        city="London",
        postcode="L1 1AA",
        phone_number=phone_number,
        email=f"{claim_id.lower()}@example.com",
        account_created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    policy = Policy(
        policy_id=f"POL-{claim_id}",
        customer_id=customer.customer_id,
        policy_type="auto",
        coverage_amount=Decimal("50000.00"),
        premium_amount=Decimal("1200.00"),
        deductible=Decimal("500.00"),
        policy_start_date=date(2025, 1, 1),
        policy_end_date=date(2026, 1, 1),
        policy_status="active",
    )
    vehicle = Vehicle(
        vehicle_id=f"VEH-{claim_id}",
        customer_id=customer.customer_id,
        make="Toyota",
        model="Rav4",
        manufacture_year=2021,
        vehicle_type="SUV",
        estimated_value=Decimal("28000.00"),
        registration_region="Greater London",
    )
    incident = Incident(
        incident_id=f"INC-{claim_id}",
        incident_type="collision",
        incident_date=date(2025, 1, 10),
        incident_city="London",
        incident_address="A10",
        weather_condition="rain",
        police_report_reference="PR-1",
        witness_count=1,
    )
    repair_shop = RepairShop(
        repair_shop_id=repair_shop_id,
        name="Northside Repairs",
        city="London",
        postcode="L1 2BB",
        owner_name="Jamie Stone",
        bank_account_reference=bank_account_reference,
        registration_date=date(2020, 1, 1),
    )
    claim = Claim(
        claim_id=claim_id,
        policy_id=policy.policy_id,
        vehicle_id=vehicle.vehicle_id,
        incident_id=incident.incident_id,
        repair_shop_id=repair_shop.repair_shop_id,
        claim_date=date(2025, 1, 12),
        claim_amount=Decimal("12000.00"),
        repair_cost=Decimal("9000.00"),
        damage_type="front_end",
        injury_reported=False,
        police_report_available=True,
        claim_description="Front bumper damage",
        claim_status="open",
        historical_fraud_label=historical_fraud_label,
        investigation_outcome=investigation_outcome,
        claim_embedding=[0.1] * 128,
    )
    participant = ClaimParticipant(
        participant_id=f"PART-{claim_id}",
        claim_id=claim_id,
        participant_type="witness",
        full_name="Morgan Lee",
        phone_number=phone_number,
        email=None,
        address=address,
        relationship_to_claim="witness",
    )
    payment = Payment(
        payment_id=f"PAY-{claim_id}",
        claim_id=claim_id,
        recipient_type="repair_shop",
        recipient_id=repair_shop_id,
        bank_account_reference=bank_account_reference,
        payment_amount=Decimal("9000.00"),
        payment_date=date(2025, 1, 20),
        payment_status="paid",
    )
    return ClaimEvidence(
        claim=claim,
        policy=policy,
        customer=customer,
        vehicle=vehicle,
        incident=incident,
        repair_shop=repair_shop,
        participants=(participant,),
        payments=(payment,),
    )


def test_pgvector_query_uses_cosine_distance_and_excludes_source() -> None:
    """Candidate retrieval uses pgvector cosine distance and source exclusion."""
    sql = str(build_vector_search_sql(exclude_source_claim=True))
    assert "claim_embedding <=> CAST(:query_vector AS vector)" in sql
    assert "claim_id <> :exclude_claim_id" in sql
    assert "ORDER BY claim_embedding <=> CAST(:query_vector AS vector)" in sql


def test_vector_serialization_and_dimension_validation() -> None:
    """Vectors are serialized for pgvector and checked before search."""
    assert vector_to_pgvector_literal([0, 1.25, 2]) == "[0.0,1.25,2.0]"
    validate_query_vector([0.0] * 128)
    with pytest.raises(InvalidClaimVectorError):
        validate_query_vector([0.0] * 127)


def test_top_k_defaults_and_max_clamp() -> None:
    """top_k uses configured defaults and clamps large values."""
    settings = SimpleNamespace(default_top_k=7, max_top_k=20)
    assert resolve_top_k(None, settings=settings) == 7
    assert resolve_top_k(100, settings=settings) == 20
    with pytest.raises(ValueError):
        resolve_top_k(0, settings=settings)


def test_empty_pgvector_results_raise_domain_error() -> None:
    """An empty indexed-candidate result is surfaced as a semantic-search error."""

    class EmptyResult:
        def all(self) -> list[object]:
            return []

    class EmptySession:
        def execute(self, statement, params):
            return EmptyResult()

    with pytest.raises(NoIndexedClaimsError):
        search_similar_to_vector(EmptySession(), [0.0] * 128, top_k=5)


def test_missing_source_claim_raises_domain_error(monkeypatch) -> None:
    """Source claim lookup failures are reported clearly."""

    class MissingRepository:
        def __init__(self, session) -> None:
            pass

        def get_claim_evidence(self, claim_id: str):
            return None

    monkeypatch.setattr(semantic_search, "ClaimRepository", MissingRepository)
    with pytest.raises(ClaimNotFoundError):
        find_similar_claims(SimpleNamespace(), "CLM-MISSING")


def test_source_claim_without_embedding_raises_domain_error(monkeypatch) -> None:
    """Claims must be indexed before semantic retrieval."""

    class MissingEmbeddingRepository:
        def __init__(self, session) -> None:
            pass

        def get_claim_evidence(self, claim_id: str):
            return SimpleNamespace(claim=SimpleNamespace(claim_id=claim_id, claim_embedding=None))

    monkeypatch.setattr(semantic_search, "ClaimRepository", MissingEmbeddingRepository)
    with pytest.raises(ClaimEmbeddingMissingError):
        find_similar_claims(SimpleNamespace(), "CLM-1")


def test_ranked_candidate_order_is_preserved_when_attaching_evidence(monkeypatch) -> None:
    """Evidence is attached without reordering pgvector-ranked candidates."""
    source = SimpleNamespace(claim=SimpleNamespace(claim_id="CLM-1"))
    evidence_2 = SimpleNamespace(claim=SimpleNamespace(claim_id="CLM-2"))
    evidence_3 = SimpleNamespace(claim=SimpleNamespace(claim_id="CLM-3"))
    candidates = [
        SimilarClaim(claim_id="CLM-3", similarity_score=0.95),
        SimilarClaim(claim_id="CLM-2", similarity_score=0.91),
    ]
    monkeypatch.setattr(semantic_search, "explain_similarity", lambda source, evidence: ([], []))
    matches = attach_evidence_and_explanations(source, candidates, [evidence_2, evidence_3])
    assert [match.claim_id for match in matches] == ["CLM-3", "CLM-2"]
    assert [match.evidence.claim.claim_id for match in matches] == ["CLM-3", "CLM-2"]


def test_similarity_explanation_reports_shared_tokens_and_entities_without_label_leakage() -> None:
    """Explanations use deterministic evidence overlap, not fraud metadata."""
    source = _claim_evidence("CLM-1", historical_fraud_label=True, investigation_outcome="fraud")
    candidate = _claim_evidence(
        "CLM-2",
        historical_fraud_label=False,
        investigation_outcome="not_fraud",
    )
    shared_tokens, shared_entities = explain_similarity(source, candidate)
    combined_reasons = " ".join(shared_tokens + shared_entities)

    assert "incident_city=london" in shared_tokens
    assert "vehicle_type=suv" in shared_tokens
    assert "repair_shop_id=RS-1" in shared_entities
    assert "payment_bank_account_reference=BANK-1" in shared_entities
    assert "historical_fraud_label" not in combined_reasons
    assert "investigation_outcome" not in combined_reasons
