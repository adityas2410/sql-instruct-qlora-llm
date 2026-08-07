"""Unit tests for joined claim feature construction."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from database.models import Claim, Customer, Incident, Policy, Vehicle
from database.repositories import ClaimEvidence
from embedding_model.features import FeatureContext, build_claim_features


def _evidence_without_shop_or_payments() -> ClaimEvidence:
    customer = Customer(
        customer_id="CUST-1",
        full_name="Alex Reed",
        date_of_birth=date(1988, 5, 4),
        occupation="Engineer",
        annual_income=Decimal("82000.00"),
        address="10 Main Street",
        city="London",
        postcode="L1 1AA",
        phone_number="555-0101",
        email="alex@example.com",
        account_created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    policy = Policy(
        policy_id="POL-1",
        customer_id="CUST-1",
        policy_type="auto",
        coverage_amount=Decimal("50000.00"),
        premium_amount=Decimal("1200.00"),
        deductible=Decimal("500.00"),
        policy_start_date=date(2025, 1, 1),
        policy_end_date=date(2026, 1, 1),
        policy_status="active",
    )
    policy.customer = customer
    vehicle = Vehicle(
        vehicle_id="VEH-1",
        customer_id="CUST-1",
        make="Toyota",
        model="Rav4",
        manufacture_year=2021,
        vehicle_type="SUV",
        estimated_value=Decimal("28000.00"),
        registration_region="Greater London",
    )
    incident = Incident(
        incident_id="INC-1",
        incident_type="collision",
        incident_date=date(2025, 1, 10),
        incident_city="London",
        incident_address="A10",
        weather_condition="rain",
        police_report_reference=None,
        witness_count=0,
    )
    claim = Claim(
        claim_id="CLM-1",
        policy_id="POL-1",
        vehicle_id="VEH-1",
        incident_id="INC-1",
        repair_shop_id=None,
        claim_date=date(2025, 1, 12),
        claim_amount=Decimal("12000.00"),
        repair_cost=Decimal("9000.00"),
        damage_type="front_end",
        injury_reported=False,
        police_report_available=False,
        claim_description="Front bumper damage",
        claim_status="open",
        historical_fraud_label=True,
        investigation_outcome="confirmed_fraud",
        claim_embedding=None,
    )
    return ClaimEvidence(
        claim=claim,
        policy=policy,
        customer=customer,
        vehicle=vehicle,
        incident=incident,
        repair_shop=None,
        participants=(),
        payments=(),
    )


def test_joined_feature_construction_handles_missing_shop_and_no_payments() -> None:
    """Feature builder keeps relational context without requiring optional links."""
    evidence = _evidence_without_shop_or_payments()
    features = build_claim_features(evidence, FeatureContext())
    assert features.claim_id == "CLM-1"
    assert features.categorical["incident_city"] == "London"
    assert features.categorical["vehicle_type"] == "SUV"
    assert "repair_shop_id" not in features.categorical
    assert features.numeric["payment_count"] == 0
    assert features.numeric["incident_to_claim_delay_days"] == 2
    assert features.numeric["policy_age_days"] == 11
    assert "historical_fraud_label" not in features.categorical
    assert "investigation_outcome" not in features.categorical
