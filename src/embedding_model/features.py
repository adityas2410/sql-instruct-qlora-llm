"""Joined relational feature construction for claim embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Sequence

from database.repositories import ClaimEvidence


def _to_float(value: Decimal | int | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _days_between(start: date | None, end: date | None) -> int | None:
    if start is None or end is None:
        return None
    return max((end - start).days, 0)


def _age_years(birth_date: date | None, reference_date: date | None) -> int | None:
    if birth_date is None or reference_date is None:
        return None
    before_birthday = (reference_date.month, reference_date.day) < (
        birth_date.month,
        birth_date.day,
    )
    return max(reference_date.year - birth_date.year - int(before_birthday), 0)


@dataclass(frozen=True)
class StructuredClaimFeatures:
    """Embedding-ready joined representation for one claim."""

    claim_id: str
    categorical: dict[str, object] = field(default_factory=dict)
    numeric: dict[str, float | int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureContext:
    """Deterministic cross-claim counts used in claim feature construction."""

    previous_claim_count_by_claim_id: dict[str, int] = field(default_factory=dict)
    previous_total_claim_amount_by_claim_id: dict[str, float] = field(default_factory=dict)
    customer_claim_frequency_by_claim_id: dict[str, int] = field(default_factory=dict)
    shared_bank_account_count_by_claim_id: dict[str, int] = field(default_factory=dict)
    shared_phone_count_by_claim_id: dict[str, int] = field(default_factory=dict)
    shared_address_count_by_claim_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_evidence(cls, evidence_items: Sequence[ClaimEvidence]) -> FeatureContext:
        """Build deterministic prior-claim and shared-identifier counts."""
        sorted_evidence = sorted(
            evidence_items,
            key=lambda evidence: (evidence.claim.claim_date, evidence.claim.claim_id),
        )
        by_customer: dict[str, list[ClaimEvidence]] = defaultdict(list)
        for evidence in sorted_evidence:
            by_customer[evidence.customer.customer_id].append(evidence)

        previous_count: dict[str, int] = {}
        previous_amount: dict[str, float] = {}
        customer_frequency: dict[str, int] = {}
        for customer_claims in by_customer.values():
            running_amount = 0.0
            for index, evidence in enumerate(customer_claims):
                claim_id = evidence.claim.claim_id
                previous_count[claim_id] = index
                previous_amount[claim_id] = running_amount
                customer_frequency[claim_id] = len(customer_claims)
                running_amount += float(evidence.claim.claim_amount)

        bank_accounts = _identifier_counts(evidence_items, _bank_accounts_for_evidence)
        phone_numbers = _identifier_counts(evidence_items, _phone_numbers_for_evidence)
        addresses = _identifier_counts(evidence_items, _addresses_for_evidence)

        return cls(
            previous_claim_count_by_claim_id=previous_count,
            previous_total_claim_amount_by_claim_id=previous_amount,
            customer_claim_frequency_by_claim_id=customer_frequency,
            shared_bank_account_count_by_claim_id=_shared_count_by_claim(
                evidence_items,
                bank_accounts,
                _bank_accounts_for_evidence,
            ),
            shared_phone_count_by_claim_id=_shared_count_by_claim(
                evidence_items,
                phone_numbers,
                _phone_numbers_for_evidence,
            ),
            shared_address_count_by_claim_id=_shared_count_by_claim(
                evidence_items,
                addresses,
                _addresses_for_evidence,
            ),
        )


def _identifier_counts(
    evidence_items: Iterable[ClaimEvidence],
    extractor,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for evidence in evidence_items:
        counter.update(set(extractor(evidence)))
    return counter


def _shared_count_by_claim(
    evidence_items: Iterable[ClaimEvidence],
    identifier_counts: Counter[str],
    extractor,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evidence in evidence_items:
        counts[evidence.claim.claim_id] = sum(
            max(identifier_counts[identifier] - 1, 0)
            for identifier in set(extractor(evidence))
        )
    return counts


def _bank_accounts_for_evidence(evidence: ClaimEvidence) -> list[str]:
    identifiers = [
        payment.bank_account_reference
        for payment in evidence.payments
        if payment.bank_account_reference
    ]
    if evidence.repair_shop and evidence.repair_shop.bank_account_reference:
        identifiers.append(evidence.repair_shop.bank_account_reference)
    return identifiers


def _phone_numbers_for_evidence(evidence: ClaimEvidence) -> list[str]:
    identifiers = [evidence.customer.phone_number] if evidence.customer.phone_number else []
    identifiers.extend(
        participant.phone_number
        for participant in evidence.participants
        if participant.phone_number
    )
    return identifiers


def _addresses_for_evidence(evidence: ClaimEvidence) -> list[str]:
    identifiers = [evidence.customer.address] if evidence.customer.address else []
    identifiers.extend(participant.address for participant in evidence.participants if participant.address)
    return identifiers


def build_claim_features(
    evidence: ClaimEvidence,
    context: FeatureContext | None = None,
) -> StructuredClaimFeatures:
    """Build joined claim features from relational evidence.

    Historical fraud labels and investigation outcomes are intentionally excluded;
    they remain retrieval metadata rather than self-supervised training signal.
    """
    context = context or FeatureContext()
    claim = evidence.claim
    policy = evidence.policy
    customer = evidence.customer
    vehicle = evidence.vehicle
    incident = evidence.incident
    repair_shop = evidence.repair_shop

    total_payment_amount = sum(float(payment.payment_amount) for payment in evidence.payments)
    distinct_bank_accounts = len(
        {
            payment.bank_account_reference
            for payment in evidence.payments
            if payment.bank_account_reference
        }
    )

    categorical: dict[str, object] = {
        "claim_status": claim.claim_status,
        "damage_type": claim.damage_type,
        "injury_reported": claim.injury_reported,
        "police_report_available": claim.police_report_available,
        "policy_type": policy.policy_type,
        "policy_status": policy.policy_status,
        "customer_city": customer.city,
        "customer_postcode": customer.postcode,
        "customer_occupation": customer.occupation,
        "vehicle_make": vehicle.make,
        "vehicle_model": vehicle.model,
        "vehicle_type": vehicle.vehicle_type,
        "vehicle_registration_region": vehicle.registration_region,
        "incident_type": incident.incident_type,
        "incident_city": incident.incident_city,
        "weather_condition": incident.weather_condition,
        "has_police_report_reference": bool(incident.police_report_reference),
    }
    if repair_shop is not None:
        categorical.update(
            {
                "repair_shop_id": repair_shop.repair_shop_id,
                "repair_shop_city": repair_shop.city,
                "repair_shop_postcode": repair_shop.postcode,
                "repair_shop_bank_account_reference": repair_shop.bank_account_reference,
            }
        )

    numeric: dict[str, float | int | None] = {
        "claim_amount": _to_float(claim.claim_amount),
        "repair_cost": _to_float(claim.repair_cost),
        "annual_income": _to_float(customer.annual_income),
        "vehicle_value": _to_float(vehicle.estimated_value),
        "policy_age_days": _days_between(policy.policy_start_date, claim.claim_date),
        "customer_age_years": _age_years(customer.date_of_birth, claim.claim_date),
        "incident_to_claim_delay_days": _days_between(incident.incident_date, claim.claim_date),
        "payment_count": len(evidence.payments),
        "total_payment_amount": total_payment_amount,
        "distinct_payment_bank_accounts": distinct_bank_accounts,
        "previous_claim_count": context.previous_claim_count_by_claim_id.get(claim.claim_id, 0),
        "previous_total_claim_amount": context.previous_total_claim_amount_by_claim_id.get(
            claim.claim_id,
            0.0,
        ),
        "customer_claim_frequency": context.customer_claim_frequency_by_claim_id.get(
            claim.claim_id,
            1,
        ),
        "shared_bank_account_count": context.shared_bank_account_count_by_claim_id.get(
            claim.claim_id,
            0,
        ),
        "shared_phone_count": context.shared_phone_count_by_claim_id.get(claim.claim_id, 0),
        "shared_address_count": context.shared_address_count_by_claim_id.get(claim.claim_id, 0),
    }
    return StructuredClaimFeatures(claim_id=claim.claim_id, categorical=categorical, numeric=numeric)
