"""Read-oriented repository helpers for insurance investigation data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from .models import Claim, ClaimParticipant, Customer, Incident, Payment, Policy, RepairShop, Vehicle


@dataclass(frozen=True)
class ClaimEvidence:
    """Complete relational evidence for a claim."""

    claim: Claim
    policy: Policy
    customer: Customer
    vehicle: Vehicle
    incident: Incident
    repair_shop: RepairShop | None
    participants: Sequence[ClaimParticipant]
    payments: Sequence[Payment]


@dataclass(frozen=True)
class PaymentAggregate:
    """Payment summary used by data generation, embeddings, and investigations."""

    claim_id: str
    payment_count: int
    total_payment_amount: Decimal
    distinct_bank_accounts: int


class ClaimRepository:
    """Repository for claim-centric read operations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_claim(self, claim_id: str) -> Claim | None:
        """Return a claim by ID without eager-loading related records."""
        return self.session.get(Claim, claim_id)

    def get_claim_embedding(self, claim_id: str) -> list[float] | None:
        """Return the stored claim vector for a claim, if indexed."""
        statement = select(Claim.claim_embedding).where(Claim.claim_id == claim_id)
        return self.session.execute(statement).scalar_one_or_none()

    def get_claim_evidence(self, claim_id: str) -> ClaimEvidence | None:
        """Load a claim with its complete relational investigation evidence."""
        statement = self._claim_evidence_statement().where(Claim.claim_id == claim_id)
        claim = self.session.execute(statement).scalar_one_or_none()
        if claim is None:
            return None
        return self._to_evidence(claim)

    def list_claim_evidence(self, claim_ids: Iterable[str]) -> list[ClaimEvidence]:
        """Load complete evidence for a set of claim IDs.

        Results are returned in the same order as the supplied IDs where records
        exist. Missing IDs are ignored so service code can report them explicitly.
        """
        ordered_ids = list(dict.fromkeys(claim_ids))
        if not ordered_ids:
            return []

        statement = self._claim_evidence_statement().where(Claim.claim_id.in_(ordered_ids))
        claims = self.session.execute(statement).scalars().all()
        by_id = {claim.claim_id: self._to_evidence(claim) for claim in claims}
        return [by_id[claim_id] for claim_id in ordered_ids if claim_id in by_id]

    def list_customer_claims(self, customer_id: str) -> list[Claim]:
        """Return all claims linked to a customer through their policies."""
        statement = (
            select(Claim)
            .join(Claim.policy)
            .where(Policy.customer_id == customer_id)
            .order_by(Claim.claim_date.desc(), Claim.claim_id)
        )
        return list(self.session.execute(statement).scalars().all())

    def list_claims_for_repair_shop(self, repair_shop_id: str) -> list[Claim]:
        """Return claims associated with a repair shop."""
        statement = (
            select(Claim)
            .where(Claim.repair_shop_id == repair_shop_id)
            .order_by(Claim.claim_date.desc(), Claim.claim_id)
        )
        return list(self.session.execute(statement).scalars().all())

    def get_payment_aggregate(self, claim_id: str) -> PaymentAggregate:
        """Compute deterministic payment aggregates for one claim."""
        statement = (
            select(
                Payment.claim_id,
                func.count(Payment.payment_id),
                func.coalesce(func.sum(Payment.payment_amount), 0),
                func.count(func.distinct(Payment.bank_account_reference)),
            )
            .where(Payment.claim_id == claim_id)
            .group_by(Payment.claim_id)
        )
        row = self.session.execute(statement).one_or_none()
        if row is None:
            return PaymentAggregate(
                claim_id=claim_id,
                payment_count=0,
                total_payment_amount=Decimal("0"),
                distinct_bank_accounts=0,
            )
        return PaymentAggregate(
            claim_id=row[0],
            payment_count=int(row[1]),
            total_payment_amount=row[2],
            distinct_bank_accounts=int(row[3]),
        )

    @staticmethod
    def _claim_evidence_statement() -> Select[tuple[Claim]]:
        """Build the eager-loading query used for full claim evidence."""
        return select(Claim).options(
            selectinload(Claim.policy).selectinload(Policy.customer),
            selectinload(Claim.vehicle),
            selectinload(Claim.incident),
            selectinload(Claim.repair_shop),
            selectinload(Claim.participants),
            selectinload(Claim.payments),
        )

    @staticmethod
    def _to_evidence(claim: Claim) -> ClaimEvidence:
        """Convert an eager-loaded claim model into a typed evidence bundle."""
        return ClaimEvidence(
            claim=claim,
            policy=claim.policy,
            customer=claim.policy.customer,
            vehicle=claim.vehicle,
            incident=claim.incident,
            repair_shop=claim.repair_shop,
            participants=tuple(claim.participants),
            payments=tuple(claim.payments),
        )


class EntityLinkRepository:
    """Repository for shared-entity lookups used in investigations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def claim_ids_for_bank_account(self, bank_account_reference: str) -> list[str]:
        """Return claim IDs connected to a bank account through payments or repair shops."""
        payment_claims = select(Payment.claim_id).where(
            Payment.bank_account_reference == bank_account_reference
        )
        repair_shop_claims = (
            select(Claim.claim_id)
            .join(Claim.repair_shop)
            .where(RepairShop.bank_account_reference == bank_account_reference)
        )
        rows = self.session.execute(payment_claims.union(repair_shop_claims)).scalars().all()
        return sorted(set(rows))

    def claim_ids_for_phone_number(self, phone_number: str) -> list[str]:
        """Return claim IDs connected to a customer or participant phone number."""
        customer_claims = (
            select(Claim.claim_id)
            .join(Claim.policy)
            .join(Policy.customer)
            .where(Customer.phone_number == phone_number)
        )
        participant_claims = select(ClaimParticipant.claim_id).where(
            ClaimParticipant.phone_number == phone_number
        )
        rows = self.session.execute(customer_claims.union(participant_claims)).scalars().all()
        return sorted(set(rows))

    def claim_ids_for_address(self, address: str) -> list[str]:
        """Return claim IDs connected to a customer or participant address."""
        customer_claims = (
            select(Claim.claim_id)
            .join(Claim.policy)
            .join(Policy.customer)
            .where(Customer.address == address)
        )
        participant_claims = select(ClaimParticipant.claim_id).where(
            ClaimParticipant.address == address
        )
        rows = self.session.execute(customer_claims.union(participant_claims)).scalars().all()
        return sorted(set(rows))
