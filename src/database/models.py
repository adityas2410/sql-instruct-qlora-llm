"""SQLAlchemy ORM models for the insurance investigation database."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .schema import CLAIM_VECTOR_DIMENSION
from .vector_types import PgVector


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Customer(Base):
    """Insurance customer with contact and demographic fields."""

    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String, primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    date_of_birth: Mapped[date] = mapped_column(Date, nullable=False)
    occupation: Mapped[str | None] = mapped_column(Text)
    annual_income: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(Text)
    postcode: Mapped[str | None] = mapped_column(Text)
    phone_number: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)
    account_created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    policies: Mapped[list[Policy]] = relationship(back_populates="customer")
    vehicles: Mapped[list[Vehicle]] = relationship(back_populates="customer")

    __table_args__ = (
        Index("idx_customers_city", "city"),
        Index("idx_customers_postcode", "postcode"),
        Index("idx_customers_phone_number", "phone_number"),
        Index("idx_customers_email", "email"),
    )


class Policy(Base):
    """Insurance policy linked to a customer."""

    __tablename__ = "policies"

    policy_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    policy_type: Mapped[str] = mapped_column(Text, nullable=False)
    coverage_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    premium_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    deductible: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    policy_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    policy_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    policy_status: Mapped[str] = mapped_column(Text, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="policies")
    claims: Mapped[list[Claim]] = relationship(back_populates="policy")

    __table_args__ = (
        Index("idx_policies_customer_id", "customer_id"),
        Index("idx_policies_type_status", "policy_type", "policy_status"),
        Index("idx_policies_start_date", "policy_start_date"),
    )


class Vehicle(Base):
    """Insured vehicle associated with a customer and claim."""

    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    make: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    manufacture_year: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_type: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    registration_region: Mapped[str | None] = mapped_column(Text)

    customer: Mapped[Customer] = relationship(back_populates="vehicles")
    claims: Mapped[list[Claim]] = relationship(back_populates="vehicle")

    __table_args__ = (
        Index("idx_vehicles_customer_id", "customer_id"),
        Index("idx_vehicles_type_region", "vehicle_type", "registration_region"),
    )


class RepairShop(Base):
    """Repair shop entity used for repair and payment-link analysis."""

    __tablename__ = "repair_shops"

    repair_shop_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[str | None] = mapped_column(Text)
    postcode: Mapped[str | None] = mapped_column(Text)
    owner_name: Mapped[str | None] = mapped_column(Text)
    bank_account_reference: Mapped[str | None] = mapped_column(Text)
    registration_date: Mapped[date | None] = mapped_column(Date)

    claims: Mapped[list[Claim]] = relationship(back_populates="repair_shop")

    __table_args__ = (
        Index("idx_repair_shops_city", "city"),
        Index("idx_repair_shops_postcode", "postcode"),
        Index("idx_repair_shops_bank_account", "bank_account_reference"),
    )


class Incident(Base):
    """Incident details attached to one or more claim records."""

    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String, primary_key=True)
    incident_type: Mapped[str] = mapped_column(Text, nullable=False)
    incident_date: Mapped[date] = mapped_column(Date, nullable=False)
    incident_city: Mapped[str | None] = mapped_column(Text)
    incident_address: Mapped[str | None] = mapped_column(Text)
    weather_condition: Mapped[str | None] = mapped_column(Text)
    police_report_reference: Mapped[str | None] = mapped_column(Text)
    witness_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    claims: Mapped[list[Claim]] = relationship(back_populates="incident")

    __table_args__ = (
        Index("idx_incidents_type_city", "incident_type", "incident_city"),
        Index("idx_incidents_date", "incident_date"),
    )


class Claim(Base):
    """Insurance claim with relational links and optional pgvector embedding."""

    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String, primary_key=True)
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.policy_id"), nullable=False)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.vehicle_id"), nullable=False)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id"), nullable=False)
    repair_shop_id: Mapped[str | None] = mapped_column(ForeignKey("repair_shops.repair_shop_id"))
    claim_date: Mapped[date] = mapped_column(Date, nullable=False)
    claim_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    repair_cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    damage_type: Mapped[str | None] = mapped_column(Text)
    injury_reported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    police_report_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    claim_description: Mapped[str | None] = mapped_column(Text)
    claim_status: Mapped[str] = mapped_column(Text, nullable=False)
    historical_fraud_label: Mapped[bool | None] = mapped_column(Boolean)
    investigation_outcome: Mapped[str | None] = mapped_column(Text)
    claim_embedding: Mapped[list[float] | None] = mapped_column(PgVector(CLAIM_VECTOR_DIMENSION))

    policy: Mapped[Policy] = relationship(back_populates="claims")
    vehicle: Mapped[Vehicle] = relationship(back_populates="claims")
    incident: Mapped[Incident] = relationship(back_populates="claims")
    repair_shop: Mapped[RepairShop | None] = relationship(back_populates="claims")
    participants: Mapped[list[ClaimParticipant]] = relationship(back_populates="claim")
    payments: Mapped[list[Payment]] = relationship(back_populates="claim")

    __table_args__ = (
        Index("idx_claims_policy_id", "policy_id"),
        Index("idx_claims_vehicle_id", "vehicle_id"),
        Index("idx_claims_incident_id", "incident_id"),
        Index("idx_claims_repair_shop_id", "repair_shop_id"),
        Index("idx_claims_claim_date", "claim_date"),
        Index("idx_claims_amount", "claim_amount"),
        Index("idx_claims_status", "claim_status"),
        Index("idx_claims_historical_fraud_label", "historical_fraud_label"),
    )


class ClaimParticipant(Base):
    """Person or organization participating in a claim."""

    __tablename__ = "claim_participants"

    participant_id: Mapped[str] = mapped_column(String, primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), nullable=False)
    participant_type: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(Text)
    phone_number: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text)
    relationship_to_claim: Mapped[str | None] = mapped_column(Text)

    claim: Mapped[Claim] = relationship(back_populates="participants")

    __table_args__ = (
        Index("idx_claim_participants_claim_id", "claim_id"),
        Index("idx_claim_participants_phone", "phone_number"),
        Index("idx_claim_participants_email", "email"),
        Index("idx_claim_participants_address", "address"),
    )


class Payment(Base):
    """Payment made against a claim."""

    __tablename__ = "payments"

    payment_id: Mapped[str] = mapped_column(String, primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), nullable=False)
    recipient_type: Mapped[str] = mapped_column(Text, nullable=False)
    recipient_id: Mapped[str | None] = mapped_column(Text)
    bank_account_reference: Mapped[str | None] = mapped_column(Text)
    payment_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    payment_status: Mapped[str] = mapped_column(Text, nullable=False)

    claim: Mapped[Claim] = relationship(back_populates="payments")

    __table_args__ = (
        Index("idx_payments_claim_id", "claim_id"),
        Index("idx_payments_bank_account", "bank_account_reference"),
        Index("idx_payments_recipient", "recipient_type", "recipient_id"),
        Index("idx_payments_date", "payment_date"),
    )
