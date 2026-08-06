"""Database package for the insurance investigation application."""

from .models import (
    Base,
    Claim,
    ClaimParticipant,
    Customer,
    Incident,
    Payment,
    Policy,
    RepairShop,
    Vehicle,
)
from .repositories import ClaimEvidence, ClaimRepository, EntityLinkRepository, PaymentAggregate
from .schema import APPROVED_COLUMNS_BY_TABLE, APPROVED_TABLE_NAMES, APPROVED_TABLES
from .session import get_db_session, get_engine, get_session_factory, session_scope

__all__ = [
    "APPROVED_COLUMNS_BY_TABLE",
    "APPROVED_TABLE_NAMES",
    "APPROVED_TABLES",
    "Base",
    "Claim",
    "ClaimEvidence",
    "ClaimParticipant",
    "ClaimRepository",
    "Customer",
    "EntityLinkRepository",
    "Incident",
    "Payment",
    "PaymentAggregate",
    "Policy",
    "RepairShop",
    "Vehicle",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "session_scope",
]
