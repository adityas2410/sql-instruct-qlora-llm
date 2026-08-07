"""Preprocessing helpers for structured claim embedding tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

MISSING_VALUE_TOKEN = "missing"


@dataclass(frozen=True)
class NumericBin:
    """Numeric interval used to convert continuous fields into stable tokens."""

    lower: float | None
    upper: float | None
    label: str

    def contains(self, value: float) -> bool:
        """Return True when a value falls inside the interval."""
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value >= self.upper:
            return False
        return True


class NumericBinner:
    """Column-aware binner for model-ready numeric feature tokens."""

    def __init__(self, bins_by_feature: dict[str, tuple[NumericBin, ...]] | None = None) -> None:
        self.bins_by_feature = bins_by_feature or default_bins_by_feature()

    def bin_value(self, feature_name: str, value: int | float | Decimal | None) -> str | None:
        """Return the configured bin label for a feature value."""
        if value is None:
            return None
        numeric_value = float(value)
        bins = self.bins_by_feature.get(feature_name)
        if bins is None:
            raise KeyError(f"No numeric bins configured for feature: {feature_name}")
        for numeric_bin in bins:
            if numeric_bin.contains(numeric_value):
                return numeric_bin.label
        raise ValueError(f"No bin matched {feature_name}={numeric_value}")

    def to_metadata(self) -> dict[str, list[dict[str, float | str | None]]]:
        """Return bin metadata for artifact export."""
        return {
            feature_name: [
                {"lower": numeric_bin.lower, "upper": numeric_bin.upper, "label": numeric_bin.label}
                for numeric_bin in bins
            ]
            for feature_name, bins in self.bins_by_feature.items()
        }


def money_bins() -> tuple[NumericBin, ...]:
    """Default bins for monetary fields."""
    return (
        NumericBin(None, 1000, "0_1000"),
        NumericBin(1000, 5000, "1000_5000"),
        NumericBin(5000, 10000, "5000_10000"),
        NumericBin(10000, 15000, "10000_15000"),
        NumericBin(15000, 25000, "15000_25000"),
        NumericBin(25000, 50000, "25000_50000"),
        NumericBin(50000, None, "50000_plus"),
    )


def day_bins() -> tuple[NumericBin, ...]:
    """Default bins for day-count fields."""
    return (
        NumericBin(None, 31, "0_30_days"),
        NumericBin(31, 91, "31_90_days"),
        NumericBin(91, 181, "91_180_days"),
        NumericBin(181, 366, "181_365_days"),
        NumericBin(366, None, "365_plus_days"),
    )


def age_bins() -> tuple[NumericBin, ...]:
    """Default bins for customer age."""
    return (
        NumericBin(None, 25, "under_25"),
        NumericBin(25, 35, "25_34"),
        NumericBin(35, 45, "35_44"),
        NumericBin(45, 60, "45_59"),
        NumericBin(60, None, "60_plus"),
    )


def count_bins() -> tuple[NumericBin, ...]:
    """Default bins for count features."""
    return (
        NumericBin(0, 1, "0"),
        NumericBin(1, 2, "1"),
        NumericBin(2, 4, "2_3"),
        NumericBin(4, 6, "4_5"),
        NumericBin(6, 11, "6_10"),
        NumericBin(11, None, "10_plus"),
    )


def default_bins_by_feature() -> dict[str, tuple[NumericBin, ...]]:
    """Return the approved numeric bins for claim embedding features."""
    money_features = (
        "claim_amount",
        "repair_cost",
        "annual_income",
        "vehicle_value",
        "previous_total_claim_amount",
        "total_payment_amount",
    )
    day_features = ("policy_age_days", "incident_to_claim_delay_days")
    count_features = (
        "payment_count",
        "previous_claim_count",
        "customer_claim_frequency",
        "distinct_payment_bank_accounts",
        "shared_bank_account_count",
        "shared_phone_count",
        "shared_address_count",
    )
    bins: dict[str, tuple[NumericBin, ...]] = {name: money_bins() for name in money_features}
    bins.update({name: day_bins() for name in day_features})
    bins["customer_age_years"] = age_bins()
    bins.update({name: count_bins() for name in count_features})
    return bins


def normalize_token_value(value: object) -> str:
    """Normalize a feature value into a compact token-safe label."""
    if value is None:
        return MISSING_VALUE_TOKEN
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip().lower()
    if not text:
        return MISSING_VALUE_TOKEN
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or MISSING_VALUE_TOKEN


def stable_unique(values: Iterable[str]) -> list[str]:
    """Return sorted unique values for deterministic unordered token bags."""
    return sorted(set(values))
