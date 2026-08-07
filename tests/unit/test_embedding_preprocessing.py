"""Unit tests for claim embedding preprocessing."""

from __future__ import annotations

from decimal import Decimal

from embedding_model.preprocessing import NumericBinner, normalize_token_value


def test_numeric_binning_boundaries_and_labels() -> None:
    """Configured numeric bins use stable inclusive-lower boundaries."""
    binner = NumericBinner()
    assert binner.bin_value("claim_amount", Decimal("999.99")) == "0_1000"
    assert binner.bin_value("claim_amount", Decimal("1000.00")) == "1000_5000"
    assert binner.bin_value("claim_amount", Decimal("50000.00")) == "50000_plus"
    assert binner.bin_value("policy_age_days", 30) == "0_30_days"
    assert binner.bin_value("policy_age_days", 31) == "31_90_days"
    assert binner.bin_value("payment_count", 3) == "2_3"


def test_token_value_normalization() -> None:
    """Feature values are normalized into compact token labels."""
    assert normalize_token_value("New York City") == "new_york_city"
    assert normalize_token_value(" RS-0001 ") == "rs_0001"
    assert normalize_token_value(True) == "true"
    assert normalize_token_value("") == "missing"
