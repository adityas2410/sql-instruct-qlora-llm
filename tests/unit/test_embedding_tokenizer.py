"""Unit tests for column-aware claim tokenization."""

from __future__ import annotations

from embedding_model.features import StructuredClaimFeatures
from embedding_model.tokenizer import tokenize_claim_features


def test_column_aware_token_formatting_and_normalization() -> None:
    """Tokens retain feature names and normalized feature values."""
    features = StructuredClaimFeatures(
        claim_id="CLM-1",
        categorical={"incident_city": "London", "vehicle_type": "SUV", "repair_shop_id": "RS-0001"},
        numeric={"claim_amount": 25000, "policy_age_days": 12},
    )
    tokens = tokenize_claim_features(features)
    assert "incident_city=london" in tokens
    assert "vehicle_type=suv" in tokens
    assert "repair_shop_id=rs_0001" in tokens
    assert "claim_amount_bin=25000_50000" in tokens
    assert "policy_age_days_bin=0_30_days" in tokens


def test_fraud_labels_and_outcomes_are_excluded_from_tokens() -> None:
    """Embedding tokens do not include supervised fraud metadata."""
    features = StructuredClaimFeatures(
        claim_id="CLM-2",
        categorical={
            "incident_city": "London",
            "historical_fraud_label": True,
            "investigation_outcome": "confirmed_fraud",
        },
        numeric={"claim_amount": 12000},
    )
    tokens = tokenize_claim_features(features)
    assert "incident_city=london" in tokens
    assert all("historical_fraud_label" not in token for token in tokens)
    assert all("investigation_outcome" not in token for token in tokens)
