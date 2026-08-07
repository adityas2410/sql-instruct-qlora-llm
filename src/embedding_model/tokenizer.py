"""Column-aware tokenization for structured claim features."""

from __future__ import annotations

from .features import StructuredClaimFeatures
from .preprocessing import NumericBinner, normalize_token_value, stable_unique

EXCLUDED_FEATURE_NAMES = frozenset({"historical_fraud_label", "investigation_outcome"})


def format_feature_token(feature_name: str, value: object) -> str | None:
    """Format one categorical feature as feature=value."""
    if feature_name in EXCLUDED_FEATURE_NAMES or value is None:
        return None
    return f"{feature_name}={normalize_token_value(value)}"


def tokenize_claim_features(
    features: StructuredClaimFeatures,
    binner: NumericBinner | None = None,
) -> list[str]:
    """Tokenize joined claim features into an unordered column-aware token bag."""
    binner = binner or NumericBinner()
    tokens: list[str] = []

    for feature_name, value in features.categorical.items():
        token = format_feature_token(feature_name, value)
        if token is not None:
            tokens.append(token)

    for feature_name, value in features.numeric.items():
        if feature_name in EXCLUDED_FEATURE_NAMES or value is None:
            continue
        bin_label = binner.bin_value(feature_name, value)
        if bin_label is not None:
            tokens.append(f"{feature_name}_bin={normalize_token_value(bin_label)}")

    return stable_unique(tokens)
