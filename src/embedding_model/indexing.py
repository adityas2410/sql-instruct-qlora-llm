"""Database indexing helpers for pooled claim vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy.orm import Session

from database.models import Claim
from database.schema import CLAIM_VECTOR_DIMENSION


def validate_claim_vector(
    vector: Sequence[float],
    expected_dimension: int = CLAIM_VECTOR_DIMENSION,
) -> None:
    """Validate vector dimensionality before writing to pgvector."""
    if len(vector) != expected_dimension:
        raise ValueError(
            f"Claim vector has dimension {len(vector)}; expected {expected_dimension}"
        )


def index_claim_vectors(
    session: Session,
    vectors_by_claim_id: Mapping[str, Sequence[float]],
    expected_dimension: int = CLAIM_VECTOR_DIMENSION,
) -> int:
    """Write pooled claim vectors into claims.claim_embedding."""
    indexed_count = 0
    for claim_id, vector in vectors_by_claim_id.items():
        validate_claim_vector(vector, expected_dimension=expected_dimension)
        claim = session.get(Claim, claim_id)
        if claim is None:
            continue
        claim.claim_embedding = [float(value) for value in vector]
        indexed_count += 1
    return indexed_count
