"""Unit tests for claim-vector pooling and indexing validation."""

from __future__ import annotations

import pytest
import torch

from database.models import Claim
from embedding_model.indexing import index_claim_vectors, validate_claim_vector
from embedding_model.pooling import pool_claim_vector, pool_claim_vectors
from embedding_model.skipgram import Vocabulary


class FakeSession:
    """Minimal session double for vector indexing tests."""

    def __init__(self, claims: dict[str, Claim]) -> None:
        self.claims = claims

    def get(self, model, primary_key: str):
        assert model is Claim
        return self.claims.get(primary_key)


def test_mean_pooling_shape_and_values_with_fake_embeddings() -> None:
    """Known token vectors are averaged into one claim vector."""
    vocabulary = Vocabulary(token_to_id={"a=1": 0, "b=2": 1}, frequencies=[1, 1])
    embeddings = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    vector = pool_claim_vector(["a=1", "unknown=x", "b=2"], embeddings, vocabulary)
    assert vector == [3.0, 5.0]


def test_pool_claim_vectors_reports_claims_with_no_known_tokens() -> None:
    """Claims without vocabulary overlap are skipped rather than indexed."""
    vocabulary = Vocabulary(token_to_id={"a=1": 0}, frequencies=[1])
    embeddings = torch.tensor([[2.0, 4.0]])
    vectors, skipped = pool_claim_vectors(
        {"CLM-1": ["a=1"], "CLM-2": ["missing=token"]},
        embeddings,
        vocabulary,
    )
    assert vectors == {"CLM-1": [2.0, 4.0]}
    assert skipped == ["CLM-2"]


def test_vector_dimension_validation_before_indexing() -> None:
    """Indexing rejects vectors that do not match pgvector dimension."""
    with pytest.raises(ValueError):
        validate_claim_vector([0.1, 0.2], expected_dimension=128)


def test_index_claim_vectors_updates_existing_claims_only() -> None:
    """Vector indexing updates found claims and ignores missing claim IDs."""
    claim = Claim(claim_id="CLM-1")
    session = FakeSession({"CLM-1": claim})
    indexed = index_claim_vectors(
        session,
        {"CLM-1": [0.1, 0.2], "CLM-2": [0.3, 0.4]},
        expected_dimension=2,
    )
    assert indexed == 1
    assert claim.claim_embedding == [0.1, 0.2]
