"""Unit tests for Skip-Gram training primitives."""

from __future__ import annotations

import torch

from embedding_model.negative_sampling import NegativeSampler
from embedding_model.skipgram import build_vocabulary, create_skipgram_pairs


def test_positive_pairs_are_created_from_unordered_claim_tokens() -> None:
    """Every token can act as context for every other token in the same claim."""
    tokens_by_claim = {"CLM-1": ["a=1", "b=2", "c=3"]}
    vocabulary = build_vocabulary(tokens_by_claim)
    pairs = create_skipgram_pairs(tokens_by_claim, vocabulary=vocabulary)
    pair_ids = {(pair.center_id, pair.context_id) for pair in pairs}
    assert len(pairs) == 6
    assert len(pair_ids) == 6
    assert all(center_id != context_id for center_id, context_id in pair_ids)


def test_negative_sampler_avoids_positive_context_when_avoidable() -> None:
    """Negative samples avoid the positive target when vocabulary has alternatives."""
    vocabulary = build_vocabulary({"CLM-1": ["a=1", "b=2", "c=3"]})
    sampler = NegativeSampler(vocabulary, seed=7)
    positives = torch.tensor([0, 1, 2])
    negatives = sampler.sample(positives, negative_samples=4)
    assert negatives.shape == (3, 4)
    assert bool((negatives != positives.view(3, 1)).all())
