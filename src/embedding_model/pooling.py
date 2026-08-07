"""Mean-pooling helpers for claim vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from .skipgram import Vocabulary


def pool_claim_vector(
    tokens: list[str],
    token_embeddings: torch.Tensor,
    vocabulary: Vocabulary,
) -> list[float] | None:
    """Mean-pool known token vectors into one claim vector."""
    token_ids = [vocabulary.token_to_id[token] for token in tokens if token in vocabulary.token_to_id]
    if not token_ids:
        return None
    vectors = token_embeddings[token_ids]
    pooled = vectors.mean(dim=0)
    return [float(value) for value in pooled.tolist()]


def pool_claim_vectors(
    tokens_by_claim: Mapping[str, list[str]],
    token_embeddings: torch.Tensor,
    vocabulary: Vocabulary,
) -> tuple[dict[str, list[float]], list[str]]:
    """Pool vectors for all claims and report claims with no known tokens."""
    vectors_by_claim_id: dict[str, list[float]] = {}
    skipped_claim_ids: list[str] = []
    for claim_id, tokens in tokens_by_claim.items():
        vector = pool_claim_vector(tokens, token_embeddings, vocabulary)
        if vector is None:
            skipped_claim_ids.append(claim_id)
            continue
        vectors_by_claim_id[claim_id] = vector
    return vectors_by_claim_id, skipped_claim_ids


def write_claim_vectors_json(vectors_by_claim_id: Mapping[str, list[float]], output_path: Path) -> None:
    """Write pooled claim vectors to JSON for later indexing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(vectors_by_claim_id), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_claim_vectors_json(path: Path) -> dict[str, list[float]]:
    """Read pooled claim vectors from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {claim_id: [float(value) for value in vector] for claim_id, vector in raw.items()}
