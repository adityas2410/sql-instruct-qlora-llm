"""Skip-Gram model primitives for claim-token embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Vocabulary:
    """Token vocabulary and observed token frequencies."""

    token_to_id: dict[str, int]
    frequencies: list[int]

    @property
    def id_to_token(self) -> dict[int, str]:
        """Return reverse vocabulary mapping."""
        return {token_id: token for token, token_id in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable vocabulary artifact."""
        return {
            "token_to_id": self.token_to_id,
            "frequencies": self.frequencies,
            "size": len(self),
        }


@dataclass(frozen=True)
class TrainingPair:
    """One positive center-context pair from a claim token bag."""

    center_id: int
    context_id: int


def build_vocabulary(
    tokens_by_claim: dict[str, list[str]],
    min_token_frequency: int = 1,
) -> Vocabulary:
    """Build deterministic vocabulary from tokenized claims."""
    counts: Counter[str] = Counter()
    for tokens in tokens_by_claim.values():
        counts.update(tokens)
    kept_tokens = sorted(
        token for token, count in counts.items() if count >= min_token_frequency
    )
    token_to_id = {token: index for index, token in enumerate(kept_tokens)}
    frequencies = [counts[token] for token in kept_tokens]
    return Vocabulary(token_to_id=token_to_id, frequencies=frequencies)


def create_skipgram_pairs(
    tokens_by_claim: dict[str, list[str]],
    vocabulary: Vocabulary | None = None,
    min_token_frequency: int = 1,
) -> list[TrainingPair]:
    """Create ordered positive pairs from unordered tokens within each claim."""
    vocabulary = vocabulary or build_vocabulary(tokens_by_claim, min_token_frequency)
    pairs: list[TrainingPair] = []
    for tokens in tokens_by_claim.values():
        token_ids = sorted({vocabulary.token_to_id[token] for token in tokens if token in vocabulary.token_to_id})
        for center_id in token_ids:
            for context_id in token_ids:
                if center_id != context_id:
                    pairs.append(TrainingPair(center_id=center_id, context_id=context_id))
    return pairs


class SkipGramModel(nn.Module):
    """Skip-Gram with Negative Sampling over structured claim tokens."""

    def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        positive_context_ids: torch.Tensor,
        negative_context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch SGNS loss."""
        center_vectors = self.input_embeddings(center_ids)
        positive_vectors = self.output_embeddings(positive_context_ids)
        negative_vectors = self.output_embeddings(negative_context_ids)

        positive_scores = torch.sum(center_vectors * positive_vectors, dim=1)
        positive_loss = F.logsigmoid(positive_scores)

        negative_scores = torch.bmm(negative_vectors, center_vectors.unsqueeze(2)).squeeze(2)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        return -(positive_loss + negative_loss).mean()

    def token_embedding_matrix(self) -> torch.Tensor:
        """Return the learned input-token embedding matrix for pooling."""
        return self.input_embeddings.weight.detach().cpu()


class PairDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper for Skip-Gram positive pairs."""

    def __init__(self, pairs: Iterable[TrainingPair]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[int, int]:
        pair = self.pairs[index]
        return pair.center_id, pair.context_id
