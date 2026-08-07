"""Training and artifact export for the custom Skip-Gram embedding model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import EmbeddingArtifactPaths, SkipGramConfig
from .negative_sampling import NegativeSampler
from .preprocessing import NumericBinner
from .skipgram import PairDataset, SkipGramModel, Vocabulary, build_vocabulary, create_skipgram_pairs


@dataclass(frozen=True)
class SkipGramTrainingResult:
    """Completed Skip-Gram training output."""

    model: SkipGramModel
    vocabulary: Vocabulary
    loss_history: list[float]

    @property
    def token_embeddings(self) -> torch.Tensor:
        """Return learned input-token embeddings used for claim-vector pooling."""
        return self.model.token_embedding_matrix()


def train_skipgram_model(
    tokens_by_claim: dict[str, list[str]],
    config: SkipGramConfig,
) -> SkipGramTrainingResult:
    """Train SGNS token embeddings from structured claim tokens."""
    torch.manual_seed(config.random_seed)
    vocabulary = build_vocabulary(
        tokens_by_claim,
        min_token_frequency=config.min_token_frequency,
    )
    if len(vocabulary) < 2:
        raise ValueError("Skip-Gram training requires at least two vocabulary tokens")

    pairs = create_skipgram_pairs(tokens_by_claim, vocabulary=vocabulary)
    if not pairs:
        raise ValueError("Skip-Gram training requires at least one positive token pair")

    model = SkipGramModel(
        vocabulary_size=len(vocabulary),
        embedding_dimension=config.embedding_dimension,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sampler = NegativeSampler(
        vocabulary,
        power=config.negative_sampling_power,
        seed=config.random_seed,
    )
    loader = DataLoader(
        PairDataset(pairs),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )

    loss_history: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        total_loss = 0.0
        batch_count = 0
        for center_ids, context_ids in loader:
            center_ids = center_ids.long()
            context_ids = context_ids.long()
            negative_ids = sampler.sample(context_ids, config.negative_samples)
            loss = model(center_ids, context_ids, negative_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        loss_history.append(total_loss / max(batch_count, 1))

    return SkipGramTrainingResult(model=model, vocabulary=vocabulary, loss_history=loss_history)


def save_embedding_artifacts(
    result: SkipGramTrainingResult,
    config: SkipGramConfig,
    paths: EmbeddingArtifactPaths,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save vocabulary, preprocessing metadata, config, and token embeddings."""
    paths.ensure_parent_dirs()
    paths.vocabulary_path.write_text(
        json.dumps(result.vocabulary.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    preprocessing_metadata = {
        "numeric_bins": NumericBinner().to_metadata(),
        "tokenization": {
            "strategy": "column_aware_feature_equals_value",
            "claim_representation": "joined_relational_claim_evidence",
            "excluded_features": ["historical_fraud_label", "investigation_outcome"],
        },
        "training": {"loss_history": result.loss_history},
    }
    if metadata:
        preprocessing_metadata.update(metadata)
    paths.preprocessing_metadata_path.write_text(
        json.dumps(preprocessing_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths.config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(
        {
            "token_embeddings": result.token_embeddings,
            "token_to_id": result.vocabulary.token_to_id,
            "frequencies": result.vocabulary.frequencies,
            "config": config.to_dict(),
        },
        paths.token_embeddings_path,
    )


def load_token_embedding_artifact(path) -> tuple[torch.Tensor, Vocabulary]:
    """Load token embeddings and vocabulary from a saved training artifact."""
    artifact = torch.load(path, map_location="cpu")
    return (
        artifact["token_embeddings"].detach().cpu(),
        Vocabulary(
            token_to_id=dict(artifact["token_to_id"]),
            frequencies=[int(value) for value in artifact["frequencies"]],
        ),
    )
