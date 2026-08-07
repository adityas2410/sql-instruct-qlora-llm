"""Frequency-based negative sampling for SGNS training."""

from __future__ import annotations

import torch

from .skipgram import Vocabulary


class NegativeSampler:
    """Sample negative context IDs from smoothed token frequencies."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        power: float = 0.75,
        seed: int = 2410,
    ) -> None:
        if len(vocabulary) == 0:
            raise ValueError("Cannot sample negatives from an empty vocabulary")
        weights = torch.tensor(vocabulary.frequencies, dtype=torch.float)
        weights = torch.pow(weights, power)
        self.probabilities = weights / weights.sum()
        self.generator = torch.Generator().manual_seed(seed)

    def sample(
        self,
        positive_context_ids: torch.Tensor,
        negative_samples: int,
    ) -> torch.Tensor:
        """Sample negatives while avoiding the positive context when possible."""
        if negative_samples <= 0:
            raise ValueError("negative_samples must be positive")
        batch_size = int(positive_context_ids.shape[0])
        vocabulary_size = int(self.probabilities.shape[0])
        samples = torch.multinomial(
            self.probabilities,
            batch_size * negative_samples,
            replacement=True,
            generator=self.generator,
        ).view(batch_size, negative_samples)

        if vocabulary_size > 1:
            positive = positive_context_ids.view(batch_size, 1)
            collisions = samples == positive
            while bool(collisions.any()):
                replacement_count = int(collisions.sum().item())
                samples[collisions] = torch.multinomial(
                    self.probabilities,
                    replacement_count,
                    replacement=True,
                    generator=self.generator,
                )
                collisions = samples == positive
        return samples
