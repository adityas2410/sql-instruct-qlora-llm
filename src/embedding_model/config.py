"""Configuration for the custom Skip-Gram claim embedding model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class SkipGramConfig:
    """Training hyperparameters for Skip-Gram with Negative Sampling."""

    embedding_dimension: int = 128
    negative_samples: int = 5
    epochs: int = 20
    min_token_frequency: int = 1
    batch_size: int = 512
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    random_seed: int = 2410
    negative_sampling_power: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    """Output locations for generated embedding artifacts."""

    artifact_dir: Path = Path("artifacts/embedding_model")
    vocabulary_path: Path = Path("artifacts/embedding_model/token_vocabulary.json")
    preprocessing_metadata_path: Path = Path("artifacts/embedding_model/preprocessing_metadata.json")
    config_path: Path = Path("artifacts/embedding_model/skipgram_config.json")
    token_embeddings_path: Path = Path("artifacts/embedding_model/token_embeddings.pt")
    claim_vectors_path: Path = Path("artifacts/claim_vectors/claim_vectors.json")

    def ensure_parent_dirs(self) -> None:
        """Create output directories when scripts generate local artifacts."""
        for path in (
            self.vocabulary_path,
            self.preprocessing_metadata_path,
            self.config_path,
            self.token_embeddings_path,
            self.claim_vectors_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


class EmbeddingModelSettings(BaseSettings):
    """Environment-backed settings for claim embedding training and indexing."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_dimension: int = Field(default=128, alias="EMBEDDING_DIMENSION")
    negative_samples: int = Field(default=5, alias="EMBEDDING_NEGATIVE_SAMPLES")
    epochs: int = Field(default=20, alias="EMBEDDING_EPOCHS")
    min_token_frequency: int = Field(default=1, alias="EMBEDDING_MIN_TOKEN_FREQUENCY")
    batch_size: int = Field(default=512, alias="EMBEDDING_BATCH_SIZE")
    learning_rate: float = Field(default=1e-3, alias="EMBEDDING_LEARNING_RATE")
    random_seed: int = Field(default=2410, alias="EMBEDDING_RANDOM_SEED")
    artifact_dir: Path = Field(default=Path("artifacts/embedding_model"), alias="EMBEDDING_ARTIFACT_DIR")
    vocabulary_path: Path | None = Field(default=None, alias="EMBEDDING_VOCAB_PATH")
    preprocessing_metadata_path: Path | None = Field(
        default=None,
        alias="EMBEDDING_PREPROCESSING_METADATA_PATH",
    )
    config_path: Path | None = Field(default=None, alias="EMBEDDING_CONFIG_PATH")
    token_embeddings_path: Path | None = Field(default=None, alias="EMBEDDING_TOKEN_EMBEDDINGS_PATH")
    claim_vectors_path: Path = Field(
        default=Path("artifacts/claim_vectors/claim_vectors.json"),
        alias="CLAIM_VECTOR_OUTPUT_PATH",
    )

    def to_training_config(self) -> SkipGramConfig:
        """Build Skip-Gram training config from environment settings."""
        return SkipGramConfig(
            embedding_dimension=self.embedding_dimension,
            negative_samples=self.negative_samples,
            epochs=self.epochs,
            min_token_frequency=self.min_token_frequency,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
        )

    def to_artifact_paths(self) -> EmbeddingArtifactPaths:
        """Build artifact paths, deriving defaults from EMBEDDING_ARTIFACT_DIR."""
        return EmbeddingArtifactPaths(
            artifact_dir=self.artifact_dir,
            vocabulary_path=self.vocabulary_path or self.artifact_dir / "token_vocabulary.json",
            preprocessing_metadata_path=(
                self.preprocessing_metadata_path
                or self.artifact_dir / "preprocessing_metadata.json"
            ),
            config_path=self.config_path or self.artifact_dir / "skipgram_config.json",
            token_embeddings_path=(
                self.token_embeddings_path or self.artifact_dir / "token_embeddings.pt"
            ),
            claim_vectors_path=self.claim_vectors_path,
        )
