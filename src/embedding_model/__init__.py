"""Custom Skip-Gram claim embedding pipeline."""

from .config import EmbeddingModelSettings, SkipGramConfig
from .features import FeatureContext, StructuredClaimFeatures, build_claim_features
from .indexing import index_claim_vectors, validate_claim_vector
from .pooling import pool_claim_vector, pool_claim_vectors
from .skipgram import SkipGramModel, TrainingPair, build_vocabulary, create_skipgram_pairs
from .tokenizer import tokenize_claim_features

__all__ = [
    "EmbeddingModelSettings",
    "FeatureContext",
    "SkipGramConfig",
    "SkipGramModel",
    "StructuredClaimFeatures",
    "TrainingPair",
    "build_claim_features",
    "build_vocabulary",
    "create_skipgram_pairs",
    "index_claim_vectors",
    "pool_claim_vector",
    "pool_claim_vectors",
    "tokenize_claim_features",
    "validate_claim_vector",
]
