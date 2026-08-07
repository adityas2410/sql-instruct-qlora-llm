"""Train custom Skip-Gram claim embeddings from PostgreSQL evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.models import Claim
from database.repositories import ClaimRepository
from database.session import session_scope
from embedding_model.config import EmbeddingModelSettings
from embedding_model.features import FeatureContext, build_claim_features
from embedding_model.pooling import pool_claim_vectors, write_claim_vectors_json
from embedding_model.tokenizer import tokenize_claim_features
from embedding_model.training import save_embedding_artifacts, train_skipgram_model


def parse_args() -> argparse.Namespace:
    """Parse embedding training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of claims to load for training.",
    )
    parser.add_argument(
        "--claim-vectors-output",
        type=Path,
        default=None,
        help="Override path for pooled claim vectors JSON.",
    )
    return parser.parse_args()


def load_claim_evidence(limit: int | None = None):
    """Load complete relational evidence for all claims in stable order."""
    with session_scope() as session:
        statement = select(Claim.claim_id).order_by(Claim.claim_date, Claim.claim_id)
        if limit is not None:
            statement = statement.limit(limit)
        claim_ids = list(session.execute(statement).scalars().all())
        return ClaimRepository(session).list_claim_evidence(claim_ids)


def main() -> None:
    """Train token embeddings, pool claim vectors, and save local artifacts."""
    args = parse_args()
    settings = EmbeddingModelSettings()
    config = settings.to_training_config()
    paths = settings.to_artifact_paths()
    if args.claim_vectors_output is not None:
        paths = type(paths)(
            artifact_dir=paths.artifact_dir,
            vocabulary_path=paths.vocabulary_path,
            preprocessing_metadata_path=paths.preprocessing_metadata_path,
            config_path=paths.config_path,
            token_embeddings_path=paths.token_embeddings_path,
            claim_vectors_path=args.claim_vectors_output,
        )

    evidence_items = load_claim_evidence(limit=args.limit)
    context = FeatureContext.from_evidence(evidence_items)
    tokens_by_claim = {
        evidence.claim.claim_id: tokenize_claim_features(build_claim_features(evidence, context))
        for evidence in evidence_items
    }

    result = train_skipgram_model(tokens_by_claim, config)
    save_embedding_artifacts(
        result,
        config,
        paths,
        metadata={"claim_count": len(tokens_by_claim), "vocabulary_size": len(result.vocabulary)},
    )
    vectors_by_claim_id, skipped_claim_ids = pool_claim_vectors(
        tokens_by_claim,
        result.token_embeddings,
        result.vocabulary,
    )
    write_claim_vectors_json(vectors_by_claim_id, paths.claim_vectors_path)
    print(
        "Trained claim embedding model: "
        f"claims={len(tokens_by_claim)}, vocabulary={len(result.vocabulary)}, "
        f"vectors={len(vectors_by_claim_id)}, skipped={len(skipped_claim_ids)}"
    )


if __name__ == "__main__":
    main()
