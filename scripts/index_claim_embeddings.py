"""Index pooled claim vectors into PostgreSQL pgvector column."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.session import session_scope
from embedding_model.config import EmbeddingModelSettings
from embedding_model.indexing import index_claim_vectors
from embedding_model.pooling import read_claim_vectors_json


def parse_args() -> argparse.Namespace:
    """Parse claim-vector indexing arguments."""
    settings = EmbeddingModelSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectors",
        type=Path,
        default=settings.claim_vectors_path,
        help="Path to claim_vectors.json produced by train_database_embeddings.py.",
    )
    return parser.parse_args()


def main() -> None:
    """Write pooled claim vectors into claims.claim_embedding."""
    args = parse_args()
    vectors_by_claim_id = read_claim_vectors_json(args.vectors)
    with session_scope() as session:
        indexed_count = index_claim_vectors(session, vectors_by_claim_id)
    print(f"Indexed {indexed_count} claim embeddings from {args.vectors}")


if __name__ == "__main__":
    main()
