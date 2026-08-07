"""Inspect semantically similar claims from indexed PostgreSQL claim vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.session import session_scope
from services.semantic_search import SimilarClaimSearchResult, find_similar_claims


def parse_args() -> argparse.Namespace:
    """Parse semantic search inspection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim_id", help="Source claim ID with an indexed claim_embedding vector.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum similar claims to return.")
    return parser.parse_args()


def serialize_result(result: SimilarClaimSearchResult) -> dict[str, object]:
    """Convert service output to compact JSON for inspection."""
    return {
        "source_claim_id": result.source_claim_id,
        "matches": [
            {
                "claim_id": match.claim_id,
                "similarity_score": match.similarity_score,
                "shared_tokens": match.shared_tokens,
                "shared_entities": match.shared_entities,
                "historical_fraud_label": (
                    match.evidence.claim.historical_fraud_label if match.evidence else None
                ),
                "investigation_outcome": (
                    match.evidence.claim.investigation_outcome if match.evidence else None
                ),
            }
            for match in result.matches
        ],
    }


def main() -> None:
    """Run backend semantic retrieval for one source claim."""
    args = parse_args()
    with session_scope() as session:
        result = find_similar_claims(session, args.claim_id, top_k=args.top_k)
    print(json.dumps(serialize_result(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
