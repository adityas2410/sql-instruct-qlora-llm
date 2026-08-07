"""Semantic claim retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from schemas import SimilarClaimResponse, SimilarClaimsResponse
from services.semantic_search import SemanticSearchError, find_similar_claims

router = APIRouter(prefix="/semantic", tags=["semantic"])


@router.post("/claims/{claim_id}", response_model=SimilarClaimsResponse)
def similar_claims(
    claim_id: str,
    top_k: int = Query(default=10, gt=0),
    session: Session = Depends(get_db_session),
) -> SimilarClaimsResponse:
    """Return semantically similar claims for one source claim."""
    try:
        result = find_similar_claims(session, claim_id=claim_id, top_k=top_k)
    except SemanticSearchError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SimilarClaimsResponse(
        source_claim_id=result.source_claim_id,
        matches=[
            SimilarClaimResponse(
                claim_id=match.claim_id,
                similarity_score=match.similarity_score,
                shared_tokens=match.shared_tokens,
                shared_entities=match.shared_entities,
                historical_fraud_label=(
                    match.evidence.claim.historical_fraud_label if match.evidence else None
                ),
                investigation_outcome=(
                    match.evidence.claim.investigation_outcome if match.evidence else None
                ),
            )
            for match in result.matches
        ],
    )
