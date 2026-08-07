"""Semantic claim retrieval over pgvector-indexed claim embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.repositories import ClaimEvidence, ClaimRepository
from database.schema import CLAIM_VECTOR_DIMENSION
from embedding_model.features import build_claim_features
from embedding_model.tokenizer import tokenize_claim_features


class SemanticSearchSettings(BaseSettings):
    """Environment-backed settings for semantic claim retrieval."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    default_top_k: int = Field(default=10, alias="SEMANTIC_SEARCH_DEFAULT_TOP_K")
    max_top_k: int = Field(default=50, alias="SEMANTIC_SEARCH_MAX_TOP_K")


class SemanticSearchError(Exception):
    """Base class for semantic retrieval failures."""


class ClaimNotFoundError(SemanticSearchError):
    """Raised when the source claim does not exist."""


class ClaimEmbeddingMissingError(SemanticSearchError):
    """Raised when the source claim has not been indexed with a vector."""


class InvalidClaimVectorError(SemanticSearchError):
    """Raised when a query vector does not match the configured dimension."""


class NoIndexedClaimsError(SemanticSearchError):
    """Raised when pgvector search returns no indexed candidate claims."""


@dataclass(frozen=True)
class SimilarClaim:
    """Ranked candidate claim returned by semantic vector retrieval."""

    claim_id: str
    similarity_score: float
    evidence: ClaimEvidence | None = None
    shared_tokens: list[str] = field(default_factory=list)
    shared_entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimilarClaimSearchResult:
    """Complete semantic search result for one source claim."""

    source_claim_id: str
    source_evidence: ClaimEvidence
    matches: list[SimilarClaim]


def resolve_top_k(top_k: int | None, settings: SemanticSearchSettings | None = None) -> int:
    """Apply semantic search defaults and maximum result limits."""
    settings = settings or SemanticSearchSettings()
    resolved = settings.default_top_k if top_k is None else int(top_k)
    if resolved <= 0:
        raise ValueError("top_k must be positive")
    return min(resolved, settings.max_top_k)


def validate_query_vector(
    query_vector: Sequence[float],
    expected_dimension: int = CLAIM_VECTOR_DIMENSION,
) -> None:
    """Validate vector dimension before using pgvector search."""
    if len(query_vector) != expected_dimension:
        raise InvalidClaimVectorError(
            f"Query vector has dimension {len(query_vector)}; expected {expected_dimension}"
        )


def vector_to_pgvector_literal(query_vector: Sequence[float]) -> str:
    """Serialize a Python vector into pgvector's text literal format."""
    return "[" + ",".join(str(float(value)) for value in query_vector) + "]"


def build_vector_search_sql(exclude_source_claim: bool = True):
    """Build the pgvector cosine-distance query used for candidate retrieval."""
    exclusions = "AND claim_id <> :exclude_claim_id" if exclude_source_claim else ""
    return text(
        f"""
        SELECT
            claim_id,
            1 - (claim_embedding <=> CAST(:query_vector AS vector)) AS similarity_score
        FROM claims
        WHERE claim_embedding IS NOT NULL
        {exclusions}
        ORDER BY claim_embedding <=> CAST(:query_vector AS vector), claim_id
        LIMIT :top_k
        """
    )


def search_similar_to_vector(
    session: Session,
    query_vector: list[float],
    top_k: int = 10,
    exclude_claim_id: str | None = None,
    settings: SemanticSearchSettings | None = None,
) -> list[SimilarClaim]:
    """Return ranked candidate claim IDs and similarity scores from pgvector."""
    validate_query_vector(query_vector)
    resolved_top_k = resolve_top_k(top_k, settings=settings)
    statement = build_vector_search_sql(exclude_source_claim=exclude_claim_id is not None)
    params = {
        "query_vector": vector_to_pgvector_literal(query_vector),
        "top_k": resolved_top_k,
    }
    if exclude_claim_id is not None:
        params["exclude_claim_id"] = exclude_claim_id

    rows = session.execute(statement, params).all()
    if not rows:
        raise NoIndexedClaimsError("No indexed candidate claims were returned by semantic search")

    return [
        SimilarClaim(claim_id=str(row.claim_id), similarity_score=float(row.similarity_score))
        for row in rows
    ]


def retrieve_candidate_evidence(session: Session, candidate_ids: list[str]) -> list[ClaimEvidence]:
    """Load complete relational evidence for ranked candidate claim IDs."""
    return ClaimRepository(session).list_claim_evidence(candidate_ids)


def find_similar_claims(
    session: Session,
    claim_id: str,
    top_k: int = 10,
    settings: SemanticSearchSettings | None = None,
) -> SimilarClaimSearchResult:
    """Find claims semantically similar to a source claim's stored vector."""
    repository = ClaimRepository(session)
    source_evidence = repository.get_claim_evidence(claim_id)
    if source_evidence is None:
        raise ClaimNotFoundError(f"Claim not found: {claim_id}")

    source_vector = source_evidence.claim.claim_embedding
    if source_vector is None:
        raise ClaimEmbeddingMissingError(f"Claim has no indexed embedding: {claim_id}")

    candidates = search_similar_to_vector(
        session,
        source_vector,
        top_k=top_k,
        exclude_claim_id=claim_id,
        settings=settings,
    )
    candidate_ids = [candidate.claim_id for candidate in candidates]
    candidate_evidence = retrieve_candidate_evidence(session, candidate_ids)
    matches = attach_evidence_and_explanations(source_evidence, candidates, candidate_evidence)
    return SimilarClaimSearchResult(
        source_claim_id=claim_id,
        source_evidence=source_evidence,
        matches=matches,
    )


def attach_evidence_and_explanations(
    source_evidence: ClaimEvidence,
    candidates: Sequence[SimilarClaim],
    evidence_items: Sequence[ClaimEvidence],
) -> list[SimilarClaim]:
    """Attach retrieved evidence and deterministic explanations in candidate order."""
    evidence_by_claim_id = {evidence.claim.claim_id: evidence for evidence in evidence_items}
    matches: list[SimilarClaim] = []
    for candidate in candidates:
        evidence = evidence_by_claim_id.get(candidate.claim_id)
        if evidence is None:
            matches.append(candidate)
            continue
        shared_tokens, shared_entities = explain_similarity(source_evidence, evidence)
        matches.append(
            replace(
                candidate,
                evidence=evidence,
                shared_tokens=shared_tokens,
                shared_entities=shared_entities,
            )
        )
    return matches


def explain_similarity(
    source_evidence: ClaimEvidence,
    candidate_evidence: ClaimEvidence,
) -> tuple[list[str], list[str]]:
    """Return deterministic overlap reasons for two retrieved claims."""
    source_tokens = set(tokenize_claim_features(build_claim_features(source_evidence)))
    candidate_tokens = set(tokenize_claim_features(build_claim_features(candidate_evidence)))
    shared_tokens = sorted(source_tokens.intersection(candidate_tokens))
    shared_entities = sorted(_shared_entity_labels(source_evidence, candidate_evidence))
    return shared_tokens, shared_entities


def _shared_entity_labels(
    source_evidence: ClaimEvidence,
    candidate_evidence: ClaimEvidence,
) -> set[str]:
    labels: set[str] = set()
    if (
        source_evidence.repair_shop is not None
        and candidate_evidence.repair_shop is not None
        and source_evidence.repair_shop.repair_shop_id == candidate_evidence.repair_shop.repair_shop_id
    ):
        labels.add(f"repair_shop_id={source_evidence.repair_shop.repair_shop_id}")

    for bank_account in _shared_values(
        _payment_bank_accounts(source_evidence),
        _payment_bank_accounts(candidate_evidence),
    ):
        labels.add(f"payment_bank_account_reference={bank_account}")
    for phone_number in _shared_values(_phone_numbers(source_evidence), _phone_numbers(candidate_evidence)):
        labels.add(f"phone_number={phone_number}")
    for address in _shared_values(_addresses(source_evidence), _addresses(candidate_evidence)):
        labels.add(f"address={address}")

    if source_evidence.incident.incident_type == candidate_evidence.incident.incident_type:
        labels.add(f"incident_type={source_evidence.incident.incident_type}")
    if source_evidence.incident.incident_city == candidate_evidence.incident.incident_city:
        labels.add(f"incident_city={source_evidence.incident.incident_city}")
    if source_evidence.vehicle.vehicle_type == candidate_evidence.vehicle.vehicle_type:
        labels.add(f"vehicle_type={source_evidence.vehicle.vehicle_type}")
    return labels


def _shared_values(left: Iterable[str | None], right: Iterable[str | None]) -> set[str]:
    left_values = {value for value in left if value}
    right_values = {value for value in right if value}
    return left_values.intersection(right_values)


def _payment_bank_accounts(evidence: ClaimEvidence) -> list[str | None]:
    return [payment.bank_account_reference for payment in evidence.payments]


def _phone_numbers(evidence: ClaimEvidence) -> list[str | None]:
    values = [evidence.customer.phone_number]
    values.extend(participant.phone_number for participant in evidence.participants)
    return values


def _addresses(evidence: ClaimEvidence) -> list[str | None]:
    values = [evidence.customer.address]
    values.extend(participant.address for participant in evidence.participants)
    return values
