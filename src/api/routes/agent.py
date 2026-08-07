"""Agentic investigation route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from agents import InvestigationAgentService
from api.dependencies import get_db_session
from schemas import AgentQueryRequest, AgentQueryResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentQueryResponse)
def query_agent(
    request: AgentQueryRequest,
    session: Session = Depends(get_db_session),
) -> AgentQueryResponse:
    """Run a natural-language investigation request through the Falcon agent."""
    result = InvestigationAgentService(session).run(request.query)
    return AgentQueryResponse(
        answer=result.answer,
        framework=result.framework,
        model_provider=result.model_provider,
        grounded=result.grounded,
    )
