"""SQL generation and execution routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from schemas import SQLExecuteRequest, SQLExecuteResponse, SQLGenerateRequest, SQLGenerateResponse
from services.sql_execution import SQLSafetyError, execute_readonly_sql
from sql_model.inference import FalconSQLGenerator

router = APIRouter(prefix="/sql", tags=["sql"])


@router.post("/generate", response_model=SQLGenerateResponse)
def generate_sql(request: SQLGenerateRequest) -> SQLGenerateResponse:
    """Generate read-only PostgreSQL from a natural-language request."""
    result = FalconSQLGenerator().generate_sql(
        instruction=request.instruction,
        schema=request.schema_text,
    )
    return SQLGenerateResponse(
        sql=result.sql,
        base_model_id=result.base_model_id,
        adapter_dir=result.adapter_dir,
        prompt_characters=result.prompt_characters,
    )


@router.post("/execute", response_model=SQLExecuteResponse)
def execute_sql(
    request: SQLExecuteRequest,
    session: Session = Depends(get_db_session),
) -> SQLExecuteResponse:
    """Execute one validated read-only SQL statement."""
    try:
        result = execute_readonly_sql(session, request.sql)
    except SQLSafetyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SQLExecuteResponse(sql=result.sql, rows=result.rows, row_count=result.row_count)
