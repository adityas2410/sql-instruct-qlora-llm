"""FastAPI route tests with mocked service boundaries."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.dependencies import get_db_session
from api.main import create_app


class FakeSession:
    pass


def _client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: FakeSession()
    return TestClient(app)


def test_health_route_returns_app_metadata() -> None:
    """Health route exposes app and agent framework metadata."""
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json()["agent_framework"] == "huggingface_smolagents"


def test_sql_generate_route_uses_generator(monkeypatch) -> None:
    """SQL generation route returns Falcon SQL metadata."""
    from api.routes import sql as sql_route

    class FakeGenerator:
        def generate_sql(self, instruction: str, schema: str | None = None):
            return SimpleNamespace(
                sql="SELECT claim_id FROM claims LIMIT 10;",
                base_model_id="tiiuae/falcon-11B",
                adapter_dir="models/sql_lora_adapter",
                prompt_characters=120,
            )

    monkeypatch.setattr(sql_route, "FalconSQLGenerator", lambda: FakeGenerator())
    response = _client().post("/sql/generate", json={"instruction": "show claims"})
    assert response.status_code == 200
    assert response.json()["sql"].startswith("SELECT")


def test_sql_execute_route_uses_execution_service(monkeypatch) -> None:
    """SQL execution route returns materialized rows."""
    from api.routes import sql as sql_route

    monkeypatch.setattr(
        sql_route,
        "execute_readonly_sql",
        lambda session, sql: SimpleNamespace(
            sql="SELECT claim_id FROM claims LIMIT 10",
            rows=[{"claim_id": "CLM-1"}],
            row_count=1,
        ),
    )
    response = _client().post("/sql/execute", json={"sql": "SELECT claim_id FROM claims"})
    assert response.status_code == 200
    assert response.json()["row_count"] == 1


def test_semantic_claim_route_uses_semantic_service(monkeypatch) -> None:
    """Semantic route returns similar claim matches."""
    from api.routes import semantic as semantic_route

    evidence = SimpleNamespace(
        claim=SimpleNamespace(historical_fraud_label=True, investigation_outcome="fraud")
    )
    result = SimpleNamespace(
        source_claim_id="CLM-1",
        matches=[
            SimpleNamespace(
                claim_id="CLM-2",
                similarity_score=0.91,
                shared_tokens=["incident_city=london"],
                shared_entities=["repair_shop_id=RS-1"],
                evidence=evidence,
            )
        ],
    )
    monkeypatch.setattr(semantic_route, "find_similar_claims", lambda session, claim_id, top_k: result)
    response = _client().post("/semantic/claims/CLM-1?top_k=5")
    assert response.status_code == 200
    assert response.json()["matches"][0]["claim_id"] == "CLM-2"


def test_agent_query_route_uses_agent_service(monkeypatch) -> None:
    """Agent route returns a grounded agent response."""
    from api.routes import agent as agent_route

    class FakeAgentService:
        def __init__(self, session) -> None:
            pass

        def run(self, query: str):
            return SimpleNamespace(
                answer="grounded answer",
                framework="huggingface_smolagents",
                model_provider="local_falcon_adapter",
                grounded=True,
            )

    monkeypatch.setattr(agent_route, "InvestigationAgentService", FakeAgentService)
    response = _client().post("/agent/query", json={"query": "find similar claims"})
    assert response.status_code == 200
    assert response.json()["framework"] == "huggingface_smolagents"
