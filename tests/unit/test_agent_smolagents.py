"""Unit tests for Hugging Face smolagents integration."""

from __future__ import annotations

from types import SimpleNamespace

from agents.falcon_model import FalconAgentMessage, FalconSmolagentsModel
from agents.prompts import INVESTIGATION_AGENT_INSTRUCTIONS
from agents.service import InvestigationAgentService


class FakeAgent:
    """Minimal stand-in for smolagents.CodeAgent."""

    captured_kwargs = None

    def __init__(self, **kwargs) -> None:
        FakeAgent.captured_kwargs = kwargs

    def run(self, query: str) -> str:
        return f"grounded answer for {query}"


def test_agent_instructions_include_investigation_boundaries() -> None:
    """Agent prompt requires grounded evidence and safe SQL behavior."""
    assert "provided project tools" in INVESTIGATION_AGENT_INSTRUCTIONS
    assert "read-only SQL" in INVESTIGATION_AGENT_INSTRUCTIONS
    assert "Historical fraud labels" in INVESTIGATION_AGENT_INSTRUCTIONS
    assert "grounded in returned tool evidence" in INVESTIGATION_AGENT_INSTRUCTIONS


def test_falcon_smolagents_wrapper_can_be_constructed_without_loading_weights() -> None:
    """The wrapper is lazy and can be mocked for tests/API wiring."""
    model = FalconSmolagentsModel()
    assert model._model is None
    assert model._tokenizer is None


def test_falcon_wrapper_returns_message_when_generate_text_is_mocked(monkeypatch) -> None:
    """The model wrapper exposes a smolagents-compatible callable shape."""
    model = FalconSmolagentsModel()
    monkeypatch.setattr(model, "generate_text", lambda prompt, stop_sequences=None: "assistant text")
    response = model([{"role": "user", "content": "hello"}])
    assert getattr(response, "content", None) == "assistant text"


def test_agent_service_builds_code_agent_with_project_tools(monkeypatch) -> None:
    """The service disables base tools and passes bounded steps."""
    import agents.service as service_module

    monkeypatch.setattr(service_module, "CodeAgent", FakeAgent, raising=False)
    monkeypatch.setattr(
        service_module,
        "build_agent_tools",
        lambda session, sql_generator=None: [lambda: "tool"],
    )

    class Settings:
        agent_max_steps = 6
        agent_use_base_tools = False
        agent_framework = "huggingface_smolagents"
        agent_model_provider = "local_falcon_adapter"
        agent_require_grounded_evidence = True

    service = InvestigationAgentService(
        session=SimpleNamespace(),
        settings=Settings(),
        model=FalconAgentMessage(role="assistant", content="mock"),
    )
    monkeypatch.setitem(__import__("sys").modules, "smolagents", SimpleNamespace(CodeAgent=FakeAgent))
    result = service.run("show similar claims")

    assert result.answer == "grounded answer for show similar claims"
    assert FakeAgent.captured_kwargs["max_steps"] == 6
    assert FakeAgent.captured_kwargs["add_base_tools"] is False
    assert FakeAgent.captured_kwargs["tools"]
