"""Application service wrapper around Hugging Face smolagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from agents.falcon_model import FalconSmolagentsModel
from agents.prompts import INVESTIGATION_AGENT_INSTRUCTIONS
from agents.tools import build_agent_tools
from core.settings import AppSettings
from sql_model.inference import FalconSQLGenerator


@dataclass(frozen=True)
class AgentQueryResult:
    """Agent response returned to API callers."""

    answer: str
    framework: str
    model_provider: str
    grounded: bool


class InvestigationAgentService:
    """Construct and run the Falcon-backed smolagents investigation agent."""

    def __init__(
        self,
        session: Session,
        settings: AppSettings | None = None,
        model: Any | None = None,
        sql_generator: FalconSQLGenerator | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or AppSettings()
        self.model = model or FalconSmolagentsModel()
        self.sql_generator = sql_generator

    def build_agent(self):
        """Build a CodeAgent with only project-defined investigation tools."""
        from smolagents import CodeAgent

        return CodeAgent(
            tools=build_agent_tools(self.session, sql_generator=self.sql_generator),
            model=self.model,
            instructions=INVESTIGATION_AGENT_INSTRUCTIONS,
            max_steps=self.settings.agent_max_steps,
            add_base_tools=self.settings.agent_use_base_tools,
        )

    def run(self, query: str) -> AgentQueryResult:
        """Run one natural-language investigation query through the agent."""
        agent = self.build_agent()
        answer = agent.run(query)
        return AgentQueryResult(
            answer=str(answer),
            framework=self.settings.agent_framework,
            model_provider=self.settings.agent_model_provider,
            grounded=self.settings.agent_require_grounded_evidence,
        )
