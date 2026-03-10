"""
AgentRouter – maps intent categories to specialist agents.

Add new intents here as you add more agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.gatekeeper.schemas import IntentCategory
from src.memory.state import AgentTask

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── Intent → agent-name mapping ───────────────────────────────────────────────
#
# To add a new agent:
#   1. Create src/agents/<your_agent>/agent.py
#   2. Register it in main_loop.py's _build_agents()
#   3. Map its intents here
#
INTENT_AGENT_MAP: dict[IntentCategory, str] = {
    IntentCategory.GENERAL_INQUIRY:   "responder",
    IntentCategory.PRODUCT_QUESTION:  "responder",
    IntentCategory.COMPLAINT:         "responder",
    IntentCategory.ORDER_STATUS:      "responder",
    IntentCategory.BILLING:           "responder",
    IntentCategory.UNKNOWN:           "responder",
    IntentCategory.TECHNICAL_SUPPORT: "responder",
    IntentCategory.IMAGE_QUERY:       "responder",
    IntentCategory.RESEARCH:          "researcher",   #← only intent with live Tavily access
    IntentCategory.CONTENT_CREATION:  "content_creator",
    IntentCategory.DATA_ANALYSIS:     "wbs_agent",
    IntentCategory.MANDAYS_PLANNING:  "mandays_agent",
}


class AgentRouter:
    """Resolves an AgentTask to the correct BaseAgent instance."""

    def __init__(self, agents: dict[str, "BaseAgent"]) -> None:
        self._agents = agents

    def resolve(self, task: AgentTask) -> "BaseAgent":
        """
        Pick the agent for task.intent.

        Falls back to 'responder' if the intent has no mapping
        or the mapped agent is not registered.
        """
        intent = IntentCategory(task.intent or IntentCategory.UNKNOWN.value)
        agent_name = INTENT_AGENT_MAP.get(intent, "responder")
        agent = self._agents.get(agent_name) or self._agents["responder"]

        logger.info(
            "Router: session=%s intent=%s → agent=%s",
            task.session_id,
            intent.value,
            agent.name,
        )
        return agent
