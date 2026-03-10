"""
GatekeeperAgent – classifies user intent and routes the task.

This is a *classifier*, not a full response-generator.
It is called by the orchestrator BEFORE routing to a specialist agent.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from src.agents.gatekeeper.openrouter import OpenRouterClient
from src.agents.gatekeeper.schemas import IntentCategory, IntentResult

logger = logging.getLogger(__name__)


class GatekeeperAgent:
    """Stateful classifier; holds an LLM client reference."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings   = settings or get_settings()
        self._llm_client = OpenRouterClient(self._settings)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def classify_intent(self, text: str, session_id: str = "") -> IntentResult:
        """
        Classify a single text string.

        Args:
            text:       Raw user text.
            session_id: Optional identifier for logging.

        Returns:
            IntentResult with intent category and confidence score.
        """
        if not text or not text.strip():
            return self._empty_result(session_id)

        normalised = text.strip()
        logger.info("Gatekeeper classifying session=%s text=%.80s…", session_id, normalised)

        llm_response = await self._llm_client.classify_intent(normalised)

        result = IntentResult(
            session_id=session_id,
            raw_text=normalised,
            intent=llm_response.intent,
            confidence=llm_response.confidence,
            tools=list(llm_response.tools),
            model_used=llm_response.model_used,
        )
        logger.info(
            "Gatekeeper → intent=%s confidence=%.2f",
            result.intent.value,
            result.confidence,
        )
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        await self._llm_client.aclose()

    async def __aenter__(self) -> "GatekeeperAgent":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(session_id: str) -> IntentResult:
        return IntentResult(
            session_id=session_id,
            raw_text="",
            intent=IntentCategory.UNKNOWN,
            confidence=0.0,
            metadata={"reason": "empty_input"},
        )
