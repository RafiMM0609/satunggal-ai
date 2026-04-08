"""
GatekeeperAgent – classifies user intent and routes the task.

This is a *classifier*, not a full response-generator.
It is called by the orchestrator BEFORE routing to a specialist agent.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from src.agents.gatekeeper.openrouter import GatekeeperLLMClient
from src.agents.gatekeeper.schemas import IntentCategory, IntentResult

logger = logging.getLogger(__name__)


class GatekeeperAgent:
    """Stateful classifier; holds an LLM client reference."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings   = settings or get_settings()
        self._llm_client = GatekeeperLLMClient(self._settings)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def classify_intent(
        self,
        text: str,
        session_id: str = "",
        history: "list[dict] | None" = None,
        allowed_intents: "list[str] | None" = None,
    ) -> IntentResult:
        """
        Classify a single text string.

        Args:
            text:             Raw user text.
            session_id:       Optional identifier for logging.
            history:          Optional list of recent conversation messages in
                              OpenAI chat-completion format (role/content dicts).
                              When provided, these are injected into the LLM
                              system prompt so follow-up commands (e.g.
                              "berikan screenshot" after a web_automation turn)
                              are classified correctly.
            allowed_intents:  Optional list of IntentCategory values (strings)
                              that the Gatekeeper is allowed to return.  When
                              provided, a mode-restriction note is appended to
                              the system prompt so the LLM narrows its choices.

        Returns:
            IntentResult with intent category and confidence score.
        """
        if not text or not text.strip():
            return self._empty_result(session_id)

        normalised = text.strip()
        logger.info("Gatekeeper classifying session=%s text=%.80s…", session_id, normalised)

        llm_response = await self._llm_client.classify_intent(
            normalised, history=history, allowed_intents=allowed_intents
        )

        # ── Self-Correction: fallback clarification question ──────────────
        clarification_question = llm_response.clarification_question
        needs_clarification    = llm_response.needs_clarification

        # Force clarification when intent is UNKNOWN or confidence is very low,
        # even if the LLM forgot to set needs_clarification in its response.
        if (
            llm_response.intent == IntentCategory.UNKNOWN
            or llm_response.confidence < 0.50
        ):
            needs_clarification = True
            if not clarification_question:
                clarification_question = (
                    "Maaf, saya belum sepenuhnya memahami permintaan Anda. "
                    "Boleh Anda jelaskan lebih detail apa yang ingin Anda lakukan? "
                    "Misalnya: membuat dokumen, mencari informasi, menganalisis data, atau kebutuhan lainnya?"
                )

        metadata: dict = {}
        if llm_response.sub_intent:
            metadata["sub_intent"] = llm_response.sub_intent

        result = IntentResult(
            session_id=session_id,
            raw_text=normalised,
            intent=llm_response.intent,
            confidence=llm_response.confidence,
            tools=list(llm_response.tools),
            model_used=llm_response.model_used,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            metadata=metadata,
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
