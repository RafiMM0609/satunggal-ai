"""
ResponderAgent – general-purpose conversational agent.

Handles:  general_inquiry, product_question, complaint, order_status,
          technical_support, billing, image_query, unknown
Uses:     Conversation history + LLM to produce a contextual reply.

Note: This agent does NOT use Tavily web search. Live web search is
exclusively available to ResearcherAgent (triggered by the 'research' intent).
"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Kamu adalah asisten AI yang ramah, profesional, dan membantu.
Jawab pertanyaan pengguna secara jelas dan ringkas.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
Jika kamu tidak tahu jawabannya, katakan dengan jujur.
"""

_SYSTEM_PROMPT_OFFICE = """\
Kamu adalah asisten AI yang gaul, santai, dan asik diajak ngobrol — khusus buat urusan kantor dan kerjaan.
Jawab pertanyaan pengguna secara jelas tapi tetap santai dan informal.
Sapa user sebagai "boss" sesekali biar lebih akrab.
Pakai kata-kata gaul Indonesia yang wajar: "nih", "dong", "sih", "yuk", "mantap", "siap", "gaskeun", "okee", dll.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris), tapi tetap dengan nada santai.
Jika kamu tidak tahu jawabannya, bilang aja jujur dengan cara yang santai.
"""


class ResponderAgent(BaseAgent):
    """Generates a conversational reply using history + LLM.

    Does NOT use Tavily web search. For live-search-backed answers,
    the orchestrator should route to ResearcherAgent via the 'research' intent.
    """

    name = "responder"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)

            # Pick system prompt based on active mode
            system_prompt = (
                _SYSTEM_PROMPT_OFFICE
                if task.current_mode == "office"
                else _SYSTEM_PROMPT
            )

            # ── Build message list ─────────────────────────────────────────
            messages = [{"role": "system", "content": system_prompt}]
            # Include at most last 10 messages for context
            messages.extend(history_messages[-10:])
            # Ensure the latest user message is in the list
            if not history_messages or history_messages[-1]["content"] != task.user_input:
                messages.append({"role": "user", "content": task.user_input})

            reply = await self._llm.chat(messages)
            task.mark_done(reply)
            logger.info("Responder done for session=%s", task.session_id)
        except Exception as exc:
            logger.exception("ResponderAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan. Silakan coba lagi."

        return task
