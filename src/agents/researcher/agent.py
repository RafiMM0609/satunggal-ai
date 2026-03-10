"""
ResearcherAgent – handles explicit research/investigation requests.

Triggered by the 'research' intent only, which is activated when the user
uses explicit investigative keywords such as:
  - Indonesian: teliti, riset, selidiki, telusuri, analisis mendalam, kaji
  - English:    research, investigate, deep dive, thoroughly analyze

Primary flow (orchestrated):
  The orchestrator runs TavilySearchTool BEFORE calling this agent and stores
  the result in task.tool_results["tavily_search"]["context_text"].  The agent
  reads that pre-fetched context and injects it into the LLM system prompt.

Fallback flow (direct):
  If tool_results["tavily_search"] is absent or empty, the agent falls back
  to calling TavilySearchTool directly so the feature still works.
"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Kamu adalah asisten riset dan teknis yang ahli.
Ketika menjawab pertanyaan teknis atau kompleks:
1. Analisis masalah terlebih dahulu.
2. Pecah menjadi langkah-langkah yang jelas.
3. Berikan jawaban yang komprehensif namun mudah dipahami.
4. Sertakan contoh atau ilustrasi jika diperlukan.
5. Jika pertanyaan di luar kemampuanmu, katakan dengan jujur.

Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
"""

_SYSTEM_PROMPT_WITH_SEARCH = """\
Kamu adalah asisten riset dan teknis yang ahli dengan akses ke data web terkini.
Kamu diberikan hasil pencarian web terbaru sebagai konteks tambahan.
Gunakan informasi tersebut untuk memastikan jawabanmu akurat dan mutakhir.

Ketika menjawab pertanyaan teknis atau kompleks:
1. Analisis masalah terlebih dahulu, manfaatkan konteks pencarian yang tersedia.
2. Pecah menjadi langkah-langkah yang jelas.
3. Berikan jawaban yang komprehensif namun mudah dipahami.
4. Sertakan referensi sumber dari hasil pencarian jika relevan.
5. Jika ada konflik antara informasi lama dan baru, utamakan informasi terbaru.
6. Jika pertanyaan di luar kemampuanmu, katakan dengan jujur.

Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
"""


class ResearcherAgent(BaseAgent):
    """Provides detailed, research-style answers for technical questions.

    Uses Tavily web-search context pre-fetched by the orchestrator
    (task.tool_results["tavily_search"]), with a direct-call fallback.
    """

    name = "researcher"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()

    async def _fetch_web_context_fallback(self, query: str) -> str | None:
        """Direct Tavily call used only when the orchestrator didn't run the tool."""
        try:
            from src.tools.tavily_search import TavilySearchTool  # noqa: PLC0415
            tool     = TavilySearchTool()
            response = await tool.search(query)
            if not response.results:
                return None
            return response.as_context_text()
        except Exception as exc:
            logger.warning("ResearcherAgent fallback Tavily call failed (non-fatal): %s", exc)
            return None

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)

            # ── Resolve Tavily context ─────────────────────────────────────
            # Prefer pre-fetched context from the orchestrator tool execution.
            tavily_tr    = task.tool_results.get("tavily_search", {})
            web_context  = tavily_tr.get("context_text") or None

            if web_context:
                logger.debug(
                    "ResearcherAgent: using orchestrator-provided Tavily context (%d chars)",
                    len(web_context),
                )
            else:
                # Fallback: call Tavily directly (e.g. key available but tool not in tools list)
                logger.info(
                    "ResearcherAgent: no pre-fetched Tavily context – attempting direct call "
                    "for session=%s", task.session_id,
                )
                web_context = await self._fetch_web_context_fallback(task.user_input)

            if web_context:
                system_content = _SYSTEM_PROMPT_WITH_SEARCH + "\n\n" + web_context
            else:
                system_content = _SYSTEM_PROMPT

            # ── Build message list ─────────────────────────────────────────
            messages = [{"role": "system", "content": system_content}]
            messages.extend(history_messages[-8:])
            if not history_messages or history_messages[-1]["content"] != task.user_input:
                messages.append({"role": "user", "content": task.user_input})

            logger.debug(
                "ResearcherAgent sending messages (count=%d) for session=%s",
                len(messages), task.session_id,
            )
            reply = await self._llm.chat(messages, max_tokens=2048)
            logger.debug("ResearcherAgent raw reply: %s", reply)

            task.mark_done(reply)
            logger.info(
                "Researcher done for session=%s (web_search=%s)",
                task.session_id, web_context is not None,
            )
        except Exception as exc:
            logger.exception("ResearcherAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

        return task

