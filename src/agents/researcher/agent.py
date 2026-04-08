"""
ResearcherAgent – handles explicit research/investigation requests.

Triggered by the 'research' intent only, which is activated when the user
uses explicit investigative keywords such as:
  - Indonesian: teliti, riset, selidiki, telusuri, analisis mendalam, kaji
  - English:    research, investigate, deep dive, thoroughly analyze

Enhanced multi-point search flow:
  1. The user's input is decomposed by the LLM into 3–4 focused sub-queries
     that together cover the topic from different angles.
  2. TavilySearchTool is called in parallel for each sub-query so the agent
     accumulates richer, more diverse context than a single search would yield.
  3. All results are merged and injected into the LLM system prompt.

Fallback chain (when Tavily is unavailable or returns no results):
  - Multi-point search → orchestrator pre-fetched context → no-search prompt.
"""

from __future__ import annotations

import asyncio
import logging
import re

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

Panduan format (output dikirim ke Telegram):
- Gunakan header markdown (## Judul Bagian) untuk setiap bagian utama.
- Gunakan bullet point (-) untuk daftar; hindari paragraf yang terlalu panjang.
- Pisahkan setiap bagian dengan baris kosong agar mudah dibaca.
- Pastikan jawabanmu **lengkap dan tidak terpotong**; selesaikan setiap kalimat dan bagian.
- Jangan potong di tengah penjelasan – jika topik luas, ringkas setiap bagian secara padat.

Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
"""

_SYSTEM_PROMPT_WITH_SEARCH = """\
Kamu adalah asisten riset dan teknis yang ahli dengan akses ke data web terkini.
Kamu diberikan hasil pencarian web dari beberapa poin penelusuran yang telah diuraikan \
dari pertanyaan pengguna, sehingga konteks yang tersedia lebih mendalam dan beragam.
Gunakan seluruh informasi tersebut untuk memastikan jawabanmu akurat, komprehensif, dan mutakhir.

Ketika menjawab pertanyaan teknis atau kompleks:
1. Analisis masalah terlebih dahulu, manfaatkan seluruh konteks pencarian yang tersedia.
2. Pecah menjadi langkah-langkah yang jelas.
3. Berikan jawaban yang komprehensif namun mudah dipahami.
4. Sertakan referensi sumber dari hasil pencarian jika relevan.
5. Jika ada konflik antara informasi lama dan baru, utamakan informasi terbaru.
6. Jika pertanyaan di luar kemampuanmu, katakan dengan jujur.

Panduan format (output dikirim ke Telegram):
- Gunakan header markdown (## Judul Bagian) untuk setiap bagian utama.
- Gunakan bullet point (-) untuk daftar; hindari paragraf yang terlalu panjang.
- Pisahkan setiap bagian dengan baris kosong agar mudah dibaca.
- Pastikan jawabanmu **lengkap dan tidak terpotong**; selesaikan setiap kalimat dan bagian.
- Jangan potong di tengah penjelasan – jika topik luas, ringkas setiap bagian secara padat.
- Cantumkan referensi URL di bagian akhir, bukan di tengah teks.

Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
"""

_DECOMPOSE_SYSTEM_PROMPT = """\
Kamu adalah asisten yang bertugas menguraikan sebuah pertanyaan atau topik riset menjadi \
beberapa poin pencarian yang spesifik dan saling melengkapi.
Tujuannya adalah mendapatkan informasi yang komprehensif dari berbagai sudut pandang.

Aturan ketat:
- Berikan tepat 3 hingga 4 query pencarian yang berbeda dan spesifik.
- Setiap query harus bisa langsung digunakan di mesin pencari (Google, Bing, dll).
- Setiap baris query WAJIB diawali dengan prefix "QUERY: " (tanpa tanda kutip).
- Jangan tulis penjelasan, komentar, penomoran, atau baris lain selain baris QUERY.
- Jangan gunakan baris kosong di antara query.
- Gunakan bahasa yang sama dengan input pengguna.

Contoh format output yang benar:
QUERY: lowongan kerja Astra International 2024
QUERY: rekrutmen fresh graduate Astra Group
QUERY: cara daftar kerja di Astra online
"""

_PLAN_SYSTEM_PROMPT = """\
Kamu adalah perencana riset yang bertugas membuat kerangka penelitian terstruktur.
Berdasarkan pertanyaan pengguna dan konteks informasi yang tersedia, buat rencana riset \
berupa daftar bagian/topik yang harus dijawab secara menyeluruh dalam jawaban akhir.

Aturan ketat:
- Buat 4–7 bagian utama yang mencakup topik secara komprehensif.
- Setiap bagian diawali dengan angka dan titik (contoh: 1. Judul Bagian).
- Di bawah setiap bagian, tuliskan 2–3 sub-poin yang harus dibahas, diawali dengan tanda (-).
- Jangan jawab pertanyaannya sekarang – hanya buat kerangka rencana.
- Gunakan bahasa yang sama dengan pertanyaan pengguna.
"""


class ResearcherAgent(BaseAgent):
    """Provides detailed, research-style answers for technical questions.

    Decomposes the user's query into multiple focused sub-queries, runs them
    through Tavily in parallel to gather rich context, then feeds the combined
    results to the LLM for a comprehensive answer.
    """

    name = "researcher"

    _MAX_SUB_QUERIES      = 4    # hard cap on parallel Tavily calls per request
    _DECOMPOSE_MAX_TOKENS = 1024  # reasoning models need budget before producing answers
    _PLAN_MAX_TOKENS      = 512  # budget for building the research outline
    _PLAN_CONTEXT_MAX_CHARS = 3000  # preview fed to planner – keeps planning call cheap
    _MAX_HISTORY_MESSAGES = 8    # keep last N turns in context window

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()

    async def _decompose_query(self, query: str) -> list[str]:
        """Use the LLM to break the user query into focused search sub-queries.

        Returns a list of 3–4 sub-queries on success, or ``[query]`` on failure
        so the caller always has at least one query to search.
        """
        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Uraikan topik/pertanyaan berikut menjadi poin-poin pencarian:\n\n"
                    + query
                ),
            },
        ]
        try:
            response = await self._llm.chat(messages, max_tokens=self._DECOMPOSE_MAX_TOKENS)

            # Strip reasoning/thinking blocks emitted by reasoning models
            # e.g. <think>...</think> or <thinking>...</thinking>
            cleaned = re.sub(r"<think(?:ing)?>[^<]*(?:<(?!/think(?:ing)?>)[^<]*)*</think(?:ing)?>", "", response, flags=re.DOTALL | re.IGNORECASE)

            # Extract only lines that start with "QUERY:" prefix
            sub_queries = [
                line[len("QUERY:"):].strip()
                for line in cleaned.splitlines()
                if line.strip().upper().startswith("QUERY:")
            ]
            sub_queries = sub_queries[:self._MAX_SUB_QUERIES]  # hard cap
            logger.debug("ResearcherAgent decomposed query into: %s", sub_queries)
            return sub_queries if sub_queries else [query]
        except Exception as exc:
            logger.warning("Query decomposition failed (non-fatal): %s", exc)
            return [query]

    async def _plan_research(self, query: str, web_context: str | None) -> str | None:
        """Generate a structured research outline from the query and available web context.

        Returns a numbered plan string on success, or ``None`` on failure so the
        caller can proceed without a plan (graceful degradation).
        """
        context_snippet = ""
        if web_context:
            # Feed only a trimmed preview to keep the planning call cheap.
            context_snippet = "\n\nKonteks informasi yang tersedia (ringkasan):\n" + web_context[:self._PLAN_CONTEXT_MAX_CHARS]

        messages = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Buat rencana riset untuk pertanyaan berikut:\n\n"
                    + query
                    + context_snippet
                ),
            },
        ]
        try:
            plan = await self._llm.chat(messages, max_tokens=self._PLAN_MAX_TOKENS)
            plan = plan.strip()
            logger.debug("ResearcherAgent research plan:\n%s", plan)
            return plan if plan else None
        except Exception as exc:
            logger.warning("Research planning failed (non-fatal): %s", exc)
            return None

    async def _search_sub_queries(self, sub_queries: list[str]) -> str | None:
        """Run Tavily in parallel for every sub-query and merge the results.

        Returns a combined context string, or ``None`` when Tavily is
        unavailable or all searches returned empty results.
        """
        try:
            from src.tools.tavily_search import TavilySearchTool  # noqa: PLC0415
            tool = TavilySearchTool()

            responses = await asyncio.gather(
                *(tool.search(q) for q in sub_queries),
                return_exceptions=True,
            )

            parts: list[str] = []
            for sub_query, resp in zip(sub_queries, responses):
                if isinstance(resp, Exception):
                    logger.warning(
                        "Sub-query search failed (skipped) query=%r: %s", sub_query, resp
                    )
                    continue
                if resp.results:
                    parts.append(resp.as_context_text())

            return "\n\n---\n\n".join(parts) if parts else None
        except Exception as exc:
            logger.warning("Multi-point Tavily search failed (non-fatal): %s", exc)
            return None

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)

            # ── Step 1: Decompose query into focused sub-points ───────────────
            logger.info(
                "ResearcherAgent: decomposing query for session=%s", task.session_id
            )
            sub_queries = await self._decompose_query(task.user_input)
            logger.info(
                "ResearcherAgent: %d sub-queries generated for session=%s: %s",
                len(sub_queries), task.session_id, sub_queries,
            )

            # ── Step 2: Multi-point Tavily search ─────────────────────────────
            web_context = await self._search_sub_queries(sub_queries)

            if web_context:
                logger.debug(
                    "ResearcherAgent: multi-point search yielded %d chars for session=%s",
                    len(web_context), task.session_id,
                )
            else:
                # Fallback 1: use orchestrator pre-fetched context (single-query search)
                logger.info(
                    "ResearcherAgent: multi-point search empty – falling back to "
                    "orchestrator pre-fetched context for session=%s", task.session_id,
                )
                tavily_tr   = task.tool_results.get("tavily_search", {})
                web_context = tavily_tr.get("context_text") or None

            # ── Step 3: Build research plan (outline) ─────────────────────────
            research_plan = await self._plan_research(task.user_input, web_context)
            if research_plan:
                logger.info(
                    "ResearcherAgent: research plan built (%d chars) for session=%s",
                    len(research_plan), task.session_id,
                )

            # ── Step 4: Build system prompt with context + plan ───────────────
            if web_context:
                system_content = _SYSTEM_PROMPT_WITH_SEARCH + "\n\n" + web_context
            else:
                system_content = _SYSTEM_PROMPT

            if research_plan:
                system_content += (
                    "\n\n## Rencana Riset (ikuti urutan dan cakupan ini persis)\n\n"
                    + research_plan
                    + "\n\nPastikan setiap bagian dalam rencana di atas dijawab secara lengkap dan berurutan."
                )

            # ── Step 5: Build message list ────────────────────────────────────
            messages = [{"role": "system", "content": system_content}]
            messages.extend(history_messages[-self._MAX_HISTORY_MESSAGES:])
            if not history_messages or history_messages[-1]["content"] != task.user_input:
                messages.append({"role": "user", "content": task.user_input})

            logger.debug(
                "ResearcherAgent sending messages (count=%d) for session=%s",
                len(messages), task.session_id,
            )
            reply = await self._llm.chat(messages, max_tokens=8192)
            logger.debug("ResearcherAgent raw reply: %s", reply)

            task.mark_done(reply)
            logger.info(
                "Researcher done for session=%s (web_search=%s, sub_queries=%d, plan=%s)",
                task.session_id, web_context is not None, len(sub_queries),
                research_plan is not None,
            )
        except Exception as exc:
            logger.exception("ResearcherAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

        return task

