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
from src.tools.telegram_formatter import sanitize_for_telegram
from src.tools.tavily_search import TavilySearchTool
from src.tools.web_reader import WebReaderTool

logger = logging.getLogger(__name__)

# Matches <think>…</think> or <thinking>…</thinking> blocks produced by reasoning
# models (e.g. DeepSeek R1).  Must be stripped before parsing structured output.
_THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)

# Prefix that every decomposed search query must start with.
_QUERY_PREFIX = "QUERY:"

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

# ── Office-mode addendum: appended to the final-answer prompt only. ───────────
_HERMES_DECISION_PROMPT = """\
Kamu adalah asisten riset mandiri (Hermes Agent) yang melakukan investigasi mendalam secara iteratif.
Tugas kamu adalah menjawab pertanyaan atau melakukan riset untuk pengguna dengan menggunakan alat pencarian (Search) dan pembaca halaman web (Read).

Setiap langkah, kamu harus menganalisis informasi yang sudah didapatkan, menentukan apakah informasi tersebut sudah cukup, dan jika belum, putuskan tindakan berikutnya.

Alat yang tersedia:
1. `search`: Melakukan pencarian web menggunakan kata kunci/query tertentu. Gunakan ini untuk menemukan link atau rangkuman awal.
   Parameter yang dibutuhkan: `query` (string)
2. `read`: Membaca teks lengkap dari halaman web tertentu berdasarkan URL. Gunakan ini setelah menemukan URL yang relevan dari pencarian.
   Parameter yang dibutuhkan: `url` (string)
3. `answer`: Memberikan jawaban/laporan akhir yang komprehensif, terstruktur, dan akurat kepada pengguna berdasarkan riset yang telah kamu lakukan.
   Parameter yang dibutuhkan: `content` (string)

Format Output Wajib:
Kamu harus membalas dalam format JSON yang valid. Jangan sertakan teks lain di luar JSON tersebut.
Struktur JSON:
{
  "thought": "Pemikiranmu tentang apa yang sudah ditemukan, apa yang kurang, dan apa rencana langkah selanjutnya.",
  "action": "Nama tindakan yang dipilih ('search', 'read', atau 'answer').",
  "query": "Query pencarian (hanya diisi jika action adalah 'search').",
  "url": "URL halaman web (hanya diisi jika action adalah 'read').",
  "content": "Jawaban akhir dalam markdown yang rapi, lengkap, dan informatif (hanya diisi jika action adalah 'answer')."
}

PENTING:
- Lakukan riset secara mendalam dan iteratif. Jika pencarian pertama kurang spesifik, lakukan pencarian kedua dengan kata kunci yang lebih tajam.
- Jika ada URL penting dalam hasil pencarian, gunakan tindakan 'read' untuk membaca isinya sebelum membuat kesimpulan.
- Cantumkan sumber/referensi URL di bagian akhir laporan jika kamu memilih tindakan 'answer'.
- Jangan membuat tindakan 'read' atau 'search' berulang-ulang tanpa kemajuan.
- Maksimal langkah riset dibatasi. Jika ini adalah langkah terakhir, kamu WAJIB menggunakan tindakan 'answer'.
"""

# ── Office-mode addendum: appended to the final-answer prompt only. ───────────
# Internal steps (decompose, plan) are unaffected so their strict QUERY: prefix
# parsing keeps working.  We only change the *tone* of the user-facing response.
class ResearcherAgent(BaseAgent):
    """Provides detailed, research-style answers for technical questions.

    Decomposes the user's query into multiple focused sub-queries, runs them
    through Tavily in parallel to gather rich context, then feeds the combined
    results to the LLM for a comprehensive answer.
    """

    name = "researcher"

    # ── Persona ───────────────────────────────────────────────────────────────
    role = "Web Research Specialist & Technical Analyst"
    goal = (
        "Deliver accurate, up-to-date, and comprehensive research reports by "
        "decomposing complex questions into focused sub-queries, gathering "
        "multi-source web evidence, and synthesising it into structured answers."
    )
    backstory = (
        "You are a meticulous research analyst who never stops at the first result. "
        "You decompose every question into 3–4 orthogonal search angles, verify "
        "facts across multiple sources, and always cite your references. "
        "When consulted by other agents, you provide concise, actionable summaries "
        "focused on what the calling agent needs to know."
    )

    _MAX_SUB_QUERIES      = 4    # hard cap on parallel Tavily calls per request
    _DECOMPOSE_MAX_TOKENS = 8192  # reasoning models need budget before producing answers
    _PLAN_MAX_TOKENS      = 512  # budget for building the research outline
    _PLAN_CONTEXT_MAX_CHARS = 3000  # preview fed to planner – keeps planning call cheap
    _MAX_HISTORY_MESSAGES = 8    # keep last N turns in context window
    _MAX_HERMES_STEPS     = 5    # max iterations for main research loop
    _MAX_DELEGATION_STEPS = 3    # max iterations for delegation research loop

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
            cleaned = _THINK_TAG_RE.sub("", response)

            # Extract only lines that start with the QUERY: prefix
            sub_queries = [
                line[len(_QUERY_PREFIX):].strip()
                for line in cleaned.splitlines()
                if line.strip().upper().startswith(_QUERY_PREFIX)
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

    async def _run_hermes_loop(
        self,
        query: str,
        session_id: str,
        max_steps: int,
        delegation_mode: bool = False,
        history_messages: list[dict] | None = None
    ) -> str:
        """Run the iterative ReAct loop to search the web and read pages."""
        tavily_tool = TavilySearchTool()
        reader_tool = WebReaderTool()

        system_prompt = _HERMES_DECISION_PROMPT
        if delegation_mode:
            system_prompt += (
                "\n\nCATATAN DELEGASI:\n"
                "Kamu sedang dihubungi oleh Agen lain yang membutuhkan informasi ringkas. "
                "Ketika kamu memberikan jawaban akhir ('action': 'answer'), pastikan kontennya "
                "sangat ringkas, terfokus pada data teknis yang dibutuhkan, dan actionable."
            )

        persona = self.get_persona_prompt()
        if persona:
            system_prompt = persona + "\n\n" + system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages[-self._MAX_HISTORY_MESSAGES:])

        if not history_messages or history_messages[-1]["content"] != query:
            messages.append({"role": "user", "content": query})

        step = 0
        final_answer = ""

        while step < max_steps:
            step += 1
            logger.info(
                "ResearcherAgent Hermes Loop: Step %d/%d for session=%s",
                step, max_steps, session_id
            )

            try:
                raw_response = await self._llm.chat(
                    messages,
                    max_tokens=self._DECOMPOSE_MAX_TOKENS,
                    json_mode=True
                )

                cleaned_response = _THINK_TAG_RE.sub("", raw_response).strip()
                # Parse action
                import json
                try:
                    action_data = json.loads(cleaned_response)
                except Exception as exc:
                    logger.warning("Failed to parse JSON cleanly: %s. Raw: %r", exc, cleaned_response)
                    # Try fallback regex
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', cleaned_response)
                    action = action_match.group(1) if action_match else "answer"
                    
                    query_match = re.search(r'"query"\s*:\s*"([^"]+)"', cleaned_response)
                    sq = query_match.group(1) if query_match else ""
                    
                    url_match = re.search(r'"url"\s*:\s*"([^"]+)"', cleaned_response)
                    su = url_match.group(1) if url_match else ""
                    
                    content_match = re.search(r'"content"\s*:\s*"(.*)"', cleaned_response, re.DOTALL)
                    sc = content_match.group(1) if content_match else ""
                    
                    action_data = {
                        "thought": "Failed to parse JSON cleanly.",
                        "action": action,
                        "query": sq,
                        "url": su,
                        "content": sc
                    }

                logger.info(
                    "ResearcherAgent Step %d: Thought: %s | Action: %s",
                    step, action_data.get("thought", ""), action_data.get("action", "")
                )

                # Append assistant message so LLM maintains its context
                messages.append({"role": "assistant", "content": raw_response})

                action = action_data.get("action", "answer")

                if action == "answer":
                    final_answer = action_data.get("content", "")
                    break

                elif action == "search":
                    search_query = action_data.get("query", "")
                    if not search_query:
                        messages.append({
                            "role": "user",
                            "content": "Error: parameter 'query' tidak boleh kosong untuk tindakan 'search'."
                        })
                        continue

                    logger.info("ResearcherAgent executing Tavily Search for: %r", search_query)
                    try:
                        search_resp = await tavily_tool.search(search_query)
                        if search_resp.results:
                            parts = []
                            for idx, r in enumerate(search_resp.results, start=1):
                                parts.append(
                                    f"Sumber {idx}: {r.title}\n"
                                    f"URL: {r.url}\n"
                                    f"Snippet: {r.content}\n"
                                )
                            tool_output = "\n".join(parts)
                        else:
                            tool_output = "Tidak ada hasil pencarian ditemukan."
                    except Exception as exc:
                        tool_output = f"Error saat melakukan pencarian: {exc}"

                    messages.append({
                        "role": "user",
                        "content": f"[Hasil Pencarian untuk: \"{search_query}\"]\n\n{tool_output}"
                    })

                elif action == "read":
                    read_url = action_data.get("url", "")
                    if not read_url:
                        messages.append({
                            "role": "user",
                            "content": "Error: parameter 'url' tidak boleh kosong untuk tindakan 'read'."
                        })
                        continue

                    logger.info("ResearcherAgent executing WebReader for: %r", read_url)
                    temp_task = AgentTask(
                        session_id=session_id,
                        user_input="",
                        metadata={"target_url": read_url}
                    )
                    try:
                        reader_result = await reader_tool.run(temp_task)
                        if "error" in reader_result:
                            tool_output = f"Error saat membaca halaman: {reader_result['error']}"
                        else:
                            title = reader_result.get("title", "No Title")
                            text = reader_result.get("page_text", "")
                            truncated = text[:4000]
                            tool_output = (
                                f"Judul: {title}\n"
                                f"URL Akhir: {reader_result.get('url', read_url)}\n"
                                f"Isi Halaman:\n{truncated}"
                            )
                    except Exception as exc:
                        tool_output = f"Error saat membaca halaman: {exc}"

                    messages.append({
                        "role": "user",
                        "content": f"[Hasil Pembacaan Web untuk: \"{read_url}\"]\n\n{tool_output}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Error: tindakan '{action}' tidak valid. Silakan pilih 'search', 'read', atau 'answer'."
                    })

            except Exception as exc:
                logger.error("Error in ResearcherAgent Hermes step: %s", exc)
                messages.append({
                    "role": "user",
                    "content": f"Terjadi kesalahan internal: {exc}. Silakan perbaiki tindakan atau berikan jawaban akhir."
                })
                if step >= max_steps - 1:
                    break

        if not final_answer:
            logger.info("ResearcherAgent: forcing final answer generation")
            force_prompt = (
                "Langkah riset maksimum telah tercapai. Kamu harus segera memberikan jawaban/laporan akhir "
                "sekarang berdasarkan semua informasi yang terkumpul. "
                "Gunakan format JSON yang valid dengan 'action': 'answer' dan 'content': '...'."
            )
            messages.append({"role": "user", "content": force_prompt})
            try:
                raw_response = await self._llm.chat(
                    messages,
                    max_tokens=self._DECOMPOSE_MAX_TOKENS,
                    json_mode=True
                )
                cleaned = _THINK_TAG_RE.sub("", raw_response).strip()
                action_data = json.loads(cleaned)
                final_answer = action_data.get("content", "")
            except Exception as exc:
                logger.error("Failed to generate forced final answer: %s", exc)
                # Ultimate fallback - plain text prompt using accumulated user/tool messages
                fallback_prompt = (
                    f"Buat laporan akhir riset komprehensif tentang '{query}' berdasarkan riwayat pencarian berikut.\n\n"
                )
                for msg in messages:
                    if msg["role"] == "user" and (msg["content"].startswith("[Hasil") or msg["content"].startswith("Hasil")):
                        fallback_prompt += f"\n---\n{msg['content']}\n"
                
                fallback_messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT_WITH_SEARCH},
                    {"role": "user", "content": fallback_prompt}
                ]
                final_answer = await self._llm.chat(fallback_messages, max_tokens=8192)

        return final_answer.strip()

    async def research_for_delegation(
        self,
        query: str,
        session_id: str = "delegation",
    ) -> str:
        """Lightweight research entry-point for inter-agent delegation.

        Called by other agents (e.g. DeveloperAgent) that need a quick,
        focused research result without going through the full task pipeline.

        Args:
            query:      The specific question or topic to research.
            session_id: Caller's session ID (used for logging; no history stored).

        Returns:
            A concise, plain-text research summary the calling agent can inject
            into its own prompt.
        """
        logger.info(
            "ResearcherAgent.research_for_delegation: query=%r session=%s",
            query[:120], session_id,
        )
        try:
            return await self._run_hermes_loop(
                query=query,
                session_id=session_id,
                max_steps=self._MAX_DELEGATION_STEPS,
                delegation_mode=True
            )
        except Exception as exc:
            logger.warning(
                "ResearcherAgent.research_for_delegation failed: %s session=%s", exc, session_id
            )
            return f"[Research unavailable: {exc}]"

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)

            reply = await self._run_hermes_loop(
                query=task.user_input,
                session_id=task.session_id,
                max_steps=self._MAX_HERMES_STEPS,
                history_messages=history_messages
            )

            reply = sanitize_for_telegram(reply)
            task.mark_done(reply)
            logger.info(
                "Researcher done for session=%s",
                task.session_id,
            )
        except Exception as exc:
            logger.exception("ResearcherAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

        return task

