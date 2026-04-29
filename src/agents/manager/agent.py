"""
ManagerAgent – Hierarchical Manager Pattern.

Peran:
  Manager senior yang duduk di atas semua specialist agent.
  Ia TIDAK menjawab pertanyaan user secara langsung, melainkan MEREVIEW
  output yang dihasilkan oleh agent lain sebelum dikirim ke user.

Tugas:
  1. Mengevaluasi relevansi jawaban terhadap pertanyaan user.
  2. Mendeteksi jawaban yang kosong, tidak lengkap, atau off-topic.
  3. Jika jawaban sudah baik → kembalikan apa adanya (verdict: OK).
  4. Jika ada masalah gaya/struktur/relevansi → perbaiki sendiri (verdict: REVISE).
  5. Jika jawaban mengandung klaim faktual berisiko → delegasikan ke ResearcherAgent
     untuk verifikasi real-time, lalu gabungkan hasilnya (verdict: VALIDATE).

Desain:
  - Non-fatal: jika review gagal, output asli dikembalikan apa adanya.
  - Efisien: prompt dibatasi 800 char (input) + 2500 char (output).
  - Deterministik: temperature=0.15 untuk konsistensi.
  - Tidak mewarisi BaseAgent karena tidak dipanggil via router.run()
    melainkan langsung via manager.review() dari main_loop.py.
  - Delegation: jika agents dict diberikan, Manager dapat mendelegasikan
    validasi faktual ke ResearcherAgent via research_for_delegation().
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient
    from src.agents.researcher.agent import ResearcherAgent

logger = logging.getLogger(__name__)

# ── Sampling parameters ────────────────────────────────────────────────────────

MANAGER_TEMPERATURE = 0.15
MANAGER_TOP_P       = 0.90

# ── Context window caps ────────────────────────────────────────────────────────

MAX_INPUT_CHARS:  int = 800
MAX_OUTPUT_CHARS: int = 2_500

# ── Regex to strip <think>…</think> blocks from reasoning models ───────────────

_THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)

# ── System prompt (structured 3-verdict) ──────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Manager Agent** — reviewer senior yang mengevaluasi kualitas jawaban \
sebelum dikirim ke pengguna.

TUGASMU:
1. Baca pertanyaan asli pengguna dan jawaban dari specialist agent.
2. Pilih TEPAT SATU dari tiga VERDICT berikut:

   • **OK** – Jawaban sudah relevan, lengkap, dan tidak mengandung klaim berisiko.
              Kembalikan tanpa perubahan.

   • **REVISE** – Ada masalah gaya/struktur/relevansi yang bisa diperbaiki TANPA data baru.
                 Tulis versi yang diperbaiki di field "output".

   • **VALIDATE** – Jawaban mengandung klaim faktual SANGAT SPESIFIK (angka, tanggal, nama,
                   statistik, event terkini) yang berisiko tinggi salah dan perlu diverifikasi
                   via pencarian web. Tulis query pencarian yang fokus di "validation_query".

ATURAN KRITIS:
- Balas dengan JSON SAJA – tanpa markdown, tanpa penjelasan di luar JSON.
- Schema: {{"verdict": "OK|REVISE|VALIDATE", "output": "<teks revisi atau null>", "validation_query": "<query atau null>"}}
- Jangan tambahkan komentar seperti "Jawaban ini kurang..." dalam field output.
- Saat REVISE: pertahankan gaya bahasa asli (formal/santai/slang/Gen Z) persis.
- Gunakan VALIDATE hanya untuk klaim sangat spesifik & berisiko. Default ke OK jika ragu.
- Gunakan REVISE hanya jika ada masalah JELAS. Default ke OK jika ragu.

---
Pertanyaan pengguna:
{user_input}

Jawaban dari agent [{agent_name}]:
{agent_output}
"""

# ── Agent yang tidak perlu direview ───────────────────────────────────────────

_SKIP_REVIEW_AGENTS: frozenset[str] = frozenset({
    "developer",           # punya DeveloperInspectorAgent sendiri
    "developer_inspector", # sudah berperan sebagai reviewer
    "researcher",          # sudah menggunakan Tavily, tidak perlu divalidasi lagi
    "web_automation",      # output teknis (screenshots, DOM)
    "wbs_agent",           # output Excel – tidak bisa direview teks
    "mandays_agent",       # output Excel – tidak bisa direview teks
    "quiz_agent",          # output HTML quiz
    "tg_quiz_agent",       # output HTML quiz
})

# ── Agent yang eligible untuk VALIDATE (bisa berisi klaim faktual) ────────────

_VALIDATE_ELIGIBLE_AGENTS: frozenset[str] = frozenset({
    "responder",
    "content_creator",
    "technical_writer",
    "sysinfo_agent",
})


class ManagerAgent:
    """
    Post-processing reviewer dengan kemampuan delegasi ke ResearcherAgent.

    Dipanggil secara otomatis oleh ``main_loop.process_message()`` setelah
    ``agent.run()`` selesai dan sebelum hasilnya dicatat ke history.

    Alur review (3 verdict):
      OK       → kembalikan output asli
      REVISE   → kembalikan revisi dari LLM manager
      VALIDATE → delegasikan ke ResearcherAgent, gabungkan fact-check ke output

    Tidak mewarisi ``BaseAgent`` karena tidak pernah di-route secara langsung
    oleh ``AgentRouter`` — ia beroperasi sebagai middleware di pipeline.
    """

    name = "manager"

    role = "Senior Quality Manager"
    goal = (
        "Memastikan setiap jawaban yang dikirim ke pengguna relevan, "
        "akurat, dan bebas dari hallusinasi — dengan mendelegasikan verifikasi "
        "faktual ke ResearcherAgent bila diperlukan."
    )
    backstory = (
        "Seorang manajer berpengalaman yang telah mengawasi ratusan specialist agent. "
        "Ia tahu kapan harus memperbaiki jawaban sendiri, kapan harus membiarkannya, "
        "dan kapan harus meminta researcher untuk memverifikasi klaim faktual."
    )

    def __init__(
        self,
        llm: "LLMClient",
        agents: Optional[dict] = None,
    ) -> None:
        self._llm    = llm
        self._agents = agents or {}

    @property
    def _researcher(self) -> "Optional[ResearcherAgent]":
        """Return ResearcherAgent instance jika tersedia, else None."""
        return self._agents.get("researcher")  # type: ignore[return-value]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_verdict(self, raw: str) -> dict:
        """Parse JSON verdict dari LLM. Return dict kosong jika gagal."""
        cleaned = _THINK_TAG_RE.sub("", raw).strip()
        # Coba ekstrak blok JSON jika LLM menambahkan teks di luar kurung kurawal
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("ManagerAgent: failed to parse verdict JSON: %r", raw[:200])
            return {}

    async def _delegate_to_researcher(
        self,
        validation_query: str,
        original_output:  str,
        session_id:       str,
    ) -> str:
        """Panggil ResearcherAgent.research_for_delegation() dan gabungkan hasilnya.

        Jika researcher tidak tersedia atau gagal, kembalikan original_output apa adanya.
        """
        researcher = self._researcher
        if researcher is None:
            logger.warning(
                "ManagerAgent: VALIDATE verdict tapi researcher tidak tersedia session=%s",
                session_id,
            )
            return original_output

        logger.info(
            "ManagerAgent: delegating fact-check to researcher query=%r session=%s",
            validation_query[:120], session_id,
        )
        try:
            fact_check = await researcher.research_for_delegation(
                query=validation_query,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(
                "ManagerAgent: researcher delegation failed (returning original) "
                "session=%s error=%s", session_id, exc,
            )
            return original_output

        if not fact_check or fact_check.startswith("[Research unavailable"):
            logger.info(
                "ManagerAgent: researcher returned no useful data session=%s", session_id,
            )
            return original_output

        # Gabungkan output asli + hasil verifikasi sebagai bagian tambahan
        combined = (
            original_output.rstrip()
            + "\n\n---\n"
            + "🔍 **Verifikasi Fakta** *(oleh Researcher Agent)*\n\n"
            + fact_check.strip()
        )
        logger.info(
            "ManagerAgent: fact-check appended (%d chars) session=%s",
            len(fact_check), session_id,
        )
        return combined

    # ── Public API ────────────────────────────────────────────────────────────

    async def review(
        self,
        user_input:   str,
        agent_name:   str,
        agent_output: str,
        session_id:   str = "unknown",
    ) -> str:
        """Review *agent_output* terhadap *user_input* dan kembalikan teks final.

        Alur:
          1. Skip jika output kosong atau agent ada di daftar skip.
          2. Kirim ke LLM manager untuk mendapat verdict JSON.
          3. OK       → kembalikan asli.
             REVISE   → kembalikan field "output" dari LLM.
             VALIDATE → delegasikan ke researcher, gabungkan fact-check.
          4. Jika parsing gagal atau LLM error → kembalikan asli (fail-safe).

        Args:
            user_input:   Teks asli dari pengguna.
            agent_name:   Nama agent yang menghasilkan output.
            agent_output: Teks jawaban yang akan direview.
            session_id:   ID sesi (untuk logging).

        Returns:
            Teks final yang sudah direview.
        """
        if not agent_output or not agent_output.strip():
            return agent_output

        if agent_name in _SKIP_REVIEW_AGENTS:
            logger.debug(
                "ManagerAgent: skipping review for agent=%s session=%s",
                agent_name, session_id,
            )
            return agent_output

        system = _SYSTEM_PROMPT.format(
            user_input   = user_input[:MAX_INPUT_CHARS],
            agent_name   = agent_name,
            agent_output = agent_output[:MAX_OUTPUT_CHARS],
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": "Lakukan review sekarang."},
        ]

        try:
            raw = await self._llm.chat(
                messages,
                max_tokens   = 512,
                temperature  = MANAGER_TEMPERATURE,
                top_p        = MANAGER_TOP_P,
            )
        except Exception as exc:
            logger.warning(
                "ManagerAgent: LLM call failed (returning original) agent=%s session=%s error=%s",
                agent_name, session_id, exc,
            )
            return agent_output

        verdict_data = self._parse_verdict(raw)
        verdict      = (verdict_data.get("verdict") or "OK").strip().upper()

        if verdict == "OK" or not verdict_data:
            logger.debug(
                "ManagerAgent: approved agent=%s session=%s", agent_name, session_id,
            )
            return agent_output

        if verdict == "REVISE":
            revised = (verdict_data.get("output") or "").strip()
            if not revised:
                logger.debug(
                    "ManagerAgent: REVISE verdict but empty output, keeping original "
                    "agent=%s session=%s", agent_name, session_id,
                )
                return agent_output
            logger.info(
                "ManagerAgent: revised agent=%s (%d → %d chars) session=%s",
                agent_name, len(agent_output), len(revised), session_id,
            )
            return revised

        if verdict == "VALIDATE" and agent_name in _VALIDATE_ELIGIBLE_AGENTS:
            validation_query = (verdict_data.get("validation_query") or "").strip()
            if not validation_query:
                logger.debug(
                    "ManagerAgent: VALIDATE verdict but no query, keeping original "
                    "agent=%s session=%s", agent_name, session_id,
                )
                return agent_output
            return await self._delegate_to_researcher(
                validation_query=validation_query,
                original_output=agent_output,
                session_id=session_id,
            )

        # VALIDATE untuk agent yang tidak eligible (e.g. researcher itu sendiri) → skip
        logger.debug(
            "ManagerAgent: verdict=%s agent=%s not eligible for delegation, keeping original "
            "session=%s", verdict, agent_name, session_id,
        )
        return agent_output
