"""
ManagerAgent – Hierarchical Manager Pattern.

Peran:
  Manager senior yang duduk di atas semua specialist agent.
  Ia TIDAK menjawab pertanyaan user secara langsung, melainkan MEREVIEW
  output yang dihasilkan oleh agent lain sebelum dikirim ke user.

Tugas:
  1. Mengevaluasi relevansi jawaban terhadap pertanyaan user.
  2. Mendeteksi jawaban yang kosong, tidak lengkap, atau off-topic.
  3. Mengembalikan output yang sudah diperbaiki HANYA jika ada masalah jelas.
  4. Mengembalikan output asli (tanpa perubahan) jika sudah baik.

Desain:
  - Non-fatal: jika review gagal, output asli dikembalikan apa adanya.
  - Efisien: prompt dibatasi 800 char (input) + 2500 char (output).
  - Deterministik: temperature=0.15 untuk konsistensi.
  - Tidak mewarisi BaseAgent karena tidak dipanggil via router.run()
    melainkan langsung via manager.review() dari main_loop.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Sampling parameters ────────────────────────────────────────────────────────

MANAGER_TEMPERATURE = 0.15
MANAGER_TOP_P       = 0.90

# ── Context window caps ────────────────────────────────────────────────────────
# Batas karakter yang dikirim ke LLM untuk menjaga latency dan cost tetap rendah.

MAX_INPUT_CHARS:  int = 800
MAX_OUTPUT_CHARS: int = 2_500

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Manager Agent** — reviewer senior yang mengevaluasi kualitas jawaban \
sebelum dikirim ke pengguna.

TUGASMU:
1. Baca pertanyaan asli pengguna dan jawaban dari specialist agent.
2. Nilai apakah jawaban tersebut:
   a. Relevan dan menjawab inti pertanyaan.
   b. Cukup lengkap (tidak terlalu singkat tanpa alasan).
   c. Tidak berisi hallusinasi atau klaim palsu yang jelas.
3. Jika jawaban sudah baik → balas dengan TEPAT SATU kata: **OK**
4. Jika ada masalah jelas → kembalikan versi yang DIPERBAIKI (bukan komentar tentang masalahnya).

ATURAN KRITIS:
- Jangan pernah menambahkan komentar seperti "Jawaban ini kurang..." atau "Saya memperbaiki...".
- Output-mu adalah LANGSUNG jawaban final yang akan diterima pengguna.
- Jangan ubah gaya bahasa, nada, atau format asli kecuali benar-benar salah.
- Jika kamu tidak yakin ada masalah, balas **OK**.

---
Pertanyaan pengguna:
{user_input}

Jawaban dari agent [{agent_name}]:
{agent_output}
"""

# ── Agent yang tidak perlu direview ───────────────────────────────────────────
# Agent-agent ini menghasilkan output yang bersifat teknis/file/binary,
# atau sudah punya mekanisme validasi sendiri, sehingga review tambahan
# oleh Manager akan membuang latency tanpa nilai tambah.

_SKIP_REVIEW_AGENTS: frozenset[str] = frozenset({
    "developer",           # punya DeveloperInspectorAgent sendiri
    "developer_inspector", # sudah berperan sebagai reviewer
    "web_automation",      # output teknis (screenshots, DOM)
    "wbs_agent",           # output Excel – tidak bisa direview teks
    "mandays_agent",       # output Excel – tidak bisa direview teks
    "quiz_agent",          # output HTML quiz
    "tg_quiz_agent",       # output HTML quiz
})


class ManagerAgent:
    """
    Lightweight post-processing reviewer untuk semua specialist agent.

    Dipanggil secara otomatis oleh ``main_loop.process_message()`` setelah
    ``agent.run()`` selesai dan sebelum hasilnya dicatat ke history.

    Tidak mewarisi ``BaseAgent`` karena tidak pernah di-route secara langsung
    oleh ``AgentRouter`` — ia beroperasi sebagai middleware di pipeline.
    """

    name = "manager"

    role = "Senior Quality Manager"
    goal = (
        "Memastikan setiap jawaban yang dikirim ke pengguna relevan, "
        "akurat, dan bebas dari hallusinasi."
    )
    backstory = (
        "Seorang manajer berpengalaman yang telah mengawasi ratusan specialist agent. "
        "Ia tahu kapan harus memperbaiki jawaban dan kapan harus membiarkannya apa adanya."
    )

    def __init__(self, llm: "LLMClient") -> None:
        self._llm = llm

    async def review(
        self,
        user_input:   str,
        agent_name:   str,
        agent_output: str,
        session_id:   str = "unknown",
    ) -> str:
        """Review *agent_output* terhadap *user_input* dan kembalikan teks final.

        Mengembalikan *agent_output* tidak berubah bila:
        * *agent_output* kosong atau hanya whitespace.
        * Agent ada di daftar ``_SKIP_REVIEW_AGENTS``.
        * LLM mengonfirmasi kualitas dengan membalas "OK".
        * Panggilan LLM gagal (fail-safe).

        Args:
            user_input:   Teks asli dari pengguna.
            agent_name:   Nama agent yang menghasilkan output (untuk logging & skip-list).
            agent_output: Teks jawaban yang akan direview.
            session_id:   ID sesi (untuk logging).

        Returns:
            Teks final yang sudah direview (atau output asli jika tidak ada masalah).
        """
        # Lewati review jika output kosong
        if not agent_output or not agent_output.strip():
            return agent_output

        # Lewati agent yang sudah punya reviewer sendiri atau output non-teks
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
            review = await self._llm.chat(
                messages,
                max_tokens   = 1024,
                temperature  = MANAGER_TEMPERATURE,
                top_p        = MANAGER_TOP_P,
            )
            review = review.strip()

            if not review:
                return agent_output

            # "OK" → output sudah bagus, kembalikan asli
            if review.upper().startswith("OK"):
                logger.debug(
                    "ManagerAgent: approved agent=%s session=%s",
                    agent_name, session_id,
                )
                return agent_output

            # LLM mengembalikan versi yang diperbaiki
            logger.info(
                "ManagerAgent: revised agent=%s (%d → %d chars) session=%s",
                agent_name, len(agent_output), len(review), session_id,
            )
            return review

        except Exception as exc:
            logger.warning(
                "ManagerAgent: review failed (returning original) agent=%s session=%s error=%s",
                agent_name, session_id, exc,
            )
            return agent_output
