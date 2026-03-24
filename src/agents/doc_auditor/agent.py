"""
DocAuditorAgent – Quality Auditor untuk dokumen .docx.

Alur kerja (Step-Planned System):
──────────────────────────────────────────────────────────────────────────────
MODE 1: ANALISIS DOKUMEN (ketika file .docx baru dikirim)
  Langkah 1: Validasi seksi yang sudah di-parse oleh DocxParserTool
  Langkah 2: Buat Daftar Isi (ToC) dari judul-judul bab
  Langkah 3: Ringkas setiap bab (batch LLM, hemat RAM)
  Langkah 4: Simpan ke SQLite (via DocIndex)
  Langkah 5: Kirim laporan: Judul Dokumen + Daftar Isi + Ringkasan per Bab

MODE 2: TANYA-JAWAB INTERAKTIF (pesan teks lanjutan dalam sesi yang sama)
  - Cari bab relevan dari indeks SQLite berdasarkan kata kunci query
  - Ambil teks asli + ringkasan bab tersebut
  - Jawab dengan LLM menggunakan konteks bab + histori percakapan

Step-planned progress update dikirim melalui status_callback agar pengguna
selalu tahu apa yang sedang dilakukan bot.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.doc_index import get_doc_index
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[str], Awaitable[None]]]

# ── Constants ─────────────────────────────────────────────────────────────────

# Max section content length passed to LLM during summarization (chars)
_MAX_SECTION_CONTENT_CHARS = 3000
# Max section content length included in Q&A context (chars)
_MAX_QA_CONTENT_CHARS = 2000

_SUMMARY_SYSTEM_PROMPT = """\
Kamu adalah Quality Auditor AI yang bertugas merangkum isi dokumen teknis.

Tugasmu adalah menganalisis teks satu bab/seksi dan menghasilkan:
1. **Ringkasan objektif** (3-5 poin penting, menggunakan bullet point)
2. **Catatan kualitas** (apakah ada bagian yang kurang jelas, inkonsisten, atau perlu perbaikan)

Format output WAJIB:
**Ringkasan:**
- Poin 1
- Poin 2
- ...

**Catatan Kualitas:**
- Catatan (atau "Tidak ada catatan khusus." jika isinya baik)

Gunakan Bahasa Indonesia yang profesional dan ringkas.
Jangan tambahkan pembukaan atau penutup di luar format di atas.
"""

_QA_SYSTEM_PROMPT = """\
Kamu adalah Quality Auditor AI yang memiliki akses ke indeks dokumen yang sudah dianalisis.

Tugas: Jawab pertanyaan pengguna berdasarkan konten bab yang relevan dari dokumen.
- Berikan jawaban yang spesifik dan mengacu pada isi dokumen.
- Jika ada beberapa bab yang relevan, jelaskan perbedaan atau hubungannya.
- Jika pertanyaan tidak dapat dijawab dari dokumen, katakan dengan jelas.
- Gunakan bahasa yang sama dengan pengguna.
"""

_CONSISTENCY_SYSTEM_PROMPT = """\
Kamu adalah Quality Auditor AI.

Tugas: Periksa konsistensi istilah antara dua bab yang berbeda.
Bandingkan apakah terminologi, definisi, atau konsep yang digunakan konsisten.
Jika ada inkonsistensi, sebutkan secara spesifik.
Gunakan Bahasa Indonesia.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _notify(cb: StatusCallback, text: str) -> None:
    if cb is None:
        return
    try:
        await cb(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("DocAuditorAgent progress callback raised: %s", exc)


def _build_step_plan_msg(
    current_step: int,
    total_steps: int,
    step_name: str,
    doc_title: str,
    total_sections: int,
) -> str:
    """Bangun pesan progres step-planned yang informatif."""
    steps = [
        ("📋", "Validasi seksi dokumen"),
        ("🗂️", "Membuat Daftar Isi"),
        ("🧠", f"Meringkas {total_sections} bab (batch)"),
        ("💾", "Menyimpan indeks ke database"),
        ("📤", "Menyiapkan laporan akhir"),
    ]
    lines = [f"⏳ *Menganalisis Dokumen*\n📄 _{doc_title}_\n"]
    for i, (icon, name) in enumerate(steps, start=1):
        if i < current_step:
            lines.append(f"  {icon} ~~{name}~~ ✅")
        elif i == current_step:
            lines.append(f"  {icon} *{name}* ← _sedang berjalan..._")
        else:
            lines.append(f"  {icon} {name} _(menunggu)_")
    lines.append(f"\n_Langkah {current_step} dari {total_steps}_")
    return "\n".join(lines)


def _build_toc(sections: list[dict]) -> str:
    """Buat Daftar Isi dari daftar seksi."""
    lines = ["📋 *DAFTAR ISI*\n"]
    for sec in sections:
        level = sec.get("level", 1)
        indent = "  " * (level - 1)
        idx = sec.get("index") or sec.get("bab_index", "?")
        title = sec.get("title") or sec.get("bab_title", "")
        lines.append(f"{indent}{idx}. {title}")
    return "\n".join(lines)


def _build_final_report(
    doc_title: str,
    sections_with_summary: list[dict],
    original_filename: str,
) -> str:
    """Buat laporan lengkap: judul + daftar isi + ringkasan per bab."""
    parts: list[str] = []

    # Header
    parts.append(f"# 📄 Laporan Analisis Dokumen\n**{doc_title}**\n_{original_filename}_\n")
    parts.append("---\n")

    # Daftar Isi
    toc_lines = ["## 📋 Daftar Isi\n"]
    for sec in sections_with_summary:
        level = sec.get("level", 1)
        indent = "  " * (level - 1)
        idx = sec.get("index") or sec.get("bab_index", "?")
        title = sec.get("title") or sec.get("bab_title", "")
        toc_lines.append(f"{indent}{idx}. {title}")
    parts.append("\n".join(toc_lines))
    parts.append("\n---\n")

    # Ringkasan per Bab
    parts.append("## 🧠 Ringkasan per Bab\n")
    for sec in sections_with_summary:
        idx = sec.get("index") or sec.get("bab_index", "?")
        title = sec.get("title") or sec.get("bab_title", "")
        summary = sec.get("summary") or "_Ringkasan tidak tersedia._"
        level = sec.get("level", 1)
        prefix = "#" * min(level + 2, 6)
        parts.append(f"{prefix} {idx}. {title}\n\n{summary}\n")

    # Footer
    parts.append(
        "\n---\n"
        "💡 *Tip:* Balas pesan ini untuk bertanya lebih lanjut tentang bab tertentu.\n"
        "Contoh: _\"Jelaskan lebih detail tentang bab 3\"_ atau _\"Apa yang dimaksud dengan X di bab 2?\"_"
    )
    return "\n".join(parts)


# ── DocAuditorAgent ───────────────────────────────────────────────────────────

class DocAuditorAgent(BaseAgent):
    """
    Quality Auditor untuk dokumen .docx dengan step-planned system.

    Mode 1 – Analisis Dokumen: dipanggil saat file .docx dikirim.
      - task.metadata["docx_sections"]   → list seksi dari DocxParserTool
      - task.metadata["doc_title"]       → judul dokumen
      - task.metadata["docx_file_id"]    → nama file sebagai identifier
      - task.metadata["status_callback"] → progress callback

    Mode 2 – Q&A Interaktif: dipanggil saat pesan teks lanjutan.
      - task.user_input → pertanyaan pengguna
      - DocIndex di-query untuk seksi relevan
    """

    name = "doc_auditor"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm = llm or LLMClient()
        self._doc_index = get_doc_index()

    # ── Main entrypoint ───────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        sections: list[dict] | None = task.metadata.get("docx_sections")
        status_cb: StatusCallback = task.metadata.get("status_callback")

        if sections is not None:
            # MODE 1: analisis dokumen baru
            return await self._analyze_document(task, sections, status_cb)
        else:
            # MODE 2: tanya-jawab lanjutan
            return await self._answer_question(task, status_cb)

    # ── MODE 1: Analisis Dokumen ──────────────────────────────────────────────

    async def _analyze_document(
        self,
        task: AgentTask,
        sections: list[dict],
        status_cb: StatusCallback,
    ) -> AgentTask:
        """Step-planned pipeline: parse → ToC → summarize → index → report."""
        session_id    = task.session_id
        doc_title     = task.metadata.get("doc_title", "Dokumen")
        file_id       = task.metadata.get("docx_file_id", "document.docx")
        total_sections = len(sections)
        original_filename = task.metadata.get("original_filename", file_id)
        total_steps   = 5

        # ── Langkah 1: Validasi seksi ─────────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_plan_msg(1, total_steps, "Validasi seksi", doc_title, total_sections),
        )

        if not sections:
            task.mark_failed("Dokumen tidak memiliki konten yang dapat dibaca.")
            task.result = (
                "❌ Gagal menganalisis dokumen: tidak ada teks yang dapat diekstrak. "
                "Pastikan file .docx berisi teks (bukan gambar)."
            )
            return task

        logger.info(
            "DocAuditor: analyzing session=%s file_id=%r sections=%d",
            session_id, file_id, total_sections,
        )

        # ── Langkah 2: Daftar Isi ────────────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_plan_msg(2, total_steps, "Membuat Daftar Isi", doc_title, total_sections),
        )
        # (ToC dibangun dari data sections, tidak perlu LLM)

        # ── Langkah 3: Ringkas per bab ───────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_plan_msg(3, total_steps, f"Meringkas {total_sections} bab", doc_title, total_sections),
        )

        sections_with_summary: list[dict[str, Any]] = []
        for sec in sections:
            summary = await self._summarize_section(
                section=sec,
                session_id=session_id,
            )
            sec_copy = dict(sec)
            sec_copy["summary"] = summary
            sections_with_summary.append(sec_copy)
            logger.debug(
                "DocAuditor: summarized bab=%r session=%s",
                sec.get("title", "?"), session_id,
            )

        # ── Langkah 4: Simpan ke database ────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_plan_msg(4, total_steps, "Menyimpan indeks", doc_title, total_sections),
        )

        total_words = task.metadata.get("total_words", 0)
        self._doc_index.save_document(
            session_id=session_id,
            file_id=file_id,
            doc_title=doc_title,
            sections=sections,  # simpan teks asli
            total_words=total_words,
        )
        # Simpan ringkasan per bab
        for sec in sections_with_summary:
            self._doc_index.save_summary(
                session_id=session_id,
                file_id=file_id,
                bab_index=sec["index"],
                summary=sec.get("summary", ""),
            )

        # ── Langkah 5: Laporan akhir ──────────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_plan_msg(5, total_steps, "Menyiapkan laporan", doc_title, total_sections),
        )

        report = _build_final_report(doc_title, sections_with_summary, original_filename)
        task.mark_done(report)

        logger.info(
            "DocAuditor: analysis done session=%s file_id=%r",
            session_id, file_id,
        )
        return task

    async def _summarize_section(
        self,
        section: dict,
        session_id: str,
    ) -> str:
        """Panggil LLM untuk meringkas satu seksi bab."""
        title   = section.get("title", "")
        content = section.get("content", "")

        if not content.strip():
            return "_Bab ini tidak memiliki konten teks._"

        # Batasi konten ke _MAX_SECTION_CONTENT_CHARS karakter agar hemat token / RAM
        truncated_content = content[:_MAX_SECTION_CONTENT_CHARS]
        if len(content) > _MAX_SECTION_CONTENT_CHARS:
            truncated_content += "\n... _(konten dipotong untuk efisiensi)_"

        user_prompt = (
            f"Bab/Seksi: **{title}**\n\n"
            f"Konten:\n{truncated_content}"
        )

        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            summary = await self._llm.chat(messages, max_tokens=1024, temperature=0.3)
            return summary
        except Exception as exc:
            logger.exception(
                "DocAuditor: LLM summarize failed for bab=%r session=%s: %s",
                title, session_id, exc,
            )
            return "_Gagal meringkas bab ini._"

    # ── MODE 2: Q&A Interaktif ────────────────────────────────────────────────

    async def _answer_question(
        self,
        task: AgentTask,
        status_cb: StatusCallback,
    ) -> AgentTask:
        """Jawab pertanyaan lanjutan menggunakan indeks dokumen yang sudah ada."""
        session_id = task.session_id
        user_query = task.user_input

        await _notify(status_cb, "🔍 *Mencari bab yang relevan di dokumen...*")

        # Cek apakah ada dokumen terindeks
        if not self._doc_index.has_document(session_id):
            task.mark_done(
                "⚠️ Belum ada dokumen yang dianalisis untuk sesi ini.\n\n"
                "Silakan kirim file `.docx` terlebih dahulu untuk memulai analisis."
            )
            return task

        # Cari seksi relevan
        relevant_sections = self._doc_index.search_sections(
            session_id=session_id,
            query=user_query,
            limit=3,
        )

        # Bangun konteks dari seksi relevan
        context_parts = []
        for sec in relevant_sections:
            idx     = sec.get("bab_index", "?")
            title   = sec.get("bab_title", "")
            content = sec.get("content_text", "")
            summary = sec.get("summary", "")
            context_parts.append(
                f"### Bab {idx}: {title}\n\n"
                f"**Teks Asli (ringkasan):**\n{content[:_MAX_QA_CONTENT_CHARS]}\n\n"
                f"**Ringkasan Bot:**\n{summary}"
            )

        doc_context = "\n\n---\n\n".join(context_parts)

        # Ambil histori percakapan untuk konteks
        history_messages = self._history.get_as_llm_messages(session_id)

        await _notify(status_cb, "🧠 *Menjawab pertanyaan berdasarkan dokumen...*")

        user_prompt = (
            f"Pertanyaan: {user_query}\n\n"
            f"Konten bab yang relevan dari dokumen:\n\n{doc_context}"
        )

        messages = [{"role": "system", "content": _QA_SYSTEM_PROMPT}]
        # Tambahkan histori percakapan (maks 6 pesan terakhir)
        messages.extend(history_messages[-6:])
        # Pastikan pesan user terakhir adalah query ini
        if not history_messages or history_messages[-1]["content"] != user_query:
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Ganti pesan user terakhir dengan versi yang sudah disertai konteks
            messages[-1] = {"role": "user", "content": user_prompt}

        try:
            answer = await self._llm.chat(messages, max_tokens=2048, temperature=0.5)
            task.mark_done(answer)
        except Exception as exc:
            logger.exception("DocAuditor QA LLM failed session=%s: %s", session_id, exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat menjawab pertanyaan. Silakan coba lagi."

        return task
