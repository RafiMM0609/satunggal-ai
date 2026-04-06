"""
AnalysisDiagramAgent – membuat flow diagram dari hasil analisa & QnA sesi dokumen aktif.

Alur:
  1. Ambil semua pasangan Q&A dari qna_log (DocIndex) untuk sesi aktif.
  2. Ambil ringkasan bab dari doc_sections sebagai konteks latar.
  3. Kirim ke LLM dengan prompt khusus → Markdown + blok Mermaid diagram.
  4. Simpan Markdown ke task.metadata["document_markdown"].
  5. Set pending_tools = ["diagram_renderer", "doc_generator"] agar diagram
     dirender ke PNG dan dikompilasi menjadi PDF/DOCX lalu dikirim ke user.

Jika belum ada QnA log (user belum bertanya apa pun), agent menggunakan
ringkasan bab dari doc_sections sebagai sumber tunggal.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.doc_index import get_doc_index
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[str], Awaitable[None]]]

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_QNA_CHARS    = 20_000   # batas total karakter dari log Q&A
_MAX_SECTION_CHARS = 8_000   # batas karakter ringkasan bab yang dimasukkan
_MAX_ANSWER_CHARS  = 1_500   # batas per jawaban dalam log Q&A

# ── System Prompt ─────────────────────────────────────────────────────────────

_DIAGRAM_SYSTEM_PROMPT = """\
Kamu adalah Senior Technical Analyst yang ahli membuat flow diagram dan dokumentasi visual.

TUGASMU:
Berdasarkan analisa dokumen dan riwayat tanya-jawab (Q&A) yang diberikan, buat:
1. Satu atau lebih **flow diagram Mermaid** yang menggambarkan proses, alur, atau konsep utama.
2. **Penjelasan naratif** yang ringkas untuk setiap diagram.

FORMAT OUTPUT (Markdown murni):

# Diagram Alur: [Judul Singkat]

## Ringkasan Konteks
[1-2 paragraf yang menjelaskan topik utama dari analisa dan diskusi]

## Diagram [N]: [Nama Diagram]

[Penjelasan singkat diagram ini menggambarkan apa]

```mermaid
[kode Mermaid di sini]
```

### Keterangan
- [bullet point menjelaskan node/step penting]

[Ulangi blok ## Diagram untuk setiap alur/konsep yang relevan]

## Catatan Analisis
[Poin-poin penting yang perlu diperhatikan berdasarkan diskusi]

ATURAN WAJIB:
1. WAJIB menggunakan blok ```mermaid ``` untuk setiap diagram.
2. Gunakan `graph TD` (top-down) untuk alur proses/workflow.
3. Gunakan `graph LR` (left-right) untuk hubungan komponen/arsitektur.
4. Gunakan `sequenceDiagram` untuk interaksi antar pihak/sistem.
5. Label node harus singkat (maks 6 kata) dan deskriptif.
6. Selalu tambahkan penjelasan naratif sebelum dan sesudah setiap diagram.
7. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
8. HANYA kembalikan konten Markdown — jangan bungkus dengan backtick tambahan di luar konten dokumen.
9. Buat minimal 1 diagram dan maksimal 3 diagram yang paling relevan.
"""


async def _notify(cb: StatusCallback, text: str) -> None:
    if cb is None:
        return
    try:
        await cb(text)
    except Exception as exc:
        logger.warning("AnalysisDiagramAgent: progress callback raised: %s", exc)


class AnalysisDiagramAgent(BaseAgent):
    """Menghasilkan flow diagram Mermaid dari hasil analisa & QnA sesi dokumen."""

    name = "analysis_diagram"

    def __init__(self, llm: LLMClient) -> None:
        self._llm       = llm
        self._doc_index = get_doc_index()

    async def run(self, task: AgentTask) -> AgentTask:
        session_id  = task.session_id
        user_input  = task.user_input or ""
        status_cb   = task.metadata.get("status_callback")

        await _notify(status_cb, "📊 *Mengumpulkan data analisa dan diskusi...*")

        # ── 1. Ambil QnA log ──────────────────────────────────────────────────
        qna_pairs = self._doc_index.get_qna_log(session_id)

        # ── 2. Ambil ringkasan bab sebagai konteks latar ──────────────────────
        sections = self._doc_index.get_sections(session_id)
        doc_meta = self._doc_index.get_doc_meta(session_id)
        doc_title = (doc_meta or {}).get("doc_title", "Dokumen")

        if not qna_pairs and not sections:
            task.mark_done(
                "⚠️ Belum ada data analisa atau diskusi untuk sesi ini.\n\n"
                "Silakan upload dokumen terlebih dahulu dan lakukan beberapa tanya-jawab, "
                "kemudian minta diagram dibuat dari hasil diskusi tersebut."
            )
            return task

        await _notify(status_cb, "🧠 *Membuat flow diagram dari hasil analisa...*")

        # ── 3. Bangun konteks untuk LLM ───────────────────────────────────────
        context_parts: list[str] = []

        # 3a. Ringkasan bab (konteks latar dokumen)
        if sections:
            section_text = ""
            for sec in sections:
                title   = sec.get("bab_title", "")
                summary = sec.get("summary", "") or ""
                if summary:
                    chunk = f"**{title}**: {summary[:_MAX_SECTION_CHARS // max(len(sections), 1)]}"
                    section_text += chunk + "\n\n"
                    if len(section_text) >= _MAX_SECTION_CHARS:
                        break
            if section_text:
                context_parts.append(
                    f"=== RINGKASAN DOKUMEN: {doc_title} ===\n\n{section_text.strip()}"
                )

        # 3b. Riwayat Q&A
        if qna_pairs:
            qna_text = ""
            for pair in qna_pairs:
                turn     = pair.get("turn_index", "?")
                question = pair.get("question", "")
                answer   = pair.get("answer", "")[:_MAX_ANSWER_CHARS]
                chunk = f"**T{turn} - Pertanyaan:** {question}\n**Jawaban:** {answer}\n"
                if len(qna_text) + len(chunk) > _MAX_QNA_CHARS:
                    break
                qna_text += chunk + "\n"
            if qna_text:
                context_parts.append(
                    f"=== RIWAYAT TANYA-JAWAB ({len(qna_pairs)} pertanyaan) ===\n\n{qna_text.strip()}"
                )

        context_combined = "\n\n---\n\n".join(context_parts)

        user_prompt = (
            f"Dokumen yang dianalisa: **{doc_title}**\n"
            f"Permintaan pengguna: {user_input}\n\n"
            f"{context_combined}\n\n"
            f"Berdasarkan data di atas, buat flow diagram Mermaid beserta penjelasannya."
        )

        messages = [
            {"role": "system", "content": _DIAGRAM_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # ── 4. Panggil LLM ────────────────────────────────────────────────────
        try:
            markdown = await self._llm.chat(messages, max_tokens=4000, temperature=0.4)
        except Exception as exc:
            logger.exception("AnalysisDiagramAgent: LLM failed session=%s: %s", session_id, exc)
            task.result = "❌ Gagal membuat diagram. Silakan coba lagi."
            task.mark_failed(str(exc))
            return task

        if not markdown or not markdown.strip():
            task.result = "❌ LLM tidak menghasilkan konten diagram."
            task.mark_failed("empty LLM response")
            return task

        # ── 5. Simpan Markdown + set pending tools ────────────────────────────
        await _notify(status_cb, "🖼️ *Merender diagram...*")

        task.metadata["document_markdown"] = markdown
        task.metadata["doc_title"]         = f"Diagram Analisa – {doc_title}"

        # Trigger DiagramRendererTool (Mermaid → PNG) lalu DocumentGeneratorTool (→ PDF/DOCX)
        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        task.result = (
            f"✅ Flow diagram dari analisa **{doc_title}** sedang disiapkan.\n\n"
            f"Diagram dibuat berdasarkan {len(qna_pairs)} pertanyaan & "
            f"{len(sections)} bab dokumen.\n\n"
            f"📎 File diagram akan dikirim setelah proses render selesai."
        )
        task.mark_done(task.result)

        logger.info(
            "AnalysisDiagramAgent done: session=%s qna_pairs=%d sections=%d",
            session_id, len(qna_pairs), len(sections),
        )
        return task
