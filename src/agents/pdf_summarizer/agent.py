"""
PDFSummarizerAgent – handles pdf_summarization intent.

Capabilities:
  1. Summarize a PDF document – produce a structured summary with key points,
     main topics, and a concise conclusion.
  2. Answer specific questions about the PDF content when the user provides
     a question alongside the document.

Flow:
  1. Read pdf_chunks from task.metadata (populated by PDFParserTool upstream).
  2. Concatenate chunks up to a 30 000-character cap to avoid token overflow.
  3. Call LLM with the document text and the user's original message.
  4. Store the LLM response in task.result and mark task done.
"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 30_000

_SYSTEM_PROMPT = """\
Kamu adalah asisten cerdas yang ahli membaca, merangkum, dan menjawab pertanyaan \
tentang dokumen PDF.

Ketika diminta meringkas dokumen:
- Buat ringkasan terstruktur yang mencakup: Topik Utama, Poin-Poin Penting, dan Kesimpulan.
- Gunakan format yang rapi dengan heading dan bullet point.
- Tulis dalam bahasa yang sama dengan dokumen atau bahasa yang digunakan pengguna.

Ketika ada pertanyaan spesifik:
- Jawab berdasarkan isi dokumen yang diberikan.
- Jika jawaban tidak ditemukan di dokumen, sampaikan dengan jelas.
- Kutip bagian relevan dari dokumen bila perlu.

Selalu berikan jawaban yang informatif, akurat, dan mudah dipahami.
"""

_SUMMARIZE_PROMPT = """\
Berikut adalah isi dokumen PDF:

{document_text}

---
{user_request}

Berikan respons yang sesuai berdasarkan isi dokumen di atas.
"""


class PDFSummarizerAgent(BaseAgent):
    """Summarises PDF documents and answers questions about their content."""

    name = "pdf_summarizer"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, task: AgentTask) -> AgentTask:
        """
        Summarise or answer questions about the PDF whose text is in
        task.metadata["pdf_chunks"].
        """
        chunks: list[str] = task.metadata.get("pdf_chunks", [])

        if not chunks:
            task.mark_done(
                "❌ Tidak ada teks yang dapat dibaca dari dokumen PDF ini. "
                "Pastikan PDF tidak terenkripsi atau berbasis gambar (scan)."
            )
            return task

        # Concatenate chunks up to the character cap
        document_text = ""
        for chunk in chunks:
            if len(document_text) + len(chunk) > _MAX_DOC_CHARS:
                remaining = _MAX_DOC_CHARS - len(document_text)
                if remaining > 0:
                    document_text += chunk[:remaining]
                break
            document_text += chunk

        user_request = task.user_input or "Tolong ringkas dokumen ini."

        prompt = _SUMMARIZE_PROMPT.format(
            document_text=document_text,
            user_request=user_request,
        )

        logger.info(
            "PDFSummarizerAgent: session=%s doc_chars=%d user_request=%.80s",
            task.session_id, len(document_text), user_request,
        )

        try:
            response = await self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.strip() if response else "❌ LLM tidak menghasilkan respons."
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDFSummarizerAgent LLM error: session=%s", task.session_id)
            result = f"❌ Gagal memproses dokumen: {exc}"

        task.mark_done(result)
        return task
