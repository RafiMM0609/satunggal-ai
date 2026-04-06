"""
PDFSummarizerAgent – merangkum, menjawab pertanyaan, dan menjelaskan isi PDF.

Alur Kerja:
  1. Menerima teks yang sudah diekstrak dari PDF (disimpan di task.metadata["pdf_chunks"])
  2. Menggabungkan chunks menjadi satu konteks (dibatasi agar tidak melebihi token LLM)
  3. Memanggil LLM dengan system prompt ringkasan
  4. Menyimpan hasil ringkasan ke task.result

Catatan:
  - Jika user menyertakan pertanyaan spesifik (caption), agent akan menjawab
    pertanyaan tersebut berdasarkan isi dokumen.
  - Jika tidak ada pertanyaan, agent membuat ringkasan komprehensif.
"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah asisten ringkasan dokumen yang cerdas dan terstruktur.

Tugas utamamu adalah membaca isi dokumen PDF yang diberikan dan:
1. Jika pengguna mengajukan pertanyaan spesifik → jawab pertanyaan tersebut secara detail \
   berdasarkan isi dokumen.
2. Jika tidak ada pertanyaan spesifik → buat ringkasan komprehensif yang mencakup:
   - Gambaran umum dokumen
   - Poin-poin utama per bagian/bab
   - Kesimpulan atau temuan penting

## PANDUAN PENULISAN ##
- Gunakan bahasa yang sama dengan bahasa dokumen atau bahasa pengguna.
- Tulis secara terstruktur menggunakan poin-poin atau heading bila diperlukan.
- Jangan mengarang informasi yang tidak ada di dokumen.
- Jika dokumen terlalu panjang, prioritaskan informasi yang paling penting.
- Sebutkan jika ada bagian yang tidak dapat diringkas karena berupa gambar atau tabel.

## FORMAT OUTPUT ##
Gunakan format Markdown untuk keterbacaan:
- **Heading** untuk judul bagian
- Bullet points untuk poin-poin penting
- *Italic* untuk penekanan
- Kutip teks penting secara langsung bila relevan
"""

# Batas karakter konten PDF yang dikirim ke LLM (untuk menghindari overflow token)
_MAX_CONTENT_CHARS = 30_000


class PDFSummarizerAgent(BaseAgent):
    """Summarizes PDF content using LLM. Handles both general summary and Q&A."""

    name = "pdf_summarizer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    # ── Public interface ──────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        """
        Main entrypoint. Expects:
          - task.metadata["pdf_chunks"]: list[str]  – text chunks from PDF
          - task.user_input: str                     – user caption/question (may be empty)

        Produces:
          - task.result: str – formatted summary or Q&A answer
        """
        chunks: list[str] = task.metadata.get("pdf_chunks", [])
        if not chunks:
            task.mark_failed("Tidak ada konten PDF yang dapat diproses.")
            task.result = (
                "❌ Gagal meringkas: teks PDF kosong atau tidak dapat dibaca. "
                "Pastikan PDF berisi teks (bukan gambar hasil scan)."
            )
            return task

        # Gabungkan chunks menjadi satu blok teks, batasi panjangnya
        full_text = "\n\n".join(chunks)
        if len(full_text) > _MAX_CONTENT_CHARS:
            truncated = True
            full_text = full_text[:_MAX_CONTENT_CHARS]
        else:
            truncated = False

        logger.info(
            "PDFSummarizerAgent: session=%s chunks=%d chars=%d truncated=%s",
            task.session_id, len(chunks), len(full_text), truncated,
        )

        # Tentukan mode: Q&A atau ringkasan umum
        user_question = _extract_user_question(task.user_input)
        if user_question:
            user_prompt = (
                f"Pertanyaan dari pengguna: {user_question}\n\n"
                f"Jawab pertanyaan di atas berdasarkan isi dokumen berikut:\n\n"
                f"---\n{full_text}\n---"
            )
        else:
            user_prompt = (
                f"Buatkan ringkasan komprehensif dari dokumen berikut:\n\n"
                f"---\n{full_text}\n---"
            )

        if truncated:
            user_prompt += (
                "\n\n_(Catatan: Dokumen sangat panjang sehingga hanya sebagian yang ditampilkan. "
                "Ringkasan mungkin tidak mencakup seluruh isi dokumen.)_"
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            reply = await self._llm.chat(messages, max_tokens=4096)
        except Exception as exc:
            logger.exception("PDFSummarizerAgent LLM call failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                "❌ Terjadi kesalahan saat meringkas dokumen. Silakan coba lagi."
            )
            return task

        if not reply or not reply.strip():
            task.mark_failed("LLM tidak memberikan respons.")
            task.result = "❌ Gagal meringkas: LLM tidak menghasilkan respons. Coba lagi."
            return task

        task.mark_done(reply.strip())
        logger.info(
            "PDFSummarizerAgent done: session=%s reply_chars=%d",
            task.session_id, len(reply),
        )
        return task


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_user_question(user_input: str) -> str:
    """
    Ekstrak pertanyaan dari input user.
    Menghilangkan kata-kata meta seperti 'ringkas', 'summarize', dll.
    Kembalikan string kosong jika tidak ada pertanyaan spesifik.
    """
    if not user_input:
        return ""

    # Kata-kata yang menunjukkan permintaan ringkasan umum (bukan pertanyaan spesifik)
    summary_keywords = {
        "ringkas", "rangkum", "summarize", "summary", "apa isi",
        "ceritakan", "jelaskan", "apa yang ada", "kesimpulan",
        "ringkasan", "rangkuman", "tolong ringkas", "tolong rangkum",
        "buat ringkasan", "buat rangkuman",
    }

    lower_input = user_input.lower().strip()

    # Jika input hanya berisi kata kunci ringkasan, tidak ada pertanyaan spesifik
    for kw in summary_keywords:
        if lower_input == kw or lower_input == kw + " dokumen ini" or lower_input == kw + " ini":
            return ""

    # Jika ada tanda tanya → kemungkinan besar pertanyaan spesifik
    if "?" in user_input:
        return user_input

    # Jika kata pertama adalah kata kunci ringkasan → ringkasan umum
    first_word = lower_input.split()[0] if lower_input.split() else ""
    if first_word in {"ringkas", "rangkum", "summarize", "jelaskan", "ceritakan"}:
        # Cek apakah ada pertanyaan spesifik di dalamnya
        if len(user_input.split()) <= 4:
            return ""

    # Input lain yang lebih spesifik → anggap sebagai pertanyaan
    return user_input if len(user_input.split()) > 4 else ""
