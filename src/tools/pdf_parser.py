"""
PDFParserTool – Ekstraksi teks bersih dari file PDF.

Menggunakan PyMuPDF (fitz) sebagai backend utama.
Membagi teks menjadi chunk-chunk berukuran ~2.000 kata untuk menghindari
penggunaan memori yang berlebihan pada proses batch.

Tool ini dipanggil SEBELUM QuizAgent oleh orchestrator khusus PDF
(process_pdf_quiz di main_loop.py), bukan melalui pipeline gatekeeper biasa.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# Jumlah kata per chunk (disesuaikan untuk VPS 2 GB RAM)
_WORDS_PER_CHUNK = 2_000
# Jumlah halaman maksimum per satu kali baca (anti-OOM)
_PAGES_PER_READ  = 10


class PDFParserTool(BaseTool):
    """
    Ekstraksi teks dari PDF menggunakan PyMuPDF (fitz).

    Input (dari task.metadata):
        "pdf_path": str – path absolut ke file PDF sementara

    Output (dict):
        "chunks":       list[str]  – potongan teks, siap diproses oleh QuizAgent
        "total_pages":  int        – jumlah halaman PDF
        "total_words":  int        – perkiraan total kata
        "filename":     str        – nama file asli
    """

    name = "pdf_parser"
    description = (
        "Extract clean text from a PDF file using PyMuPDF (fitz). "
        "Splits the text into chunks of ~2 000 words for LLM context management. "
        "Set task.metadata['pdf_path'] to the absolute path of the PDF before calling."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Absolute path to the PDF file (set in task.metadata['pdf_path']).",
            },
        },
        "required": ["pdf_path"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "chunks":      {"type": "array", "items": {"type": "string"}, "description": "Text chunks ready for LLM processing."},
            "total_pages": {"type": "integer", "description": "Total number of pages in the PDF."},
            "total_words": {"type": "integer", "description": "Estimated total word count."},
            "filename":    {"type": "string",  "description": "Original file name."},
            "error":       {"type": "string",  "description": "Present only on failure."},
        },
    }

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        pdf_path: str = task.metadata.get("pdf_path", "")
        if not pdf_path or not os.path.isfile(pdf_path):
            logger.error("PDFParserTool: pdf_path missing or invalid. session=%s", task.session_id)
            return {"error": "pdf_path tidak valid atau file tidak ditemukan."}

        filename = os.path.basename(pdf_path)
        logger.info("PDFParserTool: parsing %s session=%s", filename, task.session_id)

        try:
            import fitz  # PyMuPDF  # noqa: PLC0415
        except ImportError:
            logger.error("PyMuPDF (fitz) tidak terinstal. Jalankan: pip install PyMuPDF")
            return {"error": "PyMuPDF tidak terinstal di server ini."}

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.exception("PDFParserTool: failed to open PDF %s: %s", pdf_path, exc)
            return {"error": f"Gagal membuka PDF: {exc}"}

        total_pages = len(doc)
        all_text_parts: list[str] = []

        # Baca halaman per kelompok untuk membatasi penggunaan RAM
        # pdf_max_pages: jika di-set, hanya baca N halaman pertama (Quick Peek mode)
        max_pages = task.metadata.get("pdf_max_pages", None)
        read_up_to = min(total_pages, max_pages) if max_pages else total_pages
        for start in range(0, read_up_to, _PAGES_PER_READ):
            end = min(start + _PAGES_PER_READ, read_up_to)
            page_texts: list[str] = []
            for page_num in range(start, end):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    page_texts.append(text)
                except Exception as exc:
                    logger.warning(
                        "PDFParserTool: error on page %d: %s", page_num, exc
                    )
            all_text_parts.append("\n".join(page_texts))

        doc.close()

        # Gabungkan dan bersihkan teks
        raw_text = "\n".join(all_text_parts)
        clean_text = _clean_text(raw_text)

        if not clean_text.strip():
            logger.warning("PDFParserTool: extracted text is empty. PDF may be image-based.")
            return {
                "error": (
                    "Teks PDF kosong. Kemungkinan PDF berbasis gambar (scan). "
                    "Gunakan PDF dengan teks yang dapat dipilih."
                )
            }

        # Bagi menjadi chunk berdasarkan jumlah kata
        chunks = _split_into_chunks(clean_text, _WORDS_PER_CHUNK)
        total_words = len(clean_text.split())

        logger.info(
            "PDFParserTool: parsed OK – pages=%d read_up_to=%d words=%d chunks=%d is_partial=%s session=%s",
            total_pages, read_up_to, total_words, len(chunks), bool(max_pages and max_pages < total_pages), task.session_id,
        )

        return {
            "chunks":      chunks,
            "total_pages": total_pages,
            "total_words": total_words,
            "filename":    filename,
            **({
                "is_partial": True,
                "peeked_pages": read_up_to,
            } if max_pages and max_pages < total_pages else {}),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Bersihkan teks hasil ekstraksi PDF dari artefak OCR dan karakter tidak berguna."""
    # Hapus baris kosong berulang
    text = re.sub(r"\n{3,}", "\n\n", raw)
    # Hapus spasi berlebih
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Hapus karakter kontrol kecuali newline dan tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _split_into_chunks(text: str, words_per_chunk: int) -> list[str]:
    """
    Bagi teks menjadi chunk berdasarkan jumlah kata.

    Mencoba memotong di batas kalimat/paragraf agar konteks tidak terpotong di tengah.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])

        # Coba cari titik terakhir di chunk untuk memotong di batas kalimat
        if end < len(words):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            cut = max(last_period, last_newline)
            if cut > 0:
                chunk = chunk[:cut + 1]
                # Hitung ulang berapa kata yang dipakai
                actual_words = len(chunk.split())
                end = start + actual_words

        chunks.append(chunk.strip())
        start = end

    return [c for c in chunks if c.strip()]
