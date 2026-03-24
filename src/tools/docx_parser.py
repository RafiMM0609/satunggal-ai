"""
DocxParserTool – membedah dokumen .docx menjadi seksi-seksi berdasarkan Heading.

Output:
    {
        "doc_title":  str,                         # Judul dokumen (Heading 1 pertama atau nama file)
        "sections":   list[dict],                  # Daftar seksi terstruktur
        "total_sections": int,
        "total_words": int,
    }

Setiap elemen "sections":
    {
        "index":   int,   # Nomor urut (mulai 1)
        "title":   str,   # Teks heading
        "level":   int,   # 1 = Heading 1, 2 = Heading 2, dst.
        "content": str,   # Teks paragraf di bawah heading ini (hingga heading berikutnya)
    }

Paragraf yang berada SEBELUM heading pertama dimasukkan ke seksi dengan title "Pendahuluan"
agar tidak ada konten yang terlewat.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.tools.base_tool import BaseTool
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# Nama gaya heading yang dikenali (sesuai standar Microsoft Word)
_HEADING_STYLES = {
    "heading 1": 1,
    "heading 2": 2,
    "heading 3": 3,
    "heading 4": 4,
    "heading 5": 5,
}

# Gaya judul dokumen (bukan heading bab, tapi judul utama dokumen)
_TITLE_STYLES = {"title", "subtitle"}


def _is_heading(paragraph) -> int:
    """Return heading level (1-5) jika paragraf adalah heading, else 0."""
    style_name = paragraph.style.name.lower()
    return _HEADING_STYLES.get(style_name, 0)


def parse_docx_to_sections(docx_path: str) -> dict[str, Any]:
    """
    Membedah file .docx menjadi seksi-seksi terstruktur.

    Args:
        docx_path: Path absolut ke file .docx.

    Returns:
        Dict berisi doc_title, sections, total_sections, total_words.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        Exception: Jika python-docx gagal membaca file.
    """
    try:
        from docx import Document  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx tidak terinstal. Jalankan: pip install python-docx"
        ) from exc

    if not os.path.isfile(docx_path):
        raise FileNotFoundError(f"File tidak ditemukan: {docx_path}")

    doc = Document(docx_path)

    sections: list[dict[str, Any]] = []
    current_title: str = ""
    current_level: int = 0
    current_paragraphs: list[str] = []
    doc_title: str = ""
    found_first_heading: bool = False

    def _flush_section(title: str, level: int, paragraphs: list[str], idx: int) -> None:
        """Simpan seksi yang sedang dikumpulkan."""
        content = "\n".join(p for p in paragraphs if p.strip())
        sections.append({
            "index":   idx,
            "title":   title,
            "level":   level,
            "content": content,
        })

    section_idx = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        level = _is_heading(para)
        style_name = para.style.name.lower()

        # Tangkap judul dokumen dari gaya Title
        if style_name in _TITLE_STYLES and text and not doc_title:
            doc_title = text
            continue  # Judul tidak dihitung sebagai seksi

        if level > 0:
            # Simpan seksi sebelumnya (jika ada)
            if found_first_heading or current_paragraphs:
                section_idx += 1
                _flush_section(
                    title=current_title or "Pendahuluan",
                    level=current_level,
                    paragraphs=current_paragraphs,
                    idx=section_idx,
                )

            # Set doc_title = Heading 1 pertama
            if level == 1 and not doc_title:
                doc_title = text

            current_title = text
            current_level = level
            current_paragraphs = []
            found_first_heading = True

        else:
            # Konten biasa (bukan heading)
            if text:
                current_paragraphs.append(text)

    # Flush seksi terakhir
    if found_first_heading or current_paragraphs:
        section_idx += 1
        _flush_section(
            title=current_title or "Konten",
            level=current_level,
            paragraphs=current_paragraphs,
            idx=section_idx,
        )

    # Jika dokumen tidak memiliki heading sama sekali,
    # perlakukan seluruh dokumen sebagai satu seksi
    if not sections:
        all_text = "\n".join(
            p.text.strip() for p in doc.paragraphs if p.text.strip()
        )
        sections.append({
            "index":   1,
            "title":   doc_title or os.path.basename(docx_path),
            "level":   1,
            "content": all_text,
        })
        doc_title = doc_title or os.path.basename(docx_path)

    total_words = sum(
        len(s["content"].split()) + len(s["title"].split()) for s in sections
    )

    logger.info(
        "docx_parser: parsed %d sections, %d words from %r",
        len(sections), total_words, docx_path,
    )

    return {
        "doc_title":      doc_title or os.path.basename(docx_path),
        "sections":       sections,
        "total_sections": len(sections),
        "total_words":    total_words,
    }


class DocxParserTool(BaseTool):
    """
    Tool yang membedah file .docx menjadi seksi-seksi berdasarkan Heading.

    Input (via task.metadata):
        docx_path (str): Path absolut ke file .docx.

    Output:
        {
            "doc_title":      str,
            "sections":       list[dict],
            "total_sections": int,
            "total_words":    int,
        }
    """

    name = "docx_parser"

    async def run(self, task: AgentTask) -> dict[str, Any]:
        docx_path: str | None = task.metadata.get("docx_path")
        if not docx_path:
            return {"error": "docx_path tidak ditemukan di task.metadata"}

        try:
            result = parse_docx_to_sections(docx_path)
            return result
        except FileNotFoundError as exc:
            logger.error("DocxParserTool: %s", exc)
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("DocxParserTool unexpected error: %s", exc)
            return {"error": f"Gagal membaca file .docx: {exc}"}
