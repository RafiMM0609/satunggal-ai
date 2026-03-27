"""
DocxParserTool – membedah dokumen .docx menjadi seksi-seksi berdasarkan Heading.

Output:
    {
        "doc_title":        str,   # Judul dokumen (Heading 1 pertama atau nama file)
        "sections":         list[dict],
        "total_sections":   int,
        "total_words":      int,
        "detection_method": str,   # "formal" | "heuristic"
    }

Setiap elemen "sections":
    {
        "index":        int,   # Nomor urut (mulai 1)
        "title":        str,   # Teks heading
        "level":        int,   # 1 = Heading 1, 2 = Heading 2, dst.
        "content":      str,   # Teks paragraf di bawah heading ini (hingga heading berikutnya)
        "is_heuristic": bool,  # True jika bab dideteksi via heuristik (bukan Heading style)
    }

Paragraf yang berada SEBELUM heading pertama dimasukkan ke seksi dengan title "Pendahuluan"
agar tidak ada konten yang terlewat.

Deteksi dua tahap:
  1. Formal   – gunakan Heading style Word (jika ≥ 2 heading formal ditemukan)
  2. Heuristic – fallback untuk dokumen tanpa Heading style:
       • Bold penuh + teks pendek (< 80 char)
       • ALL CAPS + ≥ 3 kata + teks pendek
       • Pola numbering: BAB/CHAPTER/BAGIAN/SECTION + nomor, atau angka bertitik (1. / 1.1.)
"""

from __future__ import annotations

import logging
import os
import re
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

# Pola numbering untuk heading informal
_RE_CHAPTER_KEYWORD = re.compile(
    r"^(BAB|CHAPTER|BAGIAN|SECTION|PART|MODUL|MODULE)\s*[\-–]?\s*[IVXLC\d]+",
    re.IGNORECASE,
)
_RE_NUMERIC_HEADING = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?\s+\S")


def _is_heading(paragraph) -> int:
    """Return heading level (1-5) jika paragraf menggunakan Heading style Word, else 0."""
    style_name = paragraph.style.name.lower()
    return _HEADING_STYLES.get(style_name, 0)


def _is_informal_heading(paragraph) -> int:
    """
    Heuristic: deteksi bab pada dokumen tanpa Heading style formal.

    Mengembalikan level heading (1 atau 2) jika paragraf teridentifikasi
    sebagai judul bab/seksi, atau 0 jika bukan.

    Kriteria:
      - Bold penuh + teks pendek (< 80 char, ≥ 2 kata)
      - ALL CAPS + ≥ 3 kata + teks pendek (< 100 char)
      - Pola kata kunci: BAB/CHAPTER/BAGIAN/SECTION + nomor → level 1
      - Pola angka bertitik: "1. Judul" → level 1, "1.1 Subjudul" → level 2
    """
    text = paragraph.text.strip()
    if not text or len(text) > 120:
        return 0

    words = text.split()
    if len(words) < 2:
        return 0

    # Cek pola kata kunci bab (prioritas tertinggi → selalu level 1)
    if _RE_CHAPTER_KEYWORD.match(text):
        return 1

    # Cek pola angka bertitik
    num_match = _RE_NUMERIC_HEADING.match(text)
    if num_match:
        if num_match.group(3):   # "1.1.1 ..." → level 3
            return 3
        elif num_match.group(2): # "1.1 ..."   → level 2
            return 2
        else:                    # "1. ..."    → level 1
            return 1

    # Cek ALL CAPS (semua kata huruf besar, ≥ 3 kata, teks < 100 char)
    if len(text) < 100 and len(words) >= 3:
        alpha_words = [w for w in words if w.isalpha()]
        if alpha_words and all(w.isupper() for w in alpha_words):
            return 1

    # Cek bold penuh (semua run dengan teks adalah bold)
    if len(text) < 80:
        runs_with_text = [r for r in paragraph.runs if r.text.strip()]
        if runs_with_text and all(r.bold for r in runs_with_text):
            return 1

    return 0


def parse_docx_to_sections(docx_path: str) -> dict[str, Any]:
    """
    Membedah file .docx menjadi seksi-seksi terstruktur.

    Deteksi dua tahap:
      - Jika dokumen memiliki ≥ 2 paragraf dengan Heading style formal → mode "formal"
      - Jika tidak → aktifkan heuristic fallback → mode "heuristic"

    Args:
        docx_path: Path absolut ke file .docx.

    Returns:
        Dict berisi doc_title, sections, total_sections, total_words, detection_method.

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
    paragraphs = doc.paragraphs

    # ── Pass 1: hitung formal heading ──────────────────────────────────────
    formal_heading_count = sum(1 for p in paragraphs if _is_heading(p) > 0)
    use_heuristic = formal_heading_count < 2
    detection_method = "heuristic" if use_heuristic else "formal"

    logger.info(
        "docx_parser: file=%r formal_headings=%d → mode=%s",
        docx_path, formal_heading_count, detection_method,
    )

    # ── Pass 2: bangun seksi ───────────────────────────────────────────────
    sections: list[dict[str, Any]] = []
    current_title: str = ""
    current_level: int = 0
    current_paragraphs: list[str] = []
    current_heuristic: bool = False
    doc_title: str = ""
    found_first_heading: bool = False
    section_idx = 0

    def _flush_section(
        title: str, level: int, paragraphs: list[str], idx: int, is_heuristic: bool
    ) -> None:
        content = "\n".join(p for p in paragraphs if p.strip())
        sections.append({
            "index":        idx,
            "title":        title,
            "level":        level,
            "content":      content,
            "is_heuristic": is_heuristic,
        })

    for para in paragraphs:
        text = para.text.strip()
        style_name = para.style.name.lower()

        # Tangkap judul dokumen dari gaya Title
        if style_name in _TITLE_STYLES and text and not doc_title:
            doc_title = text
            continue

        # Tentukan level heading paragraf ini
        level = _is_heading(para)
        is_heuristic_para = False
        if level == 0 and use_heuristic:
            level = _is_informal_heading(para)
            if level > 0:
                is_heuristic_para = True

        if level > 0:
            # Simpan seksi sebelumnya
            if found_first_heading or current_paragraphs:
                section_idx += 1
                _flush_section(
                    title=current_title or "Pendahuluan",
                    level=current_level,
                    paragraphs=current_paragraphs,
                    idx=section_idx,
                    is_heuristic=current_heuristic,
                )

            # Set doc_title = Heading 1 pertama
            if level == 1 and not doc_title:
                doc_title = text

            current_title = text
            current_level = level
            current_paragraphs = []
            current_heuristic = is_heuristic_para
            found_first_heading = True
        else:
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
            is_heuristic=current_heuristic,
        )

    # Jika dokumen tidak memiliki heading sama sekali,
    # perlakukan seluruh dokumen sebagai satu seksi
    if not sections:
        all_text = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())
        sections.append({
            "index":        1,
            "title":        doc_title or os.path.basename(docx_path),
            "level":        1,
            "content":      all_text,
            "is_heuristic": False,
        })
        doc_title = doc_title or os.path.basename(docx_path)

    total_words = sum(
        len(s["content"].split()) + len(s["title"].split()) for s in sections
    )

    logger.info(
        "docx_parser: parsed %d sections (%s), %d words from %r",
        len(sections), detection_method, total_words, docx_path,
    )

    return {
        "doc_title":        doc_title or os.path.basename(docx_path),
        "sections":         sections,
        "total_sections":   len(sections),
        "total_words":      total_words,
        "detection_method": detection_method,
    }


class DocxParserTool(BaseTool):
    """
    Tool yang membedah file .docx menjadi seksi-seksi berdasarkan Heading.

    Input (via task.metadata):
        docx_path (str): Path absolut ke file .docx.

    Output:
        {
            "doc_title":        str,
            "sections":         list[dict],
            "total_sections":   int,
            "total_words":      int,
            "detection_method": str,   # "formal" | "heuristic"
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
