"""
DocumentGeneratorTool – mengompilasi Markdown (+ gambar) menjadi PDF atau Word.

Dependency (install sesuai format yang dibutuhkan):
    # Untuk PDF (pilih salah satu):
    pip install weasyprint markdown

    # Untuk Word (.docx):
    apt install pandoc   # atau: brew install pandoc
    # (opsional) sertakan template: data/templates/template.docx

Flow (dipanggil orchestrator SESUDAH DiagramRendererTool):
  1. Baca Markdown dari task.metadata["document_markdown"].
  2. Tentukan format dari task.metadata["output_format"] (default: pdf).
  3. Untuk PDF  → konversi Markdown ke HTML → render ke PDF via WeasyPrint.
  4. Untuk DOCX → panggil Pandoc: `pandoc input.md -o output.docx`.
  5. Simpan path file akhir ke task.metadata["document_path"].
  6. Return dict hasil untuk task.tool_results["document_generator"].
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# CSS dasar untuk tampilan PDF yang rapi
_PDF_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1a1a2e;
    padding: 0;
}

@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @bottom-center {
        content: "Halaman " counter(page) " dari " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}

h1 { font-size: 20pt; color: #0f3460; margin: 0 0 0.4em; border-bottom: 2px solid #0f3460; padding-bottom: 0.2em; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 14pt; color: #16213e; margin: 1.2em 0 0.4em; }
h3 { font-size: 12pt; color: #1a1a2e; margin: 1em 0 0.3em; }

p   { margin: 0.5em 0; }
ul, ol { margin: 0.5em 0 0.5em 1.8em; }
li  { margin-bottom: 0.25em; }

code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.5pt;
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 0.1em 0.4em;
}

pre {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #0f3460;
    border-radius: 4px;
    padding: 0.8em 1em;
    overflow-x: auto;
    margin: 0.8em 0;
}
pre code { background: none; border: none; padding: 0; }

blockquote {
    border-left: 4px solid #0f3460;
    background: #f0f4ff;
    padding: 0.5em 1em;
    margin: 0.8em 0;
    border-radius: 0 4px 4px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
}
th, td {
    border: 1px solid #ccc;
    padding: 0.5em 0.8em;
    text-align: left;
}
th { background: #0f3460; color: white; font-weight: 600; }
tr:nth-child(even) { background: #f9f9f9; }

img { max-width: 100%; height: auto; margin: 0.8em 0; display: block; border-radius: 4px; }

a { color: #0f3460; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2em 0; }
"""


class DocumentGeneratorTool(BaseTool):
    """Compile Markdown + gambar menjadi PDF atau Word (.docx)."""

    name = "document_generator"

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        markdown: str = task.metadata.get("document_markdown", "")
        if not markdown:
            logger.error(
                "DocumentGeneratorTool: document_markdown kosong. session=%s",
                task.session_id,
            )
            return {"error": "document_markdown tidak ditemukan di task.metadata"}

        output_format: str = task.metadata.get("output_format", "pdf").lower()
        out_dir = _make_output_dir(task.session_id)

        if output_format == "docx":
            result = await _generate_docx(markdown, out_dir, task.session_id)
        else:
            # Default ke PDF
            result = await _generate_pdf(markdown, out_dir, task.session_id)

        if "document_path" in result:
            task.metadata["document_path"] = result["document_path"]
            logger.info(
                "DocumentGeneratorTool: file tersimpan di %s. session=%s",
                result["document_path"], task.session_id,
            )

        return result


# ── PDF via WeasyPrint ─────────────────────────────────────────────────────────

async def _generate_pdf(
    markdown: str, out_dir: str, session_id: str
) -> dict[str, Any]:
    """Konversi Markdown → HTML → PDF menggunakan WeasyPrint."""
    try:
        import markdown as md_lib           # noqa: PLC0415
        from weasyprint import HTML, CSS   # noqa: PLC0415
    except ImportError as exc:
        logger.error("DocumentGeneratorTool: %s. Install: pip install weasyprint markdown", exc)
        return {"error": f"Dependency tidak tersedia: {exc}. Install: pip install weasyprint markdown"}

    # Markdown → HTML
    html_content = md_lib.markdown(
        markdown,
        extensions=["tables", "fenced_code", "toc", "nl2br", "attr_list"],
    )
    full_html = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
</head>
<body>
{html_content}
</body>
</html>"""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"document_{session_id}_{ts}.pdf")

    try:
        HTML(string=full_html, base_url=out_dir).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=_PDF_CSS)],
        )
        return {"document_path": pdf_path, "format": "pdf"}
    except Exception as exc:
        logger.exception("DocumentGeneratorTool: WeasyPrint gagal: %s", exc)
        return {"error": f"Gagal membuat PDF: {exc}"}


# ── DOCX via Pandoc ────────────────────────────────────────────────────────────

async def _generate_docx(
    markdown: str, out_dir: str, session_id: str
) -> dict[str, Any]:
    """Konversi Markdown → DOCX menggunakan Pandoc."""
    if not shutil.which("pandoc"):
        logger.error("DocumentGeneratorTool: pandoc tidak ditemukan.")
        return {
            "error": "pandoc tidak ditemukan. Install: sudo apt install pandoc"
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path   = os.path.join(out_dir, f"document_{session_id}_{ts}.md")
    docx_path = os.path.join(out_dir, f"document_{session_id}_{ts}.docx")

    # Tulis Markdown ke file sementara
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    # Cari template Word custom (opsional)
    template_path = _find_docx_template()
    cmd = ["pandoc", md_path, "-o", docx_path, "--from", "markdown", "--to", "docx"]
    if template_path:
        cmd += ["--reference-doc", template_path]
        logger.debug("DocumentGeneratorTool: menggunakan template %s", template_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error("DocumentGeneratorTool: pandoc error: %s", result.stderr)
            return {"error": f"Pandoc gagal: {result.stderr}"}
        return {"document_path": docx_path, "format": "docx"}
    except subprocess.TimeoutExpired:
        logger.error("DocumentGeneratorTool: pandoc timeout")
        return {"error": "Pandoc timeout (>60 detik)"}
    except Exception as exc:
        logger.exception("DocumentGeneratorTool: error: %s", exc)
        return {"error": str(exc)}
    finally:
        # Hapus file Markdown sementara
        if os.path.exists(md_path):
            os.remove(md_path)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_output_dir(session_id: str) -> str:
    """Buat direktori output dokumen."""
    out_dir = os.path.join(
        tempfile.gettempdir(), "advance_ai_docs", session_id
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _find_docx_template() -> str | None:
    """Cari template Word di direktori data/templates/."""
    candidates = [
        os.path.join("data", "templates", "template.docx"),
        os.path.join("data", "template.docx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None
