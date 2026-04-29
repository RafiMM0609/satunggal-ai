"""
DiagramRendererTool – mengekstrak blok Mermaid dari Markdown dan merendernya
menjadi gambar PNG menggunakan `mermaid-cli` (mmdc).

Dependency (install di server):
    npm install -g @mermaid-js/mermaid-cli
    # setelah itu `mmdc` tersedia di PATH

Flow (dipanggil orchestrator SESUDAH TechnicalWriterAgent selesai):
  1. Baca Markdown dari task.metadata["document_markdown"].
  2. Ekstrak semua blok ```mermaid ... ```.
  3. Render tiap blok ke file PNG via `mmdc`.
  4. Ganti blok mermaid di Markdown dengan referensi gambar Markdown:
         ![Diagram N](path/to/diagram_N.png)
  5. Simpan Markdown yang sudah diperbarui kembali ke
     task.metadata["document_markdown"] (in-place update).
  6. Simpan daftar path PNG ke task.metadata["diagram_paths"].
  7. Return dict hasil untuk task.tool_results["diagram_renderer"].

Jika `mmdc` tidak ditemukan, tool membuat placeholder teks
sehingga dokumen tetap bisa diproses ke tahap berikutnya.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# Regex untuk menemukan blok ```mermaid ... ``` (termasuk multi-line)
_MERMAID_BLOCK_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)


class DiagramRendererTool(BaseTool):
    """Render blok Mermaid di dalam Markdown menjadi gambar PNG."""

    name = "diagram_renderer"
    description = (
        "Extract Mermaid diagram blocks from a Markdown document and render each one "
        "as a PNG image using mermaid-cli (mmdc). Replaces the raw Mermaid blocks in "
        "task.metadata['document_markdown'] with Markdown image references. "
        "Called by the orchestrator after TechnicalWriterAgent finishes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_markdown": {
                "type": "string",
                "description": "Full Markdown text containing Mermaid code blocks (set in task.metadata['document_markdown']).",
            },
        },
        "required": ["document_markdown"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "diagram_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths to the rendered PNG files.",
            },
            "diagram_count": {"type": "integer", "description": "Number of diagrams rendered."},
            "error":         {"type": "string", "description": "Present only on failure."},
        },
    }

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        markdown: str = task.metadata.get("document_markdown", "")
        if not markdown:
            logger.warning(
                "DiagramRendererTool: document_markdown kosong. session=%s",
                task.session_id,
            )
            return {"diagram_paths": [], "skipped": True}

        blocks = _MERMAID_BLOCK_RE.findall(markdown)
        if not blocks:
            logger.info(
                "DiagramRendererTool: tidak ada blok mermaid. session=%s",
                task.session_id,
            )
            task.metadata.setdefault("diagram_paths", [])
            return {"diagram_paths": [], "skipped": True}

        out_dir = _make_diagram_dir(task.session_id)
        diagram_paths: list[str] = []
        mmdc_available = shutil.which("mmdc") is not None

        if not mmdc_available:
            logger.warning(
                "DiagramRendererTool: mmdc tidak ditemukan. "
                "Install dengan: npm install -g @mermaid-js/mermaid-cli. "
                "Diagram akan diganti placeholder. session=%s",
                task.session_id,
            )

        updated_markdown = markdown
        for idx, mermaid_code in enumerate(blocks, start=1):
            png_path = os.path.join(out_dir, f"diagram_{idx}.png")

            if mmdc_available:
                png_path = _render_mermaid(mermaid_code, png_path, idx)
            else:
                png_path = None  # akan gunakan placeholder teks

            # Ganti blok mermaid pertama yang cocok di markdown
            if png_path and os.path.exists(png_path):
                replacement = f"![Diagram {idx}]({png_path})"
                diagram_paths.append(png_path)
            else:
                # Placeholder jika rendering gagal / mmdc tidak ada
                replacement = (
                    f"> **[Diagram {idx}]** *(render tidak tersedia — "
                    f"install mermaid-cli untuk menghasilkan gambar)*\n"
                    f">\n"
                    f"> ```mermaid\n"
                    + "\n".join(f"> {line}" for line in mermaid_code.strip().splitlines())
                    + "\n> ```"
                )

            updated_markdown = _replace_first_mermaid_block(
                updated_markdown, mermaid_code, replacement
            )

        # Update markdown in-place di task.metadata
        task.metadata["document_markdown"] = updated_markdown
        task.metadata["diagram_paths"] = diagram_paths

        logger.info(
            "DiagramRendererTool: %d diagram diproses, %d berhasil dirender. session=%s",
            len(blocks), len(diagram_paths), task.session_id,
        )
        return {
            "diagram_paths": diagram_paths,
            "total_diagrams": len(blocks),
            "rendered": len(diagram_paths),
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_diagram_dir(session_id: str) -> str:
    """Buat direktori sementara untuk menyimpan gambar diagram."""
    out_dir = os.path.join(
        tempfile.gettempdir(), "advance_ai_docs", session_id, "diagrams"
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _render_mermaid(mermaid_code: str, png_path: str, idx: int) -> str | None:
    """Jalankan mmdc untuk merender kode Mermaid ke PNG. Return path jika sukses."""
    tmp_mmd = png_path.replace(".png", ".mmd")
    try:
        with open(tmp_mmd, "w", encoding="utf-8") as f:
            f.write(mermaid_code.strip())

        result = subprocess.run(
            ["mmdc", "-i", tmp_mmd, "-o", png_path, "--backgroundColor", "white"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "DiagramRendererTool: mmdc gagal untuk diagram %d: %s",
                idx, result.stderr,
            )
            return None

        return png_path
    except subprocess.TimeoutExpired:
        logger.error("DiagramRendererTool: mmdc timeout untuk diagram %d", idx)
        return None
    except Exception as exc:
        logger.exception("DiagramRendererTool: error render diagram %d: %s", idx, exc)
        return None
    finally:
        # Hapus file .mmd sementara
        if os.path.exists(tmp_mmd):
            os.remove(tmp_mmd)


def _replace_first_mermaid_block(markdown: str, mermaid_code: str, replacement: str) -> str:
    """Ganti kemunculan pertama blok ```mermaid yang mengandung mermaid_code."""
    # Coba direct string match dulu (lebih cepat & tidak rentan whitespace regex)
    for prefix in ("```mermaid\n", "```mermaid \n"):
        target = prefix + mermaid_code + "```"
        if target in markdown:
            return markdown.replace(target, replacement, 1)

    # Fallback: regex dengan stripped content agar whitespace leading/trailing diabaikan
    stripped = mermaid_code.strip()
    escaped_code = re.escape(stripped)
    pattern = re.compile(
        r"```mermaid\s*\n\s*" + escaped_code + r"\s*\n?```",
        re.DOTALL,
    )
    result = pattern.sub(replacement, markdown, count=1)
    if result == markdown:
        # Jika masih tidak cocok, log warning agar mudah di-debug
        logger.warning(
            "DiagramRendererTool: tidak bisa menemukan blok mermaid untuk diganti. "
            "Konten mermaid (50 char pertama): %r",
            stripped[:50],
        )
    return result
