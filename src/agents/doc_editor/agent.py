"""
DocEditorAgent – Agent untuk mengedit dokumen .docx dengan presisi XML-level.

Alur kerja:
──────────────────────────────────────────────────────────────────────────────
  Langkah 1: Baca peta paragraf dari file .docx (index, text, style)
  Langkah 2: Kirim instruksi edit + peta paragraf ke LLM
  Langkah 3: Parse operasi edit (JSON) dari output LLM
  Langkah 4: Terapkan edit menggunakan DocxEditorTool (XML-level precision)
  Langkah 5: Kirim laporan perubahan + path file yang sudah diedit

Mode edit terdeteksi ketika user_caption mengandung kata kerja edit dalam
Bahasa Indonesia (ubah, ganti, tambah, hapus, perbaiki, revisi, dll.) atau
task.metadata["edit_mode"] = True sudah di-set oleh process_docx.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[str], Awaitable[None]]]

# ── System prompt untuk LLM ───────────────────────────────────────────────────

_EDITOR_SYSTEM_PROMPT = """\
Kamu adalah Document Editor AI yang mengedit dokumen Word (.docx) dengan presisi.

Kamu diberikan:
1. Daftar paragraf bernomor dari dokumen (format: [indeks] STYLE: teks)
2. Instruksi edit dari pengguna

Tugasmu: Analisis instruksi dan hasilkan daftar operasi edit dalam format JSON.

Format operasi yang tersedia:

1. Ganti teks (run-level, tanpa merusak format bold/italic):
   {"op": "replace_text", "find": "teks lama", "replace": "teks baru", "paragraph_index": <int|null>}
   → paragraph_index=null berarti cari di seluruh dokumen

2. Tambah paragraf baru (mewarisi style/numbering dari referensi):
   {"op": "add_paragraph", "text": "teks paragraf baru", "after_paragraph_index": <int>, "style_from_index": <int|null>}
   → style_from_index=null berarti warisi dari after_paragraph_index
   → Untuk item daftar bernomor, style_from_index harus menunjuk ke item daftar di atasnya agar numId sama

3. Hapus paragraf:
   {"op": "delete_paragraph", "paragraph_index": <int>}

4. Ganti seluruh konten paragraf (mempertahankan pPr/numbering):
   {"op": "replace_paragraph", "paragraph_index": <int>, "new_text": "konten baru"}

CATATAN PENTING:
- paragraph_index adalah indeks 0-based (dimulai dari 0)
- Gunakan "replace_text" untuk perubahan kecil (lebih aman untuk format)
- Gunakan "replace_paragraph" hanya jika konten paragraf berubah total
- Untuk menambah item list/nomor, gunakan "add_paragraph" dengan style_from_index ke item list terdekat
- Jika instruksi tidak jelas, kembalikan edits: [] dan jelaskan di summary

Output HARUS berupa JSON valid di dalam blok ```json ... ```.
Format output wajib:
```json
{
  "summary": "Ringkasan perubahan dalam Bahasa Indonesia",
  "edits": [
    <operasi edit 1>,
    <operasi edit 2>,
    ...
  ]
}
```
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _notify(cb: StatusCallback, text: str) -> None:
    if cb is None:
        return
    try:
        await cb(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("DocEditorAgent progress callback raised: %s", exc)


def _build_paragraph_list(paragraph_map: list[dict]) -> str:
    """Format daftar paragraf untuk prompt LLM."""
    lines = []
    for para in paragraph_map:
        idx = para["index"]
        text = para["text"]
        style = para["style"]
        flags = []
        if para.get("is_heading"):
            flags.append("HEADING")
        if para.get("is_list"):
            flags.append("LIST")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        text_preview = text[:120] + ("..." if len(text) > 120 else "")
        lines.append(f"[{idx}] {style}{flag_str}: {text_preview}")
    return "\n".join(lines) if lines else "(dokumen kosong)"


def _extract_json_from_llm(text: str) -> dict | None:
    """Ekstrak JSON dari output LLM (cari blok ```json ... ```)."""
    # Cari blok ```json ... ```
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.warning("DocEditorAgent: JSON decode error in ```json block: %s", exc)

    # Fallback: cari JSON object di seluruh teks
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, text)
    for m in reversed(matches):  # Coba dari yang terpanjang
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    return None


def _build_step_msg(step: int, total: int, name: str, doc_title: str) -> str:
    """Bangun pesan progres step-planned."""
    steps = [
        ("📋", "Membaca peta paragraf dokumen"),
        ("🧠", "Menganalisis instruksi edit (LLM)"),
        ("✂️", "Menerapkan perubahan XML-level"),
        ("📤", "Menyiapkan dokumen hasil edit"),
    ]
    lines = [f"⏳ *Mengedit Dokumen*\n📄 _{doc_title}_\n"]
    for i, (icon, sname) in enumerate(steps, start=1):
        if i < step:
            lines.append(f"  {icon} ~~{sname}~~ ✅")
        elif i == step:
            lines.append(f"  {icon} *{sname}* ← _sedang berjalan..._")
        else:
            lines.append(f"  {icon} {sname} _(menunggu)_")
    lines.append(f"\n_Langkah {step} dari {total}_")
    return "\n".join(lines)


# ── DocEditorAgent ─────────────────────────────────────────────────────────────

class DocEditorAgent(BaseAgent):
    """
    Agent untuk mengedit dokumen .docx dengan presisi di level XML.

    Input (via task.metadata):
        docx_path (str):        Path ke file .docx yang akan diedit.
        doc_title (str):        Judul dokumen.
        original_filename (str): Nama file asli.
        status_callback:        Progress callback.

    Input (via task.user_input):
        Instruksi edit dari pengguna.

    Output:
        task.result              → Laporan teks perubahan yang dilakukan.
        task.metadata["document_path"] → Path file .docx yang sudah diedit.
    """

    name = "doc_editor"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        from src.tools.docx_editor import DocxEditorTool, get_paragraph_map

        docx_path: str | None = task.metadata.get("docx_path")
        doc_title: str = task.metadata.get("doc_title", "Dokumen")
        original_filename: str = task.metadata.get("original_filename", "document.docx")
        status_cb: StatusCallback = task.metadata.get("status_callback")
        edit_instruction: str = task.user_input or ""
        total_steps = 4
        session_id = task.session_id

        if not docx_path:
            task.mark_failed("docx_path tidak ditemukan di task.metadata")
            task.result = "❌ Tidak ada file .docx untuk diedit."
            return task

        if not edit_instruction.strip():
            task.mark_failed("Instruksi edit kosong")
            task.result = "❌ Instruksi edit tidak ditemukan. Mohon sertakan instruksi di caption file."
            return task

        # ── Langkah 1: Baca peta paragraf ─────────────────────────────────
        await _notify(
            status_cb,
            _build_step_msg(1, total_steps, "Membaca peta paragraf", doc_title),
        )

        try:
            paragraph_map = get_paragraph_map(docx_path)
        except Exception as exc:
            logger.exception("DocEditorAgent: get_paragraph_map failed session=%s: %s", session_id, exc)
            task.mark_failed(str(exc))
            task.result = f"❌ Gagal membaca struktur dokumen: {exc}"
            return task

        total_paragraphs = len(paragraph_map)
        logger.info(
            "DocEditorAgent: session=%s file=%r paragraphs=%d instruction=%r",
            session_id, original_filename, total_paragraphs, edit_instruction[:100],
        )

        # ── Langkah 2: Analisis instruksi via LLM ─────────────────────────
        await _notify(
            status_cb,
            _build_step_msg(2, total_steps, "Menganalisis instruksi edit", doc_title),
        )

        paragraph_list_str = _build_paragraph_list(paragraph_map)
        user_prompt = (
            f"Dokumen: **{doc_title}** ({original_filename})\n"
            f"Total paragraf: {total_paragraphs}\n\n"
            f"=== DAFTAR PARAGRAF ===\n{paragraph_list_str}\n\n"
            f"=== INSTRUKSI EDIT ===\n{edit_instruction}"
        )

        messages = [
            {"role": "system", "content": _EDITOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            llm_output = await self._llm.chat(messages, max_tokens=2048, temperature=0.1)
        except Exception as exc:
            logger.exception("DocEditorAgent: LLM call failed session=%s: %s", session_id, exc)
            task.mark_failed(str(exc))
            task.result = "❌ Gagal menganalisis instruksi edit. Silakan coba lagi."
            return task

        # Parse JSON dari output LLM
        parsed = _extract_json_from_llm(llm_output)
        if parsed is None:
            logger.warning(
                "DocEditorAgent: Could not parse JSON from LLM output session=%s. Output: %r",
                session_id, llm_output[:200],
            )
            task.mark_done(
                f"⚠️ Tidak dapat menghasilkan operasi edit yang valid.\n\n"
                f"Respon AI:\n{llm_output[:500]}"
            )
            return task

        edits: list[dict] = parsed.get("edits", [])
        edit_summary: str = parsed.get("summary", "Perubahan telah dianalisis.")

        if not edits:
            task.mark_done(
                f"ℹ️ *Tidak ada perubahan yang diperlukan.*\n\n"
                f"**Analisis:** {edit_summary}"
            )
            return task

        logger.info(
            "DocEditorAgent: %d edit operations planned session=%s",
            len(edits), session_id,
        )

        # ── Langkah 3: Terapkan perubahan ─────────────────────────────────
        await _notify(
            status_cb,
            _build_step_msg(3, total_steps, "Menerapkan perubahan XML-level", doc_title),
        )

        task.metadata["docx_edits"] = edits
        editor_tool = DocxEditorTool()
        edit_result = await editor_tool.run(task)

        if "error" in edit_result:
            logger.error(
                "DocEditorAgent: DocxEditorTool failed session=%s: %s",
                session_id, edit_result["error"],
            )
            task.mark_failed(edit_result["error"])
            task.result = f"❌ Gagal menerapkan perubahan: {edit_result['error']}"
            return task

        changes_made: int = edit_result.get("changes_made", 0)
        details: list[str] = edit_result.get("details", [])
        edited_path: str = edit_result.get("edited_docx_path", "")

        # ── Langkah 4: Laporan ─────────────────────────────────────────────
        await _notify(
            status_cb,
            _build_step_msg(4, total_steps, "Menyiapkan dokumen hasil edit", doc_title),
        )

        # Simpan ke history
        self._history.add(
            session_id, "assistant",
            f"[Edit DOCX] {changes_made} perubahan diterapkan pada {original_filename}",
        )

        # Bangun laporan teks
        report_lines = [
            f"# ✅ Dokumen Berhasil Diedit\n",
            f"**File:** _{original_filename}_",
            f"**Jumlah perubahan:** {changes_made}\n",
            f"**Ringkasan:** {edit_summary}\n",
        ]

        if details:
            report_lines.append("## Detail Perubahan\n")
            for d in details[:20]:  # Batas 20 item untuk readability
                report_lines.append(f"- {d}")
            if len(details) > 20:
                report_lines.append(f"- _...dan {len(details) - 20} perubahan lainnya_")

        report_lines.append(
            "\n---\n"
            "📎 *File Word yang sudah diedit terlampir di bawah.*"
        )

        task.mark_done("\n".join(report_lines))
        logger.info(
            "DocEditorAgent: done session=%s changes=%d file=%r",
            session_id, changes_made, edited_path,
        )
        return task
