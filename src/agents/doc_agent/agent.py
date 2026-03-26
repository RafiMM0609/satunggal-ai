"""
DocAgent – Gabungan DocAuditorAgent + DocEditorAgent.

Satu agent yang menangani seluruh siklus hidup dokumen .docx:

  MODE 1 – ANALISIS (saat file .docx baru dikirim)
    Step 1: Validasi seksi dari DocxParserTool
    Step 2: Buat Daftar Isi
    Step 3: Ringkas setiap bab (batch LLM)
    Step 4: Simpan ke SQLite (DocIndex) bersama docx_path asli
    Step 5: Kirim laporan: Judul + Daftar Isi + Ringkasan per Bab

  MODE 2 – Q&A INTERAKTIF (pertanyaan umum tentang dokumen)
    - Cari bab relevan dari DocIndex berdasarkan kata kunci
    - Jawab menggunakan LLM + konteks bab + histori percakapan

  MODE 3 – KUMPULKAN EDIT (instruksi edit selama sesi Q&A)
    - Baca peta paragraf dari file .docx asli
    - Kirim instruksi + peta ke LLM → hasilkan operasi edit JSON
    - Simpan operasi ke antrian `doc_pending_edits` di DocIndex
    - Konfirmasi ke user + tetap jawab pertanyaan via Q&A

  MODE 4 – TERAPKAN & KIRIM FILE (user meminta file hasil edit)
    - Ambil semua pending edits dari DocIndex
    - Terapkan SEMUANYA ke file .docx ASLI → file hasil edit
    - Set task.metadata["document_path"] agar handler mengirim file
    - Bersihkan antrian pending edits

Pendekatan XML-deferred: edit tidak langsung ditulis ke file saat instruksi
diterima; melainkan dikumpulkan sebagai JSON di SQLite, lalu diterapkan sekali
saja ke file asli ketika user meminta.  Ini mencegah akumulasi kerusakan format
akibat chaining edits berulang pada file yang sudah dimodifikasi.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import os
from typing import Any, Awaitable, Callable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.doc_index import get_doc_index
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[str], Awaitable[None]]]

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_SECTION_CONTENT_CHARS = 3000
_MAX_QA_CONTENT_CHARS      = 2000
_MAX_PARA_TEXT_CHARS       = 120   # preview di paragraph map

# ── Trigger keywords untuk "apply & send" ─────────────────────────────────────

_APPLY_TRIGGERS = frozenset({
    "berikan", "kirim", "kasih", "beri", "send", "terapkan", "apply",
})
_FILE_WORDS = frozenset({"file", "dokumen", "filenya", "dokumennya"})
_EDIT_WORDS = frozenset({
    "edit", "diedit", "editnya", "perubahan", "revisi", "semua", "sudah",
    "hasil", "yang",
})

# ── Edit intent keywords ───────────────────────────────────────────────────────

_EDIT_KEYWORDS = frozenset({
    "edit", "ubah", "ganti", "tambah", "hapus", "perbaiki", "revisi",
    "modifikasi", "perbarui", "update", "koreksi", "perbaikan", "replace",
    "rubah", "tukar", "sesuaikan", "sisipkan", "insert", "buang", "delete",
    "hilangkan", "timpa", "overwrite", "change", "modify", "remove", "correct",
    "fix", "rewrite", "revise", "append", "add", "alter", "adjustment",
    "sesuaikan", "pertegas", "perjelas", "lengkapi",
})

# ── System prompts ─────────────────────────────────────────────────────────────

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
Kamu adalah Document Assistant AI yang memiliki akses ke indeks dokumen yang sudah dianalisis.

Tugas: Jawab pertanyaan pengguna berdasarkan konten bab yang relevan dari dokumen.
- Berikan jawaban yang spesifik dan mengacu pada isi dokumen.
- Jika pertanyaan adalah instruksi edit (ubah, ganti, perbaiki, dll.), berikan draf konten
  yang telah direvisi sesuai instruksi — ini akan disimpan sebagai acuan pengeditan.
- Jika ada beberapa bab yang relevan, jelaskan perbedaan atau hubungannya.
- Jika pertanyaan tidak dapat dijawab dari dokumen, katakan dengan jelas.
- Gunakan bahasa yang sama dengan pengguna.
"""

_EDITOR_SYSTEM_PROMPT = """\
Kamu adalah Document Editor AI yang mengedit dokumen Word (.docx) dengan presisi.

Kamu diberikan:
1. Daftar paragraf bernomor dari dokumen (format: [indeks] STYLE: teks)
2. Instruksi edit dari pengguna (mungkin disertai konten draf yang sudah direvisi)

Tugasmu: Analisis instruksi dan hasilkan daftar operasi edit dalam format JSON.

Format operasi yang tersedia:

1. Ganti teks (run-level, tanpa merusak format bold/italic):
   {"op": "replace_text", "find": "teks lama", "replace": "teks baru", "paragraph_index": <int|null>}
   → paragraph_index=null berarti cari di seluruh dokumen

2. Tambah paragraf baru (mewarisi style/numbering dari referensi):
   {"op": "add_paragraph", "text": "teks paragraf baru", "after_paragraph_index": <int>, "style_from_index": <int|null>}

3. Hapus paragraf:
   {"op": "delete_paragraph", "paragraph_index": <int>}

4. Ganti seluruh konten paragraf (mempertahankan pPr/numbering):
   {"op": "replace_paragraph", "paragraph_index": <int>, "new_text": "konten baru"}

CATATAN PENTING:
- paragraph_index adalah indeks 0-based (dimulai dari 0)
- Gunakan "replace_text" untuk perubahan kecil (lebih aman untuk format)
- Gunakan "replace_paragraph" hanya jika konten paragraf berubah total
- Jika instruksi tidak jelas, kembalikan edits: [] dan jelaskan di summary

Output HARUS berupa JSON valid di dalam blok ```json ... ```.
Format output wajib:
```json
{
  "summary": "Ringkasan perubahan dalam Bahasa Indonesia",
  "edits": [
    <operasi edit 1>,
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
        logger.debug("DocAgent progress callback raised: %s", exc)


def _is_apply_trigger(text: str) -> bool:
    """Deteksi apakah user meminta file hasil penerapan semua instruksi edit."""
    if not text or not text.strip():
        return False
    words = {w.strip(",.!?;:\"'-()/\\") for w in text.lower().split()}

    # Pattern A: terapkan/apply + (edit|perubahan|revisi|semua|file|dokumen)
    if {"terapkan", "apply"} & words:
        if words & (_EDIT_WORDS | _FILE_WORDS):
            return True

    # Pattern B: berikan/kirim/kasih/beri + file/dokumen + edit/diedit/...
    if {"berikan", "kirim", "kasih", "beri", "send"} & words:
        if words & _FILE_WORDS and words & _EDIT_WORDS:
            return True

    # Pattern C: "berikan file" / "kirim dokumen" (tanpa kata edit)
    if {"berikan", "kirim", "kasih", "beri", "send"} & words:
        if words & _FILE_WORDS:
            return True

    return False


def _is_edit_intent(text: str) -> bool:
    """Deteksi apakah teks mengandung instruksi edit."""
    if not text or not text.strip():
        return False
    words = text.lower().split()
    return any(w.strip(",.!?;:\"'-()/\\") in _EDIT_KEYWORDS for w in words)


def _extract_json_from_llm(text: str) -> dict | None:
    """Ekstrak JSON dari output LLM (cari blok ```json ... ```)."""
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.warning("DocAgent: JSON decode error in ```json block: %s", exc)

    # Fallback: cari JSON object di seluruh teks
    matches = re.findall(r"\{[\s\S]*\}", text)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def _build_paragraph_list(paragraph_map: list[dict]) -> str:
    """Format daftar paragraf untuk prompt LLM editor."""
    lines = []
    for para in paragraph_map:
        idx   = para["index"]
        text  = para["text"]
        style = para["style"]
        flags = []
        if para.get("is_heading"):
            flags.append("HEADING")
        if para.get("is_list"):
            flags.append("LIST")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        preview  = text[:_MAX_PARA_TEXT_CHARS] + ("..." if len(text) > _MAX_PARA_TEXT_CHARS else "")
        lines.append(f"[{idx}] {style}{flag_str}: {preview}")
    return "\n".join(lines) if lines else "(dokumen kosong)"


def _build_step_analyze_msg(
    current_step: int,
    total_steps: int,
    doc_title: str,
    total_sections: int,
) -> str:
    steps = [
        ("📋", "Validasi seksi dokumen"),
        ("🗂️", "Membuat Daftar Isi"),
        ("🧠", f"Meringkas {total_sections} bab"),
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


def _build_step_edit_msg(step: int, total: int, doc_title: str) -> str:
    steps = [
        ("📋", "Membaca peta paragraf dokumen"),
        ("🧠", "Menganalisis instruksi edit (LLM)"),
        ("💾", "Menyimpan instruksi ke antrian"),
    ]
    lines = [f"⏳ *Menyimpan Instruksi Edit*\n📄 _{doc_title}_\n"]
    for i, (icon, name) in enumerate(steps, start=1):
        if i < step:
            lines.append(f"  {icon} ~~{name}~~ ✅")
        elif i == step:
            lines.append(f"  {icon} *{name}* ← _sedang berjalan..._")
        else:
            lines.append(f"  {icon} {name} _(menunggu)_")
    lines.append(f"\n_Langkah {step} dari {total}_")
    return "\n".join(lines)


def _build_final_report(
    doc_title: str,
    sections_with_summary: list[dict],
    original_filename: str,
) -> str:
    """Buat laporan lengkap: judul + daftar isi + ringkasan per bab."""
    parts: list[str] = []

    parts.append(f"# 📄 Laporan Analisis Dokumen\n**{doc_title}**\n_{original_filename}_\n")
    parts.append("---\n")

    toc_lines = ["## 📋 Daftar Isi\n"]
    for sec in sections_with_summary:
        level  = sec.get("level", 1)
        indent = "  " * (level - 1)
        idx    = sec.get("index") or sec.get("bab_index", "?")
        title  = sec.get("title") or sec.get("bab_title", "")
        toc_lines.append(f"{indent}{idx}. {title}")
    parts.append("\n".join(toc_lines))
    parts.append("\n---\n")

    parts.append("## 🧠 Ringkasan per Bab\n")
    for sec in sections_with_summary:
        idx     = sec.get("index") or sec.get("bab_index", "?")
        title   = sec.get("title") or sec.get("bab_title", "")
        summary = sec.get("summary") or "_Ringkasan tidak tersedia._"
        level   = sec.get("level", 1)
        prefix  = "#" * min(level + 2, 6)
        parts.append(f"{prefix} {idx}. {title}\n\n{summary}\n")

    parts.append(
        "\n---\n"
        "💡 *Tip:* Balas untuk bertanya atau mengedit bagian tertentu.\n"
        "Contoh edit: _\"ubah bagian Team Identity untuk mempertegas nama tim\"_\n"
        "Ketik _\"berikan file\"_ untuk mendapatkan dokumen dengan semua edit yang sudah dicatat."
    )
    return "\n".join(parts)


# ── DocAgent ──────────────────────────────────────────────────────────────────


class DocAgent(BaseAgent):
    """
    Merged agent: analisis + Q&A + edit interaktif + kirim file.

    Mode dipilih otomatis berdasarkan konteks task:

    ANALYZE  – task.metadata["docx_sections"] tersedia (file baru dikirim)
    APPLY    – user meminta file (berikan file / kirim / terapkan edit)
    EDIT     – user memberikan instruksi edit pada sesi aktif
    QA       – pertanyaan umum pada sesi aktif
    """

    name = "doc_agent"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history   = history
        self._llm       = llm or LLMClient()
        self._doc_index = get_doc_index()

    # ── Main entrypoint ───────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        sections:  list[dict] | None = task.metadata.get("docx_sections")
        status_cb: StatusCallback    = task.metadata.get("status_callback")
        user_text: str               = task.user_input or ""

        if sections is not None:
            return await self._analyze_document(task, sections, status_cb)

        has_doc = self._doc_index.has_document(task.session_id)
        if not has_doc:
            task.mark_done(
                "⚠️ Belum ada dokumen yang dianalisis untuk sesi ini.\n\n"
                "Silakan kirim file `.docx` terlebih dahulu untuk memulai analisis."
            )
            return task

        if _is_apply_trigger(user_text):
            n_pending = self._doc_index.get_pending_edit_count(task.session_id)
            if n_pending > 0:
                return await self._apply_and_send(task, status_cb)
            else:
                # Tidak ada pending edits – kirim file asli
                return await self._send_original(task, status_cb)

        if _is_edit_intent(user_text):
            return await self._collect_edit(task, status_cb)

        return await self._answer_question(task, status_cb)

    # ── MODE 1: Analisis Dokumen ──────────────────────────────────────────────

    async def _analyze_document(
        self,
        task: AgentTask,
        sections: list[dict],
        status_cb: StatusCallback,
    ) -> AgentTask:
        session_id        = task.session_id
        doc_title         = task.metadata.get("doc_title", "Dokumen")
        file_id           = task.metadata.get("docx_file_id", "document.docx")
        original_filename = task.metadata.get("original_filename", file_id)
        docx_path         = task.metadata.get("docx_path", "")
        total_sections    = len(sections)
        total_words       = task.metadata.get("total_words", 0)
        total_steps       = 5

        # Step 1: Validasi
        await _notify(status_cb, _build_step_analyze_msg(1, total_steps, doc_title, total_sections))

        if not sections:
            task.mark_failed("Dokumen tidak memiliki konten yang dapat dibaca.")
            task.result = (
                "❌ Gagal menganalisis dokumen: tidak ada teks yang dapat diekstrak. "
                "Pastikan file .docx berisi teks (bukan gambar)."
            )
            return task

        logger.info(
            "DocAgent: analyzing session=%s file=%r sections=%d",
            session_id, file_id, total_sections,
        )

        # Step 2: Daftar Isi (no LLM)
        await _notify(status_cb, _build_step_analyze_msg(2, total_steps, doc_title, total_sections))

        # Step 3: Ringkas per bab
        await _notify(status_cb, _build_step_analyze_msg(3, total_steps, doc_title, total_sections))

        sections_with_summary: list[dict[str, Any]] = []
        for sec in sections:
            summary = await self._summarize_section(sec, session_id)
            sec_copy = dict(sec)
            sec_copy["summary"] = summary
            sections_with_summary.append(sec_copy)

        # Step 4: Simpan ke database (dengan docx_path)
        await _notify(status_cb, _build_step_analyze_msg(4, total_steps, doc_title, total_sections))

        self._doc_index.save_document(
            session_id=session_id,
            file_id=file_id,
            doc_title=doc_title,
            sections=sections,
            total_words=total_words,
            docx_path=docx_path,
        )
        for sec in sections_with_summary:
            self._doc_index.save_summary(
                session_id=session_id,
                file_id=file_id,
                bab_index=sec["index"],
                summary=sec.get("summary", ""),
            )

        # Step 5: Laporan akhir
        await _notify(status_cb, _build_step_analyze_msg(5, total_steps, doc_title, total_sections))

        report = _build_final_report(doc_title, sections_with_summary, original_filename)
        task.mark_done(report)

        logger.info("DocAgent: analysis done session=%s file=%r", session_id, file_id)
        return task

    async def _summarize_section(self, section: dict, session_id: str) -> str:
        """Ringkas satu seksi via LLM."""
        title   = section.get("title", "")
        content = section.get("content", "")

        if not content.strip():
            return "_Bab ini tidak memiliki konten teks._"

        truncated = content[:_MAX_SECTION_CONTENT_CHARS]
        if len(content) > _MAX_SECTION_CONTENT_CHARS:
            truncated += "\n... _(konten dipotong untuk efisiensi)_"

        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Bab/Seksi: **{title}**\n\nKonten:\n{truncated}"},
        ]
        try:
            return await self._llm.chat(messages, max_tokens=1024, temperature=0.3)
        except Exception as exc:
            logger.exception("DocAgent: summarize failed bab=%r session=%s: %s", title, session_id, exc)
            return "_Gagal meringkas bab ini._"

    # ── MODE 2: Q&A Interaktif ────────────────────────────────────────────────

    async def _answer_question(
        self,
        task: AgentTask,
        status_cb: StatusCallback,
    ) -> AgentTask:
        session_id = task.session_id
        user_query = task.user_input

        await _notify(status_cb, "🔍 *Mencari bab yang relevan di dokumen...*")

        relevant_sections = self._doc_index.search_sections(
            session_id=session_id,
            query=user_query,
            limit=3,
        )

        context_parts = []
        for sec in relevant_sections:
            idx     = sec.get("bab_index", "?")
            title   = sec.get("bab_title", "")
            content = sec.get("content_text", "")
            summary = sec.get("summary", "")
            context_parts.append(
                f"### Bab {idx}: {title}\n\n"
                f"**Teks Asli:**\n{content[:_MAX_QA_CONTENT_CHARS]}\n\n"
                f"**Ringkasan:**\n{summary}"
            )

        doc_context = "\n\n---\n\n".join(context_parts)
        history_messages = self._history.get_as_llm_messages(session_id)

        await _notify(status_cb, "🧠 *Menjawab pertanyaan berdasarkan dokumen...*")

        user_prompt = (
            f"Pertanyaan/Instruksi: {user_query}\n\n"
            f"Konten bab yang relevan:\n\n{doc_context}"
        )

        messages = [{"role": "system", "content": _QA_SYSTEM_PROMPT}]
        messages.extend(history_messages[-6:])
        if not history_messages or history_messages[-1]["content"] != user_query:
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages[-1] = {"role": "user", "content": user_prompt}

        try:
            answer = await self._llm.chat(messages, max_tokens=2048, temperature=0.5)
            task.mark_done(answer)
        except Exception as exc:
            logger.exception("DocAgent QA LLM failed session=%s: %s", session_id, exc)
            task.result = "Maaf, terjadi kesalahan saat menjawab pertanyaan. Silakan coba lagi."
            task.mark_failed(str(exc))

        return task

    # ── MODE 3: Kumpulkan Edit ────────────────────────────────────────────────

    async def _collect_edit(
        self,
        task: AgentTask,
        status_cb: StatusCallback,
    ) -> AgentTask:
        """
        Terjemahkan instruksi edit ke operasi JSON, simpan ke antrian DocIndex.
        Setelah menyimpan, JUGA menjawab pertanyaan via Q&A sehingga user
        langsung mendapat draf konten yang direvisi.
        """
        from src.tools.docx_editor import get_paragraph_map

        session_id        = task.session_id
        user_instruction  = task.user_input or ""
        docx_path         = self._doc_index.get_docx_path(session_id)
        doc_meta          = self._doc_index.get_doc_meta(session_id)
        doc_title         = (doc_meta or {}).get("doc_title", "Dokumen")
        total_steps       = 3

        if not docx_path or not os.path.isfile(docx_path):
            # File tidak tersedia lagi – fallback ke Q&A
            logger.warning(
                "DocAgent: docx_path not found for session=%s path=%r; falling back to QA",
                session_id, docx_path,
            )
            return await self._answer_question(task, status_cb)

        # Step 1: Baca peta paragraf
        await _notify(status_cb, _build_step_edit_msg(1, total_steps, doc_title))

        try:
            paragraph_map = get_paragraph_map(docx_path)
        except Exception as exc:
            logger.exception("DocAgent: get_paragraph_map failed session=%s: %s", session_id, exc)
            # Fallback ke Q&A
            return await self._answer_question(task, status_cb)

        # Step 2: Analisis instruksi via LLM
        await _notify(status_cb, _build_step_edit_msg(2, total_steps, doc_title))

        paragraph_list_str = _build_paragraph_list(paragraph_map)
        user_prompt = (
            f"Dokumen: **{doc_title}**\n"
            f"Total paragraf: {len(paragraph_map)}\n\n"
            f"=== DAFTAR PARAGRAF ===\n{paragraph_list_str}\n\n"
            f"=== INSTRUKSI EDIT ===\n{user_instruction}"
        )

        messages = [
            {"role": "system", "content": _EDITOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            llm_output = await self._llm.chat(messages, max_tokens=2048, temperature=0.1)
        except Exception as exc:
            logger.exception("DocAgent: editor LLM failed session=%s: %s", session_id, exc)
            task.result = "❌ Gagal menganalisis instruksi edit. Silakan coba lagi."
            task.mark_failed(str(exc))
            return task

        parsed = _extract_json_from_llm(llm_output)

        if parsed is None or not parsed.get("edits"):
            # Tidak ada operasi – jawab Q&A + informasikan ke user
            qa_task = await self._answer_question(task, status_cb)
            n_pending = self._doc_index.get_pending_edit_count(session_id)
            suffix = (
                f"\n\n📌 *Instruksi ini dicatat sebagai catatan.* "
                f"({n_pending} edit tersimpan sejauh ini)\n"
                "Ketik _\"berikan file\"_ untuk menerapkan semua edit ke dokumen."
            )
            if qa_task.result:
                qa_task.result += suffix
            return qa_task

        edit_ops: list[dict] = parsed.get("edits", [])
        edit_summary: str    = parsed.get("summary", "")

        # Step 3: Simpan ke antrian
        await _notify(status_cb, _build_step_edit_msg(3, total_steps, doc_title))

        edit_order = self._doc_index.add_pending_edit(
            session_id=session_id,
            instruction=user_instruction,
            edit_ops=edit_ops,
        )
        n_pending = self._doc_index.get_pending_edit_count(session_id)

        logger.info(
            "DocAgent: queued edit #%d session=%s ops=%d total_pending=%d",
            edit_order, session_id, len(edit_ops), n_pending,
        )

        # Bangun respons: ringkasan edit + draf konten (via Q&A)
        # Jalankan Q&A untuk memberikan draf revisi yang bisa dilihat user
        qa_answer = ""
        try:
            qa_task_copy = AgentTask(session_id=session_id, user_input=user_instruction)
            qa_task_copy.metadata = dict(task.metadata)
            qa_result = await self._answer_question(qa_task_copy, None)
            qa_answer = qa_result.result or ""
        except Exception as exc:
            logger.debug("DocAgent: QA during edit failed: %s", exc)

        confirmation = (
            f"✅ *Edit #{edit_order} dicatat!*\n\n"
            f"**Ringkasan perubahan:** {edit_summary}\n"
            f"**Total edit tersimpan:** {n_pending} instruksi\n\n"
        )

        if qa_answer:
            confirmation += f"---\n\n{qa_answer}\n\n---\n\n"

        confirmation += (
            f"📌 Teruskan percakapan atau berikan instruksi edit lainnya.\n"
            f"Ketik _\"berikan file\"_ untuk menerapkan semua {n_pending} edit ke dokumen."
        )

        task.mark_done(confirmation)
        return task

    # ── MODE 4: Terapkan semua edit & kirim file ──────────────────────────────

    async def _apply_and_send(
        self,
        task: AgentTask,
        status_cb: StatusCallback,
    ) -> AgentTask:
        """
        Terapkan semua pending edits ke file .docx ASLI dan set document_path.
        Pendekatan deferred: semua operasi dijalankan sekali pada file asli,
        bukan chaining pada file yang sudah dimodifikasi.
        """
        from src.tools.docx_editor import DocxEditorTool

        session_id = task.session_id
        doc_meta   = self._doc_index.get_doc_meta(session_id)
        doc_title  = (doc_meta or {}).get("doc_title", "Dokumen")
        docx_path  = self._doc_index.get_docx_path(session_id)

        if not docx_path or not os.path.isfile(docx_path):
            task.mark_done(
                "⚠️ File .docx asli sudah tidak tersedia di server.\n\n"
                "Mohon upload ulang dokumen untuk melanjutkan pengeditan."
            )
            return task

        pending_edits = self._doc_index.get_pending_edits(session_id)
        n_pending     = len(pending_edits)

        await _notify(
            status_cb,
            f"⏳ *Menerapkan {n_pending} instruksi edit ke dokumen...*\n📄 _{doc_title}_",
        )

        # Gabungkan semua operasi edit dari seluruh antrian
        all_ops: list[dict] = []
        for pe in pending_edits:
            all_ops.extend(pe.get("edit_ops", []))

        if not all_ops:
            # Tidak ada ops konkret – kirim file asli
            task.metadata["document_path"] = docx_path
            task.mark_done(
                "ℹ️ *Belum ada perubahan konkret yang tercatat.*\n\n"
                "File dokumen asli dilampirkan."
            )
            return task

        logger.info(
            "DocAgent: applying %d total ops from %d pending edits session=%s",
            len(all_ops), n_pending, session_id,
        )

        # Buat output path
        out_dir = os.path.join(tempfile.gettempdir(), "advance_ai_docx_edited", session_id)
        os.makedirs(out_dir, exist_ok=True)
        basename = os.path.basename(docx_path)
        name, ext = os.path.splitext(basename)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(out_dir, f"{name}_edited_{ts}{ext}")

        # Terapkan semua ops sekaligus via DocxEditorTool
        task.metadata["docx_edits"]         = all_ops
        task.metadata["output_docx_path"]   = output_path

        editor = DocxEditorTool()
        edit_result = await editor.run(task)

        if "error" in edit_result:
            logger.error("DocAgent: DocxEditorTool failed session=%s: %s", session_id, edit_result["error"])
            task.mark_failed(edit_result["error"])
            task.result = f"❌ Gagal menerapkan perubahan: {edit_result['error']}"
            return task

        changes_made: int   = edit_result.get("changes_made", 0)
        details: list[str]  = edit_result.get("details", [])
        edited_path: str    = edit_result.get("edited_docx_path", output_path)

        # Bersihkan antrian setelah berhasil
        self._doc_index.clear_pending_edits(session_id)
        # Update docx_path ke file hasil edit agar edit selanjutnya berbasis file terbaru
        self._doc_index.save_docx_path(session_id, edited_path)

        # Simpan ke history
        self._history.add(
            session_id, "assistant",
            f"[Apply Edits] {n_pending} instruksi, {changes_made} perubahan diterapkan.",
        )

        # Set document_path agar handler mengirim file
        task.metadata["document_path"] = edited_path

        report_lines = [
            f"# ✅ Dokumen Berhasil Diedit\n",
            f"**{n_pending} instruksi edit diterapkan** ({changes_made} perubahan)\n",
        ]
        if details:
            report_lines.append("## Detail Perubahan\n")
            for d in details[:20]:
                report_lines.append(f"- {d}")
            if len(details) > 20:
                report_lines.append(f"- _...dan {len(details) - 20} perubahan lainnya_")
        report_lines.append(
            "\n---\n"
            "📎 *File Word yang sudah diedit terlampir di bawah.*\n\n"
            "💡 Anda masih bisa melanjutkan memberi instruksi edit untuk sesi ini."
        )

        task.mark_done("\n".join(report_lines))
        logger.info(
            "DocAgent: apply done session=%s changes=%d file=%r",
            session_id, changes_made, edited_path,
        )
        return task

    # ── Kirim file asli (tidak ada pending edits) ─────────────────────────────

    async def _send_original(
        self,
        task: AgentTask,
        status_cb: StatusCallback,
    ) -> AgentTask:
        """Kirim file asli ketika tidak ada pending edits."""
        session_id = task.session_id
        docx_path  = self._doc_index.get_docx_path(session_id)

        if docx_path and os.path.isfile(docx_path):
            task.metadata["document_path"] = docx_path
            task.mark_done(
                "📎 *File dokumen asli terlampir.*\n\n"
                "_(Belum ada instruksi edit yang tercatat untuk sesi ini.)_"
            )
        else:
            task.mark_done(
                "⚠️ File dokumen tidak tersedia.\n\n"
                "Silakan upload ulang file .docx untuk memulai sesi baru."
            )
        return task
