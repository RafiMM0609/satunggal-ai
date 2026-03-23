"""
QuizAgent – PDF-to-Web Quiz Generator

Alur Kerja:
  1. Menerima teks yang sudah diekstrak dari PDF (disimpan di task.metadata["pdf_chunks"])
  2. Memproses setiap chunk secara berurutan (sequential batch) untuk menghemat RAM
  3. Setiap putaran menghasilkan 10-15 soal dalam format JSON
  4. Mengumpulkan semua soal ke task.metadata["quiz_questions"]
  5. Menambahkan "web_quiz_builder" ke task.pending_tools untuk pembuatan HTML

Desain Anti-Crash:
  - Setiap batch diproses satu per satu (tidak paralel)
  - Cache teks PDF dihapus setelah chunk selesai diproses
  - Progres soal disimpan incremental di task.metadata
  - Jika LLM gagal pada satu batch, batch tersebut dilewati (tidak crash total)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Awaitable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah QuizMaster AI – spesialis pembuat soal ujian berkualitas tinggi.

Tugasmu adalah menganalisis teks yang diberikan dan menghasilkan soal-soal pilihan ganda \
yang menantang, akurat, dan edukatif. Soal harus menguji pemahaman mendalam, BUKAN sekadar \
hafalan verbatim dari teks.

## ATURAN KETAT ##

1. OUTPUT HARUS BERUPA JSON VALID SAJA. Tidak ada teks, penjelasan, atau komentar di luar JSON.
2. Kembalikan TEPAT sejumlah soal yang diminta dalam pesan user. Tidak kurang, tidak lebih.
3. Setiap soal WAJIB memiliki TEPAT 4 pilihan jawaban (A, B, C, D).
4. Field "correct" adalah INTEGER INDEX (0=A, 1=B, 2=C, 3=D) dari jawaban yang benar.
5. JANGAN pernah memberi kunci jawaban yang salah. Verifikasi kembali sebelum output.
6. Pilihan jawaban pengecoh (distraktor) harus masuk akal dan relevan dengan topik.
7. Gunakan Bahasa Indonesia yang baku dan formal.

## FORMAT JSON WAJIB ##
[
  {
    "id": 1,
    "question": "Pertanyaan yang jelas dan tidak ambigu?",
    "options": [
      "A. Pilihan pertama",
      "B. Pilihan kedua",
      "C. Pilihan ketiga",
      "D. Pilihan keempat"
    ],
    "correct": 0,
    "explanation": "Penjelasan singkat mengapa jawaban ini benar (1-2 kalimat)."
  }
]

## PANDUAN KUALITAS SOAL ##
- Variasikan tingkat kesulitan: 30% mudah, 50% sedang, 20% sulit.
- Hindari soal dengan kata "BUKAN" atau "KECUALI" lebih dari 20% dari total soal.
- Pastikan setiap soal berdiri sendiri dan tidak bergantung pada soal lain.
- Distribusikan soal secara merata di seluruh konten teks, bukan hanya bagian awal.
- Pilihan jawaban tidak boleh mengandung hint implisit (misal: pilihan terpanjang = benar).
"""


# ── QuizAgent ─────────────────────────────────────────────────────────────────

class QuizAgent(BaseAgent):
    """Generates quiz questions from PDF text chunks using batch LLM processing."""

    name = "quiz_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    # ── Public interface ──────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        """
        Main entrypoint. Expects:
          - task.metadata["pdf_chunks"]: list[str]  – text chunks from PDF
          - task.metadata["quiz_title"]: str         – optional title for the quiz
          - task.metadata["status_callback"]: callable – optional progress callback

        Produces:
          - task.metadata["quiz_questions"]: list[dict] – accumulated questions
          - task.pending_tools appended with "web_quiz_builder"
        """
        chunks: list[str] = task.metadata.get("pdf_chunks", [])
        if not chunks:
            task.mark_failed("Tidak ada konten PDF yang dapat diproses.")

            task.result = (
                "❌ Gagal membuat kuis: teks PDF kosong atau tidak dapat dibaca. "
                "Pastikan PDF berisi teks (bukan gambar hasil scan)."
            )
            return task

        quiz_title: str = task.metadata.get("quiz_title", "Kuis dari PDF")
        status_cb = task.metadata.get("status_callback")

        # Desired total question count (user-specified, e.g. "buat 30 soal")
        desired_total: int | None = task.metadata.get("quiz_question_count")

        import math as _math

        # Per-batch cap: asking for more than 15 at once causes LLM to truncate.
        _MAX_PER_BATCH = 15
        _DEFAULT_PER_BATCH = 10

        if desired_total:
            per_batch_count = min(desired_total, _MAX_PER_BATCH)
            # Ensure we have enough chunks/batches to cover the desired total.
            # If the PDF produced fewer chunks than needed, tile the existing
            # chunks so each batch can contribute per_batch_count questions.
            needed_batches = _math.ceil(desired_total / per_batch_count)
            if len(chunks) < needed_batches:
                # Repeat chunks cyclically until we have enough batches
                tiled = []
                for i in range(needed_batches):
                    tiled.append(chunks[i % len(chunks)])
                chunks = tiled
                logger.info(
                    "QuizAgent: tiled %d original chunk(s) into %d batches for %d desired questions. session=%s",
                    len(task.metadata.get("pdf_chunks", chunks)), needed_batches,
                    desired_total, task.session_id,
                )
        else:
            per_batch_count = _DEFAULT_PER_BATCH

        total_batches = len(chunks)
        all_questions: list[dict[str, Any]] = []
        question_counter = 0

        logger.info(
            "QuizAgent starting: session=%s batches=%d title=%r",
            task.session_id, total_batches, quiz_title,
        )

        for batch_idx, chunk_text in enumerate(chunks, start=1):
            # ── Update live progress ──────────────────────────────────────
            await _notify_quiz_progress(
                status_cb,
                quiz_title=quiz_title,
                questions_done=question_counter,
                batch_current=batch_idx,
                batch_total=total_batches,
                phase="generating",
            )

            # ── Call LLM for this chunk ───────────────────────────────────
            batch_questions = await self._process_chunk(
                chunk_text=chunk_text,
                batch_index=batch_idx,
                session_id=task.session_id,
                existing_count=question_counter,
                questions_per_batch=per_batch_count,
                existing_questions=all_questions,
            )

            # Free the chunk text from memory immediately after processing
            chunks[batch_idx - 1] = None  # type: ignore[assignment]

            if batch_questions:
                all_questions.extend(batch_questions)
                question_counter = len(all_questions)
                logger.info(
                    "QuizAgent batch %d/%d: +%d questions (total=%d) session=%s",
                    batch_idx, total_batches, len(batch_questions),
                    question_counter, task.session_id,
                )
                # Stop early if we've reached (or exceeded) the desired total
                if desired_total and question_counter >= desired_total:
                    logger.info(
                        "QuizAgent: reached desired total %d (have %d), stopping early. session=%s",
                        desired_total, question_counter, task.session_id,
                    )
                    break
            else:
                logger.warning(
                    "QuizAgent batch %d/%d: no questions extracted, skipping. session=%s",
                    batch_idx, total_batches, task.session_id,
                )

        # ── Build phase ───────────────────────────────────────────────────
        await _notify_quiz_progress(
            status_cb,
            quiz_title=quiz_title,
            questions_done=len(all_questions),
            batch_current=total_batches,
            batch_total=total_batches,
            phase="building",
        )

        if not all_questions:
            task.mark_failed("LLM tidak menghasilkan soal valid dari PDF ini.")
            task.result = (
                "❌ Gagal membuat kuis: tidak ada soal yang berhasil dibuat. "
                "Coba dengan PDF yang lebih padat kontennya."
            )
            return task

        # Re-number questions sequentially (trim to desired total if over)
        if desired_total and len(all_questions) > desired_total:
            all_questions = all_questions[:desired_total]
        for idx, q in enumerate(all_questions, start=1):
            q["id"] = idx

        # Store accumulated questions for the builder tool
        task.metadata["quiz_questions"] = all_questions
        task.metadata["quiz_title"] = quiz_title

        # Schedule the HTML builder tool
        task.pending_tools.append("web_quiz_builder")

        task.mark_done(
            f"✅ Kuis berhasil dibuat dengan *{len(all_questions)} soal*. "
            "Sedang membangun website interaktif..."
        )

        logger.info(
            "QuizAgent done: session=%s total_questions=%d",
            task.session_id, len(all_questions),
        )
        return task

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _process_chunk(
        self,
        chunk_text: str,
        batch_index: int,
        session_id: str,
        existing_count: int,
        questions_per_batch: int = 10,
        existing_questions: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Call the LLM with one text chunk and extract the questions JSON.

        Returns a (possibly empty) list of question dicts.
        """
        # Build anti-duplication hint when we already have questions from this text
        avoid_hint = ""
        if existing_questions:
            existing_stems = "\n".join(
                f"- {q['question']}" for q in existing_questions[-30:]
            )
            avoid_hint = (
                f"\n\nHINDARILAH soal yang mirip atau sama dengan soal yang sudah dibuat berikut ini:\n"
                f"{existing_stems}\n"
                f"Buat soal yang BERBEDA dari semua soal di atas."
            )

        user_prompt = (
            f"Berikut adalah teks sumber untuk Batch {batch_index}.\n"
            f"Buat TEPAT {questions_per_batch} soal pilihan ganda dari teks ini.\n"
            f"ID soal dimulai dari {existing_count + 1}.\n"
            f"{avoid_hint}\n\n"
            f"TEKS:\n{chunk_text}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # Scale max_tokens with the number of questions requested;
        # each question uses roughly 150 tokens of output.
        max_tokens = max(4096, questions_per_batch * 200)

        try:
            raw_reply = await self._llm.chat(
                messages,
                max_tokens=max_tokens,
                temperature=0.3,  # low temperature for consistent JSON format
                top_p=0.9,
            )
        except Exception as exc:
            logger.exception(
                "QuizAgent LLM call failed for batch=%d session=%s: %s",
                batch_index, session_id, exc,
            )
            return []

        return _parse_questions_json(raw_reply, batch_index, session_id)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_questions_json(
    raw: str, batch_index: int, session_id: str
) -> list[dict[str, Any]]:
    """Extract and validate question JSON from LLM output."""

    # 1. Try direct parse
    stripped = raw.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return _validate_questions(data)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return _validate_questions(data)
        except json.JSONDecodeError:
            pass

    # 3. Extract any JSON array
    match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return _validate_questions(data)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "QuizAgent: could not parse JSON from LLM for batch=%d session=%s. "
        "Raw (first 200 chars): %r",
        batch_index, session_id, raw[:200],
    )
    return []


def _validate_questions(raw_list: list) -> list[dict[str, Any]]:
    """Filter out malformed question objects."""
    valid: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        # Must have required fields
        if not all(k in item for k in ("question", "options", "correct")):
            continue
        # options must be a list of 4
        if not isinstance(item["options"], list) or len(item["options"]) != 4:
            continue
        # correct must be a valid index
        if not isinstance(item["correct"], int) or item["correct"] not in range(4):
            continue
        valid.append(item)
    return valid


# ── Progress notification ──────────────────────────────────────────────────────

StatusCallback = Optional[Callable[[str], Awaitable[None]]]


async def _notify_quiz_progress(
    cb: StatusCallback,
    *,
    quiz_title: str,
    questions_done: int,
    batch_current: int,
    batch_total: int,
    phase: str,
) -> None:
    """Send a structured quiz progress message via the status callback."""
    if cb is None:
        return

    pdf_status    = "✅ Selesai"
    gen_status    = f"⏳ [{questions_done} soal dibuat...]" if phase == "generating" else "✅ Selesai"
    build_status  = "🔄 Membangun..." if phase == "building" else "⏳ Menunggu"
    final_status  = "⏳ Menunggu"

    if phase == "building":
        build_status = "🔄 Membangun..."
    elif phase == "done":
        build_status = "✅ Selesai"
        final_status = "✅ Selesai"

    text = (
        f"⏳ *Proses Pembuatan Kuis Aktif*\n\n"
        f"📝 *{quiz_title}*\n\n"
        f"  • 📄 Membaca PDF: {pdf_status}\n"
        f"  • 🧠 Menghasilkan Soal: {gen_status}\n"
        f"  • 🏗️ Membangun Website: {build_status}\n"
        f"  • 📦 Finalisasi File: {final_status}\n\n"
        f"_(Batch {batch_current} dari {batch_total})_"
    )

    try:
        await cb(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("quiz progress callback raised: %s", exc)
