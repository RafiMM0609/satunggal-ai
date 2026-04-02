"""
TelegramQuizAgent – PDF-to-Telegram Poll Quiz Generator

Alur Kerja:
  1. Menerima teks PDF (task.metadata["pdf_chunks"])
  2. Memproses setiap chunk secara berurutan (sequential batch)
  3. Setiap batch menghasilkan soal dalam format JSON untuk Telegram sendPoll
  4. Menjalankan validation loop (self-correction) hingga 2 kali retry per batch
  5. Menyimpan state kuis ke SQLite via TgQuizStore
  6. Menyimpan soal tervalidasi ke task.metadata["tg_quiz_questions"]

Desain Anti-Crash:
  - Setiap batch diproses satu per satu (tidak paralel)
  - Cache teks PDF dihapus setelah chunk selesai diproses
  - Validation loop dengan feedback error agar LLM memperbaiki sendiri
  - Jika LLM gagal pada satu batch setelah retry, batch tersebut dilewati

Kompatibilitas Telegram:
  - Question: maks 300 karakter
  - Options: maks 100 karakter per pilihan (tanpa prefix "A. ")
  - Explanation: maks 200 karakter
  - correct_option_id: integer 0–3 (indeks berbasis 0)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Awaitable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.tg_quiz_agent.quiz_store import TgQuizStore
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── Telegram API limits ────────────────────────────────────────────────────────

_MAX_QUESTION_LEN  = 300
_MAX_OPTION_LEN    = 100
_MAX_EXPLANATION_LEN = 200

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah TelegramQuizMaster – spesialis pembuat soal kuis untuk dikirim via \
Telegram menggunakan fitur Poll (Quiz Mode).

Tugasmu adalah menganalisis teks yang diberikan dan menghasilkan soal-soal \
pilihan ganda yang menantang, akurat, dan edukatif.

## ATURAN KETAT ##

1. OUTPUT HARUS BERUPA JSON VALID SAJA. Tidak ada teks, penjelasan, atau komentar di luar JSON.
2. Kembalikan TEPAT sejumlah soal yang diminta. Tidak kurang, tidak lebih.
3. Setiap soal WAJIB memiliki TEPAT 4 pilihan jawaban.
4. Field "correct_option_id" adalah INTEGER INDEX 0-3 (0=pilihan pertama, 1=pilihan kedua, dst.).
5. PENTING: Pilihan jawaban TIDAK BOLEH menggunakan prefix "A. " / "B. " / "C. " / "D. ".
   Tulis langsung teks pilihan tanpa huruf depan.
6. JANGAN pernah memberi kunci jawaban yang salah. Verifikasi kembali sebelum output.
7. Pilihan jawaban pengecoh (distraktor) harus masuk akal dan relevan dengan topik.
8. Gunakan Bahasa Indonesia yang baku dan formal.
9. Panjang karakter (WAJIB dipatuhi karena ini limit API Telegram):
   - "question"    : maks 300 karakter
   - setiap option : maks 100 karakter
   - "explanation" : maks 200 karakter

## FORMAT JSON WAJIB ##
[
  {
    "question": "Pertanyaan yang jelas dan tidak ambigu? (maks 300 karakter)",
    "options": [
      "Pilihan pertama",
      "Pilihan kedua",
      "Pilihan ketiga",
      "Pilihan keempat"
    ],
    "correct_option_id": 0,
    "explanation": "Penjelasan singkat mengapa jawaban ini benar. (maks 200 karakter)"
  }
]

## PANDUAN KUALITAS SOAL ##
- Variasikan tingkat kesulitan: 30% mudah, 50% sedang, 20% sulit.
- Pastikan setiap soal berdiri sendiri dan tidak bergantung pada soal lain.
- Distribusikan soal secara merata di seluruh konten teks.
- Pilihan jawaban tidak boleh mengandung hint implisit.
"""

_CORRECTION_PROMPT = """\
Output sebelumnya tidak valid. Berikut detail errornya:
{errors}

Harap perbaiki dan kirimkan ulang HANYA JSON yang valid sesuai format yang diminta. \
Tidak ada teks di luar JSON.
"""


# ── TelegramQuizAgent ─────────────────────────────────────────────────────────

class TelegramQuizAgent(BaseAgent):
    """Generates quiz questions from PDF chunks, formatted for Telegram sendPoll."""

    name = "tg_quiz_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm   = llm or LLMClient()
        self._store = TgQuizStore()

    # ── Public interface ──────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        """
        Main entrypoint. Expects:
          - task.metadata["pdf_chunks"]:        list[str]  – text chunks from PDF
          - task.metadata["quiz_title"]:        str        – optional quiz title
          - task.metadata["quiz_question_count"]: int      – optional desired total
          - task.metadata["status_callback"]:   callable   – optional progress callback

        Produces:
          - task.metadata["tg_quiz_questions"]: list[dict] – Telegram-ready question dicts
          - task.metadata["tg_quiz_store_id"]:  int        – SQLite row ID for this session
        """
        chunks: list[str] = task.metadata.get("pdf_chunks", [])
        if not chunks:
            task.mark_failed("Tidak ada konten PDF yang dapat diproses.")
            task.result = (
                "❌ Gagal membuat kuis Telegram: teks PDF kosong atau tidak dapat dibaca. "
                "Pastikan PDF berisi teks (bukan gambar hasil scan)."
            )
            return task

        quiz_title: str = task.metadata.get("quiz_title", "Kuis Telegram dari PDF")
        status_cb       = task.metadata.get("status_callback")
        desired_total: int | None = task.metadata.get("quiz_question_count")

        import math as _math

        _MAX_PER_BATCH  = 10
        _DEFAULT_PER_BATCH = 5

        if desired_total:
            per_batch_count = min(desired_total, _MAX_PER_BATCH)
            needed_batches  = _math.ceil(desired_total / per_batch_count)
            if len(chunks) < needed_batches:
                tiled = [chunks[i % len(chunks)] for i in range(needed_batches)]
                chunks = tiled
                logger.info(
                    "TelegramQuizAgent: tiled chunks into %d batches for %d desired questions. session=%s",
                    needed_batches, desired_total, task.session_id,
                )
        else:
            per_batch_count = _DEFAULT_PER_BATCH

        total_batches = len(chunks)
        all_questions: list[dict[str, Any]] = []
        question_counter = 0

        logger.info(
            "TelegramQuizAgent starting: session=%s batches=%d title=%r",
            task.session_id, total_batches, quiz_title,
        )

        for batch_idx, chunk_text in enumerate(chunks, start=1):
            await _notify_progress(
                status_cb,
                quiz_title=quiz_title,
                questions_done=question_counter,
                batch_current=batch_idx,
                batch_total=total_batches,
            )

            batch_questions = await self._process_chunk_with_correction(
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
                    "TelegramQuizAgent batch %d/%d: +%d questions (total=%d) session=%s",
                    batch_idx, total_batches, len(batch_questions),
                    question_counter, task.session_id,
                )
                if desired_total and question_counter >= desired_total:
                    logger.info(
                        "TelegramQuizAgent: reached desired total %d (have %d), stopping early. session=%s",
                        desired_total, question_counter, task.session_id,
                    )
                    break
            else:
                logger.warning(
                    "TelegramQuizAgent batch %d/%d: no valid questions, skipping. session=%s",
                    batch_idx, total_batches, task.session_id,
                )

        if not all_questions:
            task.mark_failed("LLM tidak menghasilkan soal valid dari PDF ini.")
            task.result = (
                "❌ Gagal membuat kuis Telegram: tidak ada soal yang berhasil dibuat. "
                "Coba dengan PDF yang lebih padat kontennya."
            )
            return task

        # Trim to desired total
        if desired_total and len(all_questions) > desired_total:
            all_questions = all_questions[:desired_total]

        # Ensure Telegram API limits are respected
        all_questions = [_enforce_telegram_limits(q) for q in all_questions]

        # Persist to SQLite
        try:
            store_id = self._store.save_session(
                session_id=task.session_id,
                quiz_title=quiz_title,
                questions=all_questions,
            )
            task.metadata["tg_quiz_store_id"] = store_id
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TelegramQuizAgent: failed to persist quiz to SQLite: %s (session=%s)",
                exc, task.session_id,
            )

        task.metadata["tg_quiz_questions"] = all_questions
        task.metadata["quiz_title"]        = quiz_title

        task.mark_done(
            f"✅ Kuis Telegram berhasil dibuat dengan *{len(all_questions)} soal*. "
            "Mengirim soal ke chat..."
        )

        logger.info(
            "TelegramQuizAgent done: session=%s total_questions=%d",
            task.session_id, len(all_questions),
        )
        return task

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _process_chunk_with_correction(
        self,
        chunk_text: str,
        batch_index: int,
        session_id: str,
        existing_count: int,
        questions_per_batch: int = 5,
        existing_questions: list[dict] | None = None,
        max_retries: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Call the LLM with one text chunk, running a self-correction loop
        on validation failure (up to *max_retries* additional attempts).

        Returns a (possibly empty) list of validated question dicts.
        """
        avoid_hint = ""
        if existing_questions:
            existing_stems = "\n".join(
                f"- {q['question']}" for q in existing_questions[-20:]
            )
            avoid_hint = (
                f"\n\nHINDARILAH soal yang mirip dengan soal yang sudah dibuat:\n"
                f"{existing_stems}\n"
                f"Buat soal yang BERBEDA dari semua soal di atas."
            )

        user_prompt = (
            f"Berikut adalah teks sumber untuk Batch {batch_index}.\n"
            f"Buat TEPAT {questions_per_batch} soal pilihan ganda dari teks ini.\n"
            f"{avoid_hint}\n\n"
            f"TEKS:\n{chunk_text}"
        )

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        max_tokens = max(2048, questions_per_batch * 200)

        for attempt in range(max_retries + 1):
            try:
                raw_reply = await self._llm.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                )
            except Exception as exc:
                logger.exception(
                    "TelegramQuizAgent LLM call failed for batch=%d attempt=%d session=%s: %s",
                    batch_index, attempt, session_id, exc,
                )
                return []

            questions, errors = _parse_and_validate(raw_reply, batch_index, session_id)

            if questions:
                if attempt > 0:
                    logger.info(
                        "TelegramQuizAgent batch=%d: self-correction succeeded on attempt %d session=%s",
                        batch_index, attempt, session_id,
                    )
                return questions

            # No valid questions – attempt self-correction if retries remain
            if attempt < max_retries:
                correction = _CORRECTION_PROMPT.format(errors="; ".join(errors))
                messages.append({"role": "assistant", "content": raw_reply})
                messages.append({"role": "user",      "content": correction})
                logger.info(
                    "TelegramQuizAgent batch=%d: validation failed (attempt %d), retrying. errors=%s session=%s",
                    batch_index, attempt, errors, session_id,
                )
            else:
                logger.warning(
                    "TelegramQuizAgent batch=%d: gave up after %d attempts. session=%s",
                    batch_index, max_retries + 1, session_id,
                )

        return []


# ── JSON parsing & validation ─────────────────────────────────────────────────

def _parse_and_validate(
    raw: str,
    batch_index: int,
    session_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse LLM output and validate question structure.

    Returns:
        (valid_questions, error_messages)
        valid_questions is empty when parsing or validation completely fails.
    """
    stripped = raw.strip()

    # 1. Try direct JSON parse
    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fence
    if parsed is None:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # 3. Extract any JSON array pattern
    if parsed is None:
        match = re.search(r"\[.*\]", stripped, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if parsed is None or not isinstance(parsed, list):
        logger.warning(
            "TelegramQuizAgent: could not parse JSON from LLM for batch=%d session=%s. "
            "Raw (first 200 chars): %r",
            batch_index, session_id, raw[:200],
        )
        return [], ["Output bukan JSON array yang valid"]

    return _validate_questions(parsed)


def _validate_questions(
    raw_list: list,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Validate a list of question dicts against Telegram quiz requirements.

    Returns:
        (valid_questions, error_messages_for_invalid_items)
    """
    valid:  list[dict[str, Any]] = []
    errors: list[str]            = []

    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            errors.append(f"item[{idx}] bukan dict")
            continue

        # Required fields
        missing = [k for k in ("question", "options", "correct_option_id") if k not in item]
        if missing:
            errors.append(f"item[{idx}] kekurangan field: {missing}")
            continue

        # options must be a list of exactly 4
        opts = item["options"]
        if not isinstance(opts, list) or len(opts) != 4:
            errors.append(
                f"item[{idx}] options harus berupa list dengan tepat 4 pilihan "
                f"(ditemukan {len(opts) if isinstance(opts, list) else type(opts).__name__})"
            )
            continue

        # correct_option_id must be int in range(4)
        cid = item["correct_option_id"]
        if not isinstance(cid, int) or cid not in range(4):
            errors.append(
                f"item[{idx}] correct_option_id harus integer 0-3 (ditemukan {cid!r})"
            )
            continue

        valid.append(item)

    return valid, errors


def _enforce_telegram_limits(q: dict) -> dict:
    """
    Truncate fields to fit Telegram sendPoll API limits.

    Modifies a copy of *q* to avoid mutating the original.
    """
    result = dict(q)

    # question: max 300 chars
    if len(result.get("question", "")) > _MAX_QUESTION_LEN:
        result["question"] = result["question"][:_MAX_QUESTION_LEN - 1] + "…"

    # options: max 100 chars each, strip "A. " / "B. " / "C. " / "D. " prefixes
    cleaned_opts = []
    for opt in result.get("options", []):
        # Strip common prefix patterns like "A. ", "B. ", "(A) ", "1. " etc.
        opt = re.sub(r"^[A-Da-d1-4][\.\)]\s*", "", str(opt)).strip()
        if len(opt) > _MAX_OPTION_LEN:
            opt = opt[:_MAX_OPTION_LEN - 1] + "…"
        cleaned_opts.append(opt)
    result["options"] = cleaned_opts

    # explanation: max 200 chars
    explanation = result.get("explanation", "")
    if explanation and len(explanation) > _MAX_EXPLANATION_LEN:
        result["explanation"] = explanation[:_MAX_EXPLANATION_LEN - 1] + "…"

    return result


# ── Progress notification ──────────────────────────────────────────────────────

StatusCallback = Optional[Callable[[str], Awaitable[None]]]


async def _notify_progress(
    cb: StatusCallback,
    *,
    quiz_title: str,
    questions_done: int,
    batch_current: int,
    batch_total: int,
) -> None:
    """Send a structured quiz progress message via the status callback."""
    if cb is None:
        return

    text = (
        f"⏳ *Proses Pembuatan Kuis Telegram Aktif*\n\n"
        f"📝 *{quiz_title}*\n\n"
        f"  • 📄 Membaca PDF: ✅ Selesai\n"
        f"  • 🧠 Menghasilkan Soal: ⏳ [{questions_done} soal dibuat...]\n"
        f"  • 📨 Siap Kirim ke Telegram: ⏳ Menunggu\n\n"
        f"_(Batch {batch_current} dari {batch_total})_"
    )

    try:
        await cb(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tg_quiz progress callback raised: %s", exc)
