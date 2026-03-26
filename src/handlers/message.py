"""Message handlers: text, photo, document (PDF), and unknown messages."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import tempfile

import telegramify_markdown
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src.orchestrator.main_loop import process_message, process_pdf, process_docx, is_edit_intent

logger = logging.getLogger(__name__)


_MAX_MSG_LEN = 4096
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB upload limit for all file types
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

# ── Temp directory for doc session copies ─────────────────────────────────────
# Analyzed .docx files are copied here so they remain available for editing
# after the original temp download is deleted.
_SESSIONS_TMP_DIR = os.path.join(tempfile.gettempdir(), "advance_ai_doc_sessions")

# ── Edit suggestion constants ──────────────────────────────────────────────────
_MAX_SECTIONS_FOR_SUGGESTIONS = 6    # Jumlah bab yang diperiksa untuk saran edit
_MAX_SUGGESTION_TEXT_LEN      = 120  # Panjang maksimum teks saran per bab (karakter)
_MAX_SUGGESTIONS_DISPLAY      = 3    # Jumlah maksimum saran yang ditampilkan
# Batas karakter saran AI yang disertakan ke prompt DocEditorAgent.
# Cukup panjang untuk menangkap draf konten lengkap, namun tetap aman dari
# token limit LLM.  Pemotongan dilakukan di batas karakter (bukan kata), sehingga
# mungkin terpotong di tengah kalimat – namun LLM tetap mendapat konteks utama.
_MAX_AI_SUGGESTION_FOR_EDITOR = 2000

# ── Pending doc-auditor session store ─────────────────────────────────────────
# Maps user_id (str) → {"docx_path": str, "doc_title": str, "original_filename": str}
# Populated after DocAuditorAgent successfully analyzes a document so the user
# can later ask questions OR request edits without re-uploading the file.
_pending_doc_sessions: dict[str, dict] = {}

# ── Pending doc-edit session store ────────────────────────────────────────────
# Maps user_id (str) → {"docx_path": str, "doc_title": str, "original_filename": str}
# Populated after a successful doc edit; cleared when the user confirms they
# are done and the final file is sent.
_pending_edit_sessions: dict[str, dict] = {}

# ── Pending chat-edit queue ────────────────────────────────────────────────────
# Maps user_id (str) → list of edit instruction strings collected during a
# DocAuditorAgent Q&A session.  Accumulated while the user chats, then applied
# all at once when they say "berikan saya file yang sudah diedit".
_pending_chat_edits: dict[str, list[str]] = {}

# ── Keywords that trigger "apply all accumulated edits and send file" ──────────
_APPLY_EDITS_KEYWORDS = frozenset({
    # Indonesian send/give
    "berikan", "kirim", "kasih", "beri", "send",
    # Indonesian apply
    "terapkan", "apply",
})
_APPLY_EDITS_FILE_WORDS = frozenset({"file", "dokumen", "filenya", "dokumennya"})
_APPLY_EDITS_EDIT_WORDS = frozenset({
    "edit", "diedit", "editnya", "perubahan", "revisi", "semua",
    "sudah", "hasil", "yang",
})


def _is_apply_edits_trigger(text: str) -> bool:
    """Deteksi apakah user meminta file hasil penerapan semua instruksi edit dari chat.

    Mengembalikan True untuk frasa seperti:
    - "berikan saya file yang sudah diedit"
    - "kirim file editnya"
    - "terapkan semua edit"
    - "apply semua perubahan"
    """
    if not text or not text.strip():
        return False
    words = {w.strip(",.!?;:\"'-()/\\") for w in text.lower().split()}

    # Pattern A: terapkan/apply + (edit|perubahan|revisi|semua)
    if {"terapkan", "apply"} & words:
        if words & (_APPLY_EDITS_EDIT_WORDS | _APPLY_EDITS_FILE_WORDS):
            return True

    # Pattern B: berikan/kirim/kasih/beri + file/dokumen + edit/diedit/...
    if {"berikan", "kirim", "kasih", "beri", "send"} & words:
        if words & _APPLY_EDITS_FILE_WORDS and words & _APPLY_EDITS_EDIT_WORDS:
            return True

    return False


def _split_text(text: str, max_len: int = _MAX_MSG_LEN) -> list[str]:
    """Split *text* into chunks that each fit within Telegram's message length limit.

    Prefers splitting at newline boundaries to avoid cutting mid-sentence.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Prefer splitting at a newline
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


async def _safe_reply(message, text: str) -> None:
    """Send *text* with MarkdownV2, auto-splitting if too long.

    Each chunk is converted with telegramify-markdown then sent with MarkdownV2;
    if Telegram rejects the formatting the same chunk is retried as plain text
    so the message is never lost.
    """
    chunks = _split_text(text)
    for chunk in chunks:
        try:
            formatted = telegramify_markdown.markdownify(chunk)
            await message.reply_text(formatted, parse_mode="MarkdownV2", quote=True)
        except BadRequest as exc:
            logger.warning("MarkdownV2 parse failed (%s), retrying as plain text.", exc)
            try:
                await message.reply_text(chunk, parse_mode=None, quote=True)
            except BadRequest as exc2:
                logger.error("Failed to send chunk even as plain text: %s", exc2)


def _get_edit_suggestions(session_id: str) -> str:
    """Ambil saran pengeditan berdasarkan catatan kualitas dari DocAuditorAgent.

    Membaca ringkasan per bab dari DocIndex dan mengekstrak catatan kualitas
    yang bisa disampaikan kepada pengguna sebagai saran perbaikan lanjutan.
    """
    try:
        from src.memory.doc_index import get_doc_index
        doc_index = get_doc_index()
        sections = doc_index.get_sections(session_id)
        suggestions: list[str] = []
        for sec in sections[:_MAX_SECTIONS_FOR_SUGGESTIONS]:  # Periksa bab pertama saja
            summary = sec.get("summary") or ""
            if "**Catatan Kualitas:**" in summary:
                parts = summary.split("**Catatan Kualitas:**", 1)
                if len(parts) > 1:
                    quality_note = parts[1].strip()
                    first_line = quality_note.split("\n")[0].strip(" -•*")
                    if first_line and "Tidak ada catatan khusus" not in first_line:
                        title = sec.get("bab_title") or f"Bab {sec.get('bab_index', '?')}"
                        suggestions.append(f"• *{title}*: {first_line[:_MAX_SUGGESTION_TEXT_LEN]}")
        if suggestions:
            return "\n".join(suggestions[:_MAX_SUGGESTIONS_DISPLAY])
    except Exception as exc:
        logger.debug("_get_edit_suggestions: %s", exc)
    return ""


def _build_follow_up_question(suggestions: str = "") -> str:
    """Bangun pesan tawaran lanjutan edit, opsional disertai saran kualitas."""
    msg = (
        "✏️ *Ada bagian lain yang perlu diedit lagi?*\n\n"
        "Jika **ya**, langsung ketik instruksi editnya.\n"
        "Jika **tidak**, ketik *selesai* untuk menerima file."
    )
    if suggestions:
        msg += f"\n\n💡 *Saran perbaikan dari analisis dokumen:*\n{suggestions}"
    return msg


# Alias untuk kompatibilitas kode lama yang masih menggunakan konstanta ini
_FOLLOW_UP_QUESTION = (
    "✏️ Ada bagian lain yang perlu diedit lagi?\n\n"
    "Jika **ya**, langsung ketik instruksi editnya.\n"
    "Jika **tidak**, ketik *tidak* atau *selesai* untuk menerima file."
)


async def _send_edited_docx(message, context, user, user_id_str: str, session: dict) -> None:
    """Kirim file .docx hasil edit ke user dan hapus sesi pending."""
    docx_path = session["docx_path"]
    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action="upload_document")
        with open(docx_path, "rb") as f:
            await message.reply_document(
                document=f,
                filename=os.path.basename(docx_path),
                caption="📝 File Word yang sudah diedit siap diunduh.",
                quote=True,
            )
        logger.info("Sent edited DOCX to user=%s path=%s", user.id, docx_path)
        _pending_edit_sessions.pop(user_id_str, None)
    except Exception as exc:
        logger.exception("Failed to send edited DOCX to user=%s: %s", user.id, exc)
        _pending_edit_sessions.pop(user_id_str, None)
        await message.reply_text(
            "⚠️ Gagal mengirim file hasil edit. Coba lagi nanti.", quote=True
        )
    finally:
        try:
            os.remove(docx_path)
        except OSError as exc:
            logger.debug("Could not remove edited DOCX %s: %s", docx_path, exc)


async def _handle_pending_edit(
    message, context, user, user_id_str: str, session: dict, edit_instruction: str
) -> None:
    """Terapkan instruksi edit tambahan pada file hasil edit sebelumnya, lalu tanya lagi."""
    old_docx_path = session["docx_path"]
    original_filename = session["original_filename"]

    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify("⏳ *Menerapkan edit tambahan...*"),
        parse_mode="MarkdownV2",
        quote=True,
    )

    async def _progress_callback(rendered_text: str) -> None:
        try:
            formatted = telegramify_markdown.markdownify(rendered_text)
            await context.bot.edit_message_text(
                chat_id=progress_msg.chat_id,
                message_id=progress_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pending edit progress update skipped: %s", exc)

    try:
        task = await process_docx(
            session_id=user_id_str,
            docx_path=old_docx_path,
            original_filename=original_filename,
            user_caption=edit_instruction,
            status_callback=_progress_callback,
        )
    except Exception as exc:
        logger.exception("Re-edit failed for user=%s: %s", user.id, exc)
        task = None

    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception:  # noqa: BLE001
        pass

    if task is None or task.status.value == "failed":
        error_msg = (task.result if task else None) or "❌ Gagal mengedit dokumen. Coba lagi."
        await _safe_reply(message, error_msg)
        return

    reply = task.result or "✅ Edit tambahan berhasil diterapkan."
    await _safe_reply(message, reply)

    new_docx_path = task.metadata.get("document_path")
    if new_docx_path and new_docx_path.lower().endswith(".docx"):
        # Hapus file lama jika sudah digantikan file baru
        if old_docx_path != new_docx_path:
            try:
                os.remove(old_docx_path)
            except OSError as exc:
                logger.debug("Could not remove old DOCX %s: %s", old_docx_path, exc)
        # Perbarui sesi dan tanya lagi, sertakan saran dari analisis dokumen
        _pending_edit_sessions[user_id_str] = {
            "docx_path": new_docx_path,
            "doc_title": task.metadata.get("doc_title", original_filename),
            "original_filename": original_filename,
        }
        suggestions = _get_edit_suggestions(user_id_str)
        await _safe_reply(message, _build_follow_up_question(suggestions))
    else:
        # Tidak ada file baru – bersihkan sesi
        _pending_edit_sessions.pop(user_id_str, None)
        try:
            os.remove(old_docx_path)
        except OSError as exc:
            logger.debug("Could not remove DOCX %s: %s", old_docx_path, exc)


async def _handle_audited_doc_edit(
    message, context, user, user_id_str: str, session: dict, edit_instruction: str
) -> None:
    """Terapkan instruksi edit pada dokumen yang sebelumnya dianalisis oleh DocAuditorAgent.

    Dipanggil ketika pengguna memiliki sesi analisis aktif (_pending_doc_sessions)
    dan mengirim instruksi edit. Setelah berhasil, sesi dipindahkan ke
    _pending_edit_sessions agar pengguna dapat melakukan edit lanjutan.
    """
    docx_path = session["docx_path"]
    doc_title = session.get("doc_title", "Dokumen")
    original_filename = session.get("original_filename", "document.docx")

    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify("⏳ *Menerapkan edit pada dokumen...*"),
        parse_mode="MarkdownV2",
        quote=True,
    )

    async def _progress_callback(rendered_text: str) -> None:
        try:
            formatted = telegramify_markdown.markdownify(rendered_text)
            await context.bot.edit_message_text(
                chat_id=progress_msg.chat_id,
                message_id=progress_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Audited doc edit progress update skipped: %s", exc)

    try:
        task = await process_docx(
            session_id=user_id_str,
            docx_path=docx_path,
            original_filename=original_filename,
            user_caption=edit_instruction,
            status_callback=_progress_callback,
        )
    except Exception as exc:
        logger.exception("Audited doc edit failed for user=%s: %s", user.id, exc)
        task = None

    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception:  # noqa: BLE001
        pass

    if task is None or task.status.value == "failed":
        error_msg = (task.result if task else None) or "❌ Gagal mengedit dokumen. Coba lagi."
        await _safe_reply(message, error_msg)
        return

    reply = task.result or "✅ Edit berhasil diterapkan."
    await _safe_reply(message, reply)

    new_docx_path = task.metadata.get("document_path")
    if new_docx_path and new_docx_path.lower().endswith(".docx"):
        # Pindahkan dari doc_sessions ke edit_sessions
        _pending_doc_sessions.pop(user_id_str, None)
        if docx_path != new_docx_path:
            try:
                os.remove(docx_path)
            except OSError as exc:
                logger.debug("Could not remove analyzed DOCX %s: %s", docx_path, exc)
        _pending_edit_sessions[user_id_str] = {
            "docx_path": new_docx_path,
            "doc_title": task.metadata.get("doc_title", doc_title),
            "original_filename": original_filename,
        }
        suggestions = _get_edit_suggestions(user_id_str)
        await _safe_reply(message, _build_follow_up_question(suggestions))
        logger.info(
            "Doc session converted to edit session for user=%s path=%s",
            user.id, new_docx_path,
        )
    else:
        # Tidak ada file baru (tidak ada perubahan) – kembalikan ke sesi analisis
        logger.info(
            "No new file from edit for user=%s; doc session kept", user.id
        )


async def _apply_all_chat_edits(
    message,
    context,
    user,
    user_id_str: str,
    session: dict,
    edits: list[str],
) -> None:
    """Terapkan semua instruksi edit yang dikumpulkan dari sesi Q&A ke dokumen.

    Setiap instruksi diproses secara berurutan melalui DocEditorAgent.  File
    hasil edit terakhir dikirim ke user dan semua sesi yang terkait dibersihkan.
    """
    n_edits = len(edits)
    original_filename = session.get("original_filename", "document.docx")
    original_path = session["docx_path"]   # Jaga referensi file asli agar tidak terhapus
    current_path = original_path

    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify(
            f"⏳ *Menerapkan {n_edits} instruksi edit dari percakapan...*"
        ),
        parse_mode="MarkdownV2",
        quote=True,
    )

    applied = 0
    for i, instruction in enumerate(edits, 1):
        async def _cb(rendered_text: str, edit_number: int = i) -> None:  # noqa: ANN001
            try:
                status = (
                    f"⏳ *Edit {edit_number}/{n_edits}:* Menerapkan...\n{rendered_text}"
                )
                formatted = telegramify_markdown.markdownify(status)
                await context.bot.edit_message_text(
                    chat_id=progress_msg.chat_id,
                    message_id=progress_msg.message_id,
                    text=formatted,
                    parse_mode="MarkdownV2",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Chat-edit progress update skipped: %s", exc)

        try:
            task = await process_docx(
                session_id=user_id_str,
                docx_path=current_path,
                original_filename=original_filename,
                user_caption=instruction,
                status_callback=_cb,
            )
        except Exception as exc:
            logger.exception(
                "apply_all_chat_edits edit %d/%d failed for user=%s: %s",
                i, n_edits, user.id, exc,
            )
            task = None

        if task is not None and task.status.value != "failed":
            new_path = task.metadata.get("document_path")
            if new_path and new_path.lower().endswith(".docx"):
                # Hapus file lama (bukan file asli sesi pertama)
                if current_path != original_path and current_path != new_path:
                    try:
                        os.remove(current_path)
                    except OSError as exc:
                        logger.debug("Could not remove intermediate DOCX %s: %s", current_path, exc)
                current_path = new_path
                applied += 1

    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception:  # noqa: BLE001
        pass

    # Bersihkan semua sesi terkait
    _pending_doc_sessions.pop(user_id_str, None)
    _pending_chat_edits.pop(user_id_str, None)

    if applied == 0:
        await _safe_reply(
            message,
            "❌ Tidak ada instruksi edit yang berhasil diterapkan. Coba kirim ulang instruksinya.",
        )
        return

    await _safe_reply(
        message,
        f"✅ *{applied} dari {n_edits} instruksi edit berhasil diterapkan.*\nFile siap dikirim…",
    )

    # Kirim file langsung (tanpa menyimpan ke _pending_edit_sessions)
    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action="upload_document")
        with open(current_path, "rb") as f:
            await message.reply_document(
                document=f,
                filename=os.path.basename(current_path),
                caption="📝 File Word dengan semua edit dari percakapan siap diunduh.",
                quote=True,
            )
        logger.info(
            "Sent all-chat-edits DOCX to user=%s path=%s applied=%d/%d",
            user.id, current_path, applied, n_edits,
        )
    except Exception as exc:
        logger.exception("Failed to send all-chat-edits DOCX to user=%s: %s", user.id, exc)
        await message.reply_text(
            "⚠️ Gagal mengirim file hasil edit. Coba lagi nanti.", quote=True
        )
    finally:
        try:
            os.remove(current_path)
        except OSError as exc:
            logger.debug("Could not remove all-chat-edits DOCX %s: %s", current_path, exc)


async def echo_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a text message through the agent orchestrator and reply."""
    message = update.message
    user    = update.effective_user

    logger.info("Text from user=%s: %.100s", user.id, message.text)

    user_id_str = str(user.id)
    user_text   = message.text or ""

    # ── 1. Cek sesi edit dokumen yang sedang menunggu konfirmasi ──────────
    pending_edit = _pending_edit_sessions.get(user_id_str)
    if pending_edit:
        if is_edit_intent(user_text):
            # User memberikan instruksi edit tambahan
            await _handle_pending_edit(message, context, user, user_id_str, pending_edit, user_text)
        else:
            # User selesai mengedit – kirim file (termasuk "berikan file yang sudah diedit")
            await _send_edited_docx(message, context, user, user_id_str, pending_edit)
        return

    # ── 2. Cek sesi analisis dokumen – user bisa minta edit atau tanya-jawab
    pending_doc = _pending_doc_sessions.get(user_id_str)
    queued_edit_this_turn = False  # Apakah instruksi edit baru saja dikumpulkan
    if pending_doc:
        if _is_apply_edits_trigger(user_text):
            # User minta semua instruksi edit dari percakapan diterapkan sekaligus
            chat_edits = _pending_chat_edits.get(user_id_str, [])
            if chat_edits:
                await _apply_all_chat_edits(
                    message, context, user, user_id_str, pending_doc, chat_edits
                )
            else:
                await _safe_reply(
                    message,
                    "ℹ️ Belum ada instruksi edit yang dikumpulkan dari percakapan.\n\n"
                    "Ketikkan instruksi seperti _\"ubah bab 1 jadi lebih formal\"_ terlebih dahulu, "
                    "lalu minta file editannya.",
                )
            return
        if is_edit_intent(user_text):
            # Kumpulkan instruksi edit; biarkan DocAuditorAgent menjawab via Q&A
            _pending_chat_edits.setdefault(user_id_str, []).append(user_text)
            queued_edit_this_turn = True
            # Tidak return – lanjutkan ke process_message agar Q&A tetap berjalan
    # Jika pending_doc ada tapi bukan edit intent → lanjut ke process_message
    # (DocAuditorAgent akan menjawab pertanyaan menggunakan DocIndex)

    # ── Send initial progress message ──────────────────────────────────────
    # This message will be edited live at each pipeline stage so the user
    # always sees what the bot is doing.
    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify("⏳ *Sedang memproses permintaan...*\n`[░░░░░░░░░░]` *0%*"),
        parse_mode="MarkdownV2",
        quote=True,
    )

    async def _progress_callback(rendered_text: str) -> None:
        """Edit the progress message with the latest tracker output."""
        try:
            formatted = telegramify_markdown.markdownify(rendered_text)
            await context.bot.edit_message_text(
                chat_id=progress_msg.chat_id,
                message_id=progress_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            # Telegram raises if the text didn't change – silently ignore.
            logger.debug("edit_message_text skipped: %s", exc)

    # Show typing indicator while the pipeline runs
    await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")

    task = await process_message(
        session_id=str(user.id),
        user_text=message.text,
        status_callback=_progress_callback,
    )

    # ── Delete the progress message now that we have the real reply ────────
    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id,
            message_id=progress_msg.message_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not delete progress message: %s", exc)

    reply = task.result or "Maaf, saya tidak dapat memproses permintaan Anda."

    # Send text reply (falls back to plain text if Markdown is malformed)
    await _safe_reply(message, reply)

    # ── Perkaya instruksi edit dengan saran konten dari AI ────────────────
    # Ketika DocAuditorAgent menjawab instruksi edit dengan konten yang disarankan
    # (misal draf revisi Problem Statement), kita simpan jawaban tersebut bersama
    # instruksi asli agar DocEditorAgent bisa menerapkan teks yang persis sama
    # seperti yang ditampilkan di chat bubble – bukan interpretasi ulang.
    if queued_edit_this_turn and task.result:
        edits_list = _pending_chat_edits.get(user_id_str, [])
        if edits_list:
            original_instruction = edits_list[-1]
            # Batasi panjang saran AI agar prompt DocEditorAgent tidak terlalu panjang
            ai_suggestion = task.result[:_MAX_AI_SUGGESTION_FOR_EDITOR]
            edits_list[-1] = (
                f"{original_instruction}\n\n"
                f"[Konten yang disarankan AI untuk perubahan ini]:\n{ai_suggestion}"
            )

    # ── Notifikasi jika instruksi edit baru saja dikumpulkan ─────────────
    if queued_edit_this_turn:
        n_queued = len(_pending_chat_edits.get(user_id_str, []))
        await _safe_reply(
            message,
            f"📌 *Instruksi edit dicatat!* ({n_queued} instruksi tersimpan)\n\n"
            "Teruskan percakapan atau berikan instruksi edit lainnya.\n"
            "Ketika sudah siap, ketik _\"berikan saya file yang sudah diedit\"_ "
            "untuk menerapkan semua instruksi ke dokumen sekaligus.",
        )

    # ── Kirim file Excel jika ada (WBS / mandays) ────────────────────────────
    excel_path = task.metadata.get("excel_path")
    if excel_path:
        try:
            await context.bot.send_chat_action(
                chat_id=message.chat_id, action="upload_document"
            )
            with open(excel_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=excel_path.split("/")[-1],
                    caption="📊 File Excel siap digunakan.",
                    quote=True,
                )
            logger.info("Sent Excel to user=%s path=%s", user.id, excel_path)
        except Exception as exc:
            logger.exception("Failed to send Excel to user=%s: %s", user.id, exc)
            await message.reply_text(
                "⚠️ Gagal mengirim file Excel. Coba lagi nanti.", quote=True
            )

    # ── Kirim file dokumen jika ada (PDF / DOCX dari TechnicalWriterAgent) ───
    document_path = task.metadata.get("document_path")
    if document_path:
        import os  # noqa: PLC0415
        ext = os.path.splitext(document_path)[1].lower()
        caption_map = {
            ".pdf":  "📝 Dokumen teknis PDF siap.",
            ".docx": "📝 Dokumen teknis Word siap.",
        }
        caption = caption_map.get(ext, "📝 Dokumen siap.")
        try:
            await context.bot.send_chat_action(
                chat_id=message.chat_id, action="upload_document"
            )
            with open(document_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=document_path.split("/")[-1],
                    caption=caption,
                    quote=True,
                )
            logger.info("Sent document to user=%s path=%s", user.id, document_path)
        except Exception as exc:
            logger.exception("Failed to send document to user=%s: %s", user.id, exc)
            await message.reply_text(
                "⚠️ Gagal mengirim file dokumen. Coba lagi nanti.", quote=True
            )

    # ── Kirim screenshot dari web automation jika ada ─────────────────────────
    screenshots: list[str] = task.metadata.get("screenshots", [])
    if screenshots:
        try:
            await context.bot.send_chat_action(
                chat_id=message.chat_id, action="upload_photo"
            )
            for idx, screenshot_b64 in enumerate(screenshots, start=1):
                caption = (
                    f"📸 Screenshot {idx}/{len(screenshots)}"
                    if len(screenshots) > 1
                    else "📸 Screenshot"
                )
                try:
                    png_bytes = base64.b64decode(screenshot_b64)
                    screenshot_buffer = io.BytesIO(png_bytes)
                    screenshot_buffer.name = f"screenshot_{idx}.png"
                    await message.reply_photo(
                        photo=screenshot_buffer,
                        caption=caption,
                        quote=True,
                    )
                    logger.info(
                        "Sent screenshot %d/%d to user=%s",
                        idx, len(screenshots), user.id,
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to send screenshot %d to user=%s: %s", idx, user.id, exc
                    )
        except Exception as exc:
            logger.exception("Failed to send screenshots to user=%s: %s", user.id, exc)


async def handle_pdf_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle PDF document uploads – converts PDF to an interactive HTML quiz.

    Flow:
      1. Download the PDF to a temp file
      2. Show a live progress message (edited at each stage)
      3. Run process_pdf_quiz() through the orchestrator
      4. Send the resulting .html file back to the user
    """
    message = update.message
    user    = update.effective_user
    doc     = message.document

    original_filename = doc.file_name or "document.pdf"
    user_caption      = message.caption or ""

    # Reject clearly non-PDF files (defence-in-depth against filter bypass)
    mime_ok = doc.mime_type in (None, "application/pdf", "application/octet-stream")
    ext_ok  = original_filename.lower().endswith(".pdf")
    if not mime_ok and not ext_ok:
        await message.reply_text(
            "⚠️ Hanya file PDF yang didukung. Kirim file dengan format `.pdf`.", quote=True
        )
        return

    logger.info(
        "PDF from user=%s filename=%r size=%d bytes caption=%r",
        user.id, original_filename, doc.file_size or 0, user_caption,
    )

    # ── Size guard (max 20 MB to protect VPS RAM) ──────────────────────────
    if doc.file_size and doc.file_size > _MAX_UPLOAD_BYTES:
        await message.reply_text(
            "⚠️ File PDF terlalu besar (maks. 20 MB). Coba kompres dulu.",
            quote=True,
        )
        return

    # ── Send initial progress message ──────────────────────────────────────
    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify(
            "⏳ *Memproses PDF...*\n`[░░░░░░░░░░]` *0%*\n\nMengunduh file..."
        ),
        parse_mode="MarkdownV2",
        quote=True,
    )

    async def _progress_callback(rendered_text: str) -> None:
        try:
            formatted = telegramify_markdown.markdownify(rendered_text)
            await context.bot.edit_message_text(
                chat_id=progress_msg.chat_id,
                message_id=progress_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("PDF progress edit skipped: %s", exc)

    # ── Download PDF to temp file ──────────────────────────────────────────
    tmp_dir = os.path.join(tempfile.gettempdir(), "advance_ai_pdf_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    # Sanitize filename: keep only alphanumeric, dots, hyphens, underscores
    # and reject any path traversal sequences
    safe_name = _SAFE_FILENAME_RE.sub("_", original_filename)
    safe_name = safe_name.lstrip("._/\\") or "upload.pdf"
    if ".." in safe_name:
        safe_name = "upload.pdf"
    pdf_path  = os.path.join(tmp_dir, f"{user.id}_{safe_name}")
    # Verify the final path stays inside tmp_dir (defence-in-depth)
    pdf_path  = os.path.realpath(pdf_path)
    if not pdf_path.startswith(os.path.realpath(tmp_dir)):
        logger.warning("Path traversal attempt detected for user=%s filename=%r", user.id, original_filename)
        pdf_path = os.path.join(os.path.realpath(tmp_dir), f"{user.id}_upload.pdf")

    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action="upload_document")
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(pdf_path)
        logger.info("PDF downloaded to %s for user=%s", pdf_path, user.id)
    except Exception as exc:
        logger.exception("Failed to download PDF for user=%s: %s", user.id, exc)
        try:
            await context.bot.delete_message(
                chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
            )
        except Exception:  # noqa: BLE001
            pass
        await message.reply_text(
            "❌ Gagal mengunduh PDF. Coba kirim ulang.", quote=True
        )
        return

    # ── Run the quiz pipeline ──────────────────────────────────────────────
    try:
        task = await process_pdf(
            session_id=str(user.id),
            pdf_path=pdf_path,
            original_filename=original_filename,
            user_caption=user_caption,
            status_callback=_progress_callback,
        )
    except Exception as exc:
        logger.exception("process_pdf_quiz raised for user=%s: %s", user.id, exc)
        task = None
    finally:
        # Clean up the downloaded PDF regardless of outcome
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    # ── Delete the progress message ────────────────────────────────────────
    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not delete PDF progress message: %s", exc)

    # ── Handle failure ─────────────────────────────────────────────────────
    if task is None or task.status.value == "failed":
        error_msg = (task.result if task else None) or "❌ Gagal memproses PDF. Coba lagi."
        await _safe_reply(message, error_msg)
        return

    # ── Send text reply ────────────────────────────────────────────────────
    reply = task.result or "✅ Kuis berhasil dibuat!"
    await _safe_reply(message, reply)

    # ── Send the HTML quiz file ────────────────────────────────────────────
    html_path = task.metadata.get("html_path")
    if html_path and os.path.isfile(html_path):
        try:
            await context.bot.send_chat_action(
                chat_id=message.chat_id, action="upload_document"
            )
            quiz_title = task.metadata.get("quiz_title", "kuis")
            html_filename = f"{quiz_title.replace('Kuis: ', '').replace(' ', '_').lower()}_quiz.html"
            with open(html_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=html_filename,
                    caption=(
                        "🎉 *Website Kuis Interaktif siap!*\n\n"
                        "Buka file `.html` ini di browser untuk memulai kuis.\n"
                        "Tidak perlu koneksi internet setelah dibuka! ✅"
                    ),
                    parse_mode="Markdown",
                    quote=True,
                )
            logger.info("Sent HTML quiz to user=%s path=%s", user.id, html_path)
        except Exception as exc:
            logger.exception("Failed to send HTML quiz to user=%s: %s", user.id, exc)
            await message.reply_text(
                "⚠️ Kuis berhasil dibuat tapi gagal dikirim. Coba lagi.", quote=True
            )
        finally:
            # Clean up HTML file after sending
            try:
                os.remove(html_path)
            except OSError:
                pass
    else:
        logger.warning("html_path missing in task.metadata for user=%s", user.id)
        await message.reply_text(
            "⚠️ File HTML tidak ditemukan. Ada kesalahan saat membangun kuis.", quote=True
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Acknowledge photo uploads (image analysis is a future feature)."""
    user = update.effective_user
    logger.info("Photo from user=%s.", user.id)

    photo = update.message.photo[-1]
    await update.message.reply_text(
        f"📷 Foto diterima! (file_id: <code>{photo.file_id}</code>)\n\n"
        "Analisis gambar akan segera hadir. 🚀",
        parse_mode="HTML",
        quote=True,
    )


async def handle_docx_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle .docx document uploads – menganalisis dokumen dan membangun indeks bab.

    Flow:
      1. Download file .docx ke temp file
      2. Tampilkan progress message
      3. Jalankan process_docx() (DocxParserTool → DocAuditorAgent)
      4. Kirim laporan: judul + daftar isi + ringkasan per bab
    """
    message = update.message
    user    = update.effective_user
    doc     = message.document

    original_filename = doc.file_name or "document.docx"
    user_caption      = message.caption or ""

    # Validasi ekstensi
    if not original_filename.lower().endswith(".docx"):
        await message.reply_text(
            "⚠️ Hanya file Word (.docx) yang didukung oleh fitur ini.", quote=True
        )
        return

    logger.info(
        "DOCX from user=%s filename=%r size=%d bytes caption=%r",
        user.id, original_filename, doc.file_size or 0, user_caption,
    )

    # ── Size guard (max 20 MB) ─────────────────────────────────────────────
    if doc.file_size and doc.file_size > _MAX_UPLOAD_BYTES:
        await message.reply_text(
            "⚠️ File terlalu besar (maks. 20 MB). Coba kompres atau perpendek dokumennya.",
            quote=True,
        )
        return

    # ── Progress message ───────────────────────────────────────────────────
    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify(
            "⏳ *Memproses dokumen Word...*\n`[░░░░░░░░░░]` *0%*\n\nMengunduh file..."
        ),
        parse_mode="MarkdownV2",
        quote=True,
    )

    async def _progress_callback(rendered_text: str) -> None:
        try:
            formatted = telegramify_markdown.markdownify(rendered_text)
            await context.bot.edit_message_text(
                chat_id=progress_msg.chat_id,
                message_id=progress_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("DOCX progress edit skipped: %s", exc)

    # ── Download DOCX ──────────────────────────────────────────────────────
    tmp_dir = os.path.join(tempfile.gettempdir(), "advance_ai_docx_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    safe_name = _SAFE_FILENAME_RE.sub("_", original_filename)
    safe_name = safe_name.lstrip("._/\\") or "upload.docx"
    if ".." in safe_name:
        safe_name = "upload.docx"
    docx_path = os.path.join(tmp_dir, f"{user.id}_{safe_name}")
    docx_path = os.path.realpath(docx_path)
    if not docx_path.startswith(os.path.realpath(tmp_dir)):
        logger.warning(
            "Path traversal attempt detected for user=%s filename=%r", user.id, original_filename
        )
        docx_path = os.path.join(os.path.realpath(tmp_dir), f"{user.id}_upload.docx")

    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action="upload_document")
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(docx_path)
        logger.info("DOCX downloaded to %s for user=%s", docx_path, user.id)
    except Exception as exc:
        logger.exception("Failed to download DOCX for user=%s: %s", user.id, exc)
        try:
            await context.bot.delete_message(
                chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
            )
        except Exception:  # noqa: BLE001
            pass
        await message.reply_text("❌ Gagal mengunduh dokumen. Coba kirim ulang.", quote=True)
        return

    # ── Bersihkan sesi lama jika ada ──────────────────────────────────────
    user_id_str = str(user.id)
    old_doc_session = _pending_doc_sessions.pop(user_id_str, None)
    if old_doc_session:
        try:
            os.remove(old_doc_session["docx_path"])
        except OSError as exc:
            logger.debug("Could not remove old doc session file: %s", exc)
    old_edit_session = _pending_edit_sessions.pop(user_id_str, None)
    if old_edit_session:
        try:
            os.remove(old_edit_session["docx_path"])
        except OSError as exc:
            logger.debug("Could not remove old edit session file: %s", exc)
    # Hapus antrian instruksi edit dari percakapan (sesi baru dimulai)
    _pending_chat_edits.pop(user_id_str, None)

    # ── Deteksi mode lebih awal agar bisa membuat salinan sebelum delete ──
    is_edit_mode = is_edit_intent(user_caption)

    # ── Buat salinan file ke session dir jika mode analisis ───────────────
    # Salinan ini disimpan agar tersedia untuk edit lanjutan setelah
    # file temp asli dihapus oleh blok finally di bawah.
    session_docx_copy: str | None = None
    if not is_edit_mode:
        try:
            os.makedirs(_SESSIONS_TMP_DIR, exist_ok=True)
            session_copy_name = f"{user.id}_{safe_name}"
            session_docx_copy = os.path.join(_SESSIONS_TMP_DIR, session_copy_name)
            session_docx_copy = os.path.realpath(session_docx_copy)
            if not session_docx_copy.startswith(os.path.realpath(_SESSIONS_TMP_DIR)):
                session_docx_copy = os.path.join(
                    os.path.realpath(_SESSIONS_TMP_DIR), f"{user.id}_upload.docx"
                )
            shutil.copy2(docx_path, session_docx_copy)
        except Exception as exc:
            logger.warning("Could not copy DOCX to session dir for user=%s: %s", user.id, exc)
            session_docx_copy = None

    # ── Jalankan pipeline ──────────────────────────────────────────────────
    try:
        task = await process_docx(
            session_id=str(user.id),
            docx_path=docx_path,
            original_filename=original_filename,
            user_caption=user_caption,
            status_callback=_progress_callback,
        )
    except Exception as exc:
        logger.exception("process_docx raised for user=%s: %s", user.id, exc)
        task = None
    finally:
        # Hapus file temp setelah diproses
        try:
            os.remove(docx_path)
        except OSError:
            pass

    # ── Hapus progress message ─────────────────────────────────────────────
    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not delete DOCX progress message: %s", exc)

    # ── Handle failure ─────────────────────────────────────────────────────
    if task is None or task.status.value == "failed":
        # Bersihkan salinan jika pipeline gagal
        if session_docx_copy:
            try:
                os.remove(session_docx_copy)
            except OSError:
                pass
        error_msg = (task.result if task else None) or "❌ Gagal memproses dokumen. Coba lagi."
        await _safe_reply(message, error_msg)
        return

    # ── Kirim laporan ──────────────────────────────────────────────────────
    reply = task.result or "✅ Analisis dokumen selesai."
    await _safe_reply(message, reply)

    # ── Tanya apakah ada edit tambahan sebelum mengirim file ──────────────
    edited_docx_path = task.metadata.get("document_path")
    if edited_docx_path and edited_docx_path.lower().endswith(".docx"):
        # Mode edit: simpan sesi dan tanya dulu; file dikirim setelah user konfirmasi
        if session_docx_copy:
            try:
                os.remove(session_docx_copy)
            except OSError:
                pass
        _pending_edit_sessions[user_id_str] = {
            "docx_path": edited_docx_path,
            "doc_title": task.metadata.get("doc_title", original_filename),
            "original_filename": original_filename,
        }
        suggestions = _get_edit_suggestions(user_id_str)
        await _safe_reply(message, _build_follow_up_question(suggestions))
        logger.info(
            "Pending edit session created for user=%s path=%s", user.id, edited_docx_path
        )
    elif session_docx_copy and os.path.isfile(session_docx_copy):
        # Mode analisis berhasil: simpan salinan untuk edit lanjutan
        _pending_doc_sessions[user_id_str] = {
            "docx_path": session_docx_copy,
            "doc_title": task.metadata.get("doc_title", original_filename),
            "original_filename": original_filename,
        }
        await _safe_reply(
            message,
            "💡 *Tip:* Ingin mengedit bagian tertentu dari dokumen ini? "
            "Langsung ketikkan instruksi editnya, misalnya: "
            "_\"ubah bagian pendahuluan menjadi lebih formal\"_",
        )
        logger.info(
            "Doc session created for user=%s path=%s", user.id, session_docx_copy
        )
    elif session_docx_copy:
        # Salinan tidak jadi dibuat/tidak ada – bersihkan
        try:
            os.remove(session_docx_copy)
        except OSError:
            pass


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for unsupported message types."""
    logger.debug("Unknown message type from user=%s.", update.effective_user.id)
    await update.message.reply_text(
        "⚠️ Tipe pesan ini belum didukung. Coba kirim teks.",
        quote=True,
    )
