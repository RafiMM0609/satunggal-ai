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

from src.orchestrator.main_loop import process_message, process_pdf, process_docx, process_doc_session_message, is_edit_intent

logger = logging.getLogger(__name__)

_GROUP_CHAT_TYPES = ("group", "supergroup")


def _chat_session_id(user_id: int, chat_id: int) -> str:
    """Return a session identifier scoped to both user and chat.

    Using user_id + chat_id means conversation history is isolated between
    a direct-message session and any group session, so concurrent requests
    from different chats never share the same history context.
    """
    return f"{user_id}_{chat_id}"

_MAX_MSG_LEN   = 4096   # Telegram hard limit per message
_SPLIT_RAW_LEN = 3000   # Conservative pre-split limit: markdownify escaping can expand text ~30–50%
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB upload limit for all file types
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

# ── Temp directory for doc session copies ─────────────────────────────────────
_SESSIONS_TMP_DIR = os.path.join(tempfile.gettempdir(), "advance_ai_doc_sessions")

# Edit suggestion constants
_MAX_SECTIONS_FOR_SUGGESTIONS = 6
_MAX_SUGGESTION_TEXT_LEN      = 120
_MAX_SUGGESTIONS_DISPLAY      = 3

# ── Pending doc session store ──────────────────────────────────────────────────
# Maps user_id (str) → {"docx_path": str, "doc_title": str, "original_filename": str}
# Populated after DocAgent successfully analyzes a document.
# Used to keep the .docx file alive and to detect when sessions are active.
_pending_doc_sessions: dict[str, dict] = {}

# ── Pending doc-edit session (legacy direct-edit via upload caption) ───────────
# Maps user_id (str) → {"docx_path": str, "doc_title": str, "original_filename": str}
_pending_edit_sessions: dict[str, dict] = {}


def clear_doc_session(session_id: str) -> None:
    """Remove all in-memory doc session state for *session_id*.

    Called by the /reset command so that subsequent messages are no longer
    routed directly to DocAgent after a session reset.
    """
    _pending_doc_sessions.pop(session_id, None)
    _pending_edit_sessions.pop(session_id, None)


def _is_addressed_in_group(message, context) -> bool:
    """Return True if the bot should respond to *message* in a group/supergroup.

    The bot only responds when:
    - The message is a direct reply to one of the bot's own messages, or
    - The bot is explicitly @mentioned in the message text or caption.

    In private chats this function is never called.
    """
    # Reply to the bot's own message
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == context.bot.id:
        return True

    bot_username = context.bot.username
    if not bot_username:
        return False

    mention = f"@{bot_username}".lower()

    # @mention in regular text entities
    for entity in message.entities or []:
        if entity.type == "mention":
            text = message.text or ""
            if text[entity.offset : entity.offset + entity.length].lower() == mention:
                return True

    # @mention in caption entities (photos / documents)
    for entity in message.caption_entities or []:
        if entity.type == "mention":
            text = message.caption or ""
            if text[entity.offset : entity.offset + entity.length].lower() == mention:
                return True

    return False


def _strip_bot_mention(text: str | None, bot_username: str | None) -> str:
    """Remove *@bot_username* from *text* and strip surrounding whitespace."""
    if not bot_username or not text:
        return text or ""
    pattern = re.compile(r"@" + re.escape(bot_username), re.IGNORECASE)
    return pattern.sub("", text).strip()


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

    Splits the raw text at a conservative character limit (_SPLIT_RAW_LEN)
    *before* calling markdownify, leaving sufficient headroom for the
    MarkdownV2 escaping that the library adds (special chars like `.`, `!`,
    `(`, `)`, etc. each become two characters).  Without this, chunks near
    the 4 096-char boundary can silently exceed Telegram's limit after
    formatting, causing a BadRequest and a plain-text fallback that strips
    all structure from the response.

    Falls back to plain text if formatting fails for any reason, so
    content is never silently dropped.
    """
    chunks = _split_text(text, max_len=_SPLIT_RAW_LEN)
    for chunk in chunks:
        formatted = None
        try:
            formatted = telegramify_markdown.markdownify(chunk)
        except Exception as exc:  # noqa: BLE001
            logger.warning("markdownify failed (%s); sending chunk as plain text.", exc)

        if formatted is not None:
            try:
                await message.reply_text(formatted, parse_mode="MarkdownV2", quote=True)
                continue
            except BadRequest as exc:
                logger.warning("MarkdownV2 parse failed (%s), retrying as plain text.", exc)

        # Plain-text fallback – raw chunk is always ≤ _SPLIT_RAW_LEN chars
        try:
            await message.reply_text(chunk, parse_mode=None, quote=True)
        except BadRequest as exc2:
            logger.error("Failed to send chunk even as plain text: %s", exc2)




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
    """Terapkan instruksi edit tambahan pada file hasil edit sebelumnya."""
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
        await _safe_reply(
            message,
            "✏️ *Ada bagian lain yang perlu diedit?*\n\n"
            "Ketik instruksi edit selanjutnya, atau ketik *selesai* untuk menerima file.",
        )
    else:
        # Tidak ada file baru – bersihkan sesi
        _pending_edit_sessions.pop(user_id_str, None)
        try:
            os.remove(old_docx_path)
        except OSError as exc:
            logger.debug("Could not remove DOCX %s: %s", old_docx_path, exc)


async def echo_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a text message through the agent orchestrator and reply."""
    message = update.message
    user    = update.effective_user
    chat    = update.effective_chat

    # ── Group chat: only respond when addressed to the bot ────────────────
    if chat.type in _GROUP_CHAT_TYPES:
        if not _is_addressed_in_group(message, context):
            return

    logger.info("Text from user=%s chat_type=%s: %.100s", user.id, chat.type, message.text)

    user_id_str = str(user.id)
    session_id  = _chat_session_id(user.id, chat.id)
    # Strip @botname mention so the LLM sees clean user intent
    user_text   = _strip_bot_mention(message.text or "", context.bot.username)

    # ── 1. Cek sesi edit dokumen yang sedang menunggu konfirmasi ──────────
    pending_edit = _pending_edit_sessions.get(user_id_str)
    if pending_edit:
        if is_edit_intent(user_text):
            await _handle_pending_edit(message, context, user, user_id_str, pending_edit, user_text)
        else:
            await _send_edited_docx(message, context, user, user_id_str, pending_edit)
        return

    # ── 2. Cek sesi analisis dokumen aktif ─────────────────────────────────
    # Jika sesi dokumen aktif, bypass gatekeeper dan langsung ke DocAgent.
    # Ini mencegah gatekeeper salah mengklasifikasikan instruksi edit sebagai
    # DOCUMENT_CREATION → technical_writer → document_generator (WeasyPrint).
    has_doc_session = bool(_pending_doc_sessions.get(user_id_str))

    # ── Send initial progress message ──────────────────────────────────────
    progress_msg = await message.reply_text(
        telegramify_markdown.markdownify("⏳ *Sedang memproses permintaan...*\n`[░░░░░░░░░░]` *0%*"),
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
            logger.debug("edit_message_text skipped: %s", exc)

    await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")

    if has_doc_session:
        # Route langsung ke DocAgent tanpa melalui gatekeeper
        task = await process_doc_session_message(
            session_id=session_id,
            user_text=message.text,
            status_callback=_progress_callback,
        )
    else:
        task = await process_message(
            session_id=session_id,
            user_text=message.text,
            status_callback=_progress_callback,
        )

    try:
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id,
            message_id=progress_msg.message_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not delete progress message: %s", exc)

    reply = task.result or "Maaf, saya tidak dapat memproses permintaan Anda."
    await _safe_reply(message, reply)

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

    # ── Kirim file dokumen jika ada (PDF / DOCX / hasil edit DocAgent) ────────
    document_path = task.metadata.get("document_path")
    if document_path:
        import os  # noqa: PLC0415
        ext = os.path.splitext(document_path)[1].lower()
        caption_map = {
            ".pdf":  "📝 Dokumen PDF siap.",
            ".docx": "📝 Dokumen Word siap.",
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
    chat    = update.effective_chat
    doc     = message.document

    original_filename = doc.file_name or "document.pdf"
    user_caption      = _strip_bot_mention(message.caption or "", context.bot.username)

    # In groups: only respond when addressed to the bot
    if chat.type in _GROUP_CHAT_TYPES:
        if not _is_addressed_in_group(message, context):
            return

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
            session_id=_chat_session_id(user.id, chat.id),
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
                    caption=telegramify_markdown.markdownify(
                        "🎉 *Website Kuis Interaktif siap!*\n\n"
                        "Buka file `.html` ini di browser untuk memulai kuis.\n"
                        "Tidak perlu koneksi internet setelah dibuka! ✅"
                    ),
                    parse_mode="MarkdownV2",
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
    message = update.message
    user    = update.effective_user
    chat    = update.effective_chat

    # In groups: only respond when addressed to the bot
    if chat.type in _GROUP_CHAT_TYPES:
        if not _is_addressed_in_group(message, context):
            return

    logger.info("Photo from user=%s.", user.id)

    await message.reply_text(
        "📷 Foto diterima!\n\nAnalisis gambar akan segera hadir. 🚀",
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
    chat    = update.effective_chat
    doc     = message.document

    original_filename = doc.file_name or "document.docx"
    user_caption      = _strip_bot_mention(message.caption or "", context.bot.username)

    # In groups: only respond when addressed to the bot
    if chat.type in _GROUP_CHAT_TYPES:
        if not _is_addressed_in_group(message, context):
            return

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
    # Bersihkan DocIndex sesi lama agar DocAgent mulai fresh
    try:
        from src.memory.doc_index import get_doc_index
        get_doc_index().clear_session(user_id_str)
    except Exception as exc:
        logger.debug("Could not clear old DocIndex for user=%s: %s", user.id, exc)

    # ── Deteksi mode lebih awal agar bisa membuat salinan sebelum delete ──
    is_edit_mode = is_edit_intent(user_caption)

    # ── Buat salinan file ke session dir ──────────────────────────────────
    # Salinan ini disimpan agar tersedia untuk edit lanjutan setelah
    # file temp asli dihapus.  DocAgent menyimpan path-nya di DocIndex.
    session_docx_copy: str | None = None
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
    # Untuk mode analisis, gunakan session_docx_copy (persisten) agar DocAgent
    # bisa menyimpan path yang valid di DocIndex untuk edit lanjutan.
    pipeline_path = session_docx_copy or docx_path
    try:
        task = await process_docx(
            session_id=_chat_session_id(user.id, chat.id),
            docx_path=pipeline_path,
            original_filename=original_filename,
            user_caption=user_caption,
            status_callback=_progress_callback,
        )
    except Exception as exc:
        logger.exception("process_docx raised for user=%s: %s", user.id, exc)
        task = None
    finally:
        # Hapus file temp setelah diproses (session copy sudah terpisah)
        try:
            if docx_path != pipeline_path:
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

    # ── Setelah analisis: catat sesi agar DocAgent tersedia untuk Q&A/edit ─
    edited_docx_path = task.metadata.get("document_path")
    if edited_docx_path and edited_docx_path.lower().endswith(".docx"):
        # Mode edit langsung (caption berisi instruksi): simpan ke pending_edit_sessions
        # agar user bisa melakukan iterasi lebih lanjut via _handle_pending_edit
        if session_docx_copy and session_docx_copy != edited_docx_path:
            try:
                os.remove(session_docx_copy)
            except OSError:
                pass
        _pending_edit_sessions[user_id_str] = {
            "docx_path": edited_docx_path,
            "doc_title": task.metadata.get("doc_title", original_filename),
            "original_filename": original_filename,
        }
        await _safe_reply(
            message,
            "✏️ *Ada bagian lain yang perlu diedit?*\n\n"
            "Ketik instruksi edit selanjutnya, atau ketik *selesai* untuk menerima file.",
        )
        logger.info("Pending edit session created for user=%s path=%s", user.id, edited_docx_path)
    elif session_docx_copy and os.path.isfile(session_docx_copy):
        # Mode analisis: DocAgent sudah menyimpan session_docx_copy di DocIndex.
        # Simpan di _pending_doc_sessions hanya sebagai penanda sesi aktif.
        _pending_doc_sessions[user_id_str] = {
            "docx_path": session_docx_copy,
            "doc_title": task.metadata.get("doc_title", original_filename),
            "original_filename": original_filename,
        }
        logger.info("Doc session created for user=%s path=%s", user.id, session_docx_copy)
    elif session_docx_copy:
        # Salinan tidak jadi dibuat/tidak ada – bersihkan
        try:
            os.remove(session_docx_copy)
        except OSError:
            pass


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for unsupported message types."""
    message = update.message
    chat    = update.effective_chat

    # In groups: silently ignore unsupported messages unless addressed to bot
    if chat.type in _GROUP_CHAT_TYPES:
        if not _is_addressed_in_group(message, context):
            return

    logger.debug("Unknown message type from user=%s.", update.effective_user.id)
    await message.reply_text(
        "⚠️ Tipe pesan ini belum didukung. Coba kirim teks.",
        quote=True,
    )
