"""Message handlers: text, photo, document (PDF), and unknown messages."""

from __future__ import annotations

import logging
import os
import tempfile

import telegramify_markdown
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from src.orchestrator.main_loop import process_message, process_pdf_quiz

logger = logging.getLogger(__name__)


_MAX_MSG_LEN = 4096


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


async def echo_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a text message through the agent orchestrator and reply."""
    message = update.message
    user    = update.effective_user

    logger.info("Text from user=%s: %.100s", user.id, message.text)

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
    logger.info(
        "PDF from user=%s filename=%r size=%d bytes",
        user.id, original_filename, doc.file_size or 0,
    )

    # ── Size guard (max 20 MB to protect VPS RAM) ──────────────────────────
    max_bytes = 20 * 1024 * 1024
    if doc.file_size and doc.file_size > max_bytes:
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
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_filename)
    # Strip leading dots/slashes to prevent path traversal
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
        task = await process_pdf_quiz(
            session_id=str(user.id),
            pdf_path=pdf_path,
            original_filename=original_filename,
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


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for unsupported message types."""
    logger.debug("Unknown message type from user=%s.", update.effective_user.id)
    await update.message.reply_text(
        "⚠️ Tipe pesan ini belum didukung. Coba kirim teks.",
        quote=True,
    )
