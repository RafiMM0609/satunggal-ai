"""Telegram Application – assembles and registers all handlers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    filters,
    MessageHandler,
)

from .config import Config
from config.settings import get_settings
from src.handlers import (
    deploy,
    echo_text,
    handle_pdf_document,
    handle_photo,
    help_command,
    ping,
    reset,
    setapikey,
    setllmmodel,
    setmaxtokens,
    start,
    unknown_message,
)

logger = logging.getLogger(__name__)

_STARTUP_MESSAGE = """🟢 <b>AdvanceAI — Bot Online</b>

⏰ <i>{timestamp}</i>

<b>Fitur yang tersedia:</b>

🧠 <b>Multi-Agent Intelligence</b>
  • <b>Researcher</b> — riset topik dengan pencarian web (Tavily)
  • <b>Responder</b> — jawaban umum & percakapan kontekstual
  • <b>Gatekeeper</b> — routing cerdas ke agent yang tepat

💻 <b>Developer Suite</b>
  • <b>Developer Agent</b> — buat & edit kode, push ke GitHub/GitLab
  • <b>Developer Inspector</b> — inspeksi mendalam & root cause analysis bug
  • <b>Developer Q&amp;A</b> — tanya jawab isi repo: API, tech stack, alur, model data
  • <b>Sandbox Runner</b> — eksekusi aman via Docker

📋 <b>Project Management</b>
  • <b>WBS Agent</b> — generate Work Breakdown Structure (Excel)
  • <b>Mandays Agent</b> — estimasi man-days dari WBS
  • <b>Content Creator</b> — buat konten & dokumentasi

📄 <b>Technical Writer</b>
  • Export dokumen ke Word (.docx) & PDF
  • Render diagram Mermaid ke PNG

🧩 <b>PDF-to-Quiz Generator</b>
  • Kirim PDF untuk diubah menjadi Website Kuis Interaktif HTML
  • Progress real-time, feedback instan, dark mode & scoreboard

🌐 <b>Web Automation</b>
  • Buka URL & ringkas konten halaman web
  • Navigasi, klik tombol & isi form secara otomatis
  • Ekstrak data terstruktur dari halaman web

🖥️ <b>Monitoring &amp; Debugging</b>
  • <b>SysInfo Agent</b> — monitor CPU, RAM, disk, proses server
  • <b>Log Viewer</b> — tampilkan log bot real-time untuk debugging

⚙️ <b>Admin Commands</b>
  • /deploy — pull kode terbaru & restart otomatis

<i>Ketik pesan apa saja untuk mulai!</i>"""


async def _send_startup_notification(app: Application) -> None:
    """Kirim notifikasi startup ke admin saat bot pertama kali berjalan."""
    settings = get_settings()
    if not settings.admin_user_id:
        logger.info("ADMIN_USER_ID tidak dikonfigurasi, startup notification dilewati.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = _STARTUP_MESSAGE.format(timestamp=ts)

    try:
        bot: Bot = app.bot
        await bot.send_message(
            chat_id=settings.admin_user_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
        logger.info("Startup notification terkirim ke admin (ID: %s).", settings.admin_user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal kirim startup notification: %s", exc)


def build_application(config: Config) -> Application:
    """Build and return a fully wired Application instance."""
    app = (
        Application.builder()
        .token(config.bot_token)
        .post_init(_send_startup_notification)
        .build()
    )

    _register_handlers(app)

    logger.info("Application built with %d handler(s).", len(app.handlers[0]))
    return app


def _register_handlers(app: Application) -> None:
    """Register all command and message handlers."""

    # ── Command handlers ───────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",      help_command))
    app.add_handler(CommandHandler("ping",      ping))
    app.add_handler(CommandHandler("reset",     reset))
    app.add_handler(CommandHandler("deploy",    deploy))
    app.add_handler(CommandHandler("setapikey",   setapikey))
    app.add_handler(CommandHandler("setmaxtokens", setmaxtokens))
    app.add_handler(CommandHandler("setllmmodel",  setllmmodel))

    # ── Message handlers ───────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(
        filters.Document.PDF | filters.Document.FileExtension("pdf"),
        handle_pdf_document,
    ))

    # ── Fallback ───────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.ALL, unknown_message))
