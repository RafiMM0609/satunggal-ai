"""Telegram Application – assembles and registers all handlers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    filters,
    MessageHandler,
)

from .config import Config
from config.settings import get_settings
from src.handlers import (
    briefing_command,
    deploy,
    echo_text,
    handle_docx_document,
    handle_pdf_document,
    handle_photo,
    help_command,
    mode_callback,
    mode_command,
    ping,
    reset,
    setapikey,
    setgithubtoken,
    setgitlabtoken,
    setllmmodel,
    setmaxtokens,
    setollamamodel,
    setollamahost,
    setollamakey,
    setprovider,
    start,
    status_command,
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
  • Kirim PDF + caption <i>"buat kuis telegram"</i> → kuis polling interaktif langsung di chat
  • Progress real-time, feedback instan, dark mode &amp; scoreboard

📑 <b>Doc Auditor (Quality Auditor)</b>
  • Kirim file <b>.docx</b> untuk analisis mendalam
  • Ekstrak Daftar Isi otomatis & ringkasan per bab
  • Tanya-jawab interaktif tentang isi dokumen tanpa kirim ulang

🌐 <b>Web Automation</b>
  • Buka URL & ringkas konten halaman web
  • Navigasi, klik tombol & isi form secara otomatis
  • Ekstrak data terstruktur dari halaman web
  • <b>follow_parent</b> — browser tetap terbuka setelah setiap tugas sehingga pertanyaan lanjutan langsung berinteraksi di halaman yang sama; gunakan /reset untuk menutup browser

🖥️ <b>Monitoring &amp; Debugging</b>
  • <b>SysInfo Agent</b> — monitor CPU, RAM, disk, proses server
  • <b>Log Viewer</b> — tampilkan log bot real-time untuk debugging

⚙️ <b>Admin Commands</b>
  • /deploy — pull kode terbaru & restart otomatis
  • /setgithubtoken — atur GitHub Personal Access Token
  • /setgitlabtoken — atur GitLab Personal Access Token

<i>Ketik pesan apa saja untuk mulai!</i>"""


async def _send_startup_notification(app: Application) -> None:
    """Kirim notifikasi startup ke admin saat bot pertama kali berjalan."""
    # ── Start reminder scheduler and reschedule pending reminders ─────────
    from src.agents.reminder_agent.scheduler import (
        set_bot,
        start_scheduler,
        reschedule_pending_on_startup,
    )
    set_bot(app.bot)
    start_scheduler()
    await reschedule_pending_on_startup()

    # ── Fase 4: Proactive jobs ──────────────────────────────────────────────
    # Daily Briefing: ResearcherAgent mengirim ringkasan berita terjadwal.
    from src.proactive.daily_briefing import start_briefing_job
    start_briefing_job(app.bot)

    # Repo Watcher: Pantau commit baru di repo dan kirim laporan auto.
    from src.proactive.repo_watcher import start_repo_watcher_job
    start_repo_watcher_job(app.bot)

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


async def _shutdown_scheduler(app: Application) -> None:
    """Stop APScheduler when the bot shuts down."""
    from src.agents.reminder_agent.scheduler import stop_scheduler
    from src.proactive.daily_briefing import stop_briefing_job
    from src.proactive.repo_watcher import stop_repo_watcher_jobs
    stop_briefing_job()
    stop_repo_watcher_jobs()
    stop_scheduler()


def build_application(config: Config) -> Application:
    """Build and return a fully wired Application instance."""
    app = (
        Application.builder()
        .token(config.bot_token)
        .concurrent_updates(True)
        .post_init(_send_startup_notification)
        .post_shutdown(_shutdown_scheduler)
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
    app.add_handler(CommandHandler("mode",      mode_command))
    app.add_handler(CommandHandler("status",    status_command))
    app.add_handler(CommandHandler("setapikey",    setapikey))
    app.add_handler(CommandHandler("setmaxtokens", setmaxtokens))
    app.add_handler(CommandHandler("setllmmodel",  setllmmodel))
    app.add_handler(CommandHandler("setprovider",  setprovider))
    app.add_handler(CommandHandler("setollamakey",   setollamakey))
    app.add_handler(CommandHandler("setollamahost",  setollamahost))
    app.add_handler(CommandHandler("setollamamodel", setollamamodel))
    app.add_handler(CommandHandler("setgithubtoken", setgithubtoken))
    app.add_handler(CommandHandler("setgitlabtoken", setgitlabtoken))
    app.add_handler(CommandHandler("briefing",       briefing_command))

    # ── Callback query handlers (InlineKeyboard) ───────────────────────────
    app.add_handler(CallbackQueryHandler(mode_callback, pattern=r"^set_mode:"))

    # ── Message handlers ───────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(
        filters.Document.PDF | filters.Document.FileExtension("pdf"),
        handle_pdf_document,
    ))
    app.add_handler(MessageHandler(
        filters.Document.FileExtension("docx"),
        handle_docx_document,
    ))

    # ── Fallback ───────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.ALL, unknown_message))
