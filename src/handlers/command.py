"""Command handlers: /start, /help, /ping, /reset."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.orchestrator.main_loop import clear_session

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /start — sambut pengguna baru."""
    user = update.effective_user
    logger.info("User %s memulai bot.", user.id)

    await update.message.reply_html(
        f"Halo, <b>{user.mention_html()}</b>! 👋\n\n"
        "Saya adalah <b>AdvanceAI</b> – asisten yang cerdas dengan sistem multi-agent.\n\n"
        "Ketik pesan apa saja untuk memulai, atau /help untuk melihat perintah.",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /help — tampilkan daftar perintah."""
    logger.info("User %s meminta bantuan.", update.effective_user.id)

    help_text = (
        "<b>Daftar Perintah</b>\n\n"
        "/start   — Mulai bot\n"
        "/help    — Tampilkan pesan ini\n"
        "/ping    — Cek status bot\n"
        "/reset   — Hapus riwayat percakapan\n\n"
        "<b>Kemampuan:</b>\n"
        "• Jawab pertanyaan umum\n"
        "• Dukungan teknis & riset\n"
        "• Buat WBS & estimasi man-days proyek\n"
    )
    await update.message.reply_html(help_text)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /ping — cek latensi bot."""
    logger.info("User %s mengirim ping.", update.effective_user.id)
    await update.message.reply_text("Pong! 🏓 Bot aktif dan merespons.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /reset — hapus riwayat percakapan pengguna."""
    user = update.effective_user
    logger.info("User %s mereset sesi.", user.id)
    await clear_session(str(user.id))
    await update.message.reply_text(
        "🔄 Riwayat percakapan telah dihapus. Kita mulai dari awal!",
        quote=True,
    )
