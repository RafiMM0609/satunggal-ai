"""Command handlers: /start, /help, /ping, /reset, /deploy."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from config.settings import get_settings
from src.orchestrator.main_loop import clear_session

_DEPLOY_SCRIPT = Path(__file__).resolve().parents[3] / "helper_deploy.sh"

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
        "/deploy  — Pull kode & restart service (admin)"
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


async def deploy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /deploy — pull kode terbaru & restart service (khusus admin)."""
    settings = get_settings()
    user = update.effective_user

    # ── Cek izin admin ────────────────────────────────────────────────────────
    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /deploy tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa menjalankan deploy.")
        return

    if not _DEPLOY_SCRIPT.is_file():
        await update.message.reply_text("❌ Script deploy tidak ditemukan di server.")
        return

    logger.info("User %s memulai deploy.", user.id)
    status_msg = await update.message.reply_text("🚀 Deploy dimulai... mohon tunggu.")

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(_DEPLOY_SCRIPT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_DEPLOY_SCRIPT.parent),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode(errors="replace").strip()

        if proc.returncode == 0:
            logger.info("Deploy berhasil oleh user %s.", user.id)
            # Kirim ringkasan (potong jika terlalu panjang)
            snippet = output[-3000:] if len(output) > 3000 else output
            await status_msg.edit_text(
                f"✅ <b>Deploy berhasil!</b>\n\n<pre>{snippet}</pre>",
                parse_mode="HTML",
            )
        else:
            logger.error("Deploy gagal (exit %s) oleh user %s.", proc.returncode, user.id)
            snippet = output[-3000:] if len(output) > 3000 else output
            await status_msg.edit_text(
                f"❌ <b>Deploy gagal</b> (exit code {proc.returncode})\n\n<pre>{snippet}</pre>",
                parse_mode="HTML",
            )
    except asyncio.TimeoutError:
        logger.error("Deploy timeout oleh user %s.", user.id)
        await status_msg.edit_text("⏱️ Deploy timeout (>120 detik). Cek log server secara manual.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Deploy error: %s", exc)
        await status_msg.edit_text(f"❌ Error tak terduga: {exc}")
