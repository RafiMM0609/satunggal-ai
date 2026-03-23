"""Command handlers: /start, /help, /ping, /reset, /deploy, /setapikey."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from config.settings import get_settings
from src.memory.key_store import (
    clear_openrouter_key,
    get_openrouter_key,
    set_openrouter_key,
)
from src.orchestrator.main_loop import clear_session

_DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "helper_deploy.sh"

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
        "/deploy     — Pull kode & restart service (admin)\n"
        "/setapikey  — Atur API key OpenRouter (admin)"
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
            # jalankan di session baru agar sinyal tidak menyebar ke subprocess
            start_new_session=True,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
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


async def setapikey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setapikey — simpan atau hapus override API key OpenRouter (admin only).

    Usage:
        /setapikey <API_KEY>   — simpan key baru (menggantikan yang di .env)
        /setapikey clear       — hapus override, kembali ke key di .env
        /setapikey status      — lihat apakah override aktif
    """
    settings = get_settings()
    user = update.effective_user

    # ── Cek izin admin ────────────────────────────────────────────────────────
    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setapikey tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur API key.")
        return

    args = context.args  # list of words after the command

    # ── Tidak ada argumen → tampilkan panduan ─────────────────────────────────
    if not args:
        current = get_openrouter_key()
        status = "✅ Override aktif" if current else "ℹ️ Tidak ada override (menggunakan .env)"
        await update.message.reply_html(
            f"{status}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setapikey &lt;API_KEY&gt;</code>  — simpan key baru\n"
            "<code>/setapikey clear</code>       — hapus override\n"
            "<code>/setapikey status</code>      — cek status override",
        )
        return

    sub = args[0].strip()

    # ── status ─────────────────────────────────────────────────────────────────
    if sub.lower() == "status":
        current = get_openrouter_key()
        if current:
            masked = current[:8] + "..." + current[-4:] if len(current) > 12 else "***"
            await update.message.reply_text(f"✅ Override aktif: {masked}")
        else:
            await update.message.reply_text("ℹ️ Tidak ada override — menggunakan key dari .env")
        return

    # ── clear ──────────────────────────────────────────────────────────────────
    if sub.lower() == "clear":
        clear_openrouter_key()
        logger.info("User %s menghapus OpenRouter API key override.", user.id)
        await update.message.reply_text("🗑️ Override API key dihapus. Kembali menggunakan key dari .env.")
        return

    # ── simpan key baru ────────────────────────────────────────────────────────
    new_key = sub
    if not new_key.startswith("sk-"):
        await update.message.reply_text(
            "⚠️ API key tidak valid — key OpenRouter biasanya diawali dengan `sk-`.\n"
            "Pastikan key yang dimasukkan benar.",
            parse_mode="Markdown",
        )
        return

    set_openrouter_key(new_key)
    logger.info("User %s memperbarui OpenRouter API key override.", user.id)

    masked = new_key[:8] + "..." + new_key[-4:] if len(new_key) > 12 else "***"
    # Hapus pesan asli agar key tidak tersimpan di chat history
    try:
        await update.message.delete()
    except Exception:  # noqa: BLE001
        pass  # tidak semua chat izinkan penghapusan pesan

    await update.effective_chat.send_message(
        f"✅ API key OpenRouter berhasil disimpan ({masked}).\n"
        "Override akan digunakan untuk semua permintaan berikutnya tanpa perlu restart bot."
    )

