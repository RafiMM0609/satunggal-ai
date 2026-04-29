"""Command handlers: /start, /help, /ping, /reset, /deploy, /setapikey, /setmaxtokens."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from config.settings import get_settings
from src.memory.key_store import (
    clear_active_provider,
    clear_github_pat,
    clear_gitlab_pat,
    clear_ollama_key,
    clear_ollama_host,
    clear_ollama_model,
    clear_openrouter_key,
    clear_openrouter_max_tokens,
    clear_openrouter_model,
    get_active_provider,
    get_github_pat,
    get_gitlab_pat,
    get_ollama_key,
    get_ollama_host,
    get_ollama_model,
    get_openrouter_key,
    get_openrouter_max_tokens,
    get_openrouter_model,
    PROVIDER_OLLAMA,
    PROVIDER_OPENROUTER,
    set_active_provider,
    set_github_pat,
    set_gitlab_pat,
    set_ollama_key,
    set_ollama_host,
    set_ollama_model,
    set_openrouter_key,
    set_openrouter_max_tokens,
    set_openrouter_model,
)
from src.handlers.message import clear_doc_session, _chat_session_id
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
        "• Dukungan teknis &amp; riset\n"
        "• Buat WBS &amp; estimasi man-days proyek\n"
        "• Kirim PDF → kuis interaktif HTML\n"
        "• Kirim PDF + <i>\"buat kuis telegram\"</i> → kuis polling langsung di Telegram\n"
        "/deploy        — Pull kode &amp; restart service (admin)\n"
        "/setprovider   — Pilih provider LLM: openrouter atau ollama (admin)\n"
        "/setapikey     — Atur API key OpenRouter (admin)\n"
        "/setmaxtokens  — Atur max tokens OpenRouter (admin)\n"
        "/setllmmodel   — Atur nama model LLM OpenRouter (admin)\n"
        "/setollamakey  — Atur API key Ollama (admin)\n"
        "/setollamahost — Atur host Ollama (admin)\n"
        "/setollamamodel — Atur nama model Ollama (admin)\n"
        "/setgithubtoken — Atur GitHub Personal Access Token (admin)\n"
        "/setgitlabtoken — Atur GitLab Personal Access Token (admin)\n"
        "/briefing       — Atur daily briefing (aktifkan, jam, topik, bahasa) (admin)"
    )
    await update.message.reply_html(help_text)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /ping — cek latensi bot."""
    logger.info("User %s mengirim ping.", update.effective_user.id)
    await update.message.reply_text("Pong! 🏓 Bot aktif dan merespons.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /reset — hapus riwayat percakapan pengguna."""
    user = update.effective_user
    chat = update.effective_chat
    logger.info("User %s mereset sesi.", user.id)
    await clear_session(_chat_session_id(user.id, chat.id))
    # Also wipe in-memory doc session state so the next message is routed
    # normally instead of being sent directly to DocAgent.
    clear_doc_session(str(user.id))
    await update.message.reply_text(
        "🔄 Riwayat percakapan telah dihapus. Browser web automation juga ditutup. Kita mulai dari awal!",
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
    except Exception as exc:  # noqa: BLE001
        logger.debug("setapikey: could not delete user message: %s", exc)

    await update.effective_chat.send_message(
        f"✅ API key OpenRouter berhasil disimpan ({masked}).\n"
        "Override akan digunakan untuk semua permintaan berikutnya tanpa perlu restart bot."
    )


async def setmaxtokens(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setmaxtokens — simpan atau hapus override max_tokens OpenRouter (admin only).

    Usage:
        /setmaxtokens <NILAI>   — simpan nilai baru (misal: /setmaxtokens 4096)
        /setmaxtokens clear     — hapus override, kembali ke nilai di .env
        /setmaxtokens status    — lihat nilai yang sedang aktif
    """
    settings = get_settings()
    user = update.effective_user

    # ── Cek izin admin ────────────────────────────────────────────────────────
    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setmaxtokens tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur max tokens.")
        return

    args = context.args

    # ── Tidak ada argumen → tampilkan panduan ─────────────────────────────────
    if not args:
        stored = get_openrouter_max_tokens()
        if stored is not None:
            status_line = f"✅ Override aktif: <b>{stored}</b> tokens"
        else:
            status_line = f"ℹ️ Tidak ada override (menggunakan .env: <b>{settings.openrouter_max_tokens}</b> tokens)"
        await update.message.reply_html(
            f"{status_line}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setmaxtokens &lt;NILAI&gt;</code>  — simpan nilai baru (contoh: 4096)\n"
            "<code>/setmaxtokens clear</code>       — hapus override\n"
            "<code>/setmaxtokens status</code>      — cek status override",
        )
        return

    sub = args[0].strip()

    # ── status ─────────────────────────────────────────────────────────────────
    if sub.lower() == "status":
        stored = get_openrouter_max_tokens()
        if stored is not None:
            await update.message.reply_text(f"✅ Override aktif: {stored} tokens")
        else:
            await update.message.reply_text(
                f"ℹ️ Tidak ada override — menggunakan nilai dari .env: {settings.openrouter_max_tokens} tokens"
            )
        return

    # ── clear ──────────────────────────────────────────────────────────────────
    if sub.lower() == "clear":
        clear_openrouter_max_tokens()
        logger.info("User %s menghapus OpenRouter max_tokens override.", user.id)
        await update.message.reply_text(
            f"🗑️ Override max_tokens dihapus. Kembali menggunakan nilai dari .env ({settings.openrouter_max_tokens} tokens)."
        )
        return

    # ── simpan nilai baru ──────────────────────────────────────────────────────
    try:
        new_value = int(sub)
        if new_value < 1:
            raise ValueError("must be positive")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Nilai tidak valid. Masukkan bilangan bulat positif, misal: /setmaxtokens 4096"
        )
        return

    set_openrouter_max_tokens(new_value)
    logger.info("User %s memperbarui OpenRouter max_tokens override ke %d.", user.id, new_value)
    await update.message.reply_text(
        f"✅ max_tokens OpenRouter berhasil disimpan: <b>{new_value}</b> tokens.\n"
        "Override berlaku untuk semua permintaan berikutnya tanpa perlu restart bot.",
        parse_mode="HTML",
    )


async def setllmmodel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setllmmodel — simpan atau hapus override nama model LLM OpenRouter (admin only).

    Usage:
        /setllmmodel <MODEL_NAME>  — simpan nama model baru (misal: /setllmmodel openai/gpt-4o)
        /setllmmodel clear         — hapus override, kembali ke model di .env
        /setllmmodel status        — lihat nama model yang sedang aktif
    """
    settings = get_settings()
    user = update.effective_user

    # ── Cek izin admin ────────────────────────────────────────────────────────
    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setllmmodel tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur model LLM.")
        return

    args = context.args

    # ── Tidak ada argumen → tampilkan panduan ─────────────────────────────────
    if not args:
        stored = get_openrouter_model()
        if stored:
            status_line = f"✅ Override aktif: <b>{stored}</b>"
        else:
            status_line = f"ℹ️ Tidak ada override (menggunakan .env: <b>{settings.openrouter_model}</b>)"
        await update.message.reply_html(
            f"{status_line}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setllmmodel &lt;MODEL_NAME&gt;</code>  — simpan nama model baru\n"
            "<code>/setllmmodel clear</code>           — hapus override\n"
            "<code>/setllmmodel status</code>          — cek status override",
        )
        return

    sub = args[0].strip()

    # ── status ─────────────────────────────────────────────────────────────────
    if sub.lower() == "status":
        stored = get_openrouter_model()
        if stored:
            await update.message.reply_text(f"✅ Override aktif: {stored}")
        else:
            await update.message.reply_text(
                f"ℹ️ Tidak ada override — menggunakan model dari .env: {settings.openrouter_model}"
            )
        return

    # ── clear ──────────────────────────────────────────────────────────────────
    if sub.lower() == "clear":
        clear_openrouter_model()
        logger.info("User %s menghapus OpenRouter model name override.", user.id)
        await update.message.reply_text(
            f"🗑️ Override model LLM dihapus. Kembali menggunakan model dari .env ({settings.openrouter_model})."
        )
        return

    # ── simpan nama model baru ────────────────────────────────────────────────
    new_model = sub
    set_openrouter_model(new_model)
    logger.info("User %s memperbarui OpenRouter model name override ke %s.", user.id, new_model)
    await update.message.reply_text(
        f"✅ Model LLM OpenRouter berhasil disimpan: <b>{new_model}</b>.\n"
        "Override berlaku untuk semua permintaan berikutnya tanpa perlu restart bot.",
        parse_mode="HTML",
    )


async def setprovider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setprovider — pilih provider LLM yang aktif (admin only).

    Usage:
        /setprovider openrouter  — gunakan OpenRouter sebagai provider aktif
        /setprovider ollama      — gunakan Ollama sebagai provider aktif
        /setprovider status      — lihat provider yang sedang aktif
        /setprovider clear       — hapus override, kembali ke default (openrouter)
    """
    settings = get_settings()
    user = update.effective_user

    # ── Cek izin admin ────────────────────────────────────────────────────────
    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setprovider tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur provider LLM.")
        return

    args = context.args

    # ── Tidak ada argumen → tampilkan panduan ─────────────────────────────────
    if not args:
        current = get_active_provider()
        await update.message.reply_html(
            f"✅ Provider aktif: <b>{current}</b>\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setprovider openrouter</code> — gunakan OpenRouter\n"
            "<code>/setprovider ollama</code>      — gunakan Ollama\n"
            "<code>/setprovider status</code>      — cek provider aktif\n"
            "<code>/setprovider clear</code>       — kembali ke default (openrouter)",
        )
        return

    sub = args[0].strip().lower()

    # ── status ─────────────────────────────────────────────────────────────────
    if sub == "status":
        current = get_active_provider()
        await update.message.reply_text(f"✅ Provider LLM aktif: {current}")
        return

    # ── clear ──────────────────────────────────────────────────────────────────
    if sub == "clear":
        clear_active_provider()
        logger.info("User %s menghapus LLM provider override.", user.id)
        await update.message.reply_text(
            f"🗑️ Override provider dihapus. Kembali ke default ({PROVIDER_OPENROUTER})."
        )
        return

    # ── set provider ───────────────────────────────────────────────────────────
    if sub not in {PROVIDER_OPENROUTER, PROVIDER_OLLAMA}:
        await update.message.reply_html(
            f"⚠️ Provider tidak valid: <b>{sub}</b>\n"
            f"Pilihan yang tersedia: <code>{PROVIDER_OPENROUTER}</code>, <code>{PROVIDER_OLLAMA}</code>"
        )
        return

    set_active_provider(sub)
    logger.info("User %s mengubah LLM provider ke %s.", user.id, sub)
    await update.message.reply_text(
        f"✅ Provider LLM berhasil diubah ke <b>{sub}</b>.\n"
        "Berlaku untuk semua permintaan berikutnya tanpa perlu restart bot.",
        parse_mode="HTML",
    )


async def setollamakey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setollamakey — simpan atau hapus override API key Ollama (admin only).

    Usage:
        /setollamakey <API_KEY>  — simpan key baru
        /setollamakey clear      — hapus override, kembali ke key di .env
        /setollamakey status     — lihat apakah override aktif
    """
    settings = get_settings()
    user = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setollamakey tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur Ollama API key.")
        return

    args = context.args

    if not args:
        current = get_ollama_key()
        status = "✅ Override aktif" if current else "ℹ️ Tidak ada override (menggunakan .env)"
        await update.message.reply_html(
            f"{status}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setollamakey &lt;API_KEY&gt;</code>  — simpan key baru\n"
            "<code>/setollamakey clear</code>        — hapus override\n"
            "<code>/setollamakey status</code>       — cek status override",
        )
        return

    sub = args[0].strip()

    if sub.lower() == "status":
        current = get_ollama_key()
        if current:
            masked = current[:8] + "..." + current[-4:] if len(current) > 12 else "***"
            await update.message.reply_text(f"✅ Override aktif: {masked}")
        else:
            await update.message.reply_text("ℹ️ Tidak ada override — menggunakan key dari .env")
        return

    if sub.lower() == "clear":
        clear_ollama_key()
        logger.info("User %s menghapus Ollama API key override.", user.id)
        await update.message.reply_text("🗑️ Override Ollama API key dihapus. Kembali menggunakan key dari .env.")
        return

    new_key = sub
    set_ollama_key(new_key)
    logger.info("User %s memperbarui Ollama API key override.", user.id)

    masked = new_key[:8] + "..." + new_key[-4:] if len(new_key) > 12 else "***"
    try:
        await update.message.delete()
    except Exception as exc:  # noqa: BLE001
        logger.debug("setollamakey: could not delete user message: %s", exc)

    await update.effective_chat.send_message(
        f"✅ API key Ollama berhasil disimpan ({masked}).\n"
        "Override akan digunakan untuk semua permintaan berikutnya tanpa perlu restart bot."
    )


async def setollamahost(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setollamahost — simpan atau hapus override host Ollama (admin only).

    Usage:
        /setollamahost <URL>     — simpan host baru (misal: http://localhost:11434)
        /setollamahost clear     — hapus override, kembali ke host di .env
        /setollamahost status    — lihat host yang sedang aktif
    """
    settings = get_settings()
    user = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setollamahost tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur Ollama host.")
        return

    args = context.args

    if not args:
        stored = get_ollama_host()
        if stored:
            status_line = f"✅ Override aktif: <b>{stored}</b>"
        else:
            status_line = f"ℹ️ Tidak ada override (menggunakan .env: <b>{settings.ollama_host}</b>)"
        await update.message.reply_html(
            f"{status_line}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setollamahost &lt;URL&gt;</code>   — simpan host baru\n"
            "<code>/setollamahost clear</code>      — hapus override\n"
            "<code>/setollamahost status</code>     — cek status override",
        )
        return

    sub = args[0].strip()

    if sub.lower() == "status":
        stored = get_ollama_host()
        if stored:
            await update.message.reply_text(f"✅ Override aktif: {stored}")
        else:
            await update.message.reply_text(
                f"ℹ️ Tidak ada override — menggunakan host dari .env: {settings.ollama_host}"
            )
        return

    if sub.lower() == "clear":
        clear_ollama_host()
        logger.info("User %s menghapus Ollama host override.", user.id)
        await update.message.reply_text(
            f"🗑️ Override Ollama host dihapus. Kembali menggunakan host dari .env ({settings.ollama_host})."
        )
        return

    new_host = sub
    set_ollama_host(new_host)
    logger.info("User %s memperbarui Ollama host ke %s.", user.id, new_host)
    await update.message.reply_text(
        f"✅ Host Ollama berhasil disimpan: <b>{new_host}</b>.\n"
        "Override berlaku untuk semua permintaan berikutnya tanpa perlu restart bot.",
        parse_mode="HTML",
    )


async def setollamamodel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setollamamodel — simpan atau hapus override nama model Ollama (admin only).

    Usage:
        /setollamamodel <MODEL_NAME>  — simpan nama model baru (misal: /setollamamodel llama3.2)
        /setollamamodel clear         — hapus override, kembali ke model di .env
        /setollamamodel status        — lihat nama model yang sedang aktif
    """
    settings = get_settings()
    user = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setollamamodel tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur model Ollama.")
        return

    args = context.args

    if not args:
        stored = get_ollama_model()
        if stored:
            status_line = f"✅ Override aktif: <b>{stored}</b>"
        else:
            status_line = f"ℹ️ Tidak ada override (menggunakan .env: <b>{settings.ollama_model}</b>)"
        await update.message.reply_html(
            f"{status_line}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setollamamodel &lt;MODEL_NAME&gt;</code>  — simpan nama model baru\n"
            "<code>/setollamamodel clear</code>           — hapus override\n"
            "<code>/setollamamodel status</code>          — cek status override",
        )
        return

    sub = args[0].strip()

    if sub.lower() == "status":
        stored = get_ollama_model()
        if stored:
            await update.message.reply_text(f"✅ Override aktif: {stored}")
        else:
            await update.message.reply_text(
                f"ℹ️ Tidak ada override — menggunakan model dari .env: {settings.ollama_model}"
            )
        return

    if sub.lower() == "clear":
        clear_ollama_model()
        logger.info("User %s menghapus Ollama model name override.", user.id)
        await update.message.reply_text(
            f"🗑️ Override model Ollama dihapus. Kembali menggunakan model dari .env ({settings.ollama_model})."
        )
        return

    new_model = sub
    set_ollama_model(new_model)
    logger.info("User %s memperbarui Ollama model name override ke %s.", user.id, new_model)
    await update.message.reply_text(
        f"✅ Model Ollama berhasil disimpan: <b>{new_model}</b>.\n"
        "Override berlaku untuk semua permintaan berikutnya tanpa perlu restart bot.",
        parse_mode="HTML",
    )


async def setgithubtoken(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setgithubtoken — simpan atau hapus override GitHub Personal Access Token (admin only).

    Usage:
        /setgithubtoken <TOKEN>  — simpan token baru
        /setgithubtoken clear    — hapus override, kembali ke token di .env
        /setgithubtoken status   — lihat apakah override aktif
    """
    settings = get_settings()
    user = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setgithubtoken tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur GitHub access token.")
        return

    args = context.args

    if not args:
        current = get_github_pat()
        status = "✅ Override aktif" if current else "ℹ️ Tidak ada override (menggunakan .env)"
        await update.message.reply_html(
            f"{status}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setgithubtoken &lt;TOKEN&gt;</code>  — simpan token baru\n"
            "<code>/setgithubtoken clear</code>        — hapus override\n"
            "<code>/setgithubtoken status</code>       — cek status override",
        )
        return

    sub = args[0].strip()

    if sub.lower() == "status":
        current = get_github_pat()
        if current:
            masked = current[:4] + "****" if len(current) > 4 else "***"
            await update.message.reply_text(f"✅ Override aktif: {masked}")
        else:
            await update.message.reply_text("ℹ️ Tidak ada override — menggunakan GitHub token dari .env")
        return

    if sub.lower() == "clear":
        clear_github_pat()
        logger.info("User %s menghapus GitHub PAT override.", user.id)
        await update.message.reply_text("🗑️ Override GitHub access token dihapus. Kembali menggunakan token dari .env.")
        return

    new_token = sub
    set_github_pat(new_token)
    logger.info("User %s memperbarui GitHub PAT override.", user.id)

    masked = new_token[:4] + "****" if len(new_token) > 4 else "***"
    try:
        await update.message.delete()
    except Exception as exc:  # noqa: BLE001
        logger.debug("setgithubtoken: could not delete user message: %s", exc)

    await update.effective_chat.send_message(
        f"✅ GitHub access token berhasil disimpan ({masked}).\n"
        "Override akan digunakan untuk semua operasi git GitHub berikutnya tanpa perlu restart bot."
    )


async def setgitlabtoken(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setgitlabtoken — simpan atau hapus override GitLab Personal Access Token (admin only).

    Usage:
        /setgitlabtoken <TOKEN>  — simpan token baru
        /setgitlabtoken clear    — hapus override, kembali ke token di .env
        /setgitlabtoken status   — lihat apakah override aktif
    """
    settings = get_settings()
    user = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        logger.warning("User %s mencoba /setgitlabtoken tapi bukan admin.", user.id)
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur GitLab access token.")
        return

    args = context.args

    if not args:
        current = get_gitlab_pat()
        status = "✅ Override aktif" if current else "ℹ️ Tidak ada override (menggunakan .env)"
        await update.message.reply_html(
            f"{status}\n\n"
            "<b>Penggunaan:</b>\n"
            "<code>/setgitlabtoken &lt;TOKEN&gt;</code>  — simpan token baru\n"
            "<code>/setgitlabtoken clear</code>         — hapus override\n"
            "<code>/setgitlabtoken status</code>        — cek status override",
        )
        return

    sub = args[0].strip()

    if sub.lower() == "status":
        current = get_gitlab_pat()
        if current:
            masked = current[:4] + "****" if len(current) > 4 else "***"
            await update.message.reply_text(f"✅ Override aktif: {masked}")
        else:
            await update.message.reply_text("ℹ️ Tidak ada override — menggunakan GitLab token dari .env")
        return

    if sub.lower() == "clear":
        clear_gitlab_pat()
        logger.info("User %s menghapus GitLab PAT override.", user.id)
        await update.message.reply_text("🗑️ Override GitLab access token dihapus. Kembali menggunakan token dari .env.")
        return

    new_token = sub
    set_gitlab_pat(new_token)
    logger.info("User %s memperbarui GitLab PAT override.", user.id)

    masked = new_token[:4] + "****" if len(new_token) > 4 else "***"
    try:
        await update.message.delete()
    except Exception as exc:  # noqa: BLE001
        logger.debug("setgitlabtoken: could not delete user message: %s", exc)

    await update.effective_chat.send_message(
        f"✅ GitLab access token berhasil disimpan ({masked}).\n"
        "Override akan digunakan untuk semua operasi git GitLab berikutnya tanpa perlu restart bot."
    )


# ── Daily Briefing command ────────────────────────────────────────────────────

_BRIEFING_HELP = (
    "<b>⚙️ Pengaturan Daily Briefing</b>\n\n"
    "<b>Sub-perintah:</b>\n"
    "<code>/briefing status</code>              — lihat config aktif\n"
    "<code>/briefing on</code>                  — aktifkan briefing\n"
    "<code>/briefing off</code>                 — nonaktifkan briefing\n"
    "<code>/briefing time HH:MM</code>          — atur jam kirim (zona WIB)\n"
    "  Contoh: <code>/briefing time 07:30</code>\n"
    "<code>/briefing topics topik1, topik2</code> — atur topik (pisah koma)\n"
    "  Contoh: <code>/briefing topics AI terbaru, saham, kripto</code>\n"
    "<code>/briefing lang id</code>             — bahasa Indonesia\n"
    "<code>/briefing lang en</code>             — bahasa English\n"
    "<code>/briefing chat ID</code>             — atur target chat_id\n"
    "  Contoh: <code>/briefing chat 123456789</code>\n"
    "<code>/briefing reset</code>               — kembalikan ke default (.env)\n"
    "<code>/briefing now</code>                 — kirim briefing sekarang (test)\n"
)


async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /briefing — atur daily briefing melalui Telegram (admin only).

    Sub-perintah:
        /briefing status               — tampilkan config aktif
        /briefing on | off             — aktifkan / nonaktifkan
        /briefing time HH:MM           — ubah jam kirim (WIB)
        /briefing topics t1, t2, ...   — ubah daftar topik
        /briefing lang id | en         — ubah bahasa laporan
        /briefing chat <chat_id>       — ubah target chat_id
        /briefing reset                — hapus semua override (kembali ke .env)
        /briefing now                  — kirim briefing sekarang untuk testing
    """
    settings = get_settings()
    user     = update.effective_user

    if settings.admin_user_id and user.id != settings.admin_user_id:
        await update.message.reply_text("⛔ Akses ditolak. Hanya admin yang bisa mengatur daily briefing.")
        return

    from src.memory.key_store import (
        get_briefing_enabled,
        get_briefing_time,
        get_briefing_topics,
        get_briefing_language,
        get_briefing_chat_id,
        set_briefing_enabled,
        set_briefing_time,
        set_briefing_topics,
        set_briefing_language,
        set_briefing_chat_id,
        clear_briefing_overrides,
    )
    from src.proactive.daily_briefing import reload_briefing_job, get_current_config, _run_briefing

    args = context.args or []

    if not args:
        await update.message.reply_html(_BRIEFING_HELP)
        return

    sub = args[0].lower()

    # ── status ─────────────────────────────────────────────────────────────
    if sub == "status":
        cfg = get_current_config()
        enabled_icon = "✅ Aktif" if cfg["enabled"] else "❌ Nonaktif"
        topics_str   = ", ".join(cfg["topics"])
        lang_label   = "Indonesia 🇮🇩" if cfg["language"] == "id" else "English 🇺🇸"
        store_note   = (
            "⚡ <i>Beberapa nilai di-override via command (lebih prioritas dari .env)</i>"
            if any([
                get_briefing_enabled() is not None,
                get_briefing_time(),
                get_briefing_topics(),
                get_briefing_language(),
                get_briefing_chat_id(),
            ])
            else "📄 <i>Menggunakan nilai dari .env</i>"
        )
        await update.message.reply_html(
            f"<b>📋 Status Daily Briefing</b>\n\n"
            f"<b>Status   :</b> {enabled_icon}\n"
            f"<b>Jam Kirim:</b> {cfg['time']} WIB\n"
            f"<b>Topik    :</b> {topics_str}\n"
            f"<b>Bahasa   :</b> {lang_label}\n"
            f"<b>Chat ID  :</b> <code>{cfg['chat_id']}</code>\n\n"
            f"{store_note}\n\n"
            "Ketik /briefing untuk melihat daftar sub-perintah."
        )
        return

    # ── on / off ───────────────────────────────────────────────────────────
    if sub in ("on", "off"):
        enabled = sub == "on"
        set_briefing_enabled(enabled)
        cfg = reload_briefing_job()
        icon = "✅" if enabled else "❌"
        status_text = "diaktifkan" if enabled else "dinonaktifkan"
        detail = (
            f"\n⏰ Jadwal: <b>{cfg['time']} WIB</b>" if enabled else ""
        )
        logger.info("User %s set briefing enabled=%s.", user.id, enabled)
        await update.message.reply_html(
            f"{icon} Daily briefing <b>{status_text}</b>.{detail}"
        )
        return

    # ── time ───────────────────────────────────────────────────────────────
    if sub == "time":
        if len(args) < 2:
            await update.message.reply_html(
                "❗ Format: <code>/briefing time HH:MM</code>\n"
                "Contoh  : <code>/briefing time 07:30</code>\n"
                "<i>Waktu dalam zona WIB (UTC+7)</i>"
            )
            return

        time_str = args[1].strip()
        parts    = time_str.split(":")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            await update.message.reply_html(
                f"❌ Format waktu tidak valid: <code>{time_str}</code>\n"
                "Gunakan format <b>HH:MM</b>, contoh: <code>07:30</code>"
            )
            return

        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            await update.message.reply_html(
                f"❌ Waktu tidak valid: <code>{time_str}</code>\n"
                "Jam: 0–23, Menit: 0–59"
            )
            return

        time_str = f"{hour:02d}:{minute:02d}"  # normalise ke HH:MM
        set_briefing_time(time_str)
        cfg = reload_briefing_job()
        logger.info("User %s set briefing time to %s.", user.id, time_str)
        await update.message.reply_html(
            f"✅ Jam briefing diubah ke <b>{time_str} WIB</b>.\n"
            f"{'⚡ Briefing aktif.' if cfg['enabled'] else '⚠️ Briefing saat ini nonaktif — aktifkan dengan /briefing on'}"
        )
        return

    # ── topics ─────────────────────────────────────────────────────────────
    if sub == "topics":
        if len(args) < 2:
            await update.message.reply_html(
                "❗ Format: <code>/briefing topics topik1, topik2, topik3</code>\n"
                "Contoh  : <code>/briefing topics AI terbaru, saham hari ini, startup Indonesia</code>"
            )
            return

        raw_topics = " ".join(args[1:])
        new_topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
        if not new_topics:
            await update.message.reply_text("❌ Tidak ada topik valid yang diberikan.")
            return

        set_briefing_topics(", ".join(new_topics))
        reload_briefing_job()
        topics_display = "\n".join(f"  • {t}" for t in new_topics)
        logger.info("User %s set briefing topics: %s.", user.id, new_topics)
        await update.message.reply_html(
            f"✅ Topik briefing diperbarui ({len(new_topics)} topik):\n{topics_display}"
        )
        return

    # ── lang ───────────────────────────────────────────────────────────────
    if sub == "lang":
        if len(args) < 2 or args[1].lower() not in ("id", "en"):
            await update.message.reply_html(
                "❗ Format: <code>/briefing lang id</code> atau <code>/briefing lang en</code>\n"
                "  <b>id</b> = Bahasa Indonesia\n"
                "  <b>en</b> = English"
            )
            return

        lang = args[1].lower()
        set_briefing_language(lang)
        reload_briefing_job()
        label = "Indonesia 🇮🇩" if lang == "id" else "English 🇺🇸"
        logger.info("User %s set briefing language to %s.", user.id, lang)
        await update.message.reply_html(f"✅ Bahasa briefing diubah ke <b>{label}</b>.")
        return

    # ── chat ───────────────────────────────────────────────────────────────
    if sub == "chat":
        if len(args) < 2:
            await update.message.reply_html(
                "❗ Format: <code>/briefing chat &lt;chat_id&gt;</code>\n"
                "Contoh  : <code>/briefing chat 123456789</code>\n\n"
                "💡 Chat ID Anda saat ini: "
                f"<code>{update.effective_chat.id}</code>"
            )
            return

        new_chat_id = args[1].strip()
        if not new_chat_id or not new_chat_id.lstrip("-").isdigit():
            await update.message.reply_html(
                f"❌ Chat ID tidak valid: <code>{new_chat_id}</code>\n"
                "Chat ID harus berupa angka."
            )
            return

        set_briefing_chat_id(new_chat_id)
        reload_briefing_job()
        logger.info("User %s set briefing chat_id to %s.", user.id, new_chat_id)
        await update.message.reply_html(
            f"✅ Target chat ID briefing diubah ke <code>{new_chat_id}</code>."
        )
        return

    # ── reset ──────────────────────────────────────────────────────────────
    if sub == "reset":
        clear_briefing_overrides()
        cfg = reload_briefing_job()
        logger.info("User %s reset briefing config to .env defaults.", user.id)
        enabled_icon = "✅ Aktif" if cfg["enabled"] else "❌ Nonaktif"
        await update.message.reply_html(
            "🔄 <b>Config briefing direset ke nilai .env.</b>\n\n"
            f"<b>Status   :</b> {enabled_icon}\n"
            f"<b>Jam Kirim:</b> {cfg['time']} WIB\n"
            f"<b>Topik    :</b> {', '.join(cfg['topics'])}\n"
            f"<b>Bahasa   :</b> {'Indonesia 🇮🇩' if cfg['language'] == 'id' else 'English 🇺🇸'}\n"
            f"<b>Chat ID  :</b> <code>{cfg['chat_id']}</code>"
        )
        return

    # ── now ────────────────────────────────────────────────────────────────
    if sub == "now":
        cfg = get_current_config()
        if not cfg["chat_id"] or cfg["chat_id"] == "0":
            await update.message.reply_text(
                "❌ Tidak ada chat_id yang dikonfigurasi.\n"
                "Gunakan /briefing chat <id> atau atur PROACTIVE_BRIEFING_CHAT_ID di .env."
            )
            return

        await update.message.reply_text(
            f"⏳ Menjalankan briefing sekarang...\n"
            f"Topik: {', '.join(cfg['topics'])}\n"
            f"Target: chat_id={cfg['chat_id']}"
        )
        logger.info("User %s triggered immediate briefing run.", user.id)
        try:
            await _run_briefing(cfg["chat_id"], cfg["topics"], cfg["language"])
        except Exception as exc:
            logger.error("Immediate briefing failed: %s", exc)
            await update.message.reply_text(f"❌ Briefing gagal: {exc}")
        return

    # ── unknown sub-command ────────────────────────────────────────────────
    await update.message.reply_html(
        f"❓ Sub-perintah tidak dikenali: <code>{args[0]}</code>\n\n"
        + _BRIEFING_HELP
    )

