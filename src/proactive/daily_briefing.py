"""
ProactiveBriefingJob – Fase 4: Automated Daily Briefing.

Menjalankan ResearcherAgent secara terjadwal dan mengirimkan ringkasan
berita/topik pilihan ke Telegram setiap hari pada waktu yang dikonfigurasi.

Konfigurasi tersedia dua cara (prioritas store > .env):

A. Runtime (via command Telegram /briefing):
   Config disimpan di runtime_keys.json dan berlaku tanpa restart.

B. Default via `.env` (semua prefix PROACTIVE_BRIEFING_*):

    PROACTIVE_BRIEFING_ENABLED=true
    PROACTIVE_BRIEFING_CHAT_ID=<telegram_chat_id>   # default: ADMIN_USER_ID
    PROACTIVE_BRIEFING_TIME=07:00                    # format HH:MM, zona WIB (UTC+7)
    PROACTIVE_BRIEFING_TOPICS=AI terbaru, berita teknologi, startup Indonesia
    PROACTIVE_BRIEFING_LANGUAGE=id                   # id atau en

Cara kerja:
    1. APScheduler menembak job setiap hari pada waktu yang ditentukan.
    2. Job memanggil ResearcherAgent.research_for_delegation() untuk setiap topik.
    3. Hasil digabung menjadi satu pesan dan dikirim via Telegram Bot.

Integrasi:
    Panggil ``start_briefing_job(bot)`` sekali saat startup, setelah
    ``start_scheduler()`` dari reminder_agent.scheduler.
    Panggil ``reload_briefing_job()`` setelah config diubah via command.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Optional, TYPE_CHECKING

from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from telegram import Bot

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Internal job ID – unik agar tidak bentrok dengan reminder jobs
_JOB_ID = "proactive_daily_briefing"

# WIB = UTC+7
_WIB_OFFSET_HOURS = 7


# ── Effective config resolver ──────────────────────────────────────────────────

def get_current_config() -> dict:
    """
    Kembalikan config briefing yang sedang aktif (store override > .env default).

    Returns dict dengan keys:
        enabled  (bool)
        chat_id  (str)
        time     (str)  — "HH:MM" WIB
        topics   (list[str])
        language (str)  — "id" atau "en"
    """
    from config.settings import get_settings
    from src.memory.key_store import (
        get_briefing_enabled,
        get_briefing_time,
        get_briefing_topics,
        get_briefing_language,
        get_briefing_chat_id,
    )

    settings = get_settings()

    # enabled
    store_enabled = get_briefing_enabled()
    enabled = store_enabled if store_enabled is not None else settings.proactive_briefing_enabled

    # chat_id
    store_chat_id = get_briefing_chat_id()
    chat_id = (
        store_chat_id
        or settings.proactive_briefing_chat_id.strip()
        or str(settings.admin_user_id)
    )

    # time
    store_time = get_briefing_time()
    time_str   = store_time or settings.proactive_briefing_time.strip() or "07:00"

    # topics
    store_topics = get_briefing_topics()
    raw_topics   = store_topics or settings.proactive_briefing_topics.strip()
    topics       = [t.strip() for t in raw_topics.split(",") if t.strip()] if raw_topics else [
        "AI terbaru",
        "berita teknologi",
        "startup Indonesia",
    ]

    # language
    store_lang = get_briefing_language()
    language   = store_lang or settings.proactive_briefing_language.strip().lower() or "id"

    return {
        "enabled":  enabled,
        "chat_id":  chat_id,
        "time":     time_str,
        "topics":   topics,
        "language": language,
    }


# ── Job entry point ────────────────────────────────────────────────────────────

async def _send_long_message(bot: "Bot", chat_id: str, text: str) -> None:
    """Kirim pesan panjang dengan membaginya menjadi beberapa bagian (maksimal 4000 karakter per bagian).

    Membagi teks berdasarkan baris/paragraf agar format tetap rapi dan tidak merusak
    tag markdown. Jika terjadi kesalahan parsing markdown, bagian tersebut dikirim
    kembali sebagai teks biasa (tanpa formatting).
    """
    from telegram.constants import ParseMode

    if len(text) <= 4000:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:
            logger.warning("Failed to send markdown message, falling back to plain text: %s", exc)
            await bot.send_message(
                chat_id=chat_id,
                text=text,
            )
        return

    paragraphs = text.split("\n")
    current_chunk = ""

    for para in paragraphs:
        # Jika penambahan paragraf ini melebihi 4000 karakter, kirim chunk saat ini
        if len(current_chunk) + len(para) + 1 > 4000:
            if current_chunk:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=current_chunk,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                except Exception as exc:
                    logger.warning("Failed to send markdown chunk, falling back to plain text: %s", exc)
                    await bot.send_message(
                        chat_id=chat_id,
                        text=current_chunk,
                    )
            # Jika paragraf itu sendiri > 4000 karakter, pecah berdasarkan karakter
            if len(para) > 4000:
                sub_chunks = [para[i:i+4000] for i in range(0, len(para), 4000)]
                for sub in sub_chunks[:-1]:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=sub,
                    )
                current_chunk = sub_chunks[-1]
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para

    if current_chunk:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=current_chunk,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:
            logger.warning("Failed to send final markdown chunk, falling back to plain text: %s", exc)
            await bot.send_message(
                chat_id=chat_id,
                text=current_chunk,
            )


async def _run_briefing(chat_id: str, topics: list[str], language: str) -> None:
    """APScheduler callback: research semua topik lalu kirim ke Telegram."""
    from src.agents.llm_client import LLMClient
    from src.agents.researcher.agent import ResearcherAgent
    from src.memory.history import ConversationHistory
    from src.proactive._bot_ref import get_bot

    bot = get_bot()
    if bot is None:
        logger.error("ProactiveBriefing: bot instance not set, skipping.")
        return

    logger.info(
        "ProactiveBriefing: starting briefing for chat_id=%s topics=%s",
        chat_id, topics,
    )

    # Buat instance ringan — tidak perlu persistent history untuk proactive job
    llm      = LLMClient()
    history  = ConversationHistory(max_messages=5)
    agent    = ResearcherAgent(history, llm)

    sections: list[str] = []
    for topic in topics:
        try:
            summary = await agent.research_for_briefing(
                topic=topic,
                language=language,
                session_id="proactive_briefing",
            )
            if summary and "[Briefing research failed" not in summary:
                sections.append(f"📌 *{topic.strip().title()}*\n{summary}")
        except Exception as exc:
            logger.warning("ProactiveBriefing: topic '%s' failed: %s", topic, exc)

    if not sections:
        logger.warning("ProactiveBriefing: all topics failed, skipping send.")
        return

    from datetime import datetime, timezone, timedelta

    now_wib = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")
    header  = (
        f"☀️ *Briefing Harian — {now_wib} WIB*\n"
        f"{'─' * 30}\n\n"
    )
    body    = "\n\n".join(sections)
    message = header + body

    try:
        await _send_long_message(bot, chat_id, message)
        logger.info(
            "ProactiveBriefing: sent to chat_id=%s (%d chars)", chat_id, len(message)
        )
    except Exception as exc:
        logger.error("ProactiveBriefing: failed to send to %s: %s", chat_id, exc)


# ── Internal: schedule helper ──────────────────────────────────────────────────

def _schedule_from_config(cfg: dict) -> None:
    """Terapkan config ke APScheduler. Hapus job lama, daftarkan yang baru."""
    from src.agents.reminder_agent.scheduler import get_scheduler

    sched = get_scheduler()

    # Selalu hapus job lama terlebih dahulu
    if sched.get_job(_JOB_ID):
        sched.remove_job(_JOB_ID)

    if not cfg["enabled"]:
        logger.info("ProactiveBriefing: disabled — job not scheduled.")
        return

    chat_id = cfg["chat_id"]
    if not chat_id or chat_id == "0":
        logger.warning(
            "ProactiveBriefing: no chat_id configured. Job not scheduled."
        )
        return

    time_str = cfg["time"]
    try:
        hour_wib, minute = (int(x) for x in time_str.split(":"))
    except ValueError:
        logger.warning(
            "ProactiveBriefing: invalid time '%s', defaulting to 07:00.", time_str
        )
        hour_wib, minute = 7, 0

    hour_utc = (hour_wib - _WIB_OFFSET_HOURS) % 24
    trigger  = CronTrigger(hour=hour_utc, minute=minute, timezone=timezone.utc)

    sched.add_job(
        _run_briefing,
        trigger            = trigger,
        id                 = _JOB_ID,
        args               = [chat_id, cfg["topics"], cfg["language"]],
        misfire_grace_time = 600,
    )
    logger.info(
        "ProactiveBriefing: job scheduled at %02d:%02d WIB (%02d:%02d UTC) "
        "chat_id=%s topics=%s",
        hour_wib, minute, hour_utc, minute, chat_id, cfg["topics"],
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def start_briefing_job(bot: "Bot") -> None:
    """
    Daftarkan (atau update) daily briefing job ke APScheduler.

    Dipanggil sekali saat startup dari ``telegram_bot._send_startup_notification()``.

    Args:
        bot: Telegram Bot instance untuk mengirim pesan.
    """
    from src.proactive._bot_ref import set_bot

    set_bot(bot)
    cfg = get_current_config()
    _schedule_from_config(cfg)


def reload_briefing_job() -> dict:
    """
    Muat ulang config dari key_store / .env dan terapkan ke APScheduler.

    Dipanggil oleh command handler ``/briefing`` setelah mengubah salah satu
    pengaturan.  Tidak memerlukan argumen — selalu membaca config terbaru.

    Returns:
        dict config yang sedang aktif (sama dengan ``get_current_config()``).
    """
    cfg = get_current_config()
    _schedule_from_config(cfg)
    return cfg


def stop_briefing_job() -> None:
    """Hapus daily briefing job dari scheduler (dipanggil saat shutdown)."""
    from src.agents.reminder_agent.scheduler import get_scheduler

    sched = get_scheduler()
    if sched.get_job(_JOB_ID):
        sched.remove_job(_JOB_ID)
        logger.info("ProactiveBriefing: job removed.")

