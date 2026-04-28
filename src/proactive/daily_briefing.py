"""
ProactiveBriefingJob – Fase 4: Automated Daily Briefing.

Menjalankan ResearcherAgent secara terjadwal dan mengirimkan ringkasan
berita/topik pilihan ke Telegram setiap hari pada waktu yang dikonfigurasi.

Konfigurasi via `.env` (semua prefix PROACTIVE_BRIEFING_*):

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


# ── Job entry point ────────────────────────────────────────────────────────────

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
            summary = await agent.research_for_delegation(
                query=(
                    f"Berikan ringkasan berita dan perkembangan terkini tentang: {topic}. "
                    f"Fokus pada hal yang paling penting dan relevan hari ini. "
                    f"Gunakan bahasa {'Indonesia' if language == 'id' else 'English'}."
                ),
                session_id="proactive_briefing",
            )
            if summary and "[Research unavailable" not in summary:
                sections.append(f"📌 *{topic.strip().title()}*\n{summary}")
        except Exception as exc:
            logger.warning("ProactiveBriefing: topic '%s' failed: %s", topic, exc)

    if not sections:
        logger.warning("ProactiveBriefing: all topics failed, skipping send.")
        return

    from telegram.constants import ParseMode
    from datetime import datetime

    now_wib = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    header  = (
        f"☀️ *Briefing Harian — {now_wib} WIB*\n"
        f"{'─' * 30}\n\n"
    )
    body    = "\n\n".join(sections)
    message = header + body

    # Telegram max message = 4096 chars; potong agar tidak error
    MAX_LEN = 4000
    if len(message) > MAX_LEN:
        message = message[:MAX_LEN] + "\n\n_[pesan terpotong]_"

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(
            "ProactiveBriefing: sent to chat_id=%s (%d chars)", chat_id, len(message)
        )
    except Exception as exc:
        logger.error("ProactiveBriefing: failed to send to %s: %s", chat_id, exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def start_briefing_job(bot: "Bot") -> None:
    """
    Daftarkan (atau update) daily briefing job ke APScheduler.

    Dipanggil sekali saat startup dari ``telegram_bot._send_startup_notification()``.
    Job hanya didaftarkan jika ``PROACTIVE_BRIEFING_ENABLED=true``.

    Args:
        bot: Telegram Bot instance untuk mengirim pesan.
    """
    from config.settings import get_settings
    from src.agents.reminder_agent.scheduler import get_scheduler
    from src.proactive._bot_ref import set_bot

    settings = get_settings()

    if not settings.proactive_briefing_enabled:
        logger.info("ProactiveBriefing: disabled (PROACTIVE_BRIEFING_ENABLED not set).")
        return

    # Tentukan target chat_id
    chat_id = (
        settings.proactive_briefing_chat_id.strip()
        or str(settings.admin_user_id)
    )
    if not chat_id or chat_id == "0":
        logger.warning(
            "ProactiveBriefing: no chat_id configured "
            "(set PROACTIVE_BRIEFING_CHAT_ID or ADMIN_USER_ID). Disabled."
        )
        return

    # Parse waktu "HH:MM" WIB → UTC (APScheduler pakai UTC)
    time_str = settings.proactive_briefing_time.strip() or "07:00"
    try:
        hour_wib, minute = (int(x) for x in time_str.split(":"))
    except ValueError:
        logger.warning(
            "ProactiveBriefing: invalid PROACTIVE_BRIEFING_TIME '%s', using 07:00.", time_str
        )
        hour_wib, minute = 7, 0

    hour_utc = (hour_wib - _WIB_OFFSET_HOURS) % 24

    # Parse topik
    raw_topics = settings.proactive_briefing_topics.strip()
    topics     = [t.strip() for t in raw_topics.split(",") if t.strip()] if raw_topics else [
        "AI terbaru",
        "berita teknologi",
        "startup Indonesia",
    ]

    language = settings.proactive_briefing_language.strip().lower() or "id"

    # Simpan referensi bot agar callback bisa mengirim pesan
    set_bot(bot)

    sched   = get_scheduler()
    trigger = CronTrigger(hour=hour_utc, minute=minute, timezone=timezone.utc)

    # Hapus job lama jika ada (idempotent pada restart)
    if sched.get_job(_JOB_ID):
        sched.remove_job(_JOB_ID)

    sched.add_job(
        _run_briefing,
        trigger  = trigger,
        id       = _JOB_ID,
        args     = [chat_id, topics, language],
        misfire_grace_time = 600,  # toleransi 10 menit keterlambatan
    )

    logger.info(
        "ProactiveBriefing: job scheduled at %02d:%02d WIB (%02d:%02d UTC) "
        "chat_id=%s topics=%s",
        hour_wib, minute, hour_utc, minute, chat_id, topics,
    )


def stop_briefing_job() -> None:
    """Hapus daily briefing job dari scheduler (dipanggil saat shutdown)."""
    from src.agents.reminder_agent.scheduler import get_scheduler

    sched = get_scheduler()
    if sched.get_job(_JOB_ID):
        sched.remove_job(_JOB_ID)
        logger.info("ProactiveBriefing: job removed.")
