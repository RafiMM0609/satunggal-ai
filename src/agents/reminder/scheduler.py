"""
Reminder Scheduler – APScheduler-based job manager for timed Telegram reminders.

A single AsyncIOScheduler instance is shared across the process.  Jobs are
stored in-memory (no external job store needed since reminders are persisted in
SQLite by ReminderStore and rescheduled on startup).

Usage:
    from src.agents.reminder.scheduler import (
        start_scheduler,
        stop_scheduler,
        schedule_reminder,
        cancel_scheduled_reminder,
        reschedule_pending_on_startup,
    )

The Telegram bot instance is required to send messages.  It is injected once
via `set_bot(bot)` during application startup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

if TYPE_CHECKING:
    from telegram import Bot
    from src.tools.reminder_store import Reminder

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

_scheduler: Optional[AsyncIOScheduler] = None
_bot: Optional["Bot"] = None

_JOB_ID_PREFIX = "reminder_"


def set_bot(bot: "Bot") -> None:
    """Inject the Telegram Bot instance used to send reminder messages."""
    global _bot
    _bot = bot
    logger.info("Reminder scheduler: bot instance set.")


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
        logger.info("Reminder AsyncIOScheduler created.")
    return _scheduler


def start_scheduler() -> None:
    """Start the scheduler (idempotent)."""
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("Reminder scheduler started.")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        logger.info("Reminder scheduler stopped.")


# ── Job lifecycle ─────────────────────────────────────────────────────────────

async def schedule_reminder(reminder: "Reminder") -> None:
    """Add a one-shot job that fires at reminder.remind_at (UTC)."""
    sched = get_scheduler()
    job_id = f"{_JOB_ID_PREFIX}{reminder.id}"

    # Remove old job with same id if it exists (idempotent reschedule)
    if sched.get_job(job_id):
        sched.remove_job(job_id)

    now_utc = datetime.now(timezone.utc)
    run_at = reminder.remind_at

    # If the time is already past (e.g. on startup recovery for very old reminders),
    # fire immediately with a tiny delay so the event loop is ready.
    if run_at <= now_utc:
        run_at = now_utc + timedelta(seconds=2)

    sched.add_job(
        _fire_reminder,
        trigger=DateTrigger(run_date=run_at, timezone=timezone.utc),
        id=job_id,
        args=[reminder.id, reminder.chat_id, reminder.message],
        misfire_grace_time=300,   # allow up to 5 min latency before giving up
    )
    wib_time = run_at + timedelta(hours=7)
    logger.info(
        "Reminder job scheduled: id=%s run_at=%s WIB",
        job_id,
        wib_time.strftime("%Y-%m-%d %H:%M"),
    )


def cancel_scheduled_reminder(reminder_id: int) -> None:
    """Remove the APScheduler job for the given reminder id (if it exists)."""
    sched = get_scheduler()
    job_id = f"{_JOB_ID_PREFIX}{reminder_id}"
    if sched.get_job(job_id):
        sched.remove_job(job_id)
        logger.info("Reminder job cancelled: id=%s", job_id)


async def reschedule_pending_on_startup() -> None:
    """
    On bot startup, reload all pending (unfired) reminders from SQLite and
    create APScheduler jobs for them so reminders survive a process restart.
    """
    from src.tools.reminder_store import get_reminder_store

    store = get_reminder_store()
    pending = store.list_all_pending()
    logger.info("Rescheduling %d pending reminder(s) on startup.", len(pending))
    for reminder in pending:
        await schedule_reminder(reminder)


# ── Job callback ──────────────────────────────────────────────────────────────

async def _fire_reminder(reminder_id: int, chat_id: str, message: str) -> None:
    """Called by APScheduler when a reminder fires."""
    from src.tools.reminder_store import get_reminder_store

    store = get_reminder_store()

    # Guard: skip if already fired (e.g. duplicate job on restart)
    reminder = store.get(reminder_id)
    if reminder is None or reminder.fired:
        logger.info("Reminder #%d already fired or not found, skipping.", reminder_id)
        return

    text = f"⏰ *Pengingat!*\n\n{message}"

    if _bot is None:
        logger.error("Reminder #%d: bot instance not set, cannot send message.", reminder_id)
        return

    try:
        from telegram.constants import ParseMode
        await _bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
        store.mark_fired(reminder_id)
        logger.info("Reminder #%d fired to chat_id=%s.", reminder_id, chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send reminder #%d to %s: %s", reminder_id, chat_id, exc)
