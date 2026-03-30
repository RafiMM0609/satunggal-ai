"""
ReminderStore – SQLite-backed persistence for the Reminder Agent.

Schema (table: reminders):
  id         INTEGER PRIMARY KEY AUTOINCREMENT
  chat_id    TEXT    NOT NULL   – Telegram chat_id (same as session_id)
  message    TEXT    NOT NULL   – what to remind
  remind_at  TEXT    NOT NULL   – ISO-8601 UTC datetime string
  created_at TEXT    NOT NULL   – ISO-8601 UTC datetime string
  fired      INTEGER NOT NULL DEFAULT 0  – 0=pending, 1=sent/cancelled

All datetimes are stored in UTC as ISO-8601 strings so the DB is
timezone-agnostic and portable.
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "reminders.db"


@dataclass
class Reminder:
    id: int
    chat_id: str
    message: str
    remind_at: datetime        # always UTC
    created_at: datetime       # always UTC
    fired: bool

    @property
    def remind_at_local_str(self) -> str:
        """Return remind_at as a human-readable UTC string."""
        return self.remind_at.strftime("%Y-%m-%d %H:%M UTC")


class ReminderStore:
    """Thread-safe SQLite store for reminders."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT    NOT NULL,
                    message    TEXT    NOT NULL,
                    remind_at  TEXT    NOT NULL,
                    created_at TEXT    NOT NULL,
                    fired      INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            chat_id=row["chat_id"],
            message=row["message"],
            remind_at=datetime.fromisoformat(row["remind_at"]).replace(tzinfo=timezone.utc),
            created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
            fired=bool(row["fired"]),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, chat_id: str, message: str, remind_at: datetime) -> Reminder:
        """Insert a new reminder and return it with its generated id."""
        now_utc = datetime.now(timezone.utc)
        # Strip tzinfo before storing so SQLite doesn't choke on offset strings
        remind_at_str = remind_at.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
        created_at_str = now_utc.replace(tzinfo=None).isoformat()

        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reminders (chat_id, message, remind_at, created_at, fired) "
                "VALUES (?, ?, ?, ?, 0)",
                (chat_id, message, remind_at_str, created_at_str),
            )
            conn.commit()
            row_id = cur.lastrowid

        reminder = self.get(row_id)
        logger.info("Reminder added: id=%d chat_id=%s remind_at=%s", row_id, chat_id, remind_at_str)
        return reminder

    def get(self, reminder_id: int) -> Optional[Reminder]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return self._row_to_reminder(row) if row else None

    def list_pending(self, chat_id: str) -> List[Reminder]:
        """Return all unfired reminders for a specific chat."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE chat_id = ? AND fired = 0 ORDER BY remind_at",
                (chat_id,),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def list_all_pending(self) -> List[Reminder]:
        """Return all unfired reminders across all chats (used on startup to reschedule)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE fired = 0 ORDER BY remind_at"
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def mark_fired(self, reminder_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
            conn.commit()
        logger.info("Reminder marked fired: id=%d", reminder_id)

    def delete(self, reminder_id: int, chat_id: str) -> bool:
        """Delete a reminder owned by *chat_id*. Returns True if a row was deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE id = ? AND chat_id = ? AND fired = 0",
                (reminder_id, chat_id),
            )
            conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Reminder deleted: id=%d chat_id=%s", reminder_id, chat_id)
        return deleted


# ── Module-level singleton ────────────────────────────────────────────────────
_store: Optional[ReminderStore] = None


def get_reminder_store() -> ReminderStore:
    global _store
    if _store is None:
        _store = ReminderStore()
    return _store
