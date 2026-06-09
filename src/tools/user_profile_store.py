"""
UserProfileStore – SQLite-backed persistence for User Profiles & Long-Term Preferences.

Stores key-value pairs per chat_id/session_id to be consumed by ReminderAgent.
Contains default preferences mapping to the original hardcoded prompt guidelines.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "reminders.db"

DEFAULT_PREFERENCES = {
    "preferred_name": "User",
    "timezone_offset": "7",  # WIB (UTC+7)
    "auto_prep_important_events": "true",
    "prep_time_minutes": "30",
    "important_event_keywords": '["meeting", "interview", "presentasi", "penerbangan", "ujian", "deadline"]',
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "07:00",
    # Research preference defaults (shared database configuration)
    "explanation_style": "detailed",  # detailed, concise, or code_focused
    "ignored_domains": "[]",  # JSON list of domains to ignore in web search
    "trusted_domains": "[]",  # JSON list of domains to trust/prioritize in web search
    # Responder preference defaults (shared database configuration)
    "preferred_vibe": "auto",  # auto, formal, office, or genz
}


class UserProfileStore:
    """Thread-safe SQLite store for user preferences."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id       TEXT    NOT NULL,
                    profile_key   TEXT    NOT NULL,
                    profile_value TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    UNIQUE(chat_id, profile_key)
                )
            """)
            conn.commit()

    def set_preference(self, chat_id: str, key: str, value: str) -> None:
        """Insert or replace a user profile preference."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO user_profiles (chat_id, profile_key, profile_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, profile_key) DO UPDATE SET
                    profile_value = excluded.profile_value,
                    updated_at = excluded.updated_at
            """, (chat_id, key.strip(), value.strip(), now_str))
            conn.commit()
        logger.info("UserProfileStore updated: chat_id=%s, key=%s", chat_id, key)

    def get_preference(self, chat_id: str, key: str) -> Optional[str]:
        """Get a single preference or fall back to default."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_value FROM user_profiles WHERE chat_id = ? AND profile_key = ?",
                (chat_id, key)
            ).fetchone()
        if row:
            return row["profile_value"]
        return DEFAULT_PREFERENCES.get(key)

    def get_all_preferences(self, chat_id: str) -> dict[str, str]:
        """Get all preferences for a user, merged with defaults."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT profile_key, profile_value FROM user_profiles WHERE chat_id = ?",
                (chat_id,)
            ).fetchall()
        
        user_prefs = {row["profile_key"]: row["profile_value"] for row in rows}
        
        # Merge with defaults
        merged = DEFAULT_PREFERENCES.copy()
        merged.update(user_prefs)
        return merged

    def delete_preference(self, chat_id: str, key: str) -> bool:
        """Delete a preference. Returns True if row was deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM user_profiles WHERE chat_id = ? AND profile_key = ?",
                (chat_id, key)
            )
            conn.commit()
        return cur.rowcount > 0

    def clear_profile(self, chat_id: str) -> None:
        """Clear all user preferences."""
        with self._connect() as conn:
            conn.execute("DELETE FROM user_profiles WHERE chat_id = ?", (chat_id,))
            conn.commit()
        logger.info("UserProfileStore cleared: chat_id=%s", chat_id)


# Singleton instance
_store: Optional[UserProfileStore] = None


def get_user_profile_store() -> UserProfileStore:
    global _store
    if _store is None:
        _store = UserProfileStore()
    return _store
