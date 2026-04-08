"""
UserModeStore – persistent SQLite store for per-session active mode.

Schema:
    user_modes(session_id TEXT PRIMARY KEY, active_mode TEXT, updated_at DATETIME)

All public methods are synchronous and can be called safely from async code
via ``asyncio.to_thread()``, following the same pattern as RepoTracker.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

_DEFAULT_MODE = "all"

_DDL = """
CREATE TABLE IF NOT EXISTS user_modes (
    session_id  TEXT     PRIMARY KEY,
    active_mode TEXT     NOT NULL DEFAULT 'all',
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class UserModeStore:
    """Lightweight SQLite store for tracking each session's active mode.

    Usage::

        store = UserModeStore()
        mode  = store.get_mode(session_id)          # returns "all" by default
        store.set_mode(session_id, "dev")
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parents[2] / "data" / "user_modes.db"

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a WAL-mode connection."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema migration ──────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Ensure the schema exists; idempotent."""
        with self._connect() as conn:
            conn.executescript(_DDL)
        logger.debug("UserModeStore: schema ready at %s", self._db_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_mode(self, session_id: str) -> str:
        """Return the active mode for *session_id*, defaulting to ``"all"``."""
        sql = "SELECT active_mode FROM user_modes WHERE session_id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (session_id,)).fetchone()
        return row["active_mode"] if row else _DEFAULT_MODE

    def set_mode(self, session_id: str, mode: str) -> None:
        """Upsert the active mode for *session_id*."""
        sql = """
            INSERT INTO user_modes (session_id, active_mode, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                active_mode = excluded.active_mode,
                updated_at  = CURRENT_TIMESTAMP
        """
        with self._connect() as conn:
            conn.execute(sql, (session_id, mode))
        logger.info("UserModeStore: session=%s mode set to '%s'", session_id, mode)


# ── Module-level singleton ────────────────────────────────────────────────────

_store: UserModeStore | None = None


def get_user_mode_store() -> UserModeStore:
    """Return the shared UserModeStore singleton."""
    global _store
    if _store is None:
        _store = UserModeStore()
    return _store
