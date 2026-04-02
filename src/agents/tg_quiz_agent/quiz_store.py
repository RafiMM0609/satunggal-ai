"""
TgQuizStore – SQLite-backed persistence for Telegram quiz sessions.

Stores generated quiz questions so that sessions survive bot restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("/tmp/tg_quiz_sessions.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tg_quiz_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    quiz_title       TEXT    NOT NULL,
    questions_json   TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    total_questions  INTEGER NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'created'
);
CREATE INDEX IF NOT EXISTS idx_tg_quiz_session ON tg_quiz_sessions (session_id);
"""


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_CREATE_TABLE)
        conn.commit()
        yield conn
    finally:
        conn.close()


class TgQuizStore:
    """Persist and retrieve Telegram quiz sessions."""

    def save_session(
        self,
        session_id: str,
        quiz_title: str,
        questions: list[dict],
    ) -> int:
        """
        Persist a quiz session.

        Returns:
            The row ID of the inserted record.
        """
        questions_json = json.dumps(questions, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()

        with _get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tg_quiz_sessions
                    (session_id, quiz_title, questions_json, created_at, total_questions)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, quiz_title, questions_json, created_at, len(questions)),
            )
            conn.commit()
            row_id = cursor.lastrowid

        logger.info(
            "TgQuizStore: saved session=%s title=%r questions=%d row_id=%d",
            session_id, quiz_title, len(questions), row_id,
        )
        return row_id

    def get_latest_session(self, session_id: str) -> Optional[dict]:
        """
        Retrieve the most recent quiz session for *session_id*.

        Returns a dict with keys: id, session_id, quiz_title, questions,
        created_at, total_questions, status – or None if not found.
        """
        with _get_conn() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, quiz_title, questions_json,
                       created_at, total_questions, status
                FROM tg_quiz_sessions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "id":               row["id"],
            "session_id":       row["session_id"],
            "quiz_title":       row["quiz_title"],
            "questions":        json.loads(row["questions_json"]),
            "created_at":       row["created_at"],
            "total_questions":  row["total_questions"],
            "status":           row["status"],
        }

    def update_status(self, row_id: int, status: str) -> None:
        """Update the status column for a quiz row."""
        with _get_conn() as conn:
            conn.execute(
                "UPDATE tg_quiz_sessions SET status = ? WHERE id = ?",
                (status, row_id),
            )
            conn.commit()
        logger.debug("TgQuizStore: row_id=%d status → %s", row_id, status)
