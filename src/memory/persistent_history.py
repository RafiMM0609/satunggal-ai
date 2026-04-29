"""
PersistentConversationHistory – SQLite-backed conversation store.

Phase 3 of the Autonomous Multi-Agent Workforce transformation.

Replaces the in-memory ``ConversationHistory`` as the singleton used by the
orchestrator pipeline.  It provides the **exact same public API** as
``ConversationHistory`` so all agents work unchanged:

    ``add(session_id, role, content)``
    ``clear(session_id)``
    ``get(session_id)``               → list[Message]
    ``get_as_llm_messages(session_id)`` → list[dict]
    ``__len__()``                     → total messages across all sessions

Differences from in-memory version
------------------------------------
* Messages survive bot restarts.
* ``max_messages`` is enforced **per session** at read time (newest N messages
  are returned) AND at write time (older rows are pruned so the table doesn't
  grow unboundedly).
* All operations are **synchronous** and safe to call from async code via
  ``asyncio.to_thread()`` (same pattern used by UserModeStore and RepoTracker).

Schema
------
::

    conversation_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT    NOT NULL,
        role        TEXT    NOT NULL,   -- "user" | "assistant" | "system"
        content     TEXT    NOT NULL,
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
    )

    -- Index for fast per-session lookups
    idx_conversation_history_session (session_id, id DESC)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS conversation_history (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT     NOT NULL,
    role        TEXT     NOT NULL,
    content     TEXT     NOT NULL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conv_hist_session
    ON conversation_history (session_id, id DESC);
"""


# ── Data model (mirrors history.Message so existing consumers work) ───────────

@dataclass
class Message:
    """A single conversation turn."""

    role:      str    # "user" | "assistant" | "system"
    content:   str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Main class ────────────────────────────────────────────────────────────────

class PersistentConversationHistory:
    """SQLite-backed conversation history store (per session_id).

    Drop-in replacement for the in-memory ``ConversationHistory``.  Accepts the
    same constructor arguments and exposes the same public methods.

    Args:
        max_messages: Maximum number of messages to keep **per session**.
            Older messages are pruned on every ``add()`` call.  Also used as
            the ``LIMIT`` when fetching history for LLM context.
        db_path: Path to the SQLite file.  Defaults to
            ``<project_root>/data/conversation_history.db``.
    """

    def __init__(
        self,
        max_messages: int = 30,
        db_path: Path | str | None = None,
    ) -> None:
        self._max = max_messages

        if db_path is None:
            db_path = (
                Path(__file__).resolve().parents[2] / "data" / "conversation_history.db"
            )

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ── SQLite helpers ────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a WAL-mode connection with row_factory set."""
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

    def _migrate(self) -> None:
        """Ensure the schema and index exist; idempotent."""
        with self._connect() as conn:
            conn.executescript(_DDL)
        logger.debug(
            "PersistentConversationHistory: schema ready at %s", self._db_path
        )

    # ── Writes ────────────────────────────────────────────────────────────────

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history and prune old rows.

        After inserting the new message, any rows beyond ``max_messages`` for
        this session are deleted (keeping the most recent ones).
        """
        insert_sql = """
            INSERT INTO conversation_history (session_id, role, content)
            VALUES (?, ?, ?)
        """
        prune_sql = """
            DELETE FROM conversation_history
             WHERE session_id = ?
               AND id NOT IN (
                   SELECT id FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
               )
        """
        with self._connect() as conn:
            conn.execute(insert_sql, (session_id, role, content))
            conn.execute(prune_sql, (session_id, session_id, self._max))

    def clear(self, session_id: str) -> None:
        """Delete all stored messages for a session."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_history WHERE session_id = ?",
                (session_id,),
            )
        logger.info(
            "PersistentConversationHistory: cleared session=%s", session_id
        )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> list[Message]:
        """Return ordered list of ``Message`` objects for the session.

        Returns at most ``max_messages`` messages, ordered oldest-first.
        """
        sql = """
            SELECT role, content, timestamp
              FROM (
                    SELECT id, role, content, timestamp
                      FROM conversation_history
                     WHERE session_id = ?
                     ORDER BY id DESC
                     LIMIT ?
                   ) sub
             ORDER BY id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (session_id, self._max)).fetchall()
        return [
            Message(role=row["role"], content=row["content"], timestamp=row["timestamp"])
            for row in rows
        ]

    def get_as_llm_messages(self, session_id: str) -> list[dict]:
        """Return history in OpenAI chat-completion format (list of dicts)."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get(session_id)
        ]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return total message count across all sessions."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM conversation_history"
            ).fetchone()
        return int(row["cnt"])

    def session_count(self) -> int:
        """Return the number of distinct sessions stored."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) AS cnt FROM conversation_history"
            ).fetchone()
        return int(row["cnt"])

    def clear_all(self) -> None:
        """Delete ALL messages for ALL sessions (admin / test utility)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_history")
        logger.warning("PersistentConversationHistory: ALL history cleared")
