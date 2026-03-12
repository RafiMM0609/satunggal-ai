"""
RepoTracker – persistent SQLite store for cloned repositories.

Schema:
    local_repos(id, repo_name, repo_url, local_path,
                last_task_status, last_commit_hash, created_at)

All public methods are synchronous but run inside a thread-pool executor
so they can be called safely from async code via asyncio.to_thread().
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RepoRecord:
    """Immutable snapshot of a local_repos row."""
    id:               int
    repo_name:        str
    repo_url:         str
    local_path:       str
    last_task_status: str
    last_commit_hash: str
    created_at:       str


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS local_repos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name        TEXT    NOT NULL,
    repo_url         TEXT    NOT NULL UNIQUE,
    local_path       TEXT    NOT NULL,
    last_task_status TEXT    NOT NULL DEFAULT 'pending',
    last_commit_hash TEXT    NOT NULL DEFAULT '',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class RepoTracker:
    """
    Lightweight SQLite repository for tracking cloned repos.

    Usage:
        tracker = RepoTracker()
        await asyncio.to_thread(tracker.upsert, repo_name, repo_url, local_path)
        records = await asyncio.to_thread(tracker.list_all)
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            # Default: project root / data / repos.db
            db_path = Path(__file__).resolve().parents[2] / "data" / "repos.db"

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection with WAL mode and row_factory set."""
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
        logger.debug("RepoTracker: schema ready at %s", self._db_path)

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert(
        self,
        repo_name:   str,
        repo_url:    str,
        local_path:  str,
        *,
        status:      str = "pending",
        commit_hash: str = "",
    ) -> int:
        """
        Insert or update a repo record.

        Returns the row id.
        """
        sql = """
            INSERT INTO local_repos (repo_name, repo_url, local_path,
                                     last_task_status, last_commit_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo_url) DO UPDATE SET
                repo_name        = excluded.repo_name,
                local_path       = excluded.local_path,
                last_task_status = excluded.last_task_status,
                last_commit_hash = excluded.last_commit_hash
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (repo_name, repo_url, local_path, status, commit_hash))
            # If it was an UPDATE the lastrowid is 0 – fetch the real id.
            if cur.lastrowid:
                return cur.lastrowid
            row = conn.execute(
                "SELECT id FROM local_repos WHERE repo_url = ?", (repo_url,)
            ).fetchone()
            return int(row["id"])

    def delete_by_url(self, repo_url: str) -> None:
        """Delete the record for the given URL if it exists."""
        with self._connect() as conn:
            conn.execute("DELETE FROM local_repos WHERE repo_url = ?", (repo_url,))
        logger.info("RepoTracker: deleted record for %s", repo_url)

    def update_status(
        self,
        repo_url:    str,
        status:      str,
        commit_hash: str = "",
    ) -> None:
        """Update only the status and commit hash for an existing record."""
        sql = """
            UPDATE local_repos
               SET last_task_status = ?,
                   last_commit_hash = ?
             WHERE repo_url = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (status, commit_hash, repo_url))
        logger.info("RepoTracker: updated status=%s for %s", status, repo_url)

    # ── Read operations ───────────────────────────────────────────────────────

    def list_all(self) -> list[RepoRecord]:
        """Return all tracked repos sorted by created_at DESC."""
        sql = """
            SELECT id, repo_name, repo_url, local_path,
                   last_task_status, last_commit_hash, created_at
              FROM local_repos
             ORDER BY created_at DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [RepoRecord(**dict(row)) for row in rows]

    def find_by_url(self, repo_url: str) -> Optional[RepoRecord]:
        """Return the record for the given URL, or None if not tracked."""
        sql = """
            SELECT id, repo_name, repo_url, local_path,
                   last_task_status, last_commit_hash, created_at
              FROM local_repos
             WHERE repo_url = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (repo_url,)).fetchone()
        return RepoRecord(**dict(row)) if row else None

    def find_by_name(self, repo_name: str) -> list[RepoRecord]:
        """Return all records whose repo_name contains the given substring (case-insensitive)."""
        sql = """
            SELECT id, repo_name, repo_url, local_path,
                   last_task_status, last_commit_hash, created_at
              FROM local_repos
             WHERE LOWER(repo_name) LIKE LOWER(?)
             ORDER BY created_at DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{repo_name}%",)).fetchall()
        return [RepoRecord(**dict(row)) for row in rows]
