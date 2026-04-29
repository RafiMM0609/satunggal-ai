"""
WebTaskStore – SQLite-backed store for scheduled web automation tasks.

Scheduled tasks are persisted so they survive process restarts within the
same environment.  The ``run_scheduler()`` coroutine polls the store for due
tasks and dispatches them via a caller-supplied ``dispatch`` callable.

Usage example (in the main event loop start-up)::

    from src.memory.web_task_store import get_web_task_store, run_scheduler

    async def _dispatch(session_id: str, user_input: str) -> str:
        # run web automation and return reply
        ...

    asyncio.create_task(run_scheduler(_dispatch))
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("/tmp/web_tasks.db")


class WebTaskStore:
    """SQLite-backed store for scheduled web automation task definitions."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS web_tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                label       TEXT    NOT NULL,
                user_input  TEXT    NOT NULL,
                interval_s  INTEGER NOT NULL,
                next_run    REAL    NOT NULL,
                last_run    REAL,
                active      INTEGER NOT NULL DEFAULT 1,
                created_at  REAL    NOT NULL
            )"""
        )
        self._conn.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_task(
        self,
        session_id: str,
        label: str,
        user_input: str,
        interval_s: int,
    ) -> int:
        """Register a new scheduled task. Returns the newly created task ID."""
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO web_tasks
               (session_id, label, user_input, interval_s, next_run, active, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (session_id, label, user_input, interval_s, now + interval_s, now),
        )
        self._conn.commit()
        task_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.info(
            "WebTaskStore: added task id=%d label=%r interval=%ds session=%s",
            task_id, label, interval_s, session_id,
        )
        return task_id

    def list_tasks(self, session_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Return all active tasks, optionally filtered by *session_id*."""
        if session_id:
            rows = self._conn.execute(
                "SELECT * FROM web_tasks WHERE session_id=? AND active=1 ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM web_tasks WHERE active=1 ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_due_tasks(self) -> list[dict[str, Any]]:
        """Return active tasks whose ``next_run`` timestamp has passed."""
        now = time.time()
        rows = self._conn.execute(
            "SELECT * FROM web_tasks WHERE active=1 AND next_run <= ?",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_ran(self, task_id: int) -> None:
        """Advance ``next_run`` by ``interval_s`` and record ``last_run``."""
        now = time.time()
        self._conn.execute(
            """UPDATE web_tasks
               SET last_run=?, next_run=next_run+interval_s
               WHERE id=?""",
            (now, task_id),
        )
        self._conn.commit()

    def cancel_task(self, task_id: int) -> None:
        """Deactivate a scheduled task by ID."""
        self._conn.execute(
            "UPDATE web_tasks SET active=0 WHERE id=?",
            (task_id,),
        )
        self._conn.commit()
        logger.info("WebTaskStore: cancelled task id=%d", task_id)

    def cancel_by_label(self, session_id: str, label: str) -> int:
        """Deactivate all tasks matching *label* in *session_id*.

        Returns the number of tasks that were deactivated.
        """
        cur = self._conn.execute(
            "UPDATE web_tasks SET active=0 WHERE session_id=? AND label=?",
            (session_id, label),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ── Singleton ─────────────────────────────────────────────────────────────────

_store_singleton: Optional[WebTaskStore] = None


def get_web_task_store() -> WebTaskStore:
    """Return the process-wide singleton ``WebTaskStore``."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = WebTaskStore()
    return _store_singleton


# ── Background scheduler ──────────────────────────────────────────────────────

_scheduler_running = False


async def run_scheduler(
    dispatch: Callable[[str, str], Coroutine[Any, Any, str]],
    poll_interval_s: float = 30.0,
) -> None:
    """Long-running coroutine that polls for due tasks and dispatches them.

    Args:
        dispatch:        Async callable ``(session_id, user_input) → reply``
                         that runs a web automation task and returns the result.
        poll_interval_s: How often to check for due tasks (default 30 s).

    This coroutine is designed to run as a background ``asyncio.Task``::

        asyncio.create_task(run_scheduler(_dispatch_fn))
    """
    global _scheduler_running
    if _scheduler_running:
        logger.warning("web_task_store: scheduler already running – skipping duplicate start")
        return
    _scheduler_running = True
    store = get_web_task_store()
    logger.info(
        "web_task_store: scheduler started (poll_interval=%.0fs)", poll_interval_s
    )
    try:
        while True:
            try:
                for task in store.get_due_tasks():
                    logger.info(
                        "web_task_store: running task id=%d label=%r session=%s",
                        task["id"], task["label"], task["session_id"],
                    )
                    try:
                        await dispatch(task["session_id"], task["user_input"])
                    except Exception as exc:
                        logger.warning(
                            "web_task_store: task id=%d dispatch failed: %s",
                            task["id"], exc,
                        )
                    finally:
                        # Advance next_run regardless of success so failed tasks
                        # don't spin in tight retry loops.
                        store.mark_ran(task["id"])
            except Exception as exc:
                logger.warning("web_task_store: scheduler poll error: %s", exc)
            await asyncio.sleep(poll_interval_s)
    finally:
        _scheduler_running = False
