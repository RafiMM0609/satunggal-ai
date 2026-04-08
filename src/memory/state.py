"""
Blackboard state – a single task that travels between agents.

Every request from any interface (Telegram, REST API, CLI) is wrapped
in an AgentTask so all agents speak the same language.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING    = "pending"
    ROUTING    = "routing"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class AgentTask:
    """A unit of work passed through the orchestration pipeline."""

    session_id:    str
    user_input:    str
    intent:        Optional[str]      = None
    status:        TaskStatus         = TaskStatus.PENDING
    result:        Optional[str]      = None
    metadata:      dict[str, Any]     = field(default_factory=dict)
    agent_trace:   list[str]          = field(default_factory=list)
    tool_results:  dict[str, Any]     = field(default_factory=dict)

    # Active mode for this request (set by the orchestrator from UserModeStore).
    # Defaults to "all" which preserves full-orchestrator behaviour.
    current_mode:  str                = "all"

    # Tools requested by an agent to be executed by the orchestrator AFTER
    # the agent finishes.  Agent sets this list; orchestrator drains it.
    pending_tools: list[str]          = field(default_factory=list)

    def mark_routed(self, intent: str) -> None:
        self.intent = intent
        self.status = TaskStatus.ROUTING
        self.agent_trace.append(f"router → {intent}")

    def mark_processing(self, agent_name: str) -> None:
        self.status = TaskStatus.PROCESSING
        self.agent_trace.append(f"processing by {agent_name}")

    def mark_done(self, result: str) -> None:
        self.status = TaskStatus.DONE
        self.result = result

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.metadata["error"] = reason
        self.result = None


# ── Browser Session Store ─────────────────────────────────────────────────────

_SESSION_DIR = Path("/tmp/browser_sessions")
_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_\-.]")


class BrowserSessionStore:
    """Persists Playwright storage-state (cookies/localStorage) per website.

    Session files are written to ``/tmp/browser_sessions/<safe_key>.json``
    so they survive across agent calls within the same process lifetime but
    are cleaned up when the VPS is restarted (desired for security).

    Usage::

        store = BrowserSessionStore()
        path  = store.get_session_path("https://example.com")  # may not exist yet
        store.save_session("https://example.com", state_dict)
        store.delete_session("https://example.com")
    """

    def __init__(self, session_dir: Path = _SESSION_DIR) -> None:
        self._dir = session_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_key(url: str) -> str:
        """Convert a URL into a safe filesystem filename component."""
        # strip scheme and replace unsafe chars
        key = url.replace("://", "_").replace("/", "_")
        return _SAFE_KEY_RE.sub("_", key)[:120]

    def _path_for(self, url: str) -> Path:
        return self._dir / f"{self._safe_key(url)}.json"

    # ── public API ────────────────────────────────────────────────────────────

    def get_session_path(self, url: str) -> Path:
        """Return the Path where the session for *url* would be stored.

        The file may or may not exist; callers should check with
        ``path.exists()`` before using it.
        """
        return self._path_for(url)

    def has_session(self, url: str) -> bool:
        """Return True if a saved session exists for *url*."""
        return self._path_for(url).exists()

    def save_session(self, url: str, state: dict) -> Path:
        """Persist Playwright storage state for *url*.

        Args:
            url:   The base URL (e.g. ``"https://example.com"``).
            state: The dict returned by ``page.context.storage_state()``.

        Returns:
            The path where the file was written.
        """
        path = self._path_for(url)
        try:
            path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("BrowserSessionStore: session saved for %s → %s", url, path)
        except OSError as exc:
            logger.warning("BrowserSessionStore: failed to save session for %s: %s", url, exc)
        return path

    def load_session(self, url: str) -> Optional[dict]:
        """Load a previously saved session for *url*.

        Returns the state dict, or ``None`` if no session exists or reading
        fails.
        """
        path = self._path_for(url)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("BrowserSessionStore: failed to load session for %s: %s", url, exc)
            return None

    def delete_session(self, url: str) -> None:
        """Remove the stored session for *url* (e.g. after a logout)."""
        path = self._path_for(url)
        if path.exists():
            try:
                path.unlink()
                logger.info("BrowserSessionStore: session deleted for %s", url)
            except OSError as exc:
                logger.warning("BrowserSessionStore: failed to delete session for %s: %s", url, exc)

    def list_sessions(self) -> list[str]:
        """Return file paths of all stored sessions as strings."""
        return [str(p) for p in self._dir.glob("*.json")]
