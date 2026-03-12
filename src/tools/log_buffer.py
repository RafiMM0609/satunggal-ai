"""
In-memory ring-buffer logging handler.

A single global LogBuffer instance captures every log record emitted by the
application.  The log_viewer_agent reads from this buffer so users can
inspect recent bot activity without needing access to the host filesystem.

Usage
-----
    from src.tools.log_buffer import get_log_buffer, LogBufferHandler

    # Register once at startup:
    import logging
    logging.getLogger().addHandler(LogBufferHandler())

    # Read later:
    buf = get_log_buffer()
    lines = buf.tail(20)          # last 20 lines
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Sequence

_DEFAULT_CAPACITY = 1_000  # maximum number of lines kept in memory

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


class LogBuffer:
    """Thread-safe ring buffer that stores formatted log lines."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._buf: deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write side (called by LogBufferHandler)
    # ------------------------------------------------------------------

    def append(self, line: str) -> None:
        with self._lock:
            self._buf.append(line)

    # ------------------------------------------------------------------
    # Read side (called by LogViewerAgent)
    # ------------------------------------------------------------------

    def tail(self, n: int = 10) -> Sequence[str]:
        """Return the last *n* log lines (oldest first)."""
        n = max(1, n)
        with self._lock:
            lines = list(self._buf)
        return lines[-n:]

    def all(self) -> Sequence[str]:
        """Return all buffered log lines."""
        with self._lock:
            return list(self._buf)


class LogBufferHandler(logging.Handler):
    """Logging handler that writes formatted records to the global LogBuffer."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(_formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            _global_buffer.append(line)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ── Singleton ─────────────────────────────────────────────────────────────────

_global_buffer = LogBuffer()


def get_log_buffer() -> LogBuffer:
    """Return the application-wide LogBuffer singleton."""
    return _global_buffer
