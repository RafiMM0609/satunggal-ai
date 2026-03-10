"""
Conversation history – per-session message store.

Currently in-memory (deque with max size).
Swap _store for a Redis/DB backend in production by replacing
ConversationHistory._store with the appropriate async client.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    role:      str   # "user" | "assistant" | "system"
    content:   str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ConversationHistory:
    """Thread-safe in-memory conversation store (per session_id)."""

    def __init__(self, max_messages: int = 20) -> None:
        self._max   = max_messages
        self._store: dict[str, deque[Message]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    # ── Writes ────────────────────────────────────────────────────────────────

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        self._store[session_id].append(Message(role=role, content=content))

    def clear(self, session_id: str) -> None:
        """Wipe the entire history for a session."""
        self._store.pop(session_id, None)

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> list[Message]:
        """Return ordered list of Message objects for a session."""
        return list(self._store[session_id])

    def get_as_llm_messages(self, session_id: str) -> list[dict]:
        """Return history in OpenAI chat-completion format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get(session_id)
        ]

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())
