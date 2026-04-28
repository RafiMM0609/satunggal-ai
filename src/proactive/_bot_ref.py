"""
Shared Telegram Bot reference untuk proactive jobs.

Menyimpan satu referensi Bot yang diinjeksikan saat startup sehingga
daily_briefing.py dan repo_watcher.py bisa mengaksesnya tanpa circular import.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Bot

_bot: Optional["Bot"] = None


def set_bot(bot: "Bot") -> None:
    """Injeksikan Bot instance. Dipanggil sekali saat startup."""
    global _bot
    _bot = bot


def get_bot() -> Optional["Bot"]:
    """Kembalikan Bot instance yang tersimpan, atau None jika belum diset."""
    return _bot
