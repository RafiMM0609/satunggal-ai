"""Telegram Application – assembles and registers all handlers."""

from __future__ import annotations

import logging

from telegram.ext import (
    Application,
    CommandHandler,
    filters,
    MessageHandler,
)

from .config import Config
from src.handlers import (
    echo_text,
    handle_photo,
    help_command,
    ping,
    reset,
    start,
    unknown_message,
)

logger = logging.getLogger(__name__)


def build_application(config: Config) -> Application:
    """Build and return a fully wired Application instance."""
    app = (
        Application.builder()
        .token(config.bot_token)
        .build()
    )

    _register_handlers(app)

    logger.info("Application built with %d handler(s).", len(app.handlers[0]))
    return app


def _register_handlers(app: Application) -> None:
    """Register all command and message handlers."""

    # ── Command handlers ───────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(CommandHandler("ping",  ping))
    app.add_handler(CommandHandler("reset", reset))

    # ── Message handlers ───────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # ── Fallback ───────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.ALL, unknown_message))
