"""Konfigurasi dan eksekusi webhook server."""

from __future__ import annotations

import logging

from telegram.ext import Application

from .config import Config

logger = logging.getLogger(__name__)


def run_webhook(app: Application, config: Config) -> None:
    """
    Jalankan built-in webhook server milik python-telegram-bot.

    Server ini ringan karena hanya memproses request HTTP yang masuk
    dari Telegram (tidak ada long-polling / koneksi persisten).

    Port yang didukung Telegram untuk webhook: 443, 80, 88, 8443.
    """
    logger.info(
        "Menjalankan webhook server di %s:%d%s",
        config.host,
        config.port,
        config.webhook_path,
    )

    app.run_webhook(
        listen=config.host,
        port=config.port,
        url_path=config.webhook_path,
        webhook_url=config.listen_url,
        secret_token=config.secret_token or None,
        drop_pending_updates=True,
    )
