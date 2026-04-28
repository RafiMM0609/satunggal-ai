"""
Entry point – starts the Telegram webhook bot.

The REST API (FastAPI) can be started separately:
    uvicorn src.interfaces.rest_api:app --host 0.0.0.0 --port 8000

Or run both together (Telegram webhook + REST API) via a process manager
such as supervisord or docker-compose.
"""

import logging
import sys

from src.interfaces.config import Config
from src.interfaces.telegram_bot import build_application
from src.interfaces.webhook import run_webhook
from src.tools.log_buffer import LogBufferHandler


def setup_logging() -> None:
    log_buffer_handler = LogBufferHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), log_buffer_handler],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("=== AdvanceAI — starting ===")

    try:
        config = Config.from_env()
    except (KeyError, ValueError) as exc:
        logger.critical("Konfigurasi tidak valid: %s", exc)
        sys.exit(1)

    app = build_application(config)

    logger.info("Webhook URL  : %s", config.listen_url)
    logger.info("Listen       : %s:%d", config.host, config.port)

    run_webhook(app, config)


if __name__ == "__main__":
    main()
