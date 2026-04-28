"""
REST API authentication – API key via ``X-API-Key`` header.

Phase 4 of the Autonomous Multi-Agent Workforce transformation.

Usage
-----
Set ``REST_API_KEY`` in ``.env`` (or as an environment variable) to any
non-empty string.  Every protected endpoint then requires the header::

    X-API-Key: <your-key>

If ``REST_API_KEY`` is empty (the default), authentication is **disabled** and
all requests are allowed through.  This preserves backward compatibility for
deployments that only expose the Telegram interface.

How to use in FastAPI endpoints::

    from src.interfaces.auth import require_api_key

    @app.post("/chat")
    async def chat(req: ChatRequest, _: None = Depends(require_api_key)):
        ...

The ``/health`` liveness-probe endpoint is intentionally left unprotected so
monitoring systems and container orchestrators can query it without credentials.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ── Header scheme (OpenAPI-visible) ──────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Dependency ────────────────────────────────────────────────────────────────

def require_api_key(
    api_key_header: str | None = Security(_api_key_header),
) -> None:
    """FastAPI dependency that enforces API key authentication.

    * Reads the configured key from ``Settings.rest_api_key``.
    * If the configured key is empty, the check is skipped (auth disabled).
    * Uses :func:`secrets.compare_digest` for constant-time comparison to
      prevent timing-based key enumeration attacks.
    * Raises ``HTTP 401`` when a key is required but missing or invalid.
    """
    # Import here to avoid circular imports and to honour the lru_cache
    # singleton (settings are loaded once).
    from config.settings import get_settings  # noqa: PLC0415

    configured_key = get_settings().rest_api_key
    if not configured_key:
        # Authentication disabled — no key configured.
        return

    if not api_key_header:
        logger.warning("REST API: request rejected – missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.  Provide the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to mitigate timing attacks.
    if not secrets.compare_digest(api_key_header, configured_key):
        logger.warning("REST API: request rejected – invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
