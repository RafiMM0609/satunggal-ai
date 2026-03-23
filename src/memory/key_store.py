"""
Runtime key store — persists API key overrides to a JSON file.

Keys saved here take precedence over what is configured in .env.
The store file lives at the project root and is intentionally excluded
from version control (add ``runtime_keys.json`` to .gitignore).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root: src/memory/key_store.py → root = 2 levels up
_STORE_PATH = Path(__file__).resolve().parents[2] / "runtime_keys.json"

_KEY_OPENROUTER     = "openrouter_api_key"
_KEY_MAX_TOKENS     = "openrouter_max_tokens"


def _load() -> dict:
    if _STORE_PATH.is_file():
        try:
            return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("key_store: failed to read %s: %s", _STORE_PATH, exc)
    return {}


def _save(data: dict) -> None:
    try:
        _STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Restrict permissions: owner read/write only (Unix).
        try:
            _STORE_PATH.chmod(0o600)
        except NotImplementedError:
            pass  # Windows — chmod is a no-op, acceptable
    except Exception as exc:  # noqa: BLE001
        logger.error("key_store: failed to write %s: %s", _STORE_PATH, exc)
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def get_openrouter_key() -> Optional[str]:
    """Return the stored OpenRouter API key override, or None if not set."""
    return _load().get(_KEY_OPENROUTER) or None


def set_openrouter_key(api_key: str) -> None:
    """Persist *api_key* as the active OpenRouter override."""
    data = _load()
    data[_KEY_OPENROUTER] = api_key.strip()
    _save(data)
    logger.info("key_store: OpenRouter API key updated.")


def clear_openrouter_key() -> None:
    """Remove the stored OpenRouter API key so .env value is used again."""
    data = _load()
    data.pop(_KEY_OPENROUTER, None)
    _save(data)
    logger.info("key_store: OpenRouter API key override cleared.")


def effective_openrouter_auth_header(env_api_key: str) -> str:
    """Return the Bearer token to use, preferring the store over ``env_api_key``."""
    stored = get_openrouter_key()
    return f"Bearer {stored if stored else env_api_key}"


# ── Max Tokens override ───────────────────────────────────────────────────────

def get_openrouter_max_tokens() -> Optional[int]:
    """Return the stored max_tokens override, or None if not set."""
    value = _load().get(_KEY_MAX_TOKENS)
    return int(value) if value is not None else None


def set_openrouter_max_tokens(max_tokens: int) -> None:
    """Persist *max_tokens* as the active override."""
    data = _load()
    data[_KEY_MAX_TOKENS] = max_tokens
    _save(data)
    logger.info("key_store: OpenRouter max_tokens updated to %d.", max_tokens)


def clear_openrouter_max_tokens() -> None:
    """Remove the max_tokens override so the .env value is used again."""
    data = _load()
    data.pop(_KEY_MAX_TOKENS, None)
    _save(data)
    logger.info("key_store: OpenRouter max_tokens override cleared.")


def effective_openrouter_max_tokens(env_max_tokens: int) -> int:
    """Return the max_tokens to use, preferring the store over ``env_max_tokens``."""
    stored = get_openrouter_max_tokens()
    return stored if stored is not None else env_max_tokens
