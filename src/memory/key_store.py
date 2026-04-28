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
_KEY_MODEL_NAME     = "openrouter_model_name"

_KEY_ACTIVE_PROVIDER = "active_provider"
_KEY_OLLAMA_KEY      = "ollama_api_key"
_KEY_OLLAMA_HOST     = "ollama_host"
_KEY_OLLAMA_MODEL    = "ollama_model_name"

_KEY_GITHUB_PAT      = "github_pat"
_KEY_GITLAB_PAT      = "gitlab_pat"

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OLLAMA     = "ollama"
_VALID_PROVIDERS    = {PROVIDER_OPENROUTER, PROVIDER_OLLAMA}


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


# ── Model Name override ───────────────────────────────────────────────────────

def get_openrouter_model() -> Optional[str]:
    """Return the stored model name override, or None if not set."""
    return _load().get(_KEY_MODEL_NAME) or None


def set_openrouter_model(model_name: str) -> None:
    """Persist *model_name* as the active OpenRouter model override."""
    data = _load()
    data[_KEY_MODEL_NAME] = model_name.strip()
    _save(data)
    logger.info("key_store: OpenRouter model name updated to %s.", model_name)


def clear_openrouter_model() -> None:
    """Remove the model name override so the .env value is used again."""
    data = _load()
    data.pop(_KEY_MODEL_NAME, None)
    _save(data)
    logger.info("key_store: OpenRouter model name override cleared.")


def effective_openrouter_model(env_model: str) -> str:
    """Return the model name to use, preferring the store over ``env_model``."""
    stored = get_openrouter_model()
    return stored if stored else env_model


# ── Active Provider ───────────────────────────────────────────────────────────

def get_active_provider() -> str:
    """Return the active LLM provider ('openrouter' or 'ollama'). Defaults to 'openrouter'."""
    return _load().get(_KEY_ACTIVE_PROVIDER) or PROVIDER_OPENROUTER


def set_active_provider(provider: str) -> None:
    """Persist *provider* as the active LLM provider."""
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Valid options: {_VALID_PROVIDERS}")
    data = _load()
    data[_KEY_ACTIVE_PROVIDER] = provider
    _save(data)
    logger.info("key_store: active LLM provider set to %s.", provider)


def clear_active_provider() -> None:
    """Remove the active provider override, reverting to default (openrouter)."""
    data = _load()
    data.pop(_KEY_ACTIVE_PROVIDER, None)
    _save(data)
    logger.info("key_store: active LLM provider override cleared.")


# ── Ollama overrides ──────────────────────────────────────────────────────────

def get_ollama_key() -> Optional[str]:
    """Return the stored Ollama API key override, or None if not set."""
    return _load().get(_KEY_OLLAMA_KEY) or None


def set_ollama_key(api_key: str) -> None:
    """Persist *api_key* as the active Ollama API key override."""
    data = _load()
    data[_KEY_OLLAMA_KEY] = api_key.strip()
    _save(data)
    logger.info("key_store: Ollama API key updated.")


def clear_ollama_key() -> None:
    """Remove the stored Ollama API key so .env value is used again."""
    data = _load()
    data.pop(_KEY_OLLAMA_KEY, None)
    _save(data)
    logger.info("key_store: Ollama API key override cleared.")


def get_ollama_host() -> Optional[str]:
    """Return the stored Ollama host override, or None if not set."""
    return _load().get(_KEY_OLLAMA_HOST) or None


def set_ollama_host(host: str) -> None:
    """Persist *host* as the active Ollama host override."""
    data = _load()
    data[_KEY_OLLAMA_HOST] = host.strip()
    _save(data)
    logger.info("key_store: Ollama host updated to %s.", host)


def clear_ollama_host() -> None:
    """Remove the stored Ollama host override so .env value is used again."""
    data = _load()
    data.pop(_KEY_OLLAMA_HOST, None)
    _save(data)
    logger.info("key_store: Ollama host override cleared.")


def effective_ollama_host(env_host: str) -> str:
    """Return the Ollama host to use, preferring the store over ``env_host``."""
    stored = get_ollama_host()
    return stored if stored else env_host


def get_ollama_model() -> Optional[str]:
    """Return the stored Ollama model name override, or None if not set."""
    return _load().get(_KEY_OLLAMA_MODEL) or None


def set_ollama_model(model_name: str) -> None:
    """Persist *model_name* as the active Ollama model override."""
    data = _load()
    data[_KEY_OLLAMA_MODEL] = model_name.strip()
    _save(data)
    logger.info("key_store: Ollama model name updated to %s.", model_name)


def clear_ollama_model() -> None:
    """Remove the Ollama model name override so the .env value is used again."""
    data = _load()
    data.pop(_KEY_OLLAMA_MODEL, None)
    _save(data)
    logger.info("key_store: Ollama model name override cleared.")


def effective_ollama_auth_header(env_api_key: str) -> Optional[str]:
    """Return the Ollama Authorization header value, or None if no key is configured."""
    stored = get_ollama_key()
    key = stored if stored else env_api_key
    return f"Bearer {key}" if key and key.strip() else None


def effective_ollama_model(env_model: str) -> str:
    """Return the Ollama model name to use, preferring the store over ``env_model``."""
    stored = get_ollama_model()
    return stored if stored else env_model


# ── GitHub / GitLab PAT overrides ────────────────────────────────────────────

def get_github_pat() -> Optional[str]:
    """Return the stored GitHub PAT override, or None if not set."""
    return _load().get(_KEY_GITHUB_PAT) or None


def set_github_pat(pat: str) -> None:
    """Persist *pat* as the active GitHub Personal Access Token override."""
    data = _load()
    data[_KEY_GITHUB_PAT] = pat.strip()
    _save(data)
    logger.info("key_store: GitHub PAT updated.")


def clear_github_pat() -> None:
    """Remove the stored GitHub PAT so the .env value is used again."""
    data = _load()
    data.pop(_KEY_GITHUB_PAT, None)
    _save(data)
    logger.info("key_store: GitHub PAT override cleared.")


def effective_github_pat(env_pat: str) -> str:
    """Return the GitHub PAT to use, preferring the store over ``env_pat``."""
    stored = get_github_pat()
    return stored if stored else env_pat


def get_gitlab_pat() -> Optional[str]:
    """Return the stored GitLab PAT override, or None if not set."""
    return _load().get(_KEY_GITLAB_PAT) or None


def set_gitlab_pat(pat: str) -> None:
    """Persist *pat* as the active GitLab Personal Access Token override."""
    data = _load()
    data[_KEY_GITLAB_PAT] = pat.strip()
    _save(data)
    logger.info("key_store: GitLab PAT updated.")


def clear_gitlab_pat() -> None:
    """Remove the stored GitLab PAT so the .env value is used again."""
    data = _load()
    data.pop(_KEY_GITLAB_PAT, None)
    _save(data)
    logger.info("key_store: GitLab PAT override cleared.")


def effective_gitlab_pat(env_pat: str) -> str:
    """Return the GitLab PAT to use, preferring the store over ``env_pat``."""
    stored = get_gitlab_pat()
    return stored if stored else env_pat


# ── Daily Briefing overrides ─────────────────────────────────────────────────

_KEY_BRIEFING_ENABLED  = "briefing_enabled"
_KEY_BRIEFING_TIME     = "briefing_time"
_KEY_BRIEFING_TOPICS   = "briefing_topics"
_KEY_BRIEFING_LANGUAGE = "briefing_language"
_KEY_BRIEFING_CHAT_ID  = "briefing_chat_id"


def get_briefing_enabled() -> Optional[bool]:
    """Return the stored briefing enabled override, or None if not set."""
    val = _load().get(_KEY_BRIEFING_ENABLED)
    if val is None:
        return None
    return bool(val)


def set_briefing_enabled(enabled: bool) -> None:
    """Persist whether daily briefing is enabled."""
    data = _load()
    data[_KEY_BRIEFING_ENABLED] = enabled
    _save(data)
    logger.info("key_store: briefing_enabled set to %s.", enabled)


def get_briefing_time() -> Optional[str]:
    """Return the stored briefing time override (HH:MM WIB), or None if not set."""
    return _load().get(_KEY_BRIEFING_TIME) or None


def set_briefing_time(time_str: str) -> None:
    """Persist briefing time as HH:MM (WIB)."""
    data = _load()
    data[_KEY_BRIEFING_TIME] = time_str.strip()
    _save(data)
    logger.info("key_store: briefing_time set to %s.", time_str)


def get_briefing_topics() -> Optional[str]:
    """Return the stored briefing topics override (comma-separated), or None if not set."""
    return _load().get(_KEY_BRIEFING_TOPICS) or None


def set_briefing_topics(topics: str) -> None:
    """Persist briefing topics as a comma-separated string."""
    data = _load()
    data[_KEY_BRIEFING_TOPICS] = topics.strip()
    _save(data)
    logger.info("key_store: briefing_topics updated.")


def get_briefing_language() -> Optional[str]:
    """Return the stored briefing language override ('id' or 'en'), or None if not set."""
    return _load().get(_KEY_BRIEFING_LANGUAGE) or None


def set_briefing_language(lang: str) -> None:
    """Persist briefing language ('id' or 'en')."""
    data = _load()
    data[_KEY_BRIEFING_LANGUAGE] = lang.strip().lower()
    _save(data)
    logger.info("key_store: briefing_language set to %s.", lang)


def get_briefing_chat_id() -> Optional[str]:
    """Return the stored briefing chat_id override, or None if not set."""
    return _load().get(_KEY_BRIEFING_CHAT_ID) or None


def set_briefing_chat_id(chat_id: str) -> None:
    """Persist briefing chat_id override."""
    data = _load()
    data[_KEY_BRIEFING_CHAT_ID] = chat_id.strip()
    _save(data)
    logger.info("key_store: briefing_chat_id set to %s.", chat_id)


def clear_briefing_overrides() -> None:
    """Remove all briefing overrides so .env values are used again."""
    data = _load()
    for key in (
        _KEY_BRIEFING_ENABLED,
        _KEY_BRIEFING_TIME,
        _KEY_BRIEFING_TOPICS,
        _KEY_BRIEFING_LANGUAGE,
        _KEY_BRIEFING_CHAT_ID,
    ):
        data.pop(key, None)
    _save(data)
    logger.info("key_store: all briefing overrides cleared.")
