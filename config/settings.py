"""
Centralised application settings – loaded once from the project-root .env file.

All modules should import Settings / get_settings from here rather than
reading os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve once: config/settings.py → root = 1 level up
_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"

# Pre-populate os.environ so third-party libs that read os.environ directly also work
load_dotenv(dotenv_path=_ROOT_ENV)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ROOT_ENV),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    bot_token:    str = Field(..., alias="BOT_TOKEN")
    webhook_url:  str = Field(..., alias="WEBHOOK_URL")
    webhook_path: str = Field("/webhook", alias="WEBHOOK_PATH")
    host:         str = Field("0.0.0.0",  alias="HOST")
    port:         int = Field(8443,        alias="PORT")
    secret_token: str = Field("",          alias="SECRET_TOKEN")

    # ── OpenRouter LLM ────────────────────────────────────────────────────────
    openrouter_api_key:   str = Field(...,  alias="OPENROUTER_API_KEY")
    openrouter_base_url:  str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_model:     str = Field("openai/gpt-4o-mini",           alias="OPENROUTER_MODEL")
    openrouter_timeout:   int = Field(30,                             alias="OPENROUTER_TIMEOUT")
    openrouter_max_tokens: int = Field(8192,                          alias="OPENROUTER_MAX_TOKENS")

    # ── Tavily Search ─────────────────────────────────────────────────────────
    tavily_api_key:      str  = Field("",       alias="TAVILY_API_KEY")
    tavily_search_depth: str  = Field("advanced", alias="TAVILY_SEARCH_DEPTH")
    tavily_max_results:  int  = Field(5,          alias="TAVILY_MAX_RESULTS")

    # ── REST API ──────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000,      alias="API_PORT")

    # ── App Metadata ──────────────────────────────────────────────────────────
    app_name: str = Field("AdvanceAI", alias="APP_NAME")
    app_url:  str = Field("https://example.com", alias="APP_URL")

    @property
    def listen_url(self) -> str:
        return f"{self.webhook_url.rstrip('/')}{self.webhook_path}"

    @property
    def openrouter_headers(self) -> dict[str, str]:
        return {
            "Authorization":   f"Bearer {self.openrouter_api_key}",
            "X-Title":         self.app_name,
            "X-Referer":       self.app_url,
            "Content-Type":    "application/json",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
