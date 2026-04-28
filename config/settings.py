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

    # ── Ollama LLM ────────────────────────────────────────────────────────────
    ollama_host:    str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    ollama_api_key: str = Field("",                      alias="OLLAMA_API_KEY")
    ollama_model:   str = Field("llama3.2",              alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(120,                     alias="OLLAMA_TIMEOUT")

    # ── Tavily Search ─────────────────────────────────────────────────────────
    tavily_api_key:      str  = Field("",       alias="TAVILY_API_KEY")
    tavily_search_depth: str  = Field("advanced", alias="TAVILY_SEARCH_DEPTH")
    tavily_max_results:  int  = Field(5,          alias="TAVILY_MAX_RESULTS")
    tavily_time_range:   str  = Field("month",     alias="TAVILY_TIME_RANGE")

    # ── REST API ──────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000,      alias="API_PORT")
    # Optional API key for the REST API.  Set REST_API_KEY in .env to enable.
    # When non-empty, all protected endpoints require the header:
    #   X-API-Key: <value>
    # Leave empty (default) to disable authentication (development / Telegram-only deployments).
    rest_api_key: str = Field("", alias="REST_API_KEY")

    # ── Developer / Sandbox ───────────────────────────────────────────────────
    # Base directory where repos are cloned (absolute path string).
    sandbox_repos_dir:    str = Field("~/sandbox_repos", alias="SANDBOX_REPOS_DIR")
    # Docker image used when no Dockerfile is present in the repo.
    sandbox_python_image: str = Field("python:3.11-slim", alias="SANDBOX_PYTHON_IMAGE")
    # Max seconds for a single docker compose up --build (default: 5 min).
    sandbox_timeout:      int = Field(300, alias="SANDBOX_TIMEOUT")
    # Max retry iterations when Docker build/run fails.
    sandbox_max_retries:  int = Field(3,   alias="SANDBOX_MAX_RETRIES")

    # ── Git Credentials ───────────────────────────────────────────────────────
    # Personal Access Token – used to authenticate git clone/push for private repos.
    # Format: classic PAT with repo scope, or fine-grained PAT with Contents: RW.
    # Leave empty to rely on pre-configured SSH keys (no PAT auth).
    github_pat:       str = Field("", alias="GITHUB_PAT")
    # GitLab Personal Access Token – used to authenticate git clone/push for
    # private GitLab repositories.  The token should have at least `read_repository`
    # scope (and `write_repository` if pushing is needed).
    # Leave empty to rely on pre-configured SSH keys (no PAT auth).
    gitlab_pat:       str = Field("", alias="GITLAB_PAT")
    # Comma-separated list of ADDITIONAL hostnames that host self-managed GitLab
    # instances.  'gitlab.com' is always treated as GitLab regardless of this list.
    # Example: GITLAB_HOSTS=repo.jesica.id,git.mycompany.io
    gitlab_hosts:     str = Field("", alias="GITLAB_HOSTS")
    # Identity written to git config inside each sandbox repo before committing.
    git_user_name:    str = Field("AdvanceAI Bot", alias="GIT_USER_NAME")
    git_user_email:   str = Field("bot@advanceai.local", alias="GIT_USER_EMAIL")

    # ── Admin ─────────────────────────────────────────────────────────────────
    # Telegram user_id yang boleh menjalankan /deploy. Kosongkan untuk disable.
    admin_user_id: int = Field(0, alias="ADMIN_USER_ID")

    # ── Proactive Mode – Fase 4 ───────────────────────────────────────────────
    # Daily Briefing: ResearcherAgent mengirim ringkasan berita/topik secara terjadwal.
    #
    # PROACTIVE_BRIEFING_ENABLED   – aktifkan ("true") atau nonaktifkan ("false").
    # PROACTIVE_BRIEFING_CHAT_ID   – Telegram chat_id tujuan (default ke ADMIN_USER_ID).
    # PROACTIVE_BRIEFING_TIME      – waktu kirim dalam format "HH:MM" zona WIB (UTC+7).
    # PROACTIVE_BRIEFING_TOPICS    – topik yang diriset, dipisah koma.
    #                                Contoh: "AI terbaru, berita teknologi, startup Indonesia"
    # PROACTIVE_BRIEFING_LANGUAGE  – bahasa laporan: "id" (Indonesia) atau "en" (English).
    proactive_briefing_enabled:  bool = Field(False,  alias="PROACTIVE_BRIEFING_ENABLED")
    proactive_briefing_chat_id:  str  = Field("",     alias="PROACTIVE_BRIEFING_CHAT_ID")
    proactive_briefing_time:     str  = Field("07:00",alias="PROACTIVE_BRIEFING_TIME")
    proactive_briefing_topics:   str  = Field(
        "AI terbaru, berita teknologi, startup Indonesia",
        alias="PROACTIVE_BRIEFING_TOPICS",
    )
    proactive_briefing_language: str  = Field("id",   alias="PROACTIVE_BRIEFING_LANGUAGE")

    # Repo Watcher: Memantau repository GitHub/GitLab, auto code-review jika ada commit baru.
    #
    # PROACTIVE_REPO_WATCHER_ENABLED   – aktifkan ("true") atau nonaktifkan ("false").
    # PROACTIVE_REPO_WATCHER_CHAT_ID   – Telegram chat_id tujuan laporan.
    # PROACTIVE_REPO_WATCHER_REPOS     – daftar URL repo dipisah koma untuk dipantau.
    #                                    Contoh: "https://github.com/org/repo1,https://..."
    # PROACTIVE_REPO_WATCHER_INTERVAL  – interval polling dalam menit (default: 60).
    proactive_repo_watcher_enabled:  bool = Field(False, alias="PROACTIVE_REPO_WATCHER_ENABLED")
    proactive_repo_watcher_chat_id:  str  = Field("",    alias="PROACTIVE_REPO_WATCHER_CHAT_ID")
    proactive_repo_watcher_repos:    str  = Field("",    alias="PROACTIVE_REPO_WATCHER_REPOS")
    proactive_repo_watcher_interval: int  = Field(60,    alias="PROACTIVE_REPO_WATCHER_INTERVAL")

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
