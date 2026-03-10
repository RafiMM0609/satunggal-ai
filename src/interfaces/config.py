from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Konfigurasi aplikasi yang di-load dari environment variables."""

    bot_token: str
    webhook_url: str
    webhook_path: str
    host: str
    port: int
    secret_token: str

    # URL lengkap webhook yang dikirim ke Telegram
    listen_url: str = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True → gunakan object.__setattr__ untuk computed field
        object.__setattr__(
            self,
            "listen_url",
            f"{self.webhook_url.rstrip('/')}{self.webhook_path}",
        )

    # ── validasi ──────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validasi nilai-nilai konfigurasi kritis."""
        if not self.bot_token:
            raise ValueError("BOT_TOKEN wajib diisi.")
        if not self.webhook_url.startswith("https://"):
            raise ValueError("WEBHOOK_URL harus menggunakan HTTPS.")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"PORT tidak valid: {self.port}")

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> Config:
        """Buat instance Config dari environment variables."""
        config = cls(
            bot_token=os.environ["BOT_TOKEN"],
            webhook_url=os.environ["WEBHOOK_URL"],
            webhook_path=os.environ.get("WEBHOOK_PATH", "/webhook"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8443")),
            secret_token=os.environ.get("SECRET_TOKEN", ""),
        )
        config.validate()
        return config
