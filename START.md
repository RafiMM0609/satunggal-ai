> Panduan lengkap ada di **[RUN.md](RUN.md)**.

```bash
# Setup cepat
cp .env.example .env        # isi TELEGRAM_BOT_TOKEN, WEBHOOK_URL, OPENROUTER_API_KEY
pip install -r requirements.txt

# Terminal 1 – expose port (development)
ngrok http 8443

# Terminal 2 – Telegram bot + webhook
python main.py

# Terminal 3 – REST API (opsional)
uvicorn src.interfaces.rest_api:app --host 0.0.0.0 --port 8000 --reload
```

Lihat [CAPABILITIES.md](CAPABILITIES.md) untuk daftar lengkap kemampuan AI asisten.
