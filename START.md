> Panduan lengkap ada di **[RUN.md](RUN.md)**.

```bash
# Setup cepat
cp .env.example .env        # isi BOT_TOKEN, WEBHOOK_URL, OPENROUTER_API_KEY
pip install -r requirements.txt

# Terminal 1 – expose port
ngrok http 8443

# Terminal 2 – Telegram bot
python main.py

# Terminal 3 – REST API (opsional)
uvicorn src.interfaces.rest_api:app --port 8000 --reload
```
