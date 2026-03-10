# Telegram Webhook Bot

Bot Telegram berbasis [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) v21 yang berjalan dalam **mode webhook** — bot hanya aktif saat ada pesan masuk dari Telegram, sehingga penggunaan RAM jauh lebih efisien dibanding polling.

---

## Mengapa Webhook?

| | Polling | Webhook |
|---|---|---|
| Koneksi | Persisten (long-poll) | Hanya saat ada pesan |
| RAM | Lebih tinggi | Lebih rendah |
| Latensi | ~1 detik | < 100 ms |
| Kebutuhan | Cukup akses internet | Wajib HTTPS publik |

---

## Struktur Proyek

```
telegram/
├── src/
│   ├── config.py           # Load & validasi env vars
│   ├── bot.py              # Rakit Application + daftarkan handler
│   ├── webhook.py          # Setup & jalankan webhook server
│   └── handlers/
│       ├── command.py      # /start /help /ping
│       └── message.py      # Teks, foto, fallback
├── main.py                 # Entry point
├── .env.example            # Template environment variables
└── requirements.txt
```

---

## Cara Pemakaian

### 1. Kloning & install dependensi

```bash
cd telegram
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Buat file `.env`

```bash
cp .env.example .env
# Edit nilai BOT_TOKEN, WEBHOOK_URL, dll.
```

### 3. Ekspos server lokal (development)

Gunakan [ngrok](https://ngrok.com/) untuk mendapatkan URL HTTPS publik:

```bash
ngrok http 8443
# Salin URL seperti https://abc123.ngrok.io ke WEBHOOK_URL di .env
```

### 4. Jalankan bot

```bash
python main.py
```

---

## Environment Variables

| Variable | Wajib | Default | Keterangan |
|---|---|---|---|
| `BOT_TOKEN` | ✅ | — | Token dari @BotFather |
| `WEBHOOK_URL` | ✅ | — | URL HTTPS publik server kamu |
| `WEBHOOK_PATH` | ❌ | `/webhook` | Path endpoint |
| `HOST` | ❌ | `0.0.0.0` | Host server lokal |
| `PORT` | ❌ | `8443` | Port (80/443/88/8443) |
| `SECRET_TOKEN` | ❌ | — | Token validasi request Telegram |

---

## Menambah Handler Baru

1. Tambahkan fungsi handler di `src/handlers/command.py` atau `message.py`.
2. Export dari `src/handlers/__init__.py`.
3. Daftarkan handler di `src/bot.py` dalam fungsi `_register_handlers()`.
