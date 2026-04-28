# AdvanceAI – Panduan Menjalankan Aplikasi

Sistem multi-agent AI dengan dua interface: **Telegram Bot** dan **REST API**.

---

## Prasyarat

| Kebutuhan | Versi Minimum |
|-----------|--------------|
| Python    | 3.12+        |
| ngrok (untuk dev lokal Telegram) | [download](https://ngrok.com/download) |
| Akun OpenRouter | [openrouter.ai](https://openrouter.ai/keys) |
| Bot Telegram | Buat via [@BotFather](https://t.me/BotFather) |

---

## 1. Setup Awal (sekali saja)

### 1.1 Clone & masuk ke direktori
```bash
cd advance_ai
```

### 1.2 Buat virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 1.3 Install dependensi
```bash
pip install -r requirements.txt
```

### 1.4 Konfigurasi `.env`
```bash
cp .env.example .env
```

Edit `.env` dan isi nilai berikut:

```dotenv
# Wajib diisi
BOT_TOKEN=7869xxxxxx:AAF...          # dari @BotFather
WEBHOOK_URL=https://abc123.ngrok-free.app
OPENROUTER_API_KEY=sk-or-v1-...      # dari openrouter.ai/keys

# Opsional (sudah ada default)
OPENROUTER_MODEL=openai/gpt-4o-mini
PORT=8443
API_PORT=8000
```

Generate `SECRET_TOKEN` (dianjurkan):
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 2. Menjalankan Telegram Bot (Webhook)

Telegram memerlukan URL HTTPS publik. Gunakan **ngrok** untuk development lokal.

### Terminal 1 — Expose port via ngrok
```bash
ngrok http 8443
```
Salin URL yang muncul (contoh: `https://abc123.ngrok-free.app`) ke `WEBHOOK_URL` di `.env`.

### Terminal 2 — Jalankan bot
```bash
source .venv/bin/activate
python main.py
```

Output yang diharapkan:
```
2026-03-10 10:00:00 | INFO     | __main__ — === Telegram Webhook Bot — starting ===
2026-03-10 10:00:00 | INFO     | __main__ — Webhook URL  : https://abc123.ngrok-free.app/webhook
2026-03-10 10:00:00 | INFO     | __main__ — Listen       : 0.0.0.0:8443
```

### Perintah Bot yang Tersedia
| Perintah | Fungsi |
|----------|--------|
| `/start`  | Salam perkenalan |
| `/help`   | Daftar perintah |
| `/ping`   | Cek status bot |
| `/reset`  | Hapus riwayat percakapan |

---

## 3. Menjalankan REST API

REST API berjalan **terpisah** dari Telegram bot, tidak butuh ngrok.

```bash
source .venv/bin/activate
uvicorn src.interfaces.rest_api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi interaktif tersedia di:
- Swagger UI : http://localhost:8000/docs
- ReDoc      : http://localhost:8000/redoc

### Endpoint

| Method | Path | Fungsi |
|--------|------|--------|
| `GET`    | `/health`              | Cek status server |
| `POST`   | `/chat`                | Kirim pesan ke agent |
| `DELETE` | `/session/{session_id}` | Reset riwayat percakapan |

### Contoh Request

**Kirim pesan:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "Buatkan WBS untuk proyek website"}'
```

**Cek kesehatan:**
```bash
curl http://localhost:8000/health
```

**Reset sesi:**
```bash
curl -X DELETE http://localhost:8000/session/user-123
```

---

## 4. Menjalankan Keduanya Sekaligus

Jalankan di dua terminal terpisah:

```
Terminal 1: ngrok http 8443
Terminal 2: python main.py          ← Telegram webhook
Terminal 3: uvicorn src.interfaces.rest_api:app --port 8000 --reload  ← REST API
```

---

## 5. Alur Pipeline (Referensi)

```
Pesan Masuk (Telegram / REST API)
        │
        ▼
 orchestrator/main_loop.py :: process_message()
        │
        ├─► GatekeeperAgent   → klasifikasi intent
        │         │
        │    IntentCategory:
        │      general_inquiry / complaint / billing / order_status  → ResponderAgent
        │      technical_support / image_query                       → ResearcherAgent
        │      data_analysis                                         → WBSAgent
        │
        ├─► Agent.run(task)   → generate reply
        │
        └─► memory/history   → simpan ke riwayat percakapan
```

---

## 6. Menjalankan Tools Standalone

### WBS Generator
```bash
cd src/tools/wbs
pip install -r requirements.txt

# Excel → JSON
python extract_wbs.py input.xlsx -o output.json

# JSON → Excel
python generate_wbs.py output.json wbs_output.xlsx
```
> Panduan lengkap: [src/tools/wbs/RUN.md](src/tools/wbs/RUN.md)

### Mandays Calculator
```bash
cd src/tools/mandays
pip install -r requirements.txt
python generate_mandays.py
```
> Panduan lengkap: [src/tools/mandays/RUN.md](src/tools/mandays/RUN.md)

---

## 7. Troubleshooting

### Bot Telegram tidak merespons
- Pastikan `WEBHOOK_URL` di `.env` sama persis dengan URL ngrok yang aktif.
- Cek apakah ngrok masih berjalan di Terminal 1.
- Cek log di Terminal 2 untuk error.

### Error `OPENROUTER_API_KEY` missing
```
pydantic_core.ValidationError: Field required [OPENROUTER_API_KEY]
```
→ Pastikan `.env` sudah di-copy dari `.env.example` dan `OPENROUTER_API_KEY` sudah diisi.

### Error `BOT_TOKEN` / `WEBHOOK_URL` missing
→ Kedua variabel ini wajib diisi di `.env` untuk menjalankan Telegram bot.

### `ModuleNotFoundError`
```bash
# Pastikan venv aktif
source .venv/bin/activate

# Re-install dependensi
pip install -r requirements.txt
```

### Ganti model LLM
Edit `.env`:
```dotenv
OPENROUTER_MODEL=anthropic/claude-3-haiku   # lebih hemat
OPENROUTER_MODEL=openai/gpt-4o              # lebih pintar
```
Daftar model tersedia di [openrouter.ai/models](https://openrouter.ai/models).

## 9. Fase 4 – Proactive Mode (Background Jobs)

Bot dapat menjalankan tugas secara **proaktif** tanpa menunggu pesan dari user.

### 9.1 Daily Briefing (Ringkasan Berita Harian)

ResearcherAgent mengumpulkan berita dari Tavily dan mengirimnya ke Telegram setiap hari pada waktu yang dikonfigurasi.

```dotenv
# Aktifkan daily briefing
PROACTIVE_BRIEFING_ENABLED=true

# Target chat ID Telegram (kosongkan = gunakan ADMIN_USER_ID)
PROACTIVE_BRIEFING_CHAT_ID=123456789

# Jam kirim dalam format HH:MM zona WIB (UTC+7)
PROACTIVE_BRIEFING_TIME=07:00

# Topik yang akan diriset, dipisah koma
PROACTIVE_BRIEFING_TOPICS=AI terbaru, berita teknologi, startup Indonesia

# Bahasa laporan: id (Indonesia) atau en (English)
PROACTIVE_BRIEFING_LANGUAGE=id
```

> **Catatan:** Fitur ini membutuhkan `TAVILY_API_KEY` yang valid.

### 9.2 Repo Watcher (Pemantau Repository Otomatis)

Memantau satu atau lebih repository GitHub/GitLab secara berkala. Jika ada commit baru, DeveloperInspectorAgent akan melakukan code review dan mengirimkan laporan ke Telegram.

```dotenv
# Aktifkan repo watcher
PROACTIVE_REPO_WATCHER_ENABLED=true

# Target chat ID Telegram (kosongkan = gunakan ADMIN_USER_ID)
PROACTIVE_REPO_WATCHER_CHAT_ID=123456789

# Daftar URL repo yang dipantau, dipisah koma
PROACTIVE_REPO_WATCHER_REPOS=https://github.com/org/repo1,https://github.com/org/repo2

# Interval polling dalam menit (default: 60)
PROACTIVE_REPO_WATCHER_INTERVAL=60
```

> **Catatan:** Untuk repo private, pastikan `GITHUB_PAT` atau `GITLAB_PAT` sudah dikonfigurasi.

---


```
advance_ai/
├── main.py                          ← Entry point Telegram bot
├── .env                             ← Konfigurasi (jangan di-commit!)
├── .env.example                     ← Template konfigurasi
├── requirements.txt                 ← Dependensi Python
│
├── config/
│   └── settings.py                  ← Centralized Settings (pydantic)
│
├── src/
│   ├── orchestrator/
│   │   ├── main_loop.py             ← Pipeline utama (semua interface pakai ini)
│   │   └── router.py               ← Intent → Agent routing
│   │
│   ├── agents/
│   │   ├── gatekeeper/             ← Klasifikasi intent
│   │   ├── responder/              ← Agen percakapan umum
│   │   ├── researcher/             ← Agen riset & teknikal
│   │   └── wbs_agent/             ← Agen WBS & project planning
│   │
│   ├── memory/
│   │   ├── state.py                ← AgentTask blackboard
│   │   └── history.py             ← Riwayat percakapan per sesi
│   │
│   ├── interfaces/
│   │   ├── telegram_bot.py         ← Telegram Application builder
│   │   ├── rest_api.py             ← FastAPI endpoints
│   │   └── webhook.py             ← Webhook runner
│   │
│   ├── handlers/
│   │   ├── command.py              ← /start /help /ping /reset
│   │   └── message.py             ← Text & photo handlers
│   │
│   └── tools/
│       ├── wbs/                    ← WBS Excel tools
│       └── mandays/               ← Mandays calculator
```
