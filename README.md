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
advance_ai/
├── config/
│   ├── __init__.py
│   └── settings.py         # Konfigurasi aplikasi
├── src/
│   ├── agents/             # Modul AI agents
│   │   ├── base_agent.py   # Base class untuk semua agent
│   │   ├── llm_client.py   # Client untuk integrasi LLM
│   │   ├── content_creator/    # Agent pembuat konten
│   │   ├── developer/          # Agent developer
│   │   ├── gatekeeper/         # Agent gatekeeper (dengan OpenRouter)
│   │   ├── mandays_agent/      # Agent mandays calculator
│   │   ├── researcher/         # Agent peneliti
│   │   ├── responder/          # Agent responder
│   │   ├── technical_writer/   # Agent pembuat dokumen teknis (PDF/Word)
│   │   │   ├── __init__.py
│   │   │   └── agent.py        # TechnicalWriterAgent: Markdown → pending_tools
│   │   └── wbs_agent/          # Agent WBS (Work Breakdown Structure)
│   ├── handlers/           # Handler untuk lalu lintas pesan
│   │   ├── command.py      # Handle command dari user
│   │   └── message.py      # Handle pesan biasa
│   ├── interfaces/         # Interface ke berbagai platform
│   │   ├── config.py       # Konfigurasi interface
│   │   ├── rest_api.py     # REST API endpoints
│   │   ├── telegram_bot.py # Integrasi Telegram bot
│   │   └── webhook.py      # Webhook handler
│   ├── memory/             # Memory & state management
│   │   ├── history.py      # Riwayat percakapan
│   │   ├── repo_tracker.py # Tracking repository
│   │   └── state.py        # State management
│   ├── orchestrator/       # Orkestrasi alur proses
│   │   ├── main_loop.py    # Main event loop
│   │   └── router.py       # Message router
│   └── tools/              # Tools & utilities
│       ├── base_tool.py    # Base class tool
│       ├── cli_executor.py # Eksekusi CLI commands
│       ├── code_search.py  # Pencarian kode
│       ├── diagram_renderer.py     # Render blok Mermaid ke PNG (via mmdc)
│       ├── document_generator.py   # Compile Markdown ke PDF (WeasyPrint) / Word (Pandoc)
│       ├── git_manager.py  # Git management
│       ├── mandays_generator.py    # Generate mandays estimation
│       ├── sandbox_runner.py       # Sandbox untuk eksekusi aman
│       ├── tavily_search.py        # Search API integration
│       ├── wbs_generator.py        # Generate WBS
│       ├── mandays/        # Submodule mandays
│       └── wbs/            # Submodule WBS
├── data/                   # Direktori data
│   └── templates/          # Template dokumen (template.docx untuk Pandoc)
├── main.py                 # Entry point aplikasi
├── requirements.txt        # Python dependencies
├── README.md               # Dokumentasi proyek
├── RUN.md                  # Panduan menjalankan aplikasi
├── START.md                # Panduan memulai
├── APP_LOGIC_WORKFLOW.md   # Workflow logika aplikasi
├── CAPABILITIES.md         # Daftar capabilities
└── __init__.py
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
