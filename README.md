# satunggal-ai

Sistem **multi-agent AI** berbasis [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) v21 dan FastAPI yang berjalan dalam **mode webhook**. Bot hanya aktif saat ada pesan masuk dari Telegram, sehingga penggunaan RAM jauh lebih efisien dibanding polling.

Untuk daftar kemampuan lengkap lihat [CAPABILITIES.md](CAPABILITIES.md).  
Untuk panduan teknis pengembangan lihat [APP_LOGIC_WORKFLOW.md](APP_LOGIC_WORKFLOW.md).  
Untuk cara menjalankan lihat [RUN.md](RUN.md).

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
satunggal-ai/
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized settings (pydantic-settings)
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # ABC — semua agent wajib extends ini
│   │   ├── llm_client.py       # Wrapper HTTP ke OpenRouter/OpenAI-compatible
│   │   ├── repo_agent_base.py  # Base class untuk agent berbasis repo (clone/pull/RAG)
│   │   ├── gatekeeper/         # Klasifikasi intent → pilih agent + pre-agent tools
│   │   ├── responder/          # Percakapan umum, support, billing
│   │   ├── researcher/         # Riset mendalam + Tavily web search
│   │   ├── content_creator/    # Konten platform digital (LinkedIn, dll.)
│   │   ├── wbs_agent/          # WBS Gantt chart → Excel
│   │   ├── mandays_agent/      # Estimasi mandays → Excel
│   │   ├── developer/          # Clone repo → edit kode via LLM → Docker sandbox → push
│   │   ├── developer_inspector/# Inspeksi repo read-only, root cause analysis
│   │   ├── developer_qna/      # Tanya-jawab tentang isi codebase (API, model, dll.)
│   │   ├── technical_writer/   # Buat dokumen teknis PDF/DOCX dari repo atau topik
│   │   ├── doc_agent/          # Analisis, Q&A, dan edit dokumen .docx
│   │   ├── quiz_agent/         # Konversi PDF → kuis HTML interaktif
│   │   ├── sysinfo_agent/      # Laporan CPU, RAM, storage server
│   │   ├── log_viewer_agent/   # Tampilkan log bot untuk debugging
│   │   ├── web_automation/     # Autonomous browsing: buka URL, klik, isi form
│   │   └── reminder/           # Set / list / cancel timed reminders via Telegram
│   ├── handlers/
│   │   ├── command.py          # /start /help /ping /reset
│   │   └── message.py          # Teks, foto, dokumen PDF, dokumen DOCX
│   ├── interfaces/
│   │   ├── rest_api.py         # FastAPI: /chat, /health, /session/{id}
│   │   ├── telegram_bot.py     # Telegram Application builder
│   │   └── webhook.py          # Webhook runner (uvicorn)
│   ├── memory/
│   │   ├── state.py            # AgentTask blackboard
│   │   ├── history.py          # ConversationHistory (in-memory)
│   │   └── repo_tracker.py     # SQLite registry repo yang pernah di-clone
│   ├── orchestrator/
│   │   ├── main_loop.py        # process_message() — controller utama pipeline
│   │   └── router.py           # INTENT_AGENT_MAP: intent → agent name
│   └── tools/
│       ├── base_tool.py            # ABC — semua tool wajib extends ini
│       ├── tavily_search.py        # Pre-agent: live web search
│       ├── wbs_generator.py        # Post-agent: build WBS Excel Gantt
│       ├── mandays_generator.py    # Post-agent: build Mandays Excel
│       ├── diagram_renderer.py     # Post-agent: render blok Mermaid → PNG
│       ├── document_generator.py   # Post-agent: Markdown → PDF/DOCX (WeasyPrint/Pandoc)
│       ├── pdf_parser.py           # Ekstraksi teks dari PDF via PyMuPDF
│       ├── web_quiz_builder.py     # Generator HTML kuis interaktif
│       ├── browser_navigator.py    # Playwright browser controller (klik, isi form, screenshot)
│       ├── web_reader.py           # HTTP page reader + a11y locators
│       ├── cli_executor.py         # Async non-interactive shell runner
│       ├── sandbox_runner.py       # Docker build/run + traceback detection
│       ├── git_manager.py          # git commit/push dengan PAT auth
│       ├── docx_parser.py          # Ekstraksi seksi/bab dari .docx
│       ├── docx_editor.py          # Edit .docx via XML operations
│       ├── sysinfo_tool.py         # Metrik CPU/RAM/storage via psutil
│       ├── reminder_store.py       # SQLite store untuk reminder jobs
│       ├── log_buffer.py           # In-memory ring buffer log bot
│       ├── repo_qa.py              # Engine Q&A: ekstraksi API, model, tech stack, dll.
│       ├── identity_generator.py   # Pembuat identitas acak untuk web automation
│       ├── progress_tracker.py     # Live-update progress message di Telegram
│       ├── wbs/                    # CLI standalone: Excel ↔ WBS JSON
│       └── mandays/                # CLI standalone: Excel ↔ Mandays JSON
├── docs/
│   └── follow_parent_flowchart.md  # Flowchart fitur follow_parent WebAutomationAgent
├── main.py                  # Entry point Telegram bot + webhook
├── requirements.txt         # Python dependencies
├── README.md
├── RUN.md                   # Panduan menjalankan aplikasi
├── START.md                 # Quickstart
├── APP_LOGIC_WORKFLOW.md    # Panduan teknis pengembangan komponen
├── CAPABILITIES.md          # Daftar kemampuan & roadmap
├── AGENT_GIT.md             # Agent dengan akses Git
└── TOOLS_API_REFERENCE.md   # Referensi API internal tools
```

---

## Quickstart

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # isi variabel yang wajib

# 2. Expose port (development)
ngrok http 8443        # salin URL ke WEBHOOK_URL di .env

# 3. Jalankan bot
python main.py

# 4. REST API (opsional, terminal terpisah)
uvicorn src.interfaces.rest_api:app --host 0.0.0.0 --port 8000 --reload
```

Lihat [RUN.md](RUN.md) untuk panduan lengkap.

---

## Environment Variables

| Variabel | Wajib | Keterangan |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | Token dari @BotFather |
| `WEBHOOK_URL` | ✅ | URL HTTPS publik server (untuk Telegram webhook) |
| `OPENROUTER_API_KEY` | ✅ | API key untuk LLM via OpenRouter |
| `OPENROUTER_MODEL` | ❌ | Model LLM (default: `openai/gpt-4o-mini`) |
| `TAVILY_API_KEY` | ❌ | API key Tavily untuk live web search |
| `GITHUB_PAT` | ❌ | GitHub Personal Access Token untuk akses repo private |
| `SANDBOX_REPOS_DIR` | ❌ | Direktori clone repo (default: `~/sandbox_repos`) |
| `PORT` | ❌ | Port webhook (default: `8443`) |
| `API_PORT` | ❌ | Port REST API (default: `8000`) |

---

## Menambah Komponen Baru

Lihat [APP_LOGIC_WORKFLOW.md](APP_LOGIC_WORKFLOW.md) untuk panduan lengkap menambah agent baru, tool baru, dan intent baru.
