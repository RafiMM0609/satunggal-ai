# satunggal-ai — Kemampuan & Roadmap

> Dokumen ini menjelaskan kemampuan yang saat ini diimplementasikan dan rencana pengembangan ke depan.  
> Untuk panduan teknis alur kerja dan cara menambah komponen baru, lihat [APP_LOGIC_WORKFLOW.md](APP_LOGIC_WORKFLOW.md).

---

## Arsitektur Sistem

```
User (Telegram / REST API)
        │
        ▼
  GatekeeperAgent          ← klasifikasi intent + pilih pre-agent tools
        │
        ▼
  Pre-agent Tool Loop      ← jalankan tools sebelum agent (contoh: tavily_search)
        │
        ▼
   AgentRouter             ← pilih specialist agent berdasarkan intent
        │
        ├── ResponderAgent         → percakapan umum, support, billing
        ├── ResearcherAgent        → riset mendalam + Tavily web search
        ├── ContentCreatorAgent    → konten platform digital (LinkedIn, dll.)
        ├── WBSAgent               → WBS Gantt chart → Excel
        ├── MandaysAgent           → estimasi mandays → Excel
        ├── DeveloperAgent         → clone repo → edit kode via LLM → Docker sandbox → push
        ├── DeveloperInspectorAgent→ inspeksi repo read-only, root cause analysis
        ├── DeveloperQnAAgent      → tanya-jawab tentang isi codebase
        ├── TechnicalWriterAgent   → buat dokumen teknis PDF/DOCX dari repo atau topik
        ├── DocAgent               → analisis, Q&A, dan edit dokumen .docx
        ├── QuizAgent              → konversi PDF → kuis HTML interaktif
        ├── SysInfoAgent           → laporan CPU, RAM, storage server
        ├── LogViewerAgent         → tampilkan log bot untuk debugging
        ├── WebAutomationAgent     → autonomous browsing: buka URL, klik, isi form
        └── ReminderAgent          → set / list / cancel timed reminders
              │
              ▼
        Post-agent Tool Loop    ← jalankan tools yang diminta agent (pending_tools)
              │
              ▼
          Interface             ← kirim teks + file ke pengguna
```

Sistem menggunakan **multi-agent** berbasis LLM (via OpenRouter).  
Semua state perjalanan pipeline dibawa oleh satu objek **`AgentTask`** (blackboard pattern).  
Orchestrator adalah satu-satunya yang memanggil tools — agent **tidak** memanggil tools secara langsung.

---

## 1. Interface & Akses

| Interface | Status | Keterangan |
|---|---|---|
| Telegram Bot | ✅ Aktif | Polling & Webhook |
| REST API | ✅ Aktif | FastAPI, endpoint `/chat`, `/health`, `/session/{id}` |

### Perintah Telegram

| Perintah | Fungsi |
|---|---|
| `/start` | Sapa pengguna, tampilkan intro bot |
| `/help` | Tampilkan daftar perintah & kemampuan |
| `/ping` | Cek status & latensi bot |
| `/reset` | Hapus riwayat percakapan & tutup browser sesi aktif |

---

## 2. Klasifikasi Intent (GatekeeperAgent)

Bot secara otomatis mendeteksi maksud pesan pengguna dan meneruskannya ke agent yang tepat:

| Intent | Deskripsi | Agent |
|---|---|---|
| `general_inquiry` | Pertanyaan umum | ResponderAgent |
| `product_question` | Pertanyaan seputar produk | ResponderAgent |
| `complaint` | Keluhan pengguna | ResponderAgent |
| `order_status` | Status pesanan | ResponderAgent |
| `billing` | Pertanyaan tagihan/pembayaran | ResponderAgent |
| `technical_support` | Pertanyaan teknis ringan | ResponderAgent |
| `image_query` | Pertanyaan terkait gambar/foto | ResponderAgent |
| `unknown` | Intent tidak dikenali | ResponderAgent |
| `research` | Riset mendalam, butuh data web | ResearcherAgent |
| `content_creation` | Buat konten platform digital | ContentCreatorAgent |
| `data_analysis` | WBS, Gantt chart, project planning | WBSAgent |
| `mandays_planning` | Estimasi mandays, effort, resource | MandaysAgent |
| `code_development` | Clone, edit kode, jalankan sandbox, push | DeveloperAgent |
| `code_inspection` | Inspeksi repo, temukan root cause (read-only) | DeveloperInspectorAgent |
| `code_understanding` | Tanya-jawab tentang isi repo (API, model, dll.) | DeveloperQnAAgent |
| `document_creation` | Buat dokumen teknis PDF/DOCX | TechnicalWriterAgent |
| `doc_audit` | Analisis, Q&A, edit dokumen .docx | DocAgent |
| `quiz_generation` | Konversi PDF → kuis HTML interaktif | QuizAgent |
| `system_info` | Info CPU, RAM, storage server | SysInfoAgent |
| `log_viewer` | Tampilkan log bot untuk debugging | LogViewerAgent |
| `web_automation` | Autonomous browsing & web interaction | WebAutomationAgent |
| `reminder` | Set / list / cancel timed reminders | ReminderAgent |

---

## 3. Agent Spesialis

### ResponderAgent
- Menjawab percakapan umum, pertanyaan produk, keluhan, status order, billing, dan pertanyaan teknis ringan.
- Menggunakan riwayat percakapan untuk konteks yang relevan.
- Mendukung bahasa **Indonesia dan Inggris** secara otomatis.

### ResearcherAgent
- Menangani riset mendalam dengan pendekatan step-by-step reasoning.
- Menggunakan **TavilySearchTool** untuk live web search sebelum menjawab.
- Mendukung bahasa **Indonesia dan Inggris**.

### ContentCreatorAgent
- Membuat konten terstruktur untuk platform digital (hook / body / CTA / hashtags).
- Mendukung berbagai platform: LinkedIn, Instagram, Twitter/X, dll.

### WBSAgent
- Membuat **Work Breakdown Structure (WBS)** dalam format **Gantt chart** berdasarkan deskripsi pengguna.
- Output: file **Excel (.xlsx)** dengan layout Gantt-style — timeline per hari kerja, sprint header, sel aktif berwarna per task.
- Dipicu oleh intent `data_analysis` (kata kunci: WBS, Gantt, breakdown, timeline proyek).

### MandaysAgent
- Membuat rencana mandays dan estimasi effort berdasarkan deskripsi proyek.
- Mendukung 13 role standar: `SA`, `TL`, `BA`, `SM`, `UI`, `DBA`, `BE1`, `BE2`, `FE1`, `FE2`, `QA`, `DevOps`, `TW`.
- Output: file **Excel (.xlsx)** dengan tabel mandays per role per sprint dan grand total.

### DeveloperAgent
- Alur coding end-to-end:
  1. Parse instruksi → ekstrak `repo_url` + task
  2. Clone/Pull repo (inject `GITHUB_PAT` bila private)
  3. Cek environment (Dockerfile/docker-compose) → buat fallback bila perlu
  4. Edit kode via LLM → tulis patch ke disk
  5. Jalankan sandbox (Docker Compose) → jika error, kirim log ke LLM → retry (maks. 3×)
  6. Commit & push (git)
- Output: ringkasan teks (file changed, commit hash, status sandbox, push URL).

### DeveloperInspectorAgent
- **Read-only**: inspeksi repo, analisis root cause bug/error, beri rekomendasi.
- Tidak boleh menulis ke repo (`git add/commit/push` dilarang).
- Dilarang menggunakan `GitManager`, `SandboxRunner`, atau perintah write.

### DeveloperQnAAgent
- Tanya-jawab faktual tentang isi codebase: API endpoints, tech stack, data models, CI/CD, security, alur utama.
- Menggunakan TF-IDF RAG untuk menemukan file relevan + LLM anti-halusinasi.
- Setiap klaim harus disertai sumber `file:baris`.

### TechnicalWriterAgent
- Menghasilkan dokumen teknis profesional dari repo GitHub atau topik bebas.
- Strategi chunking: bagi semua file repo menjadi chunk ~8 KB, proses per chunk ke LLM, lalu synthesize jadi dokumen final.
- Output: file **DOCX atau PDF** dikirim langsung ke pengguna.
- Melalui tiga fase: konteks (clone repo), penulisan (chunking + LLM), kompilasi (`DiagramRendererTool` + `DocumentGeneratorTool`).

### DocAgent
- **Mode Analisis**: saat file `.docx` baru dikirim — validasi seksi, buat daftar isi, ringkas setiap bab, simpan ke SQLite.
- **Mode Q&A**: jawab pertanyaan tentang isi dokumen menggunakan RAG berbasis bab.
- **Mode Edit**: kumpulkan instruksi edit dari user → hasilkan operasi JSON → simpan ke antrian.
- **Mode Apply**: terapkan semua edit ke file `.docx` asli → kirim file hasil ke pengguna.

### QuizAgent
- Menerima file PDF dari Telegram → ekstrak teks via PyMuPDF → bagi menjadi chunk → generate 10–15 soal per chunk via LLM.
- Output: file **HTML interaktif single-file** dengan fitur: dark mode, instant feedback, scoreboard, progress bar, dan mode review.
- Distribusi kesulitan soal: 30% mudah, 50% sedang, 20% sulit.
- Maks. ukuran PDF: 20 MB.

### SysInfoAgent
- Mengumpulkan metrik host via `psutil` (tidak ada shell command).
- Melaporkan: CPU (penggunaan, frekuensi, core), RAM (total/used/available), disk (total/used/free per mount point), uptime.
- Input user hanya mengontrol presentasi laporan, bukan pengumpulan data.

### LogViewerAgent
- Menampilkan log bot terbaru dari in-memory ring buffer (`log_buffer.py`).
- Berguna untuk debugging langsung dari Telegram tanpa akses SSH.

### WebAutomationAgent
- Autonomous browsing berbasis ReAct loop (maks. 20 langkah per sesi).
- Aksi yang didukung: `navigate`, `click`, `type`, `scroll`, `get_content`, `get_full_content`, `get_links`, `extract_data`, `check_captcha`, `close_popup`, screenshot.
- **follow_parent**: setelah tugas selesai, browser **tidak ditutup** — sesi berikutnya langsung melanjutkan di halaman yang sama. Browser ditutup saat `/reset`.
- Session persistence: cookies/localStorage disimpan ke disk antar sesi.
- Keamanan: satu tab, resource blocking (gambar/media/font), timeout 30 detik per aksi.

### ReminderAgent
- Set reminder dengan waktu/tanggal dalam bahasa natural (mis. "ingatkan saya besok jam 9 untuk meeting").
- Simpan ke SQLite, dijadwalkan via APScheduler → dikirim ke Telegram pada waktunya.
- Perintah: set reminder, list reminder, cancel reminder.

---

## 4. Tools Internal

### Pre-agent Tools (diputuskan GatekeeperAgent)

| Tool | Kapan Dijalankan | Fungsi |
|---|---|---|
| `TavilySearchTool` | Sebelum ResearcherAgent | Live web search → `task.tool_results["tavily_search"]` |

### Post-agent Tools (diminta agent via `pending_tools`)

| Tool | Kapan Dijalankan | Fungsi |
|---|---|---|
| `WBSGeneratorTool` | Setelah WBSAgent | Build Excel Gantt chart dari `task.metadata["wbs_json_data"]` |
| `MandaysGeneratorTool` | Setelah MandaysAgent | Build Excel mandays dari `task.metadata["mandays_json_data"]` |
| `DiagramRendererTool` | Setelah TechnicalWriterAgent | Render blok Mermaid → PNG via mmdc |
| `DocumentGeneratorTool` | Setelah TechnicalWriterAgent | Markdown + PNG → DOCX/PDF via Pandoc/WeasyPrint |
| `WebQuizBuilderTool` | Setelah QuizAgent | Inject soal JSON → HTML kuis interaktif |

### Internal DeveloperAgent Tools (bukan melalui pipeline orchestrator)

| Tool | Fungsi |
|---|---|
| `CLIExecutor` | Jalankan perintah shell non-interaktif (timeout 5 mnt) |
| `SandboxRunner` | Build & run Docker container; fallback Dockerfile/compose |
| `GitManager` | `git add -A → commit → push` dengan PAT auth |

---

## 5. Memori & Sesi

- Setiap pengguna memiliki sesi percakapan terpisah berdasarkan `session_id`.
- Riwayat percakapan disimpan **in-memory** selama bot berjalan.
- Sesi dapat di-reset via `/reset` (Telegram) atau `DELETE /session/{session_id}` (REST API).
- Reset juga menutup browser WebAutomationAgent dan menghapus sesi browser dari disk.

---

## 6. Output yang Dikirim ke Pengguna

| Output | Dikirim Oleh |
|---|---|
| Teks Markdown | Semua agent |
| File Excel Gantt `.xlsx` | WBSAgent |
| File Excel Mandays `.xlsx` | MandaysAgent |
| File HTML kuis interaktif | QuizAgent |
| File DOCX/PDF dokumen teknis | TechnicalWriterAgent |
| File DOCX hasil edit | DocAgent |
| Screenshot browser | WebAutomationAgent |
| Laporan coding (file changed, commit hash, sandbox status, push URL) | DeveloperAgent |

---

## 7. Konfigurasi & Variabel Lingkungan

| Variabel | Keterangan |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram (wajib) |
| `WEBHOOK_URL` | URL HTTPS publik untuk Telegram webhook (wajib) |
| `OPENROUTER_API_KEY` | API key untuk LLM via OpenRouter (wajib) |
| `OPENROUTER_MODEL` | Model LLM (default: `openai/gpt-4o-mini`) |
| `TAVILY_API_KEY` | API key Tavily untuk live web search |
| `GITHUB_PAT` | GitHub PAT untuk akses repo private |
| `SANDBOX_REPOS_DIR` | Direktori clone repo (default: `~/sandbox_repos`) |
| `PORT` | Port webhook (default: `8443`) |
| `API_PORT` | Port REST API (default: `8000`) |

---

## 8. Dependensi Utama

| Package | Kegunaan |
|---|---|
| `python-telegram-bot` | Interface Telegram |
| `fastapi` + `uvicorn` | REST API server |
| `httpx` | HTTP client untuk panggilan LLM |
| `pydantic` / `pydantic-settings` | Validasi skema data & konfigurasi |
| `openpyxl` | Generate & parse file Excel |
| `playwright` | Browser automation (WebAutomationAgent) |
| `PyMuPDF` | Ekstraksi teks dari PDF (QuizAgent) |
| `python-docx` | Read/write file DOCX |
| `psutil` | Metrik sistem (SysInfoAgent) |
| `APScheduler` | Penjadwalan reminder |
| `python-dotenv` | Manajemen konfigurasi `.env` |

---

## 9. Rencana Pengembangan

### Prioritas Tinggi

- [ ] **Persistensi Memori** — simpan riwayat percakapan ke database agar tidak hilang saat bot restart.
- [ ] **Autentikasi REST API** — tambahkan API key atau JWT untuk mengamankan endpoint REST API.
- [ ] **Confidence Threshold** — tolak atau minta klarifikasi jika confidence intent terlalu rendah (< 0.5).
- [ ] **Error Recovery** — jika LLM menghasilkan JSON tidak valid, lakukan retry otomatis dengan prompt koreksi.
- [ ] **Unit & Integration Tests** — tambahkan test suite untuk semua agent dan tools.

### Pengembangan Agent Baru

- [ ] **RAGAgent** — Retrieval-Augmented Generation dari basis pengetahuan internal (FAQ, SOP, dokumentasi produk).
- [ ] **ScheduleAgent** — buat jadwal proyek Gantt chart dari WBS yang sudah ada, output PDF atau Excel.
- [ ] **ReportAgent** — laporan progres proyek berdasarkan data mandays aktual vs. rencana.
- [ ] **Image Analysis Agent** — proses gambar yang dikirim pengguna dengan model vision.

### Pengembangan Interface

- [ ] **Web Dashboard** — frontend React/Next.js untuk monitor sesi, riwayat chat, dan output file.
- [ ] **WhatsApp Interface** — tambahkan interface WhatsApp via Twilio atau WhatsApp Business API.
- [ ] **Slack / Discord Bot** — integrasikan ke platform kolaborasi tim.
- [ ] **Voice Input** — speech-to-text sebelum diproses oleh pipeline agent.

### Peningkatan Kualitas

- [ ] **Multi-turn WBS Refinement** — izinkan pengguna memperbaiki WBS secara iteratif dalam satu sesi.
- [ ] **Rate Limiting** — batasi jumlah request per sesi untuk mencegah abuse.
- [ ] **Logging & Monitoring** — integrasi Sentry / Grafana / Prometheus untuk monitoring produksi.
- [ ] **Template Proyek WBS** — sediakan template WBS per jenis proyek (e-commerce, mobile app, ERP, dll.).
