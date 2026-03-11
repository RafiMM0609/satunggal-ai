# AdvanceAI – Kemampuan & Roadmap

> Dokumen ini menjelaskan kemampuan bot yang sudah berjalan saat ini dan rencana pengembangan ke depan.  
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
        ├── ResponderAgent      → percakapan umum
        ├── ResearcherAgent     → riset mendalam + Tavily web search
        ├── ContentCreatorAgent → pembuatan konten platform (LinkedIn, dll)
        ├── WBSAgent            → WBS Gantt chart + export Excel
        ├── MandaysAgent        → estimasi mandays + export Excel
        └── DeveloperAgent      → clone repo → edit kode via LLM → Docker sandbox
              │
              ▼
        Post-agent Tool Loop    ← jalankan tools yang diminta agent (pending_tools)
              │
              ▼
          Interface             ← kirim teks + file ke pengguna
```

Bot menggunakan sistem **multi-agent** berbasis LLM (via OpenRouter).  
Semua state perjalanan pipeline dibawa oleh satu objek **`AgentTask`** (blackboard pattern).  
Orchestrator adalah satu-satunya yang memanggil tools — agent **tidak** memanggil tools secara langsung.

---

## Kemampuan Saat Ini

### 1. Interface & Akses

| Interface    | Status   | Keterangan |
|--------------|----------|------------|
| Telegram Bot | ✅ Aktif | Polling & Webhook |
| REST API     | ✅ Aktif | FastAPI, endpoint `/chat` dan `/clear/{session_id}` |
| Webhook      | ✅ Aktif | Integrasi Telegram via webhook |

#### Perintah Telegram

| Perintah | Fungsi |
|----------|--------|
| `/start`  | Sapa pengguna, tampilkan intro bot |
| `/help`   | Tampilkan daftar perintah & kemampuan |
| `/ping`   | Cek status & latensi bot |
| `/reset`  | Hapus riwayat percakapan sesi aktif |

---

### 2. Klasifikasi Intent (GatekeeperAgent)

Bot secara otomatis mendeteksi maksud pesan pengguna dan meneruskannya ke agent yang tepat.

| Intent | Deskripsi | Agent | Pre-agent Tool |
|--------|-----------|-------|----------------|
| `general_inquiry` | Pertanyaan umum | ResponderAgent | — |
| `product_question` | Pertanyaan seputar produk | ResponderAgent | — |
| `complaint` | Keluhan pengguna | ResponderAgent | — |
| `order_status` | Status pesanan | ResponderAgent | — |
| `billing` | Pertanyaan tagihan/pembayaran | ResponderAgent | — |
| `technical_support` | Masalah teknis (tanpa kata riset eksplisit) | ResponderAgent | — |
| `image_query` | Pertanyaan terkait gambar/foto | ResponderAgent | — |
| `unknown` | Intent tidak dikenali | ResponderAgent | — |
| `research` | Riset mendalam dengan kata kunci investigatif eksplisit | ResearcherAgent | `tavily_search` |
| `content_creation` | Buat konten untuk platform digital | ContentCreatorAgent | — |
| `data_analysis` | WBS / Gantt chart proyek | WBSAgent | — |
| `mandays_planning` | Estimasi mandays, effort, alokasi resource | MandaysAgent | — |
| `code_development` | Clone repo, edit kode via AI, jalankan di Docker sandbox | DeveloperAgent | — |

> **Catatan `research`:** Hanya dipicu oleh kata kunci investigatif eksplisit (riset, selidiki, deep dive, dll).  
> Pertanyaan teknis biasa tanpa kata kunci tersebut → `technical_support` → ResponderAgent.

---

### 3. Agent Spesialis

#### ResponderAgent
- Menjawab percakapan umum, pertanyaan produk, keluhan, status order, billing, dan technical support ringan.
- Menggunakan riwayat percakapan (last 10 pesan) sebagai konteks.
- Mendukung bahasa **Indonesia dan Inggris** secara otomatis.
- **Tidak** menggunakan Tavily web search.

#### ResearcherAgent
- Menangani permintaan riset mendalam yang menggunakan kata kunci investigatif eksplisit.
- **Diperkaya data web real-time** melalui `TavilySearchTool` (dijalankan orchestrator sebelum agent dipanggil).
- Memberikan jawaban komprehensif dengan analisis step-by-step.
- Menggunakan riwayat percakapan (last 8 pesan) sebagai konteks.
- Fallback ke LLM-only jika Tavily tidak dikonfigurasi.

#### ContentCreatorAgent
- Mengubah ide atau riset menjadi konten siap-publikasi untuk platform digital.
- Output terstruktur: `hook`, `body`, `cta`, `hashtags`, `platform`.
- Mendukung platform: **LinkedIn**, **Twitter/X**, **Blog**, dan platform lain.
- Menghasilkan preview teks langsung di Telegram.

#### WBSAgent
- Membuat **Work Breakdown Structure (WBS)** dalam format **Gantt chart** dari deskripsi pengguna.
- Alur: Agent → LLM → JSON → `pending_tools` → Orchestrator → `WBSGeneratorTool` → Excel.
- Output: **file Excel (.xlsx)** dengan layout Gantt-style (timeline per hari kerja, sprint header, sel aktif berwarna) dikirim langsung ke pengguna.
- Dipicu oleh intent `data_analysis`.

#### MandaysAgent
- Membuat **rencana estimasi mandays** dan alokasi sumber daya dari deskripsi proyek.
- Alur: Agent → LLM → JSON → `pending_tools` → Orchestrator → `MandaysGeneratorTool` → Excel.
- Output: **file Excel (.xlsx)** dengan tabel mandays per role per sprint + grand total.
- Mendukung 13 role standar: `SA`, `TL`, `BA`, `SM`, `UI`, `DBA`, `BE1`, `BE2`, `FE1`, `FE2`, `QA`, `DevOps`, `TW`.
- Dipicu oleh intent `mandays_planning`.

#### DeveloperAgent
- **Senior Developer Orchestrator** – mengeksekusi tugas coding end-to-end dari pesan pengguna.
- Dipicu oleh intent `code_development` (kata kunci: clone repo, perbaiki kode, tambah fitur, jalankan di sandbox, daftar repo).

**Alur kerja internal DeveloperAgent:**

```
1. Parse instruksi  → LLM ekstrak repo_url + task dari pesan pengguna
2. Clone / Pull     → git clone (repo baru) atau git pull (sudah ada)
                      · Inject GITHUB_PAT ke HTTPS URL otomatis
                      · Simpan ke RepoTracker (SQLite)
3. Environment      → Cek Dockerfile & docker-compose.yml
                      · Jika tidak ada → generate fallback otomatis
4. Edit Kode        → Mode LLM-direct (primary):
                        · Scan struktur repo (find .)
                        · Grep file relevan berdasarkan keyword task
                        · Baca isi file (max 80 KB)
                        · Kirim ke OpenRouter → JSON patch
                        · Tulis file ke disk
                      Mode claude CLI (opsional, jika claude terinstall):
                        · claude -p "<task>" --allowedTools "Read,Edit,Write,Bash"
5. Verifikasi       → docker compose up --build --abort-on-container-exit
                      · Deteksi Python traceback di log
                      · Jika gagal → kirim error log ke LLM → retry (max 3x)
6. Commit & Push    → git add -A → git commit → git push (dengan PAT auth)
7. Report           → Summary / Files Changed / Commit Message / Docker Status / Push Status
```

**Catatan penting:**
- `gh copilot suggest` **tidak digunakan** karena hanya menyarankan perintah shell interaktif, bukan mengedit file.
- Semua logika tool (CLIExecutor, SandboxRunner, GitManager) dikelola **internal** di dalam agent — tidak melalui pipeline orchestrator.
- Jika user mengirim pesan tanpa repo URL (contoh: "tampilkan daftar repo"), agent menampilkan semua repo dari SQLite tracker.

---

### 4. Tools Internal

Semua tools merupakan subclass dari `BaseTool` dan hanya dipanggil oleh **orchestrator**.

| Tool | Tipe | Kapan Dijalankan | Fungsi |
|------|------|------------------|--------|
| `TavilySearchTool` | Pre-agent | Sebelum ResearcherAgent | Live web search, hasilnya masuk `task.tool_results["tavily_search"]` |
| `WBSGeneratorTool` | Post-agent | Setelah WBSAgent selesai | Build Excel Gantt chart dari JSON di `task.metadata["wbs_json_data"]` |
| `MandaysGeneratorTool` | Post-agent | Setelah MandaysAgent selesai | Build Excel mandays dari JSON di `task.metadata["mandays_json_data"]` |

Tools internal DeveloperAgent (dikelola langsung oleh agent, **tidak** melalui pipeline orchestrator):

| Tool | File | Fungsi |
|------|------|--------|
| `CLIExecutor` | `src/tools/cli_executor.py` | Jalankan perintah shell non-interaktif dengan timeout 5 menit, capture stdout+stderr |
| `SandboxRunner` | `src/tools/sandbox_runner.py` | Build & run Docker container; generate Dockerfile/compose fallback jika tidak ada; deteksi traceback |
| `GitManager` | `src/tools/git_manager.py` | Konfigurasi identitas git, inject PAT ke URL, `git add -A → commit → push` |

Tools utility (standalone, tidak dalam pipeline):

| File | Fungsi |
|------|--------|
| `src/tools/wbs/extract_wbs.py` | Parse Excel WBS → JSON (untuk reverse engineering) |
| `src/tools/mandays/extract_mandays.py` | Parse Excel Mandays → JSON |

---

### 5. Memori & Sesi

- Setiap pengguna memiliki **sesi percakapan terpisah** berdasarkan `session_id`.
- Riwayat percakapan disimpan **in-memory** selama bot berjalan.
- Sesi dapat di-reset via `/reset` (Telegram) atau `DELETE /clear/{session_id}` (REST API).

---

### 6. Output

| Tipe Output | Siapa yang Menghasilkan | Format |
|-------------|------------------------|--------|
| Teks Markdown | Semua agent | Telegram MarkdownV2 |
| File Excel Gantt | WBSAgent + WBSGeneratorTool | `.xlsx` dikirim via Telegram |
| File Excel Mandays | MandaysAgent + MandaysGeneratorTool | `.xlsx` dikirim via Telegram |
| Draft konten | ContentCreatorAgent | Teks terstruktur (hook/body/cta/hashtags) |
| Laporan coding | DeveloperAgent | Teks: file changed, commit hash, sandbox status, push URL |

---

## Rencana Pengembangan

### Prioritas Tinggi

- [ ] **Persistensi Memori** – simpan riwayat percakapan ke database (SQLite / Redis / PostgreSQL).
- [ ] **Confidence Threshold** – tolak atau minta klarifikasi jika confidence intent < 0.5.
- [ ] **Error Recovery LLM** – jika LLM menghasilkan JSON tidak valid, retry otomatis dengan prompt koreksi.
- [ ] **Unit & Integration Tests** – test suite untuk semua agent dan tools.
- [ ] **Autentikasi REST API** – API key atau JWT untuk mengamankan endpoint.

### Pengembangan Agent Baru

- [ ] **DocumentAgent** – summarize, ekstrak, atau analisis dokumen (PDF, DOCX) yang di-upload pengguna.
- [ ] **RAGAgent** – Retrieval-Augmented Generation dari basis pengetahuan internal (FAQ, SOP, dokumentasi produk).
- [ ] **ImageAnalysisAgent** – proses gambar dengan model vision; saat ini `image_query` ditangani ResponderAgent.
- [ ] **ReportAgent** – buat laporan progres berdasarkan data mandays aktual vs. rencana.
- [ ] **ScheduleAgent** – konversi WBS menjadi jadwal proyek dengan export PDF.

### Peningkatan WBSAgent & MandaysAgent

- [ ] **Template Proyek** – template WBS per jenis proyek (e-commerce, mobile app, ERP, dll).
- [ ] **Multi-turn Refinement** – pengguna bisa minta revisi WBS/mandays secara iteratif dalam satu sesi.
- [ ] **Export PDF** – opsi export selain Excel.
- [ ] **Input dari Dokumen** – pengguna upload briefing, agent buat WBS/mandays dari isinya.
- [ ] **Kalkulasi Biaya** – estimasi biaya per role berdasarkan rate yang dikonfigurasi.

### Pengembangan Interface

- [ ] **Web Dashboard** – frontend (React/Next.js) untuk monitor sesi, riwayat chat, dan output file.
- [ ] **WhatsApp Interface** – via Twilio atau WhatsApp Business API.
- [ ] **Slack / Discord Bot** – integrasi ke platform kolaborasi tim.
- [ ] **Voice Input** – speech-to-text sebelum diproses pipeline.

### Kualitas & Operasional

- [ ] **Rate Limiting** – batasi request per sesi untuk mencegah abuse.
- [ ] **Logging & Monitoring** – integrasi Sentry / Grafana / Prometheus untuk produksi.
- [ ] **Health Check Endpoint** – endpoint `/health` dengan status semua komponen.

---

## Konfigurasi & Variabel Lingkungan

| Variabel | Keterangan |
|----------|------------|
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram |
| `OPENROUTER_API_KEY` | API key untuk LLM via OpenRouter |
| `OPENROUTER_MODEL` | Model LLM yang digunakan |
| `TAVILY_API_KEY` | API key Tavily (opsional – research intent) |
| `WEBHOOK_URL` | URL publik untuk Telegram webhook (opsional) |
| `PORT` | Port server (default: 8000) |
| `GITHUB_PAT` | Personal Access Token GitHub (scope: `repo`). Kosong = pakai SSH key |
| `GIT_USER_NAME` | Nama penulis commit (default: `AdvanceAI Bot`) |
| `GIT_USER_EMAIL` | Email penulis commit (default: `bot@advanceai.local`) |
| `SANDBOX_REPOS_DIR` | Direktori clone repo lokal (default: `~/sandbox_repos`) |
| `SANDBOX_PYTHON_IMAGE` | Docker image fallback Dockerfile (default: `python:3.11-slim`) |
| `SANDBOX_TIMEOUT` | Timeout per perintah Docker dalam detik (default: `300`) |
| `SANDBOX_MAX_RETRIES` | Maks iterasi retry sandbox jika container gagal (default: `3`) |

---

## Dependensi Utama

| Package | Kegunaan |
|---------|----------|
| `python-telegram-bot` | Interface Telegram |
| `fastapi` + `uvicorn` | REST API server |
| `httpx` | HTTP client untuk panggilan LLM & Tavily |
| `pydantic` | Validasi skema data |
| `openpyxl` | Generate & parse file Excel |
| `tavily-python` | Live web search (opsional) |
| `python-dotenv` | Manajemen konfigurasi `.env` |


---

## Arsitektur Sistem

```
User (Telegram / REST API)
        │
        ▼
  GatekeeperAgent          ← klasifikasi intent
        │
        ▼
   AgentRouter             ← pilih agent yang sesuai
        │
        ├── ResponderAgent      → percakapan umum
        ├── ResearcherAgent     → riset mendalam + Tavily web search
        ├── ContentCreatorAgent → pembuatan konten platform (LinkedIn, dll)
        ├── WBSAgent            → WBS Gantt chart + export Excel
        ├── MandaysAgent        → estimasi mandays + export Excel
        └── DeveloperAgent      → clone repo → edit kode via LLM → Docker sandbox
```

Bot menggunakan sistem **multi-agent** berbasis LLM (via OpenRouter). Setiap pesan diklasifikasikan terlebih dahulu oleh GatekeeperAgent, lalu diteruskan ke agent spesialis yang paling sesuai.

---

## Kemampuan Saat Ini

### 1. Interface & Akses

| Interface      | Status | Keterangan |
|----------------|--------|------------|
| Telegram Bot   | ✅ Aktif | Polling & Webhook |
| REST API       | ✅ Aktif | FastAPI, endpoint `/chat` dan `/clear/{session_id}` |
| Webhook        | ✅ Aktif | Integrasi Telegram via webhook |

#### Perintah Telegram

| Perintah | Fungsi |
|----------|--------|
| `/start`  | Sapa pengguna, tampilkan intro bot |
| `/help`   | Tampilkan daftar perintah & kemampuan |
| `/ping`   | Cek status & latensi bot |
| `/reset`  | Hapus riwayat percakapan sesi aktif |

---

### 2. Klasifikasi Intent (GatekeeperAgent)

Bot secara otomatis mendeteksi maksud pesan pengguna dan meneruskannya ke agent yang tepat:

| Intent | Deskripsi | Agent yang Menangani |
|--------|-----------|----------------------|
| `general_inquiry` | Pertanyaan umum | ResponderAgent |
| `product_question` | Pertanyaan seputar produk | ResponderAgent |
| `complaint` | Keluhan pengguna | ResponderAgent |
| `order_status` | Status pesanan | ResponderAgent |
| `billing` | Pertanyaan tagihan/pembayaran | ResponderAgent |
| `unknown` | Intent tidak dikenali | ResponderAgent |
| `technical_support` | Masalah teknis & riset mendalam | ResearcherAgent |
| `image_query` | Pertanyaan terkait gambar/foto | ResearcherAgent |
| `data_analysis` | WBS, perencanaan proyek berbasis struktur | WBSAgent |
| `mandays_planning` | Estimasi mandays, effort, alokasi resource | MandaysAgent |

---

### 3. Agent Spesialis

#### ResponderAgent
- Menjawab percakapan umum, pertanyaan produk, keluhan, status order, dan billing.
- Menggunakan riwayat percakapan (last 10 pesan) untuk konteks yang relevan.
- Mendukung bahasa **Indonesia dan Inggris** secara otomatis.

#### ResearcherAgent
- Menangani pertanyaan teknis kompleks dengan pendekatan **step-by-step reasoning**.
- Memberikan jawaban komprehensif dengan analisis mendalam.
- Menggunakan riwayat percakapan (last 8 pesan) untuk konteks.
- Mendukung bahasa **Indonesia dan Inggris**.

#### WBSAgent
- Membuat **Work Breakdown Structure (WBS)** dalam format **Gantt chart** berdasarkan deskripsi pengguna.
- Output Excel menggunakan layout Gantt-style: timeline per hari kerja, sprint header, dan sel aktif berwarna per task.
- Menghasilkan **file Excel (.xlsx)** yang langsung dikirim ke pengguna via Telegram.
- Dipicu oleh intent `data_analysis` (kata kunci: WBS, breakdown structure, Gantt, timeline proyek).

#### MandaysAgent
- Membuat **rencana mandays** dan estimasi effort berdasarkan deskripsi proyek atau fitur pengguna.
- Fokus pada alokasi **sumber daya per role** dan estimasi waktu yang realistis.
- Menghasilkan **file Excel (.xlsx)** yang langsung dikirim ke pengguna via Telegram.
- Mendukung 13 role standar: `SA`, `TL`, `BA`, `SM`, `UI`, `DBA`, `BE1`, `BE2`, `FE1`, `FE2`, `QA`, `DevOps`, `TW`.
- Dipicu oleh intent `mandays_planning` (kata kunci: mandays, estimasi, effort, resource, person-days).

---

### 4. Tools Internal

Tools yang dipanggil oleh **orchestrator** (subclass `BaseTool`):

| Tool | Tipe | Kapan Dijalankan | Fungsi |
|------|------|------------------|--------|
| `TavilySearchTool` | Pre-agent | Sebelum ResearcherAgent | Live web search, hasilnya masuk `task.tool_results["tavily_search"]` |
| `WBSGeneratorTool` | Post-agent | Setelah WBSAgent selesai | Build Excel Gantt chart dari JSON di `task.metadata["wbs_json_data"]` |
| `MandaysGeneratorTool` | Post-agent | Setelah MandaysAgent selesai | Build Excel mandays dari JSON di `task.metadata["mandays_json_data"]` |

Tools internal **DeveloperAgent** (dikelola langsung oleh agent, **tidak** melalui pipeline orchestrator):

| Tool | File | Fungsi |
|------|------|--------|
| `CLIExecutor` | `src/tools/cli_executor.py` | Jalankan perintah shell non-interaktif (timeout 5 mnt), capture stdout+stderr |
| `SandboxRunner` | `src/tools/sandbox_runner.py` | Build & run Docker container; generate Dockerfile/compose fallback; deteksi traceback |
| `GitManager` | `src/tools/git_manager.py` | Konfigurasi identitas git, inject GITHUB_PAT ke URL, `git add -A → commit → push` |
| `RepoTracker` | `src/memory/repo_tracker.py` | SQLite registry repo yang pernah di-clone (data/repos.db) |

---

### 5. Memori & Sesi

- Setiap pengguna memiliki **sesi percakapan terpisah** berdasarkan `session_id` (Telegram user ID atau custom session dari REST API).
- Riwayat percakapan disimpan **in-memory** selama bot berjalan.
- Sesi dapat di-reset via perintah `/reset` (Telegram) atau endpoint `DELETE /clear/{session_id}` (REST API).

---

### 6. Output Khusus

- **Teks Markdown** – semua reply teks menggunakan format Markdown via Telegram.
- **File Excel (Gantt)** – WBSAgent mengirim file `.xlsx` berformat Gantt chart (timeline per hari kerja, sprint header, sel aktif berwarna).
- **File Excel (Mandays)** – MandaysAgent mengirim file `.xlsx` berformat tabel mandays per role per sprint dengan grand total.
- **Laporan Coding** – DeveloperAgent mengirim ringkasan teks: file yang diubah, commit hash, status sandbox Docker, dan URL push.

---

## Rencana Pengembangan

### Prioritas Tinggi

- [ ] **Persistensi Memori** – simpan riwayat percakapan ke database (SQLite / Redis / PostgreSQL) agar tidak hilang saat bot di-restart.
- [ ] **Upload Dokumen Pengguna** – izinkan pengguna upload file (PDF, DOCX, Excel) sebagai input konteks untuk WBSAgent atau ResearcherAgent.
- [ ] **Image Analysis Agent** – proses gambar yang dikirim pengguna dengan model vision (saat ini `image_query` diteruskan ke ResearcherAgent teks biasa).
- [ ] **Autentikasi REST API** – tambahkan API key atau JWT untuk mengamankan endpoint REST API.

### Pengembangan Agent Baru

- [ ] **ScheduleAgent** – buat jadwal proyek (Gantt chart) dari WBS yang sudah ada, output PDF atau Excel.
- [ ] **DocumentAgent** – summarize, ekstrak, atau analisis dokumen (PDF, DOCX) yang di-upload pengguna.
- [ ] **RAGAgent** – Retrieval-Augmented Generation dari basis pengetahuan internal (FAQ, SOP, dokumentasi produk).
- [ ] **ReportAgent** – buat laporan progres proyek berdasarkan data mandays aktual vs. rencana.
- [ ] **CalendarAgent** – integrasi Google Calendar / Outlook untuk buat event dari jadwal proyek.

### Pengembangan Interface

- [ ] **Web Dashboard** – tampilan frontend (React/Next.js) untuk memonitor sesi, riwayat chat, dan output file.
- [ ] **WhatsApp Interface** – tambahkan interface WhatsApp via Twilio atau WhatsApp Business API.
- [ ] **Slack / Discord Bot** – integrasikan ke platform kolaborasi tim.
- [ ] **Voice Input** – speech-to-text sebelum diproses oleh pipeline agent.

### Peningkatan Kualitas

- [ ] **Confidence Threshold** – tolak atau minta klarifikasi jika confidence intent terlalu rendah (< 0.5).
- [ ] **Multi-turn WBS Refinement** – izinkan pengguna memperbaiki/menambah detail WBS secara iteratif dalam satu sesi.
- [ ] **Error Recovery** – jika LLM menghasilkan JSON tidak valid, lakukan retry otomatis dengan prompt koreksi.
- [ ] **Unit & Integration Tests** – tambahkan test suite untuk semua agent dan tools.
- [ ] **Rate Limiting** – batasi jumlah request per sesi untuk mencegah abuse.
- [ ] **Logging & Monitoring** – integrasi dengan Sentry / Grafana / Prometheus untuk monitoring produksi.

### Pengembangan WBSAgent

- [ ] **Template Proyek** – sediakan template WBS per jenis proyek (e-commerce, mobile app, ERP, dll).
- [ ] **Export PDF** – tambahkan opsi export WBS ke format PDF selain Excel.
- [ ] **Kalkulasi Biaya** – hitung estimasi biaya per role berdasarkan rate yang dikonfigurasi.
- [ ] **Input dari File** – pengguna bisa upload briefing dokumen, lalu WBSAgent membuat WBS dari isinya.
- [ ] **Edit Interaktif** – setelah WBS di-generate, pengguna bisa minta revisi spesifik (tambah sprint, ubah durasi, dll).

---

## Konfigurasi & Variabel Lingkungan

| Variabel | Keterangan |
|----------|------------|
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram |
| `OPENROUTER_API_KEY` | API key untuk LLM via OpenRouter |
| `OPENROUTER_MODEL` | Model LLM yang digunakan (default: sesuai config) |
| `WEBHOOK_URL` | URL publik untuk Telegram webhook (opsional) |
| `PORT` | Port server (default: 8000) |

---

## Dependensi Utama

| Package | Kegunaan |
|---------|----------|
| `python-telegram-bot` | Interface Telegram |
| `fastapi` + `uvicorn` | REST API server |
| `httpx` | HTTP client untuk panggilan LLM |
| `pydantic` | Validasi skema data |
| `openpyxl` | Generate & parse file Excel |
| `python-dotenv` | Manajemen konfigurasi `.env` |
