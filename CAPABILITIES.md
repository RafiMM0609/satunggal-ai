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
 # AdvanceAI — Ringkasan Kemampuan & Roadmap (terbarui)

 Dokumen ini merangkum kemampuan yang saat ini diimplementasikan di repo dan rencana pengembangan singkat.

 **Referensi teknis:** lihat `APP_LOGIC_WORKFLOW.md` untuk detail alur agent dan `src/` untuk implementasi.

 ---

 **Arsitektur singkat**

 User (Telegram / REST API) → GatekeeperAgent → AgentRouter → Specialist Agent → Orchestrator (tools) → Interface (kirim teks/file)

 - Sistem multi-agent berbasis LLM (OpenRouter client). Semua panggilan tool dikelola oleh orchestrator; agent mengisi `AgentTask` (blackboard).

 ---

 **Ringkasan kemampuan saat ini**

 - Interface: Telegram Bot (polling & webhook) dan REST API (`/chat`, `/clear/{session_id}`) aktif.
 - Intent classification: `GatekeeperAgent` memilih agent spesialis berdasarkan intent pesan.
 - Bahasa: mendukung bahasa Indonesia dan Inggris.

 **Agent utama yang tersedia (implementasi di `src/agents/`)**
 - `ResponderAgent`: percakapan umum, support ringan, keluhan, billing.
 - `ResearcherAgent`: riset mendalam; dapat menggunakan `TavilySearchTool` untuk hasil web real-time.
 - `ContentCreatorAgent`: buat konten terstruktur (hook/body/cta/hashtags) untuk platform digital.
 - `WBSAgent`: hasilkan WBS → JSON → `WBSGeneratorTool` → Excel Gantt (.xlsx).
 - `MandaysAgent`: hasilkan estimasi mandays → `MandaysGeneratorTool` → Excel (.xlsx).
 - `DeveloperAgent`: alur coding end‑to‑end (clone/pull, edit via LLM, sandbox run, commit & push).

 **Intent penting yang dikenali**
 - `research`, `content_creation`, `data_analysis`, `mandays_planning`, `code_development`, plus intent percakapan umum (`general_inquiry`, `technical_support`, dll.).

 ---

 **Tools & utilitas (lokasi di `src/tools/`)**
 - Pre/Post-agent tools: `TavilySearchTool`, `WBSGeneratorTool`, `MandaysGeneratorTool`.
 - Developer-agent tools (internal): `CLIExecutor`, `SandboxRunner`, `GitManager`.
 - Utility: extractor untuk WBS/mandays di `src/tools/wbs/` dan `src/tools/mandays/`.

 ---

 **DeveloperAgent — alur singkat**
 1) Parse instruksi → ekstrak repo_url + task
 2) Clone/Pull repos (inject `GITHUB_PAT` bila perlu) → simpan di `RepoTracker`
 3) Cek environment (Dockerfile/docker-compose) → buat fallback bila perlu
 4) Edit kode via LLM → tulis patch ke disk
 5) Jalankan sandbox (docker compose) → jika error kirim log ke LLM → retry (max 3x)
 6) Commit & push (git)

 ---

 **Output yang dikirim ke pengguna**
 - Teks berbentuk Markdown (Telegram MarkdownV2)
 - File Excel `.xlsx` untuk WBS dan Mandays
 - Laporan coding (file changed, commit hash, status sandbox, push URL)

 ---

 **Konfigurasi & variabel lingkungan penting**
 - `TELEGRAM_BOT_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `TAVILY_API_KEY`, `GITHUB_PAT`, `SANDBOX_REPOS_DIR`, `PORT`, dll.

 ---

 **Dependensi inti**
 - `python-telegram-bot`, `fastapi` + `uvicorn`, `httpx`, `pydantic`, `openpyxl`, `python-dotenv` (lihat `requirements.txt`).

 ---

 **Rencana pengembangan (prioritas tinggi)**
 - Persistensi memori (simpan riwayat ke DB)
 - Confidence threshold untuk klasifikasi intent
 - Error recovery untuk LLM-generated JSON
 - Unit & integration tests
 - Autentikasi pada REST API

 Untuk detail tugas implementasi, saya bisa memecah ke TODO pekerjaan atau memperbarui bagian roadmap sesuai prioritas Anda.

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
