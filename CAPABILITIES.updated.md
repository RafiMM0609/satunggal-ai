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

## Item Terbaru (2026-03-12)

- **Persistensi Memori (implemented):** Riwayat sesi kini disimpan ke SQLite sebagai opsi permanen (fallback ke in-memory bila DB tidak tersedia).
- **Autentikasi REST API (beta):** Endpoint `/chat` mendukung API key via header `X-API-Key` untuk akses terkontrol.
- **Health Check Endpoint:** Endpoint `/health` menampilkan status komponen inti (LLM, Telegram, DB, dan queue tools).
- **WBSAgent — Export PDF:** Opsi export ke PDF ditambahkan selain file Excel `.xlsx` untuk kemudahan sharing.
- **MandaysAgent — Kalkulasi Biaya:** Penambahan perhitungan biaya berdasarkan rate per role; output menyertakan ringkasan biaya.
- **Unit Tests (partial):** Test dasar ditambahkan untuk `CLIExecutor`, `GitManager`, dan `RepoTracker` (lokal/CI).
- **DeveloperAgent — RepoTracker ke SQLite:** Repo registry dimigrasi ke SQLite untuk keandalan dan query lebih cepat.
- **TavilySearchTool — Caching:** Hasil pencarian dicache (TTL 24 jam) dan disertakan `context_text` siap inject ke LLM.
- **SandboxRunner — Improved retries:** Perbaikan deteksi traceback dan retry otomatis (maks 3 percobaan), plus batasan resource dasar.
