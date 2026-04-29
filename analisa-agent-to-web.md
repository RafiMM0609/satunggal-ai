# Analisa Teknis: Migrasi / Ekstensi Agent ke Web UI

> **Konteks:** Sistem `satunggal-ai` saat ini adalah multi-agent AI yang berjalan di Telegram
> dan menyediakan REST API (FastAPI). Dokumen ini menganalisa kelayakan, kebutuhan teknis,
> pro/kontra, dan rencana pengembangan jika fitur-fitur agent ini diekspos ke satu atau lebih
> aplikasi web independen.

---

## 1. Arsitektur Saat Ini

```
┌──────────────────────────────────────────────────────────────┐
│                      Interface Layer                         │
│  ┌──────────────────┐        ┌──────────────────────────┐   │
│  │  Telegram Bot    │        │  REST API (FastAPI)       │   │
│  │  (telegram_bot.py│        │  POST /chat              │   │
│  │   + webhook.py)  │        │  DELETE /session/{id}    │   │
│  └────────┬─────────┘        └───────────┬──────────────┘   │
│           │                              │                   │
│           └──────────────┬───────────────┘                   │
│                          ▼                                   │
│              process_message(session_id, text)               │
│                  [main_loop.py]                              │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  GatekeeperAgent → TaskPlanner → AgentRouter → Agent.run()  │
│  Pre-tool loop → Post-tool loop → ManagerAgent review       │
└──────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
  ResearcherAgent   MandaysAgent    ContentCreatorAgent
  WBSAgent         DeveloperAgent  QuizAgent  ... (18+ agents)
```

### Komponen utama yang relevan untuk web:

| Komponen | File | Fungsi |
|---|---|---|
| REST API | `src/interfaces/rest_api.py` | Endpoint `/chat` & `/session` sudah ada |
| Auth | `src/interfaces/auth.py` | X-API-Key header, `secrets.compare_digest` |
| Orchestrator | `src/orchestrator/main_loop.py` | Single entry point `process_message()` |
| Persistent History | `src/memory/persistent_history.py` | SQLite, per session_id |
| MandaysAgent | `src/agents/mandays_agent/agent.py` | JSON → Excel |
| WBSAgent | `src/agents/wbs_agent/agent.py` | JSON → Gantt Excel |
| ContentCreatorAgent | `src/agents/content_creator/agent.py` | Platform-ready content JSON |
| ResearcherAgent | `src/agents/researcher/agent.py` | Tavily search + synthesis |

---

## 2. Pro / Kontra: Telegram Only vs Telegram + Web UI

### ✅ PRO membangun Web UI

| Dimensi | Keuntungan |
|---|---|
| **Aksesibilitas** | Pengguna tidak perlu install Telegram; bisa akses dari browser mana saja |
| **UX yang lebih kaya** | Tampilan tabel mandays, Gantt chart, preview konten, syntax highlighting kode langsung di browser |
| **File output** | Excel/PDF bisa ditampilkan preview atau langsung diunduh tanpa melewati batasan ukuran file Telegram (50 MB) |
| **Multi-tab / multi-session** | User bisa buka beberapa pekerjaan berbeda secara paralel di tab berbeda |
| **Branding** | Setiap produk bisa punya domain/brand sendiri (mandays.satunggal.ai, konten.satunggal.ai, dst) |
| **Onboarding lebih mudah** | Link langsung, tidak butuh username Telegram atau undangan group |
| **Integrasi 3rd party** | Mudah embed ke Notion, Confluence, internal tools via iframe atau OAuth |
| **Analitik** | Pasang Google Analytics / Posthog untuk memahami pola penggunaan per fitur |
| **Monetisasi** | Payment gateway, rate-limit per user/tier lebih mudah diimplementasi di web |

### ❌ KONTRA / Risiko

| Dimensi | Risiko |
|---|---|
| **Effort pengembangan** | Perlu bangun frontend (React/Vue/Next.js) untuk setiap web app |
| **Auth & User Management** | Telegram sudah handle identitas; web butuh sistem login sendiri (OAuth / JWT) |
| **Infrastruktur tambahan** | CDN, SSL cert, domain, reverse proxy per web app |
| **Duplikasi maintenance** | Bug fix harus dipastikan konsisten di Telegram dan semua web app |
| **File upload** | Telegram handle upload PDF/DOCX otomatis; web perlu `multipart/form-data` endpoint baru |
| **Streaming response** | Telegram polling cocok untuk long-running; web butuh SSE / WebSocket untuk UX real-time |
| **Keamanan CORS** | REST API saat ini tidak ada CORS config; perlu ditambahkan untuk browser request |
| **State management** | Session Telegram dikelola bot; web perlu token/cookie per browser tab |

---

## 3. Kebutuhan Teknis Jika AI Ini Digunakan di Website

### 3.1 Perubahan Backend (Python/FastAPI)

#### a) CORS Middleware
```
# Saat ini: tidak ada CORS → browser akan block request cross-origin
# Perlu ditambahkan di rest_api.py:
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=[...], ...)
```

#### b) Streaming Response (Server-Sent Events)
```
Saat ini: POST /chat → tunggu → balik full reply (bisa 10-30 detik)
Perlu tambah: GET /chat/stream → EventSourceResponse (SSE)
              atau WebSocket endpoint /ws/chat
```

Tanpa streaming, user melihat layar kosong sampai AI selesai berpikir → UX buruk.

#### c) File Upload Endpoint
```
Saat ini: file diterima via Telegram file_id
Perlu tambah: POST /upload  (multipart/form-data)
              → simpan sementara → inject ke session context
              → mendukung PDF, DOCX, gambar
```

#### d) User Authentication & Session Management
```
Saat ini: session_id = Telegram chat_id (tidak ada password)
Perlu: - JWT atau session token (expire per X jam)
       - Endpoint: POST /auth/login, POST /auth/register, POST /auth/refresh
       - Middleware validasi token per request
       - Opsional: OAuth (Google / GitHub) untuk SSO
```

#### e) Rate Limiting per User
```
Saat ini: tidak ada per-user rate limit
Perlu: slowapi atau custom middleware
       - Free tier: N request/hari
       - Pro tier: unlimited
```

#### f) Endpoint Khusus per Fitur Web
```
Saat ini: semua lewat POST /chat (natural language)
Tambahan opsional:
  POST /mandays/generate  → payload terstruktur → skip gatekeeper
  POST /wbs/generate
  POST /content/generate
  GET  /mandays/{job_id}/excel  → download file
```

### 3.2 Infrastruktur

| Komponen | Kebutuhan | Opsi |
|---|---|---|
| Domain | 1 per web app | Cloudflare, Namecheap |
| SSL | Wajib (HTTPS) | Let's Encrypt / Cloudflare proxy |
| Reverse Proxy | Satu server bisa host semua | Nginx, Caddy, Traefik |
| File Storage | Output Excel/PDF sementara | Local `/tmp` + cleanup job, atau S3 |
| CDN | Aset statik frontend | Cloudflare, Vercel, Netlify |
| Deploy Backend | Sama dengan Telegram bot | VPS yang sudah ada, atau Railway/Render |

### 3.3 Frontend Stack (per web app)

**Minimal (cepat dibangun):**
- Next.js 14 App Router + Tailwind CSS + shadcn/ui
- Fetch ke REST API + SSE untuk streaming
- Simpan session token di `localStorage` / httpOnly cookie

**Fitur frontend yang perlu dibangun:**
- [ ] Chat interface dengan history (mirip ChatGPT)
- [ ] File upload drag-and-drop
- [ ] Tombol download untuk output Excel/PDF
- [ ] Preview tabel/Gantt chart di browser
- [ ] Indikator loading / skeleton saat AI berpikir
- [ ] Toast notification untuk error

---

## 4. Rencana Multi-Web: 3–4 Aplikasi Web Terpisah

Daripada satu web "serba bisa", lebih efektif membuat beberapa web **focused** yang masing-masing
menyelesaikan satu masalah spesifik. Backend tetap **satu** (`satunggal-ai`), web berbeda hanya
memanggil endpoint/intent yang relevan.

```
                    ┌────────────────────────┐
                    │    satunggal-ai        │
                    │    (Backend Tunggal)   │
                    │    FastAPI + Agents    │
                    └──────────┬─────────────┘
                               │  REST API
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼                    ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │ Web App #1    │  │ Web App #2    │  │ Web App #3    │  │ Web App #4    │
  │ Mandays &     │  │ Konten &      │  │ Developer     │  │ Research &    │
  │ WBS Generator │  │ Copywriting   │  │ Assistant     │  │ Briefing Hub  │
  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

---

### Web App #1 — Mandays & WBS Generator

**URL idea:** `mandays.satunggal.ai` / `wbs.satunggal.ai`

**Target pengguna:** Project Manager, BA, Scrum Master

**Fitur:**
- Form input: nama proyek, deskripsi fitur, jumlah sprint, roles yang terlibat
- Generate WBS Gantt Chart (Excel) → preview tabel di browser → download
- Generate Mandays Plan (Excel) → tabel sprint-by-sprint → download
- Simpan riwayat project (per user login)
- Tombol "Regenerate" / "Edit prompt" tanpa ketik ulang

**Agent yang dipakai:** `wbs_agent`, `mandays_agent`

**Kompleksitas Frontend:** ⭐⭐ (form + tabel preview + download)

**Nilai bisnis:** ✅ Tinggi — langsung menggantikan manual spreadsheet

---

### Web App #2 — AI Content & Copywriting Studio

**URL idea:** `konten.satunggal.ai` / `copy.satunggal.ai`

**Target pengguna:** Marketer, Social Media Manager, Content Writer

**Fitur:**
- Input topik / brief → AI generate konten siap posting
- Pilih platform: LinkedIn, Twitter/X, Blog, Instagram caption
- Tone selector: formal, casual, Gen-Z, profesional
- Preview format sesuai platform (character count LinkedIn, hashtag Twitter)
- 1-klik copy to clipboard
- Riwayat draft konten per sesi
- Riset otomatis sebelum menulis (integrasikan `researcher_agent`)

**Agent yang dipakai:** `content_creator`, `researcher`

**Kompleksitas Frontend:** ⭐⭐⭐ (rich text preview, platform switcher, tone config)

**Nilai bisnis:** ✅ Tinggi — pasar content creator sangat besar

---

### Web App #3 — Developer Assistant & Code Tools

**URL idea:** `dev.satunggal.ai` / `code.satunggal.ai`

**Target pengguna:** Software Engineer, Tech Lead

**Fitur:**
- Chat interface untuk tanya jawab kode (seperti GitHub Copilot Chat)
- Upload repo URL (GitHub/GitLab) → inspect, QnA, root cause analysis
- Generate kode dari deskripsi → syntax highlighted preview
- Code fix: paste snippet + deskripsi bug → AI perbaiki
- Generate dokumen teknis (README, API doc) dari repo
- Sandbox output (tampilkan log eksekusi)

**Agent yang dipakai:** `developer`, `developer_inspector`, `developer_qna`, `code_fix`, `technical_writer`

**Kompleksitas Frontend:** ⭐⭐⭐⭐ (code editor, syntax highlight, diff viewer, file tree)

**Nilai bisnis:** ✅ Menengah-Tinggi — butuh diferensiasi dari Copilot / Cursor

---

### Web App #4 — Research & Intelligence Hub

**URL idea:** `riset.satunggal.ai` / `intel.satunggal.ai`

**Target pengguna:** Analis, Peneliti, Eksekutif yang butuh briefing

**Fitur:**
- Input topik → AI riset web (Tavily) → ringkasan terstruktur dengan sumber
- Daily briefing: subscribe topik, terima email / notifikasi setiap hari
- Upload PDF → ringkasan + QnA interaktif
- Upload DOCX → audit kualitas dokumen + saran perbaikan
- Export hasil riset ke PDF / Word
- Dashboard riwayat riset per topik

**Agent yang dipakai:** `researcher`, `pdf_summarizer`, `doc_agent`, `technical_writer`

**Kompleksitas Frontend:** ⭐⭐⭐ (chat + PDF viewer + export)

**Nilai bisnis:** ✅ Tinggi — niche tapi willing to pay (B2B)

---

## 5. Roadmap Implementasi yang Disarankan

### Fase 1 – Perkuat Backend (1–2 minggu)
- [ ] Tambahkan CORS middleware di `rest_api.py`
- [ ] Tambahkan SSE / streaming endpoint `GET /chat/stream`
- [ ] Tambahkan `POST /upload` untuk file PDF/DOCX
- [ ] Tambahkan rate limiting per API key
- [ ] Tambahkan endpoint khusus `POST /mandays/generate` dan `POST /wbs/generate`
- [ ] Konfigurasi file serving untuk output Excel/PDF (`GET /files/{file_id}`)

### Fase 2 – Web App #1: Mandays & WBS (2–3 minggu)
- [ ] Setup Next.js project + Tailwind + shadcn/ui
- [ ] Form input proyek → call API → download Excel
- [ ] Preview tabel mandays di browser (react-table)
- [ ] Preview Gantt chart (ag-grid atau custom SVG)
- [ ] Auth minimal (API key per organisasi, tanpa login)
- [ ] Deploy ke Vercel / VPS dengan subdomain

### Fase 3 – Web App #2: Konten Studio (2–3 minggu)
- [ ] UI input brief + platform selector + tone selector
- [ ] Streaming response (SSE) → teks muncul real-time
- [ ] Preview per platform (LinkedIn card, Twitter thread, dll)
- [ ] Copy-to-clipboard + riwayat draft
- [ ] Auth: simple login (email + password / Google OAuth)

### Fase 4 – Web App #3 & #4 (4–6 minggu)
- [ ] Developer Assistant: code editor (Monaco), diff viewer
- [ ] Research Hub: PDF viewer, export PDF/Word

---

## 6. Keputusan Arsitektur Penting

### Backend: Satu atau Banyak?
**Rekomendasi: tetap satu backend** (`satunggal-ai`), semua web memanggil REST API yang sama.
- Tidak ada duplikasi agent/logic
- Satu deployment untuk semua fitur
- Cukup tambahkan CORS whitelist per domain web

### Auth Strategy
| Opsi | Cocok untuk | Trade-off |
|---|---|---|
| API Key per organisasi | B2B, minimal friction | Tidak ada per-user tracking |
| Email + Password (JWT) | Consumer app | Perlu bangun auth flow |
| Google OAuth | Consumer app, UX terbaik | Perlu Google Cloud project |
| Magic Link (email) | No-password UX | Perlu email provider (Resend/SendGrid) |

**Rekomendasi awal:** API Key per organisasi untuk Web #1 (Mandays), lalu tambah Google OAuth
untuk Web #2 (Konten Studio) karena target audiensnya lebih consumer.

### Streaming: SSE vs WebSocket
| | SSE (Server-Sent Events) | WebSocket |
|---|---|---|
| Implementasi backend | Mudah (FastAPI StreamingResponse) | Lebih kompleks |
| Implementasi frontend | `EventSource` API native | perlu library |
| Reconnect otomatis | ✅ Built-in | Manual |
| Bidirectional | ❌ Satu arah | ✅ |
| Cocok untuk | Chat streaming | Real-time kolaborasi |

**Rekomendasi:** SSE untuk semua web app (cukup untuk use case chat).

### File Output
```
Flow saat ini (Telegram):
  Agent → Excel file di /tmp → Telegram sendDocument()

Flow untuk Web:
  Agent → Excel file di /tmp/web_outputs/{job_id}.xlsx
         → Response: { "job_id": "abc123", "download_url": "/files/abc123" }
         → Frontend: tombol download
         → Cleanup job: hapus file setelah 1 jam
```

---

## 7. Estimasi Effort & Prioritas

| Web App | Backend Tambahan | Frontend | Total Effort | Prioritas |
|---|---|---|---|---|
| #1 Mandays & WBS | 3–5 hari | 8–12 hari | **~3 minggu** | 🥇 Tinggi |
| #2 Konten Studio | 2–3 hari | 10–14 hari | **~3 minggu** | 🥈 Tinggi |
| #3 Developer Assistant | 5–7 hari | 14–21 hari | **~5 minggu** | 🥉 Menengah |
| #4 Research Hub | 3–5 hari | 10–14 hari | **~3 minggu** | 🥉 Menengah |

> Catatan: estimasi untuk 1 developer fullstack dengan familiar pada codebase ini.

---

## 8. Kesimpulan

1. **Secara teknis, transisi ke web sangat feasible** — REST API (`/chat`) sudah ada dan berfungsi.
   Hambatan utama hanya di layer UX (streaming, file download, auth) yang membutuhkan effort
   terbatas di backend.

2. **Multi-web app adalah pendekatan yang tepat** dibanding satu web "serba bisa".
   Setiap web punya target pengguna spesifik, branding tersendiri, dan lebih mudah di-marketing.
   Backend tetap satu — hemat infrastruktur.

3. **Prioritas pertama: Web Mandays & WBS** — paling konkret, output-nya (Excel) mudah
   di-demo, dan paling mudah dimonetisasi (bayar per download / per bulan).

4. **Perubahan backend minimal yang wajib dilakukan sebelum launch web apapun:**
   - CORS middleware
   - SSE streaming endpoint
   - File serving endpoint
   - Rate limiting

5. **Telegram tetap jalan paralel** — tidak ada yang perlu diubah di Telegram bot.
   Web app hanya menambahkan satu channel distribusi baru ke orchestrator yang sama.
