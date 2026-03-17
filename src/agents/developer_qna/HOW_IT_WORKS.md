# Cara Kerja DeveloperQnAAgent

## Ringkasan

`DeveloperQnAAgent` adalah agen tanya-jawab (Q&A) berbasis repositori yang menjawab pertanyaan spesifik tentang isi codebase — bukan untuk menemukan bug. Agen ini memahami konten repositori dan dapat menjawab pertanyaan seperti:

- "Ada API apa saja di repo ini?"
- "Tech stack apa yang dipakai?"
- "Bagaimana alur utama aplikasi?"
- "Jelaskan fungsi `HandleDownload`"
- "Model data apa yang ada?"

**Perbedaan utama dengan `DeveloperInspectorAgent`:**

| Aspek | DeveloperQnAAgent | DeveloperInspectorAgent |
|---|---|---|
| Tujuan | Menjawab pertanyaan tentang ISI repo | Menemukan BUG & root cause |
| Output | Jawaban singkat dan faktual | Laporan inspeksi penuh + critic pass |
| Trigger | "ada API apa", "jelaskan fungsi X" | "kenapa error", "ada bug di" |

---

## Arsitektur & Komponen

```
DeveloperQnAAgent (src/agents/developer_qna/agent.py)
    │
    ├── RepoAgentBase     (src/agents/repo_agent_base.py)
    │     ├── Repo clone/pull       → ~/sandbox_repos/<repo-name>
    │     ├── Branch checkout       → git checkout <branch>
    │     ├── RAG (file relevan)    → TF-IDF relevance ranking
    │     ├── Dir tree              → find . -not (skip-dirs)
    │     ├── Tavily search         → opsional web context
    │     └── LLM extraction        → parse repo_url, problem, branch
    │
    └── repo_qa.py        (src/tools/repo_qa.py)
          ├── classify_intent()     → tentukan topik
          ├── extract_api_endpoints()
          ├── extract_tech_stack()
          ├── extract_data_models()
          ├── extract_dependencies()
          ├── extract_ci_cd()
          ├── extract_security()
          ├── extract_main_flow()
          └── extract_specific_symbol()
```

---

## Alur Kerja Lengkap (Step-by-Step)

### Jalur Normal (dengan URL repo)

```
User Input
    │
    ▼
[1] Cek pending branch confirmation
    │ (ada?) → lanjut ke _run_qa_flow dengan branch yang dikonfirmasi
    │ (tidak ada?) ↓
    ▼
[2] classify_intent()  ← regex-based, cepat
    │  Hasilkan: API_ENDPOINTS | TECH_STACK | DATA_MODELS |
    │            DEPENDENCIES | CI_CD | SECURITY | MAIN_FLOW |
    │            SPECIFIC_SYMBOL | FULL_INSPECTION
    ▼
[3] _extract_request()  ← LLM call
    │  Parse: repo_url, problem, keywords, branch, verbosity
    ▼
[4] _resolve_repo()
    │  Clone atau pull repo → ~/sandbox_repos/<repo-name>
    │  (jika tidak ada URL → jawab dari deskripsi saja)
    ▼
[5] Branch selection
    │  a. Branch sudah ditentukan → checkout langsung → _run_qa_flow
    │  b. Tidak ada branch → deteksi branch aktif → tanya user
    │     → simpan ke _qna_pending_confirmations[session_id]
    ▼
[6] _run_qa_flow()  ← inti Q/A
    │
    ├──── Resolve symbol target (untuk SPECIFIC_SYMBOL intent)
    │       Cek apakah ada target di pesan, atau inherit dari session sebelumnya
    │
    ├──── asyncio.gather() — semua dijalankan PARALEL:
    │       ├── run_qa_extraction()      → ekstrak evidence topik-spesifik
    │       ├── _read_relevant_files()   → RAG: file paling relevan
    │       ├── _fetch_tavily_context()  → opsional: pencarian web
    │       └── _get_dir_tree()          → struktur direktori repo
    │
    ├──── Bangun evidence dict dari semua hasil
    │       (filter: untuk pertanyaan "cara kerja" + SPECIFIC_SYMBOL,
    │        skip RAG & dir-tree untuk mengurangi noise)
    │
    ├──── Jika pertanyaan "cara kerja"/"logika bisnis":
    │       Tambahkan preamble instruksi anti-code-dump ke LLM
    │       Cap evidence pada 8.000 karakter
    │
    ├──── LLM call (temperature=0.15, top_p=0.90, max_tokens=16384)
    │       System prompt: _QA_SYSTEM_PROMPT (anti-halusinasi)
    │       User message: pertanyaan + panduan verbositas + evidence
    │
    └──── mark_done() dengan jawaban + branch info + performance footer
         Simpan session context untuk follow-up
```

---

## Detail Setiap Langkah

### Langkah 1 — Cek Pending Branch Confirmation

Setiap kali ada permintaan Q/A, agen pertama-tama memeriksa apakah ada entri di dictionary `_qna_pending_confirmations[session_id]`. Ini terjadi ketika pada request sebelumnya user tidak menyebutkan branch — agen mendeteksi branch aktif dan menunggu konfirmasi.

Jika ada pending:
- Input user dicek dengan `_resolve_branch_from_reply()`.
- Kata seperti "ya", "lanjutkan", "ok", "lanjut" → gunakan branch yang terdeteksi.
- Input berupa nama branch valid (e.g., `develop`) → gunakan nama itu.
- Selain itu → fall through ke alur parse normal.

### Langkah 2 — Klasifikasi Intent (`classify_intent`)

Fungsi ini masih menggunakan regex sebagai jalur cepat (deterministik dan hampir nol-latensi), tetapi sekarang memakai pendekatan hybrid: **regex pertama, LLM fallback** hanya ketika regex gagal mengklasifikasikan (yaitu ketika regex mengembalikan `FULL_INSPECTION` sebagai fallback).

Rangka kerjanya singkat:

1. Jalur cepat: regex mengenali pola kuat seperti qualified identifiers (`controllers.DownloadFile`), API paths (`/upload`, `/api/v1/...`), kata kunci topik (`ci/cd`, `dependencies`, `tech stack`), file mentions (`main.py`, `controllers/user.go`), imperatives (`berikan/tampilkan isi dockerfile`), dan existence questions (`adakah/apakah ada endpoint ...`).
2. Jika regex menemukan kecocokan yang jelas → kembalikan intent sesuai `_INTENT_RULES` (mis. `CI_CD`, `API_ENDPOINTS`, `SPECIFIC_SYMBOL`, dll.).
3. Jika regex TIDAK menemukan kecocokan (default: `FULL_INSPECTION`), agen membuat satu panggilan LLM singkat (deterministik, 0 temperature, ~20 token output) untuk menebak intent sebenarnya. Ini memperbaiki kasus bahasa alamiah yang tidak terduga tanpa menambah latensi pada jalur yang umum.

Perubahan penting pada pola regex:
- Menangkap permintaan tentang `Dockerfile` / `docker-compose` (mis. "berikan script dockerfile").
- Menangkap permintaan "isi file" dan path file (mis. "jelaskan isi file main.py", "lihat controllers/user.go").
- Menangkap pertanyaan keberadaan fitur/handler (mis. "adakah handle upload file pada repository ini").
- Menangkap imperative verbs sebelum nama objek (mis. "tampilkan fungsi X", "berikan kode fungsi Y").

Keuntungan: jalur umum tetap cepat karena regex; pertanyaan ambigu diklasifikasikan lebih akurat oleh LLM tanpa memperlambat kasus umum.

> **Catatan:** `FULL_INSPECTION` dari Q/A agent artinya pertanyaan disambungkan ke `DeveloperInspectorAgent` jika memang terkait bug/error, atau diproses oleh Q/A secara fall-through ketika cocok.

### Langkah 3 — Ekstraksi Terstruktur via LLM (`_extract_request`)

LLM dipanggil dengan prompt yang meminta output JSON:

```json
{
  "repo_url":  "https://github.com/user/repo",
  "problem":   "deskripsi singkat pertanyaan",
  "keywords":  ["keyword1", "keyword2"],
  "branch":    "main"
}
```

Konversi ke `RepoExtractionRequest` (Pydantic model) yang juga berisi:
- `verbosity`: "detailed" atau "concise" (dari kata kunci di pesan user)
- `candidate_route_filenames`: nama file routing yang disebutkan user

**Session context inheritance:** Jika user tidak menyebut `repo_url`/`branch`, prompt menyertakan riwayat percakapan sebelumnya sehingga LLM dapat mengisi dari konteks yang sudah ada.

### Langkah 4 — Resolusi Repo (`_resolve_repo`)

Repo dikloning ke `~/sandbox_repos/<repo-name>/`. Jika sudah ada, dilakukan `git pull`. PAT (Personal Access Token) di-inject ke URL untuk akses private repo (GitHub PAT / GitLab PAT dari settings).

Jika tidak ada URL sama sekali → agen tetap menjawab berdasarkan deskripsi user saja, dengan catatan warning bahwa tidak ada data repo yang diakses.

### Langkah 5 — Branch Selection

- **Branch disebutkan** di request → langsung `git checkout <branch>` dan lanjut ke `_run_qa_flow`.
- **Branch tidak disebutkan** → agen membaca branch aktif (`git rev-parse --abbrev-ref HEAD`) dan meminta konfirmasi user sebelum melanjutkan.

Strategi checkout (5 percobaan):
1. `git checkout <branch>` biasa
2. Recover dari konflik/unmerged (`git rebase --abort` + `git reset --hard`) lalu retry
3. Force checkout jika branch ada lokal: `git checkout -f <branch>`
4. Fetch semua remote lalu checkout: `git checkout -b <branch> origin/<branch>`
5. Jika branch sudah ada lokal setelah fetch: `git checkout -f <branch>`

### Langkah 6 — Q/A Flow Utama (`_run_qa_flow`)

#### 6a. Resolve Symbol Target

Berlaku hanya untuk intent `SPECIFIC_SYMBOL`. Fungsi `extract_specific_target()` mengekstrak nama target dari input user dengan prioritas:

`extract_specific_target()` telah disempurnakan untuk menangani lebih banyak variasi alami (file mentions, prepositional phrases, imperative verbs) dan untuk menghindari false-positive dari parsing path/file order. Prioritas pengecekan sekarang adalah:

1. **Qualified identifier** (`controllers.DownloadFile`) — regex yang menangkap `pkg.CamelCase`.
2. **Keyword + intermediary** (mis. "jelaskan fungsi X", "tampilkan kode fungsi X") — menangkap nama fungsi/kelas/handler sebelum pengecekan file/path sehingga kasus seperti "jelaskan method handle_request di agent.py" mengekstrak `handle_request` (bukan `agent.py`).
3. **File mention / filename.extension** (mis. `main.py`, `controllers/user.go`) — mendukung multi-segment paths sehingga `controllers/user.go` diekstrak utuh.
4. **API path** (mis. `/upload`, `/api/v1/:id`) — deteksi path HTTP.
5. **General "jelaskan X" / existence questions** — menangkap target hingga 3 kata dan melewatkan preposisi umum (mis. "fungsi untuk login" → `login`).

Jika tidak ditemukan target → **inherit dari session sebelumnya** (`ctx["last_symbol_target"]`), berguna untuk pertanyaan follow-up seperti "bisa detailkan logika bisnis di api ini?".

**Dual-trace (pertanyaan gabungan):** Ketika target adalah qualified name (e.g., `controllers.DownloadFile`) **dan** `user_input` juga mengandung API path berparameter (e.g., `GET /appuuid/:uuid/:processoption/:outputtype`), `extract_specific_symbol()` menjalankan **dua trace secara paralel** via `asyncio.gather()`:
- `_trace_api_route(repo_path, api_path)` → cari registrasi route + handler body
- `_find_symbol_definition(repo_path, "DownloadFile", package_hint="controllers")` → cari definisi fungsi di direktori yang sesuai

Sebelumnya, hanya qualified name yang ditelusuri (karena `extract_specific_target()` mengembalikan satu target saja), sehingga registrasi route tidak pernah ditemukan saat pertanyaan mengandung keduanya.

#### 6b. Pengumpulan Evidence (Paralel)

Empat operasi dijalankan secara bersamaan dengan `asyncio.gather()`:

| Operasi | Fungsi | Hasil | Disertakan untuk SPECIFIC_SYMBOL? |
|---|---|---|---|
| Topic extraction | `run_qa_extraction()` | Evidence utama sesuai intent | ✅ selalu |
| RAG | `_read_relevant_files()` | File-file paling relevan (TF-IDF) | ❌ dilewati |
| Tavily | `_fetch_tavily_context()` | Konteks web opsional | ✅ jika relevan |
| Dir tree | `_get_dir_tree()` | Struktur folder repo | ✅ kecuali explanation |

**Optimasi untuk `SPECIFIC_SYMBOL`:** RAG dilewati untuk **semua** pertanyaan `SPECIFIC_SYMBOL`, bukan hanya yang bersifat explanation. Alasannya: symbol tracer (`_trace_api_route` / `_find_symbol_definition`) sudah memindai seluruh repo secara langsung; RAG TF-IDF sering menarik file yang tidak relevan hanya karena kesamaan kata (contoh: kata "download" cocok dengan handler WhatsApp yang tidak ada hubungannya), sehingga LLM menyimpulkan "data tidak cukup".

- **SPECIFIC_SYMBOL + non-explanation:** dir-tree tetap disertakan agar LLM tahu struktur direktori.
- **SPECIFIC_SYMBOL + explanation** (terdeteksi via `_EXPLANATION_Q_RE`): dir-tree juga dilewati untuk memaksimalkan fokus pada kode spesifik.
- **Intent lain** (bukan SPECIFIC_SYMBOL): RAG dan dir-tree disertakan seperti biasa.

#### 6c. Penyusunan User Message ke LLM

Dua mode:

**Mode penjelasan** (terdeteksi via `_EXPLANATION_Q_RE` — kata seperti "cara kerja", "logika bisnis", "alur api"):
- Ditambahkan preamble instruksi wajib ke LLM agar menjawab dalam **prosa bahasa natural**, bukan menyalin kode.
- Evidence di-cap pada **8.000 karakter** untuk menghindari LLM "tersesat" dalam kode.
- Verbosity note: "Jelaskan secara LENGKAP tapi dalam prosa bahasa natural, bukan kode."

**Mode reguler:**
- Verbosity note menyesuaikan `req.verbosity`: "singkat" (concise) atau "step-by-step" (detailed).
- Evidence dikirim sepenuhnya ke LLM.

#### 6d. LLM Call

```python
_llm.chat(
    messages=[
        {"role": "system", "content": _QA_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ],
    temperature=0.15,   # sangat deterministik
    top_p=0.90,
    max_tokens=16384,
)
```

**System prompt anti-halusinasi** memastikan LLM:
- Hanya menjawab berdasarkan data yang diberikan.
- Setiap poin harus disertai sumber `file:baris`.
- Menggunakan label: 🟢 `[CONFIRMED]`, 🟡 `[LIKELY]`, 🔴 `[UNVERIFIED]`.
- Dilarang mengarang detail atau menampilkan ulang kode secara verbatim.

#### 6e. Format Jawaban Final

Jawaban LLM dibungkus dengan:
- Header branch: `🌿 **Branch:** \`main\``
- Footer performa: `⏱️ 📡 API Endpoints · 3.2s (ekstraksi: 1.8s)`

Session context disimpan untuk follow-up: `repo_url`, `branch`, `candidate_route_filenames`, `last_symbol_target`.

---

## Extractor Topik-Spesifik (`run_qa_extraction`)

Setiap intent dipetakan ke satu extractor di `repo_qa.py`:

| Intent | Extractor | Strategi Ekstraksi |
|---|---|---|
| `API_ENDPOINTS` | `extract_api_endpoints()` | Cari file OpenAPI/Swagger → grep pola route decorator → baca file routing (urls.py, routes.go, dll.) |
| `TECH_STACK` | `extract_tech_stack()` | Baca requirements.txt/package.json/go.mod → grep import framework → cari Dockerfile |
| `DATA_MODELS` | `extract_data_models()` | Grep pattern ORM (SQLAlchemy, Pydantic, GORM, Sequelize) → baca file models/schemas |
| `DEPENDENCIES` | `extract_dependencies()` | Baca requirements.txt, package.json, pyproject.toml, go.mod, Gemfile, composer.json |
| `CI_CD` | `extract_ci_cd()` | Baca .github/workflows/*.yml, .gitlab-ci.yml, Jenkinsfile, Dockerfile, docker-compose.yml |
| `SECURITY` | `extract_security()` | Grep pola auth middleware, JWT, OAuth, env vars, secret patterns |
| `MAIN_FLOW` | `extract_main_flow()` | Baca entry points (main.py, app.py, index.js), grep startup sequence, request lifecycle |
| `SPECIFIC_SYMBOL` | `extract_specific_symbol(repo_path, target, user_input)` | Cari definisi + penggunaan simbol target. Jika target adalah qualified name (`pkg.Func`) **dan** `user_input` mengandung API path berparameter → dual-trace paralel: `_trace_api_route()` + `_find_symbol_definition()`. Jika target dimulai dengan `/` → `_trace_api_route()` saja. Jika symbol biasa → `_find_symbol_definition()` saja. |

Semua extractor bersifat **READ-ONLY**, menggunakan kombinasi `Path.rglob()` dan `re` untuk membaca dan menganalisis file secara lokal.

---

## Pembatasan & Prinsip Keamanan

1. **READ-ONLY** — tidak ada `git add`, `git commit`, atau `git push`.
2. **Tidak ada eksekusi kode** — hanya membaca dan menganalisis file statis.
3. **Anti-halusinasi** — LLM wajib menyebut sumber (file + baris) untuk setiap klaim.
4. **Anti-code-dump** — untuk pertanyaan "cara kerja", LLM dilarang menyalin ulang kode; harus menjelaskan dalam prosa.
5. **Batas ukuran** — per-file max 40.000 karakter, max 12 file per extractor, max 100 baris per grep result.

---

## Contoh Skenario Lengkap

### Skenario: "Jelaskan cara kerja endpoint `/download/:uuid`"

```
User: "Di repo https://github.com/user/myapp jelaskan cara kerja endpoint /download/:uuid"

1. classify_intent() → SPECIFIC_SYMBOL
   └─ pattern: route path berparam `/download/:uuid`

2. _extract_request() via LLM →
   { repo_url: "github.com/user/myapp", problem: "cara kerja /download/:uuid",
     branch: "" }

3. _resolve_repo() → clone ke ~/sandbox_repos/myapp/

4. Branch tidak ada → deteksi "main" → tanya konfirmasi user

5. User balas "lanjutkan" → checkout "main"

6. _run_qa_flow():
   ├── extract_specific_target() → "/download/:uuid"
   ├── _EXPLANATION_Q_RE.search() → True ("cara kerja")
   ├── asyncio.gather():
   │   ├── extract_specific_symbol(repo_path, "/download/:uuid")
   │   │     → grep route + baca handler file
   │   ├── [RAG dilewati karena is_explanation_q=True + SPECIFIC_SYMBOL]
   │   ├── _fetch_tavily_context() → (tidak relevan, dilewati)
   │   └── [dir_tree dilewati]
   ├── evidence di-cap 8.000 karakter
   ├── preamble instruksi anti-code-dump ditambahkan
   └── LLM menjawab dalam prosa bahasa natural step-by-step

Output:
🌿 Branch: `main`

## 💬 Jawaban
Endpoint `/download/:uuid` bekerja dalam 4 langkah utama:
1. **Validasi UUID** — handler memverifikasi format UUID (file: handlers/download.go:42) 🟢 [CONFIRMED]
2. **Cek otorisasi** — middleware auth memverifikasi token JWT sebelum request sampai handler ...
...

⏱️ 🔍 Symbol Q/A · 4.1s (ekstraksi: 2.3s)
```

---

## File yang Terlibat

| File | Peran |
|---|---|
| [agent.py](agent.py) | Class utama `DeveloperQnAAgent`, orkestrasi alur |
| [src/agents/repo_agent_base.py](../repo_agent_base.py) | Base class: repo clone/pull, branch, RAG, Tavily, LLM extraction |
| [src/tools/repo_qa.py](../../tools/repo_qa.py) | Engine Q/A: `classify_intent`, semua extractor topik, `run_qa_extraction` |
| [src/agents/llm_client.py](../llm_client.py) | Abstraksi pemanggilan LLM (OpenRouter/OpenAI compatible) |
| [src/memory/state.py](../../memory/state.py) | `AgentTask` — data task in/out |
| [src/memory/repo_tracker.py](../../memory/repo_tracker.py) | Tracking repo yang sudah di-clone |
| [src/tools/cli_executor.py](../../tools/cli_executor.py) | Eksekutor shell command async (git, find, grep) |
