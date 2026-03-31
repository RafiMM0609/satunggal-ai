# Agent dengan Akses Git

Dokumen ini merangkum agent yang berinteraksi dengan repositori Git dan batas aksesnya masing-masing.

---

## Ringkasan Akses

| Agent | Mode | Clone/Pull | Edit Kode | Commit & Push | Docker Sandbox |
|---|---|---|---|---|---|
| `DeveloperAgent` | Read-Write | ✅ | ✅ | ✅ (`GitManager`) | ✅ (`SandboxRunner`) |
| `DeveloperInspectorAgent` | Read-Only | ✅ | ❌ | ❌ | ❌ |
| `DeveloperQnAAgent` | Read-Only | ✅ | ❌ | ❌ | ❌ |
| `TechnicalWriterAgent` | Read-Only | ✅ | ❌ | ❌ | ❌ |

---

## DeveloperAgent (`src/agents/developer/agent.py`)

**Satu-satunya agent yang boleh menulis ke repository.**

Alur kerja:
1. Parse instruksi → ekstrak `repo_url` dan deskripsi task.
2. Clone atau pull repo ke direktori sandbox (`SANDBOX_REPOS_DIR`). Inject `GITHUB_PAT` ke URL untuk repo private.
3. Cek environment (Dockerfile / docker-compose.yml) → buat fallback bila tidak ada.
4. Edit kode via LLM → tulis patch ke disk.
5. Jalankan sandbox (Docker Compose) → jika ada error/traceback, kirim log ke LLM → retry (maks. 3×).
6. `GitManager.commit_and_push()` → `git add -A → commit → push`.

Tools internal (dikelola langsung oleh agent, **tidak** melalui pipeline orchestrator):

| Tool | File | Fungsi |
|---|---|---|
| `CLIExecutor` | `src/tools/cli_executor.py` | Jalankan perintah shell non-interaktif (timeout 5 mnt) |
| `SandboxRunner` | `src/tools/sandbox_runner.py` | Build & run Docker; generate Dockerfile fallback; deteksi traceback |
| `GitManager` | `src/tools/git_manager.py` | Konfigurasi git identity, inject PAT, `git add -A → commit → push` |
| `RepoTracker` | `src/memory/repo_tracker.py` | SQLite registry repo yang pernah di-clone |

---

## DeveloperInspectorAgent (`src/agents/developer_inspector/agent.py`)

**Read-only.** Menginspeksi codebase untuk menemukan root cause bug/error dan memberi rekomendasi perbaikan.

Perintah yang **boleh** digunakan:
`git log`, `git diff`, `git show`, `git status`, `find`, `grep`, `cat`, `head`, `tail`, `ls`, `wc`

Perintah yang **dilarang keras**:

| Larangan | Alasan |
|---|---|
| `GitManager.commit_and_push()` | Menulis ke repo |
| `SandboxRunner` | Eksekusi kode |
| `echo > file`, `sed -i`, dll. | Mengubah state repo |
| `git add`, `git commit`, `git push` | Write ke VCS |

---

## DeveloperQnAAgent (`src/agents/developer_qna/agent.py`)

**Read-only.** Menjawab pertanyaan faktual tentang isi codebase: API endpoints, tech stack, data models, CI/CD, security, alur utama, definisi simbol/fungsi tertentu.

- Menggunakan TF-IDF RAG untuk menemukan file relevan.
- Setiap klaim harus disertai sumber `file:baris`.
- Session context disimpan untuk follow-up pertanyaan dalam sesi yang sama.
- Tidak boleh menulis ke disk atau menjalankan perintah yang mengubah state.

Lihat [src/agents/developer_qna/HOW_IT_WORKS.md](src/agents/developer_qna/HOW_IT_WORKS.md) untuk detail alur kerja lengkap.

---

## TechnicalWriterAgent (`src/agents/technical_writer/agent.py`)

**Read-only pada repository.** Membuat dokumen teknis profesional (DOCX/PDF) dari isi codebase.

- Clone/pull repo → baca semua file → bagi menjadi chunk → generate dokumen per chunk via LLM → synthesize jadi dokumen final.
- Menggunakan `DiagramRendererTool` (render Mermaid → PNG) dan `DocumentGeneratorTool` (Markdown → DOCX/PDF).
- Tidak menulis kembali ke repository sumber — hanya membaca.

---

## Variabel Lingkungan Relevan

| Variabel | Kegunaan |
|---|---|
| `GITHUB_PAT` | Personal Access Token untuk clone/pull repo private GitHub |
| `SANDBOX_REPOS_DIR` | Direktori lokal tempat repo di-clone (default: `~/sandbox_repos`) |
