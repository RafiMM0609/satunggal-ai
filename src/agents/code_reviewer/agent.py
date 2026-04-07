"""
CodeReviewerAgent – Code Quality Review Agent (Read-Only).

Peran:
  Reviewer senior yang membaca repositori dan menghasilkan laporan kualitas kode.
  Berfokus pada:
    - Konvensi penulisan kode (naming, formatting, DRY, SOLID)
    - Anti-pattern dan technical debt
    - Kerentanan keamanan (hardcoded secrets, injection risks, insecure deps)
    - Performa (N+1 queries, unbounded loops, memory leaks)
    - Test coverage dan kualitas test

Perbedaan dengan DeveloperInspectorAgent:
  - Inspector  → menemukan BUG RUNTIME, root cause analysis, laporan inspeksi + critic pass
  - Reviewer   → menilai KUALITAS KODE, best practices, security audit (tidak terkait crash)

Workflow:
  1. Ekstrak repo_url + fokus review dari user via LLM (shared base).
  2. Clone / pull repo jika belum ada.
  3. Detect branch aktif.
  4. Kumpulkan bukti: struktur direktori, file kunci, dependency list.
  5. Kirim ke LLM untuk menghasilkan laporan review terstruktur.
  6. Kembalikan laporan dengan kategori temuan per jenis kualitas.

Batasan:
  - READ-ONLY: tidak ada git add/commit/push.
  - Tidak ada eksekusi kode.
  - Tidak menghasilkan patch atau kode pengganti.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from src.agents.repo_agent_base import RepoAgentBase, RepoExtractionRequest
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── Sampling parameters ────────────────────────────────────────────────────────

REVIEWER_TEMPERATURE = 0.15
REVIEWER_TOP_P       = 0.90
REVIEWER_MAX_TOKENS  = 16384

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Senior Code Reviewer** yang mengevaluasi kualitas kode repositori.

Identitasmu:
- Kamu adalah REVIEWER, bukan programmer.
- Kamu membaca dan MENILAI kode terhadap standar kualitas industri.
- Kamu TIDAK menulis, mengedit, atau mengeksekusi kode apapun.
- Seperti seorang arsitek teknis senior: menilai kualitas berdasarkan FAKTA dari kode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ATURAN KRITIS – ANTI-HALUSINASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Setiap temuan HARUS disertai kutipan langsung dari kode (file + baris).
2. Gunakan label kepercayaan: 🟢 [CONFIRMED] / 🟡 [LIKELY] / 🔴 [UNVERIFIED].
3. Tulis [DATA TIDAK CUKUP] jika kode yang tersedia tidak cukup untuk menilai aspek tertentu.
4. JANGAN mengarang masalah yang tidak terlihat dalam kode.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hasilkan **Laporan Review Kualitas Kode** dengan struktur berikut:

---

## 🔍 LAPORAN REVIEW KUALITAS KODE

### 1. Ringkasan Eksekutif
Gambaran umum kondisi kualitas kode (2-4 kalimat). Sebutkan bahasa/framework yang dideteksi.

### 2. 🏗️ Arsitektur & Struktur
Evaluasi:
- Pemisahan concerns (separation of concerns)
- Struktur direktori dan modularitas
- Keterbacaan dan konsistensi organisasi kode

### 3. 🔐 Keamanan (Security)
Identifikasi:
- Hardcoded secrets atau credential
- SQL injection / command injection risks
- Insecure dependency versions (jika requirements/package.json tersedia)
- Missing authentication/authorization checks
- Sensitive data exposure

### 4. 🧹 Kualitas Kode & Konvensi
Evaluasi:
- Naming conventions (variable, function, class names)
- DRY violations (Don't Repeat Yourself)
- Code smell (long methods, deep nesting, magic numbers)
- Dead code / unused imports
- Error handling quality

### 5. ⚡ Performa
Identifikasi:
- N+1 query patterns
- Unbounded loops atau recursion
- Inefficient data structure usage
- Missing indexes or caching opportunities

### 6. 🧪 Testing
Evaluasi:
- Keberadaan dan kualitas test files
- Test coverage yang terlihat (berdasarkan file test yang ada)
- Kualitas test: meaningful assertions, test isolation

### 7. 📊 Ringkasan Temuan
| Kategori | Tingkat | Deskripsi Singkat |
|----------|---------|-------------------|
| Keamanan | 🔴 Kritis / 🟡 Sedang / 🟢 Ringan | ... |
| ... | ... | ... |

### 8. 💡 Rekomendasi Prioritas
Top 5 perbaikan yang paling berdampak, diurutkan berdasarkan urgensi:
1. ...
2. ...

---

**Format:**
- Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
- Spesifik: kutip file, baris, dan nama fungsi/variabel.
- Profesional dan konstruktif.
- Jika fokus review disebutkan user (misal "fokus di security"), prioritaskan bagian tersebut.
"""


class CodeReviewerAgent(RepoAgentBase):
    """
    Read-only code quality review agent.

    Reviews code against style, security, performance, and best-practice standards.
    For bug hunting and root-cause analysis, use DeveloperInspectorAgent.
    """

    name = "code_reviewer"

    def __init__(
        self,
        llm: LLMClient | None = None,
        history=None,
    ) -> None:
        super().__init__(llm=llm, history=history)

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            logger.info("CodeReviewerAgent: starting for session=%s", task.session_id)

            req = await self._extract_request(task.user_input, session_id=task.session_id)
            logger.info(
                "CodeReviewer: repo_url=%r focus=%r branch=%r",
                req.repo_url, req.problem, req.branch,
            )

            repo_path = await self._resolve_repo(req.repo_url)

            if repo_path is None:
                task.mark_done(
                    "⚠️ **Tidak ada repositori yang dapat diakses.**\n\n"
                    "Untuk code review, sertakan URL repositori GitHub/GitLab "
                    "atau sebutkan repo yang sebelumnya di-clone.\n\n"
                    "Contoh: `review kualitas kode repo github.com/username/repo`"
                )
                return task

            # Detect/confirm branch
            if not req.branch:
                req = RepoExtractionRequest(
                    repo_url=req.repo_url,
                    problem=req.problem,
                    keywords=req.keywords,
                    branch=await self._get_current_branch(repo_path),
                    verbosity=req.verbosity,
                    candidate_route_filenames=req.candidate_route_filenames,
                )

            t_start = time.monotonic()
            evidence = await self._gather_evidence(repo_path, req)

            logger.info(
                "CodeReviewer: evidence gathered in %.2fs, calling LLM",
                time.monotonic() - t_start,
            )

            focus_note = f"\n\n**Fokus review yang diminta:** {req.problem}" if req.problem else ""
            user_msg = (
                f"**Permintaan:**\n{task.user_input}{focus_note}\n\n"
                f"---\n\n"
                f"**Data dari repositori:**\n\n{evidence}"
            )

            answer = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=REVIEWER_TEMPERATURE,
                top_p=REVIEWER_TOP_P,
                max_tokens=REVIEWER_MAX_TOKENS,
            )

            t_total = time.monotonic() - t_start
            branch_note = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            perf_footer = f"\n\n---\n⏱️ *🔍 Code Review · {t_total:.1f}s*"
            task.mark_done(branch_note + (answer or "[DATA TIDAK CUKUP]").strip() + perf_footer)

            self._save_session_context(task.session_id, req.repo_url, req.branch)

        except Exception as exc:
            logger.exception("CodeReviewerAgent: error for session=%s: %s", task.session_id, exc)
            task.mark_failed(f"❌ Code review gagal: {exc}")

        return task

    # ── Evidence gathering ────────────────────────────────────────────────────

    async def _gather_evidence(self, repo_path: Path, req: RepoExtractionRequest) -> str:
        """Collect directory tree + key files + dependency list concurrently."""
        dir_tree_coro   = self._get_dir_tree(repo_path)
        key_files_coro  = self._read_key_files(repo_path)
        dep_list_coro   = self._read_relevant_files(
            repo_path, req.problem or "security quality review"
        )

        dir_tree, key_files, dep_files = await asyncio.gather(
            dir_tree_coro, key_files_coro, dep_list_coro,
            return_exceptions=True,
        )

        sections: dict[str, str] = {}
        if isinstance(dir_tree, str) and dir_tree.strip():
            sections["🗂️ Struktur Direktori"] = dir_tree
        if isinstance(key_files, str) and key_files.strip():
            sections["📂 File Kunci"] = key_files
        if isinstance(dep_files, str) and dep_files.strip():
            sections["🔎 File Relevan (RAG)"] = dep_files

        return self._build_evidence_text(sections) or "(data repositori tidak tersedia)"

    async def _read_key_files(self, repo_path: Path) -> str:
        """Read common config / dependency files that are most useful for review."""
        targets = [
            "requirements.txt", "requirements-dev.txt", "Pipfile",
            "package.json", "package-lock.json",
            "pyproject.toml", "setup.py", "setup.cfg",
            "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
            ".env.example", "config.py", "settings.py",
            "README.md",
        ]
        sections: list[str] = []
        total = 0
        MAX_BYTES = 30_000

        for name in targets:
            abs_path = repo_path / name
            if not abs_path.is_file():
                continue
            try:
                text = abs_path.read_text(errors="replace")
                chunk = f"### {name}\n```\n{text[:3000]}\n```\n"
                if total + len(chunk) > MAX_BYTES:
                    break
                sections.append(chunk)
                total += len(chunk)
            except Exception:
                continue

        return "\n".join(sections) or "(tidak ada file konfigurasi ditemukan)"

    async def _get_current_branch(self, repo_path: Path) -> str:
        """Return name of current branch, defaulting to 'main'."""
        from src.tools.cli_executor import CLIExecutor
        executor = CLIExecutor(work_dir=repo_path, timeout=15)
        result   = await executor.run("git rev-parse --abbrev-ref HEAD")
        branch   = (result.stdout or "").strip()
        return branch or "main"
