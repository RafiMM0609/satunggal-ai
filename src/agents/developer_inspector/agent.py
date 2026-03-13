"""
DeveloperInspectorAgent – Repository Inspector & Diagnostician.

Peran:
  Inspektor senior yang MEMBACA dan MENGANALISIS repositori kode.
  Ia TIDAK menulis kode, TIDAK mengedit file, TIDAK melakukan commit atau push.
  Tugasnya adalah:
    1. Menginspeksi struktur dan isi repositori secara menyeluruh.
    2. Mengidentifikasi akar penyebab masalah (root cause analysis).
    3. Menyusun laporan inspeksi yang jelas dan actionable.
    4. Memberikan rekomendasi perbaikan yang spesifik kepada developer.

Workflow:
  1. Ekstrak repo_url + deskripsi masalah dari input pengguna via LLM.
  2. Clone repo (jika URL disertakan) atau gunakan repo yang sudah ada via RepoTracker.
  3. Jalankan pemeriksaan read-only:
       - Struktur direktori (ls -R)
       - Git log & diff terbaru
       - Grep untuk pola error / keyword masalah
       - Baca file-file kunci (entry points, config, bagian yang dicurigai)
  4. Kirim semua temuan ke LLM untuk analisis mendalam.
  5. Kembalikan laporan inspeksi terstruktur.

Batasan penting:
  - READ-ONLY: tidak ada git add/commit/push.
  - Tidak ada eksekusi kode (compile, run, docker).
  - Tidak ada penulisan atau pengeditan file apapun di repo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.repo_tracker import RepoTracker
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor, CommandResult
from src.tools.git_utils import (
    inject_pat_into_url as _inject_pat_into_url,
    is_gitlab_url       as _is_gitlab_url,
    repo_name_from_url  as _repo_name_from_url,
)
from src.tools.repo_qa import (
    QAIntent,
    classify_intent,
    extract_specific_target,
    run_qa_extraction,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

REPOS_BASE_DIR      = Path.home() / "sandbox_repos"
MAX_FILE_BYTES      = 40_000   # max bytes per file snippet sent to LLM
MAX_GREP_LINES      = 80       # max lines from grep output per pattern
MAX_LOG_LINES       = 50       # max git log lines
MAX_DIFF_LINES      = 120      # max git diff lines
MAX_LS_LINES        = 150      # max lines from directory listing

# RAG: how many top-relevant source files to read in full.
MAX_RELEVANT_FILES  = 6

# LLM sampling parameters – low temperature for determinism, less hallucination.
INSPECTOR_TEMPERATURE = 0.15
INSPECTOR_TOP_P       = 0.90

# Critic (second-pass) uses even lower temperature for strict fact-checking.
CRITIC_TEMPERATURE    = 0.10
CRITIC_TOP_P          = 0.85

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Inspektor Kode Senior** (Developer Inspector).

Identitasmu:
- Kamu adalah INSPEKTOR, bukan programmer.
- Kamu membaca dan MENGANALISIS kode, tapi kamu TIDAK menulis, mengedit, \
  atau mengeksekusi kode apapun.
- Kamu seperti seorang detektif teknis: mengumpulkan bukti, mengidentifikasi \
  akar masalah, dan memberikan rekomendasi yang akurat berdasarkan FAKTA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ATURAN KRITIS – ANTI-HALUSINASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **DILARANG KERAS** membuat klaim tanpa bukti dari data yang diberikan.
2. Setiap temuan HARUS disertai kutipan langsung (exact quote) dari \
file/log/diff yang ada dalam evidence.
3. Tulis **[PERLU VERIFIKASI]** jika kamu menduga adanya masalah tapi tidak ada bukti \
langsung dalam data yang diberikan.
4. Tulis **[DATA TIDAK CUKUP]** jika data tidak memungkinkan diagnosis akurat \
daripada menebak-nebak.
5. **JANGAN** mengasumsikan struktur kode, naming convention, atau bug yang tidak \
terlihat dalam evidence.
6. Jika ada ketidakpastian, nyatakan tingkat kepercayaan:
   - 🟢 **[CONFIRMED]** – bukti kuat, dikutip langsung dari kode/log.
   - 🟡 **[LIKELY]** – indikasi kuat tapi perlu verifikasi tambahan.
   - 🔴 **[UNVERIFIED]** – dugaan tanpa bukti langsung; harus diverifikasi developer.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tugasmu adalah menghasilkan Laporan Inspeksi Repositori yang mencakup:

---

## 📋 LAPORAN INSPEKSI REPOSITORI

### 1. Ringkasan Eksekutif
Jelaskan secara singkat apa yang ditemukan (2-4 kalimat). Nyatakan secara jelasnya \
apa yang SUDAH DIVERIFIKASI vs yang MASIH DUGAAN.

### 2. Struktur Proyek
Deskripsikan arsitektur dan organisasi kode **berdasarkan directory tree yang diberikan**. \
Hanya deskripsikan yang TERLIHAT dalam data.

### 3. 🔍 Temuan Masalah
Daftar semua masalah teridentifikasi dengan tingkat keparahan:
- 🔴 **KRITIS**: Masalah yang menyebabkan sistem tidak berfungsi.
- 🟡 **SEDANG**: Degradasi performa atau fungsionalitas.
- 🟢 **RINGAN**: Masalah kode yang tidak urgent.

Untuk setiap masalah, sertakan wajib:
  - **Lokasi**: Nama file dan nomor baris (exact, bukan perkiraan).
  - **Bukti** (WAJIB): Cuplikan kode/log yang dikutip PERSIS dari evidence. \
Jika tidak ada kutipan, tambahkan tanda 🔴 **[UNVERIFIED]**.
  - **Deskripsi**: Apa yang salah dan mengapa ini masalah.
  - **Kepercayaan**: 🟢 CONFIRMED / 🟡 LIKELY / 🔴 UNVERIFIED.

### 4. 🎯 Analisis Akar Masalah (Root Cause)
Jelaskan mengapa masalah ini terjadi secara teknis dan mendalam, dengan merujuk \
pada bukti spesifik dari kode.

### 5. 💡 Rekomendasi Perbaikan
Langkah perbaikan spesifik dan actionable, diurutkan berdasarkan prioritas.
- Sertakan nama file dan fungsi yang perlu diubah (bukan secara umum).
- Berikan pseudocode atau contoh pattern yang harus diterapkan developer.
- Hanya rekomendasikan perubahan yang didukung oleh temuan nyata.

### 6. ⚠️ Risiko Jika Tidak Diperbaiki
Dampak potensial jika masalah dibiarkan (berbasis temuan yang CONFIRMED).

### 7. 📊 Ringkasan Kepercayaan
| Temuan | Status | Dasar Bukti |
|--------|--------|-------------|
(Tabel semua temuan dengan status CONFIRMED/LIKELY/UNVERIFIED)

---

**Aturan Format:**
1. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
2. Jujur: jika data tidak cukup, katakan demikian — jangan mengarang.
3. Spesifik: kutip file, baris, dan nama fungsi dengan tepat.
4. Profesional: objektif, berbasis data, hindari menyalahkan developer.
"""

# ── Critic (second-pass verification) prompt ─────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """\
Kamu adalah **Reviewer Laporan Inspeksi Kode** yang tugasnya mem-verifikasi \
setiap klaim dalam laporan inspeksi terhadap evidence yang diberikan.

Tugasmu:
1. Baca setiap temuan dalam Laporan Inspeksi.
2. Periksa apakah temuan tersebut didukung oleh kutipan langsung dalam evidence.
3. Perbarui status kepercayaan setiap temuan:
   - 🟢 **[CONFIRMED]** jika ada kutipan langsung dari kode/log.
   - 🟡 **[LIKELY]** jika ada indikasi kuat tapi tidak dikutip langsung.
   - 🔴 **[UNVERIFIED]** jika tidak ada bukti dalam evidence.
4. Hapus atau tandai klaim yang tidak bisa diverifikasi.
5. Pertahankan semua temuan yang valid, tambahkan kutipan yang terlewat jika ada dalam evidence.
6. Tambahkan catatan reviewer di awal laporan: berapa temuan CONFIRMED, LIKELY, UNVERIFIED.

Jangan ubah gaya penulisan laporan. Kembalikan laporan lengkap yang sudah diverifikasi.
"""

_CRITIC_USER_TEMPLATE = """\
## LAPORAN INSPEKSI (perlu diverifikasi)

{report}

---

## EVIDENCE YANG TERSEDIA

{evidence}

---

Verifikasi setiap temuan dalam laporan terhadap evidence di atas. \
Perbarui status [CONFIRMED/LIKELY/UNVERIFIED] dan tambahkan/perbaiki kutipan bukti.
"""

# ── Q/A mode LLM prompt ───────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
Kamu adalah **Asisten Analisis Repositori** yang menjawab pertanyaan langsung
tentang sebuah codebase berdasarkan data yang diekstrak dari repositori.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ATURAN KRITIS – ANTI-HALUSINASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Jawab HANYA berdasarkan data repositori yang diberikan.
2. Setiap poin jawaban HARUS disertai sumber: nama file + nomor baris jika ada.
3. Gunakan label:
   - 🟢 **[CONFIRMED]** – ditemukan langsung dalam kode.
   - 🟡 **[LIKELY]** – dapat disimpulkan dari konteks.
   - 🔴 **[UNVERIFIED]** – tidak ada bukti langsung, tulis ini jika harus menduga.
4. Jika data tidak cukup, tulis **[DATA TIDAK CUKUP]** dan jelaskan apa yang perlu diperiksa.
5. DILARANG mengarang detail yang tidak ada dalam data.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format jawaban:

## 💬 Jawaban
<Jawaban langsung dan padat, 2-5 kalimat>

## 📋 Detail & Bukti
<Daftar poin dengan sumber file:baris dan kutipan kode>

## 🗺️ Lokasi di Repo
<Tabel ringkas: nama/path | file | baris | status>

## 💡 Catatan Tambahan
<Hal-hal penting yang relevan dengan pertanyaan>

**Format:**
- Gunakan bahasa yang sama dengan pertanyaan pengguna.
- Singkat dan faktual, tidak perlu struktur laporan inspeksi penuh.
- Jangan tampilkan template kosong jika tidak ada konten.
"""

# ── Extract prompt ─────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Ekstrak informasi berikut dari pesan pengguna dan balas dalam JSON:
{{
  "repo_url":   "<URL lengkap repository (GitHub, GitLab, Bitbucket, dll.) atau string kosong jika tidak ada>",
  "problem":    "<deskripsi ringkas masalah yang dilaporkan atau area yang ingin diinspeksi>",
  "keywords":   ["<keyword error atau simbol yang relevan untuk dicari di kode>"],
  "branch":     "<nama git branch jika disebutkan secara eksplisit dalam pesan, jika tidak ada biarkan string kosong>"
}}

Perhatian: repo_url bisa berupa URL GitHub (github.com), GitLab (gitlab.com), atau platform git lainnya.
Salin URL persis seperti yang disebutkan pengguna, termasuk scheme https://.

Pesan pengguna: {user_input}
"""


# ── Pydantic schema untuk parsing LLM extract ─────────────────────────────────

class InspectionRequest(BaseModel):
    repo_url:   str       = ""
    problem:    str       = ""
    keywords:   list[str] = []
    branch:     str       = ""
    # Q/A mode: diisi oleh agent berdasarkan classify_intent(), bukan oleh LLM extractor.
    qa_mode:    bool      = False
    qa_intent:  str       = ""    # nilai QAIntent
    verbosity:  str       = "detailed"  # detailed | concise
    # Optional: user-provided candidate routing filenames (e.g. ['routes.go']).
    candidate_route_filenames: list[str] = []

# ── Branch confirmation state (per session, in-process) ─────────────────────
#
# Mirrors the pattern in developer/agent.py: when no branch is specified the
# agent returns a confirmation request to the client via the orchestrator, and
# resumes on the next message from the stored pending state.

_inspector_pending_confirmations: dict[str, dict] = {}

# Q/A pending confirmations: digunakan saat branch belum diketahui pada Q/A mode
_qa_pending_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}


def _resolve_branch_from_reply(user_input: str, detected_branch: str) -> str | None:
    """
    Parse the user's confirmation reply and return the branch to use.
    Returns the branch name or None if the reply is not a recognizable confirmation.
    """
    clean = user_input.strip()
    lower = clean.lower()
    if lower in _CONFIRMATION_ANSWERS:
        return detected_branch
    if len(clean) <= 100 and " " not in clean and re.match(r"^[\w\-./]+$", clean):
        return clean
    return None

# ── Q/A Intent display labels ─────────────────────────────────────────────────────

_QA_INTENT_LABELS: dict["QAIntent", str] = {}


def _build_qa_intent_labels() -> None:
    """Populate _QA_INTENT_LABELS lazily after QAIntent is imported."""
    _QA_INTENT_LABELS.update({
        QAIntent.API_ENDPOINTS:   "📡 API Endpoints",
        QAIntent.TECH_STACK:      "🛠️ Tech Stack",
        QAIntent.DATA_MODELS:     "🗃️ Data Models",
        QAIntent.DEPENDENCIES:    "📦 Dependencies",
        QAIntent.CI_CD:           "🚀 CI/CD",
        QAIntent.SECURITY:        "🔐 Security",
        QAIntent.MAIN_FLOW:       "🔄 Main Flow",
        QAIntent.SPECIFIC_SYMBOL: "🔍 Symbol Q/A",
    })


_build_qa_intent_labels()

# ── Agent ──────────────────────────────────────────────────────────────────────

class DeveloperInspectorAgent(BaseAgent):
    """
    Read-only repository inspector.

    Collects evidence from the codebase via shell commands,
    then delegates root-cause analysis to the LLM.
    """

    name = "developer_inspector"

    def __init__(self, llm: LLMClient | None = None) -> None:
        from config.settings import get_settings
        _settings          = get_settings()
        self._llm          = llm or LLMClient()
        self._repo_tracker = RepoTracker()
        self._cli          = CLIExecutor(timeout=30)
        self._repos_dir    = Path(_settings.sandbox_repos_dir).expanduser()
        self._github_pat   = _settings.github_pat
        self._gitlab_pat   = _settings.gitlab_pat
        self._repos_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers: read-only shell commands ─────────────────────────────────────

    async def _run_cmd(self, cmd: str, cwd: Path | None = None) -> str:
        """Run a shell command and return stdout (truncated). Never write-mode commands."""
        result: CommandResult = await self._cli.run(cmd, work_dir=cwd)
        out = (result.stdout or "") + (result.stderr or "")
        return out.strip()
    async def _get_current_branch(self, repo_path: Path) -> str:
        """Return the name of the currently checked-out branch (read-only)."""
        out = await self._run_cmd("git rev-parse --abbrev-ref HEAD", cwd=repo_path)
        return out.strip() or "main"

    async def _checkout_branch(self, repo_path: Path, branch: str) -> None:
        """
        Checkout the requested branch for read-only inspection.

        Strategy:
          1. Try a simple `git checkout <branch>`.
          2. If checkout fails due to conflict/unmerged state, abort any
             in-progress rebase/merge and reset hard, then retry.
          3. If branch already exists locally but checkout still fails,
             force-checkout it.
          4. If the branch is not found locally, fetch all remotes and retry
             as a tracking branch (`-b <branch> origin/<branch>`).
          5. If `-b` fails because the branch now exists locally (race), fall
             back to a plain checkout.
        """
        cli = CLIExecutor(timeout=30)

        # ── Attempt 1: plain checkout ──────────────────────────────────────
        result = await cli.run(f"git checkout {branch}", work_dir=repo_path)
        if result.succeeded:
            logger.info("Inspector: checked out branch '%s'", branch)
            return

        stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 2: recover from conflict / unmerged-files state ───────
        if any(kw in stderr.lower() for kw in ("unmerged", "conflict", "merge", "rebase")):
            logger.warning(
                "Inspector: checkout blocked by dirty state – recovering: %s", stderr[:200]
            )
            await cli.run("git rebase --abort", work_dir=repo_path)
            await cli.run("git merge --abort",  work_dir=repo_path)
            await cli.run("git reset --hard HEAD", work_dir=repo_path)
            result = await cli.run(f"git checkout {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("Inspector: checked out branch '%s' after state recovery", branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 3: branch exists locally but checkout still failed ────
        branch_check = await cli.run(f"git branch --list {branch}", work_dir=repo_path)
        if branch in (branch_check.stdout or ""):
            # Exists locally – force-checkout to discard any local changes.
            result = await cli.run(f"git checkout -f {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("Inspector: force-checked out existing branch '%s'", branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 4: branch missing locally – fetch all remotes ─────────
        await cli.run("git remote set-branches origin '*'", work_dir=repo_path)
        await cli.run("git fetch --all --prune", work_dir=repo_path)
        result = await cli.run(
            f"git checkout -b {branch} origin/{branch}", work_dir=repo_path
        )
        if result.succeeded:
            logger.info("Inspector: checked out remote branch '%s'", branch)
            return

        stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 5: '-b' failed because branch already exists locally ──
        if "already exists" in stderr:
            result = await cli.run(f"git checkout -f {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("Inspector: checked out (already-local) branch '%s'", branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        raise RuntimeError(
            f"Branch '{branch}' tidak dapat di-checkout:\n{stderr[:400]}"
        )
    async def _get_dir_tree(self, repo_path: Path) -> str:
        """Return a pruned directory listing."""
        out = await self._run_cmd(
            "find . -not \\( "
            + " ".join(f"-path './{d}' -prune -o" for d in _SKIP_DIRS)
            + " -false \\) -print | head -" + str(MAX_LS_LINES),
            cwd=repo_path,
        )
        return out or "(empty)"

    async def _get_git_log(self, repo_path: Path) -> str:
        out = await self._run_cmd(
            f"git log --oneline -n {MAX_LOG_LINES}",
            cwd=repo_path,
        )
        return out or "(no git log)"

    async def _get_git_diff(self, repo_path: Path) -> str:
        out = await self._run_cmd(
            f"git diff HEAD~1 HEAD --stat 2>/dev/null | head -{MAX_DIFF_LINES}",
            cwd=repo_path,
        )
        return out or "(no diff)"

    async def _grep_keywords(self, repo_path: Path, keywords: list[str]) -> str:
        if not keywords:
            return "(no keywords specified)"
        pattern = "|".join(re.escape(k) for k in keywords[:5])
        out = await self._run_cmd(
            f"grep -rn "
            f"--include='*.py' --include='*.js' --include='*.ts' "
            f"--include='*.go' --include='*.java' --include='*.rb' "
            f"--include='*.php' --include='*.cs' --include='*.rs' "
            f"--include='*.vue' --include='*.tsx' --include='*.jsx' "
            f"-E '{pattern}' . 2>/dev/null | head -{MAX_GREP_LINES}",
            cwd=repo_path,
        )
        return out or f"(no matches for: {', '.join(keywords)})"

    async def _grep_error_patterns(self, repo_path: Path) -> str:
        """Grep for generic error/exception patterns across common source files."""
        error_pattern = (
            r"(Exception|Error|Traceback|panic:|FATAL|CRITICAL"
            r"|undefined is not|cannot read property|NullPointerException"
            r"|segfault|SIGSEGV|stack overflow|out of memory)"
        )
        out = await self._run_cmd(
            f"grep -rn "
            f"--include='*.py' --include='*.js' --include='*.ts' "
            f"--include='*.go' --include='*.java' --include='*.rs' "
            f"-iE '{error_pattern}' . 2>/dev/null | head -{MAX_GREP_LINES}",
            cwd=repo_path,
        )
        return out or "(no generic error patterns found)"

    async def _read_key_files(self, repo_path: Path) -> str:
        """Read common entry-point and config files if they exist."""
        candidates = [
            "README.md", "README.rst",
            "main.py", "app.py", "server.py", "index.js", "index.ts",
            "manage.py", "wsgi.py", "asgi.py",
            "package.json", "pyproject.toml", "setup.py", "requirements.txt",
            "docker-compose.yml", "Dockerfile",
            ".env.example",
        ]
        snippets: list[str] = []
        for name in candidates:
            fpath = repo_path / name
            if fpath.exists() and fpath.is_file():
                try:
                    text = fpath.read_text(errors="replace")[:MAX_FILE_BYTES]
                    snippets.append(f"### {name}\n```\n{text}\n```")
                except OSError:
                    pass
        return "\n\n".join(snippets) if snippets else "(no key files found)"

    async def _find_error_logs(self, repo_path: Path) -> str:
        """Search for error patterns in log files and common output files."""
        out = await self._run_cmd(
            "find . -name '*.log' -o -name 'error.txt' -o -name 'crash.txt' "
            "2>/dev/null | head -5 | xargs -I{} sh -c "
            f"'echo \"=== {{}} ===\"; head -{MAX_GREP_LINES} {{}}' 2>/dev/null",
            cwd=repo_path,
        )
        return out or "(no log files found)"
    async def _read_relevant_files(self, repo_path: Path, problem: str) -> str:
        """
        RAG step: index the repo with AST/TF-IDF ranking and read the top-N
        source files most relevant to the reported problem.

        Falls back gracefully if code_search dependencies are not available.
        """
        try:
            from src.tools.code_search import build_ast_index, rank_files_by_relevance
        except ImportError:
            logger.warning("Inspector: code_search not available; skipping RAG step")
            return "(code_search unavailable)"

        if not problem:
            return "(no problem description for relevance ranking)"

        try:
            logger.info("Inspector: building AST index for RAG at %s", repo_path)
            t0 = time.monotonic()
            symbol_index = build_ast_index(repo_path)
            candidates   = list(symbol_index.keys())

            if not candidates:
                return "(no indexable source files found)"

            ranked  = rank_files_by_relevance(candidates, symbol_index, problem)
            top_n   = ranked[:MAX_RELEVANT_FILES]
            elapsed = time.monotonic() - t0
            logger.info(
                "Inspector: RAG indexed %d files in %.2fs; top %d selected",
                len(candidates), elapsed, len(top_n),
            )

            snippets: list[str] = []
            for rel_path in top_n:
                abs_path = repo_path / rel_path
                try:
                    text = abs_path.read_text(errors="replace")[:MAX_FILE_BYTES]
                    snippets.append(
                        f"### 📄 {rel_path} (relevant to problem)\n"
                        f"```\n{text}\n```"
                    )
                except OSError as exc:
                    logger.debug("Inspector: could not read %s: %s", rel_path, exc)

            return (
                "\n\n".join(snippets)
                if snippets
                else "(no relevant files could be read)"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Inspector: RAG step failed: %s", exc)
            return f"(RAG error: {exc})"

    async def _verify_report(self, report: str, evidence_text: str) -> str:
        """
        Critic second-pass: ask the LLM to cross-check every finding in the
        initial report against the raw evidence and update confidence labels.

        Uses a very low temperature (CRITIC_TEMPERATURE) for strict fact-checking.
        """
        logger.info("Inspector: running critic verification pass")
        critic_user = _CRITIC_USER_TEMPLATE.format(
            report=report,
            evidence=evidence_text[:60_000],  # cap to avoid context overflow
        )
        try:
            verified = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": critic_user},
                ],
                temperature=CRITIC_TEMPERATURE,
                top_p=CRITIC_TOP_P,
            )
            return verified.strip() or report
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inspector: critic pass failed (%s); using initial report", exc)
            return report
    # ── Repo resolution ────────────────────────────────────────────────────────

    async def _resolve_repo(self, repo_url: str) -> Path | None:
        """
        Return a local path for the repo.

        Priority:
          1. If repo_url given → clone (or update) and return path.
          2. If no URL → check RepoTracker for any previously cloned repo
             and return the most recent one.
        """
        if repo_url:
            # Clone into sandbox dir if not already there.
            # Use owner-repo slug (same convention as developer/agent.py) so
            # both agents share the same local directory for the same repo.
            repo_name  = _repo_name_from_url(repo_url)
            local_path = self._repos_dir / repo_name
            self._repos_dir.mkdir(parents=True, exist_ok=True)

            # Inject PAT for private HTTPS repos (GitLab or GitHub).
            _pat     = self._gitlab_pat if _is_gitlab_url(repo_url) else self._github_pat
            auth_url = _inject_pat_into_url(repo_url, _pat)

            if local_path.exists():
                logger.info("Inspector: repo already exists, pulling. path=%s", local_path)
                # Expand remote refspec so 'git fetch --all' picks up ALL branches,
                # even if the repo was previously cloned with --single-branch.
                await self._run_cmd(
                    "git remote set-branches origin '*'", cwd=local_path
                )
                fetch_out = await self._run_cmd(
                    f"git fetch {auth_url} --all --prune", cwd=local_path
                )
                pull_out = await self._run_cmd(
                    f"git pull {auth_url} --rebase --quiet", cwd=local_path
                )
                if "fatal" in pull_out.lower() or "error" in pull_out.lower():
                    logger.warning("Inspector: git pull may have failed: %s", pull_out)
                    # Recover from conflict / unmerged-files state so subsequent
                    # git commands (checkout, etc.) are not blocked.
                    if "unmerged" in pull_out.lower() or "conflict" in pull_out.lower() or "rebase" in pull_out.lower():
                        logger.info("Inspector: recovering from conflict/rebase state")
                        await self._run_cmd("git rebase --abort", cwd=local_path)
                        await self._run_cmd("git merge --abort", cwd=local_path)
                        await self._run_cmd("git reset --hard HEAD", cwd=local_path)
            else:
                logger.info("Inspector: cloning %s → %s", repo_url, local_path)
                # --no-single-branch ensures ALL remote branches are fetched,
                # not just the default branch (which --depth implies by default).
                result = await self._run_cmd(
                    f"git clone --no-single-branch {auth_url} {local_path}"
                )
                if "fatal" in result.lower() or "error" in result.lower():
                    logger.warning("Inspector: clone may have failed: %s", result)

            # Track in RepoTracker
            self._repo_tracker.upsert(
                    repo_name,
                    repo_url,         # store the clean URL (no PAT) in DB
                    str(local_path),
                    status="cloned",
                )
            return local_path if local_path.exists() else None

        # No URL – try last known repo from tracker
        repos = self._repo_tracker.list_all()
        if repos:
            latest = repos[-1]
            # list_all() returns RepoRecord dataclass objects, not dicts –
            # access the attribute directly instead of calling .get().
            path = Path(latest.local_path) if latest.local_path else None
            if path and path.exists():
                logger.info("Inspector: no URL given, using tracked repo=%s", path)
                return path
        return None

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _extract_request(self, user_input: str) -> InspectionRequest:
        prompt   = _EXTRACT_PROMPT.format(user_input=user_input)
        response = await self._llm.chat(
            messages=[
                {"role": "system", "content": "You are a JSON extractor. Reply with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.strip()
        # strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$",          "", raw, flags=re.MULTILINE)
        try:
            data = json.loads(raw)
            return InspectionRequest(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Inspector: failed to parse extraction JSON: %s", exc)
            return InspectionRequest(problem=user_input)

    def _build_evidence_text(self, evidence: dict[str, str]) -> str:
        """Serialize evidence dict into a markdown string for LLM consumption."""
        return "\n\n".join(
            f"## {title}\n{content}"
            for title, content in evidence.items()
            if content.strip()
        )

    async def _fetch_tavily_context(self, query: str) -> str:
        """Attempt to fetch web research context from Tavily.

        Returns an empty string on any failure (missing API key, network, etc.).
        """
        try:
            from src.tools.tavily_search import TavilySearchTool  # type: ignore

            tool = TavilySearchTool()
            resp = await tool.search(query)
            ctx = resp.as_context_text()
            return ctx or ""
        except Exception as exc:  # pragma: no cover - best-effort external call
            logger.debug("Tavily fetch failed or unavailable: %s", exc)
            return ""

    async def _run_qa_flow(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       InspectionRequest,
    ) -> AgentTask:
        """
        Q/A mode: jawab pertanyaan spesifik user secara langsung dan ringkas.

        Flow:
          1. Jalankan extractor yang sesuai intent (API, tech, model, dll.).
          2. Sertakan RAG file relevan sebagai konteks tambahan.
          3. Kirim ke LLM dengan prompt Q/A (bukan template inspeksi penuh).
          4. Kembalikan jawaban langsung tanpa template laporan.
        """
        try:
            intent = QAIntent(req.qa_intent) if req.qa_intent else QAIntent.FULL_INSPECTION
            logger.info(
                "Inspector Q/A: intent=%s repo=%s problem=%r",
                intent, repo_path, req.problem,
            )
            t_start = time.monotonic()

            # Run topic extractor + RAG concurrently
            qa_evidence_task = asyncio.create_task(
                run_qa_extraction(
                    repo_path,
                    intent,
                    req.problem or task.user_input,
                    candidate_route_filenames=req.candidate_route_filenames,
                )
            )
            rag_task = asyncio.create_task(
                self._read_relevant_files(repo_path, req.problem or task.user_input)
            )
            tavily_task = asyncio.create_task(
                self._fetch_tavily_context(req.problem or task.user_input)
            )
            dir_tree_task = asyncio.create_task(
                self._get_dir_tree(repo_path)
            )

            qa_evidence, rag_files, tavily_ctx, dir_tree = await asyncio.gather(
                qa_evidence_task, rag_task, tavily_task, dir_tree_task,
                return_exceptions=True,
            )

            def _safe_str(r: object, fallback: str) -> str:
                return str(r) if not isinstance(r, Exception) else fallback

            evidence: dict[str, str] = {}

            # Primary: topic-specific extraction
            if isinstance(qa_evidence, dict):
                evidence.update(qa_evidence)
            else:
                evidence["Extraction Error"] = str(qa_evidence)

            # Secondary: RAG-relevant files as additional context
            rag_text = _safe_str(rag_files, "(RAG unavailable)")
            if rag_text.strip() and "unavailable" not in rag_text and "error" not in rag_text.lower():
                evidence["📂 File Relevan (RAG)"] = rag_text

            # Tertiary: directory tree for structural context
            tree = _safe_str(dir_tree, "")
            if tree.strip():
                evidence["🗂️ Struktur Direktori"] = tree

            # Optional: Tavily web search context for broader research
            tavily_text = _safe_str(tavily_ctx, "")
            if tavily_text.strip() and "hasil pencarian" in tavily_text.lower():
                evidence["🔎 Pencarian Web (Tavily)"] = tavily_text

            t_extract = time.monotonic()
            logger.info("Inspector Q/A: extraction done in %.2fs", t_extract - t_start)

            # Build Q/A LLM prompt
            evidence_text = self._build_evidence_text(evidence)
            verbosity_note = (
                "Jawab secara SINGKAT dan padat (maksimal 10 poin)."
                if req.verbosity == "concise"
                else "Jawab secara LENGKAP dengan detail dan contoh kode bila ada."
            )
            user_msg = (
                f"**Pertanyaan pengguna:**\n{task.user_input}\n\n"
                f"**Panduan verbositas:** {verbosity_note}\n\n"
                f"---\n\n"
                f"**Data dari repositori:**\n\n{evidence_text}"
            )

            qa_response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _QA_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=INSPECTOR_TEMPERATURE,
                top_p=INSPECTOR_TOP_P,
            )

            t_total = time.monotonic() - t_start
            logger.info("Inspector Q/A: done in %.2fs total", t_total)

            branch_note = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            intent_badge = _QA_INTENT_LABELS.get(intent, "💬 Q/A")
            perf_footer = (
                f"\n\n---\n"
                f"⏱️ *{intent_badge} · {t_total:.1f}s "
                f"(ekstraksi: {t_extract - t_start:.1f}s)*"
            )
            task.mark_done(branch_note + qa_response.strip() + perf_footer)

        except Exception as exc:
            logger.exception("Inspector Q/A flow error: %s", exc)
            task.mark_failed(f"❌ Q/A gagal: {exc}")

        return task

    async def _run_inspection_llm(
        self,
        user_input: str,
        problem:    str,
        evidence:   dict[str, str],
    ) -> str:
        """
        Phase 1 – Generate initial report.
        Phase 2 – Critic pass: verify every finding against raw evidence.
        """
        evidence_text = self._build_evidence_text(evidence)

        # Log evidence sizes for telemetry
        for title, content in evidence.items():
            logger.debug(
                "Inspector evidence '%s': %d chars",
                title, len(content),
            )
        logger.info(
            "Inspector: sending %d evidence sections (%d total chars) to LLM",
            len(evidence), len(evidence_text),
        )

        user_msg = (
            f"**Permintaan inspeksi:**\n{user_input}\n\n"
            f"**Masalah yang dilaporkan:**\n{problem}\n\n"
            f"---\n\n"
            f"**Hasil pengumpulan data dari repositori:**\n\n{evidence_text}"
        )

        t0 = time.monotonic()
        initial_report = await self._llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=INSPECTOR_TEMPERATURE,
            top_p=INSPECTOR_TOP_P,
        )
        logger.info(
            "Inspector: initial report generated in %.2fs (%d chars)",
            time.monotonic() - t0, len(initial_report),
        )

        # Phase 2: critic verification
        verified_report = await self._verify_report(initial_report.strip(), evidence_text)
        return verified_report

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            logger.info("DeveloperInspectorAgent: starting for session=%s", task.session_id)

            # ── Check for pending branch confirmation ──────────────────────
            pending = _inspector_pending_confirmations.get(task.session_id)
            if pending:
                branch_choice = _resolve_branch_from_reply(
                    task.user_input, pending["detected_branch"]
                )
                if branch_choice is not None:
                    del _inspector_pending_confirmations[task.session_id]
                    repo_path = Path(pending["repo_path"])
                    await self._checkout_branch(repo_path, branch_choice)
                    req = InspectionRequest(
                        repo_url=pending["repo_url"],
                        problem=pending["problem"],
                        keywords=pending["keywords"],
                        branch=branch_choice,
                        qa_mode=pending.get("qa_mode", False),
                        qa_intent=pending.get("qa_intent", ""),
                        verbosity=pending.get("verbosity", "detailed"),
                    )
                    if req.qa_mode and req.qa_intent:
                        return await self._run_qa_flow(task, repo_path, req)
                    return await self._run_inspection_task(task, repo_path, req)
                # Not a recognizable confirmation – fall through to normal parse.

            # ── Step 1: Classify intent FIRST (fast, no LLM call) ─────────
            intent = classify_intent(task.user_input)
            is_qa  = (intent != QAIntent.FULL_INSPECTION)
            logger.info("Inspector: intent=%s qa_mode=%s", intent.value, is_qa)

            # ── Step 2: Extract structured request ────────────────────────
            req = await self._extract_request(task.user_input)
            req.qa_mode   = is_qa
            req.qa_intent = intent.value if is_qa else ""

            lower = task.user_input.lower()
            if any(w in lower for w in ["singkat", "brief", "concise", "ringkas"]):
                req.verbosity = "concise"

            logger.info(
                "Inspector: repo_url=%r problem=%r keywords=%s branch=%r intent=%s",
                req.repo_url, req.problem, req.keywords, req.branch, intent.value,
            )

            # ── Step 3: Resolve local repo path ───────────────────────────
            repo_path = await self._resolve_repo(req.repo_url)

            if repo_path is None:
                # No repo – answer from description only (inspection-style).
                logger.info("Inspector: no local repo – description-only analysis.")
                warning = (
                    "\n\n> ⚠️ **Catatan:** Tidak ada repositori yang dapat diakses "
                    "(URL tidak diberikan atau tidak ada repo yang sebelumnya di-clone). "
                    "Analisis ini didasarkan pada deskripsi pengguna saja.\n"
                )
                evidence = {"Deskripsi dari Pengguna": task.user_input}
                report   = await self._run_inspection_llm(
                    task.user_input, req.problem or task.user_input, evidence,
                )
                task.mark_done(report + warning)
                return task

            # ── Step 4: Branch selection ───────────────────────────────────
            if req.branch:
                await self._checkout_branch(repo_path, req.branch)
                if req.qa_mode:
                    return await self._run_qa_flow(task, repo_path, req)
                return await self._run_inspection_task(task, repo_path, req)

            # No branch specified → detect and ask for confirmation.
            detected_branch = await self._get_current_branch(repo_path)
            _inspector_pending_confirmations[task.session_id] = {
                "repo_url":        req.repo_url,
                "repo_path":       str(repo_path),
                "problem":         req.problem,
                "keywords":        req.keywords,
                "detected_branch": detected_branch,
                "qa_mode":         req.qa_mode,
                "qa_intent":       req.qa_intent,
                "verbosity":       req.verbosity,
            }
            mode_label = (
                f"mode **Q/A** (`{_QA_INTENT_LABELS.get(intent, intent.value)}`)"
                if req.qa_mode
                else "mode **Inspeksi Penuh**"
            )
            task.mark_done(
                f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                f"Repository berhasil diakses. Branch aktif saat ini: **`{detected_branch}`**\n\n"
                f"Akan dijalankan dalam {mode_label} pada branch **`{detected_branch}`**.\n\n"
                f"Balas **`lanjutkan`** untuk melanjutkan, "
                f"atau ketik nama branch yang diinginkan "
                f"(contoh: `develop`, `feature/my-feature`)."
            )
            return task

        except Exception as exc:
            logger.exception("DeveloperInspectorAgent: unexpected error: %s", exc)
            task.mark_failed(
                f"❌ Gagal karena error tidak terduga: {exc}\n\n"
                "Mohon periksa log untuk detail lebih lanjut."
            )

        return task

    async def _run_inspection_task(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       InspectionRequest,
    ) -> AgentTask:
        """
        Execute the actual inspection after the branch has been confirmed
        and checked out.  Separated from run() so it can be called both on
        the first turn (branch explicit) and on the confirmation turn.
        """
        try:
            logger.info(
                "Inspector: inspecting repo at %s (branch=%s)",
                repo_path, req.branch,
            )
            t_start = time.monotonic()

            (
                dir_tree,
                git_log,
                git_diff,
                grep_result,
                grep_errors,
                key_files,
                error_logs,
                relevant_files,
            ) = await _gather_evidence(self, repo_path, req.keywords, req.problem)

            t_gather = time.monotonic()
            logger.info(
                "Inspector: evidence gathered in %.2fs",
                t_gather - t_start,
            )

            evidence = {
                "Struktur Direktori":           dir_tree,
                "Git Log (terbaru)":            git_log,
                "Git Diff (terakhir)":          git_diff,
                "Grep Keyword Masalah":         grep_result,
                "Grep Pola Error Umum":         grep_errors,
                "File Kunci":                   key_files,
                "Log & Error Files":            error_logs,
                "File Relevan (RAG/TF-IDF)":    relevant_files,
            }

            report = await self._run_inspection_llm(
                req.problem or "",
                req.problem or "",
                evidence,
            )

            t_total = time.monotonic() - t_start
            logger.info(
                "Inspector: inspection complete in %.2fs total "
                "(gather=%.2fs, llm=%.2fs)",
                t_total,
                t_gather - t_start,
                t_total - (t_gather - t_start),
            )

            branch_header = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            perf_footer = (
                f"\n\n---\n"
                f"\u23f1\ufe0f *Inspeksi selesai dalam {t_total:.1f}s "
                f"(pengumpulan data: {t_gather - t_start:.1f}s)*"
            )
            task.mark_done(branch_header + report + perf_footer)

        except Exception as exc:
            logger.exception("Inspector._run_inspection_task: error: %s", exc)
            task.mark_failed(
                f"❌ Inspeksi gagal pada branch `{req.branch}`: {exc}"
            )

        return task


# ── Git URL helpers ───────────────────────────────────────────────────────────

# is_gitlab_url, inject_pat_into_url, repo_name_from_url are imported from
# src.tools.git_utils at the top of this module.
# Self-hosted GitLab instances are handled via the GITLAB_HOSTS setting.


# ── Private gather helper (module-level to avoid 'self' closure issues) ────────

async def _gather_evidence(
    agent:     "DeveloperInspectorAgent",
    repo_path: Path,
    keywords:  list[str],
    problem:   str = "",
) -> tuple[str, str, str, str, str, str, str, str]:
    """
    Run all read-only inspection commands concurrently, then run RAG sequentially.

    Returns an 8-tuple:
        (dir_tree, git_log, git_diff, grep_keywords, grep_errors,
         key_files, error_logs, relevant_files)
    """
    parallel_results = await asyncio.gather(
        agent._get_dir_tree(repo_path),
        agent._get_git_log(repo_path),
        agent._get_git_diff(repo_path),
        agent._grep_keywords(repo_path, keywords),
        agent._grep_error_patterns(repo_path),
        agent._read_key_files(repo_path),
        agent._find_error_logs(repo_path),
        return_exceptions=True,
    )

    def _safe(r: object, fallback: str) -> str:
        if isinstance(r, Exception):
            logger.warning("Inspector evidence gather error: %s", r)
            return f"(error: {r})"
        return str(r) if r else fallback

    # RAG step runs after parallel commands (needs repo path to be intact).
    relevant_files = await agent._read_relevant_files(repo_path, problem)

    return (
        _safe(parallel_results[0], "(no dir tree)"),
        _safe(parallel_results[1], "(no git log)"),
        _safe(parallel_results[2], "(no diff)"),
        _safe(parallel_results[3], "(no grep result)"),
        _safe(parallel_results[4], "(no error patterns)"),
        _safe(parallel_results[5], "(no key files)"),
        _safe(parallel_results[6], "(no log files)"),
        relevant_files,
    )
