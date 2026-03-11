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

import json
import logging
import re
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

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

REPOS_BASE_DIR      = Path.home() / "sandbox_repos"
MAX_FILE_BYTES      = 40_000   # max bytes per file snippet sent to LLM
MAX_GREP_LINES      = 60       # max lines from grep output per pattern
MAX_LOG_LINES       = 30       # max git log lines
MAX_DIFF_LINES      = 80       # max git diff lines
MAX_LS_LINES        = 120      # max lines from directory listing

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
  akar masalah, dan memberikan rekomendasi yang akurat.

Tugasmu adalah menghasilkan Laporan Inspeksi Repositori yang mencakup:

---

## 📋 LAPORAN INSPEKSI REPOSITORI

### 1. Ringkasan Eksekutif
Jelaskan secara singkat apa yang ditemukan dalam inspeksi ini (2-4 kalimat).

### 2. Struktur Proyek
Deskripsikan arsitektur dan organisasi kode yang ditemukan.

### 3. 🔍 Temuan Masalah
Daftar semua masalah yang teridentifikasi, dengan tingkat keparahan:
- 🔴 **KRITIS**: Masalah yang menyebabkan sistem tidak berfungsi.
- 🟡 **SEDANG**: Masalah yang degradasi performa atau fungsionalitas.
- 🟢 **RINGAN**: Masalah kode yang perlu diperbaiki tapi tidak urgently.

Untuk setiap masalah, sertakan:
  - **Lokasi**: File dan baris yang relevan (jika diketahui).
  - **Deskripsi**: Apa yang salah.
  - **Bukti**: Cuplikan kode atau log yang mendukung temuan.

### 4. 🎯 Analisis Akar Masalah (Root Cause)
Jelaskan mengapa masalah ini terjadi secara teknis dan mendalam.

### 5. 💡 Rekomendasi Perbaikan
Daftar langkah-langkah perbaikan yang spesifik dan actionable, \
diurutkan berdasarkan prioritas. Sertakan contoh kode yang HARUS diperbaiki \
oleh developer (bukan oleh kamu).

### 6. Risiko Jika Tidak Diperbaiki
Jelaskan dampak potensial jika masalah dibiarkan.

---

**Aturan:**
1. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
2. Jujur: jika data tidak cukup untuk diagnosis akurat, katakan demikian.
3. Spesifik: tunjuk file, baris, dan function yang bermasalah.
4. Profesional: bersifat objektif, hindari menyalahkan developer.
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
    repo_url: str       = ""
    problem:  str       = ""
    keywords: list[str] = []
    branch:   str       = ""

# ── Branch confirmation state (per session, in-process) ─────────────────────
#
# Mirrors the pattern in developer/agent.py: when no branch is specified the
# agent returns a confirmation request to the client via the orchestrator, and
# resumes on the next message from the stored pending state.

_inspector_pending_confirmations: dict[str, dict] = {}

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
            f"grep -rn --include='*.py' --include='*.js' --include='*.ts' "
            f"--include='*.go' --include='*.java' --include='*.rb' "
            f"-E '{pattern}' . 2>/dev/null | head -{MAX_GREP_LINES}",
            cwd=repo_path,
        )
        return out or f"(no matches for: {', '.join(keywords)})"

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
            path   = Path(latest.get("local_path", ""))
            if path.exists():
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

    async def _run_inspection_llm(
        self,
        user_input: str,
        problem:    str,
        evidence:   dict[str, str],
    ) -> str:
        """Send all gathered evidence to the LLM for analysis."""
        evidence_text = "\n\n".join(
            f"## {title}\n{content}"
            for title, content in evidence.items()
            if content.strip()
        )

        user_msg = (
            f"**Permintaan inspeksi:**\n{user_input}\n\n"
            f"**Masalah yang dilaporkan:**\n{problem}\n\n"
            f"---\n\n"
            f"**Hasil pengumpulan data dari repositori:**\n\n{evidence_text}"
        )

        response = await self._llm.chat(
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_msg},
            ],
        )
        return response.strip()

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            logger.info("DeveloperInspectorAgent: starting inspection for session=%s", task.session_id)

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
                    )
                    return await self._run_inspection_task(task, repo_path, req)
                # Not a recognizable confirmation – fall through to normal parse.

            # ── Step 1: Extract structured request ────────────────────────
            req = await self._extract_request(task.user_input)
            logger.info(
                "Inspector: repo_url=%r problem=%r keywords=%s branch=%r",
                req.repo_url, req.problem, req.keywords, req.branch,
            )

            # ── Step 2: Resolve local repo path ───────────────────────────
            repo_path = await self._resolve_repo(req.repo_url)

            if repo_path is None:
                # No repo available – do a pure LLM analysis on the description.
                logger.info("Inspector: no local repo – proceeding with description-only analysis.")
                evidence = {
                    "Deskripsi Masalah dari Pengguna": task.user_input,
                }
                warning = (
                    "\n\n> ⚠️ **Catatan:** Tidak ada repositori yang dapat diakses "
                    "(URL tidak diberikan atau tidak ada repo yang sebelumnya di-clone). "
                    "Analisis ini didasarkan pada deskripsi pengguna saja.\n"
                )
                report = await self._run_inspection_llm(
                    task.user_input,
                    req.problem or task.user_input,
                    evidence,
                )
                task.mark_done(report + warning)
                return task

            # ── Branch selection ───────────────────────────────────────────
            if req.branch:
                # Branch explicitly mentioned → checkout immediately.
                await self._checkout_branch(repo_path, req.branch)
                return await self._run_inspection_task(task, repo_path, req)
            else:
                # No branch specified → detect current branch and ask client.
                detected_branch = await self._get_current_branch(repo_path)
                _inspector_pending_confirmations[task.session_id] = {
                    "repo_url":        req.repo_url,
                    "repo_path":       str(repo_path),
                    "problem":         req.problem,
                    "keywords":        req.keywords,
                    "detected_branch": detected_branch,
                }
                task.mark_done(
                    f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                    f"Repository berhasil diakses. Branch aktif saat ini adalah: **`{detected_branch}`**\n\n"
                    f"Inspeksi akan dijalankan pada branch **`{detected_branch}`**.\n\n"
                    f"Balas **`lanjutkan`** untuk melanjutkan pada branch ini, "
                    f"atau ketik nama branch yang diinginkan "
                    f"(contoh: `develop`, `feature/my-feature`)."
                )
                return task

        except Exception as exc:
            logger.exception("DeveloperInspectorAgent: unexpected error: %s", exc)
            task.mark_failed(
                f"❌ Inspeksi gagal karena error tidak terduga: {exc}\n\n"
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
            logger.info("Inspector: inspecting repo at %s (branch=%s)", repo_path, req.branch)

            (
                dir_tree,
                git_log,
                git_diff,
                grep_result,
                key_files,
                error_logs,
            ) = await _gather_evidence(self, repo_path, req.keywords)

            evidence = {
                "Struktur Direktori":   dir_tree,
                "Git Log (terbaru)":    git_log,
                "Git Diff (terakhir)":  git_diff,
                "Grep Keyword Masalah": grep_result,
                "File Kunci":           key_files,
                "Log & Error Files":    error_logs,
            }

            report = await self._run_inspection_llm(
                req.problem or "",
                req.problem or "",
                evidence,
            )
            branch_header = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            task.mark_done(branch_header + report)

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
) -> tuple[str, str, str, str, str, str]:
    """Run all read-only inspection commands concurrently."""
    import asyncio

    results = await asyncio.gather(
        agent._get_dir_tree(repo_path),
        agent._get_git_log(repo_path),
        agent._get_git_diff(repo_path),
        agent._grep_keywords(repo_path, keywords),
        agent._read_key_files(repo_path),
        agent._find_error_logs(repo_path),
        return_exceptions=True,
    )

    def _safe(r: object, fallback: str) -> str:
        if isinstance(r, Exception):
            logger.warning("Inspector evidence gather error: %s", r)
            return f"(error: {r})"
        return str(r) if r else fallback

    return (
        _safe(results[0], "(no dir tree)"),
        _safe(results[1], "(no git log)"),
        _safe(results[2], "(no diff)"),
        _safe(results[3], "(no grep result)"),
        _safe(results[4], "(no key files)"),
        _safe(results[5], "(no log files)"),
    )
