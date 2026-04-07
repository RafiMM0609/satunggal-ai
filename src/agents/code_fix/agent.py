"""
CodeFixAgent – Auto-Detect + Auto-Fix Pipeline.

Peran:
  Gabungan pipeline: Inspector (read) → LLM recommendation extractor → Developer (write).

  1. **Inspection Phase**: Jalankan DeveloperInspectorAgent untuk menemukan masalah.
  2. **Planning Phase**: LLM mengekstrak task-task perbaikan yang spesifik dari laporan inspeksi.
  3. **Fix Phase**: CodeEditorTool menerapkan perbaikan ke repo.
  4. **Verification Phase**: Docker sandbox memverifikasi hasil.
  5. **Push Phase**: Commit dan push jika sandbox hijau.

Perbedaan dengan agent lain:
  - DeveloperInspectorAgent → HANYA menemukan masalah, tidak memperbaiki.
  - DeveloperAgent → memperbaiki berdasarkan instruksi user (tidak ada fase inspect).
  - CodeFixAgent → temukan DULU lalu auto-perbaiki (combined pipeline).

Penggunaan:
  "temukan dan perbaiki semua bug di repo github.com/foo/bar"
  "diagnosa masalah lalu otomatis perbaiki"
  "cari masalah dan langsung fix repo ini"
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.key_store import effective_github_pat, effective_gitlab_pat
from src.memory.repo_tracker import RepoTracker
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor
from src.tools.code_editor_tool import (
    CodeEditorTool,
    detect_available_cli,
)
from src.tools.git_manager import GitManager, GitPushResult
from src.tools.git_utils import (
    inject_pat_into_url as _inject_pat_into_url,
    is_gitlab_url       as _is_gitlab_url,
    repo_name_from_url  as _repo_name_from_url,
)
from src.tools.sandbox_runner import SandboxResult, SandboxRunner

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Extract the following from the user message below and respond in JSON:
{{
  "repo_url": "<full GitHub or GitLab URL or empty string>",
  "problem":  "<brief description of what to find and fix>",
  "branch":   "<git branch name if explicitly mentioned, otherwise empty string>"
}}

User message: {user_input}
"""

_EXTRACT_SYSTEM = "Extract structured data from the user message. Reply in JSON only."

_FIX_PLAN_SYSTEM = """\
Kamu adalah senior software engineer yang membaca laporan inspeksi kode dan menghasilkan daftar task perbaikan.

Dari laporan inspeksi yang diberikan, ekstrak semua REKOMENDASI PERBAIKAN yang konkret dan actionable.
Buat task deskripsi yang singkat dan spesifik untuk setiap perbaikan.

Format output WAJIB (JSON array):
[
  "Fix: <deskripsi perbaikan spesifik 1>",
  "Fix: <deskripsi perbaikan spesifik 2>",
  ...
]

Aturan:
- Maksimal 5 task terpenting (lihat MAX_FIX_TASKS).
- Setiap task harus SPESIFIK: sebutkan nama file atau fungsi jika tersedia.
- Hanya task yang CONFIRMED atau LIKELY dari laporan inspeksi.
- Balas HANYA dengan JSON array yang valid.
"""


# Maximum characters of inspection report sent to the LLM fix-plan extractor.
_MAX_INSPECTION_CHARS_FOR_PLAN = 8_000

# Maximum number of fix tasks to apply in one CodeFix run (must match prompt above).
MAX_FIX_TASKS = 5

# ── Branch confirmation state ──────────────────────────────────────────────────

_pending_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}


def _resolve_branch_from_reply(user_input: str, detected_branch: str) -> str | None:
    clean = user_input.strip()
    lower = clean.lower()
    if lower in _CONFIRMATION_ANSWERS:
        return detected_branch
    if len(clean) <= 100 and " " not in clean and re.match(r"^[\w\-./]+$", clean):
        return clean
    return None


# ── Agent ──────────────────────────────────────────────────────────────────────


class CodeFixAgent(BaseAgent):
    """
    Combined inspect → auto-fix pipeline agent.

    Phase 1: DeveloperInspectorAgent runs a read-only inspection.
    Phase 2: LLM extracts actionable fix tasks from the inspection report.
    Phase 3: CodeEditorTool applies the fixes.
    Phase 4: Docker sandbox verifies the result.
    Phase 5: GitManager commits and pushes if verification passes.
    """

    name = "code_fix"

    def __init__(
        self,
        llm:       LLMClient | None = None,
        repos_dir: Path | str | None = None,
    ) -> None:
        from config.settings import get_settings
        _settings          = get_settings()
        self._llm          = llm or LLMClient()
        self._repos_dir    = (
            Path(repos_dir).expanduser()
            if repos_dir
            else Path(_settings.sandbox_repos_dir).expanduser()
        )
        self._python_image = _settings.sandbox_python_image
        self._timeout      = _settings.sandbox_timeout
        self._max_retries  = _settings.sandbox_max_retries
        self._github_pat   = _settings.github_pat
        self._gitlab_pat   = _settings.gitlab_pat
        self._git_user_name  = _settings.git_user_name
        self._git_user_email = _settings.git_user_email
        self._repos_dir.mkdir(parents=True, exist_ok=True)
        self._tracker = RepoTracker()
        self._ai_cli  = detect_available_cli()
        self._editor  = CodeEditorTool(
            llm=self._llm,
            ai_cli=self._ai_cli,
            timeout=self._timeout,
        )
        logger.info(
            "CodeFixAgent: AI CLI mode=%s",
            self._ai_cli or "LLM-direct",
        )

    # ── BaseAgent contract ────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)

        try:
            # Handle branch confirmation replies
            pending = _pending_confirmations.get(task.session_id)
            if pending:
                branch_choice = _resolve_branch_from_reply(
                    task.user_input, pending["detected_branch"]
                )
                if branch_choice is not None:
                    del _pending_confirmations[task.session_id]
                    local_path = Path(pending["local_path"])
                    await self._checkout_branch(local_path, branch_choice)
                    return await self._execute_fix(
                        task,
                        repo_url=pending["repo_url"],
                        problem=pending["problem"],
                        local_path=local_path,
                        branch=branch_choice,
                    )

            # Parse instruction
            repo_url, problem, branch = await self._parse_instruction(task.user_input)

            if not repo_url:
                task.mark_done(
                    "⚠️ **URL repository tidak ditemukan.**\n\n"
                    "Untuk auto-fix, sertakan URL repo GitHub/GitLab.\n"
                    "Contoh: `temukan dan perbaiki bug di repo github.com/foo/bar`"
                )
                return task

            local_path = await self._clone_or_pull(repo_url)

            if branch:
                await self._checkout_branch(local_path, branch)
                return await self._execute_fix(task, repo_url, problem, local_path, branch)
            else:
                detected_branch = await self._get_current_branch(local_path)
                _pending_confirmations[task.session_id] = {
                    "repo_url":        repo_url,
                    "problem":         problem,
                    "local_path":      str(local_path),
                    "detected_branch": detected_branch,
                }
                task.mark_done(
                    f"⚠️ **Branch tidak ditentukan.**\n\n"
                    f"Branch aktif: **`{detected_branch}`**\n\n"
                    f"Pipeline inspect → auto-fix akan dijalankan pada branch ini.\n\n"
                    f"Balas **`lanjutkan`** untuk melanjutkan, atau ketik nama branch lain."
                )
                return task

        except Exception as exc:
            logger.exception("CodeFixAgent: unexpected error: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                f"❌ Terjadi kesalahan: {exc}\n\n"
                "Pastikan repo URL valid dan bot memiliki akses ke repository."
            )

        return task

    # ── Execution pipeline ────────────────────────────────────────────────────

    async def _execute_fix(
        self,
        task:       AgentTask,
        repo_url:   str,
        problem:    str,
        local_path: Path,
        branch:     str,
    ) -> AgentTask:
        """Run the full inspect → fix → verify → push pipeline."""
        try:
            # ── Phase 1: Inspection ───────────────────────────────────────────
            inspection_report = await self._run_inspection(task, local_path, problem)

            # ── Phase 2: Extract fix tasks from report ────────────────────────
            fix_tasks = await self._extract_fix_tasks(inspection_report)

            if not fix_tasks:
                task.mark_done(
                    f"🌿 **Branch:** `{branch}`\n\n"
                    "✅ **Inspeksi selesai – tidak ada masalah yang membutuhkan perbaikan otomatis.**\n\n"
                    f"**Laporan Inspeksi:**\n\n{inspection_report[:2000]}"
                )
                return task

            fix_task_desc = "\n".join(f"- {t}" for t in fix_tasks)
            combined_task = (
                f"{problem or 'Auto-fix berdasarkan hasil inspeksi'}\n\n"
                f"Perbaikan yang diperlukan:\n{fix_task_desc}"
            )
            logger.info(
                "CodeFixAgent: %d fix tasks identified → applying changes",
                len(fix_tasks),
            )

            # ── Phase 3: Apply fixes ──────────────────────────────────────────
            await self._editor.apply_changes(combined_task, local_path)

            # ── Phase 4: Sandbox verification ─────────────────────────────────
            sandbox = SandboxRunner(
                repo_path=local_path,
                python_image=self._python_image,
                timeout=self._timeout,
            )
            sandbox_result = await sandbox.run(max_attempts=1)

            # ── Phase 5: Commit & push if sandbox passed ───────────────────────
            push_result: Optional[GitPushResult] = None
            if sandbox_result.succeeded:
                git_mgr = GitManager(
                    repo_path=local_path,
                    github_pat=effective_github_pat(self._github_pat),
                    gitlab_pat=effective_gitlab_pat(self._gitlab_pat),
                    user_name=self._git_user_name,
                    user_email=self._git_user_email,
                    timeout=self._timeout,
                )
                commit_msg  = f"fix: auto-fix via CodeFixAgent [{problem[:60] or 'auto'}]"
                push_result = await git_mgr.commit_and_push(commit_msg)

            commit_hash = ""
            if push_result:
                commit_hash = push_result.commit_hash or ""
            status = "success" if (sandbox_result.succeeded and push_result and push_result.succeeded) else "failed"
            await asyncio.to_thread(
                self._tracker.update_status, repo_url, status, commit_hash
            )

            report = self._build_report(
                inspection_report, fix_tasks, sandbox_result, push_result, commit_hash, branch
            )
            task.mark_done(report)

        except Exception as exc:
            logger.exception("CodeFixAgent._execute_fix: error: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                f"❌ Terjadi kesalahan pada pipeline inspect → fix: {exc}\n\n"
                "Pastikan repo URL valid dan bot memiliki akses ke repository."
            )

        return task

    # ── Phase 1: Inspection ───────────────────────────────────────────────────

    async def _run_inspection(
        self,
        task:       AgentTask,
        local_path: Path,
        problem:    str,
    ) -> str:
        """
        Run DeveloperInspectorAgent on a copy of the task to obtain an inspection report.

        Returns the inspection report as a string.
        """
        from src.agents.developer_inspector.agent import DeveloperInspectorAgent
        from src.memory.state import AgentTask as _AgentTask

        inspection_task = _AgentTask(
            session_id=task.session_id + "_inspect",
            user_input=(
                f"Inspeksi repo di {local_path} untuk: {problem or 'masalah umum'}"
            ),
        )
        inspection_task.metadata["local_path_override"] = str(local_path)

        inspector = DeveloperInspectorAgent(llm=self._llm)
        inspection_task = await inspector.run(inspection_task)
        return inspection_task.result or ""

    # ── Phase 2: Extract fix tasks ────────────────────────────────────────────

    async def _extract_fix_tasks(self, inspection_report: str) -> list[str]:
        """Ask LLM to extract actionable fix tasks from the inspection report."""
        if not inspection_report.strip():
            return []

        try:
            import json
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _FIX_PLAN_SYSTEM},
                    {"role": "user",   "content": inspection_report[:_MAX_INSPECTION_CHARS_FOR_PLAN]},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )
            clean = response.strip()
            # Strip markdown fences
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean, flags=re.MULTILINE)
                clean = re.sub(r"\n?```$", "", clean.strip())
            data = json.loads(clean)
            if isinstance(data, list):
                return [str(t) for t in data if isinstance(t, str)][:MAX_FIX_TASKS]
        except Exception as exc:
            logger.warning("CodeFixAgent._extract_fix_tasks: failed (%s)", exc)

        return []

    # ── Git helpers ───────────────────────────────────────────────────────────

    async def _parse_instruction(self, user_input: str) -> tuple[str, str, str]:
        """Extract repo_url, problem, branch from free-form input via LLM."""
        import json
        prompt = _EXTRACT_PROMPT.format(user_input=user_input)
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        raw = await self._llm.chat(messages, max_tokens=300)

        results: list[str] = []
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                results = [str(data.get(k, "")).strip() for k in ("repo_url", "problem", "branch")]
                return (results[0], results[1], results[2])
        except (json.JSONDecodeError, ValueError):
            pass

        for key in ("repo_url", "problem", "branch"):
            match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
            results.append(match.group(1).strip() if match else "")
        return (
            results[0] if len(results) > 0 else "",
            results[1] if len(results) > 1 else "",
            results[2] if len(results) > 2 else "",
        )

    async def _clone_or_pull(self, repo_url: str) -> Path:
        """Clone or pull the repository."""
        repo_name  = _repo_name_from_url(repo_url)
        local_path = self._repos_dir / repo_name
        executor   = CLIExecutor(work_dir=self._repos_dir, timeout=120)

        _pat     = (effective_gitlab_pat(self._gitlab_pat)
                    if _is_gitlab_url(repo_url)
                    else effective_github_pat(self._github_pat))
        auth_url = _inject_pat_into_url(repo_url, _pat)

        if local_path.exists():
            pull_exec = CLIExecutor(work_dir=local_path, timeout=120)
            stash = await pull_exec.run("git stash --include-untracked")
            stashed = stash.succeeded and "No local changes to save" not in stash.stdout
            await pull_exec.run(f"git pull {auth_url} HEAD --rebase")
            if stashed:
                await pull_exec.run("git stash pop")
        else:
            result = await executor.run(f"git clone {auth_url} {repo_name}")
            if not result.succeeded:
                raise RuntimeError(
                    f"git clone failed:\n{result.combined_output}"
                )

        await asyncio.to_thread(
            self._tracker.upsert,
            repo_name, repo_url, str(local_path), status="cloned",
        )
        return local_path

    async def _get_current_branch(self, local_path: Path) -> str:
        executor = CLIExecutor(work_dir=local_path, timeout=15)
        result   = await executor.run("git rev-parse --abbrev-ref HEAD")
        return (result.stdout or "").strip() or "main"

    async def _checkout_branch(self, local_path: Path, branch: str) -> None:
        executor = CLIExecutor(work_dir=local_path, timeout=30)
        result   = await executor.run(f"git checkout {branch}")
        if result.succeeded:
            return
        await executor.run("git remote set-branches origin '*'")
        await executor.run("git fetch --all --prune")
        result = await executor.run(f"git checkout -b {branch} origin/{branch}")
        if not result.succeeded:
            raise RuntimeError(
                f"Branch '{branch}' tidak ditemukan: {result.stderr[:400]}"
            )

    # ── Report ────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        inspection_report: str,
        fix_tasks:         list[str],
        sandbox_result:    SandboxResult,
        push_result:       Optional[GitPushResult],
        commit_hash:       str,
        branch:            str,
    ) -> str:
        status_emoji = "✅" if sandbox_result.succeeded else "❌"
        test_status  = (
            "PASSED"
            if sandbox_result.succeeded
            else f"FAILED\n```\n{sandbox_result.error_summary}\n```"
        )

        if push_result is None:
            push_section = "_(tidak dilakukan – sandbox gagal)_"
        elif push_result.succeeded:
            push_section = f"✅ Pushed ke `{push_result.remote_url}`"
        else:
            push_section = f"❌ Push gagal: {push_result.error}"

        fix_list = "\n".join(f"  - {t}" for t in fix_tasks) or "_(tidak ada task perbaikan yang teridentifikasi)_"

        # Include inspection summary (first 1500 chars)
        inspection_summary = inspection_report[:1500]
        if len(inspection_report) > 1500:
            inspection_summary += "\n\n_[laporan inspeksi dipotong untuk ringkasan]_"

        return (
            f"🌿 **Branch:** `{branch}`\n\n"
            f"{status_emoji} **CodeFix Report (Inspect → Auto-Fix)**\n\n"
            f"**Fix Tasks yang Diterapkan:**\n{fix_list}\n\n"
            f"**Test Result (Docker):** {test_status}\n\n"
            f"**Push Status:** {push_section}\n\n"
            f"**Commit Hash:** `{commit_hash or 'N/A'}`\n\n"
            f"---\n\n"
            f"**📋 Ringkasan Inspeksi:**\n\n{inspection_summary}"
        )
