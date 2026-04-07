"""
DeveloperAgent – Senior Developer Orchestrator.

Workflow per request:
  1. Extract repo_url + task description from user input via LLM.
  2. Clone repo (or git pull if already present); upsert in RepoTracker.
  3. Locate or generate Docker files via SandboxRunner.
  4. Run AI CLI (gh copilot suggest / claude) to apply the requested change.
  5. Verification loop (max MAX_SANDBOX_RETRIES):
       a. docker compose up --build
       b. Detect traceback / non-zero exit
       c. If failure, send error log back to AI CLI for another fix attempt.
  6. Report: summary, files changed (git diff --name-only), commit message,
             test result.

System prompt is intentionally opinionated:
  - Zero guesswork: use `ls -R` / `grep` before editing.
  - Sandbox first: never declare success without a green Docker run.
  - Transparency: admit failures honestly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel  # noqa: F401 (kept for subclass compatibility)

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.key_store import effective_github_pat, effective_gitlab_pat
from src.memory.repo_tracker import RepoTracker
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor, CommandResult
from src.tools.code_editor_tool import (
    CodeEditorTool,
    CodePatch as _CodePatch,
    apply_patches_to_disk as _apply_patches_to_disk,
    build_ai_cli_command   as _build_ai_cli_command,
    detect_available_cli   as _detect_available_cli,
    make_search_command    as _make_search_command,
)
from src.tools.git_manager import GitManager, GitPushResult
from src.tools.git_utils import (
    inject_pat_into_url as _inject_pat_into_url,
    is_gitlab_url       as _is_gitlab_url,
    repo_name_from_url  as _repo_name_from_url,
)
from src.tools.sandbox_runner import SandboxResult, SandboxRunner

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SANDBOX_RETRIES  = 3
REPOS_BASE_DIR       = Path.home() / "sandbox_repos"

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Senior Developer Orchestrator**.

Tugasmu adalah menerima instruksi perbaikan kode atau penambahan fitur, \
lalu mengeksekusinya menggunakan AI CLI (seperti GitHub Copilot atau Claude Code).

**Aturan Utama:**
1. **Zero Guesswork**: Jangan berasumsi lokasi file. \
   Gunakan perintah `ls -R` atau `grep` jika ragu.
2. **Sandbox First**: Setiap perubahan kode **WAJIB** diuji di dalam \
   Docker Sandbox sebelum dinyatakan selesai.
3. **CLI Specialist**: Kamu tidak menulis kode secara manual. \
   Kamu memberikan perintah ke terminal seperti: \
   `gh copilot suggest -t "perbaiki bug X di file Y"` \
   atau `claude -p "tambahkan unit test untuk fungsi Z"`.
4. **Transparency**: Berikan laporan jujur jika CLI gagal melakukan perbaikan. \
   Jangan memberikan jawaban manis jika sistem error.

**Format Output Laporan:**
- **Summary**: Penjelasan singkat apa yang diubah (seperti menjelaskan ke anak 12 tahun).
- **Files Changed**: Daftar file yang diedit.
- **Commit Message**: Pesan commit sesuai standar konvensional.
- **Test Result**: Status aplikasi di dalam Docker.
"""

# ── LLM helpers prompt ────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Extract the following from the user message below and respond in JSON:
{{
  "repo_url": "<full GitHub or GitLab URL or empty string>",
  "task":     "<concise description of the code change requested>",
  "branch":   "<git branch name if explicitly mentioned in the message, otherwise empty string>"
}}

User message: {user_input}
"""

_FIX_PROMPT = """\
The sandbox returned this error after the last edit attempt:

{error_log}

Generate a single `gh copilot suggest` or `claude` CLI command that addresses \
this error.  Output ONLY the shell command, nothing else.
"""


# ── Branch confirmation state (per session, in-process) ─────────────────────
#
# When no branch is specified in the request, the agent stores the pending
# task here and returns a confirmation message to the client via the
# orchestrator.  On the user's next reply the agent resumes from this state.

_pending_branch_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}


def _resolve_branch_from_reply(user_input: str, detected_branch: str) -> str | None:
    """
    Parse the user's confirmation reply and return the branch to use.

    Returns:
        The branch name to use (detected or user-specified),
        or None if the reply is not a recognizable confirmation.
    """
    clean = user_input.strip()
    lower = clean.lower()

    # Simple confirmation → use the previously detected branch.
    if lower in _CONFIRMATION_ANSWERS:
        return detected_branch

    # Looks like an explicit branch name (no spaces, valid git chars).
    if len(clean) <= 100 and " " not in clean and re.match(r"^[\w\-./]+$", clean):
        return clean

    return None


# ── Module-level helpers re-exported from code_editor_tool ───────────────────
# These are kept here for backward compatibility with any external code that
# imports them from this module directly.

class DeveloperAgent(BaseAgent):
    """
    Orchestrates: clone → AI CLI edit → Docker sandbox verification → report.
    """

    name = "developer"

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
        # Git credentials – read once at startup from .env
        self._github_pat   = _settings.github_pat
        self._gitlab_pat   = _settings.gitlab_pat
        self._git_user_name  = _settings.git_user_name
        self._git_user_email = _settings.git_user_email
        self._repos_dir.mkdir(parents=True, exist_ok=True)
        self._tracker      = RepoTracker()
        # Detect which AI CLI can actually edit files non-interactively.
        # gh copilot suggest is excluded – it cannot write files.
        self._ai_cli = _detect_available_cli()
        if self._ai_cli:
            logger.info("DeveloperAgent: AI CLI mode → %s", self._ai_cli)
        else:
            logger.info(
                "DeveloperAgent: LLM-direct mode active "
                "(no claude CLI found; using OpenRouter to read & patch files)"
            )
        self._editor = CodeEditorTool(
            llm=self._llm,
            ai_cli=self._ai_cli,
            timeout=self._timeout,
        )

    # ── BaseAgent contract ────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)

        try:
            # ── Check if this is a reply to a branch-confirmation request ──────
            pending = _pending_branch_confirmations.get(task.session_id)
            if pending:
                branch_choice = _resolve_branch_from_reply(
                    task.user_input, pending["detected_branch"]
                )
                if branch_choice is not None:
                    del _pending_branch_confirmations[task.session_id]
                    local_path = Path(pending["local_path"])
                    await self._checkout_branch(local_path, branch_choice)
                    return await self._execute_task(
                        task,
                        repo_url=pending["repo_url"],
                        dev_task=pending["dev_task"],
                        local_path=local_path,
                        branch=branch_choice,
                    )
                # Reply was not a valid confirmation → fall through to normal parse.

            # ── Normal flow ───────────────────────────────────────────────────
            repo_url, dev_task, branch = await self._parse_instruction(task.user_input)

            if not repo_url:
                # No repo URL → list tracked repos instead.
                result = await self._list_repos()
                task.mark_done(result)
                return task

            local_path = await self._clone_or_pull(repo_url)

            # ── Branch selection ───────────────────────────────────────────────
            if branch:
                # Branch explicitly mentioned → checkout immediately.
                await self._checkout_branch(local_path, branch)
                return await self._execute_task(task, repo_url, dev_task, local_path, branch)
            else:
                # No branch specified → detect current branch and ask client.
                detected_branch = await self._get_current_branch(local_path)
                _pending_branch_confirmations[task.session_id] = {
                    "repo_url":        repo_url,
                    "dev_task":        dev_task,
                    "local_path":      str(local_path),
                    "detected_branch": detected_branch,
                }
                task.mark_done(
                    f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                    f"Repository berhasil di-clone/update. "
                    f"Branch aktif saat ini adalah: **`{detected_branch}`**\n\n"
                    f"Proses pengerjaan akan dijalankan pada branch **`{detected_branch}`**.\n\n"
                    f"Balas **`lanjutkan`** untuk melanjutkan pada branch ini, "
                    f"atau ketik nama branch yang diinginkan "
                    f"(contoh: `develop`, `feature/my-feature`)."
                )
                return task

        except Exception as exc:
            logger.exception("DeveloperAgent: unexpected error: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                f"❌ Terjadi kesalahan saat memproses permintaan: {exc}\n\n"
                "Pastikan repo URL valid dan bot memiliki akses ke repository."
            )

        return task

    async def _execute_task(
        self,
        task:       AgentTask,
        repo_url:   str,
        dev_task:   str,
        local_path: Path,
        branch:     str,
    ) -> AgentTask:
        """
        Run the actual development work after the branch has been confirmed
        and checked out.  Separated from run() so it can be called both on
        the first turn (branch explicit) and on the confirmation turn.
        """
        try:
            executor = CLIExecutor(work_dir=local_path, timeout=self._timeout)
            sandbox  = SandboxRunner(
                repo_path=local_path,
                python_image=self._python_image,
                timeout=self._timeout,
            )
            git_mgr = GitManager(
                repo_path=local_path,
                github_pat=effective_github_pat(self._github_pat),
                gitlab_pat=effective_gitlab_pat(self._gitlab_pat),
                user_name=self._git_user_name,
                user_email=self._git_user_email,
                timeout=self._timeout,
            )

            # Apply AI CLI changes then verify in sandbox.
            sandbox_result = await self._apply_and_verify(
                dev_task, executor, sandbox, max_retries=self._max_retries
            )

            # Commit & push only when sandbox is green.
            push_result: Optional[GitPushResult] = None
            if sandbox_result.succeeded:
                commit_msg  = _suggest_commit_message(dev_task, succeeded=True)
                push_result = await git_mgr.commit_and_push(commit_msg)

            # Persist final status.
            commit_hash = push_result.commit_hash if push_result else await git_mgr.get_short_hash()
            status      = "success" if (sandbox_result.succeeded and push_result and push_result.succeeded) else "failed"
            await asyncio.to_thread(
                self._tracker.update_status, repo_url, status, commit_hash
            )

            diff   = await git_mgr.get_diff()
            report = self._build_report(dev_task, sandbox_result, diff, commit_hash, push_result)
            report = f"🌿 **Branch:** `{branch}`\n\n" + report
            task.mark_done(report)

        except Exception as exc:
            logger.exception("DeveloperAgent._execute_task: error: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                f"❌ Terjadi kesalahan saat mengeksekusi task pada branch `{branch}`: {exc}\n\n"
                "Pastikan repo URL valid dan bot memiliki akses ke repository."
            )

        return task

    # ── Step 1: Parse instruction ─────────────────────────────────────────────

    async def _parse_instruction(self, user_input: str) -> tuple[str, str, str]:
        """
        Use LLM to extract repo_url, task description, and branch from free-form input.

        Returns:
            (repo_url, task_description, branch)
            repo_url is empty string when user is asking about tracked repos.
            branch is empty string when not explicitly specified.
        """
        prompt = _EXTRACT_PROMPT.format(user_input=user_input)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        raw = await self._llm.chat(messages, max_tokens=300)
        return _parse_json_fields(raw, keys=("repo_url", "task", "branch"))

    # ── Branch helpers ─────────────────────────────────────────────────────────

    async def _get_current_branch(self, local_path: Path) -> str:
        """Return the name of the currently checked-out branch."""
        executor = CLIExecutor(work_dir=local_path, timeout=15)
        result   = await executor.run("git rev-parse --abbrev-ref HEAD")
        branch   = (result.stdout or "").strip()
        return branch or "main"

    async def _checkout_branch(self, local_path: Path, branch: str) -> None:
        """
        Checkout the requested branch.

        Strategy:
          1. Try a simple `git checkout <branch>`.
          2. If the branch is not found locally, fetch all remotes and retry
             as a tracking branch (`-b <branch> origin/<branch>`).
        """
        executor = CLIExecutor(work_dir=local_path, timeout=30)
        result   = await executor.run(f"git checkout {branch}")
        if result.succeeded:
            logger.info("DeveloperAgent: checked out branch '%s'", branch)
            return
        # Branch missing locally – expand refspec, fetch all, then retry.
        await executor.run("git remote set-branches origin '*'")
        await executor.run("git fetch --all --prune")
        result = await executor.run(f"git checkout -b {branch} origin/{branch}")
        if not result.succeeded:
            raise RuntimeError(
                f"Branch '{branch}' tidak ditemukan di lokal maupun remote:\n"
                f"{result.stderr[:400]}"
            )
        logger.info("DeveloperAgent: checked out remote branch '%s'", branch)

    # ── Step 2: Clone or pull ─────────────────────────────────────────────────

    async def _clone_or_pull(self, repo_url: str) -> Path:
        """
        Git clone if the repo is new; git pull if it already exists locally.

        Injects the correct PAT into the HTTPS URL automatically:
        - GitHub repos use GITHUB_PAT  → ``<PAT>@github.com``
        - GitLab repos use GITLAB_PAT  → ``oauth2:<PAT>@gitlab.com``
        Private repos can be cloned without interactive prompts when the
        matching PAT is configured.

        Updates the RepoTracker with the current local path.

        Returns:
            Absolute Path to the local repo root.
        """
        repo_name  = _repo_name_from_url(repo_url)
        local_path = self._repos_dir / repo_name
        executor   = CLIExecutor(work_dir=self._repos_dir, timeout=120)

        # Build an authenticated URL for HTTPS repos when PAT is available.
        # Select the appropriate PAT based on the git host (GitLab vs GitHub).
        _pat     = effective_gitlab_pat(self._gitlab_pat) if _is_gitlab_url(repo_url) else effective_github_pat(self._github_pat)
        auth_url = _inject_pat_into_url(repo_url, _pat)

        if local_path.exists():
            logger.info("DeveloperAgent: repo exists, pulling %s", repo_url)
            pull_exec = CLIExecutor(work_dir=local_path, timeout=120)

            # Stash any unstaged / uncommitted changes so rebase can proceed.
            stash_result = await pull_exec.run("git stash --include-untracked")
            stashed = (
                stash_result.succeeded
                and "No local changes to save" not in stash_result.stdout
            )
            if stashed:
                logger.info("DeveloperAgent: stashed local changes before pull")

            result = await pull_exec.run(f"git pull {auth_url} HEAD --rebase")

            # Restore stashed changes regardless of pull outcome.
            if stashed:
                pop_result = await pull_exec.run("git stash pop")
                if not pop_result.succeeded:
                    logger.warning(
                        "DeveloperAgent: git stash pop failed: %s",
                        pop_result.stderr[:300],
                    )

            _raise_if_failed(result, "git pull")
        else:
            logger.info("DeveloperAgent: cloning %s", repo_url)
            result = await executor.run(f"git clone {auth_url} {repo_name}")
            _raise_if_failed(result, "git clone")

        # Upsert in database.
        await asyncio.to_thread(
            self._tracker.upsert,
            repo_name,
            repo_url,         # store the clean URL (no PAT) in DB
            str(local_path),
            status="cloned",
        )

        return local_path

    # ── Step 3+4: Apply AI CLI / LLM-direct and verify ───────────────────────

    async def _apply_and_verify(
        self,
        dev_task:    str,
        executor:    CLIExecutor,
        sandbox:     SandboxRunner,
        *,
        max_retries: int = MAX_SANDBOX_RETRIES,
    ) -> SandboxResult:
        """
        Apply code changes first (single pass), then verify in Docker sandbox.

        Flow:
          1. Apply all code changes (AI CLI or LLM-direct) – ONCE via CodeEditorTool.
          2. Run Docker sandbox ONCE to verify the result.
          3. Return SandboxResult.
        """
        repo_path = executor.work_dir

        # ── Step 1: Apply code changes ────────────────────────────────────
        logger.info("DeveloperAgent: applying code changes (single pass)")
        await self._editor.apply_changes(dev_task, repo_path)

        # ── Step 2: Run sandbox once after all changes are done ───────────
        logger.info("DeveloperAgent: all changes applied – starting sandbox verification")
        sandbox_result = await sandbox.run(max_attempts=1)

        logger.info(
            "DeveloperAgent: sandbox done | succeeded=%s phase=%s",
            sandbox_result.succeeded,
            sandbox_result.phase,
        )
        return sandbox_result

    # ── CLI mode helpers ──────────────────────────────────────────────────────
    # These delegate directly to CodeEditorTool; kept for backward compatibility
    # in case any subclass or test references them.

    async def _apply_with_cli(self, dev_task: str, executor: CLIExecutor) -> None:
        """Delegate to CodeEditorTool._apply_with_cli."""
        await self._editor._apply_with_cli(dev_task, executor)

    async def _ask_llm_for_cli_fix(self, error_log: str) -> str:
        """Ask LLM to produce a CLI fix command based on sandbox error output."""
        prompt   = _FIX_PROMPT.format(error_log=error_log[-3000:])
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        raw = await self._llm.chat(messages, max_tokens=256)
        cmd = raw.strip().strip("`").strip()
        logger.info("DeveloperAgent: LLM suggested CLI fix: %s", cmd)
        return cmd or _build_ai_cli_command("fix the error shown in logs", self._ai_cli)

    # ── LLM-direct mode helpers ───────────────────────────────────────────────
    # These delegate directly to CodeEditorTool; kept for backward compatibility.

    async def _apply_with_llm_direct(self, dev_task: str, repo_path: Path) -> None:
        """Delegate to CodeEditorTool._apply_with_llm_direct."""
        await self._editor._apply_with_llm_direct(dev_task, repo_path)

    async def _apply_llm_direct_fix(self, error_log: str, repo_path: Path) -> None:
        """Delegate to CodeEditorTool._apply_llm_direct_fix."""
        await self._editor._apply_llm_direct_fix(error_log, repo_path)

    async def _parse_code_patches(self, raw_json: str) -> list[_CodePatch]:
        """Delegate to CodeEditorTool._parse_code_patches."""
        return await self._editor._parse_code_patches(raw_json)

    async def _get_repo_tree(self, repo_path: Path) -> str:
        """Delegate to CodeEditorTool.get_repo_tree."""
        return await self._editor.get_repo_tree(repo_path)

    async def _read_relevant_files(self, repo_path: Path, hint: str) -> str:
        """Delegate to CodeEditorTool.read_relevant_files."""
        return await self._editor.read_relevant_files(repo_path, hint)

    # ── List repos ────────────────────────────────────────────────────────────

    async def _list_repos(self) -> str:
        records = await asyncio.to_thread(self._tracker.list_all)
        if not records:
            return "📂 Belum ada repository yang di-clone."

        lines = ["📂 **Repository yang di-track:**\n"]
        for r in records:
            status_emoji = "✅" if r.last_task_status == "success" else "❌" if r.last_task_status == "failed" else "📦"
            lines.append(
                f"{status_emoji} **{r.repo_name}**\n"
                f"   URL: {r.repo_url}\n"
                f"   Path: `{r.local_path}`\n"
                f"   Status: {r.last_task_status} | Commit: `{r.last_commit_hash[:8] or 'N/A'}`\n"
                f"   Ditambahkan: {r.created_at}\n"
            )
        return "\n".join(lines)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _build_report(
        self,
        dev_task:       str,
        sandbox_result: SandboxResult,
        diff:           str,
        commit_hash:    str,
        push_result:    Optional[GitPushResult] = None,
    ) -> str:
        status_emoji  = "✅" if sandbox_result.succeeded else "❌"
        phase_label   = f" [{sandbox_result.phase} phase]" if not sandbox_result.succeeded and sandbox_result.phase else ""
        test_status   = "PASSED" if sandbox_result.succeeded else f"FAILED{phase_label}\n```\n{sandbox_result.error_summary}\n```"
        files_changed = _extract_changed_files(diff) or "_(tidak ada perubahan)_"
        commit_msg    = _suggest_commit_message(dev_task, sandbox_result.succeeded)

        # Push status section
        if push_result is None:
            push_section = "_(tidak dilakukan – sandbox gagal)_"
        elif push_result.succeeded:
            push_section = f"✅ Pushed ke `{push_result.remote_url}`"
        else:
            push_section = f"❌ Push gagal: {push_result.error}"

        return (
            f"{status_emoji} **Developer Report**\n\n"
            f"**Summary:**\n{dev_task}\n\n"
            f"**Files Changed:**\n{files_changed}\n\n"
            f"**Commit Message:**\n`{commit_msg}`\n\n"
            f"**Test Result (Docker):** {test_status}\n\n"
            f"**Push Status:** {push_section}\n\n"
            f"**Commit Hash:** `{commit_hash or 'N/A'}`\n\n"
            f"**Attempts:** {sandbox_result.attempts}/{self._max_retries}"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

# Git URL helpers (is_gitlab_url, inject_pat_into_url, repo_name_from_url)
# are imported from src.tools.git_utils at the top of this file.
# Self-hosted GitLab instances are handled via the GITLAB_HOSTS setting.
#
# Code-editing helpers (_build_ai_cli_command, _make_search_command,
# _apply_diff_patch, _apply_patches_to_disk) are imported from
# src.tools.code_editor_tool at the top of this file.


def _parse_json_fields(raw: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    """
    Naively extract JSON field values from an LLM response string.
    Returns empty strings for any key not found.
    """
    import json
    results: list[str] = []

    # Try proper JSON parse first.
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            results = [str(data.get(k, "")).strip() for k in keys]
            return tuple(results)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: regex extraction.
    for key in keys:
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
        results.append(match.group(1).strip() if match else "")

    return tuple(results)


def _extract_changed_files(diff: str) -> str:
    """Return a bulleted list of file paths that appear in the diff header."""
    files = re.findall(r"^diff --git a/(.+?) b/", diff, re.MULTILINE)
    return "\n".join(f"- `{f}`" for f in files)


def _suggest_commit_message(task: str, succeeded: bool) -> str:
    """Generate a conventional commit message from the task description."""
    prefix = "feat" if "tambah" in task.lower() or "add" in task.lower() else "fix"
    suffix = "" if succeeded else " [sandbox failed]"
    short  = task[:72].rstrip(".")
    return f"{prefix}: {short}{suffix}"


def _raise_if_failed(result: CommandResult, label: str) -> None:
    """Raise RuntimeError if a CLI command returned a non-zero exit code."""
    if not result.succeeded:
        raise RuntimeError(
            f"{label} failed (exit={result.returncode}):\n{result.combined_output}"
        )
