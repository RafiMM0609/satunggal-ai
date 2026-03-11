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
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, model_validator

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.repo_tracker import RepoTracker
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor, CommandResult
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

# ── LLM-direct editing prompt (fallback when no AI CLI is installed) ──────────

_DIRECT_EDIT_SYSTEM = """\
Kamu adalah senior software engineer yang ahli membaca dan mengedit kode.
Kamu HANYA merespons dengan JSON array – tidak ada teks lain, tidak ada penjelasan, tidak ada markdown.

Format respons – pilih salah satu per file:

Opsi A – Preferred (unified diff, lebih aman):
[{"path":"path/relatif/file.ext","diff":"--- a/path\\n+++ b/path\\n@@ -1,3 +1,4 @@\\n context\\n-baris lama\\n+baris baru"}]

Opsi B – Fallback (full file content, hanya jika patch tidak mungkin):
[{"path":"path/relatif/file.ext","content":"isi lengkap file baru setelah diedit"}]

Aturan wajib:
1. Sertakan HANYA file yang benar-benar diubah.
2. Jika menggunakan "diff": format harus valid unified diff, dimulai dengan "--- a/" dan "+++ b/".
3. Jika menggunakan "content": isi LENGKAP file baru.
4. Gunakan path relatif dari root repository (contoh: "src/App.vue", "styles/main.css").
5. Mulai respons langsung dengan "[" – jangan tambahkan apapun sebelum atau sesudah JSON.
6. Pastikan JSON valid: escape newline sebagai \\n, escape backslash sebagai \\\\.
"""

_DIRECT_EDIT_USER = """\
Repository structure:
{repo_tree}

File contents:
{file_contents}

Instruction: {task}

Respond with JSON array of changed files only.
"""

_DIRECT_FIX_USER = """\
Repository structure:
{repo_tree}

File contents:
{file_contents}

Docker sandbox returned this error:
{error_log}

Fix the error. Respond with JSON array of changed files only.
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


# ── CLI detection & code-patch helpers ────────────────────────────────────────

import shutil
from dataclasses import dataclass as _dataclass

from src.tools.code_search import build_ast_index, rank_files_by_relevance


@_dataclass
class _CodePatch:
    """A single file change produced by the LLM-direct editing mode."""
    path:    str        # repo-relative path
    content: str = ""   # full new content (legacy / fallback)
    diff:    str = ""   # unified diff string (preferred over content)


class _PatchItem(BaseModel):
    """Pydantic model for strict validation of one LLM-patch JSON entry."""
    path:    str
    content: str = ""
    diff:    str = ""

    @model_validator(mode="after")
    def at_least_one_content(self) -> "_PatchItem":
        if not self.content and not self.diff:
            raise ValueError("patch entry must have 'content' or 'diff'")
        return self


def _detect_available_cli() -> str | None:
    """
    Return the name of a supported AI CLI that can NON-INTERACTIVELY edit files,
    or None (→ LLM-direct mode via OpenRouter).

    NOTE: `gh copilot suggest` is intentionally excluded here.
    It only suggests shell commands interactively – it cannot read or write
    source files, and it requires human confirmation to run the suggestion.
    Only `claude` CLI is supported because it can write files non-interactively
    via: claude -p "<task>" --allowedTools Edit,Write --output-format json
    """
    if shutil.which("claude"):
        return "claude"
    return None


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
                github_pat=self._github_pat,
                gitlab_pat=self._gitlab_pat,
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
        _pat     = self._gitlab_pat if _is_gitlab_url(repo_url) else self._github_pat
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
          1. Apply all code changes (AI CLI or LLM-direct) – ONCE.
          2. Run Docker sandbox ONCE to verify the result.
          3. Return SandboxResult.

        Sandbox is intentionally run AFTER all edits are complete, not
        interleaved. This avoids triggering long Docker build/run cycles
        on every intermediate patch and prevents compose-up hangs on
        server-style apps (Vue, React, etc.).
        """
        repo_path = executor.work_dir

        # ── Step 1: Apply code changes ────────────────────────────────────
        logger.info("DeveloperAgent: applying code changes (single pass)")
        if self._ai_cli:
            await self._apply_with_cli(dev_task, executor)
        else:
            await self._apply_with_llm_direct(dev_task, repo_path)

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

    async def _apply_with_cli(self, dev_task: str, executor: CLIExecutor) -> None:
        """Run the AI CLI tool to make code changes."""
        cmd    = _build_ai_cli_command(dev_task, self._ai_cli)
        result = await executor.run(cmd)
        if not result.succeeded:
            logger.warning("DeveloperAgent: AI CLI non-zero exit: %s", result.stderr[:500])

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

    async def _apply_with_llm_direct(self, dev_task: str, repo_path: Path) -> None:
        """
        Use internal LLM (OpenRouter) to generate and apply code patches.

        Flow:
          1. Scan repo structure.
          2. Read files most likely relevant to the task.
          3. Ask LLM to output JSON patches.
          4. Write patches to disk.
        """
        repo_tree     = await self._get_repo_tree(repo_path)
        file_contents = await self._read_relevant_files(repo_path, dev_task)

        prompt = _DIRECT_EDIT_USER.format(
            repo_tree=repo_tree,
            file_contents=file_contents,
            task=dev_task,
        )
        messages = [
            {"role": "system", "content": _DIRECT_EDIT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        raw_json = await self._llm.chat(messages, max_tokens=8192)
        patches  = await self._parse_code_patches(raw_json)
        executor = CLIExecutor(work_dir=repo_path, timeout=60)
        await _apply_patches_to_disk(patches, repo_path, executor)
        logger.info("DeveloperAgent: LLM-direct applied %d file patch(es)", len(patches))

    async def _apply_llm_direct_fix(self, error_log: str, repo_path: Path) -> None:
        """LLM-direct variant for sandbox error retry."""
        repo_tree     = await self._get_repo_tree(repo_path)
        file_contents = await self._read_relevant_files(repo_path, error_log[:200])

        prompt = _DIRECT_FIX_USER.format(
            repo_tree=repo_tree,
            file_contents=file_contents,
            error_log=error_log[-3000:],
        )
        messages = [
            {"role": "system", "content": _DIRECT_EDIT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        raw_json = await self._llm.chat(messages, max_tokens=8192)
        patches  = await self._parse_code_patches(raw_json)
        executor = CLIExecutor(work_dir=repo_path, timeout=60)
        await _apply_patches_to_disk(patches, repo_path, executor)
        logger.info("DeveloperAgent: LLM-direct fix applied %d file patch(es)", len(patches))

    # ── JSON patch validation (3-tier + LLM retry) ───────────────────────────

    async def _parse_code_patches(self, raw_json: str) -> list[_CodePatch]:
        """
        Parse LLM output into a list of _CodePatch objects using a 3-tier
        validation strategy with an automatic LLM-retry fallback.

        Tier 1 – Strict   : json.loads + Pydantic model_validate on every item.
        Tier 2 – Partial  : json.loads + Pydantic; silently drop invalid items.
        Tier 3 – Regex    : regex extraction when JSON is totally malformed.
        Tier 4 – LLM retry: final attempt via LLM re-prompt if 0 patches found.
        """
        import json

        # Strip markdown code fences if present.
        clean = re.sub(r"^```[a-z]*\n?", "", raw_json.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\n?```$", "", clean.strip())

        # ── Tier 1: strict JSON + Pydantic ──────────────────────────────────
        try:
            items = json.loads(clean)
            if isinstance(items, list):
                patches = [
                    _CodePatch(path=item.path, content=item.content, diff=item.diff)
                    for raw in items
                    for item in [_PatchItem.model_validate(raw)]
                ]
                if patches:
                    logger.debug("_parse_code_patches: Tier 1 OK – %d patches", len(patches))
                    return patches
        except (json.JSONDecodeError, Exception):
            pass

        # ── Tier 2: partial JSON (skip invalid items) ────────────────────────
        patches: list[_CodePatch] = []
        try:
            items = json.loads(clean)
            if isinstance(items, list):
                for raw in items:
                    try:
                        item = _PatchItem.model_validate(raw)
                        patches.append(
                            _CodePatch(path=item.path, content=item.content, diff=item.diff)
                        )
                    except ValidationError as ve:
                        logger.warning(
                            "_parse_code_patches: Tier 2 skipped item (%s)", ve.error_count()
                        )
        except json.JSONDecodeError:
            pass

        if patches:
            logger.debug("_parse_code_patches: Tier 2 OK – %d patches", len(patches))
            return patches

        # ── Tier 3: regex extraction ───────────────────────────────────────
        for m in re.finditer(
            r'"path"\s*:\s*"([^"]+)".*?(?:"diff"\s*:\s*"((?:[^"\\]|\\.)*)"'
            r'|"content"\s*:\s*"((?:[^"\\]|\\.)*?)")',
            clean,
            re.DOTALL,
        ):
            path, diff_raw, content_raw = m.group(1), m.group(2) or "", m.group(3) or ""
            # Unescape \n and \\ in the captured value.
            diff_val    = diff_raw.replace("\\n", "\n").replace("\\\\", "\\")
            content_val = content_raw.replace("\\n", "\n").replace("\\\\", "\\")
            if path:
                patches.append(_CodePatch(path=path.strip(), diff=diff_val, content=content_val))

        if patches:
            logger.warning("_parse_code_patches: Tier 3 regex – %d patches", len(patches))
            return patches

        # ── Tier 4: LLM retry ───────────────────────────────────────────────
        logger.warning(
            "_parse_code_patches: all tiers failed – requesting LLM reformat; raw=%s",
            raw_json[:200],
        )
        retry_messages = [
            {
                "role": "system",
                "content": _DIRECT_EDIT_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON.\n"
                    "Previous response:\n"
                    f"{raw_json[:1000]}\n\n"
                    "Output ONLY a valid JSON array: "
                    '[{\"path\": \"...\", \"diff\": \"...\"}]  '
                    "or [{\"path\": \"...\", \"content\": \"...\"}].  "
                    "Start with \"[\" immediately."
                ),
            },
        ]
        try:
            retry_raw  = await self._llm.chat(retry_messages, max_tokens=4096)
            retry_clean = re.sub(r"^```[a-z]*\n?", "", retry_raw.strip(), flags=re.MULTILINE)
            retry_clean = re.sub(r"\n?```$", "", retry_clean.strip())
            items = json.loads(retry_clean)
            if isinstance(items, list):
                for raw in items:
                    try:
                        item = _PatchItem.model_validate(raw)
                        patches.append(
                            _CodePatch(path=item.path, content=item.content, diff=item.diff)
                        )
                    except ValidationError:
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.error("_parse_code_patches: Tier 4 LLM retry failed (%s)", exc)

        if not patches:
            logger.error("_parse_code_patches: 0 patches extracted after all tiers")
        else:
            logger.info("_parse_code_patches: Tier 4 LLM retry – %d patches", len(patches))
        return patches

    async def _get_repo_tree(self, repo_path: Path) -> str:
        """Return a trimmed directory listing for the repo."""
        executor = CLIExecutor(work_dir=repo_path, timeout=15)
        # Exclude hidden dirs, node_modules, __pycache__, venv, .git
        result = await executor.run(
            "find . -not \\( -path './.git' -prune \\) "
            "-not \\( -path './node_modules' -prune \\) "
            "-not \\( -path './__pycache__' -prune \\) "
            "-not \\( -path './.venv' -prune \\) "
            "-type f | sort | head -120"
        )
        return result.stdout or "(empty repo)"

    async def _read_relevant_files(self, repo_path: Path, hint: str) -> str:
        """
        Read files most likely relevant to the task.

        Strategy:
          1. rg (ripgrep) / grep keyword search to collect candidate files.
          2. Build AST symbol index for the repo (tree-sitter multi-language).
          3. Re-rank candidates + AST top picks via TF-IDF cosine similarity.
          4. Cap total read at 80 KB to stay within LLM context.
        """
        executor = CLIExecutor(work_dir=repo_path, timeout=20)

        # Extract simple keywords from hint (skip short words).
        keywords = [w for w in re.findall(r"\w+", hint) if len(w) > 3][:6]

        candidate_files: list[str] = []
        if keywords:
            pattern       = "|".join(keywords)
            search_cmd    = _make_search_command(pattern)
            search_result = await executor.run(search_cmd)
            candidate_files = [
                line.strip().lstrip("./")
                for line in search_result.stdout.splitlines()
                if line.strip()
            ]

        # Always include common entry-point files regardless of search results.
        common = [
            "index.html", "App.vue", "App.jsx", "App.tsx", "main.py",
            "src/App.vue", "src/App.jsx", "src/App.tsx",
            "src/main.css", "src/style.css", "src/assets/main.css",
            "src/styles/main.css", "src/index.css",
        ]
        for f in common:
            if (repo_path / f).exists() and f not in candidate_files:
                candidate_files.append(f)

        # ── AST index + TF-IDF re-ranking ────────────────────────────────────
        try:
            symbol_index    = await asyncio.to_thread(build_ast_index, repo_path)
            candidate_files = rank_files_by_relevance(
                candidate_files, symbol_index, hint
            )
            # Supplement with high-scoring files not yet in the candidate list.
            new_from_index = [
                p for p in symbol_index if p not in set(candidate_files)
            ]
            extra = rank_files_by_relevance(new_from_index, symbol_index, hint)
            candidate_files += extra[:3]   # cap extra to avoid bloat
        except Exception as exc:  # noqa: BLE001
            logger.warning("_read_relevant_files: AST/TF-IDF ranking failed (%s)", exc)

        MAX_BYTES = 80_000
        sections:  list[str] = []
        total      = 0

        for rel_path in candidate_files:
            abs_path = repo_path / rel_path
            if not abs_path.is_file():
                continue
            try:
                text = abs_path.read_text(errors="replace")
            except Exception:
                continue
            chunk = f"### {rel_path}\n```\n{text}\n```\n"
            if total + len(chunk) > MAX_BYTES:
                break
            sections.append(chunk)
            total += len(chunk)

        return "\n".join(sections) or "(no relevant files read)"

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


def _build_ai_cli_command(task: str, cli: str = "claude") -> str:
    """
    Build the shell command to invoke the AI CLI for non-interactive file editing.

    - claude: uses --allowedTools to permit file Read/Edit/Write so it can
              actually modify source files without interactive confirmation.
    NOTE: `gh copilot suggest` is NOT used here because it cannot edit files.
    """
    safe_task = task.replace('"', '\\"').replace("'", "\\'")
    if cli == "claude":
        return (
            f'claude -p "{safe_task}" '
            f'--allowedTools "Read,Edit,Write,Bash" '
            f'--output-format json'
        )
    # Fallback – should not be reached with current detection logic.
    return f'echo "No supported AI CLI available for: {safe_task}"'


def _make_search_command(pattern: str) -> str:
    """
    Build a file-search shell command for the given keyword pattern.

    Prefers ripgrep (rg) when available – it is significantly faster than
    GNU grep on large repos and has better default ignore behaviour (.gitignore
    is respected automatically).

    Falls back to GNU grep with explicit --include globs when rg is not found.
    """
    _INCLUDE_EXTS = (
        "html", "css", "scss", "sass",
        "js", "ts", "tsx", "jsx",
        "vue", "svelte", "py", "json",
    )
    safe_pattern = pattern.replace("'", r"\'")  # basic shell-safety

    if shutil.which("rg"):
        # ripgrep: faster, respects .gitignore, multi-type in one pass.
        type_args = " ".join(
            f"--type-add '{ext}:*.{ext}' --type {ext}" for ext in _INCLUDE_EXTS
        )
        return (
            f"rg --files-with-matches -e '{safe_pattern}' "
            f"{type_args} "
            f"--max-count 1 . 2>/dev/null | head -20"
        )

    # Fallback: GNU grep.
    includes = " ".join(f"--include='*.{e}'" for e in _INCLUDE_EXTS)
    return (
        f"grep -rl {includes} "
        f"-e '{safe_pattern}' . 2>/dev/null | head -20"
    )


async def _apply_diff_patch(
    patch_text: str,
    repo_path:  Path,
    executor:   CLIExecutor,
) -> bool:
    """
    Apply a unified diff patch to the repository.

    Strategy:
      1. Use the system `patch` CLI (fastest, most battle-tested on Linux).
      2. Fall back to the Python `patch` library if the CLI is unavailable.

    *repo_path* is used as the working directory so relative paths in the diff
    resolve correctly.  --backup writes .orig files which can be used for
    manual recovery if needed.

    Returns True on success, False on failure.
    """
    if not patch_text.strip():
        return False

    if shutil.which("patch"):
        # Write patch text to a temp file; shell-redirect into `patch`.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(patch_text)
            tmp_path = tmp.name
        try:
            result = await executor.run(
                f"patch -p1 --backup --forward --reject-file=- < {tmp_path}"
            )
            if result.succeeded:
                logger.info("_apply_diff_patch: patch CLI succeeded")
                return True
            logger.warning(
                "_apply_diff_patch: patch CLI failed (exit=%d): %s",
                result.returncode, result.stderr[:400],
            )
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
    else:
        # Fallback: Python `patch` library.
        try:
            import patch as patch_lib  # type: ignore

            ps = patch_lib.fromstring(patch_text.encode("utf-8"))
            success = bool(ps and ps.apply(root=str(repo_path)))
            if success:
                logger.info("_apply_diff_patch: python-patch library succeeded")
            else:
                logger.warning("_apply_diff_patch: python-patch library failed")
            return success
        except Exception as exc:  # noqa: BLE001
            logger.error("_apply_diff_patch: python-patch error (%s)", exc)
            return False


async def _apply_patches_to_disk(
    patches:   list[_CodePatch],
    repo_root: Path,
    executor:  CLIExecutor,
) -> None:
    """
    Write each patch to disk.

    Per-patch strategy:
      1. If *patch.diff* is non-empty → try unified-diff apply via
         `_apply_diff_patch`.  On failure, fall back to full-content write.
      2. If *patch.content* is non-empty → write full file content with a
         .bak backup of the original.
      3. If neither field is populated → log and skip.
    """
    for patch in patches:
        target = repo_root / patch.path
        target.parent.mkdir(parents=True, exist_ok=True)

        if patch.diff:
            ok = await _apply_diff_patch(patch.diff, repo_root, executor)
            if ok:
                logger.info("DeveloperAgent: diff-patched → %s", target)
                continue
            # Diff failed – fall through to full-content write if available.
            if not patch.content:
                logger.error(
                    "DeveloperAgent: diff patch failed and no content fallback for %s", target
                )
                continue
            logger.warning(
                "DeveloperAgent: diff failed for %s – falling back to full-content write", target
            )

        if patch.content:
            # Write a .bak backup before overwriting.
            if target.is_file():
                bak_path = target.with_suffix(target.suffix + ".bak")
                try:
                    bak_path.write_bytes(target.read_bytes())
                except Exception:  # noqa: BLE001
                    pass  # non-fatal
            target.write_text(patch.content, encoding="utf-8")
            logger.info("DeveloperAgent: wrote patch → %s", target)
        else:
            logger.warning("DeveloperAgent: patch for %s has no diff or content – skipping", target)


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
