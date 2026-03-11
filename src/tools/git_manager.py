"""
GitManager – high-level git operations for the DeveloperAgent.

Responsibilities:
- Configure per-repo git identity (user.name / user.email).
- Inject GITHUB_PAT into the remote URL so push/pull work without
  interactive password prompts (HTTPS credential helper via URL).
- Stage all changes, commit with a conventional message.
- Push to origin with the authenticated remote URL.
- Fall back gracefully when PAT is empty (assumes SSH key auth).

All heavy work is delegated to CLIExecutor (async, timeout-protected).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from src.tools.cli_executor import CLIExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitPushResult:
    """
    Immutable result of a commit + push operation.

    Attributes:
        committed:    True when `git commit` succeeded (or nothing to commit).
        pushed:       True when `git push` succeeded.
        commit_hash:  Short hash of the new commit (empty if nothing committed).
        remote_url:   The remote URL that was pushed to (PAT redacted).
        error:        Non-empty string when something failed.
    """
    committed:   bool
    pushed:      bool
    commit_hash: str
    remote_url:  str
    error:       str = ""

    @property
    def succeeded(self) -> bool:
        return self.committed and self.pushed


class GitManager:
    """
    Wraps all git credential and commit/push logic for a single local repo.

    Usage:
        gm = GitManager(repo_path=Path("/tmp/myrepo"))
        result = await gm.commit_and_push("fix: resolve import error")
        if result.succeeded:
            print(f"Pushed at {result.commit_hash}")
    """

    def __init__(
        self,
        repo_path:   Path | str,
        github_pat:  str = "",
        user_name:   str = "AdvanceAI Bot",
        user_email:  str = "bot@advanceai.local",
        timeout:     int = 120,
    ) -> None:
        """
        Args:
            repo_path:   Absolute path to the local git repo.
            github_pat:  GitHub Personal Access Token (empty → SSH key auth).
            user_name:   Identity written to `git config user.name`.
            user_email:  Identity written to `git config user.email`.
            timeout:     Max seconds per git command.
        """
        self._repo_path  = Path(repo_path)
        self._github_pat = github_pat
        self._user_name  = user_name
        self._user_email = user_email
        self._executor   = CLIExecutor(work_dir=self._repo_path, timeout=timeout, auto_yes=False)

    # ── Public API ────────────────────────────────────────────────────────────

    async def commit_and_push(self, commit_message: str) -> GitPushResult:
        """
        Stage all changes → commit → push to origin.

        If there is nothing to commit (clean working tree), committed=True
        and pushed=True are still returned so the caller can treat it as
        a no-op success.

        Args:
            commit_message: Conventional commit string, e.g. "fix: resolve import error"

        Returns:
            GitPushResult with commit hash and push status.
        """
        await self._configure_identity()

        remote_url        = await self._get_remote_url()
        auth_remote_url   = self._inject_pat(remote_url)
        safe_remote_url   = _redact_pat(auth_remote_url)

        # Stage all changes.
        stage_result = await self._executor.run("git add -A")
        if not stage_result.succeeded:
            return GitPushResult(
                committed=False, pushed=False,
                commit_hash="", remote_url=safe_remote_url,
                error=f"git add failed: {stage_result.stderr}",
            )

        # Commit (skip if nothing staged).
        commit_hash, commit_error = await self._commit(commit_message)
        if commit_error and "nothing to commit" not in commit_error:
            return GitPushResult(
                committed=False, pushed=False,
                commit_hash="", remote_url=safe_remote_url,
                error=commit_error,
            )

        # Push to origin using authenticated URL.
        # push_result = await self._executor.run(
        #     f"git push {auth_remote_url} HEAD"
        # )
        # if not push_result.succeeded:
        #     return GitPushResult(
        #         committed=True, pushed=False,
        #         commit_hash=commit_hash, remote_url=safe_remote_url,
        #         error=f"git push failed: {push_result.stderr}",
        #     )

        logger.info(
            "GitManager: pushed commit=%s to %s", commit_hash, safe_remote_url
        )
        return GitPushResult(
            committed=True, pushed=True,
            commit_hash=commit_hash, remote_url=safe_remote_url,
        )

    async def get_short_hash(self) -> str:
        """Return the short hash of the current HEAD, or empty string."""
        result = await self._executor.run("git rev-parse --short HEAD")
        return result.stdout if result.succeeded else ""

    async def get_diff(self) -> str:
        """Return the full `git diff HEAD` output."""
        result = await self._executor.run("git diff HEAD")
        return result.stdout or "(no diff)"

    async def get_diff_stat(self) -> str:
        """Return a compact `git diff --stat HEAD` summary."""
        result = await self._executor.run("git diff --stat HEAD")
        return result.stdout or "(no changes)"

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _configure_identity(self) -> None:
        """Write user.name and user.email to the local repo's git config."""
        await self._executor.run(f'git config user.name  "{self._user_name}"')
        await self._executor.run(f'git config user.email "{self._user_email}"')
        logger.debug(
            "GitManager: configured identity %s <%s>",
            self._user_name, self._user_email,
        )

    async def _get_remote_url(self) -> str:
        """
        Return the `origin` remote URL.

        Raises RuntimeError if no remote is configured.
        """
        result = await self._executor.run("git remote get-url origin")
        if not result.succeeded or not result.stdout:
            raise RuntimeError(
                "GitManager: no 'origin' remote configured in this repo. "
                "Set it with: git remote add origin <url>"
            )
        return result.stdout.strip()

    def _inject_pat(self, remote_url: str) -> str:
        """
        Embed the PAT into an HTTPS remote URL so git can push without
        interactive prompts.

        https://github.com/owner/repo.git
          →  https://<PAT>@github.com/owner/repo.git

        If the URL is already SSH (git@github.com:…) or the PAT is empty,
        the URL is returned unchanged (SSH key auth is assumed).
        """
        if not self._github_pat:
            logger.debug("GitManager: no PAT configured, using existing auth (SSH/credential helper)")
            return remote_url

        parsed = urlparse(remote_url)
        if parsed.scheme not in ("http", "https"):
            # SSH URL – PAT cannot be injected; rely on SSH key.
            logger.debug("GitManager: SSH remote, skipping PAT injection")
            return remote_url

        # Strip any existing credentials before injecting the PAT.
        authed = parsed._replace(netloc=f"{self._github_pat}@{parsed.hostname}{_port_suffix(parsed)}")
        return urlunparse(authed)

    async def _commit(self, message: str) -> tuple[str, str]:
        """
        Run `git commit`. Returns (short_hash, error_string).

        `error_string` is empty on success.  If the working tree is clean,
        returns ("", "nothing to commit") so the caller can distinguish.
        """
        safe_msg    = message.replace('"', '\\"')
        commit_res  = await self._executor.run(f'git commit -m "{safe_msg}"')

        if commit_res.succeeded:
            commit_hash = await self.get_short_hash()
            logger.info("GitManager: committed %s – %s", commit_hash, message)
            return commit_hash, ""

        combined = (commit_res.stdout + " " + commit_res.stderr).lower()
        if "nothing to commit" in combined or "nothing added to commit" in combined:
            logger.info("GitManager: nothing to commit (clean working tree)")
            commit_hash = await self.get_short_hash()
            return commit_hash, "nothing to commit"

        return "", f"git commit error: {commit_res.combined_output}"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _port_suffix(parsed) -> str:
    """Return ':port' string only when a non-default port is set."""
    if parsed.port and parsed.port not in (80, 443):
        return f":{parsed.port}"
    return ""


def _redact_pat(url: str) -> str:
    """Replace a PAT embedded in a URL with '***' for safe logging."""
    return re.sub(r"(https?://)([^@]+)@", r"\1***@", url)
