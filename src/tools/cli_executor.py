"""
CLIExecutor – non-interactive terminal command runner.

Features:
- Configurable timeout (default: 5 minutes) to prevent server hang.
- Captures both stdout and stderr for LLM analysis on failure.
- Automatically confirms interactive Y/N prompts via stdin.
- Runs commands inside a specific working directory.
- Returns a structured result dataclass.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes


@dataclass(frozen=True)
class CommandResult:
    """
    Immutable result of a CLI command execution.

    Attributes:
        command:     The command that was run (as a string).
        returncode:  Process exit code (0 = success).
        stdout:      Captured standard output.
        stderr:      Captured standard error.
        timed_out:   True if the process was killed due to timeout.
    """
    command:    str
    returncode: int
    stdout:     str
    stderr:     str
    timed_out:  bool

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        """Concatenated stdout + stderr for LLM analysis."""
        parts = []
        if self.stdout.strip():
            parts.append(f"[STDOUT]\n{self.stdout}")
        if self.stderr.strip():
            parts.append(f"[STDERR]\n{self.stderr}")
        return "\n".join(parts) or "(no output)"

    def __str__(self) -> str:
        status = "OK" if self.succeeded else ("TIMEOUT" if self.timed_out else f"ERR:{self.returncode}")
        return f"<CommandResult [{status}] cmd={self.command!r}>"


class CLIExecutor:
    """
    Async-friendly wrapper around subprocess for CLI command execution.

    All heavy work runs in a thread via asyncio.to_thread() to avoid
    blocking the event loop.

    Example:
        executor = CLIExecutor(work_dir="/tmp/myrepo")
        result = await executor.run("git status")
        if not result.succeeded:
            print(result.combined_output)
    """

    def __init__(
        self,
        work_dir: Path | str | None = None,
        timeout:  int               = _DEFAULT_TIMEOUT_SECONDS,
        auto_yes: bool              = True,
    ) -> None:
        """
        Args:
            work_dir:  Default working directory for all commands.
            timeout:   Max seconds to wait before killing the process.
            auto_yes:  If True, pipes 'yes\\n' to stdin so interactive
                       Y/N prompts are answered automatically.
        """
        self._work_dir = Path(work_dir) if work_dir else Path.cwd()
        self._timeout  = timeout
        self._auto_yes = auto_yes

    # ── Public async API ──────────────────────────────────────────────────────

    @property
    def work_dir(self) -> Path:
        """The default working directory for this executor."""
        return self._work_dir

    async def run(
        self,
        command:  str | Sequence[str],
        *,
        work_dir: Path | str | None = None,
        env:      dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a shell command asynchronously.

        Args:
            command:  Shell string (e.g. 'git clone ...') or argv list.
            work_dir: Override the default working directory for this call.
            env:      Extra environment variables merged with os.environ.

        Returns:
            CommandResult with stdout, stderr, and returncode.
        """
        cwd = Path(work_dir) if work_dir else self._work_dir
        cmd_str = command if isinstance(command, str) else shlex.join(command)

        logger.info("CLIExecutor.run cwd=%s cmd=%s", cwd, cmd_str)

        return await asyncio.to_thread(
            self._run_sync, cmd_str, cwd, env
        )

    # ── Private sync implementation ───────────────────────────────────────────

    def _run_sync(
        self,
        cmd_str:  str,
        cwd:      Path,
        extra_env: dict[str, str] | None,
    ) -> CommandResult:
        """Synchronous execution – called via asyncio.to_thread()."""
        import os

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        stdin_data = b"y\ny\ny\n" if self._auto_yes else None

        try:
            proc = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                cwd=str(cwd),
                env=env,
                input=stdin_data,
                timeout=self._timeout,
            )
            result = CommandResult(
                command=cmd_str,
                returncode=proc.returncode,
                stdout=proc.stdout.decode(errors="replace").strip(),
                stderr=proc.stderr.decode(errors="replace").strip(),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace").strip()
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            result = CommandResult(
                command=cmd_str,
                returncode=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
            logger.warning("CLIExecutor: command timed out after %ds: %s", self._timeout, cmd_str)
        except Exception as exc:
            result = CommandResult(
                command=cmd_str,
                returncode=-1,
                stdout="",
                stderr=str(exc),
                timed_out=False,
            )
            logger.exception("CLIExecutor: unexpected error running %s: %s", cmd_str, exc)

        log_level = logging.DEBUG if result.succeeded else logging.WARNING
        logger.log(
            log_level,
            "CLIExecutor: returncode=%d stdout_len=%d stderr_len=%d",
            result.returncode,
            len(result.stdout),
            len(result.stderr),
        )
        return result
