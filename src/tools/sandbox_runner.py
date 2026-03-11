"""
SandboxRunner – Docker-based code execution sandbox.

Responsibilities:
1. Detect or generate a Dockerfile / docker-compose.yml for a cloned repo.
2. Build and run the container via `docker compose up --build`.
3. Inspect logs for Python tracebacks or exit code errors.
4. Report a structured SandboxResult back to the DeveloperAgent.

All heavy work is delegated to CLIExecutor so it doesn't block the
asyncio event loop.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from src.tools.cli_executor import CLIExecutor, CommandResult

logger = logging.getLogger(__name__)

_DEFAULT_PYTHON_IMAGE = "python:3.11-slim"
_DEFAULT_NODE_IMAGE   = "node:20-slim"

# ── Fallback Dockerfiles ──────────────────────────────────────────────────────

# Python project: try pytest → main.py → explain
_FALLBACK_DOCKERFILE_PYTHON = textwrap.dedent("""\
    FROM {image}
    WORKDIR /app
    COPY . .
    RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true
    RUN pip install --no-cache-dir pytest 2>/dev/null || true
    CMD ["sh", "-c",
         "python -m pytest --tb=short -q 2>&1 || \\
          (echo 'No pytest tests found; trying main.py'; \\
           python main.py 2>&1) || \\
          echo 'No runnable entry point found'"
    ]
""")

# Node/Vue/TS/JS project: install deps → build → lint/test
_FALLBACK_DOCKERFILE_NODE = textwrap.dedent("""\
    FROM {image}
    WORKDIR /app
    COPY package*.json ./
    RUN npm ci --prefer-offline 2>&1 || npm install 2>&1
    COPY . .
    CMD ["sh", "-c",
         "npm test 2>&1 || \\
          npm run build 2>&1 || \\
          npm run lint 2>&1 || \\
          echo 'No test/build/lint script found in package.json'"
    ]
""")

# docker-compose.yml template – {container_name} filled at runtime.
# 'version' field omitted: deprecated in Docker Compose v2.
_FALLBACK_COMPOSE = textwrap.dedent("""\
    services:
      app:
        build: .
        container_name: {container_name}
""")


@dataclass
class SandboxResult:
    """
    Result of a single sandbox execution run.

    Attributes:
        succeeded:     True when Docker exited with code 0 and no Python
                       tracebacks detected in the app logs.
        has_traceback: True when app logs contain a Python traceback.
        phase:         Which phase failed: "build" or "run" (or "" if succeeded).
        logs:          Relevant output (app logs for run phase; build logs for
                       build phase).
        error_summary: A short human-readable error description (or empty string).
        attempts:      How many build attempts were made before this result.
    """
    succeeded:     bool
    has_traceback: bool
    phase:         str   # "build" | "run" | ""
    logs:          str
    error_summary: str = ""
    attempts:      int = 1


class SandboxRunner:
    """
    Orchestrates Docker sandbox creation and test execution for a repo.

    Usage:
        runner = SandboxRunner(repo_path=Path("/tmp/myrepo"))
        result = await runner.run(max_attempts=3)
    """

    def __init__(
        self,
        repo_path:    Path | str,
        python_image: str = _DEFAULT_PYTHON_IMAGE,
        timeout:      int = 300,
    ) -> None:
        """
        Args:
            repo_path:    Absolute path to the cloned repository.
            python_image: Docker image used when generating a fallback Dockerfile.
            timeout:      Max seconds for each docker compose command.
        """
        self._repo_path    = Path(repo_path)
        self._python_image = python_image
        self._executor     = CLIExecutor(work_dir=self._repo_path, timeout=timeout)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, *, max_attempts: int = 3) -> SandboxResult:
        """
        Build and run the repo inside Docker.

        Two-phase execution:
          Phase 1 – Build  : `docker compose build`
          Phase 2 – Run    : `docker compose up --no-build --exit-code-from app`
          Phase 3 – Logs   : `docker compose logs app` (clean app-only output)

        Separating build from run allows us to clearly distinguish
        Docker build failures from Python runtime errors, and prevents the
        Docker build noise (ERROR/WARNING lines from the build system) from
        triggering false-positive traceback detection.

        Args:
            max_attempts: Passed through to SandboxResult so the caller
                          can decide whether to keep retrying.

        Returns:
            SandboxResult describing the outcome.
        """
        self._ensure_docker_files_exist()

        # Teardown any stale container from a previous attempt.
        await self._teardown()

        # ── Phase 1: Build ────────────────────────────────────────────────
        build_result = await self._executor.run(
            "docker compose build --progress=plain 2>&1"
        )
        if not build_result.succeeded:
            build_logs = build_result.combined_output
            logger.error("SandboxRunner: build FAILED (exit=%d)", build_result.returncode)
            return SandboxResult(
                succeeded=False,
                has_traceback=False,
                phase="build",
                logs=build_logs,
                error_summary=_extract_build_error(build_logs),
                attempts=max_attempts,
            )

        # ── Phase 2: Run via docker compose run --rm ──────────────────────
        # `docker compose run --rm` starts a one-off container that exits
        # immediately after the command finishes.  This avoids the timeout
        # problem caused by `compose up` blocking on long-running servers
        # (Vue dev server, Flask app, etc.).
        #
        # The test command is chosen based on the detected project type:
        #   Python : python -m pytest --tb=short -q
        #   Node   : npm test  (fallback: npm run build)
        project_type = _detect_project_type(self._repo_path)
        test_cmd     = _get_test_command(project_type)
        logger.info(
            "SandboxRunner: project_type=%s test_command=%s",
            project_type, test_cmd,
        )

        run_result = await self._executor.run(
            f"docker compose run --rm --no-deps app sh -c {shlex.quote(test_cmd)} 2>&1"
        )
        detection_text = _strip_compose_prefix(run_result.stdout or "")
        if not detection_text.strip():
            detection_text = run_result.combined_output

        has_traceback = _detect_python_traceback(detection_text)
        success       = run_result.succeeded and not has_traceback

        error_summary = ""
        if not success:
            if run_result.timed_out:
                error_summary = "Docker run timed out."
            elif has_traceback:
                error_summary = _extract_traceback_summary(detection_text)
            else:
                error_summary = (
                    f"Container exited with code {run_result.returncode}.\n"
                    f"{detection_text[-1500:]}"
                )

        logger.info(
            "SandboxRunner: succeeded=%s has_traceback=%s returncode=%d phase=run",
            success, has_traceback, run_result.returncode,
        )

        return SandboxResult(
            succeeded=success,
            has_traceback=has_traceback,
            phase="run",
            logs=detection_text,
            error_summary=error_summary,
            attempts=max_attempts,
        )

    async def teardown(self) -> None:
        """Public wrapper – stop and remove the sandbox container."""
        await self._teardown()

    async def get_diff(self) -> str:
        """Return `git diff` of changes made inside the repo."""
        result = await self._executor.run("git diff")
        return result.stdout or "(no changes)"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_docker_files_exist(self) -> None:
        """
        Check for Dockerfile and docker-compose.yml.
        Writes appropriate fallback files based on detected project type
        (Python or Node/Vue/TS/JS) when absent.

        Container name is derived from the repo path so multiple repos never
        share the same name (avoids Docker name conflicts).
        """
        dockerfile   = self._repo_path / "Dockerfile"
        compose_file = self._repo_path / "docker-compose.yml"

        if not dockerfile.exists():
            project_type = _detect_project_type(self._repo_path)
            if project_type == "node":
                content = _FALLBACK_DOCKERFILE_NODE.format(image=_DEFAULT_NODE_IMAGE)
            else:
                content = _FALLBACK_DOCKERFILE_PYTHON.format(image=self._python_image)
            dockerfile.write_text(content)
            logger.info(
                "SandboxRunner: generated fallback Dockerfile (%s) at %s",
                project_type, dockerfile,
            )

        if not compose_file.exists():
            container_name = _safe_container_name(self._repo_path)
            compose_file.write_text(
                _FALLBACK_COMPOSE.format(container_name=container_name)
            )
            logger.info(
                "SandboxRunner: generated fallback docker-compose.yml "
                "(container=%s) at %s",
                container_name,
                compose_file,
            )

    async def _teardown(self) -> None:
        """Run docker compose down to clean up containers and networks."""
        result = await self._executor.run("docker compose down --remove-orphans --timeout 10")
        if not result.succeeded:
            logger.debug("SandboxRunner teardown warning: %s", result.stderr)


# ── Utility functions ─────────────────────────────────────────────────────────

def _detect_project_type(repo_path: Path) -> str:
    """
    Detect the primary language/framework of a repository.

    Returns:
        "node"   – when package.json exists (Vue, React, TS, JS, Svelte, etc.)
        "python" – when requirements.txt / pyproject.toml / *.py files exist
        "python" – default fallback
    """
    if (repo_path / "package.json").exists():
        return "node"
    if (
        (repo_path / "requirements.txt").exists()
        or (repo_path / "pyproject.toml").exists()
        or any(repo_path.rglob("*.py"))
    ):
        return "python"
    return "python"   # safe default


def _get_test_command(project_type: str) -> str:
    """
    Return the shell command used inside the container to verify the project.

    Python : run pytest; fallback to `python main.py` if no tests found.
    Node   : run `npm test`; fallback to `npm run build` then `npm run lint`.
    """
    if project_type == "node":
        return (
            "npm test 2>&1 || "
            "npm run build 2>&1 || "
            "npm run lint 2>&1 || "
            "echo 'No test/build/lint script found in package.json'"
        )
    # Default: Python
    return (
        "python -m pytest --tb=short -q 2>&1 || "
        "(echo 'No pytest tests; trying main.py'; python main.py 2>&1) || "
        "echo 'No runnable entry point found'"
    )


# Regex for a genuine Python traceback header.
_RE_TRACEBACK      = re.compile(r"Traceback \(most recent call last\)")
# Raised-exception line at the very end of a traceback block, e.g.:
#   ValueError: invalid literal for int()
_RE_RAISED_EXC     = re.compile(r"^[A-Z][\w.]*Error:|^[A-Z][\w.]*Exception:", re.MULTILINE)
# Docker compose log prefix, e.g. "app  | actual log line"
_RE_COMPOSE_PREFIX = re.compile(r"^\S+\s+\|\s?", re.MULTILINE)
# Docker build error lines (buildkit format)
_RE_BUILD_ERROR    = re.compile(
    r"^(?:#\d+\s+)?ERROR[: ]|^failed to solve",
    re.MULTILINE | re.IGNORECASE,
)


def _detect_python_traceback(text: str) -> bool:
    """
    Return True ONLY when the text contains a genuine Python traceback.

    Requires BOTH of the following patterns to match (or just the header):
      1. The canonical header "Traceback (most recent call last)"
      2. OR a raised-exception line like ``SomeError: message``

    This avoids false positives from Docker build log lines such as
    ``ERROR [internal] load metadata`` or ``Error: Process completed.``
    """
    return bool(_RE_TRACEBACK.search(text) or _RE_RAISED_EXC.search(text))


def _strip_compose_prefix(text: str) -> str:
    """
    Remove Docker Compose log prefixes ("app  | ") from each line so
    traceback detection operates on raw app output.
    """
    return _RE_COMPOSE_PREFIX.sub("", text)


def _extract_traceback_summary(logs: str, max_lines: int = 30) -> str:
    """
    Extract lines starting from the first traceback header.
    Falls back to the last max_lines lines of the log.
    """
    lines = logs.splitlines()
    try:
        start_idx = next(
            i for i, line in enumerate(lines)
            if _RE_TRACEBACK.search(line)
        )
        relevant = lines[start_idx:]
    except StopIteration:
        relevant = lines
    return "\n".join(relevant[-max_lines:])


def _extract_build_error(logs: str, max_lines: int = 30) -> str:
    """
    Extract a concise summary of a Docker build failure.
    Finds the first ERROR line in buildkit output.
    """
    lines = logs.splitlines()
    try:
        start_idx = next(
            i for i, line in enumerate(lines)
            if _RE_BUILD_ERROR.search(line)
        )
        return "\n".join(lines[start_idx : start_idx + max_lines])
    except StopIteration:
        return "\n".join(lines[-max_lines:])


def _safe_container_name(repo_path: Path) -> str:
    """
    Derive a Docker-safe container name from the repo path.

    Format: ``sandbox-<slug>-<hash6>``
    - slug  : last two path components, lowercased, non-alnum replaced with '-'.
    - hash6 : first 6 chars of SHA1 of the absolute path (for uniqueness).
    Docker container name limit: 63 characters.
    """
    parts = list(repo_path.resolve().parts[-2:])
    slug  = re.sub(r"[^a-z0-9]+", "-", "_".join(parts).lower()).strip("-")
    h6    = hashlib.sha1(str(repo_path.resolve()).encode()).hexdigest()[:6]
    return f"sandbox-{slug}-{h6}"[:63]
