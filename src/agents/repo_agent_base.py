"""
RepoAgentBase – Shared base class for all repository-aware read-only agents.

Menyediakan infrastruktur bersama yang digunakan oleh:
  - DeveloperInspectorAgent  (inspeksi + root cause analysis)
  - DeveloperQnAAgent        (tanya-jawab tentang isi repository)

Metode yang disediakan:
  - Repo resolution: clone / pull / RepoTracker
  - Branch management: detect, checkout (multi-attempt)
  - Evidence gathering: dir tree, key files, log files
  - RAG: read_relevant_files via AST index + TF-IDF
  - Tavily: optional web search context
  - LLM extraction: parse repo_url, problem, keywords, branch dari user input
  - CLI helper: run read-only shell commands

Subclass HARUS mengimplementasikan `run(task)`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, ValidationError

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.key_store import effective_github_pat, effective_gitlab_pat
from src.memory.repo_tracker import RepoTracker
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor, CommandResult
from src.tools.git_utils import (
    inject_pat_into_url as _inject_pat_into_url,
    is_gitlab_url       as _is_gitlab_url,
    repo_name_from_url  as _repo_name_from_url,
)

logger = logging.getLogger(__name__)

# ── Shared Constants ───────────────────────────────────────────────────────────

REPOS_BASE_DIR     = Path.home() / "sandbox_repos"
MAX_FILE_BYTES     = 40_000   # max bytes per file snippet sent to LLM
MAX_GREP_LINES     = 80       # max lines from grep output per pattern
MAX_LOG_LINES      = 50       # max git log lines
MAX_DIFF_LINES     = 120      # max git diff lines
MAX_LS_LINES       = 150      # max lines from directory listing
MAX_RELEVANT_FILES = 10       # RAG: how many top-relevant source files to read

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

# ── Context compression constants ─────────────────────────────────────────────
# A section whose text exceeds this length gets compressed via LLM before
# the sections are assembled into a single evidence string.
EVIDENCE_COMPRESS_THRESHOLD = 3_000   # chars; sections shorter than this skip compression
EVIDENCE_COMPRESS_MAX_TOKENS = 512    # max tokens for the compression LLM response

# Rolling-summary threshold: once history for a session reaches this many
# messages, create a compressed summary and use it instead of raw history.
HISTORY_SUMMARY_THRESHOLD = 10   # messages
# Minimum new messages since last summary before regenerating the summary.
HISTORY_SUMMARY_REGEN_DELTA = 4

# ── Session repo context ──────────────────────────────────────────────────────
# Stores the last successfully-used {repo_url, branch} per session_id,
# so follow-up questions can inherit context without re-stating the repo/branch.
_REPO_SESSION_CONTEXT: dict[str, dict] = {}

# ── LLM prompt: conversation rolling summary ──────────────────────────────────
_SUMMARY_SYSTEM_PROMPT = """\
Kamu adalah ringkaser percakapan teknis. Ringkas percakapan berikut menjadi
3-5 poin konteks kunci yang relevan untuk pertanyaan lanjutan, mencakup:
- URL repositori dan branch yang sedang dibahas
- Masalah atau pertanyaan utama yang sedang diselidiki
- Temuan penting yang sudah dikonfirmasi (file, baris, fungsi kunci)
- Status terakhir: apa yang sudah dijawab dan apa yang belum

Balas HANYA dalam bentuk bullet points yang singkat dan padat.
"""

# ── LLM prompt: evidence section compression ──────────────────────────────────
_COMPRESS_SYSTEM_PROMPT = """\
Kamu adalah ringkaser teknis untuk evidence kode. Ringkas teks berikut menjadi
poin-poin kunci yang padat. WAJIB pertahankan semua detail berikut jika ada:
- Nama file dan nomor baris (mis. src/auth.py:42)
- Nama fungsi, class, dan method
- Nilai konstanta, environment variable, dan konfigurasi penting
- Pesan error/exception spesifik
- Route HTTP, endpoint URL, dan method (GET/POST/PUT/DELETE)
Balas dengan poin-poin bullet yang informatif. Jangan tambahkan penjelasan meta.
"""

# ── Shared LLM extraction prompt ──────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Ekstrak informasi berikut dari pesan pengguna dan balas dalam JSON:
{{
  "repo_url":   "<URL lengkap repository (GitHub, GitLab, Bitbucket, dll.) atau string kosong jika tidak ada>",
  "problem":    "<deskripsi ringkas masalah yang dilaporkan atau area yang ingin diinspeksi>",
  "keywords":   ["<keyword error atau simbol yang relevan untuk dicari di kode>"],
  "branch":     "<nama git branch jika disebutkan secara eksplisit dalam pesan atau dalam percakapan sebelumnya, jika tidak ada biarkan string kosong>"
}}

Perhatian:
- repo_url bisa berupa URL GitHub (github.com), GitLab (gitlab.com), atau platform git lainnya.
- Salin URL persis seperti yang disebutkan pengguna, termasuk scheme https://.
- Jika pesan saat ini adalah pertanyaan lanjutan (follow-up) dan repo_url / branch disebutkan di percakapan sebelumnya, ekstrak dari sana.

{history_section}Pesan pengguna saat ini: {user_input}
"""


class RepoExtractionRequest(BaseModel):
    """Parsed structured request extracted from user input via LLM."""
    repo_url:   str       = ""
    problem:    str       = ""
    keywords:   list[str] = []
    branch:     str       = ""
    # Verbosity hint parsed from user message ("singkat", "concise", etc.)
    verbosity:  str       = "detailed"  # "detailed" | "concise"
    # Optional: user-provided candidate routing filenames (e.g. ['routes.go'])
    candidate_route_filenames: list[str] = []


class RepoAgentBase(BaseAgent):
    """
    Abstract base for repository read-only agents.

    Provides shared infrastructure: repo resolution, branch management,
    evidence collection, RAG, and LLM-based request extraction.

    Subclasses must implement `run(task: AgentTask) -> AgentTask`.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        history: ConversationHistory | None = None,
    ) -> None:
        from config.settings import get_settings
        _settings          = get_settings()
        self._llm          = llm or LLMClient()
        self._history      = history
        self._repo_tracker = RepoTracker()
        self._cli          = CLIExecutor(timeout=30)
        self._repos_dir    = Path(_settings.sandbox_repos_dir).expanduser()
        self._github_pat   = _settings.github_pat
        self._gitlab_pat   = _settings.gitlab_pat
        self._repos_dir.mkdir(parents=True, exist_ok=True)

    # ── CLI helper ─────────────────────────────────────────────────────────────

    async def _run_cmd(self, cmd: str, cwd: Path | None = None) -> str:
        """Run a read-only shell command and return stdout+stderr (truncated)."""
        result: CommandResult = await self._cli.run(cmd, work_dir=cwd)
        out = (result.stdout or "") + (result.stderr or "")
        return out.strip()

    # ── Branch management ──────────────────────────────────────────────────────

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
            logger.info("%s: checked out branch '%s'", self.name, branch)
            return

        stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 2: recover from conflict / unmerged-files state ───────
        if any(kw in stderr.lower() for kw in ("unmerged", "conflict", "merge", "rebase")):
            logger.warning("%s: checkout blocked by dirty state – recovering: %s", self.name, stderr[:200])
            await cli.run("git rebase --abort", work_dir=repo_path)
            await cli.run("git merge --abort",  work_dir=repo_path)
            await cli.run("git reset --hard HEAD", work_dir=repo_path)
            result = await cli.run(f"git checkout {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("%s: checked out branch '%s' after state recovery", self.name, branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 3: branch exists locally but checkout still failed ────
        branch_check = await cli.run(f"git branch --list {branch}", work_dir=repo_path)
        if branch in (branch_check.stdout or ""):
            result = await cli.run(f"git checkout -f {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("%s: force-checked out existing branch '%s'", self.name, branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 4: branch missing locally – fetch all remotes ─────────
        await cli.run("git remote set-branches origin '*'", work_dir=repo_path)
        await cli.run("git fetch --all --prune", work_dir=repo_path)
        result = await cli.run(
            f"git checkout -b {branch} origin/{branch}", work_dir=repo_path
        )
        if result.succeeded:
            logger.info("%s: checked out remote branch '%s'", self.name, branch)
            return

        stderr = (result.stdout or "") + (result.stderr or "")

        # ── Attempt 5: '-b' failed because branch already exists locally ──
        if "already exists" in stderr:
            result = await cli.run(f"git checkout -f {branch}", work_dir=repo_path)
            if result.succeeded:
                logger.info("%s: checked out (already-local) branch '%s'", self.name, branch)
                return
            stderr = (result.stdout or "") + (result.stderr or "")

        raise RuntimeError(
            f"Branch '{branch}' tidak dapat di-checkout:\n{stderr[:400]}"
        )

    # ── Evidence helpers ───────────────────────────────────────────────────────

    async def _get_dir_tree(self, repo_path: Path) -> str:
        """Return a pruned directory listing."""
        out = await self._run_cmd(
            "find . -not \\( "
            + " ".join(f"-path './{d}' -prune -o" for d in _SKIP_DIRS)
            + " -false \\) -print | head -" + str(MAX_LS_LINES),
            cwd=repo_path,
        )
        return out or "(empty)"

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
                except OSError as _exc:
                    logger.debug("repo_agent_base: could not read %s: %s", fpath, _exc)
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
        RAG step: index the repo with AST + semantic (if available) or TF-IDF
        ranking and read the top-N source files most relevant to *problem*.

        Ranking priority:
          1. Semantic (sentence-transformers + FAISS) – if installed.
          2. TF-IDF cosine similarity (scikit-learn) – always available.

        Falls back gracefully if code_search dependencies are not available.
        """
        try:
            from src.tools.code_search import (
                build_ast_index,
                rank_files_by_relevance,
                rank_files_by_relevance_semantic,
            )
        except ImportError:
            logger.warning("%s: code_search not available; skipping RAG step", self.name)
            return "(code_search unavailable)"

        if not problem:
            return "(no problem description for relevance ranking)"

        try:
            logger.info("%s: building AST index for RAG at %s", self.name, repo_path)
            t0 = time.monotonic()
            symbol_index = build_ast_index(repo_path)
            candidates   = list(symbol_index.keys())

            if not candidates:
                return "(no indexable source files found)"

            # Try semantic ranking first (Item 4); fall back to TF-IDF.
            ranked = rank_files_by_relevance_semantic(
                candidates, symbol_index, problem, repo_path
            )
            if ranked is None:
                ranked = rank_files_by_relevance(candidates, symbol_index, problem)
                logger.debug("%s: using TF-IDF ranking (semantic unavailable)", self.name)
            else:
                logger.debug("%s: using semantic ranking (sentence-transformers)", self.name)

            top_n   = ranked[:MAX_RELEVANT_FILES]
            elapsed = time.monotonic() - t0
            logger.info(
                "%s: RAG indexed %d files in %.2fs; top %d selected",
                self.name, len(candidates), elapsed, len(top_n),
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
                    logger.debug("%s: could not read %s: %s", self.name, rel_path, exc)

            return (
                "\n\n".join(snippets)
                if snippets
                else "(no relevant files could be read)"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: RAG step failed: %s", self.name, exc)
            return f"(RAG error: {exc})"

    # ── Tavily context ─────────────────────────────────────────────────────────

    async def _fetch_tavily_context(self, query: str) -> str:
        """Attempt to fetch web research context from Tavily (best-effort)."""
        try:
            from src.tools.tavily_search import TavilySearchTool  # type: ignore
            tool = TavilySearchTool()
            resp = await tool.search(query)
            ctx = resp.as_context_text()
            return ctx or ""
        except Exception as exc:  # pragma: no cover
            logger.debug("Tavily fetch failed or unavailable: %s", exc)
            return ""

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
            repo_name  = _repo_name_from_url(repo_url)
            local_path = self._repos_dir / repo_name
            self._repos_dir.mkdir(parents=True, exist_ok=True)

            _pat     = effective_gitlab_pat(self._gitlab_pat) if _is_gitlab_url(repo_url) else effective_github_pat(self._github_pat)
            auth_url = _inject_pat_into_url(repo_url, _pat)

            if local_path.exists():
                logger.info("%s: repo already exists, pulling. path=%s", self.name, local_path)
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
                    logger.warning("%s: git pull may have failed: %s", self.name, pull_out)
                    if any(kw in pull_out.lower() for kw in ("unmerged", "conflict", "rebase")):
                        logger.info("%s: recovering from conflict/rebase state", self.name)
                        await self._run_cmd("git rebase --abort", cwd=local_path)
                        await self._run_cmd("git merge --abort", cwd=local_path)
                        await self._run_cmd("git reset --hard HEAD", cwd=local_path)
            else:
                logger.info("%s: cloning %s → %s", self.name, repo_url, local_path)
                result = await self._run_cmd(
                    f"git clone --no-single-branch {auth_url} {local_path}"
                )
                if "fatal" in result.lower() or "error" in result.lower():
                    logger.warning("%s: clone may have failed: %s", self.name, result)

            self._repo_tracker.upsert(
                repo_name,
                repo_url,
                str(local_path),
                status="cloned",
            )
            return local_path if local_path.exists() else None

        # No URL – try last known repo from tracker
        repos = self._repo_tracker.list_all()
        if repos:
            latest = repos[-1]
            path = Path(latest.local_path) if latest.local_path else None
            if path and path.exists():
                logger.info("%s: no URL given, using tracked repo=%s", self.name, path)
                return path
        return None

    # ── Session repo context ───────────────────────────────────────────────────

    def _save_session_context(
        self,
        session_id: str,
        repo_url: str,
        branch: str,
        candidate_route_filenames: list[str] | None = None,
        last_symbol_target: str = "",
    ) -> None:
        """
        Persist the repo context used in this turn so follow-up questions
        can inherit repo_url and branch without re-stating them.
        """
        if repo_url or branch:
            ctx = _REPO_SESSION_CONTEXT.setdefault(session_id, {})
            if repo_url:
                ctx["repo_url"] = repo_url
            if branch:
                ctx["branch"] = branch
            if candidate_route_filenames:
                ctx["candidate_route_filenames"] = candidate_route_filenames
            if last_symbol_target:
                ctx["last_symbol_target"] = last_symbol_target
            logger.debug(
                "%s: saved session context for %s → repo=%s branch=%s",
                self.name, session_id, repo_url, branch,
            )

    def _get_session_context(self, session_id: str) -> dict:
        """Return saved repo context for the session (empty dict if none)."""
        return _REPO_SESSION_CONTEXT.get(session_id, {})

    # ── LLM request extraction ─────────────────────────────────────────────────

    async def _extract_request(
        self,
        user_input: str,
        session_id: str = "",
    ) -> RepoExtractionRequest:
        """
        Call LLM to parse repo_url, problem, keywords, and branch from user input.

        When session_id is provided:
          - For short sessions (< HISTORY_SUMMARY_THRESHOLD messages): includes the
            last 6 messages verbatim.
          - For long sessions: uses a compressed rolling summary + the 3 most recent
            messages, preventing context drift and reducing token waste.
          - Falls back to the last saved session context when the LLM returns
            empty repo_url or branch (common in follow-up questions).

        Falls back to a minimal request if JSON parsing fails.
        """
        # Build optional history section for the prompt.
        # For long sessions, use rolling summary + last 3 messages (Item 3).
        history_section = ""
        if session_id and self._history:
            recent_messages = self._history.get_as_llm_messages(session_id)
            if len(recent_messages) > 1:
                if len(recent_messages) >= HISTORY_SUMMARY_THRESHOLD:
                    # Long session: compressed summary + last 3 verbatim turns.
                    conv_summary = await self._get_conversation_summary(session_id)
                    last_turns   = recent_messages[-3:]
                    last_lines   = "\n".join(
                        f"[{m['role']}]: {m['content'][:600]}" for m in last_turns
                    )
                    if conv_summary:
                        history_section = (
                            f"Ringkasan percakapan sebelumnya:\n{conv_summary}\n\n"
                            f"Pesan terbaru:\n{last_lines}\n\n"
                        )
                    else:
                        history_section = (
                            f"Percakapan sebelumnya (untuk konteks):\n{last_lines}\n\n"
                        )
                else:
                    # Short session: keep last 6 messages verbatim.
                    history_turns = recent_messages[-6:]
                    history_lines = "\n".join(
                        f"[{m['role']}]: {m['content'][:600]}"
                        for m in history_turns
                    )
                    history_section = (
                        f"Percakapan sebelumnya (untuk konteks):\n"
                        f"{history_lines}\n\n"
                    )

        prompt   = _EXTRACT_PROMPT.format(
            user_input=user_input,
            history_section=history_section,
        )
        response = await self._llm.chat(
            messages=[
                {"role": "system", "content": "You are a JSON extractor. Reply with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$",          "", raw, flags=re.MULTILINE)
        try:
            data = json.loads(raw)
            req  = RepoExtractionRequest(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("%s: failed to parse extraction JSON: %s", self.name, exc)
            req = RepoExtractionRequest(problem=user_input)

        # ── Fallback: inherit repo context from previous turn in session ───
        if session_id:
            ctx = self._get_session_context(session_id)
            if not req.repo_url and ctx.get("repo_url"):
                req.repo_url = ctx["repo_url"]
                logger.info(
                    "%s: inherited repo_url=%r from session context",
                    self.name, req.repo_url,
                )
            if not req.branch and ctx.get("branch"):
                req.branch = ctx["branch"]
                logger.info(
                    "%s: inherited branch=%r from session context",
                    self.name, req.branch,
                )
            if not req.candidate_route_filenames and ctx.get("candidate_route_filenames"):
                req.candidate_route_filenames = ctx["candidate_route_filenames"]

        # Verbosity hint from user message (applies to both agents)
        lower = user_input.lower()
        if any(w in lower for w in ["singkat", "brief", "concise", "ringkas"]):
            req.verbosity = "concise"

        # Detect explicitly mentioned source file names from user input
        # e.g. "routes.go", "router.py", "api.ts" → prioritized in API extraction
        if not req.candidate_route_filenames:
            file_mentions = re.findall(
                r"\b([\w\-]+\.(?:go|py|js|ts|rb|php|java|cs))\b",
                user_input,
                re.IGNORECASE,
            )
            if file_mentions:
                # Deduplicate while preserving order
                seen_files: dict[str, None] = {}
                for f in file_mentions:
                    seen_files[f.lower()] = None
                req.candidate_route_filenames = list(seen_files.keys())

        return req

    # ── Evidence text builder ──────────────────────────────────────────────────

    def _build_evidence_text(self, evidence: dict[str, str]) -> str:
        """Serialize evidence dict into a markdown string for LLM consumption."""
        return "\n\n".join(
            f"## {title}\n{content}"
            for title, content in evidence.items()
            if content.strip()
        )

    # ── Hierarchical evidence compression (Item 1) ─────────────────────────────

    async def _compress_evidence_section(
        self,
        title: str,
        content: str,
        *,
        threshold: int = EVIDENCE_COMPRESS_THRESHOLD,
        max_tokens: int = EVIDENCE_COMPRESS_MAX_TOKENS,
    ) -> str:
        """
        Compress a single evidence section via LLM if it exceeds *threshold* chars.

        Preserves critical technical details (file:line references, function names,
        error messages) while reducing token count. Falls back to a hard truncation
        if the LLM call fails.
        """
        if len(content) <= threshold:
            return content

        # Cap input to avoid sending huge prompts to the compression LLM
        input_text = content[:6_000]
        prompt = (
            f"Ringkas bagian evidence berikut. Pertahankan semua referensi file:baris, "
            f"nama fungsi/class, pesan error, dan route HTTP.\n\n"
            f"=== {title} ===\n{input_text}"
        )
        try:
            compressed = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.05,
                top_p=0.90,
                max_tokens=max_tokens,
            )
            if compressed.strip():
                logger.debug(
                    "%s: compressed evidence '%s': %d → %d chars",
                    self.name, title, len(content), len(compressed),
                )
                return f"[RINGKASAN]\n{compressed.strip()}"
        except Exception as exc:
            logger.warning(
                "%s: evidence compression failed for '%s': %s", self.name, title, exc
            )

        # Fallback: hard truncation with notice
        return content[:threshold] + f"\n... [dipotong pada {threshold} karakter]"

    async def _compress_evidence_dict(
        self,
        evidence: dict[str, str],
        *,
        threshold: int = EVIDENCE_COMPRESS_THRESHOLD,
    ) -> dict[str, str]:
        """
        Compress all large sections in *evidence* concurrently.

        Sections shorter than *threshold* chars are returned unchanged.
        This ensures every section is represented in the final evidence
        even when the total would exceed the LLM's context window.
        """
        keys = list(evidence.keys())
        results = await asyncio.gather(
            *[
                self._compress_evidence_section(k, evidence[k], threshold=threshold)
                for k in keys
            ],
            return_exceptions=True,
        )
        compressed: dict[str, str] = {}
        for k, r in zip(keys, results):
            if isinstance(r, Exception):
                logger.warning(
                    "%s: compression gather error for '%s': %s", self.name, k, r
                )
                compressed[k] = evidence[k]
            else:
                compressed[k] = str(r)
        return compressed

    # ── RAG: individual file list (for Map-Reduce, Item 2) ────────────────────

    async def _read_relevant_files_list(
        self,
        repo_path: Path,
        problem: str,
    ) -> list[tuple[str, str]]:
        """
        Like `_read_relevant_files()` but returns individual (rel_path, content)
        tuples instead of a single concatenated string.

        Used by the Map-Reduce flow in DeveloperQnAAgent.
        Falls back to an empty list if code_search is unavailable.
        """
        try:
            from src.tools.code_search import build_ast_index, rank_files_by_relevance
        except ImportError:
            logger.warning("%s: code_search not available; skipping RAG file list", self.name)
            return []

        if not problem:
            return []

        try:
            symbol_index = build_ast_index(repo_path)
            candidates   = list(symbol_index.keys())
            if not candidates:
                return []

            ranked = rank_files_by_relevance(candidates, symbol_index, problem)
            top_n  = ranked[:MAX_RELEVANT_FILES]

            file_list: list[tuple[str, str]] = []
            for rel_path in top_n:
                abs_path = repo_path / rel_path
                try:
                    text = abs_path.read_text(errors="replace")[:MAX_FILE_BYTES]
                    file_list.append((rel_path, text))
                except OSError as exc:
                    logger.debug(
                        "%s: could not read %s: %s", self.name, rel_path, exc
                    )
            return file_list

        except Exception as exc:
            logger.warning("%s: RAG file list step failed: %s", self.name, exc)
            return []

    # ── Rolling conversation summary (Item 3) ──────────────────────────────────

    async def _get_conversation_summary(self, session_id: str) -> str:
        """
        Return a compressed summary of the conversation history for *session_id*.

        Behaviour:
        - If history has fewer than HISTORY_SUMMARY_THRESHOLD messages, return
          the cached summary (or empty string if none exists yet).
        - When history is long enough, create a new summary via LLM and cache it
          in the session context.  The summary is regenerated only when at least
          HISTORY_SUMMARY_REGEN_DELTA new messages have appeared since the last
          summary, to avoid unnecessary LLM calls.
        """
        ctx = self._get_session_context(session_id)
        existing_summary: str = ctx.get("conversation_summary", "")

        if not self._history:
            return existing_summary

        messages = self._history.get_as_llm_messages(session_id)
        if len(messages) < HISTORY_SUMMARY_THRESHOLD:
            return existing_summary

        # Avoid regenerating if the message count hasn't grown enough.
        last_count: int = ctx.get("summary_message_count", 0)
        if existing_summary and len(messages) - last_count < HISTORY_SUMMARY_REGEN_DELTA:
            return existing_summary

        # Summarise all messages except the most recent few (kept verbatim).
        older_messages = messages[:-3]
        history_text   = "\n".join(
            f"[{m['role']}]: {m['content'][:800]}"
            for m in older_messages
        )

        try:
            summary = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Percakapan:\n\n{history_text}"},
                ],
                temperature=0.05,
                top_p=0.90,
                max_tokens=256,
            )
            if summary.strip():
                ctx_mut = _REPO_SESSION_CONTEXT.setdefault(session_id, {})
                ctx_mut["conversation_summary"]  = summary.strip()
                ctx_mut["summary_message_count"] = len(messages)
                logger.info(
                    "%s: created conversation summary for session %s (%d msgs → %d chars)",
                    self.name, session_id, len(messages), len(summary),
                )
                return summary.strip()
        except Exception as exc:
            logger.warning(
                "%s: conversation summary LLM call failed: %s", self.name, exc
            )

        return existing_summary

