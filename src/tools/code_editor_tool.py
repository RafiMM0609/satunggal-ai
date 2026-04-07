"""
CodeEditorTool – shared code-editing logic for developer agents.

Extracted from DeveloperAgent so that multiple agents can reuse the same
AI-CLI + LLM-direct patch workflow without duplicating code.

Used by:
  - DeveloperAgent  (src/agents/developer/agent.py)
  - CodeFixAgent    (src/agents/code_fix/agent.py)
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator

from src.agents.llm_client import LLMClient
from src.tools.cli_executor import CLIExecutor

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

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

# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class CodePatch:
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


# ── Module-level helpers ──────────────────────────────────────────────────────

def detect_available_cli() -> str | None:
    """
    Return the name of a supported AI CLI that can NON-INTERACTIVELY edit files,
    or None (→ LLM-direct mode via OpenRouter).

    NOTE: `gh copilot suggest` is intentionally excluded here.
    Only `claude` CLI is supported because it can write files non-interactively.
    """
    if shutil.which("claude"):
        return "claude"
    return None


def build_ai_cli_command(task: str, cli: str = "claude") -> str:
    """Build the shell command to invoke the AI CLI for non-interactive file editing."""
    safe_task = task.replace('"', '\\"').replace("'", "\\'")
    if cli == "claude":
        return (
            f'claude -p "{safe_task}" '
            f'--allowedTools "Read,Edit,Write,Bash" '
            f'--output-format json'
        )
    return f'echo "No supported AI CLI available for: {safe_task}"'


def make_search_command(pattern: str) -> str:
    """
    Build a file-search shell command for the given keyword pattern.

    Prefers ripgrep (rg); falls back to GNU grep.
    """
    _INCLUDE_EXTS = (
        "html", "css", "scss", "sass",
        "js", "ts", "tsx", "jsx",
        "vue", "svelte", "py", "json",
    )
    safe_pattern = pattern.replace("'", r"\'")

    if shutil.which("rg"):
        type_args = " ".join(
            f"--type-add '{ext}:*.{ext}' --type {ext}" for ext in _INCLUDE_EXTS
        )
        return (
            f"rg --files-with-matches -e '{safe_pattern}' "
            f"{type_args} "
            f"--max-count 1 . 2>/dev/null | head -20"
        )

    includes = " ".join(f"--include='*.{e}'" for e in _INCLUDE_EXTS)
    return (
        f"grep -rl {includes} "
        f"-e '{safe_pattern}' . 2>/dev/null | head -20"
    )


async def apply_diff_patch(
    patch_text: str,
    repo_path:  Path,
    executor:   CLIExecutor,
) -> bool:
    """
    Apply a unified diff patch to the repository.

    Strategy:
      1. Use the system `patch` CLI.
      2. Fall back to the Python `patch` library if unavailable.

    Returns True on success, False on failure.
    """
    if not patch_text.strip():
        return False

    if shutil.which("patch"):
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
                logger.info("apply_diff_patch: patch CLI succeeded")
                return True
            logger.warning(
                "apply_diff_patch: patch CLI failed (exit=%d): %s",
                result.returncode, result.stderr[:400],
            )
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
    else:
        try:
            import patch as patch_lib  # type: ignore

            ps = patch_lib.fromstring(patch_text.encode("utf-8"))
            success = bool(ps and ps.apply(root=str(repo_path)))
            if success:
                logger.info("apply_diff_patch: python-patch library succeeded")
            else:
                logger.warning("apply_diff_patch: python-patch library failed")
            return success
        except Exception as exc:  # noqa: BLE001
            logger.error("apply_diff_patch: python-patch error (%s)", exc)
            return False


async def apply_patches_to_disk(
    patches:   list[CodePatch],
    repo_root: Path,
    executor:  CLIExecutor,
) -> None:
    """
    Write each patch to disk.

    Per-patch strategy:
      1. If *patch.diff* is non-empty → try unified-diff apply.
         On failure, fall back to full-content write.
      2. If *patch.content* is non-empty → write full file content with .bak backup.
      3. If neither → log and skip.
    """
    for patch in patches:
        target = repo_root / patch.path
        target.parent.mkdir(parents=True, exist_ok=True)

        if patch.diff:
            ok = await apply_diff_patch(patch.diff, repo_root, executor)
            if ok:
                logger.info("CodeEditorTool: diff-patched → %s", target)
                continue
            if not patch.content:
                logger.error(
                    "CodeEditorTool: diff patch failed and no content fallback for %s", target
                )
                continue
            logger.warning(
                "CodeEditorTool: diff failed for %s – falling back to full-content write", target
            )

        if patch.content:
            if target.is_file():
                bak_path = target.with_suffix(target.suffix + ".bak")
                try:
                    bak_path.write_bytes(target.read_bytes())
                except Exception:  # noqa: BLE001
                    pass
            target.write_text(patch.content, encoding="utf-8")
            logger.info("CodeEditorTool: wrote patch → %s", target)
        else:
            logger.warning(
                "CodeEditorTool: patch for %s has no diff or content – skipping", target
            )


# ── Main tool class ───────────────────────────────────────────────────────────


class CodeEditorTool:
    """
    Shared code-editing tool used by DeveloperAgent and CodeFixAgent.

    Supports two editing modes:
      - AI CLI mode  : delegates to `claude` CLI (when available).
      - LLM-direct mode: uses OpenRouter to generate JSON patches.

    Methods:
      apply_changes(task_desc, repo_path)   – apply requested changes
      apply_fix(error_log, repo_path)       – fix sandbox error (LLM-direct only)
      get_repo_tree(repo_path)              – list repo structure
      read_relevant_files(repo_path, hint)  – RAG file reader
    """

    def __init__(
        self,
        llm:     LLMClient,
        ai_cli:  str | None = None,
        timeout: int = 300,
    ) -> None:
        self._llm     = llm
        self._ai_cli  = ai_cli
        self._timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    async def apply_changes(self, task_desc: str, repo_path: Path) -> None:
        """Apply code changes for *task_desc* to the files in *repo_path*."""
        if self._ai_cli:
            executor = CLIExecutor(work_dir=repo_path, timeout=self._timeout)
            await self._apply_with_cli(task_desc, executor)
        else:
            await self._apply_with_llm_direct(task_desc, repo_path)

    async def apply_fix(self, error_log: str, repo_path: Path) -> None:
        """Apply a sandbox-error fix using LLM-direct mode."""
        await self._apply_llm_direct_fix(error_log, repo_path)

    async def get_repo_tree(self, repo_path: Path) -> str:
        """Return a trimmed directory listing for the repo."""
        executor = CLIExecutor(work_dir=repo_path, timeout=15)
        result = await executor.run(
            "find . -not \\( -path './.git' -prune \\) "
            "-not \\( -path './node_modules' -prune \\) "
            "-not \\( -path './__pycache__' -prune \\) "
            "-not \\( -path './.venv' -prune \\) "
            "-type f | sort | head -120"
        )
        return result.stdout or "(empty repo)"

    async def read_relevant_files(self, repo_path: Path, hint: str) -> str:
        """
        Read files most likely relevant to the task.

        Strategy:
          1. rg / grep keyword search to collect candidate files.
          2. Build AST symbol index for the repo (tree-sitter multi-language).
          3. Re-rank candidates + AST top picks via TF-IDF cosine similarity.
          4. Cap total read at 80 KB.
        """
        from src.tools.code_search import build_ast_index, rank_files_by_relevance

        executor = CLIExecutor(work_dir=repo_path, timeout=20)

        keywords = [w for w in re.findall(r"\w+", hint) if len(w) > 3][:6]

        candidate_files: list[str] = []
        if keywords:
            pattern    = "|".join(keywords)
            search_cmd = make_search_command(pattern)
            search_res = await executor.run(search_cmd)
            candidate_files = [
                line.strip().lstrip("./")
                for line in search_res.stdout.splitlines()
                if line.strip()
            ]

        common = [
            "index.html", "App.vue", "App.jsx", "App.tsx", "main.py",
            "src/App.vue", "src/App.jsx", "src/App.tsx",
            "src/main.css", "src/style.css", "src/assets/main.css",
            "src/styles/main.css", "src/index.css",
        ]
        for f in common:
            if (repo_path / f).exists() and f not in candidate_files:
                candidate_files.append(f)

        try:
            symbol_index    = await asyncio.to_thread(build_ast_index, repo_path)
            candidate_files = rank_files_by_relevance(candidate_files, symbol_index, hint)
            new_from_index  = [p for p in symbol_index if p not in set(candidate_files)]
            extra           = rank_files_by_relevance(new_from_index, symbol_index, hint)
            candidate_files += extra[:3]
        except Exception as exc:  # noqa: BLE001
            logger.warning("CodeEditorTool.read_relevant_files: AST/TF-IDF failed (%s)", exc)

        MAX_BYTES = 80_000
        sections: list[str] = []
        total = 0

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

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _apply_with_cli(self, dev_task: str, executor: CLIExecutor) -> None:
        """Run the AI CLI tool to make code changes."""
        cmd    = build_ai_cli_command(dev_task, self._ai_cli)
        result = await executor.run(cmd)
        if not result.succeeded:
            logger.warning(
                "CodeEditorTool: AI CLI non-zero exit: %s", result.stderr[:500]
            )

    async def _apply_with_llm_direct(self, dev_task: str, repo_path: Path) -> None:
        """Use internal LLM to generate and apply code patches."""
        repo_tree     = await self.get_repo_tree(repo_path)
        file_contents = await self.read_relevant_files(repo_path, dev_task)

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
        await apply_patches_to_disk(patches, repo_path, executor)
        logger.info("CodeEditorTool: LLM-direct applied %d file patch(es)", len(patches))

    async def _apply_llm_direct_fix(self, error_log: str, repo_path: Path) -> None:
        """LLM-direct variant for sandbox error retry."""
        repo_tree     = await self.get_repo_tree(repo_path)
        file_contents = await self.read_relevant_files(repo_path, error_log[:200])

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
        await apply_patches_to_disk(patches, repo_path, executor)
        logger.info(
            "CodeEditorTool: LLM-direct fix applied %d file patch(es)", len(patches)
        )

    async def _parse_code_patches(self, raw_json: str) -> list[CodePatch]:
        """
        Parse LLM output into a list of CodePatch objects using a 3-tier
        validation strategy with an automatic LLM-retry fallback.

        Tier 1 – Strict  : json.loads + Pydantic model_validate on every item.
        Tier 2 – Partial : json.loads + Pydantic; silently drop invalid items.
        Tier 3 – Regex   : regex extraction when JSON is totally malformed.
        Tier 4 – LLM retry: final attempt via LLM re-prompt if 0 patches found.
        """
        import json

        clean = re.sub(r"^```[a-z]*\n?", "", raw_json.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\n?```$", "", clean.strip())

        # Tier 1: strict JSON + Pydantic
        try:
            items = json.loads(clean)
            if isinstance(items, list):
                patches = [
                    CodePatch(path=item.path, content=item.content, diff=item.diff)
                    for raw in items
                    for item in [_PatchItem.model_validate(raw)]
                ]
                if patches:
                    logger.debug("_parse_code_patches: Tier 1 OK – %d patches", len(patches))
                    return patches
        except (json.JSONDecodeError, Exception):
            pass

        # Tier 2: partial JSON (skip invalid items)
        patches: list[CodePatch] = []
        try:
            items = json.loads(clean)
            if isinstance(items, list):
                for raw in items:
                    try:
                        item = _PatchItem.model_validate(raw)
                        patches.append(
                            CodePatch(path=item.path, content=item.content, diff=item.diff)
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

        # Tier 3: regex extraction
        for m in re.finditer(
            r'"path"\s*:\s*"([^"]+)".*?(?:"diff"\s*:\s*"((?:[^"\\]|\\.)*)"'
            r'|"content"\s*:\s*"((?:[^"\\]|\\.)*?)")',
            clean,
            re.DOTALL,
        ):
            path, diff_raw, content_raw = m.group(1), m.group(2) or "", m.group(3) or ""
            diff_val    = diff_raw.replace("\\n", "\n").replace("\\\\", "\\")
            content_val = content_raw.replace("\\n", "\n").replace("\\\\", "\\")
            if path:
                patches.append(CodePatch(path=path.strip(), diff=diff_val, content=content_val))

        if patches:
            logger.warning("_parse_code_patches: Tier 3 regex – %d patches", len(patches))
            return patches

        # Tier 4: LLM retry
        logger.warning(
            "_parse_code_patches: all tiers failed – requesting LLM reformat; raw=%s",
            raw_json[:200],
        )
        retry_messages = [
            {"role": "system", "content": _DIRECT_EDIT_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON.\n"
                    "Previous response:\n"
                    f"{raw_json[:1000]}\n\n"
                    "Output ONLY a valid JSON array: "
                    '[{"path": "...", "diff": "..."}]  '
                    'or [{"path": "...", "content": "..."}].  '
                    'Start with "[" immediately.'
                ),
            },
        ]
        try:
            retry_raw   = await self._llm.chat(retry_messages, max_tokens=4096)
            retry_clean = re.sub(r"^```[a-z]*\n?", "", retry_raw.strip(), flags=re.MULTILINE)
            retry_clean = re.sub(r"\n?```$", "", retry_clean.strip())
            items = json.loads(retry_clean)
            if isinstance(items, list):
                for raw in items:
                    try:
                        item = _PatchItem.model_validate(raw)
                        patches.append(
                            CodePatch(path=item.path, content=item.content, diff=item.diff)
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
