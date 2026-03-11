"""
TechnicalWriterAgent – menghasilkan dokumen teknis profesional dalam format Markdown.

Alur 2-arah (bidirectional) antara agent dan orkestrator:

Round 1 — Konteks (jika ada repo URL):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  User: "buat dokumen PDF untuk repo github.com/xxx/yyy"            │
  │  Gatekeeper → intent = document_creation                           │
  │  Orchestrator → TechnicalWriterAgent.run(task)                     │
  │  Agent: clone/pull repo → baca struktur + file kunci              │
  │  Agent: bangun repo_context string                                  │
  └─────────────────────────────────────────────────────────────────────┘

Round 2 — Penulisan dokumen (LLM dengan konteks lengkap):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Agent: panggil LLM dengan system_prompt + repo_context + request  │
  │  LLM → Markdown lengkap dengan diagram Mermaid                     │
  │  Agent: simpan ke task.metadata["document_markdown"]               │
  │  Agent: tambah pending_tools = ["diagram_renderer","doc_generator"]│
  └─────────────────────────────────────────────────────────────────────┘

Round 3 — Kompilasi (orchestrator menjalankan tools):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DiagramRendererTool  → render blok Mermaid → PNG                  │
  │  DocumentGeneratorTool → Markdown + PNG → PDF/DOCX                 │
  │  Orchestrator → task.metadata["document_path"] = "/tmp/.../doc.pdf"│
  │  Responder → kirim file ke Telegram                                │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor
from src.tools.git_utils import (
    inject_pat_into_url as _inject_pat_into_url,
    is_gitlab_url       as _is_gitlab_url,
    repo_name_from_url  as _repo_name_from_url,
)

logger = logging.getLogger(__name__)

# ── Regex ─────────────────────────────────────────────────────────────────────

_REPO_URL_RE = re.compile(
    r"(https?://[^\s/]+/[\w.\-]+/[\w.\-]+(?:\.git)?)",
    re.IGNORECASE,
)
_FORMAT_PATTERNS = {
    "docx": re.compile(r"\b(word|docx|\.docx)\b", re.IGNORECASE),
    "pdf":  re.compile(r"\b(pdf|\.pdf)\b",         re.IGNORECASE),
}
# Regex untuk mendeteksi nama branch dari input pengguna
_BRANCH_RE = re.compile(
    r"(?:branch|cabang)[\s:]+([\w\-./]+)",
    re.IGNORECASE,
)

# ── Branch confirmation state (per session) ───────────────────────────────────

_tw_pending_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}


def _resolve_branch_from_reply(user_input: str, detected_branch: str) -> str | None:
    """
    Parse balasan konfirmasi user dan kembalikan nama branch yang akan digunakan.
    Mengembalikan None jika bukan konfirmasi yang dikenali.
    """
    clean = user_input.strip()
    lower = clean.lower()
    if lower in _CONFIRMATION_ANSWERS:
        return detected_branch
    if len(clean) <= 100 and " " not in clean and re.match(r"^[\w\-./]+$", clean):
        return clean
    return None

# File kunci yang selalu dibaca jika ada (urutan prioritas)
_KEY_FILES = [
    "README.md", "README.rst", "readme.md",
    "requirements.txt", "package.json", "Pipfile", "pyproject.toml",
    "docker-compose.yml", "Dockerfile",
    "main.py", "app.py", "index.js", "index.ts", "src/index.ts", "src/main.ts",
]

# Batas ukuran konten repo yang dikirim ke LLM
_MAX_CONTEXT_CHARS = 12_000

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
Kamu adalah **Senior Technical Writer** yang berpengalaman membuat dokumen teknis \
profesional untuk tim engineering dan manajemen.

TUGASMU:
Susun dokumen teknis yang objektif, rapi, dan mudah dipahami dalam format **Markdown**.

ATURAN WAJIB:
1. Gunakan struktur BAB dengan Heading 1 (`# Judul BAB`) dan Sub-bab (`## Sub-bab`).
2. Setiap alur proses, arsitektur, atau flow data WAJIB digambarkan dengan blok \
kode **mermaid**:
   ```mermaid
   graph TD
       A[Mulai] --> B[Proses] --> C[Selesai]
   ```
3. Penjelasan teknis namun mudah dipahami oleh engineer junior.
4. Fokus pada fakta; hindari opini tanpa dasar.
5. Sertakan bagian yang relevan:
   - Ringkasan Eksekutif
   - Arsitektur & Komponen Utama
   - Alur Kerja (Workflow)
   - Spesifikasi Teknis & Dependensi
   - Cara Instalasi & Penggunaan
   - Pertimbangan & Catatan
6. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
7. HANYA kembalikan konten Markdown — jangan bungkus dengan backtick tambahan \
di luar konten dokumen.
"""

_SYSTEM_PROMPT_WITH_REPO = """\
{base}

KONTEKS REPOSITORI:
Kamu telah diberi akses ke struktur dan isi dari repositori berikut.
Gunakan data ini sebagai SUMBER UTAMA untuk dokumen — jangan mengarang fitur \
yang tidak ada di kode.

{repo_context}
"""


class TechnicalWriterAgent(BaseAgent):
    """Menghasilkan dokumen teknis Markdown dengan konteks repo penuh,
    lalu mendelegasikan render + kompilasi ke tools via pending_tools.
    """

    name = "technical_writer"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        from config.settings import get_settings
        _settings          = get_settings()
        self._history      = history
        self._llm          = llm or LLMClient()
        self._github_pat   = _settings.github_pat
        self._gitlab_pat   = _settings.gitlab_pat
        self._repos_dir    = Path(_settings.sandbox_repos_dir).expanduser()
        self._repos_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)
        logger.info("TechnicalWriterAgent: start session=%s", task.session_id)

        # ── Cek pending branch confirmation ───────────────────────────────────
        pending = _tw_pending_confirmations.get(task.session_id)
        if pending:
            branch_choice = _resolve_branch_from_reply(
                task.user_input, pending["detected_branch"]
            )
            if branch_choice is not None:
                del _tw_pending_confirmations[task.session_id]
                repo_path = Path(pending["repo_path"])
                await self._checkout_branch(repo_path, branch_choice)
                return await self._build_document(
                    task,
                    repo_url=pending["repo_url"],
                    repo_path=repo_path,
                    branch=branch_choice,
                )
            # Bukan konfirmasi yang dikenali – lanjut ke parse normal.

        # ── 1. Tentukan format output ──────────────────────────────────────────
        output_format = task.metadata.get("output_format", "").lower()
        if not output_format:
            output_format = _detect_format(task.user_input)
        task.metadata["output_format"] = output_format

        # ── 2. Kumpulkan konteks repo (bidirectional round 1) ─────────────────
        repo_url = _extract_repo_url(task.user_input)

        if repo_url:
            logger.info(
                "TechnicalWriterAgent: repo URL terdeteksi: %s session=%s",
                repo_url, task.session_id,
            )
            task.agent_trace.append(f"technical_writer → clone/pull {repo_url}")

            # Clone / pull terlebih dahulu agar bisa cek branch
            try:
                repo_path = await self._clone_or_pull(repo_url, task)
            except Exception as exc:
                logger.error("TechnicalWriterAgent: clone/pull gagal: %s", exc)
                task.mark_failed(f"❌ Gagal mengakses repository: {exc}")
                return task

            # ── Branch selection ───────────────────────────────────────────────
            branch = _extract_branch(task.user_input)
            if branch:
                await self._checkout_branch(repo_path, branch)
                return await self._build_document(task, repo_url, repo_path, branch)
            else:
                detected_branch = await self._get_current_branch(repo_path)
                _tw_pending_confirmations[task.session_id] = {
                    "repo_url":        repo_url,
                    "repo_path":       str(repo_path),
                    "detected_branch": detected_branch,
                }
                task.mark_done(
                    f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                    f"Repository berhasil diakses. Branch aktif saat ini adalah: **`{detected_branch}`**\n\n"
                    f"Dokumen akan dibuat dari branch **`{detected_branch}`**.\n\n"
                    f"Balas **`lanjutkan`** untuk melanjutkan pada branch ini, "
                    f"atau ketik nama branch yang diinginkan "
                    f"(contoh: `develop`, `feature/my-feature`)."
                )
                return task

        # Tidak ada repo URL – langsung ke penulisan tanpa konteks repo
        return await self._build_document(task, repo_url=None, repo_path=None, branch=None)

    # ── Clone / pull ──────────────────────────────────────────────────────────

    async def _clone_or_pull(self, repo_url: str, task: AgentTask) -> Path:
        """
        Clone repo jika belum ada, atau pull jika sudah ada.
        Inject PAT untuk GitHub maupun GitLab secara otomatis.
        """
        repo_name  = _repo_name_from_url(repo_url)
        repo_path  = self._repos_dir / repo_name
        _pat       = self._gitlab_pat if _is_gitlab_url(repo_url) else self._github_pat
        auth_url   = _inject_pat_into_url(repo_url, _pat)
        cli_base   = CLIExecutor(work_dir=self._repos_dir, timeout=120)

        if repo_path.exists():
            logger.info("TechnicalWriterAgent: repo ada, git pull: %s", repo_path)
            task.agent_trace.append(f"technical_writer → git pull {repo_name}")
            cli_repo = CLIExecutor(work_dir=repo_path, timeout=120)
            stash = await cli_repo.run("git stash --include-untracked")
            stashed = stash.succeeded and "No local changes to save" not in (stash.stdout or "")
            await cli_repo.run(f"git pull {auth_url} HEAD --rebase")
            if stashed:
                await cli_repo.run("git stash pop")
        else:
            logger.info("TechnicalWriterAgent: git clone %s → %s", repo_url, repo_path)
            task.agent_trace.append(f"technical_writer → git clone {repo_url}")
            result = await cli_base.run(f"git clone {auth_url} {repo_name}")
            if not result.succeeded:
                raise RuntimeError(
                    f"git clone gagal (exit={result.returncode}):\n"
                    f"{result.stderr[:400]}"
                )

        return repo_path

    # ── Branch helpers ────────────────────────────────────────────────────────

    async def _get_current_branch(self, repo_path: Path) -> str:
        """Kembalikan nama branch yang sedang aktif."""
        cli    = CLIExecutor(work_dir=repo_path, timeout=15)
        result = await cli.run("git rev-parse --abbrev-ref HEAD")
        return (result.stdout or "").strip() or "main"

    async def _checkout_branch(self, repo_path: Path, branch: str) -> None:
        """
        Checkout branch yang diminta.
        Jika branch tidak ada di lokal, fetch remote lalu tracking checkout.
        """
        cli    = CLIExecutor(work_dir=repo_path, timeout=30)
        result = await cli.run(f"git checkout {branch}")
        if result.succeeded:
            logger.info("TechnicalWriterAgent: checked out branch '%s'", branch)
            return
        await cli.run("git remote set-branches origin '*'")
        await cli.run("git fetch --all --prune")
        result = await cli.run(f"git checkout -b {branch} origin/{branch}")
        if not result.succeeded:
            raise RuntimeError(
                f"Branch '{branch}' tidak ditemukan di lokal maupun remote:\n"
                f"{(result.stderr or '')[:400]}"
            )
        logger.info("TechnicalWriterAgent: checked out remote branch '%s'", branch)

    # ── Build document (round 2 & 3) ─────────────────────────────────────────

    async def _build_document(
        self,
        task:      AgentTask,
        repo_url:  Optional[str],
        repo_path: Optional[Path],
        branch:    Optional[str],
    ) -> AgentTask:
        """
        Baca konteks repo (jika ada), panggil LLM, daftarkan pending tools.
        Dapat dipanggil dari run() langsung (branch eksplisit) maupun dari
        alur konfirmasi (setelah user balas).
        """
        output_format = task.metadata.get("output_format") or _detect_format(task.user_input)
        task.metadata["output_format"] = output_format

        # ── Kumpulkan konteks repo ─────────────────────────────────────────────
        repo_context: Optional[str] = None
        if repo_url and repo_path:
            task.agent_trace.append(f"technical_writer → gathering repo context: {repo_url}")
            repo_context = await self._gather_repo_context(repo_url, repo_path, task)
            if repo_context:
                task.metadata["repo_context_chars"] = len(repo_context)
                logger.info(
                    "TechnicalWriterAgent: konteks repo %d chars. session=%s",
                    len(repo_context), task.session_id,
                )

        # ── Bangun system prompt ───────────────────────────────────────────────
        if repo_context:
            system_prompt = _SYSTEM_PROMPT_WITH_REPO.format(
                base=_SYSTEM_PROMPT_BASE,
                repo_context=repo_context,
            )
        else:
            system_prompt = _SYSTEM_PROMPT_BASE

        # ── Bangun messages ────────────────────────────────────────────────────
        messages = self._history.get_as_llm_messages(task.session_id)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": task.user_input})

        # ── Panggil LLM ────────────────────────────────────────────────────────
        task.agent_trace.append("technical_writer → calling LLM for document draft")
        try:
            markdown_doc = await self._llm.complete(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=16_384,
            )
        except Exception as exc:
            logger.exception("TechnicalWriterAgent: LLM call failed: %s", exc)
            task.mark_failed(f"LLM error: {exc}")
            return task

        # ── Simpan markdown ────────────────────────────────────────────────────
        if markdown_doc.count("```") % 2 != 0:
            logger.warning(
                "TechnicalWriterAgent: response LLM kemungkinan terpotong. session=%s",
                task.session_id,
            )
            markdown_doc += "\n```"
        task.metadata["document_markdown"] = markdown_doc
        task.agent_trace.append(
            f"technical_writer → markdown generated ({len(markdown_doc)} chars)"
        )

        # ── Daftarkan tools ────────────────────────────────────────────────────
        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        # ── Mark done ──────────────────────────────────────────────────────────
        fmt_label   = output_format.upper() if output_format else "PDF"
        branch_note = f" branch `{branch}`" if branch else ""
        repo_note   = f" dari repo `{repo_url}`{branch_note}" if repo_url else ""
        reply = (
            f"✅ Dokumen teknis{repo_note} berhasil disusun.\n"
            f"Sedang mengompilasi ke **{fmt_label}** — file akan dikirim sebentar lagi."
        )
        task.mark_done(reply)
        self._history.add(task.session_id, "assistant", reply)

        logger.info(
            "TechnicalWriterAgent: done session=%s pending_tools=%s",
            task.session_id, task.pending_tools,
        )
        return task

    # ── Gather repo context ───────────────────────────────────────────────────

    async def _gather_repo_context(
        self,
        repo_url:  str,
        repo_path: Path,
        task:      AgentTask,
    ) -> Optional[str]:
        """
        Baca struktur direktori + file-file kunci dari repo yang sudah di-clone.
        Dikembalikan sebagai string konteks untuk injeksi ke LLM.
        """
        cli = CLIExecutor(work_dir=repo_path, timeout=30)

        # Struktur direktori
        context_parts: list[str] = [f"## Repo: {repo_url}\n"]
        struct_result = await cli.run(
            "find . -type f "
            r"! -path '*/.git/*' ! -path '*/node_modules/*' ! -path '*/__pycache__/*' "
            "! -path '*/.venv/*' ! -path '*/dist/*' ! -path '*/build/*' "
            "| sort | head -150",
        )
        if struct_result.stdout:
            context_parts.append(
                "### Struktur File\n```\n" + struct_result.stdout.strip() + "\n```\n"
            )
            task.agent_trace.append("technical_writer → repo structure read")

        # File-file kunci
        for fname in _KEY_FILES:
            fpath = repo_path / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 3000:
                        content = content[:3000] + "\n... [dipotong]"
                    context_parts.append(f"### {fname}\n```\n{content}\n```\n")
                    task.agent_trace.append(f"technical_writer → read {fname}")
                except Exception as exc:
                    logger.warning("TechnicalWriterAgent: gagal baca %s: %s", fname, exc)

        full_context = "\n".join(context_parts)
        if len(full_context) > _MAX_CONTEXT_CHARS:
            full_context = full_context[:_MAX_CONTEXT_CHARS] + "\n... [konteks dipotong]"

        return full_context or None


# ── Git URL helpers ───────────────────────────────────────────────────────────

# is_gitlab_url, inject_pat_into_url, repo_name_from_url are imported from
# src.tools.git_utils at the top of this module.
# Self-hosted GitLab instances are handled via the GITLAB_HOSTS setting.


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_repo_url(text: str) -> Optional[str]:
    """Ekstrak URL dari teks pengguna (GitHub, GitLab, Bitbucket, self-hosted)."""
    m = _REPO_URL_RE.search(text)
    return m.group(1).rstrip("/") if m else None


def _extract_branch(text: str) -> str:
    """Ekstrak nama branch dari teks pengguna. Mengembalikan string kosong jika tidak ditemukan."""
    m = _BRANCH_RE.search(text)
    return m.group(1).strip() if m else ""


def _detect_format(text: str) -> str:
    """Deteksi format output dari teks pengguna. Default: pdf."""
    for fmt, pattern in _FORMAT_PATTERNS.items():
        if pattern.search(text):
            return fmt
    return "pdf"
