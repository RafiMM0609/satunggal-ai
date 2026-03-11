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

logger = logging.getLogger(__name__)

# Direktori repo yang sudah di-clone oleh DeveloperAgent / TechnicalWriterAgent
REPOS_BASE_DIR = Path.home() / "sandbox_repos"

# ── Regex ─────────────────────────────────────────────────────────────────────

_REPO_URL_RE = re.compile(
    r"(https?://(?:github|gitlab|bitbucket)\.com/[\w.\-]+/[\w.\-]+(?:\.git)?)",
    re.IGNORECASE,
)
_FORMAT_PATTERNS = {
    "docx": re.compile(r"\b(word|docx|\.docx)\b", re.IGNORECASE),
    "pdf":  re.compile(r"\b(pdf|\.pdf)\b",         re.IGNORECASE),
}

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
        self._history = history
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)
        logger.info("TechnicalWriterAgent: start session=%s", task.session_id)

        # ── 1. Tentukan format output ──────────────────────────────────────────
        output_format = task.metadata.get("output_format", "").lower()
        if not output_format:
            output_format = _detect_format(task.user_input)
        task.metadata["output_format"] = output_format

        # ── 2. Kumpulkan konteks repo (bidirectional round 1) ─────────────────
        repo_context: Optional[str] = None
        repo_url = _extract_repo_url(task.user_input)

        if repo_url:
            logger.info(
                "TechnicalWriterAgent: repo URL terdeteksi: %s session=%s",
                repo_url, task.session_id,
            )
            task.agent_trace.append(f"technical_writer → gathering repo context: {repo_url}")
            repo_context = await _gather_repo_context(repo_url, task)
            if repo_context:
                task.metadata["repo_context_chars"] = len(repo_context)
                logger.info(
                    "TechnicalWriterAgent: konteks repo %d chars. session=%s",
                    len(repo_context), task.session_id,
                )

        # ── 3. Bangun system prompt dengan konteks ─────────────────────────────
        if repo_context:
            system_prompt = _SYSTEM_PROMPT_WITH_REPO.format(
                base=_SYSTEM_PROMPT_BASE,
                repo_context=repo_context,
            )
        else:
            system_prompt = _SYSTEM_PROMPT_BASE

        # ── 4. Bangun messages (history + user request) ───────────────────────
        messages = self._history.get_as_llm_messages(task.session_id)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": task.user_input})

        # ── 5. Panggil LLM (bidirectional round 2) ────────────────────────────
        task.agent_trace.append("technical_writer → calling LLM for document draft")
        try:
            markdown_doc = await self._llm.complete(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=16_384,  # dokumen teknis bisa panjang; naikkan batas
            )
        except Exception as exc:
            logger.exception("TechnicalWriterAgent: LLM call failed: %s", exc)
            task.mark_failed(f"LLM error: {exc}")
            return task

        # ── 6. Simpan markdown ─────────────────────────────────────────────────
        # Deteksi response terpotong: blok mermaid/code tidak tertutup
        open_fences = markdown_doc.count("```") % 2 != 0
        if open_fences:
            logger.warning(
                "TechnicalWriterAgent: kemungkinan response LLM terpotong "
                "(jumlah fence ``` ganjil). Tambah OPENROUTER_MAX_TOKENS. session=%s",
                task.session_id,
            )
            # Tutup blok yang terbuka agar tools downstream tidak crash
            markdown_doc += "\n```"
        task.metadata["document_markdown"] = markdown_doc
        task.agent_trace.append(
            f"technical_writer → markdown generated ({len(markdown_doc)} chars)"
        )

        # ── 7. Daftarkan tools → orchestrator akan menjalankan (round 3) ──────
        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        # ── 8. Mark done ───────────────────────────────────────────────────────
        fmt_label = output_format.upper() if output_format else "PDF"
        repo_note = f" dari repo `{repo_url}`" if repo_url else ""
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


# ── Repo context gathering ────────────────────────────────────────────────────

async def _gather_repo_context(repo_url: str, task: AgentTask) -> Optional[str]:
    """
    Clone (atau pull jika sudah ada) repo, baca struktur + file kunci,
    kembalikan string konteks siap injeksi ke LLM.
    Ini adalah 'sisi agent' dari komunikasi 2 arah dengan orkestrator —
    agent secara mandiri mengambil data dari sumber eksternal.
    """
    try:
        from src.tools.cli_executor import CLIExecutor  # noqa: PLC0415
    except ImportError:
        logger.warning("TechnicalWriterAgent: CLIExecutor tidak tersedia.")
        return None

    repo_name = _repo_name_from_url(repo_url)
    repo_path = REPOS_BASE_DIR / repo_name
    cli = CLIExecutor(work_dir=str(REPOS_BASE_DIR), timeout=120)

    # Clone atau pull
    if repo_path.exists():
        logger.info("TechnicalWriterAgent: repo sudah ada, git pull: %s", repo_path)
        task.agent_trace.append(f"technical_writer → git pull {repo_name}")
        await cli.run(f"git -C {repo_path} pull --ff-only", work_dir=str(repo_path))
    else:
        logger.info("TechnicalWriterAgent: git clone %s → %s", repo_url, repo_path)
        task.agent_trace.append(f"technical_writer → git clone {repo_url}")
        REPOS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        result = await cli.run(f"git clone --depth=1 {repo_url} {repo_path}")
        if result.returncode != 0:
            logger.error(
                "TechnicalWriterAgent: clone gagal: %s", result.stderr
            )
            task.agent_trace.append(
                f"technical_writer → clone failed: {result.stderr[:200]}"
            )
            return None

    # Baca struktur direktori
    context_parts: list[str] = [f"## Repo: {repo_url}\n"]
    struct_result = await cli.run(
        "find . -type f "
        r"! -path '*/.git/*' ! -path '*/node_modules/*' ! -path '*/__pycache__/*' "
        "! -path '*/.venv/*' ! -path '*/dist/*' ! -path '*/build/*' "
        "| sort | head -150",
        work_dir=str(repo_path),
    )
    if struct_result.stdout:
        context_parts.append("### Struktur File\n```\n" + struct_result.stdout.strip() + "\n```\n")
        task.agent_trace.append("technical_writer → repo structure read")

    # Baca file-file kunci
    for fname in _KEY_FILES:
        fpath = repo_path / fname
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                # Truncate per file agar tidak terlalu panjang
                if len(content) > 3000:
                    content = content[:3000] + "\n... [dipotong]"
                context_parts.append(f"### {fname}\n```\n{content}\n```\n")
                task.agent_trace.append(f"technical_writer → read {fname}")
            except Exception as exc:
                logger.warning("TechnicalWriterAgent: gagal baca %s: %s", fname, exc)

    full_context = "\n".join(context_parts)

    # Batasi total karakter agar tidak overflow context LLM
    if len(full_context) > _MAX_CONTEXT_CHARS:
        full_context = full_context[:_MAX_CONTEXT_CHARS] + "\n... [konteks dipotong]"

    return full_context


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_repo_url(text: str) -> Optional[str]:
    """Ekstrak URL GitHub/GitLab/Bitbucket dari teks pengguna."""
    m = _REPO_URL_RE.search(text)
    return m.group(1).rstrip("/") if m else None


def _repo_name_from_url(url: str) -> str:
    """Ambil nama repo dari URL: 'github.com/user/repo' → 'repo'."""
    return url.rstrip("/").split("/")[-1].removesuffix(".git")


def _detect_format(text: str) -> str:
    """Deteksi format output dari teks pengguna. Default: pdf."""
    for fmt, pattern in _FORMAT_PATTERNS.items():
        if pattern.search(text):
            return fmt
    return "pdf"
