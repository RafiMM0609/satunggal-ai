"""
TechnicalWriterAgent – menghasilkan dokumen teknis profesional dalam format Markdown.

Alur dengan strategi chunking (untuk repo berskala besar):

Round 1 — Konteks (jika ada repo URL):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  User: "buat dokumen DOCX untuk repo github.com/xxx/yyy"           │
  │  Gatekeeper → intent = document_creation                           │
  │  Orchestrator → TechnicalWriterAgent.run(task)                     │
  │  Agent: clone/pull repo → konfirmasi branch                        │
  └─────────────────────────────────────────────────────────────────────┘

Round 2 — Penulisan dokumen (strategi chunking):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Agent: kumpulkan SEMUA file repo (bukan hanya file kunci)         │
  │  Agent: bagi file menjadi chunks (~8 KB per chunk)                 │
  │  Untuk setiap chunk:                                               │
  │    LLM → section Markdown untuk file-file dalam chunk              │
  │    Append ke file draft .md sementara (tahan crash)                │
  │  Setelah semua chunk: LLM synthesize draft → dokumen final         │
  │  Agent: simpan ke task.metadata["document_markdown"]               │
  │  Agent: tambah pending_tools = ["diagram_renderer","doc_generator"]│
  └─────────────────────────────────────────────────────────────────────┘

Round 3 — Kompilasi (orchestrator menjalankan tools):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DiagramRendererTool  → render blok Mermaid → PNG                  │
  │  DocumentGeneratorTool → Markdown + PNG → PDF/DOCX                 │
  │  Orchestrator → task.metadata["document_path"] = "/tmp/.../doc.docx"│
  │  Responder → kirim file ke Telegram                                │
  └─────────────────────────────────────────────────────────────────────┘

Keunggulan strategi chunking:
  - Hemat memori (RAM): hanya satu chunk yang ada di LLM context setiap saat.
  - Mengatasi batas context window LLM: setiap LLM call kecil dan terfokus.
  - Tahan crash: progres tersimpan bertahap di file draft .md sementara.
  - Tidak ada informasi yang hilang akibat pemotongan paksa.

Tanpa repo URL: fallback ke single-pass LLM call (pertanyaan langsung).
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
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

# ── Chunking constants ─────────────────────────────────────────────────────────

# Ukuran maksimum konten per chunk yang dikirim ke LLM (dalam karakter)
_CHUNK_SIZE_CHARS = 8_000

# Jumlah maksimum file yang dikumpulkan dari repo untuk dokumentasi
_MAX_REPO_FILES = 200

# Maksimum token output LLM per section/chunk
_SECTION_MAX_TOKENS = 2_048

# Maksimum ukuran draft content yang dikirim ke LLM saat synthesis (dalam karakter)
_MAX_DRAFT_SYNTHESIS_CHARS = 30_000

# Maksimum ukuran konten satu file yang disertakan per chunk (dalam karakter)
_MAX_FILE_CONTENT_CHARS = 4_000

# Minimum ukuran konten file agar diikutsertakan (file terlalu pendek diabaikan)
_MIN_FILE_CONTENT_CHARS = 10

# Ekstensi file teks/kode yang akan diproses (ekstensi biner dilewati)
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".scala", ".r",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".sql", ".graphql", ".proto",
    ".xml", ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".tf", ".hcl",
})

# Direktori/file yang dilewati saat mengumpulkan file repo
_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", "coverage", ".nyc_output",
    ".pytest_cache", ".mypy_cache", ".tox",
})

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

# ── Chunking system prompts ────────────────────────────────────────────────────

_CHUNK_ANALYSIS_PROMPT = """\
Kamu adalah **Senior Technical Writer** yang sedang mendokumentasikan repositori \
kode secara bertahap, satu kelompok file per iterasi.

KONTEKS:
- Repositori: {repo_url}
- Sedang memproses chunk {chunk_num} dari {total_chunks}

TUGASMU:
Buat dokumentasi Markdown yang ringkas dan informatif untuk file-file yang diberikan.

Untuk setiap file, dokumentasikan:
- Tujuan/fungsi file tersebut
- Fungsi, kelas, atau komponen utama beserta deskripsi singkatnya
- Dependensi atau import penting yang perlu diketahui

ATURAN:
1. Gunakan heading `## nama/file.ext` untuk setiap file
2. Gunakan sub-heading `### NamaFungsi / NamaKelas` untuk komponen utama
3. Sertakan diagram Mermaid HANYA jika ada alur proses yang jelas dan signifikan
4. Jangan mengarang informasi yang tidak ada di kode
5. Jangan menyertakan blok kode yang terlalu panjang — cukup deskripsikan
6. Gunakan bahasa Indonesia
7. HANYA kembalikan konten Markdown

FILE-FILE YANG PERLU DIDOKUMENTASIKAN:
{chunk_content}
"""

_FINAL_SYNTHESIS_PROMPT = """\
{base}

INSTRUKSI TAMBAHAN – SINTESIS DOKUMEN FINAL:
Kamu diberikan draft dokumentasi yang telah dikumpulkan secara bertahap (chunked) \
dari seluruh file dalam repositori.
Tugasmu adalah menyusun dokumen teknis final yang kohesif dan profesional.

PANDUAN SINTESIS:
1. Buat **Daftar Isi** (Table of Contents) di bagian awal dengan tautan Markdown
2. Tulis bagian **Ringkasan Eksekutif** yang merangkum tujuan dan fungsi utama sistem
3. Tulis bagian **Arsitektur Sistem** dengan diagram Mermaid yang menggambarkan \
komponen-komponen utama dan hubungannya
4. Kelompokkan section-section terkait menjadi bab-bab yang terstruktur logis
5. Hapus duplikasi informasi dan selaraskan terminologi
6. Pastikan konsistensi gaya bahasa di seluruh dokumen
7. HANYA kembalikan konten Markdown final — tanpa pembungkus backtick tambahan

DRAFT DOKUMENTASI (dari seluruh chunk):
{draft_content}
"""


# ── Chunking helper functions ─────────────────────────────────────────────────

def _get_draft_md_path(session_id: str) -> Path:
    """Kembalikan path file draft .md sementara untuk session ini."""
    draft_dir = Path(tempfile.gettempdir()) / "advance_ai_docs" / session_id
    draft_dir.mkdir(parents=True, exist_ok=True)
    return draft_dir / "draft.md"


def _split_files_into_chunks(
    files: list[tuple[str, str]],
    chunk_size: int = _CHUNK_SIZE_CHARS,
) -> list[list[tuple[str, str]]]:
    """
    Bagi list of (filename, content) menjadi chunks berdasarkan total ukuran konten.
    Setiap chunk tidak melebihi chunk_size karakter.
    """
    chunks: list[list[tuple[str, str]]] = []
    current_chunk: list[tuple[str, str]] = []
    current_size = 0

    for fname, content in files:
        entry_size = len(fname) + len(content)
        # Jika menambahkan entry ini akan melampaui batas dan chunk tidak kosong, simpan chunk
        if current_chunk and current_size + entry_size > chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append((fname, content))
        current_size += entry_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


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
        # Coba checkout biasa dulu (branch mungkin sudah ada di lokal setelah fetch)
        result = await cli.run(f"git checkout {branch}")
        if result.succeeded:
            logger.info("TechnicalWriterAgent: checked out branch '%s' after fetch", branch)
            return
        # Branch belum ada di lokal, buat tracking branch dari remote
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
        Titik masuk utama untuk pembuatan dokumen.

        Jika repo tersedia, gunakan strategi chunking (memori hemat, aman untuk
        repo berskala besar).  Tanpa repo, lakukan single-pass LLM call.

        Dapat dipanggil dari run() langsung (branch eksplisit) maupun dari
        alur konfirmasi (setelah user balas).
        """
        output_format = task.metadata.get("output_format") or _detect_format(task.user_input)
        task.metadata["output_format"] = output_format

        # ── Gunakan strategi chunking jika repo tersedia ───────────────────────
        if repo_url and repo_path:
            return await self._chunked_document_generation(task, repo_url, repo_path, branch)

        # ── No-repo: single-pass (tanpa konteks repo) ─────────────────────────
        system_prompt = _SYSTEM_PROMPT_BASE

        messages = self._history.get_as_llm_messages(task.session_id)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": task.user_input})

        task.agent_trace.append("technical_writer → calling LLM for document draft (no repo)")
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

        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        fmt_label = output_format.upper() if output_format else "PDF"
        reply = (
            f"✅ Dokumen teknis berhasil disusun.\n"
            f"Sedang mengompilasi ke **{fmt_label}** — file akan dikirim sebentar lagi."
        )
        task.mark_done(reply)
        self._history.add(task.session_id, "assistant", reply)

        logger.info(
            "TechnicalWriterAgent: done (no-repo) session=%s pending_tools=%s",
            task.session_id, task.pending_tools,
        )
        return task

    # ── Chunked document generation ───────────────────────────────────────────

    async def _chunked_document_generation(
        self,
        task:      AgentTask,
        repo_url:  str,
        repo_path: Path,
        branch:    Optional[str],
    ) -> AgentTask:
        """
        Hasilkan dokumen teknis menggunakan strategi chunking bertahap:

        1. Kumpulkan semua file repo (bukan hanya file kunci).
        2. Bagi menjadi chunks berdasarkan ukuran konten.
        3. Untuk setiap chunk: panggil LLM → hasilkan section Markdown.
        4. Append setiap section ke file draft .md sementara (tahan crash).
        5. Setelah semua chunk selesai, synthesize menjadi dokumen final.
        6. Daftarkan diagram_renderer & document_generator ke pending_tools.

        Strategi ini memastikan:
        - Hemat memori: hanya satu chunk yang ada di memori LLM setiap saat.
        - Toleran crash: progres tersimpan di file draft .md.
        - Tidak ada informasi yang hilang akibat batas konteks LLM.
        """
        output_format = task.metadata.get("output_format", "docx")
        task.metadata["output_format"] = output_format

        # 1. Kumpulkan semua file
        files = await self._gather_all_repo_files(repo_url, repo_path, task)

        if not files:
            # Fallback ke pendekatan lama berbasis file kunci
            logger.warning(
                "TechnicalWriterAgent: tidak ada file terkumpul, fallback ke key-files. session=%s",
                task.session_id,
            )
            return await self._build_document_from_key_files(task, repo_url, repo_path, branch)

        # 2. Bagi menjadi chunks
        chunks = _split_files_into_chunks(files)
        total_chunks = len(chunks)
        logger.info(
            "TechnicalWriterAgent: %d files → %d chunks. session=%s",
            len(files), total_chunks, task.session_id,
        )
        task.agent_trace.append(
            f"technical_writer → chunked mode: {len(files)} files, {total_chunks} chunks"
        )

        # 3. Buat file draft .md sementara
        draft_path = _get_draft_md_path(task.session_id)
        task.metadata["draft_md_path"] = str(draft_path)
        branch_note = f" (branch: {branch})" if branch else ""
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(f"# Draft Dokumentasi: {repo_url}{branch_note}\n\n")

        # 4. Proses setiap chunk → append ke draft
        for i, chunk in enumerate(chunks, start=1):
            logger.info(
                "TechnicalWriterAgent: processing chunk %d/%d. session=%s",
                i, total_chunks, task.session_id,
            )
            section_md = await self._generate_section_for_chunk(
                chunk, repo_url, i, total_chunks
            )
            with open(draft_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n<!-- CHUNK {i}/{total_chunks} -->\n\n")
                f.write(section_md)
            task.agent_trace.append(f"technical_writer → chunk {i}/{total_chunks} selesai")

        # 5. Baca draft dan synthesize menjadi dokumen final
        draft_content = draft_path.read_text(encoding="utf-8")
        logger.info(
            "TechnicalWriterAgent: draft siap (%d chars), memulai sintesis. session=%s",
            len(draft_content), task.session_id,
        )
        final_md = await self._synthesize_final_document(draft_content, repo_url, task)

        if final_md.count("```") % 2 != 0:
            logger.warning(
                "TechnicalWriterAgent: sintesis kemungkinan terpotong. session=%s",
                task.session_id,
            )
            final_md += "\n```"

        task.metadata["document_markdown"] = final_md
        task.agent_trace.append(
            f"technical_writer → final markdown synthesized ({len(final_md)} chars)"
        )

        # 6. Daftarkan tools
        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        # 7. Mark done
        fmt_label = output_format.upper()
        reply = (
            f"✅ Dokumen teknis dari repo `{repo_url}`{branch_note} berhasil disusun "
            f"menggunakan strategi chunking "
            f"({total_chunks} chunks, {len(files)} files).\n"
            f"Sedang mengompilasi ke **{fmt_label}** — file akan dikirim sebentar lagi."
        )
        task.mark_done(reply)
        self._history.add(task.session_id, "assistant", reply)

        logger.info(
            "TechnicalWriterAgent: chunked generation done session=%s pending_tools=%s",
            task.session_id, task.pending_tools,
        )
        return task

    # ── Gather all repo files ─────────────────────────────────────────────────

    async def _gather_all_repo_files(
        self,
        repo_url:  str,
        repo_path: Path,
        task:      AgentTask,
    ) -> list[tuple[str, str]]:
        """
        Kumpulkan semua file teks/kode dari repo (bukan hanya file kunci).
        Kembalikan list of (relative_path, content).
        File biner dan direktori yang di-skip diabaikan.
        """
        files: list[tuple[str, str]] = []
        count = 0

        for fpath in sorted(repo_path.rglob("*")):
            if count >= _MAX_REPO_FILES:
                logger.info(
                    "TechnicalWriterAgent: batas %d file tercapai. session=%s",
                    _MAX_REPO_FILES, task.session_id,
                )
                break

            if not fpath.is_file():
                continue

            # Lewati direktori yang di-skip dan file tersembunyi (hidden)
            parts = fpath.relative_to(repo_path).parts
            if any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in parts[:-1]):
                continue
            if parts[-1].startswith("."):
                continue

            # Lewati file dengan ekstensi biner
            suffix = fpath.suffix.lower()
            if suffix and suffix not in _TEXT_EXTENSIONS:
                continue
            # File tanpa ekstensi: coba baca (mungkin script)
            # File dengan ekstensi teks: baca
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Lewati file yang terlalu pendek (mungkin kosong atau tidak informatif)
            if len(content.strip()) < _MIN_FILE_CONTENT_CHARS:
                continue

            rel_path = str(fpath.relative_to(repo_path))
            # Potong file yang sangat panjang agar satu file tidak mendominasi chunk
            if len(content) > _MAX_FILE_CONTENT_CHARS:
                content = content[:_MAX_FILE_CONTENT_CHARS] + "\n... [dipotong]"

            files.append((rel_path, content))
            count += 1

        task.agent_trace.append(
            f"technical_writer → gathered {len(files)} files for chunked processing"
        )
        logger.info(
            "TechnicalWriterAgent: %d files dikumpulkan dari %s. session=%s",
            len(files), repo_url, task.session_id,
        )
        return files

    # ── LLM calls per chunk ────────────────────────────────────────────────────

    async def _generate_section_for_chunk(
        self,
        chunk_files:   list[tuple[str, str]],
        repo_url:      str,
        chunk_num:     int,
        total_chunks:  int,
    ) -> str:
        """Panggil LLM untuk mendokumentasikan satu chunk file."""
        chunk_content_parts: list[str] = []
        for fname, content in chunk_files:
            # Deteksi bahasa dari ekstensi file untuk syntax highlighting
            lang = Path(fname).suffix.lstrip(".") or "text"
            chunk_content_parts.append(f"### File: `{fname}`\n```{lang}\n{content}\n```")
        chunk_content = "\n\n".join(chunk_content_parts)

        system_prompt = _CHUNK_ANALYSIS_PROMPT.format(
            repo_url=repo_url,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            chunk_content=chunk_content,
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Dokumentasikan file-file pada chunk {chunk_num}/{total_chunks} "
                    f"dari repositori {repo_url}."
                ),
            }
        ]

        try:
            section_md = await self._llm.complete(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=_SECTION_MAX_TOKENS,
            )
            return section_md.strip()
        except Exception as exc:
            logger.warning(
                "TechnicalWriterAgent: chunk %d LLM gagal: %s", chunk_num, exc
            )
            # Kembalikan placeholder agar proses tidak terhenti
            return f"<!-- chunk {chunk_num} gagal diproses: {exc} -->"

    async def _synthesize_final_document(
        self,
        draft_content: str,
        repo_url:      str,
        task:          AgentTask,
    ) -> str:
        """
        Buat dokumen final yang kohesif dari seluruh section draft.
        Draft content dipotong jika melebihi batas aman untuk LLM.
        """
        # Potong draft jika terlalu besar agar tidak melebihi context window
        if len(draft_content) > _MAX_DRAFT_SYNTHESIS_CHARS:
            draft_content = (
                draft_content[:_MAX_DRAFT_SYNTHESIS_CHARS]
                + "\n\n... [draft dipotong karena melebihi batas]"
            )

        system_prompt = _FINAL_SYNTHESIS_PROMPT.format(
            base=_SYSTEM_PROMPT_BASE,
            draft_content=draft_content,
        )
        messages = [
            {
                "role": "user",
                "content": f"Susun dokumen teknis final untuk repositori: {repo_url}",
            }
        ]
        task.agent_trace.append("technical_writer → synthesizing final document from chunks")

        try:
            final_md = await self._llm.complete(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=16_384,
            )
            return final_md.strip()
        except Exception as exc:
            logger.exception(
                "TechnicalWriterAgent: synthesis LLM gagal: %s", exc
            )
            # Fallback: kembalikan draft mentah
            logger.warning(
                "TechnicalWriterAgent: menggunakan draft mentah sebagai fallback. session=%s",
                task.session_id,
            )
            return draft_content

    # ── Fallback: key-files only (digunakan ketika tidak ada file yang terkumpul) ──

    async def _build_document_from_key_files(
        self,
        task:      AgentTask,
        repo_url:  str,
        repo_path: Path,
        branch:    Optional[str],
    ) -> AgentTask:
        """
        Fallback ke pendekatan lama: baca hanya file kunci dan buat dokumen
        dalam satu LLM call. Digunakan bila _gather_all_repo_files tidak menemukan file.
        """
        repo_context = await self._gather_repo_context(repo_url, repo_path, task)
        output_format = task.metadata.get("output_format", "pdf")

        if repo_context:
            task.metadata["repo_context_chars"] = len(repo_context)
            system_prompt = _SYSTEM_PROMPT_WITH_REPO.format(
                base=_SYSTEM_PROMPT_BASE,
                repo_context=repo_context,
            )
        else:
            system_prompt = _SYSTEM_PROMPT_BASE

        messages = self._history.get_as_llm_messages(task.session_id)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": task.user_input})

        task.agent_trace.append("technical_writer → calling LLM (key-files fallback)")
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

        if markdown_doc.count("```") % 2 != 0:
            markdown_doc += "\n```"
        task.metadata["document_markdown"] = markdown_doc
        task.agent_trace.append(
            f"technical_writer → markdown generated ({len(markdown_doc)} chars)"
        )

        task.pending_tools.extend(["diagram_renderer", "document_generator"])

        fmt_label   = output_format.upper() if output_format else "PDF"
        branch_note = f" branch `{branch}`" if branch else ""
        reply = (
            f"✅ Dokumen teknis dari repo `{repo_url}`{branch_note} berhasil disusun.\n"
            f"Sedang mengompilasi ke **{fmt_label}** — file akan dikirim sebentar lagi."
        )
        task.mark_done(reply)
        self._history.add(task.session_id, "assistant", reply)
        return task

    # ── Gather repo context (key-files only, dipertahankan untuk fallback) ────

    async def _gather_repo_context(
        self,
        repo_url:  str,
        repo_path: Path,
        task:      AgentTask,
    ) -> Optional[str]:
        """
        Baca struktur direktori + file-file kunci dari repo yang sudah di-clone.
        Dikembalikan sebagai string konteks untuk injeksi ke LLM.
        Digunakan sebagai fallback oleh _build_document_from_key_files.
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
