"""
DeveloperInspectorAgent – Repository Inspector & Diagnostician.

Peran:
  Inspektor senior yang MEMBACA dan MENGANALISIS repositori kode.
  Ia TIDAK menulis kode, TIDAK mengedit file, TIDAK melakukan commit atau push.
  Tugasnya adalah:
    1. Menginspeksi struktur dan isi repositori secara menyeluruh.
    2. Mengidentifikasi akar penyebab masalah (root cause analysis).
    3. Menyusun laporan inspeksi yang jelas dan actionable.
    4. Memberikan rekomendasi perbaikan yang spesifik kepada developer.

Workflow:
  1. Ekstrak repo_url + deskripsi masalah dari input pengguna via LLM.
  2. Clone repo (jika URL disertakan) atau gunakan repo yang sudah ada via RepoTracker.
  3. Jalankan pemeriksaan read-only secara paralel:
       - Struktur direktori (ls -R)
       - Git log & diff terbaru
       - Grep untuk pola error / keyword masalah
       - Baca file-file kunci (entry points, config, bagian yang dicurigai)
       - RAG: file relevan via AST index + TF-IDF
  4. Kirim semua temuan ke LLM untuk analisis mendalam (Phase 1).
  5. Jalankan critic pass (Phase 2) untuk verifikasi setiap temuan.
  6. Kembalikan laporan inspeksi terstruktur yang telah diverifikasi.

Batasan penting:
  - READ-ONLY: tidak ada git add/commit/push.
  - Tidak ada eksekusi kode (compile, run, docker).
  - Tidak ada penulisan atau pengeditan file apapun di repo.

Untuk tanya-jawab tentang isi repositori (API apa, tech stack apa, dll.),
gunakan DeveloperQnAAgent (intent: code_understanding).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from src.agents.repo_agent_base import RepoAgentBase, RepoExtractionRequest
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask
from src.tools.cli_executor import CLIExecutor

logger = logging.getLogger(__name__)

# ── Inspector-specific constants ───────────────────────────────────────────────

# LLM sampling parameters – low temperature for determinism, less hallucination.
INSPECTOR_TEMPERATURE = 0.15
INSPECTOR_TOP_P       = 0.90

# Critic (second-pass) uses even lower temperature for strict fact-checking.
CRITIC_TEMPERATURE = 0.10
CRITIC_TOP_P       = 0.85

MAX_GREP_LINES = 80

# Maximum characters sent as evidence to the LLM in a single request.
# Keeps the total prompt within a safe context window (~60 k tokens ≈ 240 k chars,
# but we cap conservatively to leave room for the system prompt and the response).
MAX_EVIDENCE_CHARS = 60_000

# ── Progressive Deepening constants (Items 5, 6) ──────────────────────────────
# Maximum LLM calls for a single inspection (token budget).
MAX_INVESTIGATION_PHASES = 3

# If evidence is truncated by more than this fraction, add the "Data Not Enough"
# instruction so the LLM can explicitly flag what's missing (Item 7).
_TRUNCATION_WARN_FRACTION = 0.20   # 20 %

# Maximum files that the hypothesis LLM may request in phase 1.
_MAX_SUSPECTED_FILES = 5

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah **Inspektor Kode Senior** (Developer Inspector).

Identitasmu:
- Kamu adalah INSPEKTOR, bukan programmer.
- Kamu membaca dan MENGANALISIS kode, tapi kamu TIDAK menulis, mengedit, \
  atau mengeksekusi kode apapun.
- Kamu seperti seorang detektif teknis: mengumpulkan bukti, mengidentifikasi \
  akar masalah, dan memberikan rekomendasi yang akurat berdasarkan FAKTA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ATURAN KRITIS – ANTI-HALUSINASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **DILARANG KERAS** membuat klaim tanpa bukti dari data yang diberikan.
2. Setiap temuan HARUS disertai kutipan langsung (exact quote) dari \
file/log/diff yang ada dalam evidence.
3. Tulis **[PERLU VERIFIKASI]** jika kamu menduga adanya masalah tapi tidak ada bukti \
langsung dalam data yang diberikan.
4. Tulis **[DATA TIDAK CUKUP]** jika data tidak memungkinkan diagnosis akurat \
daripada menebak-nebak.
5. **JANGAN** mengasumsikan struktur kode, naming convention, atau bug yang tidak \
terlihat dalam evidence.
6. Jika ada ketidakpastian, nyatakan tingkat kepercayaan:
   - 🟢 **[CONFIRMED]** – bukti kuat, dikutip langsung dari kode/log.
   - 🟡 **[LIKELY]** – indikasi kuat tapi perlu verifikasi tambahan.
   - 🔴 **[UNVERIFIED]** – dugaan tanpa bukti langsung; harus diverifikasi developer.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tugasmu adalah menghasilkan Laporan Inspeksi Repositori yang mencakup:

---

## 📋 LAPORAN INSPEKSI REPOSITORI

### 1. Ringkasan Eksekutif
Jelaskan secara singkat apa yang ditemukan (2-4 kalimat). Nyatakan secara jelasnya \
apa yang SUDAH DIVERIFIKASI vs yang MASIH DUGAAN.

### 2. Struktur Proyek
Deskripsikan arsitektur dan organisasi kode **berdasarkan directory tree yang diberikan**. \
Hanya deskripsikan yang TERLIHAT dalam data.

### 3. 🔍 Temuan Masalah
Daftar semua masalah teridentifikasi dengan tingkat keparahan:
- 🔴 **KRITIS**: Masalah yang menyebabkan sistem tidak berfungsi.
- 🟡 **SEDANG**: Degradasi performa atau fungsionalitas.
- 🟢 **RINGAN**: Masalah kode yang tidak urgent.

Untuk setiap masalah, sertakan wajib:
  - **Lokasi**: Nama file dan nomor baris (exact, bukan perkiraan).
  - **Bukti** (WAJIB): Cuplikan kode/log yang dikutip PERSIS dari evidence. \
Jika tidak ada kutipan, tambahkan tanda 🔴 **[UNVERIFIED]**.
  - **Deskripsi**: Apa yang salah dan mengapa ini masalah.
  - **Kepercayaan**: 🟢 CONFIRMED / 🟡 LIKELY / 🔴 UNVERIFIED.

### 4. 🎯 Analisis Akar Masalah (Root Cause)
Jelaskan mengapa masalah ini terjadi secara teknis dan mendalam, dengan merujuk \
pada bukti spesifik dari kode.

### 5. 💡 Rekomendasi Perbaikan
Langkah perbaikan spesifik dan actionable, diurutkan berdasarkan prioritas.
- Sertakan nama file dan fungsi yang perlu diubah (bukan secara umum).
- Berikan pseudocode atau contoh pattern yang harus diterapkan developer.
- Hanya rekomendasikan perubahan yang didukung oleh temuan nyata.

### 6. ⚠️ Risiko Jika Tidak Diperbaiki
Dampak potensial jika masalah dibiarkan (berbasis temuan yang CONFIRMED).

### 7. 📊 Ringkasan Kepercayaan
| Temuan | Status | Dasar Bukti |
|--------|--------|-------------|
(Tabel semua temuan dengan status CONFIRMED/LIKELY/UNVERIFIED)

---

**Aturan Format:**
1. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
2. Jujur: jika data tidak cukup, katakan demikian — jangan mengarang.
3. Spesifik: kutip file, baris, dan nama fungsi dengan tepat.
4. Profesional: objektif, berbasis data, hindari menyalahkan developer.
"""

# ── Critic (second-pass verification) prompt ─────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """\
Kamu adalah **Reviewer Laporan Inspeksi Kode** yang tugasnya mem-verifikasi \
setiap klaim dalam laporan inspeksi terhadap evidence yang diberikan.

Tugasmu:
1. Baca setiap temuan dalam Laporan Inspeksi.
2. Periksa apakah temuan tersebut didukung oleh kutipan langsung dalam evidence.
3. Perbarui status kepercayaan setiap temuan:
   - 🟢 **[CONFIRMED]** jika ada kutipan langsung dari kode/log.
   - 🟡 **[LIKELY]** jika ada indikasi kuat tapi tidak dikutip langsung.
   - 🔴 **[UNVERIFIED]** jika tidak ada bukti dalam evidence.
4. Hapus atau tandai klaim yang tidak bisa diverifikasi.
5. Pertahankan semua temuan yang valid, tambahkan kutipan yang terlewat jika ada dalam evidence.
6. Tambahkan catatan reviewer di awal laporan: berapa temuan CONFIRMED, LIKELY, UNVERIFIED.

Jangan ubah gaya penulisan laporan. Kembalikan laporan lengkap yang sudah diverifikasi.
"""

_CRITIC_USER_TEMPLATE = """\
## LAPORAN INSPEKSI (perlu diverifikasi)

{report}

---

## EVIDENCE YANG TERSEDIA

{evidence}

---

Verifikasi setiap temuan dalam laporan terhadap evidence di atas. \
Perbarui status [CONFIRMED/LIKELY/UNVERIFIED] dan tambahkan/perbaiki kutipan bukti.
"""

# ── Progressive Deepening: phase-1 hypothesis prompt (Item 5) ─────────────────

_HYPOTHESIS_SYSTEM_PROMPT = f"""\
Kamu adalah inspektor kode yang melakukan investigasi bertahap.
Berdasarkan data awal berikut (directory tree + git log + grep), berikan:
1. Hipotesis awal singkat tentang penyebab masalah (1-2 kalimat).
2. Daftar file yang perlu dibuka untuk mengkonfirmasi atau membantah hipotesis.
3. Keyword tambahan untuk grep lanjutan jika dibutuhkan.

Balas HANYA dengan JSON valid (tidak ada teks lain):
{{
  "hypothesis": "<hipotesis dalam 1-2 kalimat>",
  "suspected_files": ["<repo/relative/path/file1.py>", "<path/file2.go>"],
  "additional_keywords": ["<keyword>"]
}}

Maksimal {_MAX_SUSPECTED_FILES} file dalam suspected_files. Jika data sudah cukup, gunakan daftar kosong [].
"""

# Regex to detect the "Data Not Enough" signal emitted by the LLM (Item 7).
_DATA_NEEDED_RE = re.compile(
    r"\[(?:DATA TIDAK CUKUP|PERLU DATA TAMBAHAN)[^\]]*\]"
    r"|PERLU VERIFIKASI TAMBAHAN\s*[:\-]?\s*([\w/\.]+)",
    re.IGNORECASE,
)

# Q/A mode LLM prompt
# (Q/A prompt has moved to DeveloperQnAAgent – src/agents/developer_qna/agent.py)

# ── Branch confirmation state ─────────────────────────────────────────────────

_inspector_pending_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}


def _resolve_branch_from_reply(user_input: str, detected_branch: str) -> str | None:
    """Parse the user's confirmation reply; return branch name or None."""
    clean = user_input.strip()
    lower = clean.lower()
    if lower in _CONFIRMATION_ANSWERS:
        return detected_branch
    if len(clean) <= 100 and " " not in clean and re.match(r"^[\w\-./]+$", clean):
        return clean
    return None


# ── Agent ──────────────────────────────────────────────────────────────────────

class DeveloperInspectorAgent(RepoAgentBase):
    """
    Read-only repository inspector.

    Collects evidence from the codebase via shell commands,
    then delegates root-cause analysis to the LLM (2-phase: report + critic).

    For Q/A about repository content (APIs, tech stack, data models, etc.),
    use DeveloperQnAAgent instead (intent: code_understanding).
    """

    name = "developer_inspector"

    def __init__(
        self,
        llm: LLMClient | None = None,
        history=None,
    ) -> None:
        super().__init__(llm=llm, history=history)

    # ── Inspection-specific evidence helpers ───────────────────────────────────

    async def _get_git_log(self, repo_path: Path) -> str:
        from src.agents.repo_agent_base import MAX_LOG_LINES
        out = await self._run_cmd(
            f"git log --oneline -n {MAX_LOG_LINES}",
            cwd=repo_path,
        )
        return out or "(no git log)"

    async def _get_git_diff(self, repo_path: Path) -> str:
        from src.agents.repo_agent_base import MAX_DIFF_LINES
        out = await self._run_cmd(
            f"git diff HEAD~1 HEAD --stat 2>/dev/null | head -{MAX_DIFF_LINES}",
            cwd=repo_path,
        )
        return out or "(no diff)"

    async def _grep_keywords(self, repo_path: Path, keywords: list[str]) -> str:
        if not keywords:
            return "(no keywords specified)"
        pattern = "|".join(re.escape(k) for k in keywords[:5])
        out = await self._run_cmd(
            f"grep -rn "
            f"--include='*.py' --include='*.js' --include='*.ts' "
            f"--include='*.go' --include='*.java' --include='*.rb' "
            f"--include='*.php' --include='*.cs' --include='*.rs' "
            f"--include='*.vue' --include='*.tsx' --include='*.jsx' "
            f"-E '{pattern}' . 2>/dev/null | head -{MAX_GREP_LINES}",
            cwd=repo_path,
        )
        return out or f"(no matches for: {', '.join(keywords)})"

    async def _grep_error_patterns(self, repo_path: Path) -> str:
        """Grep for generic error/exception patterns across common source files."""
        error_pattern = (
            r"(Exception|Error|Traceback|panic:|FATAL|CRITICAL"
            r"|undefined is not|cannot read property|NullPointerException"
            r"|segfault|SIGSEGV|stack overflow|out of memory)"
        )
        out = await self._run_cmd(
            f"grep -rn "
            f"--include='*.py' --include='*.js' --include='*.ts' "
            f"--include='*.go' --include='*.java' --include='*.rs' "
            f"-iE '{error_pattern}' . 2>/dev/null | head -{MAX_GREP_LINES}",
            cwd=repo_path,
        )
        return out or "(no generic error patterns found)"

    # ── Critic (second-pass verification) ─────────────────────────────────────

    async def _verify_report(self, report: str, evidence_text: str) -> str:
        """
        Critic second-pass: ask the LLM to cross-check every finding in the
        initial report against the raw evidence and update confidence labels.
        """
        logger.info("Inspector: running critic verification pass")
        critic_user = _CRITIC_USER_TEMPLATE.format(
            report=report,
            evidence=evidence_text[:MAX_EVIDENCE_CHARS],
        )
        try:
            verified = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": critic_user},
                ],
                temperature=CRITIC_TEMPERATURE,
                top_p=CRITIC_TOP_P,
            )
            return verified.strip() or report
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inspector: critic pass failed (%s); using initial report", exc)
            return report

    # ── LLM inspection pipeline ────────────────────────────────────────────────

    async def _run_inspection_llm(
        self,
        user_input: str,
        problem:    str,
        evidence:   dict[str, str],
    ) -> str:
        """
        Generate an inspection report via two LLM passes:
          Phase 1 – Initial report from compressed evidence.
          Phase 2 – Critic pass: verify every finding against raw evidence.

        Before assembling the evidence string, large sections are compressed via
        LLM so that every section is represented rather than simply chopped off
        (Item 1: Hierarchical Summarization).

        When evidence is still too large after compression, a "Data Not Enough"
        instruction is injected so the LLM explicitly flags missing information
        instead of hallucinating (Item 7).
        """
        # ── Step 1: Compress oversized sections (Item 1) ──────────────────
        compressed_evidence = await self._compress_evidence_dict(evidence)
        evidence_text = self._build_evidence_text(compressed_evidence)

        original_len = sum(len(v) for v in evidence.values())
        logger.debug(
            "Inspector: evidence compressed %d → %d chars (%d sections)",
            original_len, len(evidence_text), len(evidence),
        )

        # ── Step 2: Hard-truncate if still too large ──────────────────────
        _TRUNCATION_NOTICE = (
            f"\n\n... [evidence truncated at {MAX_EVIDENCE_CHARS} characters due to LLM context limit]"
        )
        truncated = False
        if len(evidence_text) > MAX_EVIDENCE_CHARS:
            truncation_amount = len(evidence_text) - MAX_EVIDENCE_CHARS
            truncation_fraction = truncation_amount / len(evidence_text)
            logger.warning(
                "Inspector: evidence still too large after compression (%d chars), "
                "truncating to %d chars (%.0f%% lost)",
                len(evidence_text), MAX_EVIDENCE_CHARS, truncation_fraction * 100,
            )
            evidence_text = (
                evidence_text[:MAX_EVIDENCE_CHARS - len(_TRUNCATION_NOTICE)]
                + _TRUNCATION_NOTICE
            )
            truncated = truncation_fraction > _TRUNCATION_WARN_FRACTION

        for title, content in evidence.items():
            logger.debug("Inspector evidence '%s': %d chars", title, len(content))
        logger.info(
            "Inspector: sending %d evidence sections (%d total chars) to LLM",
            len(evidence), len(evidence_text),
        )

        # ── Step 3: Add "Data Not Enough" instruction if evidence was truncated ─
        # (Item 7): Guide the LLM to explicitly signal missing information.
        data_needed_instruction = ""
        if truncated:
            data_needed_instruction = (
                "\n\n**⚠️ PERHATIAN: Evidence mungkin tidak lengkap akibat pemotongan.**\n"
                "Jika ada temuan yang tidak dapat dikonfirmasi karena data kurang:\n"
                "1. Tandai dengan 🔴 **[DATA TIDAK CUKUP]**\n"
                "2. Sebutkan secara eksplisit file atau informasi apa yang masih dibutuhkan\n"
                "   dengan format: `[PERLU DATA TAMBAHAN: path/ke/file.py]`\n"
            )

        user_msg = (
            f"**Permintaan inspeksi:**\n{user_input}\n\n"
            f"**Masalah yang dilaporkan:**\n{problem}"
            f"{data_needed_instruction}\n\n"
            f"---\n\n"
            f"**Hasil pengumpulan data dari repositori:**\n\n{evidence_text}"
        )

        t0 = time.monotonic()
        initial_report = await self._llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=INSPECTOR_TEMPERATURE,
            top_p=INSPECTOR_TOP_P,
        )
        logger.info(
            "Inspector: initial report generated in %.2fs (%d chars)",
            time.monotonic() - t0, len(initial_report),
        )

        verified_report = await self._verify_report(initial_report.strip(), evidence_text)
        return verified_report

    # ── Progressive Deepening helpers (Items 5, 6, 7) ─────────────────────────

    async def _get_investigation_hypothesis(
        self,
        problem: str,
        phase1_evidence: dict[str, str],
    ) -> dict:
        """
        Phase-1 LLM call: given lightweight evidence (dir_tree + git_log + grep),
        produce a JSON hypothesis with suspected files + additional keywords.

        Returns a dict with keys "hypothesis", "suspected_files",
        "additional_keywords" — or a default dict on failure.
        """
        evidence_text = self._build_evidence_text(phase1_evidence)
        # Cap phase-1 evidence to keep the hypothesis call cheap.
        _PHASE1_CAP = 8_000
        if len(evidence_text) > _PHASE1_CAP:
            evidence_text = evidence_text[:_PHASE1_CAP] + "\n... [dipotong]"

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _HYPOTHESIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"**Masalah:** {problem}\n\n"
                            f"**Data awal:**\n\n{evidence_text}"
                        ),
                    },
                ],
                temperature=0.10,
                top_p=0.90,
                max_tokens=256,
            )
            raw = response.strip()
            # Strip optional markdown fences.
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$",           "", raw, flags=re.MULTILINE)
            data = json.loads(raw)
            # Sanitise types.
            data.setdefault("hypothesis", "")
            data.setdefault("suspected_files", [])
            data.setdefault("additional_keywords", [])
            if not isinstance(data["suspected_files"], list):
                data["suspected_files"] = []
            if not isinstance(data["additional_keywords"], list):
                data["additional_keywords"] = []
            # Cap to maximum allowed files.
            data["suspected_files"] = data["suspected_files"][:_MAX_SUSPECTED_FILES]
            logger.info(
                "Inspector: hypothesis='%s'; suspected_files=%s; extra_keywords=%s",
                data["hypothesis"][:100],
                data["suspected_files"],
                data["additional_keywords"],
            )
            return data
        except Exception as exc:
            logger.warning("Inspector: hypothesis LLM call failed (%s); skipping phase 1", exc)
            return {"hypothesis": "", "suspected_files": [], "additional_keywords": []}

    async def _read_suspected_files(
        self,
        repo_path: Path,
        suspected_files: list[str],
    ) -> str:
        """
        Read the files requested by the hypothesis LLM, returning a formatted
        evidence string.  Gracefully skips files that do not exist or are
        unreadable.
        """
        from src.agents.repo_agent_base import MAX_FILE_BYTES
        snippets: list[str] = []
        for rel_path in suspected_files:
            abs_path = repo_path / rel_path
            if not abs_path.exists() or not abs_path.is_file():
                logger.debug("Inspector: suspected file not found: %s", rel_path)
                continue
            try:
                text = abs_path.read_text(errors="replace")[:MAX_FILE_BYTES]
                snippets.append(
                    f"### 📄 {rel_path} (suspected by hypothesis)\n```\n{text}\n```"
                )
            except OSError as exc:
                logger.debug("Inspector: could not read suspected file %s: %s", rel_path, exc)
        return "\n\n".join(snippets) if snippets else "(nenhum dos arquivos suspeitos encontrado)"

    async def _grep_additional_keywords(
        self,
        repo_path: Path,
        keywords: list[str],
    ) -> str:
        """Run an extra grep pass for *keywords* suggested by the hypothesis LLM."""
        if not keywords:
            return ""
        return await self._grep_keywords(repo_path, keywords[:5])

    async def _progressive_inspection(
        self,
        user_input:  str,
        problem:     str,
        repo_path:   Path,
        keywords:    list[str],
    ) -> str:
        """
        Iterative investigation with a token budget (Items 5, 6, 7).

        Strategy:
          Iteration 1 (phase 1) — lightweight evidence:
            dir_tree + git_log + grep_keywords → LLM hypothesis
          Iteration 2 (phase 2) — targeted evidence:
            phase-1 evidence + suspected files + refined grep + full evidence
          Iteration 3 (optional) — if report still contains [DATA TIDAK CUKUP]:
            read additional files mentioned by the LLM → produce final report

        The loop stops when:
          a) LLM report has no [DATA TIDAK CUKUP] signals, or
          b) MAX_INVESTIGATION_PHASES is reached, or
          c) No new files are identified.
        """
        phases_used = 0

        # ── Iteration 1: Lightweight evidence → hypothesis ─────────────────
        dir_tree, git_log, grep_result, grep_errors = await asyncio.gather(
            self._get_dir_tree(repo_path),
            self._get_git_log(repo_path),
            self._grep_keywords(repo_path, keywords),
            self._grep_error_patterns(repo_path),
            return_exceptions=True,
        )

        def _safe(r: object, fallback: str) -> str:
            return str(r) if not isinstance(r, Exception) else fallback

        phase1_evidence: dict[str, str] = {
            "Struktur Direktori":   _safe(dir_tree,     "(no dir tree)"),
            "Git Log (terbaru)":    _safe(git_log,      "(no git log)"),
            "Grep Keyword Masalah": _safe(grep_result,  "(no grep result)"),
            "Grep Pola Error Umum": _safe(grep_errors,  "(no error patterns)"),
        }

        hypothesis_data = await self._get_investigation_hypothesis(problem, phase1_evidence)
        phases_used += 1

        # ── Iteration 2: Full evidence + suspected files ───────────────────
        git_diff, key_files, error_logs, relevant_files = await asyncio.gather(
            self._get_git_diff(repo_path),
            self._read_key_files(repo_path),
            self._find_error_logs(repo_path),
            self._read_relevant_files(repo_path, problem),
            return_exceptions=True,
        )

        suspected_files_text    = await self._read_suspected_files(
            repo_path, hypothesis_data["suspected_files"]
        )
        extra_grep_text = await self._grep_additional_keywords(
            repo_path, hypothesis_data["additional_keywords"]
        )

        evidence: dict[str, str] = {
            **phase1_evidence,
            "Git Diff (terakhir)":          _safe(git_diff,        "(no diff)"),
            "File Kunci":                   _safe(key_files,       "(no key files)"),
            "Log & Error Files":            _safe(error_logs,      "(no log files)"),
            "File Relevan (RAG/TF-IDF)":    _safe(relevant_files,  "(no RAG results)"),
        }
        if suspected_files_text.strip():
            evidence["File Dicurigai (Hypothesis)"] = suspected_files_text
        if extra_grep_text.strip():
            evidence["Grep Tambahan (Hypothesis)"] = extra_grep_text
        if hypothesis_data["hypothesis"]:
            evidence["Hipotesis Awal"] = hypothesis_data["hypothesis"]

        report = await self._run_inspection_llm(user_input, problem, evidence)
        # Counts as 1 investigation phase. _run_inspection_llm internally uses
        # 2 LLM sub-calls (initial report + critic verification pass), but MAX_INVESTIGATION_PHASES
        # tracks investigation phases, not individual sub-calls.
        phases_used += 1

        # ── Iteration 3: Handle [DATA TIDAK CUKUP] signal (Item 7) ────────
        if phases_used < MAX_INVESTIGATION_PHASES and _DATA_NEEDED_RE.search(report):
            logger.info(
                "Inspector: [DATA TIDAK CUKUP] detected in report — running iteration 3"
            )
            # Extract file paths mentioned after the signal.
            extra_paths: list[str] = re.findall(
                r"\[PERLU DATA TAMBAHAN:\s*([^\]]+)\]",
                report,
                re.IGNORECASE,
            )
            extra_paths = [p.strip() for p in extra_paths if p.strip()][:_MAX_SUSPECTED_FILES]

            if extra_paths:
                extra_files_text = await self._read_suspected_files(repo_path, extra_paths)
                if extra_files_text.strip():
                    evidence["File Tambahan (Iterasi 3)"] = extra_files_text
                    logger.info(
                        "Inspector: iteration 3 — added %d extra file(s): %s",
                        len(extra_paths), extra_paths,
                    )
                    # Run a final targeted LLM pass with the augmented evidence.
                    report = await self._run_inspection_llm(user_input, problem, evidence)
                    phases_used += 1  # iteration 3 = phase 3

        logger.info(
            "Inspector: progressive inspection complete — %d LLM call(s) used (max=%d)",
            phases_used, MAX_INVESTIGATION_PHASES,
        )
        return report

    # ── Inspection task ────────────────────────────────────────────────────────

    async def _run_inspection_task(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       RepoExtractionRequest,
    ) -> AgentTask:
        """
        Execute the full inspection using Progressive Deepening (Items 5, 6, 7).

        Replaces the single-shot evidence-gather approach with an iterative
        investigation that:
          1. Forms a hypothesis from lightweight evidence (dir_tree + git_log).
          2. Opens only the files the hypothesis identifies as suspicious.
          3. Re-runs if the LLM signals [DATA TIDAK CUKUP] (Item 7).
          4. Enforces a MAX_INVESTIGATION_PHASES budget (Item 6).
        """
        try:
            logger.info(
                "Inspector: starting progressive inspection at %s (branch=%s)",
                repo_path, req.branch,
            )
            t_start = time.monotonic()

            report = await self._progressive_inspection(
                user_input=req.problem or task.user_input,
                problem=req.problem or task.user_input,
                repo_path=repo_path,
                keywords=req.keywords,
            )

            t_total = time.monotonic() - t_start
            logger.info("Inspector: progressive inspection complete in %.2fs", t_total)

            branch_header = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            perf_footer = (
                f"\n\n---\n"
                f"⏱️ *Inspeksi selesai dalam {t_total:.1f}s (progressive deepening, max {MAX_INVESTIGATION_PHASES} iterasi)*"
            )
            task.mark_done(branch_header + report + perf_footer)

            # Persist context so follow-up questions inherit repo + branch.
            self._save_session_context(
                task.session_id,
                req.repo_url,
                req.branch,
            )

        except Exception as exc:
            logger.exception("Inspector._run_inspection_task: error: %s", exc)
            task.mark_failed(f"❌ Inspeksi gagal pada branch `{req.branch}`: {exc}")

        return task

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            logger.info("DeveloperInspectorAgent: starting for session=%s", task.session_id)

            # ── Check for pending branch confirmation ──────────────────────
            pending = _inspector_pending_confirmations.get(task.session_id)
            if pending:
                branch_choice = _resolve_branch_from_reply(
                    task.user_input, pending["detected_branch"]
                )
                if branch_choice is not None:
                    del _inspector_pending_confirmations[task.session_id]
                    repo_path = Path(pending["repo_path"])
                    await self._checkout_branch(repo_path, branch_choice)
                    req = RepoExtractionRequest(
                        repo_url=pending["repo_url"],
                        problem=pending["problem"],
                        keywords=pending["keywords"],
                        branch=branch_choice,
                    )
                    # Persist confirmed context so subsequent follow-ups inherit it.
                    self._save_session_context(
                        task.session_id,
                        req.repo_url,
                        branch_choice,
                    )
                    return await self._run_inspection_task(task, repo_path, req)
                # Not a recognizable confirmation – fall through to normal parse.

            # ── Step 1: Extract structured request via LLM ─────────────────
            req = await self._extract_request(task.user_input, session_id=task.session_id)

            logger.info(
                "Inspector: repo_url=%r problem=%r keywords=%s branch=%r",
                req.repo_url, req.problem, req.keywords, req.branch,
            )

            # ── Step 2: Resolve local repo path ───────────────────────────
            repo_path = await self._resolve_repo(req.repo_url)

            if repo_path is None:
                logger.info("Inspector: no local repo – description-only analysis.")
                warning = (
                    "\n\n> ⚠️ **Catatan:** Tidak ada repositori yang dapat diakses "
                    "(URL tidak diberikan atau tidak ada repo yang sebelumnya di-clone). "
                    "Analisis ini didasarkan pada deskripsi pengguna saja.\n"
                )
                evidence = {"Deskripsi dari Pengguna": task.user_input}
                report   = await self._run_inspection_llm(
                    task.user_input, req.problem or task.user_input, evidence,
                )
                task.mark_done(report + warning)
                return task

            # ── Step 3: Branch selection ───────────────────────────────────
            if req.branch:
                await self._checkout_branch(repo_path, req.branch)
                return await self._run_inspection_task(task, repo_path, req)

            # No branch specified → detect and ask for confirmation.
            detected_branch = await self._get_current_branch(repo_path)
            _inspector_pending_confirmations[task.session_id] = {
                "repo_url":        req.repo_url,
                "repo_path":       str(repo_path),
                "problem":         req.problem,
                "keywords":        req.keywords,
                "detected_branch": detected_branch,
            }
            task.mark_done(
                f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                f"Repository berhasil diakses. Branch aktif saat ini: **`{detected_branch}`**\n\n"
                f"Akan dijalankan Inspeksi Penuh pada branch **`{detected_branch}`**.\n\n"
                f"Balas **`lanjutkan`** untuk melanjutkan, "
                f"atau ketik nama branch yang diinginkan "
                f"(contoh: `develop`, `feature/my-feature`)."
            )
            return task

        except Exception as exc:
            logger.exception("DeveloperInspectorAgent: unexpected error: %s", exc)
            task.mark_failed(
                f"❌ Gagal karena error tidak terduga: {exc}\n\n"
                "Mohon periksa log untuk detail lebih lanjut."
            )

        return task


# ── Private gather helper ──────────────────────────────────────────────────────

async def _gather_evidence(
    agent:     "DeveloperInspectorAgent",
    repo_path: Path,
    keywords:  list[str],
    problem:   str = "",
) -> tuple[str, str, str, str, str, str, str, str]:
    """
    Run all read-only inspection commands concurrently, then run RAG sequentially.

    Returns an 8-tuple:
        (dir_tree, git_log, git_diff, grep_keywords, grep_errors,
         key_files, error_logs, relevant_files)
    """
    parallel_results = await asyncio.gather(
        agent._get_dir_tree(repo_path),
        agent._get_git_log(repo_path),
        agent._get_git_diff(repo_path),
        agent._grep_keywords(repo_path, keywords),
        agent._grep_error_patterns(repo_path),
        agent._read_key_files(repo_path),
        agent._find_error_logs(repo_path),
        return_exceptions=True,
    )

    def _safe(r: object, fallback: str) -> str:
        if isinstance(r, Exception):
            logger.warning("Inspector evidence gather error: %s", r)
            return f"(error: {r})"
        return str(r) if r else fallback

    # RAG step runs after parallel commands (needs repo path to be intact).
    relevant_files = await agent._read_relevant_files(repo_path, problem)

    return (
        _safe(parallel_results[0], "(no dir tree)"),
        _safe(parallel_results[1], "(no git log)"),
        _safe(parallel_results[2], "(no diff)"),
        _safe(parallel_results[3], "(no grep result)"),
        _safe(parallel_results[4], "(no error patterns)"),
        _safe(parallel_results[5], "(no key files)"),
        _safe(parallel_results[6], "(no log files)"),
        relevant_files,
    )
