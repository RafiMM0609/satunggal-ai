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

# ── Q/A mode LLM prompt ───────────────────────────────────────────────────────

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
            evidence=evidence_text[:60_000],
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
        Phase 1 – Generate initial report.
        Phase 2 – Critic pass: verify every finding against raw evidence.
        """
        evidence_text = self._build_evidence_text(evidence)

        for title, content in evidence.items():
            logger.debug("Inspector evidence '%s': %d chars", title, len(content))
        logger.info(
            "Inspector: sending %d evidence sections (%d total chars) to LLM",
            len(evidence), len(evidence_text),
        )

        user_msg = (
            f"**Permintaan inspeksi:**\n{user_input}\n\n"
            f"**Masalah yang dilaporkan:**\n{problem}\n\n"
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

    # ── Inspection task ────────────────────────────────────────────────────────

    async def _run_inspection_task(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       RepoExtractionRequest,
    ) -> AgentTask:
        """
        Execute the full inspection after branch has been confirmed and checked out.
        """
        try:
            logger.info(
                "Inspector: inspecting repo at %s (branch=%s)",
                repo_path, req.branch,
            )
            t_start = time.monotonic()

            (
                dir_tree,
                git_log,
                git_diff,
                grep_result,
                grep_errors,
                key_files,
                error_logs,
                relevant_files,
            ) = await _gather_evidence(self, repo_path, req.keywords, req.problem)

            t_gather = time.monotonic()
            logger.info("Inspector: evidence gathered in %.2fs", t_gather - t_start)

            evidence = {
                "Struktur Direktori":           dir_tree,
                "Git Log (terbaru)":            git_log,
                "Git Diff (terakhir)":          git_diff,
                "Grep Keyword Masalah":         grep_result,
                "Grep Pola Error Umum":         grep_errors,
                "File Kunci":                   key_files,
                "Log & Error Files":            error_logs,
                "File Relevan (RAG/TF-IDF)":    relevant_files,
            }

            report = await self._run_inspection_llm(
                req.problem or task.user_input,
                req.problem or task.user_input,
                evidence,
            )

            t_total = time.monotonic() - t_start
            logger.info(
                "Inspector: inspection complete in %.2fs total "
                "(gather=%.2fs, llm=%.2fs)",
                t_total,
                t_gather - t_start,
                t_total - (t_gather - t_start),
            )

            branch_header = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            perf_footer = (
                f"\n\n---\n"
                f"⏱️ *Inspeksi selesai dalam {t_total:.1f}s "
                f"(pengumpulan data: {t_gather - t_start:.1f}s)*"
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
