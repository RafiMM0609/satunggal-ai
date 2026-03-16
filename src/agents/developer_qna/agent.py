"""
DeveloperQnAAgent – Repository Q&A (Code Understanding) Agent.

Peran:
  Asisten yang membantu memahami isi repositori secara mendalam melalui
  tanya-jawab topik-spesifik. Bukan untuk mencari bug — melainkan untuk
  menjawab pertanyaan seperti:
    - "Ada API apa saja di repo ini?"
    - "Tech stack apa yang dipakai?"
    - "Bagaimana alur utama aplikasi?"
    - "Model data apa yang ada?"
    - "Dependency apa yang digunakan?"
    - "Bagaimana CI/CD-nya dikonfigurasi?"
    - "Apa itu fungsi X di repo ini?"

Perbedaan dengan DeveloperInspectorAgent:
  - Inspector  → menemukan BUG, root cause analysis, laporan inspeksi penuh + critic pass
  - QnA Agent  → menjawab pertanyaan tentang ISI repo: API, tech stack, model, dll.

Workflow:
  1. Ekstrak repo_url + pertanyaan dari user via LLM (shared base).
  2. Clone / pull repo jika belum ada (shared base).
  3. Detect branch aktif → konfirmasi dengan user jika tidak disebutkan.
  4. Klasifikasi Q/A sub-intent via regex (classify_intent dari repo_qa).
  5. Jalankan extractor topik-spesifik + RAG secara paralel.
  6. Kirim hasil ekstraksi ke LLM untuk jawaban langsung dan ringkas.
  7. Kembalikan jawaban dengan sumber file:baris dan label [CONFIRMED/LIKELY/UNVERIFIED].

Batasan penting:
  - READ-ONLY: tidak ada git add/commit/push.
  - Tidak ada eksekusi kode.
  - Tidak menghasilkan laporan inspeksi penuh — hanya menjawab pertanyaan spesifik.
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
from src.tools.repo_qa import (
    QAIntent,
    classify_intent,
    extract_specific_target,
    run_qa_extraction,
)

logger = logging.getLogger(__name__)

# ── LLM sampling parameters ────────────────────────────────────────────────────

QNA_TEMPERATURE = 0.15
QNA_TOP_P       = 0.90
QNA_MAX_TOKENS  = 16384   # increased from global default to allow richer answers

# ── Q/A Intent display labels ──────────────────────────────────────────────────

_QA_INTENT_LABELS: dict[QAIntent, str] = {
    QAIntent.API_ENDPOINTS:   "📡 API Endpoints",
    QAIntent.TECH_STACK:      "🛠️ Tech Stack",
    QAIntent.DATA_MODELS:     "🗃️ Data Models",
    QAIntent.DEPENDENCIES:    "📦 Dependencies",
    QAIntent.CI_CD:           "🚀 CI/CD",
    QAIntent.SECURITY:        "🔐 Security",
    QAIntent.MAIN_FLOW:       "🔄 Main Flow",
    QAIntent.SPECIFIC_SYMBOL: "🔍 Symbol Q/A",
    QAIntent.FULL_INSPECTION: "💬 General Q/A",
}

# ── Q/A System Prompt ──────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
Kamu adalah **Asisten Analisis Repositori** yang menjawab pertanyaan langsung
tentang sebuah codebase berdasarkan data yang diekstrak dari repositori.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ATURAN KRITIS – ANTI-HALUSINASI & ANTI-CODE-DUMP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Jawab HANYA berdasarkan data repositori yang diberikan.
2. Setiap poin jawaban HARUS disertai sumber: nama file + nomor baris jika ada.
3. Gunakan label:
   - 🟢 **[CONFIRMED]** – ditemukan langsung dalam kode.
   - 🟡 **[LIKELY]** – dapat disimpulkan dari konteks.
   - 🔴 **[UNVERIFIED]** – tidak ada bukti langsung, tulis ini jika harus menduga.
4. Jika data tidak cukup, tulis **[DATA TIDAK CUKUP]** dan jelaskan apa yang perlu diperiksa.
5. DILARANG mengarang detail yang tidak ada dalam data.
6. **DILARANG menampilkan ulang kode sumber secara verbatim sebagai jawaban.**
   Kode dalam "Data dari repositori" adalah REFERENSI untuk kamu analisis.
   Gunakan kode tersebut untuk MENJELASKAN logika dalam bahasa natural (prosa).
   Kutip hanya potongan kode yang sangat spesifik sebagai bukti (maksimal 5 baris per kutipan).
   Jika pertanyaan tentang "cara kerja" / "logika bisnis": jawab step-by-step
   dalam bahasa natural — BUKAN dengan menempel ulang seluruh function body.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Panduan untuk pertanyaan tentang API / endpoint spesifik

Jika data mengandung **Route Registration** dan **Handler Implementation**,
lakukan analisis end-to-end:

1. **Lokasi Route** – di file apa, HTTP method apa, path exact-nya apa,
   middleware apa yang dipasang sebelum handler.
2. **Handler Function** – nama fungsi, file, nomor baris. Jelaskan parameter
   yang diterima (path params, query params, request body).
3. **Logika Utama** – apa yang dilakukan handler: validasi, akses DB/service,
   transformasi data, dll. Kutip kode yang relevan.
4. **Response** – apa yang dikembalikan: format data, HTTP status code, file
   (jika download), dsb.
5. **Fungsi/Service yang Dipanggil** – sebutkan nama fungsi downstream yang
   dipanggil oleh handler dan perannya berdasarkan kode yang terlihat.

Format jawaban umum:

## 💬 Jawaban
<Jawaban langsung dan padat>

## 📋 Detail & Bukti
<Daftar poin dengan sumber file:baris dan kutipan kode>

## 🔄 Alur End-to-End (untuk pertanyaan API/route)
<Alur: request masuk → middleware → handler → service/DB → response>

## 🗺️ Lokasi di Repo
<Tabel ringkas: nama/path | file | baris | status>

## 💡 Catatan Tambahan
<Hal-hal penting yang relevan dengan pertanyaan>

**Format:**
- Gunakan bahasa yang sama dengan pertanyaan pengguna.
- Singkat dan faktual, tidak perlu struktur laporan inspeksi penuh.
- Jangan tampilkan template kosong jika tidak ada konten.
"""

# ── Branch confirmation state ─────────────────────────────────────────────────

_qna_pending_confirmations: dict[str, dict] = {}

_CONFIRMATION_ANSWERS = {
    "ya", "yes", "ok", "lanjutkan", "continue", "iya",
    "proceed", "y", "yep", "sure", "lanjut",
}

# Regex to detect explanation-style questions ("cara kerja", "logika bisnis", etc.)
# Compiled once at module level for efficiency.
_EXPLANATION_Q_RE = re.compile(
    r"cara\s*kerja|bagaimana\s*bekerja|logika\s*bisnis|business\s*logic"
    r"|alur\s*(?:api|endpoint|handler|bisnis|request)"
    r"|jelaskan\s*(?:alur|logika|flow|cara)"
    r"|detailkan\s*(?:logika|alur|flow|cara)"
    r"|how\s*(?:does|it\s*work)",
    re.IGNORECASE,
)


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

class DeveloperQnAAgent(RepoAgentBase):
    """
    Repository Q/A agent for understanding codebase content.

    Answers topic-specific questions about a repository:
    APIs, tech stack, data models, dependencies, CI/CD, security,
    main flow, or specific symbols.

    For bug hunting, root cause analysis, or full code inspection,
    use DeveloperInspectorAgent (intent: code_inspection).
    """

    name = "developer_qna"

    def __init__(
        self,
        llm: LLMClient | None = None,
        history=None,
    ) -> None:
        super().__init__(llm=llm, history=history)

    # ── Q/A flow ───────────────────────────────────────────────────────────────

    async def _run_qa_flow(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       RepoExtractionRequest,
        intent:    QAIntent,
    ) -> AgentTask:
        """
        Run topic-specific extraction + RAG, then answer via LLM.

        Flow:
          1. Run topic extractor + RAG + Tavily + dir tree concurrently.
          2. Build LLM prompt from aggregated evidence.
          3. Return direct factual answer with [CONFIRMED/LIKELY/UNVERIFIED] labels.
        """
        try:
            logger.info(
                "QnA: intent=%s repo=%s problem=%r",
                intent.value, repo_path, req.problem,
            )
            t_start = time.monotonic()

            # ── Resolve symbol target for SPECIFIC_SYMBOL intent ──────────
            # If the user's follow-up question doesn't include an explicit
            # API path or function name (e.g. "bisa detailkan logika bisnis
            # di api ini"), inherit the target from the previous turn so the
            # route tracer can re-run against the same handler.
            symbol_target = ""
            if intent == QAIntent.SPECIFIC_SYMBOL:
                symbol_target = extract_specific_target(req.problem or task.user_input)
                if not symbol_target:
                    ctx = self._get_session_context(task.session_id)
                    inherited = ctx.get("last_symbol_target", "")
                    if inherited:
                        symbol_target = inherited
                        logger.info(
                            "QnA: follow-up – inherited symbol_target=%r from session",
                            symbol_target,
                        )

            # Run topic extractor + RAG + optional Tavily concurrently
            # Detect explanation-style questions BEFORE gather so we can skip
            # unnecessary data (RAG dumps, dir-tree) for SPECIFIC_SYMBOL intent.
            is_explanation_q = _EXPLANATION_Q_RE.search(task.user_input) is not None

            qa_evidence, rag_files, tavily_ctx, dir_tree = await asyncio.gather(
                run_qa_extraction(
                    repo_path,
                    intent,
                    req.problem or task.user_input,
                    candidate_route_filenames=req.candidate_route_filenames,
                    symbol_target=symbol_target,
                ),
                self._read_relevant_files(repo_path, req.problem or task.user_input),
                self._fetch_tavily_context(req.problem or task.user_input),
                self._get_dir_tree(repo_path),
                return_exceptions=True,
            )

            def _safe_str(r: object, fallback: str) -> str:
                return str(r) if not isinstance(r, Exception) else fallback

            evidence: dict[str, str] = {}

            # Primary: topic-specific extraction
            if isinstance(qa_evidence, dict):
                evidence.update(qa_evidence)
            else:
                evidence["Extraction Error"] = str(qa_evidence)

            # For explanation questions about a specific symbol, skip RAG file
            # dumps and dir-tree — they add noise (often 10+ full Go files) that
            # tempts the LLM to reproduce code rather than explain logic.
            if not (is_explanation_q and intent == QAIntent.SPECIFIC_SYMBOL):
                # Secondary: RAG-relevant files
                rag_text = _safe_str(rag_files, "(RAG unavailable)")
                if rag_text.strip() and "unavailable" not in rag_text and "error" not in rag_text.lower():
                    evidence["📂 File Relevan (RAG)"] = rag_text

                # Tertiary: directory tree for structural context
                tree = _safe_str(dir_tree, "")
                if tree.strip():
                    evidence["🗂️ Struktur Direktori"] = tree

            # Optional: Tavily web search context
            tavily_text = _safe_str(tavily_ctx, "")
            if tavily_text.strip() and "hasil pencarian" in tavily_text.lower():
                evidence["🔎 Pencarian Web (Tavily)"] = tavily_text

            t_extract = time.monotonic()
            logger.info("QnA: extraction done in %.2fs", t_extract - t_start)

            evidence_text = self._build_evidence_text(evidence)

            if is_explanation_q:
                # Cap evidence size so the LLM focuses on explaining, not copying.
                # 8000 chars covers ~80 lines of code comfortably.
                _MAX_EVIDENCE_CHARS = 8_000
                if len(evidence_text) > _MAX_EVIDENCE_CHARS:
                    evidence_text = (
                        evidence_text[:_MAX_EVIDENCE_CHARS]
                        + f"\n\n... [evidence dipotong pada {_MAX_EVIDENCE_CHARS} karakter "
                        "untuk fokus pada analisis logika]"
                    )
                # Directive placed FIRST so the LLM reads it before the evidence.
                explanation_preamble = (
                    "**🔴 INSTRUKSI WAJIB (baca sebelum menjawab):**\n"
                    "Pengguna ingin memahami CARA KERJA / LOGIKA BISNIS API, "
                    "BUKAN melihat kode sumbernya.\n"
                    "- Jawab dalam bahasa natural (bahasa Indonesia), step-by-step.\n"
                    "- JANGAN salin atau tampilkan ulang kode sumber.\n"
                    "- Gunakan kode di bagian 'Data dari repositori' HANYA sebagai "
                    "referensi untuk menjelaskan ALUR LOGIKA.\n"
                    "- Format jawaban: nomor langkah + penjelasan prose + sumber file:baris.\n"
                    "- Jika ada bagian yang di-truncate, simpulkan berdasarkan pola "
                    "yang terlihat.\n"
                    "---\n\n"
                )
                verbosity_note = "Jelaskan secara LENGKAP tapi dalam prosa bahasa natural, bukan kode."
                user_msg = (
                    f"{explanation_preamble}"
                    f"**Pertanyaan pengguna:**\n{task.user_input}\n\n"
                    f"**Panduan verbositas:** {verbosity_note}\n\n"
                    f"---\n\n"
                    f"**Data dari repositori (REFERENSI SAJA — jangan salin ulang):**\n\n{evidence_text}"
                )
            else:
                verbosity_note = (
                    "Jawab secara SINGKAT dan padat (maksimal 10 poin)."
                    if req.verbosity == "concise"
                    else "Jawab secara LENGKAP dengan penjelasan step-by-step."
                )
                user_msg = (
                    f"**Pertanyaan pengguna:**\n{task.user_input}\n\n"
                    f"**Panduan verbositas:** {verbosity_note}\n\n"
                    f"---\n\n"
                    f"**Data dari repositori:**\n\n{evidence_text}"
                )

            qa_response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _QA_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=QNA_TEMPERATURE,
                top_p=QNA_TOP_P,
                max_tokens=QNA_MAX_TOKENS,
            )

            t_total = time.monotonic() - t_start
            logger.info("QnA: done in %.2fs total", t_total)

            branch_note  = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            intent_badge = _QA_INTENT_LABELS.get(intent, "💬 Q/A")
            perf_footer  = (
                f"\n\n---\n"
                f"⏱️ *{intent_badge} · {t_total:.1f}s "
                f"(ekstraksi: {t_extract - t_start:.1f}s)*"
            )
            task.mark_done(branch_note + qa_response.strip() + perf_footer)

            # Persist context so follow-up questions inherit repo + branch.
            self._save_session_context(
                task.session_id,
                req.repo_url,
                req.branch,
                req.candidate_route_filenames or None,
                last_symbol_target=symbol_target if intent == QAIntent.SPECIFIC_SYMBOL else "",
            )

        except Exception as exc:
            logger.exception("QnA flow error: %s", exc)
            task.mark_failed(f"❌ Q/A gagal: {exc}")

        return task

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            logger.info("DeveloperQnAAgent: starting for session=%s", task.session_id)

            # ── Check for pending branch confirmation ──────────────────────
            pending = _qna_pending_confirmations.get(task.session_id)
            if pending:
                branch_choice = _resolve_branch_from_reply(
                    task.user_input, pending["detected_branch"]
                )
                if branch_choice is not None:
                    del _qna_pending_confirmations[task.session_id]
                    repo_path = Path(pending["repo_path"])
                    await self._checkout_branch(repo_path, branch_choice)
                    intent = QAIntent(pending["qa_intent"])
                    req = RepoExtractionRequest(
                        repo_url=pending["repo_url"],
                        problem=pending["problem"],
                        branch=branch_choice,
                        verbosity=pending.get("verbosity", "detailed"),
                        candidate_route_filenames=pending.get("candidate_route_filenames", []),
                    )
                    # Persist confirmed context so subsequent follow-ups inherit it.
                    self._save_session_context(
                        task.session_id,
                        req.repo_url,
                        branch_choice,
                        req.candidate_route_filenames or None,
                    )
                    return await self._run_qa_flow(task, repo_path, req, intent)
                # Not a recognizable confirmation – fall through to normal parse.

            # ── Step 1: Classify Q/A sub-intent (fast, regex-based) ────────
            intent = classify_intent(task.user_input)
            logger.info("QnA: classified intent=%s", intent.value)

            # ── Step 2: Extract structured request via LLM ─────────────────
            req = await self._extract_request(task.user_input, session_id=task.session_id)

            logger.info(
                "QnA: repo_url=%r problem=%r branch=%r intent=%s",
                req.repo_url, req.problem, req.branch, intent.value,
            )

            # ── Step 3: Resolve local repo path ───────────────────────────
            repo_path = await self._resolve_repo(req.repo_url)

            if repo_path is None:
                logger.info("QnA: no local repo – answering from description only.")
                warning = (
                    "\n\n> ⚠️ **Catatan:** Tidak ada repositori yang dapat diakses "
                    "(URL tidak diberikan atau tidak ada repo yang sebelumnya di-clone). "
                    "Jawaban ini didasarkan pada deskripsi pengguna saja.\n"
                )
                evidence_text = f"## Deskripsi dari Pengguna\n{task.user_input}"
                verbosity_note = (
                    "Jawab secara SINGKAT dan padat."
                    if req.verbosity == "concise"
                    else "Jawab secara LENGKAP dengan detail."
                )
                user_msg = (
                    f"**Pertanyaan pengguna:**\n{task.user_input}\n\n"
                    f"**Panduan verbositas:** {verbosity_note}\n\n"
                    f"---\n\n{evidence_text}"
                )
                answer = await self._llm.chat(
                    messages=[
                        {"role": "system", "content": _QA_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=QNA_TEMPERATURE,
                    top_p=QNA_TOP_P,
                    max_tokens=QNA_MAX_TOKENS,
                )
                task.mark_done(answer.strip() + warning)
                return task

            # ── Step 4: Branch selection ───────────────────────────────────
            if req.branch:
                await self._checkout_branch(repo_path, req.branch)
                return await self._run_qa_flow(task, repo_path, req, intent)

            # No branch specified → detect and ask for confirmation.
            detected_branch = await self._get_current_branch(repo_path)
            intent_badge    = _QA_INTENT_LABELS.get(intent, "💬 Q/A")
            _qna_pending_confirmations[task.session_id] = {
                "repo_url":                 req.repo_url,
                "repo_path":                str(repo_path),
                "problem":                  req.problem,
                "detected_branch":          detected_branch,
                "qa_intent":                intent.value,
                "verbosity":                req.verbosity,
                "candidate_route_filenames": req.candidate_route_filenames,
            }
            task.mark_done(
                f"⚠️ **Branch tidak ditentukan dalam permintaan.**\n\n"
                f"Repository berhasil diakses. Branch aktif saat ini: **`{detected_branch}`**\n\n"
                f"Akan dijawab dalam mode **{intent_badge}** pada branch **`{detected_branch}`**.\n\n"
                f"Balas **`lanjutkan`** untuk melanjutkan, "
                f"atau ketik nama branch yang diinginkan "
                f"(contoh: `develop`, `feature/my-feature`)."
            )
            return task

        except Exception as exc:
            logger.exception("DeveloperQnAAgent: unexpected error: %s", exc)
            task.mark_failed(
                f"❌ Gagal karena error tidak terduga: {exc}\n\n"
                "Mohon periksa log untuk detail lebih lanjut."
            )

        return task
