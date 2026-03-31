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
import json
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

# ── LLM-based intent fallback prompt ──────────────────────────────────────────
# Used only when regex classify_intent() returns FULL_INSPECTION (catch-all),
# to accurately classify genuinely ambiguous natural-language questions.

_LLM_INTENT_CLASSIFY_SYSTEM = """\
Kamu adalah intent classifier untuk agen Q&A repositori kode.
Klasifikasikan pertanyaan pengguna ke dalam SATU intent berikut:

api_endpoints   – tentang route HTTP, REST API, endpoint, daftar URL
tech_stack      – tentang teknologi, framework, bahasa, library yang dipakai
data_models     – tentang skema database, ORM model, struktur data
dependencies    – tentang package, library, requirements, modul, dependensi
ci_cd           – tentang CI/CD, deployment, Docker, Dockerfile, pipeline
security        – tentang autentikasi, otorisasi, keamanan, permission, JWT
main_flow       – tentang arsitektur, alur utama sistem, cara kerja secara umum
specific_symbol – tentang file spesifik, fungsi, class, method, handler, atau fitur tertentu
full_inspection – laporan bug, error, crash, atau masalah yang perlu dianalisis

Balas HANYA dengan nama intent (satu kata tanpa spasi), contoh: api_endpoints\
"""

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


# ── Comparison question detection ──────────────────────────────────────────────
# Matches questions asking to compare/contrast two targets. Strong signals only.
_COMPARISON_RE = re.compile(
    r"\bkomparasi\b"                         # "komparasi X dan Y"
    r"|\bbandingkan\b"                        # "bandingkan X dengan Y"
    r"|\bperbandingan\b"                      # "perbandingan antara X dan Y"
    r"|\bapa\s+beda(?:nya)?\b"               # "apa bedanya X dan Y"
    r"|\bbedanya\s"                           # "bedanya X dan Y"
    r"|\bperbedaan\s+(?:antara\s+)?"         # "perbedaan (antara) X dan Y"
    r"|\bcompare\b"                           # "compare X and Y"
    r"|\bversus\b"                            # "X versus Y"
    r"|\bvs\.?\s"                             # "X vs Y" / "X vs. Y"
    r"|\bdibandingkan\s+dengan\b",           # "X dibandingkan dengan Y"
    re.IGNORECASE,
)

# ── LLM prompt: task decomposition for comparison questions ───────────────────
_TASK_DECOMPOSE_SYSTEM = """\
Kamu adalah task planner untuk agen analisis repositori kode.
User mengajukan pertanyaan perbandingan/komparasi tentang codebase.
Pecah pertanyaan tersebut menjadi daftar sub-task terstruktur dalam format JSON.

Format output WAJIB (JSON valid, tanpa teks lain):
{
  "aspect": "<topik yang dibandingkan, mis: 'implementasi auth', 'elastic apm'>",
  "tasks": [
    {"id": 1, "label": "<nama singkat target A>", "scope": "<dir/ atau kosong>", "query": "<pertanyaan spesifik untuk target A>"},
    {"id": 2, "label": "<nama singkat target B>", "scope": "<dir/ atau kosong>", "query": "<pertanyaan spesifik untuk target B>"},
    {"id": 3, "label": "komparasi", "scope": "", "query": "<permintaan perbandingan final>"}
  ]
}

Aturan:
- "scope": direktori relatif di repo (mis. "consumer/", "api/"). Kosongkan jika tidak diketahui.
- Buat TEPAT 2 task pencarian + 1 task komparasi (total 3 tasks).
- "label": singkat, nama komponen/direktori yang dibandingkan.
- "query": pertanyaan spesifik untuk setiap sub-task.

Contoh:
Input: "apa bedanya implementasi elastic apm di consumer/ dan api/"
Output:
{
  "aspect": "implementasi elastic apm",
  "tasks": [
    {"id": 1, "label": "consumer", "scope": "consumer/", "query": "implementasi elastic apm di consumer"},
    {"id": 2, "label": "api", "scope": "api/", "query": "implementasi elastic apm di api"},
    {"id": 3, "label": "komparasi", "scope": "", "query": "bandingkan implementasi elastic apm di consumer vs api"}
  ]
}

Contoh 2:
Input: "komparasi penerapan auth pada consumer dan producer"
Output:
{
  "aspect": "penerapan autentikasi (auth)",
  "tasks": [
    {"id": 1, "label": "consumer", "scope": "consumer/", "query": "penerapan auth di consumer"},
    {"id": 2, "label": "producer", "scope": "producer/", "query": "penerapan auth di producer"},
    {"id": 3, "label": "komparasi", "scope": "", "query": "bandingkan penerapan auth di consumer vs producer"}
  ]
}

Balas HANYA dengan JSON valid, tanpa penjelasan tambahan.\
"""

# ── LLM prompt: comparison answer generation ──────────────────────────────────
_COMPARE_ANSWER_SYSTEM = """\
Kamu adalah analis kode yang membuat perbandingan mendalam antara dua implementasi.
Buat jawaban komparasi terstruktur berdasarkan evidence yang diberikan dari repositori.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ ATURAN KRITIS – ANTI-HALUSINASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Hanya jawab berdasarkan evidence yang diberikan.
2. Setiap klaim HARUS disertai sumber: nama file + nomor baris.
3. Gunakan label: 🟢 [CONFIRMED] / 🟡 [LIKELY] / 🔴 [UNVERIFIED].
4. Jika data tidak cukup untuk salah satu target, tulis [DATA TIDAK CUKUP] dan
   jelaskan apa yang perlu diperiksa.
5. DILARANG mengarang detail yang tidak ada dalam evidence.
6. Gunakan bahasa yang sama dengan pertanyaan pengguna.

Format jawaban yang diharapkan:

## 🔍 Perbandingan: {aspect}

### 📦 {label_a}
<ringkasan implementasi A dengan sumber file:baris>

### 📦 {label_b}
<ringkasan implementasi B dengan sumber file:baris>

### ⚖️ Tabel Komparasi
| Aspek | {label_a} | {label_b} |
|-------|-----------|-----------|
| ...   | ...       | ...       |

### 💡 Kesimpulan
<poin utama perbedaan dan persamaan>\
"""

# ── Stopwords for keyword extraction ──────────────────────────────────────────
_KW_STOPWORDS = {
    # Indonesian
    "cari", "implementasi", "pada", "di", "dalam", "dan", "yang", "ada", "apa",
    "berikan", "tampilkan", "tunjukkan", "list", "daftar", "dari", "bagaimana",
    "cara", "kerja", "fungsi", "class", "modul", "module", "repo", "repositori",
    "kode", "semua", "setiap", "untuk", "ini", "itu", "bisa", "boleh", "tolong",
    "jelaskan", "lihat", "komparasi", "bandingkan", "perbandingan", "bedanya",
    "beda", "perbedaan", "antara", "dengan", "versus",
    # English
    "find", "search", "show", "list", "give", "get", "look", "check",
    "implementation", "of", "in", "at", "the", "a", "an", "is", "are",
    "on", "for", "from", "with", "how", "what", "where", "all", "compare", "vs",
}

# Directories to skip when scanning repository files (mirrors repo_qa._SKIP_DIRS).
_SCAN_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

# Number of source-code lines of context to show before/after each keyword hit.
_CONTEXT_LINES_AROUND_HIT = 4

# Prefixes that indicate an evidence string carries no useful information
# (returned by _scan_scope_for_topic when nothing was found).
_EMPTY_EVIDENCE_PREFIXES = (
    "(tidak ditemukan",
    "(gagal mengumpulkan",
    "(tidak ada keyword",
)


def _extract_query_keywords(query: str) -> list[str]:
    """
    Extract meaningful search keywords from *query*, excluding stopwords.

    Returns a list ordered longest-first (more specific phrases before single words).
    """
    text = re.sub(r"https?://\S+", "", query)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [
        w.lower() for w in text.split()
        if len(w) >= 2 and w.lower() not in _KW_STOPWORDS
    ]
    candidates: list[str] = []
    for i, w in enumerate(words):
        candidates.append(w)
        if i + 1 < len(words):
            candidates.append(f"{w} {words[i + 1]}")
    seen: set[str] = set()
    result: list[str] = []
    for c in sorted(candidates, key=len, reverse=True):
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


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

    # ── Intent classification (regex + LLM fallback) ───────────────────────────

    async def _classify_intent(self, user_input: str) -> QAIntent:
        """
        Classify Q/A sub-intent with maximum accuracy.

        Strategy:
          1. Fast regex-based classify_intent() — zero latency, deterministic.
          2. If regex returns FULL_INSPECTION (the catch-all default), make a
             single cheap LLM call (~20 output tokens) to accurately classify
             genuinely ambiguous or free-form natural-language questions.

        The LLM fallback adds ~0.5–1 s only for questions that the regex
        could not confidently classify, keeping the common path fast.
        """
        intent = classify_intent(user_input)
        if intent != QAIntent.FULL_INSPECTION:
            logger.debug("QnA intent: regex → %s", intent.value)
            return intent

        logger.debug("QnA intent: regex → FULL_INSPECTION, trying LLM fallback")
        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _LLM_INTENT_CLASSIFY_SYSTEM},
                    {"role": "user",   "content": user_input[:600]},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=20,
            )
            raw = response.strip().lower()
            # Guard against LLM adding extra text — extract first token only
            raw = re.split(r"[\s\n,.;:()\[\]]+", raw)[0]
            _intent_map: dict[str, QAIntent] = {v.value: v for v in QAIntent}
            fallback = _intent_map.get(raw)
            if fallback:
                logger.info(
                    "QnA intent: LLM fallback → %s (raw=%r)", fallback.value, raw
                )
                return fallback
            logger.warning(
                "QnA intent: LLM returned unrecognised token %r → FULL_INSPECTION", raw
            )
        except Exception as exc:
            logger.warning("QnA intent: LLM fallback failed (%s) → FULL_INSPECTION", exc)

        return QAIntent.FULL_INSPECTION

    # ── Comparison: task decomposition ────────────────────────────────────────

    async def _decompose_comparison_tasks(self, user_input: str) -> dict | None:
        """
        Use LLM to decompose a comparison question into structured sub-tasks.

        Returns a dict with:
          "aspect" – the topic being compared (string)
          "tasks"  – list of dicts: {id, label, scope, query}
                     Last task is always the comparison synthesis task.

        Returns None on any failure (LLM error, bad JSON, unexpected structure).
        Warnings are logged for all failure paths so nothing is silent.
        """
        logger.info("QnA comparison: calling LLM to decompose question=%r", user_input[:120])
        response = ""  # initialised here so it's reachable in the JSONDecodeError handler
        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _TASK_DECOMPOSE_SYSTEM},
                    {"role": "user",   "content": user_input[:800]},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )
            if not response.strip():
                logger.warning(
                    "QnA comparison: task decomposition returned EMPTY response "
                    "(possible max-token hit or model error). question=%r",
                    user_input[:120],
                )
                return None

            # Extract JSON block – LLM sometimes wraps in markdown code fences
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                logger.warning(
                    "QnA comparison: task decomposition returned non-JSON response: %r",
                    response[:300],
                )
                return None

            data = json.loads(json_match.group(0))

            if not isinstance(data.get("tasks"), list) or len(data["tasks"]) < 2:
                logger.warning(
                    "QnA comparison: unexpected task structure (need ≥2 tasks): %r", data
                )
                return None

            logger.info(
                "QnA comparison: decomposed into %d tasks, aspect=%r, labels=%r",
                len(data["tasks"]),
                data.get("aspect", ""),
                [t.get("label") for t in data["tasks"]],
            )
            return data

        except json.JSONDecodeError as exc:
            logger.warning(
                "QnA comparison: task decomposition produced invalid JSON: %s (response=%r)",
                exc, response[:300],
            )
            return None
        except Exception as exc:
            logger.warning("QnA comparison: task decomposition failed with error: %s", exc)
            return None

    # ── Comparison: evidence gathering for a single scope ─────────────────────

    async def _scan_scope_for_topic(
        self,
        repo_path: Path,
        scope: str,
        query: str,
        *,
        max_files: int = 8,
    ) -> str:
        """
        Search files within *scope* directory for topic keywords extracted from *query*.

        Args:
            repo_path: Root of the cloned repository.
            scope:     Relative directory path to restrict search (e.g. "consumer/").
                       Empty string or None means search the whole repo.
            query:     Natural-language description of what to look for; keywords are
                       extracted from this string.
            max_files: Maximum number of matching files to include in output.

        Returns markdown-formatted evidence string, or an informative message if
        nothing was found.  Warnings are logged for missing directories, unreadable
        files, and empty results.
        """
        # Resolve scope directory
        scope_clean = (scope or "").strip("/")
        if scope_clean:
            scope_path = repo_path / scope_clean
            if not scope_path.exists():
                logger.warning(
                    "QnA comparison: scope directory %r not found in repo %s – "
                    "will search full repo and add scope as keyword",
                    scope_clean, repo_path,
                )
                # Treat scope label as an extra keyword and search full repo
                scope_path = repo_path
                extra_keyword = scope_clean.lower()
            else:
                extra_keyword = ""
        else:
            scope_path = repo_path
            extra_keyword = ""

        # Extract keywords from query
        keywords = _extract_query_keywords(query)
        if extra_keyword and extra_keyword not in keywords:
            keywords = [extra_keyword] + keywords

        if not keywords:
            logger.warning(
                "QnA comparison: no keywords extracted from query=%r scope=%r",
                query, scope_clean,
            )
            return "(tidak ada keyword yang dapat diekstrak dari pertanyaan)"

        logger.info(
            "QnA comparison: scanning scope=%r for keywords=%r (top 5)",
            scope_clean or "(root)", keywords[:5],
        )

        source_exts = {
            ".py", ".go", ".js", ".ts", ".jsx", ".tsx", ".java", ".php",
            ".rb", ".rs", ".kt", ".cs",
            # Config/infra files – often contain APM/auth config
            ".yaml", ".yml", ".json", ".toml", ".env", ".ini", ".cfg",
        }
        findings: list[str] = []

        for fpath in sorted(scope_path.rglob("*")):
            if fpath.is_dir() or fpath.suffix.lower() not in source_exts:
                continue
            # Skip undesired directories (use module-level constant)
            rel_parts = fpath.relative_to(repo_path).parts
            if any(p in _SCAN_SKIP_DIRS for p in rel_parts[:-1]):
                continue

            try:
                text = fpath.read_text(errors="replace")
            except OSError as exc:
                logger.warning("QnA comparison: cannot read %s: %s", fpath, exc)
                continue

            file_lines = text.splitlines()
            hit_line_indices: set[int] = set()
            matched_kw = ""

            for kw in keywords[:10]:
                kw_lower = kw.lower()
                kw_compact = kw_lower.replace(" ", "")
                for i, line in enumerate(file_lines):
                    line_lower = line.lower()
                    if kw_lower in line_lower or (
                        kw_compact != kw_lower and kw_compact in line_lower
                    ):
                        hit_line_indices.add(i)
                if hit_line_indices:
                    matched_kw = kw
                    break  # stop at first keyword with matches

            if not hit_line_indices:
                continue

            rel = fpath.relative_to(repo_path).as_posix()
            # Expand hits with context lines (module-level constant)
            context_lines: list[str] = []
            for idx in sorted(hit_line_indices)[:15]:  # cap to 15 hits per file
                start = max(0, idx - _CONTEXT_LINES_AROUND_HIT)
                end = min(len(file_lines), idx + _CONTEXT_LINES_AROUND_HIT + 1)
                for j in range(start, end):
                    marker = "→" if j in hit_line_indices else " "
                    context_lines.append(f"  {marker} L{j + 1}: {file_lines[j].rstrip()}")

            findings.append(
                f"**`{rel}`** — keyword: `{matched_kw}` "
                f"({len(hit_line_indices)} hit):\n"
                + "\n".join(context_lines[:60])  # cap output lines per file
            )
            if len(findings) >= max_files:
                logger.debug(
                    "QnA comparison: reached max_files=%d for scope=%r", max_files, scope_clean
                )
                break

        if not findings:
            logger.warning(
                "QnA comparison: NO evidence found for scope=%r keywords=%r – "
                "evidence will be marked as insufficient",
                scope_clean or "(root)", keywords[:5],
            )
            return (
                f"(tidak ditemukan implementasi terkait di `{scope_clean or 'root'}` "
                f"untuk keyword: {', '.join(keywords[:3])})"
            )

        logger.info(
            "QnA comparison: found %d relevant file(s) in scope=%r",
            len(findings), scope_clean or "(root)",
        )
        return "\n\n".join(findings)

    # ── Comparison: orchestration ──────────────────────────────────────────────

    async def _run_comparison_flow(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       RepoExtractionRequest,
        intent:    QAIntent,
    ) -> AgentTask:
        """
        Handle comparison/contrast questions by decomposing into sub-tasks:

          Sub-task 1 – Gather evidence for target A (scoped directory scan).
          Sub-task 2 – Gather evidence for target B (scoped directory scan).
          Sub-task 3 – LLM generates structured comparison from A + B evidence.

        All sub-tasks are logged at INFO level so the full execution trace is
        visible.  Warnings are emitted for empty evidence, LLM empty responses,
        and evidence truncation.

        Falls back to the normal Q/A flow (with _skip_comparison=True) if task
        decomposition fails, so the user always gets an answer.
        """
        try:
            t_start = time.monotonic()
            logger.info(
                "QnA comparison flow: starting — session=%s problem=%r",
                task.session_id, req.problem,
            )

            # ── Step 1: Decompose question into tasks via LLM ────────────
            decomposed = await self._decompose_comparison_tasks(task.user_input)
            if not decomposed:
                logger.warning(
                    "QnA comparison: task decomposition failed – falling back to "
                    "normal Q/A flow for session=%s",
                    task.session_id,
                )
                return await self._run_qa_flow(
                    task, repo_path, req, intent, _skip_comparison=True
                )

            aspect = decomposed.get("aspect") or req.problem or task.user_input
            all_tasks: list[dict] = decomposed["tasks"]

            # Search tasks = all but last; synthesis task = last
            search_tasks = all_tasks[:-1]
            synthesis_task = all_tasks[-1]

            # ── Log all sub-tasks clearly ─────────────────────────────────
            logger.info(
                "QnA comparison: %d sub-tasks planned, aspect=%r",
                len(all_tasks), aspect,
            )
            for st in all_tasks:
                logger.info(
                    "QnA comparison: sub-task id=%s label=%r scope=%r query=%r",
                    st.get("id"), st.get("label"), st.get("scope", ""), st.get("query", ""),
                )

            # ── Step 2: Gather evidence for each search target (parallel) ─
            logger.info(
                "QnA comparison: gathering evidence for %d targets in parallel",
                len(search_tasks),
            )
            evidence_coros = [
                self._scan_scope_for_topic(
                    repo_path,
                    st.get("scope", ""),
                    st.get("query") or aspect,
                )
                for st in search_tasks
            ]
            raw_results = await asyncio.gather(*evidence_coros, return_exceptions=True)

            evidence_per_target: list[tuple[str, str]] = []
            for st, result in zip(search_tasks, raw_results):
                label = st.get("label", f"target{len(evidence_per_target) + 1}")
                if isinstance(result, Exception):
                    logger.warning(
                        "QnA comparison: evidence gathering raised exception for "
                        "label=%r scope=%r: %s",
                        label, st.get("scope", ""), result,
                    )
                    evidence_per_target.append(
                        (label, f"(gagal mengumpulkan evidence: {result})")
                    )
                else:
                    ev = str(result)
                    if not ev.strip() or any(ev.startswith(p) for p in _EMPTY_EVIDENCE_PREFIXES):
                        logger.warning(
                            "QnA comparison: EMPTY evidence for label=%r scope=%r – "
                            "answer may be marked [DATA TIDAK CUKUP]",
                            label, st.get("scope", ""),
                        )
                    evidence_per_target.append((label, ev))

            t_extract = time.monotonic()
            logger.info(
                "QnA comparison: evidence gathering done in %.2fs — targets=%r",
                t_extract - t_start,
                [label for label, _ in evidence_per_target],
            )

            # ── Step 3: Build prompt and call LLM for comparison ──────────
            label_a = evidence_per_target[0][0] if len(evidence_per_target) > 0 else "target_a"
            label_b = evidence_per_target[1][0] if len(evidence_per_target) > 1 else "target_b"

            ev_sections: list[str] = []
            for label, evidence in evidence_per_target:
                ev_sections.append(f"## 📦 Evidence: {label}\n\n{evidence}")
            evidence_text = "\n\n---\n\n".join(ev_sections)

            # Cap total evidence to avoid max-token issues
            _MAX_COMPARISON_EVIDENCE = 14_000
            if len(evidence_text) > _MAX_COMPARISON_EVIDENCE:
                logger.warning(
                    "QnA comparison: evidence text too large (%d chars), truncating to %d – "
                    "some context may be lost",
                    len(evidence_text), _MAX_COMPARISON_EVIDENCE,
                )
                evidence_text = (
                    evidence_text[:_MAX_COMPARISON_EVIDENCE]
                    + f"\n\n... [evidence dipotong pada {_MAX_COMPARISON_EVIDENCE} karakter "
                    "untuk menghindari batas token]"
                )

            system_prompt = _COMPARE_ANSWER_SYSTEM.format(
                aspect=aspect,
                label_a=label_a,
                label_b=label_b,
            )
            user_msg = (
                f"**Pertanyaan pengguna:**\n{task.user_input}\n\n"
                f"**Aspek yang dibandingkan:** {aspect}\n"
                f"**Target A:** {label_a}\n"
                f"**Target B:** {label_b}\n\n"
                f"---\n\n"
                f"**Data dari repositori:**\n\n{evidence_text}"
            )

            logger.info(
                "QnA comparison: calling LLM for synthesis — "
                "evidence_chars=%d aspect=%r",
                len(evidence_text), aspect,
            )
            qa_response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=QNA_TEMPERATURE,
                top_p=QNA_TOP_P,
                max_tokens=QNA_MAX_TOKENS,
            )

            if not qa_response.strip():
                logger.warning(
                    "QnA comparison: LLM synthesis returned EMPTY response — "
                    "possible max-token hit or model error. "
                    "aspect=%r session=%s",
                    aspect, task.session_id,
                )
                qa_response = (
                    "[DATA TIDAK CUKUP] LLM tidak menghasilkan jawaban perbandingan. "
                    "Kemungkinan batas token tercapai. "
                    "Coba persempit scope pertanyaan atau sebutkan aspek yang lebih spesifik."
                )

            t_total = time.monotonic() - t_start
            logger.info(
                "QnA comparison: synthesis complete in %.2fs total "
                "(extract: %.2fs, llm: %.2fs)",
                t_total, t_extract - t_start, t_total - (t_extract - t_start),
            )

            branch_note = f"🌿 **Branch:** `{req.branch}`\n\n" if req.branch else ""
            target_labels = ", ".join(label for label, _ in evidence_per_target)
            perf_footer = (
                f"\n\n---\n"
                f"⏱️ *⚖️ Komparasi · {t_total:.1f}s "
                f"(ekstraksi: {t_extract - t_start:.1f}s · target: {target_labels})*"
            )
            task.mark_done(branch_note + qa_response.strip() + perf_footer)

            # Save session context for follow-ups
            self._save_session_context(
                task.session_id,
                req.repo_url,
                req.branch,
                req.candidate_route_filenames or None,
            )

        except Exception as exc:
            logger.exception(
                "QnA comparison flow: unexpected error — session=%s error=%s",
                task.session_id, exc,
            )
            task.mark_failed(f"❌ Komparasi gagal karena error tidak terduga: {exc}")

        return task

    # ── Q/A flow ───────────────────────────────────────────────────────────────

    async def _run_qa_flow(
        self,
        task:      AgentTask,
        repo_path: Path,
        req:       RepoExtractionRequest,
        intent:    QAIntent,
        *,
        _skip_comparison: bool = False,
    ) -> AgentTask:
        """
        Run topic-specific extraction + RAG, then answer via LLM.

        Flow:
          0. If question is a comparison and _skip_comparison is False,
             delegate to _run_comparison_flow().
          1. Run topic extractor + RAG + Tavily + dir tree concurrently.
          2. Build LLM prompt from aggregated evidence.
          3. Return direct factual answer with [CONFIRMED/LIKELY/UNVERIFIED] labels.
        """
        try:
            # ── Step 0: Route comparison questions to dedicated flow ───────
            if not _skip_comparison and _COMPARISON_RE.search(task.user_input):
                logger.info(
                    "QnA: comparison question detected — routing to comparison flow "
                    "(session=%s)",
                    task.session_id,
                )
                return await self._run_comparison_flow(task, repo_path, req, intent)

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

            # For SPECIFIC_SYMBOL intent: skip RAG entirely.
            # The symbol tracer (_trace_api_route / _find_symbol_definition) already
            # scans all repo files directly. Including TF-IDF RAG here often pulls in
            # irrelevant files (e.g. WhatsApp download handlers when asking about a
            # different "download" endpoint) that pollute the LLM context and cause
            # false "data tidak cukup" responses.
            # For explanation-style SPECIFIC_SYMBOL: also skip dir-tree to reduce noise.
            # For other intents: include both RAG and dir-tree as usual.
            if intent == QAIntent.SPECIFIC_SYMBOL:
                if not is_explanation_q:
                    # Keep dir tree so LLM knows file structure (useful for locating
                    # which directory the handler/controller lives in)
                    tree = _safe_str(dir_tree, "")
                    if tree.strip():
                        evidence["🗂️ Struktur Direktori"] = tree
            else:
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

            if not qa_response.strip():
                logger.warning(
                    "QnA: LLM returned EMPTY response — possible max-token hit or "
                    "silent model error. intent=%s session=%s problem=%r",
                    intent.value, task.session_id, req.problem,
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

            # ── Step 1: Classify Q/A sub-intent (regex + LLM fallback) ────
            intent = await self._classify_intent(task.user_input)
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
