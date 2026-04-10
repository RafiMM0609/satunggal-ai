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

from src.agents.repo_agent_base import RepoAgentBase, RepoExtractionRequest, MAX_FILE_BYTES
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask
from src.tools.repo_qa import (
    QAIntent,
    classify_intent,
    extract_specific_target,
    run_qa_extraction,
    _search_config_files_for_keyword,
    _evidence_is_empty,
    _find_symbol_definition,
)

logger = logging.getLogger(__name__)

# ── LLM sampling parameters ────────────────────────────────────────────────────

QNA_TEMPERATURE = 0.15
QNA_TOP_P       = 0.90
QNA_MAX_TOKENS  = 16384   # increased from global default to allow richer answers

# ── Map-Reduce constants (Item 2) ─────────────────────────────────────────────
# When the combined RAG evidence exceeds this threshold, activate Map-Reduce.
# Each file is analysed individually (map), then answers are synthesised (reduce).
_MAP_REDUCE_THRESHOLD  = 8_000   # chars; total RAG text above this triggers map-reduce
_MAP_MAX_FILE_CHARS    = 4_000   # chars sent to LLM per file in the map phase
_MAP_MAX_TOKENS        = 8192     # max output tokens for each file-level answer
_REDUCE_MAX_TOKENS     = 8192   # max tokens for the reduce synthesis pass
# _MAP_MAX_TOKENS        = 300     # max output tokens for each file-level answer
# _REDUCE_MAX_TOKENS     = 1_024   # max tokens for the reduce synthesis pass

# ── Adaptive deepening: [DATA TIDAK CUKUP] detector ──────────────────────────
# When the LLM response contains this signal, the agent will automatically
# gather additional file content and re-run the LLM with the enriched evidence.
_DATA_NEEDED_RE = re.compile(
    r"\[(?:DATA TIDAK CUKUP|PERLU DATA TAMBAHAN)[^\]]*\]"
    r"|\bDATA TIDAK CUKUP\b"
    r"|PERLU VERIFIKASI TAMBAHAN\s*[:\-]?\s*([\w/\.]+)",
    re.IGNORECASE,
)

# Maximum extra files to read in the adaptive deepening retry pass.
_MAX_ADAPTIVE_FILES = 5

# Maximum number of evidence-gathering + LLM rounds in the deep-search loop.
# Round 0 is the initial call; rounds 1...N are evidence-enrichment retries.
_MAX_EVIDENCE_LOOP_ROUNDS = 3

# ── Evidence section compression constants (Item 8) ──────────────────────────
# When the total assembled evidence string exceeds _EVIDENCE_COMPRESS_TRIGGER,
# each section is split into chunks of _SECTION_CHUNK_SIZE chars and each chunk
# is summarised by the LLM (map phase, all concurrent). The chunk summaries are
# then joined into a single section summary (reduce phase). This prevents context-
# window overflow while guaranteeing that every section's key details are present
# in the final prompt — unlike hard truncation which silently drops tail content.
_EVIDENCE_COMPRESS_TRIGGER = 80_000   # total chars; compression activated above this
_SECTION_CHUNK_SIZE        = 8_000    # max chars per chunk in the map phase
_SECTION_COMPRESS_TOKENS   = 768      # max output tokens per chunk summary

# ── Concurrency limiter: prevent 429 rate-limit bursts ────────────────────────
# Shared across all asyncio.gather LLM call sites in this module.
# Limits simultaneous in-flight LLM requests to avoid exhausting free-tier quota.
_LLM_CONCURRENCY = asyncio.Semaphore(3)

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

# ── Sub-intent mapping from gatekeeper metadata ────────────────────────────────
# Maps the string sub_intent set by the gatekeeper to the QAIntent enum.
# When the gatekeeper has already classified the sub-topic, we skip the
# internal regex + LLM classify_intent() call and use this mapping directly.

_SUB_INTENT_MAP: dict[str, QAIntent] = {
    "api_endpoints":   QAIntent.API_ENDPOINTS,
    "tech_stack":      QAIntent.TECH_STACK,
    "data_models":     QAIntent.DATA_MODELS,
    "dependencies":    QAIntent.DEPENDENCIES,
    "ci_cd":           QAIntent.CI_CD,
    "security":        QAIntent.SECURITY,
    "main_flow":       QAIntent.MAIN_FLOW,
    "specific_symbol": QAIntent.SPECIFIC_SYMBOL,
    "full_inspection": QAIntent.FULL_INSPECTION,
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

# ── Map-Reduce prompts (Item 2) ───────────────────────────────────────────────

_MAP_FILE_SYSTEM = """\
Kamu adalah analis kode. Berdasarkan SATU file berikut, jawab pertanyaan pengguna:
- Tuliskan 1-3 temuan spesifik dan relevan (nama fungsi, nomor baris, nilai).
- Jika file tidak relevan sama sekali terhadap pertanyaan, tulis hanya: TIDAK RELEVAN
Balas singkat dan padat — maksimal 4 kalimat.
"""

_REDUCE_SYSTEM = """\
Kamu adalah analis kode. Berikut adalah temuan per-file dari beberapa file di repositori.
Sintesekan semua temuan yang relevan menjadi jawaban komprehensif untuk pertanyaan pengguna.
Buang temuan yang ditandai TIDAK RELEVAN.
Sertakan sumber file + baris untuk setiap klaim.
Gunakan label: 🟢 [CONFIRMED] / 🟡 [LIKELY] / 🔴 [UNVERIFIED].
"""

# ── LLM prompt: evidence section chunk compression (Item 8) ──────────────────
_SECTION_COMPRESS_SYSTEM = """\
Kamu adalah ringkaser teknis untuk evidence kode. Berdasarkan bagian evidence berikut,
buat ringkasan padat yang mempertahankan semua detail teknis penting:
- Nama file dan nomor baris (mis. src/auth.py:42)
- Nama fungsi, class, dan method
- Nilai konstanta, environment variable, dan konfigurasi penting
- Pesan error/exception spesifik
- Route HTTP, endpoint URL, dan method (GET/POST/PUT/DELETE)
Fokus pada informasi yang relevan dengan pertanyaan pengguna.
Balas dengan bullet points informatif. Jangan tambahkan penjelasan meta.
"""


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


def _extract_symbols_from_response(text: str) -> list:
    """
    Extract function/symbol names from an LLM response that need further
    investigation via ``_find_symbol_definition``.

    Looks for:
    - Backtick-quoted identifiers that look like function or class names.
    - Names mentioned next to "tidak ditemukan" / "not found" sentinels.

    Returns a deduplicated list, capped at 10 symbols.
    """
    symbols: list = []

    # Backtick-quoted names: `FunctionName`, `HandleRequest`, `ProcessOrder`
    for m in re.finditer(r'`([A-Za-z_][A-Za-z0-9_]{2,})`', text):
        name = m.group(1)
        # Skip file paths (contain '/') and short tokens
        if "/" not in name and "." not in name:
            symbols.append(name)

    # "tidak ditemukan: FunctionName" patterns
    for m in re.finditer(
        r'(?:tidak\s+ditemukan|not\s+found|TIDAK\s+DITEMUKAN)\s*[:\-]?\s*'
        r'[`"\'\']?([A-Za-z_][A-Za-z0-9_]{3,})[`"\'\'\']?',
        text, re.IGNORECASE,
    ):
        symbols.append(m.group(1))

    seen: set = set()
    unique: list = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:10]


def _validate_response_confidence(response: str) -> bool:
    """
    Return True when the LLM response is considered confident.

    A response is *not* confident when:
    - It contains a [DATA TIDAK CUKUP] / [PERLU DATA TAMBAHAN] marker.
    - It has no file:line references (e.g. handler.go:42 or main.py:10).
    - It is empty or very short (< 100 chars).
    """
    if not response or len(response.strip()) < 100:
        return False
    if _DATA_NEEDED_RE.search(response):
        return False
    has_file_ref = bool(re.search(r'\b[\w/.-]+\.\w{1,5}:\d+', response))
    has_confirmed = bool(re.search(r'\[CONFIRMED\]|\U0001f7e2', response))
    return has_file_ref or has_confirmed


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

    # ── Map-Reduce for large RAG evidence (Item 2) ─────────────────────────────

    async def _map_file_to_answer(
        self,
        rel_path: str,
        content: str,
        question: str,
    ) -> str:
        """
        Map phase: ask the LLM about a *single file* in isolation.

        Returns a short (≤4 sentence) summary of the file's relevance to
        *question*, or the sentinel string "TIDAK RELEVAN" if the file is
        unrelated. This keeps each individual prompt well within the LLM's
        context window.
        """
        user_msg = (
            f"**Pertanyaan:** {question}\n\n"
            f"**File:** `{rel_path}`\n"
            f"```\n{content[:_MAP_MAX_FILE_CHARS]}\n```"
        )
        try:
            answer = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _MAP_FILE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.10,
                top_p=0.90,
                max_tokens=_MAP_MAX_TOKENS,
            )
            return f"**{rel_path}:** {answer.strip()}"
        except Exception as exc:
            logger.warning("QnA map_file_answer failed for %s: %s", rel_path, exc)
            return f"**{rel_path}:** (analisis gagal)"

    async def _map_reduce_rag(
        self,
        repo_path: "Path",
        question: str,
    ) -> str:
        """
        Map-Reduce over all top-N RAG files.

        Map phase   – each file is analysed in parallel by the LLM (small prompt).
        Filter step – discard "TIDAK RELEVAN" results.
        Reduce phase – synthesise the remaining per-file answers into one answer.

        Returns a synthesised evidence string, or empty string on failure so the
        caller can fall back to the standard concatenated-RAG approach.
        """
        file_list = await self._read_relevant_files_list(repo_path, question)
        if not file_list:
            return ""

        logger.info(
            "QnA: map-reduce over %d RAG files for question %r",
            len(file_list), question[:80],
        )

        # Map phase – all files in parallel, throttled by _LLM_CONCURRENCY.
        async def _guarded_map(rel_path: str, content: str) -> str:
            async with _LLM_CONCURRENCY:
                return await self._map_file_to_answer(rel_path, content, question)

        map_results = await asyncio.gather(
            *[
                _guarded_map(rel_path, content)
                for rel_path, content in file_list
            ],
            return_exceptions=True,
        )

        # Filter: keep only informative results.
        relevant_answers: list[str] = []
        for r in map_results:
            if isinstance(r, Exception):
                logger.warning("QnA map-reduce gather error: %s", r)
                continue
            ans = str(r)
            if "TIDAK RELEVAN" not in ans.upper():
                relevant_answers.append(ans)

        if not relevant_answers:
            logger.info("QnA map-reduce: all files marked TIDAK RELEVAN")
            return ""

        # If only a few results, no need for a separate reduce LLM call —
        # just join them and let the main QA LLM synthesise.
        combined = "\n\n".join(relevant_answers)
        if len(relevant_answers) <= 3:
            return combined

        # Reduce phase – synthesise into a single coherent evidence block.
        try:
            reduce_user = (
                f"**Pertanyaan pengguna:** {question}\n\n"
                f"**Temuan per-file:**\n\n{combined}"
            )
            synthesis = await self._llm.chat(
                messages=[
                    {"role": "system", "content": _REDUCE_SYSTEM},
                    {"role": "user",   "content": reduce_user},
                ],
                temperature=0.10,
                top_p=0.90,
                max_tokens=_REDUCE_MAX_TOKENS,
            )
            if synthesis.strip():
                return synthesis.strip()
        except Exception as exc:
            logger.warning("QnA map-reduce reduce phase failed: %s", exc)

        # Fallback: return raw combined map results.
        return combined

    # ── Evidence map-reduce compression (Item 8) ───────────────────────────────

    async def _compress_section_for_qna(
        self,
        title: str,
        content: str,
        question: str,
    ) -> str:
        """
        Compress one evidence section via chunked map-reduce.

        Map phase   – split *content* into _SECTION_CHUNK_SIZE char chunks and
                      summarise each chunk in parallel via LLM.
        Reduce phase – join the chunk summaries into a single section summary.

        Sections that fit within one chunk are returned unchanged.
        Falls back to returning the first chunk if all LLM calls fail.
        """
        if len(content) <= _SECTION_CHUNK_SIZE:
            return content

        chunks = [
            content[i : i + _SECTION_CHUNK_SIZE]
            for i in range(0, len(content), _SECTION_CHUNK_SIZE)
        ]
        logger.debug(
            "QnA compress '%s': %d chars → %d chunks",
            title, len(content), len(chunks),
        )

        async def _summarize_chunk(idx: int, chunk: str) -> str:
            user_msg = (
                f"**Pertanyaan:** {question[:200]}\n\n"
                f"**Bagian evidence (chunk {idx + 1}/{len(chunks)}) — {title}:**\n"
                f"```\n{chunk}\n```"
            )
            try:
                result = await self._llm.chat(
                    messages=[
                        {"role": "system", "content": _SECTION_COMPRESS_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.05,
                    top_p=0.90,
                    max_tokens=_SECTION_COMPRESS_TOKENS,
                )
                return result.strip() if result.strip() else ""
            except Exception as exc:
                logger.warning(
                    "QnA compress chunk %d/%d for '%s' failed: %s",
                    idx + 1, len(chunks), title, exc,
                )
                return ""

        async def _guarded_chunk(idx: int, chunk: str) -> str:
            async with _LLM_CONCURRENCY:
                return await _summarize_chunk(idx, chunk)

        chunk_results = await asyncio.gather(
            *[_guarded_chunk(i, chunk) for i, chunk in enumerate(chunks)],
            return_exceptions=True,
        )

        summaries: list[str] = []
        for i, r in enumerate(chunk_results):
            if isinstance(r, Exception):
                logger.warning(
                    "QnA compress gather error for '%s' chunk %d: %s", title, i, r
                )
            elif r:
                summaries.append(r)

        if not summaries:
            # Fallback: first chunk verbatim so the section is not entirely lost.
            return content[:_SECTION_CHUNK_SIZE] + "\n... [ringkasan gagal — dipotong]"

        combined = "\n\n".join(summaries)
        logger.debug(
            "QnA: compressed section '%s': %d → %d chars (%d chunks)",
            title, len(content), len(combined), len(chunks),
        )
        return f"[RINGKASAN — {len(chunks)} bagian]\n{combined}"

    async def _compress_evidence_for_qna(
        self,
        evidence: dict[str, str],
        question: str,
    ) -> dict[str, str]:
        """
        Compress all evidence sections concurrently via per-section map-reduce.

        Each section whose content exceeds _SECTION_CHUNK_SIZE chars is split into
        chunks and summarised by _compress_section_for_qna(). Sections that fit in
        one chunk are passed through unchanged.

        All sections are processed in parallel (asyncio.gather) to minimise latency.
        """
        keys = list(evidence.keys())
        logger.info(
            "QnA: compressing %d evidence section(s) for question %r",
            len(keys), question[:80],
        )
        async def _guarded_section(key: str) -> str:
            async with _LLM_CONCURRENCY:
                return await self._compress_section_for_qna(key, evidence[key], question)

        results = await asyncio.gather(
            *[_guarded_section(k) for k in keys],
            return_exceptions=True,
        )
        compressed: dict[str, str] = {}
        for k, r in zip(keys, results):
            if isinstance(r, Exception):
                logger.warning(
                    "QnA: compress_evidence_for_qna error for '%s': %s", k, r
                )
                compressed[k] = evidence[k]   # keep original on failure
            else:
                compressed[k] = str(r)
        return compressed

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
            async def _guarded_scope(scope: str, query: str) -> str:
                async with _LLM_CONCURRENCY:
                    return await self._scan_scope_for_topic(repo_path, scope, query)

            evidence_coros = [
                _guarded_scope(st.get("scope", ""), st.get("query") or aspect)
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
                # ── Adaptive deepening for SPECIFIC_SYMBOL ─────────────────
                # When the symbol tracer returns insufficient evidence (nothing
                # found in source code), activate a fallback that:
                #   1. Searches config/data files (.json, .yaml, etc.)
                #   2. Falls back to TF-IDF RAG on ALL files including configs
                # This ensures questions about Postman collections, OpenAPI specs,
                # CI/CD YAML, Dockerfile, etc. are answered correctly.
                symbol_evidence_values = list(evidence.values()) if evidence else []
                symbol_evidence_text = "\n".join(symbol_evidence_values)
                symbol_has_content = (
                    bool(symbol_evidence_text.strip())
                    and not _evidence_is_empty(symbol_evidence_text)
                    and "Extraction Error" not in evidence
                )

                if not symbol_has_content:
                    logger.info(
                        "QnA: SPECIFIC_SYMBOL evidence is insufficient for target=%r — "
                        "activating config-file + RAG fallback",
                        symbol_target,
                    )
                    # Search config / data files for the target keyword
                    config_fallback = await _search_config_files_for_keyword(
                        repo_path,
                        symbol_target or (req.problem or task.user_input)[:60],
                        req.problem or task.user_input,
                    )
                    if config_fallback and not config_fallback.startswith("(tidak ditemukan"):
                        evidence["📋 File Konfigurasi & Data"] = config_fallback
                        logger.info("QnA: config-file fallback added %d chars", len(config_fallback))

                    # Also include TF-IDF RAG as a broader safety net
                    rag_fallback = _safe_str(rag_files, "")
                    if (
                        rag_fallback.strip()
                        and "unavailable" not in rag_fallback
                        and "error" not in rag_fallback.lower()
                    ):
                        evidence["📂 File Relevan (RAG Fallback)"] = rag_fallback
                        logger.info("QnA: RAG fallback added for SPECIFIC_SYMBOL")

                if not is_explanation_q:
                    # Keep dir tree so LLM knows file structure (useful for locating
                    # which directory the handler/controller lives in)
                    tree = _safe_str(dir_tree, "")
                    if tree.strip():
                        evidence["🗂️ Struktur Direktori"] = tree
            else:
                # Secondary: RAG-relevant files.
                # If the combined RAG text is large, use Map-Reduce (Item 2) to
                # avoid flooding the context window with unfiltered file dumps.
                rag_text = _safe_str(rag_files, "(RAG unavailable)")
                if rag_text.strip() and "unavailable" not in rag_text and "error" not in rag_text.lower():
                    if len(rag_text) > _MAP_REDUCE_THRESHOLD:
                        logger.info(
                            "QnA: RAG text is %d chars (> %d) — activating Map-Reduce",
                            len(rag_text), _MAP_REDUCE_THRESHOLD,
                        )
                        mr_text = await self._map_reduce_rag(
                            repo_path, req.problem or task.user_input
                        )
                        if mr_text.strip():
                            evidence["📂 File Relevan (Map-Reduce)"] = mr_text
                        else:
                            # Map-reduce returned nothing useful; fall back to direct RAG.
                            evidence["📂 File Relevan (RAG)"] = rag_text
                    else:
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

            # ── Item 8: Evidence map-reduce compression ────────────────────
            # When total evidence exceeds the safe limit, compress every section
            # individually via LLM (map phase, all sections run in parallel), then
            # rebuild the evidence string from the compressed dict (reduce phase).
            # This keeps every section represented in the final prompt while
            # staying well within the model's context window — no silent data loss.
            if len(evidence_text) > _EVIDENCE_COMPRESS_TRIGGER:
                logger.info(
                    "QnA: evidence %d chars (> %d) — starting per-section map-reduce compression",
                    len(evidence_text), _EVIDENCE_COMPRESS_TRIGGER,
                )
                t_compress_start = time.monotonic()
                evidence = await self._compress_evidence_for_qna(
                    evidence, req.problem or task.user_input
                )
                evidence_text = self._build_evidence_text(evidence)
                logger.info(
                    "QnA: evidence after compression: %d chars (%.2fs)",
                    len(evidence_text), time.monotonic() - t_compress_start,
                )

            # ── Item 7: Explicit "Data Not Enough" signal ─────────────────
            # If evidence is still large even after compression, add an
            # instruction so the LLM can explicitly flag missing information
            # instead of hallucinating.
            _STANDARD_LIMIT = 30_000  # chars that fit safely without truncation
            evidence_was_truncated = len(evidence_text) > _STANDARD_LIMIT
            data_not_enough_instruction = ""
            if evidence_was_truncated:
                data_not_enough_instruction = (
                    "\n\n**⚠️ INSTRUKSI TAMBAHAN (evidence mungkin tidak lengkap):**\n"
                    "Jika data yang diberikan tidak cukup untuk menjawab dengan keyakinan tinggi,\n"
                    "tulis di akhir jawaban:\n"
                    "> **[DATA TIDAK CUKUP]** – diperlukan: `<nama file atau informasi tambahan>`\n"
                    "Jangan mengarang detail yang tidak ada dalam evidence.\n"
                )
                logger.info(
                    "QnA: evidence is %d chars (> %d limit) — adding 'data not enough' instruction",
                    len(evidence_text), _STANDARD_LIMIT,
                )

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
                    f"**Panduan verbositas:** {verbosity_note}\n"
                    f"{data_not_enough_instruction}\n"
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
                    f"**Panduan verbositas:** {verbosity_note}\n"
                    f"{data_not_enough_instruction}\n"
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

            # ── Multi-round adaptive deepening (looping evidence search) ──────
            # Round 0 is the initial LLM call above.  Each subsequent round:
            #   1. Validate whether the response is already confident.
            #   2. Read explicitly mentioned file paths.
            #   3. Search for function/symbol definitions mentioned in the response.
            #   4. Fallback: search config/data files.
            #   5. If new evidence found, re-run LLM; otherwise stop.
            # Runs up to _MAX_EVIDENCE_LOOP_ROUNDS - 1 extra retries.
            current_evidence = evidence.copy()
            current_user_msg = user_msg

            for _round in range(1, _MAX_EVIDENCE_LOOP_ROUNDS):
                if _validate_response_confidence(qa_response):
                    logger.info(
                        "QnA: response is confident after round %d — stopping loop",
                        _round - 1,
                    )
                    break

                logger.info(
                    "QnA: response not confident (round %d/%d) — gathering more evidence",
                    _round, _MAX_EVIDENCE_LOOP_ROUNDS - 1,
                )
                new_parts: list[str] = []

                # 1. Read explicitly mentioned file paths
                extra_paths_found: list[str] = re.findall(
                    r"\[(?:PERLU DATA TAMBAHAN|DATA TIDAK CUKUP)[:\s]*([^\]]+)\]",
                    qa_response,
                    re.IGNORECASE,
                )
                extra_paths_found = [
                    p.strip() for p in extra_paths_found if p.strip()
                ][:_MAX_ADAPTIVE_FILES]

                for rel_path in extra_paths_found:
                    abs_path = repo_path / rel_path
                    if abs_path.exists() and abs_path.is_file():
                        try:
                            file_text = abs_path.read_text(errors="replace")[:MAX_FILE_BYTES]
                            ext = abs_path.suffix.lstrip(".")
                            new_parts.append(
                                f"### \U0001f4c4 `{rel_path}` (diminta oleh LLM)\n"
                                f"```{ext}\n{file_text}\n```"
                            )
                            logger.debug("QnA loop: read extra file %s", rel_path)
                        except OSError as exc:
                            logger.debug("QnA loop: cannot read %s: %s", rel_path, exc)
                    else:
                        logger.debug("QnA loop: file not found: %s", rel_path)

                # 2. Search for function/symbol definitions mentioned in the response
                needed_symbols = _extract_symbols_from_response(qa_response)
                if needed_symbols:
                    logger.info(
                        "QnA loop round %d: searching definitions for symbols=%r",
                        _round, needed_symbols,
                    )
                    sym_results = await asyncio.gather(
                        *[
                            _find_symbol_definition(repo_path, sym)
                            for sym in needed_symbols
                        ],
                        return_exceptions=True,
                    )
                    for sym, result in zip(needed_symbols, sym_results):
                        if isinstance(result, Exception):
                            logger.debug(
                                "QnA loop: definition lookup error for %r: %s",
                                sym, result,
                            )
                            continue
                        defn = str(result)
                        if defn and not _evidence_is_empty(defn):
                            new_parts.append(
                                f"### \U0001f50d Definisi: `{sym}` (search-for-definition)\n"
                                f"{defn}"
                            )
                            logger.debug("QnA loop: found definition for %r", sym)

                # 3. Fallback: config/data file search when nothing else worked
                if not new_parts:
                    logger.info(
                        "QnA loop round %d: no new evidence from files/symbols — "
                        "trying config file search",
                        _round,
                    )
                    config_enrichment = await _search_config_files_for_keyword(
                        repo_path,
                        symbol_target or (req.problem or task.user_input)[:60],
                        req.problem or task.user_input,
                    )
                    if config_enrichment and not config_enrichment.startswith(
                        "(tidak ditemukan"
                    ):
                        new_parts.append(config_enrichment)

                if not new_parts:
                    logger.info(
                        "QnA loop round %d: no additional evidence found — stopping loop",
                        _round,
                    )
                    break

                # 4. Append new evidence and re-run LLM
                round_evidence_key = f"\U0001f50d Pencarian Tambahan Ronde {_round}"
                current_evidence[round_evidence_key] = "\n\n".join(new_parts)
                current_user_msg = current_user_msg + (
                    f"\n\n---\n\n"
                    f"**\u26a0\ufe0f TAMBAHAN RONDE {_round}: Evidence baru ditemukan.**\n"
                    f"Gunakan data berikut untuk melengkapi/memperbaiki jawaban:\n\n"
                    f"{current_evidence[round_evidence_key]}"
                )

                logger.info(
                    "QnA loop: re-running LLM (round %d) with enriched evidence",
                    _round,
                )
                try:
                    retry_response = await self._llm.chat(
                        messages=[
                            {"role": "system", "content": _QA_SYSTEM_PROMPT},
                            {"role": "user",   "content": current_user_msg},
                        ],
                        temperature=QNA_TEMPERATURE,
                        top_p=QNA_TOP_P,
                        max_tokens=QNA_MAX_TOKENS,
                    )
                    if retry_response.strip():
                        qa_response = retry_response
                        logger.info("QnA loop: round %d retry succeeded", _round)
                    else:
                        logger.warning(
                            "QnA loop: round %d LLM returned empty — stopping loop",
                            _round,
                        )
                        break
                except Exception as retry_exc:
                    logger.warning(
                        "QnA loop: round %d retry failed (%s); keeping previous response",
                        _round, retry_exc,
                    )
                    break
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
            # Optimization: if the gatekeeper already resolved sub_intent,
            # skip the internal classify step and use the gatekeeper result.
            gatekeeper_sub_intent = task.metadata.get("sub_intent")
            if gatekeeper_sub_intent and gatekeeper_sub_intent in _SUB_INTENT_MAP:
                intent = _SUB_INTENT_MAP[gatekeeper_sub_intent]
                logger.info(
                    "QnA: using gatekeeper sub_intent=%s → QAIntent=%s",
                    gatekeeper_sub_intent, intent.value,
                )
            else:
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
