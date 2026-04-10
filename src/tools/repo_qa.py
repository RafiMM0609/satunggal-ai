"""
repo_qa.py – Repository Q/A Engine untuk DeveloperInspectorAgent.

Menyediakan:
  1. classify_intent(user_input) → QAIntent
     Mendeteksi apakah input adalah Q/A singkat atau permintaan inspeksi penuh.

  2. Extractor per-topik (semua read-only, berbasis grep + file parse):
     - extract_api_endpoints   : rute HTTP, OpenAPI, Swagger
     - extract_tech_stack      : framework, bahasa, DB, queue, library
     - extract_data_models     : ORM class, Pydantic model, schema
     - extract_dependencies    : requirements.txt, package.json, pyproject.toml, go.mod
     - extract_ci_cd           : GitHub Actions, GitLab CI, Jenkinsfile, Dockerfile
     - extract_security        : auth middleware, env vars, secrets pattern
     - extract_main_flow       : entry points, startup sequence, request lifecycle
     - extract_specific_symbol : definisi + penggunaan simbol tertentu (API path / fungsi)

  3. run_qa_extraction(repo_path, intent, user_input) → dict[str, str]
     Menjalankan extractor yang sesuai dan mengembalikan evidence dict.

Konvensi:
  - Semua fungsi adalah async.
  - Semua operasi READ-ONLY (grep, find, cat) — tidak ada write/exec.
  - Setiap extractor mengembalikan string markdown yang siap dikirim ke LLM.
  - Fallback gracefully jika tidak ada file yang cocok.
"""

from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_BYTES_PER_FILE   = 40_000   # max chars per file snippet
MAX_GREP_LINES       = 100      # max lines per grep result
MAX_FILES_PER_TOPIC  = 12       # max files read per extractor

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

# ── Q/A Intent Enum ───────────────────────────────────────────────────────────

class QAIntent(str, Enum):
    """
    Jenis pertanyaan yang dapat dijawab Q/A.
    FULL_INSPECTION = bukan pertanyaan singkat, jalankan inspeksi penuh.
    """
    API_ENDPOINTS    = "api_endpoints"      # "ada api apa", "endpoint apa saja"
    TECH_STACK       = "tech_stack"         # "teknologi apa", "framework apa"
    DATA_MODELS      = "data_models"        # "model data", "schema", "database"
    DEPENDENCIES     = "dependencies"       # "library apa", "dependencies"
    CI_CD            = "ci_cd"              # "CI/CD", "pipeline", "deploy flow"
    SECURITY         = "security"           # "auth", "autentikasi", "keamanan"
    MAIN_FLOW        = "main_flow"          # "flow utama", "alur request", "startup"
    SPECIFIC_SYMBOL  = "specific_symbol"    # "jelaskan /upload", "apa itu UserModel"
    FULL_INSPECTION  = "full_inspection"    # bug report, error log, masalah → inspeksi penuh


# ── Intent patterns ───────────────────────────────────────────────────────────
# Setiap entry: (QAIntent, list of regex patterns, list of negative patterns)
# Pattern pertama yang match menentukan intent.

_INTENT_RULES: list[tuple[QAIntent, list[str], list[str]]] = [
    (
        QAIntent.CI_CD,
        [
            r"ci[/ \-]?cd", r"pipeline", r"github.action", r"gitlab.ci",
            r"jenkins", r"deploy.flow", r"workflow.deploy", r"bagaimana.deploy",
            r"proses.deploy", r"alur.deploy", r"cara.deploy",
            # Dockerfile / Docker Compose mentions (build & deployment config)
            r"dockerfile", r"docker.?compose", r"docker.?file\b",
            # Imperative requests: "berikan script dockerfile", "tampilkan docker-compose"
            r"(?:berikan|tampilkan|tunjukkan|kasih|lihat)\s+(?:script\s+)?(?:dockerfile|docker.?compose)",
        ],
        [],
    ),
    (
        QAIntent.API_ENDPOINTS,
        [
            # match various spellings and spacing for 'endpoint' / 'end poin'
            r"api.apa", r"endpoint.apa", r"end.?point", r"end.?poin", r"route.apa", r"ada.api",
            r"list.api", r"daftar.api", r"daftar.endpoint", r"ada.endpoint",
            r"what.api", r"what.endpoint", r"list.*endpoint", r"list.*route",
            r"available.*api", r"available.*endpoint",
            r"show.*route", r"all.*route", r"all.*endpoint", r"semua.*api",
            r"semua.*endpoint", r"semua.*route",
            # Go / specific routing file mentions → treat as API_ENDPOINTS query
            r"routes?\.go\b", r"router?\.go\b", r"routing\.go\b",
            r"(?:di|dalam|pada|in|file)\s+routes?\.(?:go|py|js|ts)\b",
            r"(?:fitur|feature|route|endpoint|fungsi)\s+.*\.(?:go|py|js|ts)\b",
            r"\.go\s+.*(?:fitur|feature|download|unduh|endpoint|route)",
            # download / upload feature queries (commonly maps to HTTP routes)
            r"(?:fitur|feature).*(?:download|unduh|upload|ekspor|export)",
            r"(?:download|unduh|upload).*(?:route|endpoint|api|fitur|feature)",
            # Existence questions about API/endpoints: "apakah ada endpoint untuk X"
            r"(?:adakah|apakah.ada)\s+(?:api|endpoint|route|rute|path)\b",
            r"(?:adakah|apakah.ada).*\b(?:endpoint|route|rute)\b",
        ],
        [],
    ),
    (
        QAIntent.TECH_STACK,
        [
            r"teknologi.apa", r"tech.stack", r"framework.apa", r"bahasa.apa",
            r"library.apa", r"stack.apa", r"dibangun.dengan", r"dibuat.dengan",
            r"what.tech", r"what.framework", r"what.language", r"stack.*used",
            r"technology.*used", r"bahasa.pemrograman", r"programming.language",
            r"language.*used", r"dipakai.bahasa",
        ],
        [],
    ),
    (
        QAIntent.DATA_MODELS,
        [
            r"model.data", r"data.model", r"schema.apa", r"database.schema",
            r"schema.database", r"tabel.apa", r"structure.database",
            r"orm.*model", r"entity.apa", r"data.struktur",
            r"what.*model", r"what.*schema", r"what.*table",
            r"apa.saja.*model", r"apa.saja.*schema",
        ],
        [],
    ),
    (
        QAIntent.DEPENDENCIES,
        [
            r"dependenc", r"library.yang.dipakai", r"package.apa",
            r"requirements", r"modul.apa", r"what.*librar", r"what.*package",
            r"what.*dependenc", r"list.*package", r"apa.*package",
            r"package.yang.dipakai", r"packages",
        ],
        [],
    ),
    (
        QAIntent.SECURITY,
        [
            r"keamanan", r"autentika", r"autorisas", r"auth(entika|orisas|ensikasi)?",
            r"jwt", r"oauth", r"middleware.auth", r"security", r"api.key",
            r"secret", r"enkripsi", r"proteksi", r"permission", r"role.*access",
            # Existence questions about security/auth
            r"(?:adakah|apakah.ada)\s+(?:middleware|autentika|autorisas|auth|keamanan|permission|role|jwt|oauth)\b",
        ],
        [],
    ),
    (
        QAIntent.MAIN_FLOW,
        [
            r"flow.utama", r"alur.utama", r"alur.request", r"request.lifecycle",
            r"startup.*flow", r"how.*works", r"bagaimana.*bekerja",
            r"main.flow", r"request.*flow", r"arsitektur.*flow",
            r"flow.*arsitektur", r"cara.kerja", r"alur.kerja",
            r"alur.sistem", r"sistem.bekerja",
            # Directory / project structure queries
            r"struktur.dir", r"struktur.project", r"struktur.projek",
            r"struktur.folder", r"struktur.aplikasi", r"struktur.app",
            r"directory.struct", r"folder.struct", r"project.struct",
            r"susunan.folder", r"susunan.file", r"layout.project",
            r"arsitektur.aplikasi", r"arsitektur.project", r"arsitektur.projek",
            r"(?:jabarkan|jelaskan|describe|show|tampilkan)\s+(?:struktur|directory|folder|layout|tree)",
            r"(?:struktur|directory|folder|layout|tree)\s+(?:yang.ada|project|repo|aplikasi|app)",
        ],
        [],
    ),
    (
        QAIntent.SPECIFIC_SYMBOL,
        [
            # Match explicit path patterns like /upload, /api/v1/users (slash required)
            r"(?:jelaskan|jelasin|jabarkan|explain|apa.itu|what.is|describe|tentang|cari)\s+/[a-zA-Z]",
            # Match "jelaskan fungsi X", "apa itu class Y", "jabarkan method Z"
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|cari)\s+(?:fungsi|function|class|method|api|endpoint)\s+\w",
            # Match standalone /path questions (not inside a URL)
            r"(?:endpoint|api|route)\s+[/]\S+",
            r"[/][a-zA-Z][a-zA-Z0-9_/\-]+\s+(?:itu|adalah|digunakan|bekerja|fungsi)",
            # Match "jelaskan CamelCase or snake_case identifier"
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|cari)\s+[A-Z][a-zA-Z0-9]+",
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|cari)\s+[a-z][a-z0-9]*(?:_[a-z0-9]+)+",
            # Business logic / implementation detail follow-up patterns (commonly referential)
            r"logika\s*bisnis",
            r"business\s*logic",
            r"alur\s*bisnis",
            r"(?:detailkan|jelaskan|jelasin|jabarkan)\s+(?:logika|alur|implementasi|flow|cara\s*kerja)",
            r"(?:logika|alur|implementasi|cara\s*kerja)\s+(?:dari\s+)?(?:api|endpoint|handler|controller)",
            r"(?:bisa\s+)?(?:detailkan|elaborasi|expand)\s+(?:logika|alur|flow|implementasi)",
            r"(?:lebih\s+)?detail\s+(?:logika|alur|flow|implementasi)",
            # "cara kerja" / "bagaimana bekerja" with explicit API path in message
            r"cara\s*kerja\s+api",
            r"bagaimana\s*(?:cara\s*)?(?:api|endpoint|route)\s+(?:ini\s+)?bekerja",
            r"ingin\s+tahu\s+(?:cara|bagaimana)",
            r"(?:cara|how)\s+kerja\s+(?:api|endpoint|handler)",
            # ── Directory-scoped requests ─────────────────────────────────────
            # "jelasin ... pada ./src/agents/developer"
            # "jelaskan pengolahan request yang ada pada ./src/..."
            r"(?:pada|di|dalam|in|at)\s+\.?/[a-zA-Z]",
            r"yang\s+ada\s+(?:pada|di|dalam)\s+\.?/[a-zA-Z]",
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|coba\s+(?:dong|deh|lah|yuk|sih|ya)?\s*jelas\w*)\s+[^./\n]{0,60}\.?/[a-zA-Z]",
            # ── File content requests ─────────────────────────────────────────
            # "jelaskan isi file main.py", "tampilkan config.yaml", "lihat router.go"
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|tampilkan|tunjukkan|berikan"
            r"|kasih|lihat|buka|cek|periksa|apa.isi)\s+"
            r"(?:isi\s+(?:dari\s+)?)?(?:file\s+)?\S+\.(?:py|go|js|ts|jsx|tsx|java|php|rb|rs|kt"
            r"|cs|yaml|yml|json|toml|env|sh|md|txt|cfg|ini|sql|html|css|scss)\b",
            # "isi file X.ext" / "isi dari X.ext" without a leading verb
            r"\bisi\s+(?:dari\s+)?(?:file\s+)?\S+\.(?:py|go|js|ts|jsx|tsx|java|php|rb|rs|kt"
            r"|cs|yaml|yml|json|toml|env|sh|md|txt|cfg|ini|sql|html|css|scss)\b",
            # "jelaskan isi file main" (filename without extension)
            r"(?:jelaskan|jelasin|jabarkan|explain|describe|tampilkan|tunjukkan|berikan)\s+(?:isi\s+)?(?:dari\s+)?file\s+\S+",
            # ── File path without leading slash in "pada/di" context ──────────
            # "berikan list function yang ada pada src/agents/agent.py"
            # "lihat implementasi di controllers/user.go"
            r"(?:pada|di|dalam|in|at)\s+\S+\.(?:py|go|js|ts|jsx|tsx|java|php|rb|rs|kt"
            r"|cs|yaml|yml|json|toml|env|sh|md|txt|cfg|ini|sql|html|css|scss)\b",
            r"yang\s+ada\s+(?:pada|di|dalam)\s+\S+\.(?:py|go|js|ts|jsx|tsx|java|php|rb|rs|kt"
            r"|cs|yaml|yml|json|toml|env|sh|md|txt|cfg|ini|sql|html|css|scss)\b",
            # ── "list/daftar function/method" requests ────────────────────────
            # "berikan list function yang ada pada ...",
            # "tampilkan daftar function di file X"
            r"(?:berikan|tampilkan|tunjukkan|kasih|lihat|list|daftar)\s+"
            r"(?:list\s+|daftar\s+)?(?:fungsi|function|method|class|def)\s+",
            # ── Existence questions about specific features / handlers ────────
            # "adakah handle upload file", "apakah ada fungsi untuk login"
            # Negative lookahead prevents matching bug/error/masalah reports
            r"(?:adakah|apakah\s+ada|ada\s+tidak)\s+"
            r"(?!(?:bug|error|crash|masalah|broken|gagal|exception|fix|perbaik))\w",
            # ── Imperative verbs for specific code artifacts ──────────────────
            # "tampilkan fungsi X", "berikan kode method Y", "tunjukkan class Z"
            r"(?:berikan|tampilkan|tunjukkan|kasih|lihat)\s+(?:kode\s+|script\s+|isi\s+|implementasi\s+)?"
            r"(?:dari\s+)?(?:fungsi|function|class|method|handler)\s+\w",
        ],
        [],
    ),
]

_INSPECTION_TRIGGERS = [
    r"error", r"bug", r"crash", r"fix", r"masalah", r"gagal", r"broken",
    r"exception", r"traceback", r"tidak berfungsi", r"not working",
    r"dibandingkan", r"diagnos", r"root.cause", r"kenapa.*gagal",
    r"mengapa.*error", r"apa.penyebab",
]


def classify_intent(user_input: str) -> QAIntent:
    """
    Klasifikasi intent dari input pengguna.

    Returns:
        QAIntent.FULL_INSPECTION jika user melaporkan bug/error.
        QAIntent.SPECIFIC_SYMBOL jika user menanyakan simbol/endpoint spesifik.
        QAIntent lainnya sesuai topik pertanyaan.
    """
    text = user_input.lower().strip()

    # Detect qualified identifiers like controllers.DownloadFile or pkg.Method.
    # This is a very strong signal for SPECIFIC_SYMBOL — check before all other rules
    # so it is not shadowed by e.g. MAIN_FLOW patterns.
    if re.search(r'\b[a-z_]\w*\.[A-Z][a-zA-Z0-9]+\b', user_input):
        logger.debug("QA classify: qualified identifier (pkg.Export) detected → SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # Jika user meminta penjelasan dan menyertakan path spesifik (mis. "/upload"),
    # anggap ini permintaan `SPECIFIC_SYMBOL` dan beri prioritas sebelum pattern
    # Q/A umum seperti "ada api apa".
    # Handles both " /path" (space before slash) and "./path" (dot-slash prefix).
    _explain_trigger = r"(?:jelaskan|jelasin|jabarkan|explain|apa.itu|what.is|describe|cari)"
    _path_present    = r"(?:(?:^|\s)\.?/[a-z0-9_\-/]+)"
    if re.search(_explain_trigger, text) and re.search(_path_present, text):
        logger.debug("QA classify: specific symbol detected (path present) -> SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # Directory-scoped requests without explicit explain trigger:
    # "pengolahan request yang ada pada ./src/agents/developer"
    # "yang ada di ./src/...", "pada ./src/..."
    if re.search(r"(?:pada|di|dalam|in|at)\s+\.?/[a-z0-9_\-/]+", text):
        logger.debug("QA classify: directory-scoped request (pada/di ./path) → SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # Jika user menyebutkan API path dengan path-parameter (/:param) — ciri khas
    # route berparam seperti /download/:appuuid/:uuid — ini hampir pasti SPECIFIC_SYMBOL
    # meski tanpa kata pemicu eksplisit (jelaskan/explain/dll.).
    # Pattern ini juga menangkap follow-up questions yang hanya menyebut path API.
    if re.search(r"(?:^|\s|api\s*:?\s*\n?\s*)/[a-z0-9_\-]+(?:/[a-z0-9_\-:]+){1,}", text):
        logger.debug("QA classify: route path pattern detected -> SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # If user explicitly requests Q/A mode ("qna", "q/a", "use qna"),
    # prefer matching Q/A intents first and fall back to a SPECIFIC_SYMBOL
    # Q/A intent so the agent runs in Q/A mode instead of forcing full
    # inspection when words like "error" might also be present.
    if re.search(r"\b(qna|q\/a|q and a|q&a|qna mode|q\/a mode|use qna|use q\/a)\b", text):
        for intent, patterns, neg_patterns in _INTENT_RULES:
            for pat in patterns:
                if re.search(pat, text):
                    if any(re.search(n, text) for n in neg_patterns):
                        continue
                    logger.debug("QA classify: explicit Q/A override -> %s matched by %r", intent, pat)
                    return intent
        logger.debug("QA classify: explicit Q/A override but no topic matched -> SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # Match topik Q/A berdasarkan urutan prioritas
    for intent, patterns, neg_patterns in _INTENT_RULES:
        for pat in patterns:
            if re.search(pat, text):
                # Check negative conditions
                if any(re.search(n, text) for n in neg_patterns):
                    continue
                logger.debug("QA classify: %s matched by %r", intent, pat)
                return intent

    # Error/bug trigger → inspeksi penuh
    for pat in _INSPECTION_TRIGGERS:
        if re.search(pat, text):
            logger.debug("QA classify: FULL_INSPECTION triggered by %r", pat)
            return QAIntent.FULL_INSPECTION

    # Fallback: detect "jabarkan/jelaskan/cari <CamelCase|snake_case>" using the
    # ORIGINAL (non-lowercased) input — catches identifiers like HandleDownload,
    # get_user_data that can't be detected in lowercase text reliably.
    # Also handles informal Indonesian "jelasin", "tolong jelasin", "coba jelasin",
    # and "coba dong jelasin" (with a known particle word like "dong/deh/lah").
    if re.search(
        r"(?:jelaskan|jelasin|jabarkan|explain|describe|cari|apa.itu|tentang"
        r"|coba\s+(?:dong|deh|lah|yuk|sih|ya)?\s*jelas\w*"
        r"|tolong\s+(?:dong|deh|lah|ya)?\s*jelas\w*)\s+"
        r"(?:[A-Z][a-zA-Z0-9]{2,}|[a-z][a-z0-9]+(?:_[a-z0-9]+)+)\b",
        user_input,  # original case
    ):
        logger.debug("QA classify: CamelCase/snake_case identifier detected → SPECIFIC_SYMBOL")
        return QAIntent.SPECIFIC_SYMBOL

    # Default: inspeksi penuh jika tidak ada Q/A pattern yang cocok
    logger.debug("QA classify: no Q/A pattern matched → FULL_INSPECTION")
    return QAIntent.FULL_INSPECTION


def extract_specific_target(user_input: str) -> str:
    """
    Ekstrak target spesifik dari query (nama path API, fungsi, class, dsb.).

    Contoh:
      "jelaskan api /upload" → "/upload"
      "apa itu fungsi process_payment" → "process_payment"
      "endpoint /api/v1/users bagaimana" → "/api/v1/users"
      "jabarkan fungsi downloadFile" → "downloadFile"
      "cari fungsi HandleDownload" → "HandleDownload"
      "/download/:appuuid/:uuid" → "/download/:appuuid/:uuid"
      "controllers.DownloadFile" → "controllers.DownloadFile"
      "jelasin ... pada ./src/agents/developer" → "/src/agents/developer"
    """
    # Broad trigger-word pattern — includes informal Indonesian variants
    # (jelasin, coba jelasin, coba dong jelasin, tolong jelaskan, etc.)
    # and imperative/existence verbs.
    _TRIGGER = (
        r"(?:jelaskan|jelasin|jabarkan|explain|apa.itu|what.is|describe"
        r"|tentang|cari"
        r"|coba\s+(?:dong|deh|lah|yuk|sih|ya)?\s*jelas\w*"
        r"|tolong\s+(?:dong|deh|lah|ya)?\s*jelas\w*"
        r"|adakah|apakah.ada|ada.tidak"
        r"|berikan|tampilkan|tunjukkan|kasih|lihat.isi|apa.isi)"
    )

    # Optional Indonesian/English prepositions that can sit between the keyword
    # and the actual function / symbol name (e.g. "fungsi untuk login").
    _PREP = r"(?:(?:untuk|buat|bagi|dengan|dari|yang|terkait|berkaitan|tentang|about|for|of)\s+)?"

    # Qualified identifier: controllers.DownloadFile, pkg.Method, etc.
    # Recognised as `lowercase_word.CamelCaseWord` — very common in Go/Java/JS.
    # Checked BEFORE everything to avoid false-positives on paths/filenames.
    text_no_url = re.sub(r"https?://\S+", "", user_input)
    qualified_match = re.search(
        r"\b([A-Za-z_]\w*)\.([A-Z][a-zA-Z0-9_]+)\b",
        text_no_url,
    )
    if qualified_match:
        return qualified_match.group(0)  # e.g. "controllers.DownloadFile"

    # "jelaskan fungsi X" — checked BEFORE path/file so it correctly captures
    # the function name even when a filename appears later in the sentence
    # (e.g. "jelaskan method handle_request di agent.py" → "handle_request").
    kw_sym_match = re.search(
        rf"{_TRIGGER}\s+(?:fungsi|function|class|method|api|endpoint|handler|fitur|feature|script)\s+{_PREP}([a-zA-Z_]\w*)",
        user_input,
        re.IGNORECASE,
    )
    if kw_sym_match:
        return kw_sym_match.group(kw_sym_match.lastindex)

    # Filename.extension — checked BEFORE api-path so that relative file paths
    # like "controllers/user.go" are not truncated by the /path extractor.
    # Character class includes "/" to handle multi-segment paths.
    file_match = re.search(
        r"\b([\w\-./]+\.(?:py|go|js|ts|jsx|tsx|java|php|rb|rs|kt|cs"
        r"|yaml|yml|json|toml|env|sh|md|txt|cfg|ini|sql|html|css|scss))\b",
        text_no_url,
        re.IGNORECASE,
    )
    if file_match:
        return file_match.group(1)

    # Directory target expressed as "./path/to/dir" or "/path/to/dir".
    # Must be extracted BEFORE the general /path extractor so that we preserve
    # the full path (e.g. "./src/agents/developer" → "/src/agents/developer").
    # Handles both "./dir" and "/dir" forms; always normalises to leading "/".
    dotslash_match = re.search(r"\./([a-zA-Z][a-zA-Z0-9_/\-]*)", text_no_url)
    if dotslash_match:
        return "/" + dotslash_match.group(1)

    # API path: /upload, /api/v1/:param — strip URL schemes first
    path_match = re.search(r"(/[a-zA-Z][a-zA-Z0-9_/\-:*]*)", text_no_url)
    if path_match:
        return path_match.group(1)

    # "jelaskan X" / "adakah X" directly — capture the first meaningful word(s)
    sym_match = re.search(
        rf"{_TRIGGER}\s+{_PREP}([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*){{0,2}})",
        user_input,
        re.IGNORECASE,
    )
    if sym_match:
        # Return up to 3 words but trim trailing stop-words
        raw = sym_match.group(sym_match.lastindex).strip()
        # Remove trailing prepositions / articles common in Indonesian
        raw = re.sub(
            r"\s+(?:di|pada|dalam|untuk|dari|yang|ini|itu|ke|dengan)\s*$",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        return raw

    return ""


# ── Internal helper ───────────────────────────────────────────────────────────

def _should_skip(rel_parts: tuple[str, ...]) -> bool:
    """Return True if any path component is in the skip-list."""
    return bool(set(rel_parts[:-1]) & _SKIP_DIRS)


def _read_snippet(path: Path, max_bytes: int = MAX_BYTES_PER_FILE) -> str:
    """Read a file and return a truncated snippet. Empty string on error."""
    try:
        return path.read_text(errors="replace")[:max_bytes]
    except OSError:
        return ""


def _format_file_snippet(rel_path: str, content: str, label: str = "") -> str:
    tag = f" ({label})" if label else ""
    return f"### 📄 `{rel_path}`{tag}\n```\n{content}\n```"


# ── Extractor: API Endpoints ──────────────────────────────────────────────────

# Patterns for route decorators across major frameworks
_ROUTE_PATTERNS = [
    # FastAPI / Flask / Starlette: @app.get("/path")
    r"@(?:app|router|api)\.(get|post|put|patch|delete|options|head|route)\s*\(\s*['\"]([^'\"]+)['\"]",
    # Django urls.py: path('...', view)
    r"(?:path|re_path|url)\s*\(\s*['\"]([^'\"]+)['\"]",
    # Express.js: app.get('/path', ...) or router.get('/path', ...)
    r"(?:app|router)\.(get|post|put|patch|delete|options|use)\s*\(\s*['\"]([^'\"]+)['\"]",
    # Laravel/Symphony style: @Route("/path")
    r"@Route\s*\(\s*['\"]([^'\"]+)['\"]",
    # Spring Boot: @RequestMapping / @GetMapping etc.
    r"@(?:Request|Get|Post|Put|Delete|Patch)Mapping\s*\(?['\"]?([^'\")\s]+)",

    # Go: support many common router patterns including gin, gorilla/mux, stdlib
    # Examples supported:
    #   r.HandleFunc("/path", handler)
    #   http.HandleFunc("/path", handler)
    #   router.Methods("GET").Path("/path")
    #   r.GET("/path", handler)   (gin)
    # Allow single/double quotes and raw backtick strings; allow whitespace/newlines.
    r"(?:\b(?:r|router|mux|http|engine|gin|e)\b)\.(?:GET|POST|PUT|PATCH|DELETE|OPTIONS|HandleFunc|Handle|GET)\s*\(\s*(?:['\"`])([^'\"`]*)['\"`]",
    r"HandleFunc\s*\(\s*(?:['\"`])([^'\"`]*)['\"`]\s*,",
    r"\.Methods\s*\(\s*['\"]?([A-Z,\s]+)['\"]?\s*\)\.Path\s*\(\s*(?:['\"`])([^'\"`]*)['\"`]\s*\)",
]
_ROUTE_RE = re.compile("|".join(_ROUTE_PATTERNS), re.MULTILINE | re.DOTALL)

_OPENAPI_FILES = {
    "openapi.yaml", "openapi.yml", "openapi.json",
    "swagger.yaml", "swagger.yml", "swagger.json",
    "api.yaml", "api.yml", "api.json",
}


async def extract_api_endpoints(repo_path: Path, candidate_filenames: list[str] | None = None) -> str:
    """
    Extract semua endpoint/rute HTTP dari repo.

    Strategi:
      1. Baca file OpenAPI/Swagger jika ada.
      2. Grep pola route decorator di semua source files.
      3. Baca file routing utama (urls.py, routes.py, router.ts, dll.).
    """
    sections: list[str] = []

    # 1. OpenAPI/Swagger spec
    for name in _OPENAPI_FILES:
        for fpath in repo_path.rglob(name):
            if _should_skip(fpath.relative_to(repo_path).parts):
                continue
            content = _read_snippet(fpath)
            if content:
                sections.append(_format_file_snippet(
                    str(fpath.relative_to(repo_path)), content, "OpenAPI/Swagger"
                ))
                break
        if sections:
            break

    # 2. Grep route patterns across source files
    findings: list[str] = []
    source_exts = {".py", ".js", ".ts", ".go", ".rb", ".php", ".java", ".cs"}
    files_scanned = 0
    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir() or fpath.suffix not in source_exts:
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            logger.debug("repo_qa: skipping file in skip-dir: %s", fpath)
            continue
        try:
            text = fpath.read_text(errors="replace")
        except OSError:
            logger.debug("repo_qa: could not read file %s (OSError)", fpath)
            continue
        matches = _ROUTE_RE.findall(text)
        if matches:
            rel = str(fpath.relative_to(repo_path))
            # Extract all decorated lines with line numbers
            route_lines = [
                f"  L{i+1}: {line.strip()}"
                for i, line in enumerate(text.splitlines())
                if re.search(r"@(?:app|router|api)\.(get|post|put|patch|delete)|"
                             r"(?:app|router)\.(get|post|put|patch|delete)|"
                             r"path\s*\(|re_path\s*\(|r\.(GET|POST|PUT|PATCH|DELETE)|"
                             r"@(?:Request|Get|Post|Put|Delete|Patch)Mapping|"
                             r"(?:router|r|mux|http)\.(?:GET|POST|PUT|PATCH|DELETE|HandleFunc|Handle)",
                             line, re.IGNORECASE)
            ][:30]
            if route_lines:
                findings.append(f"**`{rel}`**\n" + "\n".join(route_lines))
        files_scanned += 1

    if findings:
        sections.append("## Endpoint/Route yang Ditemukan\n\n" + "\n\n".join(findings))

    # 3. Route-specific files
    route_filenames = {
        # Common routing file names (add Go variants)
        "urls.py", "routes.py", "router.py", "routes.ts", "router.ts",
        "routes.js", "router.js", "api.py", "views.py", "controllers.py",
        "routes.rb", "web.php", "api.php",
        # Go routing files
        "routes.go", "router.go", "routing.go", "main.go",
    }
    # Allow caller to inject candidate filenames with higher priority
    def _route_filename_priority(candidates: set[str] | None) -> list[str]:
        if not candidates:
            return sorted(route_filenames)
        # Candidate filenames first, then defaults (deduped)
        ordered = []
        seen = set()
        for c in list(candidates):
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        for r in sorted(route_filenames):
            if r not in seen:
                ordered.append(r)
                seen.add(r)
        return ordered
    candidates_set = set(candidate_filenames) if candidate_filenames else None
    for name in _route_filename_priority(candidates_set):
        for fpath in sorted(repo_path.rglob(name)):
            if _should_skip(fpath.relative_to(repo_path).parts):
                logger.debug("repo_qa: skipping route file in skip-dir: %s", fpath)
                continue
            content = _read_snippet(fpath)
            if content:
                sections.append(_format_file_snippet(
                    str(fpath.relative_to(repo_path)), content, "routing file"
                ))

    if not sections:
        return "(tidak ditemukan definisi API/route di repositori ini)"

    return "\n\n".join(sections)


# ── Extractor: Tech Stack ─────────────────────────────────────────────────────

_MANIFEST_FILES = [
    ("requirements.txt", "Python deps"),
    ("pyproject.toml",   "Python project config"),
    ("setup.py",         "Python setup"),
    ("package.json",     "Node.js deps"),
    ("go.mod",           "Go modules"),
    ("Cargo.toml",       "Rust deps"),
    ("pom.xml",          "Java Maven"),
    ("build.gradle",     "Java Gradle"),
    ("composer.json",    "PHP Composer"),
    ("Gemfile",          "Ruby Gems"),
    ("mix.exs",          "Elixir Mix"),
    ("pubspec.yaml",     "Dart/Flutter"),
]

_INFRA_FILES = [
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example", ".env.sample",
]


async def extract_tech_stack(repo_path: Path) -> str:
    """Extract tech stack dari manifest, infra files, dan import patterns."""
    sections: list[str] = []

    # Manifest files
    for filename, label in _MANIFEST_FILES:
        fpath = repo_path / filename
        if fpath.exists():
            sections.append(_format_file_snippet(filename, _read_snippet(fpath), label))

    # Infrastructure files
    for filename in _INFRA_FILES:
        fpath = repo_path / filename
        if fpath.exists():
            sections.append(_format_file_snippet(filename, _read_snippet(fpath)))
        # Also check one level deep
        for sub in repo_path.iterdir():
            if sub.is_dir() and sub.name not in _SKIP_DIRS:
                fpath2 = sub / filename
                if fpath2.exists():
                    rel = str(fpath2.relative_to(repo_path))
                    sections.append(_format_file_snippet(rel, _read_snippet(fpath2)))

    # README for tech mentions
    for readme in ["README.md", "README.rst", "README.txt"]:
        fpath = repo_path / readme
        if fpath.exists():
            sections.append(_format_file_snippet(readme, _read_snippet(fpath, 5_000), "README"))
            break

    if not sections:
        return "(tidak ditemukan file manifest atau infra di repositori ini)"

    return "\n\n".join(sections)


# ── Extractor: Data Models ────────────────────────────────────────────────────

_MODEL_FILE_PATTERNS = [
    "models.py", "model.py", "models/*.py", "entities/*.py",
    "schemas.py", "schema.py", "types.py",
    "models/*.ts", "entities/*.ts", "schemas/*.ts",
    "models/*.go", "entities/*.go",
    "models/*.rb", "app/models/*.rb",
    "*.model.ts", "*.entity.ts", "*.schema.ts",
]

_MODEL_CLASS_RE = re.compile(
    r"(?:"
    r"class\s+(\w+)\s*\((?:BaseModel|Model|db\.Model|Document|Schema|Base)\)"  # Python ORM/Pydantic
    r"|@Entity\s*\n\s*(?:export\s+)?class\s+(\w+)"                              # TypeScript @Entity
    r"|type\s+(\w+)\s+struct\s*\{"                                               # Go struct
    r"|class\s+(\w+)\s+<\s*ApplicationRecord"                                    # Rails model
    r")",
    re.MULTILINE,
)


async def extract_data_models(repo_path: Path) -> str:
    """Extract definisi model data, ORM class, Pydantic schema, DB entity."""
    sections: list[str] = []
    seen: set[str] = set()

    # Scan for model files
    for fpath in sorted(repo_path.rglob("*.py")):
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        if "test" in fpath.name.lower() or "migration" in fpath.name.lower():
            continue
        content = _read_snippet(fpath)
        if _MODEL_CLASS_RE.search(content):
            rel = str(fpath.relative_to(repo_path))
            if rel not in seen:
                seen.add(rel)
                sections.append(_format_file_snippet(rel, content, "data model"))
                if len(seen) >= MAX_FILES_PER_TOPIC:
                    break

    # TypeScript / Go model files
    for ext in (".ts", ".go", ".java"):
        for fpath in sorted(repo_path.rglob(f"*{ext}")):
            if _should_skip(fpath.relative_to(repo_path).parts):
                continue
            if "test" in fpath.name.lower():
                continue
            content = _read_snippet(fpath)
            if _MODEL_CLASS_RE.search(content) or (
                ext == ".ts" and re.search(r"@Entity|@Column|@PrimaryColumn", content)
            ):
                rel = str(fpath.relative_to(repo_path))
                if rel not in seen:
                    seen.add(rel)
                    sections.append(_format_file_snippet(rel, content, f"entity{ext}"))
                    if len(seen) >= MAX_FILES_PER_TOPIC:
                        break

    # Alembic / Django migrations summary (just filenames, not content)
    migration_dirs = list(repo_path.rglob("migrations"))
    if migration_dirs:
        mig_list = "\n".join(
            f"  - {f.name}"
            for d in migration_dirs[:3]
            for f in sorted(d.iterdir())[:10]
            if f.is_file() and not f.name.startswith("__")
        )
        if mig_list:
            sections.append(f"## 🗃️ Migration Files\n{mig_list}")

    if not sections:
        return "(tidak ditemukan definisi model data di repositori ini)"

    return "\n\n".join(sections)


# ── Extractor: Dependencies ───────────────────────────────────────────────────

async def extract_dependencies(repo_path: Path) -> str:
    """Extract semua dependency manifest dari berbagai package manager."""
    sections: list[str] = []

    for filename, label in _MANIFEST_FILES:
        fpath = repo_path / filename
        if fpath.exists():
            sections.append(_format_file_snippet(filename, _read_snippet(fpath), label))

    # lock files (abbreviated)
    for lockfile in ["poetry.lock", "package-lock.json", "yarn.lock", "Pipfile.lock"]:
        fpath = repo_path / lockfile
        if fpath.exists():
            # Only show first 50 lines to avoid flooding
            try:
                lines = fpath.read_text(errors="replace").splitlines()[:50]
                sections.append(_format_file_snippet(
                    lockfile, "\n".join(lines) + "\n... (truncated)", "lock file"
                ))
            except OSError:
                pass

    if not sections:
        return "(tidak ditemukan file dependency manifest)"

    return "\n\n".join(sections)


# ── Extractor: CI/CD ─────────────────────────────────────────────────────────

_CICD_PATTERNS = [
    ".github/workflows",
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
    "Jenkinsfile",
    ".circleci/config.yml",
    ".travis.yml",
    "azure-pipelines.yml",
    "bitbucket-pipelines.yml",
    "cloudbuild.yaml",
    "cloudbuild.yml",
    ".drone.yml",
    "Makefile",
    "deploy.sh",
    "helper_deploy.sh",
    "start.sh",
    "scripts/deploy.sh",
    "scripts/ci.sh",
]


async def extract_ci_cd(repo_path: Path) -> str:
    """Extract pipeline/CI-CD config, Dockerfile, dan deploy scripts."""
    sections: list[str] = []

    for pattern in _CICD_PATTERNS:
        fpath = repo_path / pattern
        if fpath.is_file():
            sections.append(_format_file_snippet(pattern, _read_snippet(fpath), "CI/CD"))
        elif fpath.is_dir():
            # GitHub Actions workflows
            for wf in sorted(fpath.glob("*.yml"))[:5]:
                rel = str(wf.relative_to(repo_path))
                sections.append(_format_file_snippet(rel, _read_snippet(wf), "GitHub Actions workflow"))

    # Dockerfiles
    for fpath in sorted(repo_path.rglob("Dockerfile*"))[:4]:
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        rel = str(fpath.relative_to(repo_path))
        sections.append(_format_file_snippet(rel, _read_snippet(fpath), "Dockerfile"))

    if not sections:
        return "(tidak ditemukan konfigurasi CI/CD, pipeline, atau Dockerfile)"

    return "\n\n".join(sections)


# ── Extractor: Security ───────────────────────────────────────────────────────

_AUTH_PATTERNS = re.compile(
    r"(?:"
    r"jwt|oauth|bearer|api[_\-]key|secret[_\-]key|password|authentication"
    r"|authorization|middleware.*auth|auth.*middleware|permission|role"
    r"|session.*secret|csrf|cors|rate.?limit|throttl"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_ENV_VAR_RE = re.compile(r"(?:os\.(?:getenv|environ)|process\.env|sys\.argv)\[?['\"]?([A-Z_]{3,})")


async def extract_security(repo_path: Path) -> str:
    """
    Extract pola keamanan: auth middleware, JWT, OAuth, env vars sensitif,
    CORS, rate limiting.
    """
    sections: list[str] = []

    # .env.example / .env.sample
    for envfile in [".env.example", ".env.sample", ".env.template"]:
        fpath = repo_path / envfile
        if fpath.exists():
            sections.append(_format_file_snippet(envfile, _read_snippet(fpath), "env template"))

    # Grep for auth patterns in source files
    findings: list[str] = []
    source_exts = {".py", ".js", ".ts", ".go", ".java", ".php"}
    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir() or fpath.suffix not in source_exts:
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        try:
            text = fpath.read_text(errors="replace")
        except OSError:
            continue

        auth_lines = [
            f"  L{i+1}: {line.strip()}"
            for i, line in enumerate(text.splitlines())
            if _AUTH_PATTERNS.search(line) and len(line.strip()) < 200
        ][:15]

        if auth_lines:
            rel = str(fpath.relative_to(repo_path))
            findings.append(f"**`{rel}`** — auth/security references:\n" + "\n".join(auth_lines))

        if len(findings) >= MAX_FILES_PER_TOPIC:
            break

    if findings:
        sections.append("## 🔐 Auth & Security References\n\n" + "\n\n".join(findings))

    if not sections:
        return "(tidak ditemukan pola keamanan/auth di repositori ini)"

    return "\n\n".join(sections)


# ── Extractor: Main Flow ──────────────────────────────────────────────────────

_ENTRY_POINTS = [
    "main.py", "app.py", "server.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "server.js", "server.ts",
    "main.go", "cmd/main.go",
    "main.rb", "config.ru",
    "Program.cs",
]

_LIFECYCLE_KEYWORDS = re.compile(
    r"(?:startup|lifespan|on_start|on_shutdown|middleware|include_router"
    r"|app\.use|app\.listen|http\.ListenAndServe|gin\.Default"
    r"|@app\.on_event|asynccontextmanager|create_app|make_app)",
    re.IGNORECASE,
)


async def extract_main_flow(repo_path: Path) -> str:
    """
    Extract alur utama aplikasi: entry points, middleware registration,
    startup sequence, dan request lifecycle.
    """
    sections: list[str] = []

    # Entry point files
    for filename in _ENTRY_POINTS:
        fpath = repo_path / filename
        if fpath.exists():
            sections.append(_format_file_snippet(filename, _read_snippet(fpath), "entry point"))

    # Files with lifecycle/startup keywords — scan Python and Go sources
    seen: set[str] = set()
    _already_handled = {ep for ep in _ENTRY_POINTS}
    for ext, label in ((".py", "startup/lifecycle"), (".go", "Go startup/lifecycle")):
        for fpath in sorted(repo_path.rglob(f"*{ext}")):
            if _should_skip(fpath.relative_to(repo_path).parts):
                continue
            if fpath.name in _already_handled:
                continue  # already handled as entry point above
            content = _read_snippet(fpath)
            if _LIFECYCLE_KEYWORDS.search(content):
                rel = str(fpath.relative_to(repo_path))
                if rel not in seen:
                    seen.add(rel)
                    sections.append(_format_file_snippet(rel, content, label))
                    if len(seen) >= MAX_FILES_PER_TOPIC - 1:
                        break

    if not sections:
        return "(tidak ditemukan entry point atau lifecycle hooks yang jelas)"


    return "\n\n".join(sections)


# ── Filename-target helpers ────────────────────────────────────────────────────

# Source-code and config extensions that can appear as filename targets
# (e.g. "main.py", "config.go", "api/routes.ts").
_FILENAME_EXTS = {
    ".py", ".go", ".js", ".ts", ".jsx", ".tsx", ".java", ".php", ".rb",
    ".rs", ".kt", ".cs", ".c", ".cpp", ".h", ".hpp",
    ".yaml", ".yml", ".json", ".toml", ".env", ".sh", ".md",
    ".txt", ".cfg", ".ini", ".sql", ".html", ".css", ".scss",
    ".proto", ".vue", ".svelte", ".mjs", ".cjs",
}

# Indonesian/English stopwords to skip when building keyword lists
_STOPWORDS = {
    # Indonesian
    "cari", "implementasi", "pada", "file", "di", "dalam", "dan", "yang",
    "ada", "apa", "berikan", "tampilkan", "tunjukkan", "list", "daftar",
    "bagaimana", "cara", "kerja", "fungsi", "class", "modul", "module",
    "repo", "repositori", "kode", "semua", "setiap", "untuk", "dari",
    "ini", "itu", "bisa", "boleh", "tolong", "jelaskan", "lihat",
    # English
    "find", "search", "show", "list", "give", "get", "look", "check",
    "implementation", "of", "in", "at", "the", "a", "an", "is", "are",
    "on", "for", "from", "with", "how", "what", "where", "all",
}


def _is_filename_target(target: str) -> bool:
    """Return True if *target* looks like a file path (has a recognised extension)."""
    ext = Path(target).suffix.lower()
    return ext in _FILENAME_EXTS


def _extract_search_keywords(user_input: str, exclude_token: str = "") -> list[str]:
    """
    Extract meaningful search keywords from *user_input*, excluding the
    filename token and common stopwords.

    Returns a list of lowercase keyword strings ordered by length (longest
    first so more specific multi-word patterns are tried first).

    Examples:
      "cari implementasi elastic apm pada file main.py"
        → ["elastic apm", "elastic", "apm"]
      "bagaimana konfigurasi jwt di middleware.py"
        → ["jwt", "konfigurasi"]
    """
    # Remove URLs
    text = re.sub(r"https?://\S+", "", user_input)
    # Remove the filename token itself (e.g. "main.py")
    if exclude_token:
        text = re.sub(re.escape(exclude_token), " ", text, flags=re.IGNORECASE)
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    words = [lw for w in text.split() if len(w) >= 2 and (lw := w.lower()) not in _STOPWORDS]

    # Build candidates: individual words + consecutive bigrams
    candidates: list[str] = []
    for i, w in enumerate(words):
        candidates.append(w)
        if i + 1 < len(words):
            candidates.append(f"{w} {words[i + 1]}")

    # Deduplicate, sort longest first (prefer multi-word patterns)
    seen: set[str] = set()
    result: list[str] = []
    for c in sorted(candidates, key=len, reverse=True):
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


async def _search_keyword_in_file(
    repo_path: Path,
    filename_target: str,
    user_input: str = "",
) -> str:
    """
    Locate *filename_target* in *repo_path* and grep for keywords extracted
    from *user_input* within that file, returning matching line ranges.

    If no keywords are found in the query (or no matches in the file), fall
    back to returning the full file content (truncated to MAX_BYTES_PER_FILE).

    Supports partial path matching: "main.py" matches "api/main.py".
    """
    # ── 1. Locate the file ──────────────────────────────────────────────────
    target_norm = filename_target.replace("\\", "/").lstrip("./")
    target_name = Path(target_norm).name  # e.g. "main.py"

    matches: list[Path] = []
    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir():
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        rel = fpath.relative_to(repo_path).as_posix()
        # Prefer exact suffix match (handles "api/main.py" vs just "main.py")
        if rel.endswith(target_norm) or fpath.name == target_name:
            matches.append(fpath)

    if not matches:
        # ── Fuzzy fallback: match stem keywords across the repo tree ──────────
        # Handles flattened filenames like "developer_qna_agent.py" that map to
        # an actual path "src/agents/developer_qna/agent.py" — all stem tokens
        # ("developer", "qna", "agent") appear somewhere in the real path.
        stem = Path(target_name).stem                              # e.g. "developer_qna_agent"
        stem_kws = [kw.lower() for kw in re.split(r"[_\-]+", stem) if len(kw) >= 3]
        if stem_kws:
            for fpath in sorted(repo_path.rglob("*")):
                if fpath.is_dir() or _should_skip(fpath.relative_to(repo_path).parts):
                    continue
                rel_lower = fpath.relative_to(repo_path).as_posix().lower()
                if all(kw in rel_lower for kw in stem_kws):
                    matches.append(fpath)
        if not matches:
            return f"(file `{filename_target}` tidak ditemukan di repositori)"
        logger.info(
            "_search_keyword_in_file: exact match failed for %r — "
            "fuzzy stem match found %d candidate(s): %s",
            filename_target, len(matches),
            [str(m.relative_to(repo_path)) for m in matches[:3]],
        )

    # Sort: prefer path that ends with the full target (most specific first),
    # then files whose stem exactly matches one of the filename stem keywords
    # (handles fuzzy matches where "agent.py" should rank above "HOW_IT_WORKS.md"),
    # then alphabetically.
    _stem_kws_sort = [kw.lower() for kw in re.split(r"[_\-]+", Path(target_name).stem) if len(kw) >= 3]
    matches.sort(key=lambda p: (
        0 if p.relative_to(repo_path).as_posix().endswith(target_norm) else
        1 if p.stem.lower() in _stem_kws_sort else
        2,
        str(p),
    ))
    fpath = matches[0]
    rel_path = fpath.relative_to(repo_path).as_posix()

    try:
        file_text = fpath.read_text(errors="replace")
        file_lines = file_text.splitlines()
    except OSError as exc:
        return f"(gagal membaca `{rel_path}`: {exc})"

    # ── 2. Detect "list all functions/methods" requests ────────────────────
    # When the user asks for a list of ALL functions/methods/classes in a file,
    # skip keyword search and return full file content so the LLM can enumerate
    # every definition — not just the 3 lines where "function" appears in a comment.
    _LIST_DEFS_RE = re.compile(
        r"(?:list|daftar|semua|tampilkan|berikan|tunjukkan|lihat)\s+"
        r"(?:list\s+|daftar\s+|semua\s+)?"
        r"(?:fungsi|function|functions|method|methods|class|classes|def|"
        r"semua\s+fungsi|all\s+function)",
        re.IGNORECASE,
    )
    if user_input and _LIST_DEFS_RE.search(user_input):
        logger.info(
            "_search_keyword_in_file: 'list function' intent detected — "
            "returning full file content for %r",
            rel_path,
        )
        content = file_text[:MAX_BYTES_PER_FILE]
        ext = fpath.suffix.lstrip(".")
        return (
            f"### 📄 `{rel_path}` (full content — list all definitions)\n"
            f"```{ext}\n{content}\n```"
        )

    # ── 3. Extract keywords from the user query ─────────────────────────────
    keywords = _extract_search_keywords(user_input, exclude_token=filename_target)
    logger.info(
        "_search_keyword_in_file: file=%r keywords=%r", rel_path, keywords[:5]
    )

    # ── 4. Grep for each keyword, collect matching line ranges ─────────────
    _CONTEXT_LINES = 5  # lines of context around each match
    hit_line_indices: set[int] = set()

    for kw in keywords:
        kw_lower = kw.lower()
        # Also try compact form: "elastic apm" → "elasticapm"
        kw_compact = kw_lower.replace(" ", "")
        for i, line in enumerate(file_lines):
            line_lower = line.lower()
            if kw_lower in line_lower or (kw_compact and kw_compact != kw_lower and kw_compact in line_lower):
                hit_line_indices.add(i)

        if hit_line_indices:
            # Found matches for this keyword – stop searching less-specific ones
            break

    if hit_line_indices:
        # Expand each hit with context and merge overlapping ranges
        ranges: list[tuple[int, int]] = []
        for idx in sorted(hit_line_indices):
            start = max(0, idx - _CONTEXT_LINES)
            end   = min(len(file_lines), idx + _CONTEXT_LINES + 1)
            ranges.append((start, end))

        # Merge overlapping/adjacent ranges
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        snippet_parts: list[str] = []
        for start, end in merged:
            block = []
            for j in range(start, end):
                marker = "→" if j in hit_line_indices else " "
                block.append(f"  {marker} L{j + 1}: {file_lines[j]}")
            snippet_parts.append("\n".join(block))

        kw_used = keywords[0] if keywords else ""
        total_lines = len(file_lines)
        header = (
            f"### 🔍 `{rel_path}` — keyword: `{kw_used}` "
            f"({len(hit_line_indices)} hit dari {total_lines} baris)\n"
        )
        ext = fpath.suffix.lstrip(".")
        body = f"\n\n---\n\n".join(
            f"```{ext}\n{part}\n```" for part in snippet_parts
        )
        return header + body

    # ── 5. No keyword hits – fall back to full file content ─────────────────
    logger.info(
        "_search_keyword_in_file: no keyword hits in %r, returning full content", rel_path
    )
    content = file_text[:MAX_BYTES_PER_FILE]
    ext = fpath.suffix.lstrip(".")
    return (
        f"### 📄 `{rel_path}` (full content — keyword tidak ditemukan)\n"
        f"```{ext}\n{content}\n```"
    )


# ── Config / data file fallback search ───────────────────────────────────────

# File extensions for configuration / data files searched as fallback.
_CONFIG_EXTS = {
    ".json", ".yaml", ".yml", ".toml", ".env", ".xml",
    ".ini", ".cfg", ".conf", ".properties",
}

# Source + config extensions that are meaningful when reading a whole directory.
_DIR_SOURCE_EXTS = {
    ".py", ".go", ".js", ".ts", ".jsx", ".tsx", ".java", ".php",
    ".rb", ".rs", ".kt", ".cs", ".yaml", ".yml", ".json", ".toml",
    ".env", ".sh", ".md",
}

# Signals returned by extractors that indicate no useful evidence was found.
_EMPTY_EVIDENCE_SIGNALS = (
    "(simbol",
    "(path ",
    "(target tidak",
    "tidak ditemukan",
    "not found",
    "(file `",
)


def _evidence_is_empty(text: str) -> bool:
    """Return True when *text* only contains an 'evidence not found' sentinel."""
    stripped = text.strip()
    if not stripped:
        return True
    return any(stripped.startswith(s) for s in _EMPTY_EVIDENCE_SIGNALS)


# ── Directory target helpers ───────────────────────────────────────────────────

def _is_directory_target(repo_path: Path, target: str) -> bool:
    """Return True if *target* (possibly prefixed with '/' or './') resolves to
    an existing directory inside *repo_path*."""
    normalized = target.lstrip(".").strip("/")
    return bool(normalized) and (repo_path / normalized).is_dir()


async def _scan_directory_files(
    repo_path: Path,
    dir_path: str,
    user_input: str = "",
    *,
    max_files: int = 12,
) -> str:
    """
    Read all source files inside *dir_path* (relative to *repo_path*) and
    return their contents formatted as evidence for the LLM.

    Called when the user explicitly targets a directory such as
    "./src/agents/developer" — instead of guessing which single file is
    relevant, we read ALL meaningful files in the directory.

    Returns a markdown-formatted string, or a sentinel if no files found.
    """
    normalized = dir_path.lstrip(".").strip("/")
    target_dir = repo_path / normalized

    if not target_dir.is_dir():
        return f"(direktori `{normalized}` tidak ditemukan di repositori)"

    sections: list[str] = []
    for fpath in sorted(target_dir.rglob("*")):
        if fpath.is_dir():
            continue
        if fpath.suffix.lower() not in _DIR_SOURCE_EXTS:
            continue
        rel_parts = fpath.relative_to(repo_path).parts
        if _should_skip(rel_parts):
            continue
        rel = fpath.relative_to(repo_path).as_posix()
        content = _read_snippet(fpath)
        ext = fpath.suffix.lstrip(".")
        sections.append(f"### 📄 `{rel}`\n```{ext}\n{content}\n```")
        if len(sections) >= max_files:
            break

    if not sections:
        return f"(tidak ada file sumber yang ditemukan di direktori `{normalized}`)"

    logger.info(
        "_scan_directory_files: read %d file(s) from directory %r",
        len(sections), normalized,
    )
    return (
        f"## 📁 Isi Direktori: `{normalized}`\n\n"
        + "\n\n".join(sections)
    )


async def _search_config_files_for_keyword(
    repo_path: Path,
    keyword: str,
    user_input: str = "",
    *,
    max_files: int = 6,
    context_lines: int = 8,
) -> str:
    """
    Search for *keyword* in configuration and data files (.json, .yaml, .yml,
    .toml, .env, .xml, etc.) that are normally skipped by source-code extractors.

    Used as a fallback when `extract_specific_symbol()` returns no evidence from
    source code.  This lets agents answer questions about Postman collections,
    OpenAPI specs, Docker Compose, CI pipeline YAML, and similar artifacts.

    Returns a markdown-formatted evidence string, or an informative sentinel if
    nothing was found.
    """
    if not keyword:
        return "(tidak ada keyword untuk dicari di file konfigurasi)"

    kw_lower = keyword.lower()
    kw_compact = kw_lower.replace(" ", "").replace("_", "").replace("-", "")

    # Also derive additional keywords from user_input
    extra_kws: list[str] = []
    if user_input:
        extra_kws = _extract_search_keywords(user_input, exclude_token=keyword)[:5]

    findings: list[str] = []

    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir() or fpath.suffix.lower() not in _CONFIG_EXTS:
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue

        try:
            text = fpath.read_text(errors="replace")
        except OSError:
            continue

        file_lines = text.splitlines()
        hit_indices: set[int] = set()
        matched_kw = ""

        # Check primary keyword first, then extras
        for kw in [kw_lower] + extra_kws:
            kw_norm = kw.lower()
            kw_c = kw_norm.replace(" ", "").replace("_", "").replace("-", "")
            for i, line in enumerate(file_lines):
                line_lower = line.lower()
                if kw_norm in line_lower or (kw_c and kw_c != kw_norm and kw_c in line_lower):
                    hit_indices.add(i)
            if hit_indices:
                matched_kw = kw
                break

        if not hit_indices:
            continue

        rel = fpath.relative_to(repo_path).as_posix()
        ext = fpath.suffix.lstrip(".")

        # Expand hits with context and merge overlapping ranges
        ranges: list[tuple[int, int]] = []
        for idx in sorted(hit_indices)[:20]:  # cap hits per file
            start = max(0, idx - context_lines)
            end = min(len(file_lines), idx + context_lines + 1)
            ranges.append((start, end))

        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        snippet_parts: list[str] = []
        for start, end in merged[:5]:  # cap output blocks per file
            block = []
            for j in range(start, end):
                marker = "→" if j in hit_indices else " "
                block.append(f"  {marker} L{j + 1}: {file_lines[j].rstrip()}")
            snippet_parts.append("\n".join(block))

        body = "\n\n---\n\n".join(f"```{ext}\n{p}\n```" for p in snippet_parts)
        findings.append(
            f"**`{rel}`** — keyword: `{matched_kw}` ({len(hit_indices)} hit):\n{body}"
        )

        if len(findings) >= max_files:
            break

    if not findings:
        return (
            f"(tidak ditemukan referensi `{keyword}` di file konfigurasi/data "
            f"(.json, .yaml, .toml, dll.) di repositori ini)"
        )

    logger.info(
        "_search_config_files_for_keyword: found %d config file(s) for keyword=%r",
        len(findings), keyword,
    )
    return (
        f"## 📋 Konfigurasi & Data Files — `{keyword}`\n\n"
        + "\n\n".join(findings)
    )


# ── Extractor: Specific Symbol ────────────────────────────────────────────────

async def extract_specific_symbol(
    repo_path: Path, target: str, user_input: str = ""
) -> str:
    """
    Temukan definisi + penggunaan simbol tertentu (API path, fungsi, class).

    Untuk API path (target starts with "/"): lakukan 2-phase trace:
      Phase 1 – Route registration: cari file routing, temukan baris registrasi
                route yang sesuai, ekstrak nama handler function.
      Phase 2 – Handler implementation: baca body lengkap handler (bukan hanya
                snippet), lalu cari 1 level fungsi yang dipanggil handler tersebut.

    Untuk symbol biasa (fungsi/class): cari definisi dengan regex, tampilkan
    body lengkap.

    Untuk qualified name (pkg.FunctionName / controllers.DownloadFile): split
    dan prioritaskan file di dalam direktori yang sesuai. Jika user_input juga
    mengandung API path eksplisit (misal "GET /appuuid/:uuid → controllers.DownloadFile"),
    trace KEDUANYA secara paralel sehingga LLM menerima: registrasi route +
    full handler body + definisi fungsi yang dirujuk.

    Untuk filename target (e.g. "main.py"): cari file di repo, lalu grep untuk
    keyword yang disebutkan dalam user_input (e.g. "elastic apm") di dalam file
    tersebut, mengembalikan baris-baris yang relevan dengan konteks.

    Untuk directory target (e.g. "./src/agents/developer"): baca SEMUA file
    sumber di dalam direktori tersebut dan kembalikan isinya sebagai evidence.

    Fallback: jika source-code search tidak menemukan hasil, secara otomatis
    cari di file konfigurasi/data (.json, .yaml, .toml, dll.) agar pertanyaan
    tentang Postman collection, OpenAPI spec, CI/CD config, dsb. tetap terjawab.
    """
    if not target:
        return "(target tidak ditentukan)"

    if target.startswith("/"):
        # Check if the path is a directory inside the repo BEFORE treating it
        # as an API route — e.g. "./src/agents/developer" → read directory files.
        if _is_directory_target(repo_path, target):
            logger.info(
                "extract_specific_symbol: directory target detected → %r", target
            )
            return await _scan_directory_files(repo_path, target, user_input)

        result = await _trace_api_route(repo_path, target)
        # If route tracing found nothing in source code, also search config files
        if _evidence_is_empty(result):
            logger.info(
                "extract_specific_symbol: route %r not found in source — "
                "falling back to config file search",
                target,
            )
            return await _extract_symbol_with_config_fallback(
                repo_path, target.lstrip("/"), user_input, result
            )
        return result

    # Filename target: e.g. "main.py", "api/config.py", "worker.go"
    # Must be checked BEFORE the qualified-name splitter so that "main.py" is
    # NOT misinterpreted as package "main" + symbol "py".
    if _is_filename_target(target):
        logger.info("extract_specific_symbol: file target detected → %r", target)
        result = await _search_keyword_in_file(repo_path, target, user_input)
        if _evidence_is_empty(result):
            logger.info(
                "extract_specific_symbol: file target %r not found — "
                "falling back to config file search",
                target,
            )
            return await _extract_symbol_with_config_fallback(
                repo_path, Path(target).stem, user_input, result
            )
        return result

    # Qualified name: controllers.DownloadFile → search in controllers/ first
    if "." in target and not target.startswith("."):
        parts = target.rsplit(".", 1)
        package_hint = parts[0]   # e.g. "controllers"
        func_name    = parts[1]   # e.g. "DownloadFile"

        # If the raw question also contains an explicit parameterised API path
        # (e.g. "GET /appuuid/:uuid/:processoption/:outputtype → controllers.DownloadFile"),
        # trace BOTH the route AND the symbol definition in parallel so the LLM
        # receives full context: where the route is registered + full handler body.
        text_no_url = re.sub(r"https?://\S+", "", user_input) if user_input else ""
        api_path_match = re.search(
            r"/[a-zA-Z][a-zA-Z0-9_/\-]*(?:/:[a-zA-Z][a-zA-Z0-9_]*)+",
            text_no_url,
        ) if text_no_url else None

        if api_path_match:
            api_path = api_path_match.group(0)
            logger.info(
                "extract_specific_symbol: dual-trace → route=%r + symbol=%r (pkg=%r)",
                api_path, func_name, package_hint,
            )
            route_result, sym_result = await asyncio.gather(
                _trace_api_route(repo_path, api_path),
                _find_symbol_definition(repo_path, func_name, package_hint=package_hint),
            )
            parts_out: list[str] = []
            if route_result and "tidak ditemukan" not in route_result:
                parts_out.append(route_result)
            if sym_result and "tidak ditemukan" not in sym_result:
                parts_out.append(sym_result)
            # Include both even if only one succeeded (so LLM can at least report
            # what was found and what was missing)
            if not parts_out:
                parts_out = [r for r in (route_result, sym_result) if r]
            combined = (
                "\n\n".join(parts_out)
                if parts_out
                else f"(tidak ditemukan: route `{api_path}` maupun simbol `{target}`)"
            )
            # Fallback: if nothing found in source, search config files too
            if _evidence_is_empty(combined):
                logger.info(
                    "extract_specific_symbol: dual-trace found nothing — "
                    "falling back to config file search for %r",
                    target,
                )
                return await _extract_symbol_with_config_fallback(
                    repo_path, func_name, user_input, combined
                )
            return combined

        src_result = await _find_symbol_definition(
            repo_path, func_name, package_hint=package_hint
        )
        if _evidence_is_empty(src_result):
            logger.info(
                "extract_specific_symbol: qualified symbol %r not found in source — "
                "falling back to config file search",
                target,
            )
            return await _extract_symbol_with_config_fallback(
                repo_path, func_name, user_input, src_result
            )
        return src_result

    src_result = await _find_symbol_definition(repo_path, target)
    if _evidence_is_empty(src_result):
        logger.info(
            "extract_specific_symbol: symbol %r not found in source — "
            "falling back to config file search",
            target,
        )
        return await _extract_symbol_with_config_fallback(
            repo_path, target, user_input, src_result
        )
    return src_result


async def _extract_symbol_with_config_fallback(
    repo_path: Path,
    target: str,
    user_input: str,
    source_result: str,
) -> str:
    """
    When source-code extraction for *target* returns insufficient evidence,
    fall back to searching configuration and data files (.json, .yaml, etc.).

    The original *source_result* is prepended (if non-empty) so the LLM still
    sees any partial source-code context alongside the config-file findings.
    """
    config_result = await _search_config_files_for_keyword(
        repo_path, target, user_input
    )
    parts: list[str] = []
    if source_result and not any(source_result.startswith(s) for s in _EMPTY_EVIDENCE_SIGNALS):
        parts.append(source_result)
    if config_result and not config_result.startswith("(tidak ditemukan"):
        parts.append(config_result)
    return "\n\n".join(parts) if parts else source_result



_ROUTE_FILE_NAMES = {
    "routes.go", "router.go", "routing.go", "main.go",
    "urls.py", "routes.py", "router.py",
    "routes.ts", "router.ts", "routes.js", "router.js",
    "api.php", "web.php", "routes.rb",
}

# ── Downstream function call extraction ───────────────────────────────────────

# Identifiers to skip when extracting called function names from handler bodies.
_FUNC_CALL_SKIP: frozenset = frozenset({
    "Sprintf", "Printf", "Println", "Fprintf", "Errorf", "Fatalf", "Panicf",
    "Marshal", "Unmarshal", "MarshalJSON", "UnmarshalJSON",
    "Error", "String", "Len", "Cap",
    "Background", "WithCancel", "WithTimeout", "WithDeadline", "WithValue", "TODO",
    "StatusOK", "StatusBadRequest", "StatusNotFound", "StatusCreated",
    "StatusInternalServerError", "StatusUnauthorized", "StatusForbidden",
    "StatusConflict", "StatusNoContent", "StatusAccepted",
    "JSON", "XML", "HTML", "Data", "File", "Redirect", "Stream",
    "Bind", "BindJSON", "ShouldBind", "ShouldBindJSON", "ShouldBindQuery",
    "Abort", "AbortWithError", "AbortWithStatus", "AbortWithStatusJSON",
    "Param", "Query", "Header", "Cookie", "PostForm", "FormFile",
    "Set", "Keys", "Next", "Done", "Deadline",
    "Begin", "Commit", "Rollback", "Save", "Close", "Open",
    "Exec", "QueryRow", "Prepare", "Scan",
    "Info", "Debug", "Warn", "Warning", "Fatal", "Panic",
    "GetLogger", "WithField", "WithFields", "WithError",
    "New", "Init", "Main", "Make", "Append", "Copy", "Delete",
    "Parse", "Format", "Convert", "Wrap", "Handle",
    "Read", "Write", "Flush", "Reset",
    "Depends", "Response", "HTTPException", "Request",
    "JSONResponse", "StreamingResponse", "RedirectResponse",
    "APIRouter", "FastAPI", "Middleware",
})

# Maximum downstream function definitions to include in Phase 3.
_MAX_DOWNSTREAM_FUNCS = 8


def _extract_called_functions(body: str) -> list:
    """Extract unique user-defined function names called within *body*.

    Filters stdlib/framework names via _FUNC_CALL_SKIP.
    Returns names in first-occurrence order, capped at 20.
    """
    hits: list = []
    # Qualified calls: obj.Method(
    for m in re.finditer(r'\b[a-zA-Z_]\w*\.([A-Za-z_][A-Za-z0-9_]+)\s*\(', body):
        name = m.group(1)
        if name not in _FUNC_CALL_SKIP and len(name) >= 4:
            hits.append(name)
    # Standalone PascalCase calls (user-defined Go/Java functions)
    for m in re.finditer(r'(?<![.\w])([A-Z][a-zA-Z0-9]{3,})\s*\(', body):
        name = m.group(1)
        if name not in _FUNC_CALL_SKIP:
            hits.append(name)
    seen: set = set()
    unique: list = []
    for name in hits:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique[:20]




def _build_path_search_terms(api_path: str) -> list[str]:
    """
    Build a list of search terms (from most specific to least) for an API path.

    "/download/:appuuid/:uuid/:processoption/:outputtype"
    → ["/download/:appuuid/:uuid/:processoption/:outputtype",
       ":processoption/:outputtype",
       "processoption",
       "outputtype",
       "download"]
    """
    terms: list[str] = []
    # 1. Exact path (most specific)
    terms.append(api_path)
    # 2. Unique param names (path params are often unique to this endpoint)
    params = re.findall(r":([a-zA-Z0-9_]+)", api_path)
    # pairs of consecutive params help disambiguate
    for i in range(len(params) - 1):
        terms.append(f":{params[i]}/:{params[i+1]}")
    # 3. Individual param names (less specific but useful fallback)
    for p in params:
        terms.append(p)
    # 4. First non-param path segment (anchor for group-based routers)
    segs = [s for s in api_path.split("/") if s and not s.startswith(":")]
    if segs:
        terms.append(segs[0])
    return terms


def _extract_handler_from_line(line: str) -> list[str]:
    """
    Try to extract handler function reference(s) from a route registration line.

    Handles patterns like:
      r.GET("/path", ctrl.Method)
      r.GET("/path", ctrl.Method, middleware1)
      router.HandleFunc("/path", handleFunc)
      @app.get("/path")  → next function is the handler (Python), caller must handle
      route("/path", handler)
    Returns a list of candidate identifiers (may include obj.Method and just Method).
    """
    candidates: list[str] = []
    # Match all word.word or word tokens after the first string argument
    # Typical: r.GET("...", ctrl.Method) or r.GET("...", handlerFunc, middleware)
    # Skip the first string literal, then collect function references
    after_path = re.sub(r"""^[^,)]*,\s*(?:["'`][^"'`]*["'`]\s*,\s*)""", "", line)
    for m in re.finditer(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*[,)]", after_path):
        ref = m.group(1)
        # Skip common non-handler tokens
        if ref.lower() in {"true", "false", "nil", "null", "none"}:
            continue
        candidates.append(ref)
        # Also add just the method part
        if "." in ref:
            candidates.append(ref.split(".")[-1])
    return candidates


def _read_full_function_body(lines: list[str], start_line: int, lang_ext: str) -> str:
    """
    Read the COMPLETE function/method body starting at start_line (0-indexed).

    No line cap is applied here — the caller is responsible for size reduction
    (e.g. via _compress_handler_body) and/or a character cap afterwards.
    Reading the full body is important so that business logic later in the
    function (error handling, response building, downstream calls) is not lost.

    Supports:
      - Go / Java / JS / TS / C#: brace counting ({ ... })
      - Python: indentation-based
    Hard ceiling: 2000 lines as a safety guard against runaway files.
    """
    _HARD_CEILING = 2000

    if not lines:
        return ""

    body_lines = [lines[start_line]]

    if lang_ext == ".py":
        # Python: collect lines with deeper indentation than the def line
        def_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        for i in range(start_line + 1, min(start_line + _HARD_CEILING, len(lines))):
            line = lines[i]
            stripped = line.rstrip()
            if not stripped:
                body_lines.append(line)
                continue
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent <= def_indent and line.lstrip():
                break
            body_lines.append(line)
    else:
        # Brace-based languages: count { and } to find end of function
        open_braces = lines[start_line].count("{") - lines[start_line].count("}")
        i = start_line + 1
        while i < len(lines) and i < start_line + _HARD_CEILING:
            line = lines[i]
            body_lines.append(line)
            open_braces += line.count("{") - line.count("}")
            if open_braces <= 0:
                break
            i += 1

    return "\n".join(body_lines)


# ── Semantic compression of handler bodies ────────────────────────────────────

# Detects single-line field assignments:  respDetail.DocUuid = doc.Uuid
_FIELD_ASSIGN_RE = re.compile(r'^\s+(\w+)\.(\w+)\s*=\s*\w+\.\w+\s*$')
# Detects struct-literal field lines:     Uuid:  kf.Uuid,
_STRUCT_FIELD_RE = re.compile(r'^\s+([A-Za-z_]\w*)\s*:\s*\w+\.\w+\s*,?\s*$')
_FIELD_COLLAPSE_MIN = 4   # collapse if this many consecutive matching lines


def _compress_handler_body(body: str, max_chars: int = 12_000) -> str:
    """
    Collapse repetitive field-assignment and struct-literal-field blocks
    into a single summary comment, then apply a character cap.

    Pipeline:
      1. Receive FULL function body (no line cap — caller reads everything).
      2. Collapse consecutive field-assignment lines into one summary comment.
      3. If compressed result still exceeds max_chars, truncate with a marker.

    This ensures the LLM always sees:
      - Full business logic (validations, DB queries, error paths, calls)
      - Field mapping summarised as "// [FIELD MAPPING: N fields → a, b, c...]"
      - Never more than max_chars of raw text

    Transforms:
        respDetail.DocUuid = doc.Uuid
        respDetail.DocName = doc.DocName
        ...  (40 more lines)
    Into:
        // [FIELD MAPPING: 42 fields → DocUuid, DocName, DocType, ... +39 more]
    """
    lines  = body.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        # Try to accumulate a consecutive block of field-like lines
        block: list[str] = []
        names: list[str] = []

        j = i
        while j < len(lines):
            line = lines[j]
            m_a = _FIELD_ASSIGN_RE.match(line)
            m_s = _STRUCT_FIELD_RE.match(line)
            if m_a:
                block.append(line)
                names.append(m_a.group(2))   # dest field name
                j += 1
            elif m_s:
                block.append(line)
                names.append(m_s.group(1))   # struct key name
                j += 1
            else:
                break

        if len(block) >= _FIELD_COLLAPSE_MIN:
            shown = names[:4]
            rest  = len(names) - len(shown)
            label = ', '.join(shown) + (f', ... +{rest} more' if rest > 0 else '')
            # Preserve current indentation from the first field line
            indent = len(block[0]) - len(block[0].lstrip())
            result.append(
                ' ' * indent
                + f'// [FIELD MAPPING: {len(names)} fields → {label}]'
            )
            i = j
        else:
            result.append(lines[i])
            i += 1

    compressed = '\n'.join(result)

    # Post-compression character cap — applied AFTER noise removal so real
    # logic is never discarded in favour of field assignment blocks.
    if len(compressed) > max_chars:
        compressed = (
            compressed[:max_chars]
            + f"\n\n// ... [body truncated at {max_chars} chars after compression;"
            " full logic available in the source file]"
        )

    return compressed


async def _find_symbol_definition(repo_path: Path, name: str, package_hint: str = "") -> str:
    """
    Find function/class/method definition for a named symbol (non-path).
    Returns full body (compressed), not just a snippet.

    Args:
        repo_path:     Root of the local repository.
        name:          Simple function/class name (e.g. "DownloadFile").
        package_hint:  Optional package/directory name to prioritise
                       (e.g. "controllers" from "controllers.DownloadFile").
                       Files inside a path segment matching this hint are
                       searched before the rest of the repo.
    """
    sections: list[str] = []
    escaped = re.escape(name)

    func_def_re = re.compile(
        rf"\b(?:"
        rf"func\s+(?:\(\w[^)]*\)\s+)?{escaped}"    # Go method/function
        rf"|def\s+{escaped}"                         # Python
        rf"|function\s+{escaped}"                    # JS/TS function keyword
        rf"|(?:async\s+function\s+){escaped}"        # async JS function
        rf"|const\s+{escaped}\s*=\s*(?:async\s+)?\("  # JS const arrow
        rf"|class\s+{escaped}"                        # class def
        rf")\s*[\s({{]",
        re.IGNORECASE,
    )

    # Sort files: package_hint directory first, then everything else.
    # This ensures controllers/downloadController.go is searched before an
    # unrelated helper that might happen to contain the same function name.
    hint_lower = package_hint.lower() if package_hint else ""

    def _file_priority(p: Path) -> int:
        if not hint_lower:
            return 0
        rel_parts = p.relative_to(repo_path).parts
        return 0 if any(hint_lower in part.lower() for part in rel_parts[:-1]) else 1

    ordered_files = sorted(
        (
            f for f in sorted(repo_path.rglob("*"))
            if not f.is_dir()
            and f.suffix in _SOURCE_EXTS_ALL
            and not _should_skip(f.relative_to(repo_path).parts)
        ),
        key=_file_priority,
    )

    hits = 0
    for fpath in ordered_files:
        try:
            text = fpath.read_text(errors="replace")
            file_lines = text.splitlines()
        except OSError:
            continue

        for i, line in enumerate(file_lines):
            if func_def_re.search(line):
                body = _compress_handler_body(
                    _read_full_function_body(file_lines, i, fpath.suffix)
                )
                rel = str(fpath.relative_to(repo_path))
                lang = fpath.suffix.lstrip(".")
                sections.append(
                    f"### \U0001f527 `{name}` \u2014 `{rel}` (L{i + 1})\n"
                    f"```{lang}\n{body}\n```"
                )
                hits += 1
                break  # one hit per file

        if hits >= 5:
            break

    # Fallback: plain text search if no definition found
    if not sections:
        for fpath in ordered_files:
            try:
                file_lines = fpath.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(file_lines):
                if name in line:
                    start = max(0, i - 2)
                    end = min(len(file_lines), i + 12)
                    _arrow = "\u2192"
                    ctx = "\n".join(
                        f"  {_arrow if j == i else ' '} L{j+1}: {file_lines[j]}"
                        for j in range(start, end)
                    )
                    rel = str(fpath.relative_to(repo_path))
                    sections.append(f"### \U0001f4c4 `{rel}` (reference)\n```\n{ctx}\n```")
                    break
            if len(sections) >= 5:
                break

    return "\n\n".join(sections) if sections else f"(simbol `{name}` tidak ditemukan)"


async def _trace_api_route(repo_path: Path, api_path: str) -> str:
    """
    2-phase trace for an API path:
      Phase 1 – Locate route registration line; extract handler function reference(s).
      Phase 2 – Find and read full handler implementation body.
               Optionally trace 1 level of inner calls for service/repo layer.

    Handles group-based routers (Gin groups, Express Router, etc.) where the
    full path is split across Group("/download") + GET("/:uuid/...").
    """
    sections: list[str] = []
    search_terms = _build_path_search_terms(api_path)

    # Collect all source files; routing files get first priority
    all_files: list[Path] = []
    route_files: list[Path] = []
    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir() or fpath.suffix not in _SOURCE_EXTS_ALL:
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        if fpath.name.lower() in _ROUTE_FILE_NAMES:
            route_files.append(fpath)
        else:
            all_files.append(fpath)

    ordered_files = route_files + all_files

    # ── Phase 1: Find route registration ────────────────────────────────────
    registration_hits: list[str] = []  # markdown snippets
    handler_names: list[str] = []      # extracted handler function names

    for fpath in ordered_files:
        try:
            text = fpath.read_text(errors="replace")
            file_lines = text.splitlines()
        except OSError:
            continue

        matched_line_idx: int | None = None
        matched_term: str = ""

        # Try search terms from most specific to least
        for term in search_terms:
            for i, line in enumerate(file_lines):
                if term in line:
                    # Exclude comment-only lines
                    stripped = line.lstrip()
                    if stripped.startswith("//") or stripped.startswith("#"):
                        continue
                    matched_line_idx = i
                    matched_term = term
                    break
            if matched_line_idx is not None:
                break

        if matched_line_idx is None:
            continue

        # Show context around the registration line
        start = max(0, matched_line_idx - 3)
        end   = min(len(file_lines), matched_line_idx + 8)
        ctx_lines = []
        for j in range(start, end):
            marker = "→" if j == matched_line_idx else " "
            ctx_lines.append(f"  {marker} L{j+1}: {file_lines[j]}")
        ctx = "\n".join(ctx_lines)
        rel = str(fpath.relative_to(repo_path))
        registration_hits.append(
            f"**`{rel}`** (matched: `{matched_term}`):\n```\n{ctx}\n```"
        )

        # Extract handler reference(s) from matched line
        raw_handlers = _extract_handler_from_line(file_lines[matched_line_idx])
        # Also scan ±2 lines for inline anonymous func or decorator patterns
        for extra_i in range(max(0, matched_line_idx - 1), min(len(file_lines), matched_line_idx + 3)):
            if extra_i == matched_line_idx:
                continue
            raw_handlers.extend(_extract_handler_from_line(file_lines[extra_i]))

        for h in raw_handlers:
            if h not in handler_names and len(h) > 2:
                handler_names.append(h)

        if len(registration_hits) >= 4:
            break  # enough route hits

    if registration_hits:
        sections.append(
            "## 📍 Route Registration\n\n" + "\n\n".join(registration_hits)
        )
    else:
        # Hard fallback: show any file containing first path segment
        segs = [s for s in api_path.split("/") if s and not s.startswith(":")]
        sections.append(
            f"## ⚠️ Route Tidak Ditemukan Secara Eksplisit\n"
            f"Path `{api_path}` tidak ditemukan verbatim di codebase.\n"
            f"Kemungkinan terdapat di file routing dengan group/prefix. "
            f"Coba cari file yang mengandung: {', '.join(segs)}"
        )

    # ── Phase 2: Find handler implementations ───────────────────────────────
    if handler_names:
        impl_sections: list[str] = []
        impl_bodies:   list[str] = []   # raw compressed bodies for Phase 3
        seen_handlers: set[str] = set()

        for handler_ref in dict.fromkeys(handler_names):  # deduplicate, preserve order
            # handler_ref may be "ctrl.DownloadFile" or just "DownloadFile"
            simple_name = handler_ref.split(".")[-1]
            if simple_name in seen_handlers or len(simple_name) < 3:
                continue
            seen_handlers.add(simple_name)

            # Build a regex that matches Go / Python / JS / TS function definitions
            esc = re.escape(simple_name)
            func_def_re = re.compile(
                rf"\b(?:"
                rf"func\s+(?:\(\w[^)]*\)\s+)?{esc}"   # Go method or function
                rf"|def\s+{esc}"                         # Python
                rf"|function\s+{esc}"                    # JS/TS
                rf"|const\s+{esc}\s*=\s*(?:async\s+)?\("  # JS arrow
                rf")\s*",
                re.IGNORECASE,
            )

            for fpath in ordered_files:
                try:
                    file_lines = fpath.read_text(errors="replace").splitlines()
                except OSError:
                    continue

                for i, line in enumerate(file_lines):
                    if func_def_re.search(line):
                        body = _compress_handler_body(
                            _read_full_function_body(file_lines, i, fpath.suffix)
                        )
                        rel = str(fpath.relative_to(repo_path))
                        lang = fpath.suffix.lstrip(".")
                        impl_sections.append(
                            f"### \U0001f527 Handler: `{simple_name}` \u2014 `{rel}` (L{i + 1})\n"
                            f"```{lang}\n{body}\n```"
                        )
                        impl_bodies.append(body)
                        break  # found in this file

                if len(impl_sections) >= len(seen_handlers):
                    break  # found one file per handler

            if len(impl_sections) >= 6:
                break  # cap at 6 handler implementations

        if impl_sections:
            sections.append("## \U0001f527 Handler Implementation\n\n" + "\n\n".join(impl_sections))

        # ── Phase 3: Deep trace – definitions of functions called by handler ──
        # Extract function call names from every collected handler body and look
        # up their definitions in parallel.  This surfaces service / repository /
        # utility layers one level deeper than the handler, giving the LLM the
        # evidence needed to answer "how does this endpoint work end-to-end".
        if impl_bodies:
            all_bodies_text = "\n".join(impl_bodies)
            called_funcs = _extract_called_functions(all_bodies_text)
            already_traced: set[str] = set(seen_handlers)
            funcs_to_lookup = [fn for fn in called_funcs if fn not in already_traced]

            if funcs_to_lookup:
                logger.debug(
                    "_trace_api_route Phase 3: looking up %d downstream function(s): %r",
                    len(funcs_to_lookup), funcs_to_lookup[:8],
                )
                lookup_results = await asyncio.gather(
                    *[_find_symbol_definition(repo_path, fn) for fn in funcs_to_lookup],
                    return_exceptions=True,
                )
                downstream_sections: list[str] = []
                for func_name, result in zip(funcs_to_lookup, lookup_results):
                    if isinstance(result, Exception):
                        logger.debug(
                            "_trace_api_route Phase 3: lookup error for %r: %s",
                            func_name, result,
                        )
                        continue
                    defn = str(result)
                    if defn and not _evidence_is_empty(defn):
                        downstream_sections.append(defn)
                        logger.debug(
                            "_trace_api_route Phase 3: found definition for %r", func_name
                        )
                    if len(downstream_sections) >= _MAX_DOWNSTREAM_FUNCS:
                        break

                if downstream_sections:
                    sections.append(
                        "## \U0001f517 Downstream Functions (called by handler)\n\n"
                        + "\n\n".join(downstream_sections)
                    )

    return "\n\n".join(sections) if sections else f"(path `{api_path}` tidak ditemukan di repositori)"


# ── Top-level Q/A extraction dispatcher ──────────────────────────────────────

async def run_qa_extraction(
    repo_path:  Path,
    intent:     QAIntent,
    user_input: str,
    *,
    candidate_route_filenames: list[str] | None = None,
    symbol_target: str = "",
) -> dict[str, str]:
    """
    Jalankan extractor yang sesuai berdasarkan intent dan kembalikan evidence dict.

    Returns:
        dict { "section_title": "markdown_content", ... }
    """
    logger.info("QA extraction: intent=%s repo=%s", intent, repo_path)

    _extractor_map: dict[QAIntent, Callable[..., Awaitable[str]]] = {
        QAIntent.API_ENDPOINTS:  lambda: extract_api_endpoints(repo_path, candidate_route_filenames),
        QAIntent.TECH_STACK:     lambda: extract_tech_stack(repo_path),
        QAIntent.DATA_MODELS:    lambda: extract_data_models(repo_path),
        QAIntent.DEPENDENCIES:   lambda: extract_dependencies(repo_path),
        QAIntent.CI_CD:          lambda: extract_ci_cd(repo_path),
        QAIntent.SECURITY:       lambda: extract_security(repo_path),
        QAIntent.MAIN_FLOW:      lambda: extract_main_flow(repo_path),
        QAIntent.SPECIFIC_SYMBOL: lambda: extract_specific_symbol(
            repo_path,
            symbol_target or extract_specific_target(user_input),
            user_input=user_input,
        ),
    }

    assert intent != QAIntent.FULL_INSPECTION, \
        "run_qa_extraction tidak boleh dipanggil untuk FULL_INSPECTION"

    extractor = _extractor_map.get(intent)
    if extractor is None:
        return {"Error": f"Tidak ada extractor untuk intent {intent}"}

    try:
        content = await extractor()
        section_labels: dict[QAIntent, str] = {
            QAIntent.API_ENDPOINTS:   "📡 API Endpoints & Routes",
            QAIntent.TECH_STACK:      "🛠️ Tech Stack & Infrastruktur",
            QAIntent.DATA_MODELS:     "🗃️ Data Models & Schema",
            QAIntent.DEPENDENCIES:    "📦 Dependencies & Packages",
            QAIntent.CI_CD:           "🚀 CI/CD & Deployment Pipeline",
            QAIntent.SECURITY:        "🔐 Security & Auth",
            QAIntent.MAIN_FLOW:       "🔄 Main Flow & Arsitektur",
            QAIntent.SPECIFIC_SYMBOL: f"🔍 Detail: `{extract_specific_target(user_input) or user_input}`",
        }
        label = section_labels.get(intent, "Evidence")
        return {label: content}
    except Exception as exc:  # noqa: BLE001
        logger.warning("QA extractor error (intent=%s): %s", intent, exc)
        return {"Error": f"Extractor gagal: {exc}"}
