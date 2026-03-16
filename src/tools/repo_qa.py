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
            r"(?:jelaskan|jabarkan|explain|apa.itu|what.is|describe|tentang|cari)\s+/[a-zA-Z]",
            # Match "jelaskan fungsi X", "apa itu class Y", "jabarkan method Z"
            r"(?:jelaskan|jabarkan|explain|describe|cari)\s+(?:fungsi|function|class|method|api|endpoint)\s+\w",
            # Match standalone /path questions (not inside a URL)
            r"(?:endpoint|api|route)\s+[/]\S+",
            r"[/][a-zA-Z][a-zA-Z0-9_/\-]+\s+(?:itu|adalah|digunakan|bekerja|fungsi)",
            # Match "jelaskan CamelCase or snake_case identifier"
            r"(?:jelaskan|jabarkan|explain|describe|cari)\s+[A-Z][a-zA-Z0-9]+",
            r"(?:jelaskan|jabarkan|explain|describe|cari)\s+[a-z][a-z0-9]*(?:_[a-z0-9]+)+",
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

    # Jika user meminta penjelasan dan menyertakan path spesifik (mis. "/upload"),
    # anggap ini permintaan `SPECIFIC_SYMBOL` dan beri prioritas sebelum pattern
    # Q/A umum seperti "ada api apa".
    if re.search(r"(?:jelaskan|jabarkan|explain|apa.itu|what.is|describe|cari)", text) and re.search(r"(?:^|\s)/[a-z0-9_\-/]+", text):
        logger.debug("QA classify: specific symbol detected (path present) -> SPECIFIC_SYMBOL")
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
    if re.search(
        r"(?:jelaskan|jabarkan|explain|describe|cari|apa.itu|tentang)\s+"
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
    """
    # Match path like /upload, /api/v1/something — strip HTTP URLs first to
    # avoid picking up repo URL paths (e.g. /okai-ai-internal/okai-v2.git)
    text_no_url = re.sub(r"https?://\S+", "", user_input)
    path_match = re.search(r"(/[a-zA-Z][a-zA-Z0-9_/\-]*)", text_no_url)
    if path_match:
        return path_match.group(1)

    # Match "explain/jelaskan/jabarkan <keyword> <name>" — captures the actual name, not keyword
    kw_sym_match = re.search(
        r"(?:jelaskan|jabarkan|explain|apa.itu|what.is|describe|tentang|cari)\s+"
        r"(?:fungsi|function|class|method|api|endpoint)\s+([a-zA-Z_]\w*)",
        user_input,
        re.IGNORECASE,
    )
    if kw_sym_match:
        return kw_sym_match.group(1)

    # Match "explain/jelaskan/jabarkan <name>" directly (no intermediate keyword)
    sym_match = re.search(
        r"(?:jelaskan|jabarkan|explain|apa.itu|what.is|describe|tentang|cari)\s+([a-zA-Z_]\w*)",
        user_input,
        re.IGNORECASE,
    )
    if sym_match:
        return sym_match.group(1)

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


# ── Extractor: Specific Symbol ────────────────────────────────────────────────

async def extract_specific_symbol(repo_path: Path, target: str) -> str:
    """
    Temukan definisi + penggunaan simbol tertentu (API path, fungsi, class).

    Args:
        target: string yang dicari, bisa path API (/upload) atau nama simbol.
    """
    if not target:
        return "(target tidak ditentukan)"

    sections: list[str] = []
    escaped = re.escape(target)
    pattern = re.compile(
        rf"(?:"
        rf'["\'{re.escape(target)}["\']'      # string literal "/upload"
        rf"|def\s+{escaped}"                   # Python function def
        rf"|class\s+{escaped}"                 # class def
        rf"|function\s+{escaped}"              # JS function
        rf"|const\s+{escaped}\s*="             # JS const
        rf"|{escaped}\s*[(:{{]"                # Go/Java/TS definition
        rf")",
        re.IGNORECASE,
    )

    hits: list[tuple[str, list[str]]] = []
    source_exts = {".py", ".js", ".ts", ".go", ".java", ".rb", ".php", ".cs"}

    for fpath in sorted(repo_path.rglob("*")):
        if fpath.is_dir() or fpath.suffix not in source_exts:
            continue
        if _should_skip(fpath.relative_to(repo_path).parts):
            continue
        try:
            lines = fpath.read_text(errors="replace").splitlines()
        except OSError:
            continue

        matched_lines: list[str] = []
        for i, line in enumerate(lines):
            if pattern.search(line) or target in line:
                # Show context: 3 lines before and after
                start = max(0, i - 3)
                end   = min(len(lines), i + 10)
                ctx = "\n".join(
                    f"  {'→' if j == i else ' '} L{j+1}: {lines[j]}"
                    for j in range(start, end)
                )
                matched_lines.append(ctx)
                if len(matched_lines) >= 5:
                    break

        if matched_lines:
            rel = str(fpath.relative_to(repo_path))
            hits.append((rel, matched_lines))

        if len(hits) >= MAX_FILES_PER_TOPIC:
            break

    if not hits:
        return f"(simbol/path `{target}` tidak ditemukan di repositori ini)"

    for rel, match_ctxs in hits:
        combined = "\n\n---\n".join(match_ctxs)
        sections.append(f"### 📄 `{rel}`\n```\n{combined}\n```")

    return "\n\n".join(sections)


# ── Top-level Q/A extraction dispatcher ──────────────────────────────────────

async def run_qa_extraction(
    repo_path:  Path,
    intent:     QAIntent,
    user_input: str,
    *,
    candidate_route_filenames: list[str] | None = None,
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
            repo_path, extract_specific_target(user_input)
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
