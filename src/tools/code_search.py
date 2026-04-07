"""
code_search.py – Semantic code-file discovery helpers for DeveloperAgent.

Provides two main public functions:

  build_ast_index(repo_path)
      Walk the repo, parse each source file with tree-sitter (Python, JS, TS,
      Vue) and return a dict mapping relative path → list of extracted symbols
      (function names, class names, identifiers). Falls back to a fast
      regex-based extractor for any file that tree-sitter cannot parse.
      Go (.go) and Protobuf (.proto) files are indexed using a regex-based
      extractor so that routes, handlers, and message types are discoverable.

  rank_files_by_relevance(candidates, symbol_index, task)
      Score each candidate file against the task description using TF-IDF
      cosine similarity (scikit-learn). Returns candidates sorted from most to
      least relevant, dropping files whose similarity falls below MIN_SCORE.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum TF-IDF cosine similarity for a file to be considered relevant.
MIN_SCORE = 0.05

# Minimum semantic cosine similarity for a file to be in the "relevant" tier
# when sentence-transformers are available.
MIN_SEMANTIC_SCORE = 0.20

# Disk cache directory for FAISS indexes (one per repo commit hash).
_FAISS_CACHE_DIR = Path.home() / ".cache" / "satunggal" / "faiss"

# Extensions processed per language.
_PY_EXTS    = {".py"}
_JS_EXTS    = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS    = {".ts", ".tsx"}
_VUE_EXTS   = {".vue", ".svelte"}
_GO_EXTS    = {".go"}
_PROTO_EXTS = {".proto"}
_ALL_EXTS   = _PY_EXTS | _JS_EXTS | _TS_EXTS | _VUE_EXTS | _GO_EXTS | _PROTO_EXTS

# Paths/dirs to skip during traversal.
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

# Maximum file size to parse (bytes) – skip huge generated bundles.
_MAX_FILE_BYTES = 500_000

# ── Lazy tree-sitter language loaders ─────────────────────────────────────────

def _load_ts_languages() -> dict:
    """
    Lazily import individual tree-sitter language packages (tree-sitter >= 0.22
    new-API style) and return a map:  language_name → tree_sitter.Language.

    Packages tried:
      tree-sitter-python, tree-sitter-javascript, tree-sitter-typescript

    Returns an empty dict if none are installed, so callers can fall back to
    the regex extractor gracefully.
    """
    try:
        from tree_sitter import Language  # type: ignore
    except Exception:
        logger.debug("code_search: tree-sitter package not available; using regex fallback")
        return {}

    languages: dict = {}
    _candidates = [
        ("python",     "tree_sitter_python",     "language"),
        ("javascript", "tree_sitter_javascript", "language"),
        ("typescript", "tree_sitter_typescript", "language_typescript"),
    ]
    for lang_name, module_name, func_name in _candidates:
        try:
            mod  = __import__(module_name)
            fn   = getattr(mod, func_name)
            languages[lang_name] = Language(fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug("code_search: could not load %s (%s)", lang_name, exc)

    if not languages:
        logger.warning(
            "code_search: no tree-sitter language packages found; using regex fallback"
        )
    else:
        logger.debug("code_search: loaded tree-sitter languages: %s", list(languages))
    return languages


def _ext_to_lang(ext: str) -> Optional[str]:
    """Map a file extension to a tree-sitter language name."""
    if ext in _PY_EXTS:
        return "python"
    if ext in _JS_EXTS:
        return "javascript"
    if ext in _TS_EXTS:
        return "typescript"
    return None  # Vue/Svelte handled separately


# ── Symbol extractors ─────────────────────────────────────────────────────────

# Compiled regex patterns for the fallback extractor.
_RE_PY_DEF     = re.compile(r"^(?:def|class|async def)\s+(\w+)", re.MULTILINE)
_RE_JS_DEF     = re.compile(r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|class\s+(\w+))", re.MULTILINE)
_RE_IMPORT     = re.compile(r'(?:import|from)\s+["\']?(\w[\w./]*)["\']?', re.MULTILINE)
_RE_VUE_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)

# Go: match top-level func declarations and import paths.
# Import regex matches both single-import lines and aliased imports inside import blocks.
_RE_GO_DEF    = re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE)
_RE_GO_IMPORT = re.compile(r'^\s*(?:\w+\s+)?"([\w./]+)"', re.MULTILINE)

# Proto: match message, service, rpc, and enum declarations (may be indented).
_RE_PROTO_DEF = re.compile(r"(?:^|\s)(?:message|service|rpc|enum)\s+(\w+)", re.MULTILINE)


def _extract_symbols_regex(text: str, ext: str) -> list[str]:
    """Fallback symbol extractor using regex – no tree-sitter required."""
    symbols: list[str] = []

    if ext in _PY_EXTS:
        symbols += _RE_PY_DEF.findall(text)

    elif ext in _JS_EXTS | _TS_EXTS:
        for m in _RE_JS_DEF.finditer(text):
            symbols += [g for g in m.groups() if g]

    elif ext in _VUE_EXTS:
        # Extract only from <script> block.
        script_match = _RE_VUE_SCRIPT.search(text)
        inner = script_match.group(1) if script_match else text
        for m in _RE_JS_DEF.finditer(inner):
            symbols += [g for g in m.groups() if g]

    elif ext in _GO_EXTS:
        # Extract function names and imported package paths.
        symbols += _RE_GO_DEF.findall(text)
        symbols += _RE_GO_IMPORT.findall(text)

    elif ext in _PROTO_EXTS:
        # Extract message, service, rpc, and enum names.
        symbols += _RE_PROTO_DEF.findall(text)

    # Always add imported module names (useful for task-keyword matching).
    if ext not in _GO_EXTS | _PROTO_EXTS:
        symbols += _RE_IMPORT.findall(text)

    return list(set(symbols))


def _extract_symbols_ts(
    languages: dict,
    source_code: bytes,
    lang_name: str,
) -> list[str]:
    """
    Parse source with tree-sitter (new API, >= 0.22) and extract identifier
    names from a curated node-type allowlist.
    """
    try:
        from tree_sitter import Parser  # type: ignore

        parser = Parser(languages[lang_name])
        tree   = parser.parse(source_code)

        _IDENTIFIER_TYPES = {
            "identifier",
            "type_identifier",
            "property_identifier",
        }
        _DEF_TYPES = {
            "function_definition",      # Python
            "class_definition",         # Python
            "function_declaration",     # JS/TS
            "class_declaration",        # JS/TS
            "method_definition",        # JS/TS
            "lexical_declaration",      # JS/TS (const/let)
            "variable_declarator",      # JS/TS
            "interface_declaration",    # TS
            "type_alias_declaration",   # TS
        }

        symbols: list[str] = []

        def _walk(node) -> None:  # type: ignore[no-untyped-def]
            if node.type in _DEF_TYPES:
                for child in node.children:
                    if child.type in _IDENTIFIER_TYPES:
                        name = child.text.decode("utf-8", errors="replace")
                        if name:
                            symbols.append(name)
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)
        return list(set(symbols))

    except Exception as exc:  # noqa: BLE001
        logger.debug("code_search: tree-sitter parse error (%s)", exc)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def build_ast_index(repo_path: Path) -> dict[str, list[str]]:
    """
    Walk *repo_path* recursively and build a symbol index.

    Returns:
        {
            "src/App.vue":  ["mounted", "fetchData", ...],
            "main.py":      ["main", "run_app", ...],
            ...
        }

    Key is the repo-relative POSIX path (no leading "./").
    Value is a deduplicated list of symbol strings extracted from that file.

    Files that cannot be read or are too large are silently skipped.
    """
    languages = _load_ts_languages()
    index:    dict[str, list[str]] = {}

    for abs_path in sorted(repo_path.rglob("*")):
        # Use helper to read file and obtain skip reasons for diagnostics
        content, skip_reason = _read_file_with_reason(abs_path, repo_path)
        if skip_reason is not None:
            logger.debug("code_search: skipping %s (%s)", abs_path, skip_reason)
            continue
        text = content

        rel_path = abs_path.relative_to(repo_path).as_posix()
        ext      = abs_path.suffix.lower()
        lang     = _ext_to_lang(ext)

        # Vue/Svelte: extract script block, then parse as JS.
        if ext in _VUE_EXTS:
            script_m = _RE_VUE_SCRIPT.search(text)
            inner    = script_m.group(1) if script_m else text
            if "javascript" in languages:
                symbols = _extract_symbols_ts(languages, inner.encode(), "javascript")
            else:
                symbols = _extract_symbols_regex(inner, ".js")
            # Complement with regex for template event handlers (@click="foo").
            symbols += re.findall(r'@\w+=["\'](\w+)', text)
        elif lang and lang in languages:
            symbols = _extract_symbols_ts(languages, text.encode(), lang)
        else:
            symbols = _extract_symbols_regex(text, ext)

        index[rel_path] = list(set(symbols))

    logger.info(
        "code_search: indexed %d files in %s",
        len(index),
        repo_path,
    )
    return index


def _read_file_with_reason(abs_path: Path, repo_path: Path, max_bytes: int = _MAX_FILE_BYTES) -> tuple[str, str | None]:
    """Try to read a file and return (content, skip_reason).

    If the file should be skipped, content will be an empty string and
    skip_reason will be a short string explaining why.
    """
    try:
        if abs_path.is_dir():
            return "", "is_dir"
        ext = abs_path.suffix.lower()
        if ext not in _ALL_EXTS:
            return "", "ext_not_supported"
        parts = set(abs_path.relative_to(repo_path).parts[:-1])
        if parts & _SKIP_DIRS:
            return "", "skip_dir"
        try:
            file_size = abs_path.stat().st_size
        except OSError:
            return "", "stat_failed"
        if file_size > _MAX_FILE_BYTES:
            return "", "too_large"
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return "", "read_error"
        return text[:max_bytes], None
    except Exception:
        return "", "unknown_error"


def rank_files_by_relevance(
    candidates: list[str],
    symbol_index: dict[str, list[str]],
    task: str,
    *,
    min_score: float = MIN_SCORE,
) -> list[str]:
    """
    Rank *candidates* by TF-IDF cosine similarity to *task*.

    Each candidate is represented as a text document: its path tokens
    joined with the symbols extracted by build_ast_index.

    Files not present in *symbol_index* are kept at the end of the list
    (they have an implicit score of 0 but are not dropped, because they
    may still be relevant for non-indexed languages).

    Args:
        candidates:   Repo-relative file paths to rank.
        symbol_index: Output of build_ast_index().
        task:         Natural-language task description.
        min_score:    Files scoring below this threshold are moved to the end.

    Returns:
        Re-ordered list of candidates (most relevant first).
    """
    if not candidates:
        return candidates

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity        # type: ignore
    except ImportError:
        logger.warning("code_search: scikit-learn not installed; skipping re-ranking")
        return candidates

    # Build a text document for each candidate.
    docs: list[str] = []
    for path in candidates:
        path_tokens = re.sub(r"[/._\-]", " ", path)
        symbols     = " ".join(symbol_index.get(path, []))
        docs.append(f"{path_tokens} {symbols}")

    # Append the task as the last document; we'll compare everything to it.
    corpus = docs + [task]

    try:
        vectorizer   = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        task_vec     = tfidf_matrix[-1]          # last row = task
        file_vecs    = tfidf_matrix[:-1]         # all other rows = files
        scores       = cosine_similarity(task_vec, file_vecs).flatten()
    except Exception as exc:  # noqa: BLE001
        logger.warning("code_search: TF-IDF scoring failed (%s); returning original order", exc)
        return candidates

    # Partition into relevant (score >= min_score) and low-relevance.
    indexed_scores = sorted(
        enumerate(candidates),
        key=lambda t: scores[t[0]],
        reverse=True,
    )
    ranked      = [c for i, c in indexed_scores if scores[i] >= min_score]
    low_scoring = [c for i, c in indexed_scores if scores[i] < min_score]

    logger.debug(
        "code_search: ranked %d relevant / %d low-score files for task %r",
        len(ranked), len(low_scoring), task[:60],
    )
    return ranked + low_scoring


# ── Semantic ranking via sentence-transformers + FAISS (Item 4) ───────────────
# This section is fully optional.  If sentence-transformers or faiss are not
# installed the functions below return None / fall back to the TF-IDF ranker.


def _get_repo_commit_hash(repo_path: Path) -> str:
    """Return the current HEAD commit hash for cache-key purposes."""
    try:
        head_file = repo_path / ".git" / "HEAD"
        ref = head_file.read_text().strip()
        if ref.startswith("ref: "):
            ref_path = repo_path / ".git" / ref[5:]
            return ref_path.read_text().strip()
        return ref  # detached HEAD – the hash is the content
    except Exception:
        return hashlib.md5(str(repo_path).encode()).hexdigest()


def _semantic_cache_path(repo_path: Path) -> Path:
    """Return the path to the FAISS index cache file for *repo_path*."""
    commit_hash = _get_repo_commit_hash(repo_path)
    repo_key    = hashlib.md5(str(repo_path.resolve()).encode()).hexdigest()[:8]
    return _FAISS_CACHE_DIR / f"{repo_key}_{commit_hash[:12]}.pkl"


def _build_semantic_index(
    candidates: list[str],
    symbol_index: dict[str, list[str]],
    repo_path: Path,
) -> object | None:
    """
    Build a FAISS flat-L2 index over sentence-transformer embeddings of each
    candidate file's description (path tokens + extracted symbols).

    Returns the FAISS index (or None if sentence-transformers / faiss are not
    installed).  The result is pickled to *_FAISS_CACHE_DIR* so subsequent
    calls for the same repo commit are instant.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import faiss                                            # type: ignore
        import numpy as np                                      # type: ignore
    except ImportError:
        return None

    cache_path = _semantic_cache_path(repo_path)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                cached = pickle.load(fh)
            if cached.get("candidates") == candidates:
                logger.debug("code_search: loaded FAISS index from cache %s", cache_path)
                return cached["index"]
        except Exception as exc:
            logger.debug("code_search: FAISS cache load failed (%s); rebuilding", exc)

    logger.info("code_search: building sentence-transformer FAISS index for %s", repo_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs: list[str] = []
    for path in candidates:
        path_tokens = re.sub(r"[/._\-]", " ", path)
        symbols     = " ".join(symbol_index.get(path, []))
        docs.append(f"{path_tokens} {symbols}")

    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner-product ≈ cosine after norm
    index.add(embeddings)

    _FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as fh:
            pickle.dump({"candidates": candidates, "index": index, "model": "all-MiniLM-L6-v2"}, fh)
        # Clean up stale caches for the same repo (different commit hashes).
        repo_key = cache_path.stem.split("_")[0]
        for old in _FAISS_CACHE_DIR.glob(f"{repo_key}_*.pkl"):
            if old != cache_path:
                try:
                    old.unlink()
                except OSError:
                    pass
    except Exception as exc:
        logger.debug("code_search: FAISS cache write failed (%s); continuing without cache", exc)

    return index


def rank_files_by_relevance_semantic(
    candidates: list[str],
    symbol_index: dict[str, list[str]],
    task: str,
    repo_path: Path,
    *,
    min_score: float = MIN_SEMANTIC_SCORE,
) -> list[str] | None:
    """
    Rank *candidates* using sentence-transformer embeddings + FAISS ANN search.

    Returns a re-ordered list (most semantically similar first) or **None** if
    sentence-transformers / faiss are not installed, so the caller can fall back
    to the TF-IDF ranker.

    Benefits over TF-IDF:
    - Captures synonyms: "error handler" ↔ "exception_handler", "on_failure"
    - Understands intent: "cara handle error" finds files with `try/except`, `catch`
    - Embeddings are cached per commit hash → rebuild only when code changes
    """
    if not candidates:
        return candidates

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import faiss                                            # type: ignore
        import numpy as np                                      # type: ignore
    except ImportError:
        return None

    index = _build_semantic_index(candidates, symbol_index, repo_path)
    if index is None:
        return None

    try:
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        task_emb   = model.encode([task], convert_to_numpy=True, show_progress_bar=False)
        task_emb   = task_emb.astype("float32")
        faiss.normalize_L2(task_emb)

        k          = min(len(candidates), 50)
        scores_arr, indices = index.search(task_emb, k)
        scores_flat = scores_arr[0]
        indices_flat = indices[0]

        # Build sorted list: above threshold first, rest in original order.
        above = [
            candidates[i]
            for s, i in sorted(zip(scores_flat, indices_flat), reverse=True)
            if s >= min_score and 0 <= i < len(candidates)
        ]
        above_set = set(above)
        below     = [c for c in candidates if c not in above_set]

        logger.info(
            "code_search: semantic ranked %d relevant / %d below-threshold for task %r",
            len(above), len(below), task[:60],
        )
        return above + below

    except Exception as exc:
        logger.warning("code_search: semantic ranking search failed (%s); falling back", exc)
        return None

