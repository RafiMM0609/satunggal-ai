"""
code_search.py – Semantic code-file discovery helpers for DeveloperAgent.

Provides two main public functions:

  build_ast_index(repo_path)
      Walk the repo, parse each source file with tree-sitter (Python, JS, TS,
      Vue) and return a dict mapping relative path → list of extracted symbols
      (function names, class names, identifiers). Falls back to a fast
      regex-based extractor for any file that tree-sitter cannot parse.

  rank_files_by_relevance(candidates, symbol_index, task)
      Score each candidate file against the task description using TF-IDF
      cosine similarity (scikit-learn). Returns candidates sorted from most to
      least relevant, dropping files whose similarity falls below MIN_SCORE.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum TF-IDF cosine similarity for a file to be considered relevant.
MIN_SCORE = 0.05

# Extensions processed per language.
_PY_EXTS   = {".py"}
_JS_EXTS   = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS   = {".ts", ".tsx"}
_VUE_EXTS  = {".vue", ".svelte"}
_GO_EXTS   = {".go"}
_PROTO_EXT = {".proto"}

# Include Go and proto files so the inspector can find Go REST endpoints
_ALL_EXTS = _PY_EXTS | _JS_EXTS | _TS_EXTS | _VUE_EXTS | _GO_EXTS | _PROTO_EXT

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
    from tree_sitter import Language  # type: ignore

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
_RE_PY_DEF   = re.compile(r"^(?:def|class|async def)\s+(\w+)", re.MULTILINE)
_RE_JS_DEF   = re.compile(r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|class\s+(\w+))", re.MULTILINE)
_RE_IMPORT   = re.compile(r'(?:import|from)\s+["\']?(\w[\w./]*)["\']?', re.MULTILINE)
_RE_VUE_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)

# Go-specific patterns (simple, regex-based fallback)
_RE_GO_FUNC   = re.compile(r"^func\s+([A-Za-z_][\w]*)\s*\(", re.MULTILINE)
_RE_GO_IMPORT = re.compile(r'import\s+(?:\((.*?)\)|"([^"]+)")', re.DOTALL)


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
        # Extract top-level function names and import paths from Go files.
        symbols += _RE_GO_FUNC.findall(text)
        # import blocks may contain multiple lines; extract each quoted path
        for m in _RE_GO_IMPORT.finditer(text):
            block = m.group(1)
            single = m.group(2)
            if single:
                symbols.append(single)
            elif block:
                # find all quoted import paths inside the block
                symbols += re.findall(r'"([^"]+)"', block)

    # Always add imported module names (useful for task-keyword matching).
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
        # Skip dirs and unrecognised extensions.
        if abs_path.is_dir():
            continue
        if abs_path.suffix.lower() not in _ALL_EXTS:
            continue

        # Skip paths inside blacklisted directories.
        parts = set(abs_path.relative_to(repo_path).parts[:-1])
        if parts & _SKIP_DIRS:
            continue

        # Skip oversized files (probably generated bundles).
        try:
            file_size = abs_path.stat().st_size
        except OSError:
            continue
        if file_size > _MAX_FILE_BYTES:
            logger.debug("code_search: skipping large file %s (%d B)", abs_path, file_size)
            continue

        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

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
