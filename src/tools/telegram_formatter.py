"""telegram_formatter.py – sanitize LLM-generated text for Telegram delivery.

The Telegram MarkdownV2 parser is strict about unmatched markers and unsupported
constructs.  ``telegramify_markdown`` (used in ``_safe_reply``) can raise or fall
back to plain text when the input contains HTML tags, orphaned markdown symbols,
or other artefacts that LLMs occasionally produce.

``sanitize_for_telegram(text)`` is a lightweight pre-processing step that cleans
up the most common issues *before* the text reaches the Telegram layer, so the
formatted output reaches the user intact.
"""

from __future__ import annotations

import html
import re

# ── Patterns compiled once at import time ─────────────────────────────────────

# <think> / <thinking> blocks from reasoning models (e.g. DeepSeek R1)
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)

# Remaining HTML tags (e.g. <b>, </b>, <br>, <p>, <strong>, ...)
_HTML_TAG_RE = re.compile(r"<[^>]{1,60}>")

# Consecutive blank lines – collapse to at most two
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Orphaned bold/italic markers at beginning or end of a line (common LLM artefact)
# e.g. a line that is just "**" or starts with "* " (stray bullet-in-bold)
_STRAY_BOLD_RE  = re.compile(r"^\*{2,3}\s*$", flags=re.MULTILINE)
_STRAY_ITALIC_RE = re.compile(r"^_{1,2}\s*$", flags=re.MULTILINE)


def _fix_unclosed_code_fences(text: str) -> str:
    """Ensure every opening triple-backtick fence has a matching closing fence.

    LLMs occasionally open a code block but forget to close it, which causes
    everything that follows to be treated as code by Telegram.
    """
    lines = text.split("\n")
    open_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            open_fence = not open_fence
    if open_fence:
        text += "\n```"
    return text


def sanitize_for_telegram(text: str) -> str:
    """Return *text* cleaned up so it can be safely rendered by Telegram.

    Steps applied (in order):
    1. Strip ``<think>``/``<thinking>`` reasoning blocks.
    2. Decode HTML entities (``&amp;`` → ``&``, ``&lt;`` → ``<``, …).
    3. Remove remaining HTML tags (``<b>``, ``</b>``, ``<br>``, …).
    4. Remove stray/orphaned bold/italic markers on their own line.
    5. Close any unclosed triple-backtick fences.
    6. Collapse runs of 3+ blank lines to 2.
    7. Strip leading/trailing whitespace.
    """
    if not text:
        return text

    # 1 – reasoning blocks
    text = _THINK_BLOCK_RE.sub("", text)

    # 2 – HTML entities (do this *before* stripping tags so &amp; → & not &amp;tag)
    text = html.unescape(text)

    # 3 – remaining HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # 4 – stray bold / italic markers on a line by themselves
    text = _STRAY_BOLD_RE.sub("", text)
    text = _STRAY_ITALIC_RE.sub("", text)

    # 5 – unclosed code fences
    text = _fix_unclosed_code_fences(text)

    # 6 – collapse excessive blank lines
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    # 7 – trim
    return text.strip()
