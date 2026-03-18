"""
WebReaderTool – headless URL fetcher with Accessibility-Tree extraction.

Implements the "Mapping Phase" described in the Autonomous Browsing brief:
  1. Opens the URL in a Playwright headless Chromium browser.
  2. Blocks images / video / fonts to keep RAM usage low on a 2 GB VPS.
  3. Extracts:
       - ``page_text``    : cleaned visible text (for LLM context injection)
       - ``a11y_tree``    : simplified Accessibility-Tree snapshot (role + name pairs)
       - ``title``        : document <title>
       - ``url``          : final URL after redirects
  4. Optionally uses a stored Playwright storage-state file so the browser
     can start in a logged-in position (supplied via ``task.metadata``).

Resource constraints (VPS-safe)
─────────────────────────────────
  * Headless Chromium, one tab at a time.
  * Images, media, fonts, and ads blocked via route interception.
  * 30-second navigation timeout.
  * Browser is always closed in a ``finally`` block.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# Resource types that are blocked to save RAM / bandwidth
_BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}

# Maximum characters extracted from the page text (avoid LLM token overflow)
_MAX_TEXT_CHARS = 8_000

# Navigation timeout in milliseconds
_NAV_TIMEOUT_MS = 30_000


class WebReaderTool(BaseTool):
    """
    Headless web reader using Playwright Chromium.

    Orchestrator usage
    ------------------
    The orchestrator (or a web_automation agent) calls ``run(task)`` with::

        task.metadata["target_url"] = "https://example.com"
        # Optional: pre-loaded session
        task.metadata["session_path"] = "/tmp/browser_sessions/example_com.json"

    Result keys
    -----------
    ``title``     : str   – page <title>
    ``url``       : str   – final URL after redirects
    ``page_text`` : str   – visible text (up to _MAX_TEXT_CHARS chars)
    ``a11y_tree`` : list  – list of {role, name} dicts from the accessibility tree
    ``error``     : str   – present only on failure
    """

    name = "web_reader"

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """Fetch a URL and return structured page content.

        Reads ``task.metadata["target_url"]`` as the URL to visit.
        Optionally reads ``task.metadata["session_path"]`` for a stored
        Playwright storage-state JSON file.
        """
        url: str = task.metadata.get("target_url", "").strip()
        if not url:
            logger.warning("WebReaderTool: no target_url in task.metadata")
            return {"error": "target_url not provided in task.metadata"}

        session_path: str | None = task.metadata.get("session_path")

        logger.info(
            "WebReaderTool: fetching url=%s session=%s session_path=%s",
            url,
            task.session_id,
            session_path or "none",
        )

        try:
            result = await asyncio.wait_for(
                self._fetch(url, session_path=session_path),
                timeout=35.0,  # slightly above playwright's own timeout
            )
            logger.info(
                "WebReaderTool: done url=%s title=%r text_chars=%d a11y_nodes=%d",
                url,
                result.get("title", ""),
                len(result.get("page_text", "")),
                len(result.get("a11y_tree", [])),
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("WebReaderTool: timed out fetching %s", url)
            return {"error": f"Timeout fetching {url} (>35 s)"}
        except Exception as exc:
            logger.exception("WebReaderTool: unexpected error for %s: %s", url, exc)
            return {"error": str(exc)}

    # ── Private implementation ─────────────────────────────────────────────────

    async def _fetch(
        self,
        url: str,
        *,
        session_path: str | None = None,
    ) -> dict[str, Any]:
        """Open the URL with Playwright and extract page content."""
        from playwright.async_api import async_playwright  # lazy import

        async with async_playwright() as p:
            browser_kwargs: dict[str, Any] = {
                "headless": True,
            }

            browser = await p.chromium.launch(**browser_kwargs)
            try:
                context_kwargs: dict[str, Any] = {
                    "java_script_enabled": True,
                    "accept_downloads": False,
                }
                if session_path and Path(session_path).exists():
                    context_kwargs["storage_state"] = session_path
                    logger.debug("WebReaderTool: loading session from %s", session_path)

                context = await browser.new_context(**context_kwargs)

                # Block resource-heavy types to conserve RAM
                async def _block(route, request):  # noqa: ANN001
                    if request.resource_type in _BLOCKED_RESOURCES:
                        await route.abort()
                    else:
                        await route.continue_()

                await context.route("**/*", _block)

                page = await context.new_page()
                page.set_default_navigation_timeout(_NAV_TIMEOUT_MS)
                page.set_default_timeout(_NAV_TIMEOUT_MS)

                await page.goto(url, wait_until="domcontentloaded")

                title       = await page.title()
                final_url   = page.url
                page_text   = await self._extract_text(page)
                a11y_tree   = await self._extract_a11y(page)

                await context.close()
                return {
                    "title":     title,
                    "url":       final_url,
                    "page_text": page_text[:_MAX_TEXT_CHARS],
                    "a11y_tree": a11y_tree,
                }
            finally:
                await browser.close()

    @staticmethod
    async def _extract_text(page) -> str:  # noqa: ANN001
        """Extract visible text from the page, stripping scripts/styles."""
        try:
            text: str = await page.evaluate(
                """() => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript, svg').forEach(el => el.remove());
                    return (clone.innerText || clone.textContent || '').replace(/\\s+/g, ' ').trim();
                }"""
            )
            return text
        except Exception as exc:
            logger.warning("WebReaderTool: text extraction failed: %s", exc)
            return ""

    @staticmethod
    async def _extract_a11y(page) -> list[dict[str, str]]:  # noqa: ANN001
        """Return a simplified accessibility tree as a list of {role, name} dicts."""
        try:
            snapshot = await page.accessibility.snapshot(interesting_only=True)
            if not snapshot:
                return []

            nodes: list[dict[str, str]] = []

            def _walk(node: dict) -> None:
                role = node.get("role", "")
                name = (node.get("name") or "").strip()
                if role and name:
                    nodes.append({"role": role, "name": name})
                for child in node.get("children", []):
                    _walk(child)

            _walk(snapshot)
            return nodes
        except Exception as exc:
            logger.warning("WebReaderTool: a11y extraction failed: %s", exc)
            return []
