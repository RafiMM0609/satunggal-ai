"""
BrowserNavigatorTool – performs physical browser interactions.

Implements the "Interaction" phase of the Autonomous Browsing brief:
  - navigate(url)         : go to a URL
  - click_by_text(text)   : click the first element whose accessible name
                            matches *text* (case-insensitive, partial match)
  - type_text(selector, text) : fill a text field by CSS selector or label
  - scroll(direction)     : scroll up/down the page
  - screenshot()          : capture the current viewport as PNG bytes
  - get_content()         : read text + accessibility tree from the current page
  - extract_data(selector): extract structured list data via CSS selector
  - save_session(url)     : persist Playwright storage-state for future logins

All actions run in a single Chromium context kept alive across sequential
commands within the same ``AgentTask``.  The browser is closed when the
agent is done (via ``close()``).

Resource constraints (VPS-safe)
─────────────────────────────────
  * Headless Chromium, one tab at a time.
  * Blocked: images, media, fonts (saves >40 % RAM).
  * 30-second timeout on every action.
  * Browser is always closed in a ``finally`` / ``close()`` block.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}
_TIMEOUT_MS              = 30_000
_CLICK_LOAD_TIMEOUT_MS   = 15_000   # post-click domcontentloaded settle wait (covers login redirects)
_NETWORK_IDLE_TIMEOUT_MS = 15_000   # post-click/navigate networkidle wait (SPA-safe)
_MAX_PAGE_TEXT_CHARS     = 8_000    # truncation limit for get_content page text
_CLICK_NAV_TEXT_CHARS    = 3_000    # page-text snippet captured inside click result on navigation
_SPA_RENDER_WAIT_MS      = 3_000    # extra wait for SPA to render content after navigation/click
_CLICK_LOCATE_TIMEOUT_MS = 8_000    # timeout for each individual click locator attempt
# Maximum ratio of element text length to search text length for JS click fallback.
# Elements whose text is more than this many times longer than the search term are
# treated as containers (e.g. a nav bar holding many menu items) and skipped so
# that the deepest, most-specific matching leaf element is clicked instead.
_JS_CLICK_MAX_TEXT_RATIO = 10

# Human-like delay range (milliseconds) to reduce bot-detection risk
_HUMAN_DELAY_MIN_MS = 300
_HUMAN_DELAY_MAX_MS = 1_200

# URL path segments that indicate an error page (case-insensitive match)
_ERROR_URL_SEGMENTS = frozenset(["/500", "/error", "/not-found", "/404", "/503"])

# Content phrases that indicate an error state (lowercase, case-insensitive match).
# Add site-specific strings here as new web applications are supported.
_ERROR_CONTENT_PHRASES = (
    "proses peningkatan layanan",   # MyPertamina 500 maintenance message
    "something went wrong",         # generic English error
    "internal server error",        # HTTP 500 body text
    "service unavailable",          # HTTP 503 body text
    "koneksi internet terputus",    # Indonesian "internet connection lost"
    "cors",                         # CORS error message fragments
)


async def _random_delay(
    min_ms: int = _HUMAN_DELAY_MIN_MS,
    max_ms: int = _HUMAN_DELAY_MAX_MS,
) -> None:
    """Sleep for a random duration in [min_ms, max_ms] milliseconds.

    Simulates human think-time between browser actions, reducing the likelihood
    of bot-detection by server-side rate limiters and browser fingerprinting heuristics.
    """
    delay_s = random.SystemRandom().uniform(min_ms, max_ms) / 1_000
    await asyncio.sleep(delay_s)


class BrowserNavigatorTool(BaseTool):
    """
    Stateful browser interaction tool.

    Maintains a single open Playwright browser + page across multiple calls.
    The caller is responsible for calling ``await close()`` when done.

    Orchestrator usage
    ------------------
    The web_automation agent (or orchestrator) calls ``run(task)`` with::

        task.metadata["browser_action"]  = "navigate" | "click" | "type" |
                                           "scroll" | "screenshot" |
                                           "get_content" | "extract_data" |
                                           "save_session"
        task.metadata["target_url"]      = "https://..."          # for navigate
        task.metadata["click_text"]      = "Login"                # for click
        task.metadata["type_selector"]   = "#email"               # for type (CSS selector)
        task.metadata["type_label"]      = "Email"                # for type (label/placeholder text)
        task.metadata["type_text"]       = "user@example.com"     # for type
        task.metadata["scroll_direction"]= "down" | "up"          # for scroll
        task.metadata["session_url"]     = "https://..."          # for save_session
        task.metadata["extract_selector"]= "article h2 a"        # for extract_data (optional)
        task.metadata["extract_attribute"]= "text" | "href"       # for extract_data (optional)
        task.metadata["extract_limit"]   = 50                     # for extract_data (optional)

    Result keys
    -----------
    ``action``        : str  – action that was performed
    ``success``       : bool – whether the action succeeded
    ``screenshot_b64``: str  – base64-encoded PNG (only for "screenshot")
    ``session_path``  : str  – path to saved session file (only for "save_session")
    ``page_text``     : str  – visible page text (only for "get_content")
    ``a11y_tree``     : list – accessibility tree nodes (only for "get_content")
    ``items``         : list – extracted data items (only for "extract_data")
    ``message``       : str  – human-readable outcome description
    ``error``         : str  – present only on failure
    """

    name = "browser_navigator"

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser:    Any = None
        self._context:    Any = None
        self._page:       Any = None

    # ── BaseTool.run interface ────────────────────────────────────────────────

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """Dispatch the browser action specified in ``task.metadata``."""
        action: str = task.metadata.get("browser_action", "").strip().lower()
        if not action:
            return {"error": "browser_action not specified in task.metadata", "success": False}

        logger.info(
            "BrowserNavigatorTool: action=%s session=%s", action, task.session_id
        )

        try:
            if action == "navigate":
                return await self._action_navigate(task)
            elif action == "click":
                return await self._action_click(task)
            elif action == "type":
                return await self._action_type(task)
            elif action == "scroll":
                return await self._action_scroll(task)
            elif action == "screenshot":
                return await self._action_screenshot(task)
            elif action == "get_content":
                return await self._action_get_content(task)
            elif action == "extract_data":
                return await self._action_extract_data(task)
            elif action == "save_session":
                return await self._action_save_session(task)
            else:
                return {
                    "error": f"Unknown browser_action: '{action}'",
                    "success": False,
                }
        except Exception as exc:
            logger.exception("BrowserNavigatorTool: action=%s failed: %s", action, exc)
            return {"error": str(exc), "success": False, "action": action}

    # ── Browser lifecycle ─────────────────────────────────────────────────────

    async def _ensure_browser(self, session_path: Optional[str] = None) -> None:
        """Launch browser + context + page if not already open."""
        if self._page is not None:
            return  # already open

        from playwright.async_api import async_playwright  # lazy import

        self._playwright = await async_playwright().start()
        self._browser    = await self._playwright.chromium.launch(headless=True)

        context_kwargs: dict[str, Any] = {
            "java_script_enabled": True,
            "accept_downloads":    False,
        }
        if session_path and Path(session_path).exists():
            context_kwargs["storage_state"] = session_path
            logger.debug("BrowserNavigatorTool: loading session from %s", session_path)

        self._context = await self._browser.new_context(**context_kwargs)

        # Block resource-heavy types to conserve RAM
        async def _block(route, request):  # noqa: ANN001
            if request.resource_type in _BLOCKED_RESOURCES:
                await route.abort()
            else:
                await route.continue_()

        await self._context.route("**/*", _block)

        self._page = await self._context.new_page()
        self._page.set_default_timeout(_TIMEOUT_MS)
        self._page.set_default_navigation_timeout(_TIMEOUT_MS)

    async def close(self) -> None:
        """Close the browser and release all resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:
            logger.warning("BrowserNavigatorTool.close: error during cleanup: %s", exc)
        finally:
            self._playwright = None
            self._browser    = None
            self._context    = None
            self._page       = None

    # ── Action implementations ────────────────────────────────────────────────

    async def _action_navigate(self, task: "AgentTask") -> dict[str, Any]:
        url: str = task.metadata.get("target_url", "").strip()
        if not url:
            return {"error": "target_url not provided", "success": False, "action": "navigate"}

        session_path = task.metadata.get("session_path")
        await self._ensure_browser(session_path=session_path)
        assert self._page is not None  # noqa: S101

        await self._page.goto(url, wait_until="domcontentloaded")

        # Wait for network to become idle so that SPA bootstrapping and any
        # redirects triggered by the page finish before we read the result.
        try:
            await self._page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            logger.debug("navigate: networkidle wait timed out after %dms", _NETWORK_IDLE_TIMEOUT_MS)

        # Extra wait for SPA frameworks (React/Vue/Angular) to complete their
        # initial render cycle and mount interactive elements before we try to
        # interact with the page.
        await self._wait_for_spa_stable()

        # Small human-like pause after navigation
        await _random_delay(min_ms=200, max_ms=700)

        title = await self._page.title()
        return {
            "action":  "navigate",
            "success": True,
            "url":     self._page.url,
            "title":   title,
            "message": f"Navigated to {self._page.url} – \"{title}\"",
        }

    async def _action_click(self, task: "AgentTask") -> dict[str, Any]:
        """Click the first element whose accessible name or text contains *click_text*.

        Tries multiple locator strategies in order from most to least specific,
        including ARIA roles used by SPA menu items (menuitem, option, tab,
        treeitem) and a JavaScript-based text-content search as a final fallback
        that works even for plain ``<div>``/``<span>`` elements with click handlers.

        When the click triggers a full-page navigation (e.g. a login form
        submission redirecting to a dashboard), the resulting page title and a
        brief text snippet are captured automatically so callers can report
        the post-login state without a separate ``get_content`` call.
        """
        click_text: str = task.metadata.get("click_text", "").strip()
        if not click_text:
            return {"error": "click_text not provided", "success": False, "action": "click"}

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        # Remember URL before the click so we can detect post-click navigation
        url_before: str = self._page.url

        # Wait for the page to be interactive (SPA may still be mounting menus)
        await self._wait_for_spa_stable()

        # ── Try each locator strategy in order ────────────────────────────────
        # Extended set: standard button/link roles first, then SPA-specific menu
        # roles, then generic text match, and finally JS DOM walk as last resort.
        located = False
        locator_factories = [
            lambda: self._page.get_by_role("button",   name=click_text, exact=False),
            lambda: self._page.get_by_role("link",     name=click_text, exact=False),
            lambda: self._page.get_by_role("menuitem", name=click_text, exact=False),
            lambda: self._page.get_by_role("option",   name=click_text, exact=False),
            lambda: self._page.get_by_role("tab",      name=click_text, exact=False),
            lambda: self._page.get_by_role("treeitem", name=click_text, exact=False),
            lambda: self._page.get_by_text(click_text, exact=False),
        ]

        for locator_fn in locator_factories:
            try:
                loc = locator_fn()
                # Wait for at least one matching element to be visible before clicking
                await loc.first.wait_for(state="visible", timeout=_CLICK_LOCATE_TIMEOUT_MS)
                await _random_delay(min_ms=200, max_ms=600)
                await loc.first.click(timeout=_CLICK_LOCATE_TIMEOUT_MS)
                located = True
                logger.debug("click: located '%s' via %s", click_text, locator_fn)
                break
            except Exception:  # noqa: BLE001
                continue

        # ── JS fallback: walk DOM for any element whose text contains click_text ──
        if not located:
            located = await self._click_by_js_text(click_text)
            if located:
                logger.debug("click: located '%s' via JS text walk", click_text)

        if not located:
            # Take a screenshot to aid debugging before returning failure
            try:
                png_bytes = await self._page.screenshot(type="png", full_page=False)
                debug_b64 = base64.b64encode(png_bytes).decode()
            except Exception:  # noqa: BLE001
                debug_b64 = ""
            result: dict[str, Any] = {
                "error":   f"Element with text '{click_text}' not found",
                "success": False,
                "action":  "click",
            }
            if debug_b64:
                result["screenshot_b64"] = debug_b64
            return result

        # ── Post-click: wait for page/SPA to settle ───────────────────────────
        # Step 1: wait for domcontentloaded in case the click triggered navigation.
        # Timeout is intentionally generous (15 s) to cover server-side login
        # processing + redirect chains (e.g. 302 → dashboard page).
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=_CLICK_LOAD_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            pass  # Ignore – page may not have triggered navigation

        # Step 2: wait for networkidle so that SPA AJAX responses (login API calls,
        # redirects, dynamic content updates) complete before we read the result.
        try:
            await self._page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            logger.debug("click: networkidle wait timed out after %dms", _NETWORK_IDLE_TIMEOUT_MS)

        # Step 3: extra SPA render wait so dynamic content (menus, modals) finishes
        await self._wait_for_spa_stable()

        # Step 4: human-like pause after action
        await _random_delay()

        url_after: str = self._page.url
        navigated: bool = url_after != url_before
        title: str = await self._page.title()

        # ── Detect error pages ────────────────────────────────────────────────
        error_info = await self._detect_error_page()

        result = {
            "action":    "click",
            "success":   True,
            "message":   f"Clicked element: \"{click_text}\"",
            "url":       url_after,
            "title":     title,
            "navigated": navigated,
        }

        if error_info:
            result["page_error"] = error_info
            logger.warning(
                "click: error page detected after clicking '%s': %s", click_text, error_info
            )

        # When a full-page navigation occurred (e.g. login redirect), eagerly
        # capture the landing page text so the summariser can describe the
        # post-login state without requiring a separate get_content step.
        if navigated:
            page_text = await self._extract_page_text()
            result["page_text"] = (page_text or "")[:_CLICK_NAV_TEXT_CHARS]
            result["url_before"] = url_before
            logger.info(
                "click: navigation detected %s → %s title=%r",
                url_before, url_after, title,
            )

        return result

    async def _action_type(self, task: "AgentTask") -> dict[str, Any]:
        """Fill a text field identified by CSS selector, label/placeholder text, or auto-detect.

        Resolution order:
          1. ``type_selector`` – exact CSS selector (fastest, most reliable).
          2. ``type_label``    – label text / placeholder / accessible name; tries
                                 get_by_label, get_by_placeholder and get_by_role in turn.
          3. Fallback          – fills the next unfilled visible input using
                                 Playwright's ``fill()`` (handles React controlled components).
        """
        selector: str = task.metadata.get("type_selector", "").strip()
        label:    str = task.metadata.get("type_label",    "").strip()
        text:     str = task.metadata.get("type_text",     "").strip()

        if not text:
            return {"error": "type_text not provided", "success": False, "action": "type"}

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        filled = False
        used_target = selector or label or "(auto-detected)"

        # ── 1. CSS selector ───────────────────────────────────────────────────
        if selector and not filled:
            try:
                await self._page.fill(selector, text, timeout=_TIMEOUT_MS)
                filled = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("type via selector %r failed: %s", selector, exc)

        # ── 2. Label / placeholder / accessible name ──────────────────────────
        if label and not filled:
            for locator_fn in (
                lambda: self._page.get_by_label(label, exact=False),
                lambda: self._page.get_by_placeholder(label, exact=False),
                lambda: self._page.get_by_role("textbox", name=label, exact=False),
            ):
                try:
                    loc = locator_fn()
                    await loc.first.fill(text, timeout=10_000)
                    filled = True
                    used_target = f"label: {label}"
                    break
                except Exception:  # noqa: BLE001
                    continue

        # ── 3. Auto-detect: next unfilled visible text input ──────────────────
        if not filled:
            try:
                # Use Playwright's fill() via JS evaluation to find the first
                # visible, non-disabled input that is currently empty (or has
                # a value shorter than the text we want to type) so that when
                # the agent sends two sequential type actions (e.g. email then
                # PIN) each action targets a different field rather than always
                # overwriting the same first input.
                # We use the native HTMLInputElement value setter so that
                # React / Vue controlled-component listeners fire correctly.
                await self._page.evaluate(
                    """(text) => {
                        const SKIP_TYPES = new Set([
                            'hidden', 'submit', 'button', 'checkbox', 'radio',
                            'file', 'image', 'reset', 'range', 'color'
                        ]);
                        const inputs = [...document.querySelectorAll(
                            'input:not([type=hidden]):not([type=submit]):not([type=button]),' +
                            'input:not([type=checkbox]):not([type=radio]), textarea'
                        )].filter(el => {
                            if (SKIP_TYPES.has((el.type || '').toLowerCase())) return false;
                            const style = window.getComputedStyle(el);
                            return (
                                style.display !== 'none' &&
                                style.visibility !== 'hidden' &&
                                parseFloat(style.opacity) > 0 &&
                                !el.disabled &&
                                !el.readOnly
                            );
                        });
                        // Prefer the first empty input; if all are filled pick the first one
                        const el = inputs.find(i => !i.value) || inputs[0];
                        if (!el) return;
                        el.focus();
                        // Use the native setter so React/Vue synthetic events fire properly
                        const nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value'
                        ).set;
                        nativeSetter.call(el, text);
                        el.dispatchEvent(new Event('input',  {bubbles: true, cancelable: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                        el.blur();
                    }""",
                    text,
                )
                filled = True
                used_target = "(auto-detected)"
            except Exception as exc:  # noqa: BLE001
                return {
                    "error":   f"Could not type text into any input field: {exc}",
                    "success": False,
                    "action":  "type",
                }

        # Small random delay after typing to simulate human data-entry pace
        await _random_delay(min_ms=200, max_ms=700)

        return {
            "action":   "type",
            "success":  filled,
            "selector": used_target,
            "message":  f"Typed text into {used_target}",
        }

    async def _action_scroll(self, task: "AgentTask") -> dict[str, Any]:
        direction: str = task.metadata.get("scroll_direction", "down").strip().lower()
        if direction not in ("up", "down"):
            direction = "down"

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        pixels = 600 if direction == "down" else -600
        await self._page.evaluate(f"window.scrollBy(0, {pixels})")

        return {
            "action":    "scroll",
            "success":   True,
            "direction": direction,
            "message":   f"Scrolled {direction} by {abs(pixels)} px",
        }

    async def _action_screenshot(self, task: "AgentTask") -> dict[str, Any]:
        """Capture the current viewport as a base64-encoded PNG."""
        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        png_bytes: bytes = await self._page.screenshot(type="png", full_page=False)
        b64 = base64.b64encode(png_bytes).decode()

        return {
            "action":         "screenshot",
            "success":        True,
            "screenshot_b64": b64,
            "url":            self._page.url,
            "message":        f"Screenshot captured ({len(png_bytes):,} bytes)",
        }

    async def _action_get_content(self, task: "AgentTask") -> dict[str, Any]:
        """Extract text content and accessibility tree from the current page."""
        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "get_content",
            }

        title     = await self._page.title()
        page_text = await self._extract_page_text()
        a11y_tree = await self._extract_page_a11y()

        return {
            "action":    "get_content",
            "success":   True,
            "title":     title,
            "url":       self._page.url,
            "page_text": page_text[:_MAX_PAGE_TEXT_CHARS],
            "a11y_tree": a11y_tree,
            "message":   f"Content extracted from {self._page.url} – \"{title}\"",
        }

    async def _action_extract_data(self, task: "AgentTask") -> dict[str, Any]:
        """Extract structured list data from the current page.

        Uses a CSS selector when provided; otherwise auto-detects common list
        patterns (``<li>``, ``<tr>``, ``<article>``, ARIA listitem/row).
        """
        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "extract_data",
            }

        selector:  str = task.metadata.get("extract_selector",  "").strip()
        attribute: str = task.metadata.get("extract_attribute", "text").strip()
        limit:     int = int(task.metadata.get("extract_limit", 50))

        try:
            if selector:
                items: list[str] = await self._page.evaluate(
                    """([sel, attr, lim]) => {
                        const els = [...document.querySelectorAll(sel)].slice(0, lim);
                        return els.map(el => {
                            if (attr === 'href') return el.href || el.getAttribute('href') || '';
                            if (attr === 'src')  return el.src  || el.getAttribute('src')  || '';
                            return (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                        }).filter(v => v);
                    }""",
                    [selector, attribute, limit],
                )
            else:
                # Auto-detect: try common list/row/article containers
                items = await self._page.evaluate(
                    """(lim) => {
                        const candidates = [
                            'article', '[role="listitem"]', '[role="row"]', 'li', 'tr'
                        ];
                        for (const sel of candidates) {
                            const els = [...document.querySelectorAll(sel)].slice(0, lim);
                            if (els.length > 2) {
                                return els
                                    .map(el => (el.innerText || el.textContent || '')
                                        .replace(/\\s+/g, ' ').trim())
                                    .filter(v => v.length > 3);
                            }
                        }
                        return [];
                    }""",
                    limit,
                )

            return {
                "action":   "extract_data",
                "success":  True,
                "items":    items,
                "count":    len(items),
                "selector": selector or "(auto)",
                "url":      self._page.url,
                "message":  f"Extracted {len(items)} items from {self._page.url}",
            }
        except Exception as exc:
            return {
                "action":  "extract_data",
                "success": False,
                "error":   str(exc),
                "items":   [],
            }

    # ── Page content helpers ──────────────────────────────────────────────────

    async def _wait_for_spa_stable(self) -> None:
        """Wait for SPA frameworks (React/Vue/Angular) to finish rendering.

        After a navigate or click, SPAs may need additional time beyond
        ``networkidle`` to complete their virtual-DOM reconciliation and mount
        interactive elements (menu items, modals, etc.) into the DOM.  This
        helper uses a short polling loop that checks whether the number of DOM
        nodes has stabilised, and falls back to a fixed delay if the page is
        already stable or if no browser page is open.
        """
        if self._page is None:
            return
        try:
            prev_count: int = -1
            for _ in range(6):  # up to ~3 s of polling
                await asyncio.sleep(0.5)
                count: int = await self._page.evaluate(
                    "() => document.querySelectorAll('*').length"
                )
                if count == prev_count:
                    break
                prev_count = count
        except Exception:  # noqa: BLE001
            # If evaluation fails (e.g. page navigating), just sleep briefly
            await asyncio.sleep(1.0)

    async def _click_by_js_text(self, text: str) -> bool:
        """Click the first visible DOM element whose text content contains *text*.

        This is the most permissive fallback for SPA menu items rendered as
        ``<div>`` or ``<span>`` elements with JavaScript click handlers that
        cannot be reached by Playwright's ARIA-role locators.  It searches the
        entire DOM tree, skips script/style/input elements, and prefers the most
        specific (deepest) matching element to avoid accidentally clicking a
        container that holds multiple items.

        Returns ``True`` if an element was found and clicked, ``False`` otherwise.
        """
        if self._page is None:
            return False
        try:
            clicked: bool = await self._page.evaluate(
                """([searchText, maxRatio]) => {
                    const lower = searchText.toLowerCase();
                    const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'INPUT', 'TEXTAREA']);

                    // Collect all elements whose direct/inner text matches.
                    // querySelectorAll('*') returns elements in document order (depth-first
                    // pre-order traversal), so later entries are deeper in the DOM tree.
                    const allEls = [...document.querySelectorAll('*')];
                    const candidates = allEls.filter(el => {
                        if (SKIP_TAGS.has(el.tagName)) return false;
                        const style = window.getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') return false;
                        if (parseFloat(style.opacity) <= 0) return false;
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return false;
                        const txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                        // Skip container elements whose combined text is much longer than
                        // the search term (they hold many children, not just the target item).
                        return txt.includes(lower) && txt.length < lower.length * maxRatio;
                    });

                    if (candidates.length === 0) return false;

                    // The last candidate in document order is the deepest (most specific)
                    // matching element – prefer it to avoid clicking a parent container
                    // that merely contains the target text among other children.
                    const target = candidates[candidates.length - 1];
                    target.click();
                    return true;
                }""",
                [text, _JS_CLICK_MAX_TEXT_RATIO],
            )
            return bool(clicked)
        except Exception as exc:  # noqa: BLE001
            logger.debug("_click_by_js_text: failed for '%s': %s", text, exc)
            return False

    async def _detect_error_page(self) -> str:
        """Check whether the current page is displaying an error state.

        Returns a human-readable description of the error if detected, or an
        empty string if the page appears normal.  Recognises common patterns
        using the module-level ``_ERROR_URL_SEGMENTS`` and
        ``_ERROR_CONTENT_PHRASES`` constants, which can be extended to support
        additional sites without modifying this method.
        """
        if self._page is None:
            return ""
        try:
            url = self._page.url
            url_lower = url.lower()
            page_text_raw = await self._extract_page_text()
            snippet = page_text_raw[:300] if page_text_raw else ""

            # URL-based detection (e.g. /500, /error, /not-found)
            if any(seg in url_lower for seg in _ERROR_URL_SEGMENTS):
                title = await self._page.title()
                return f"Error page detected at {url} (title: {title!r}). {snippet}"

            # Content-based detection: look for known error phrases
            page_text_lower = page_text_raw.lower()
            matched = [p for p in _ERROR_CONTENT_PHRASES if p in page_text_lower]
            if matched:
                return (
                    f"Possible error page at {url} "
                    f"(matched phrases: {matched}). Content: {snippet}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("_detect_error_page: check failed: %s", exc)
        return ""

    async def _extract_page_text(self) -> str:
        """Extract visible text from the current page, stripping scripts/styles."""
        assert self._page is not None  # noqa: S101
        try:
            text: str = await self._page.evaluate(
                """() => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript, svg').forEach(el => el.remove());
                    return (clone.innerText || clone.textContent || '').replace(/\\s+/g, ' ').trim();
                }"""
            )
            return text
        except Exception as exc:
            logger.warning("BrowserNavigatorTool: text extraction failed: %s", exc)
            return ""

    async def _extract_page_a11y(self) -> list[dict[str, str]]:
        """Return a simplified accessibility tree from the current page."""
        assert self._page is not None  # noqa: S101
        try:
            snapshot = await self._page.accessibility.snapshot(interesting_only=True)
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
            logger.warning("BrowserNavigatorTool: a11y extraction failed: %s", exc)
            return []

    async def _action_save_session(self, task: "AgentTask") -> dict[str, Any]:
        """Persist the current browser context's cookies & storage."""
        from src.memory.state import BrowserSessionStore

        session_url: str = task.metadata.get("session_url", "").strip()
        if not session_url:
            # Fall back to the current page URL
            if self._page is not None:
                session_url = self._page.url
        if not session_url:
            return {
                "error":   "session_url not provided and no page is open",
                "success": False,
                "action":  "save_session",
            }

        if self._context is None:
            return {
                "error":   "No browser context open; navigate to a page first",
                "success": False,
                "action":  "save_session",
            }

        state: dict = await self._context.storage_state()
        store = BrowserSessionStore()
        path  = store.save_session(session_url, state)

        return {
            "action":       "save_session",
            "success":      True,
            "session_path": str(path),
            "session_url":  session_url,
            "message":      f"Session saved to {path}",
        }
