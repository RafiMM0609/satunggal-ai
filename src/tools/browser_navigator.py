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
  - get_links()           : extract all navigable links (text + href) from the
                            current page for exploration / topic discovery
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
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import urlparse

from src.tools.base_tool import BaseTool

try:
    from playwright_stealth import Stealth as _Stealth
    _STEALTH_INSTANCE = _Stealth(
        # Override navigator.languages to Indonesian + English
        navigator_languages_override=("id-ID", "en-US"),
        # Keep navigator.platform as Win32 (common desktop fingerprint)
        navigator_platform_override="Win32",
        # Enable all evasions (the defaults cover webdriver, plugins, etc.)
    )
    _STEALTH_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _STEALTH_AVAILABLE = False
    _STEALTH_INSTANCE = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_BLOCKED_RESOURCES = {"media"}
_TIMEOUT_MS              = 30_000
_CLICK_LOAD_TIMEOUT_MS   = 15_000   # post-click domcontentloaded settle wait (covers login redirects)
_NETWORK_IDLE_TIMEOUT_MS = 15_000   # post-click/navigate networkidle wait (SPA-safe)
_MAX_PAGE_TEXT_CHARS     = 8_000    # truncation limit for get_content page text
_CLICK_NAV_TEXT_CHARS    = 3_000    # page-text snippet captured inside click result on navigation
_SPA_RENDER_WAIT_MS      = 3_000    # extra wait for SPA to render content after navigation/click
_CLICK_LOCATE_TIMEOUT_MS = 4_000    # timeout for each individual click locator attempt
_MAX_LOCATORS            = 60       # max interactive elements returned in get_content "locators"

# ── Anti-bot / stealth constants ──────────────────────────────────────────────
# Pool of realistic Chrome user-agent strings (Windows + macOS, recent versions).
# One is chosen at random per browser session to avoid a fixed fingerprint.
_UA_POOL = [
    # Chrome 124 / Windows 10
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 123 / Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome 124 / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 122 / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome 124 / Linux (Ubuntu)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]
# Pool of common desktop viewport sizes (width × height).
_VIEWPORT_POOL = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1280, "height": 800},
]

# Cloudflare / bot challenge page markers (title prefix and body class)
_BOT_CHALLENGE_TITLES = ("just a moment", "attention required", "ddos-guard", "robot check")
_BOT_CHALLENGE_BODY_MARKERS = (
    "cf-browser-verification",
    "cf_chl_",
    "__cf_chl_captcha",
    "ddos-guard",
    "perisai ddos",
)

# Captcha detection: iframe src patterns and text phrases that indicate a CAPTCHA challenge.
_CAPTCHA_IFRAME_PATTERNS = (
    "google.com/recaptcha",
    "recaptcha.net/recaptcha",
    "hcaptcha.com",
    "turnstile.cloudflare.com",
    "funcaptcha.com",
    "arkoselabs.com",
)
_CAPTCHA_TEXT_PHRASES = (
    "i'm not a robot",
    "saya bukan robot",
    "verify you are human",
    "verifikasi bahwa anda manusia",
    "complete the captcha",
    "selesaikan captcha",
    "recaptcha",
    "hcaptcha",
    "captcha challenge",
)

# Popup/overlay close: text labels that typically appear on dismiss/close buttons.
_POPUP_CLOSE_TEXTS = (
    "×", "✕", "✗", "✖", "close", "tutup", "dismiss",
    "tidak, terima kasih", "no thanks", "no, thanks", "skip",
    "lewati", "accept", "terima", "ok", "got it", "mengerti",
    "setuju", "agree", "allow", "izinkan", "continue", "lanjutkan",
    # Cookie consent / GDPR
    "accept all", "accept cookies", "i accept", "i agree",
    "saya setuju", "izinkan semua", "terima semua", "oke",
    "understood", "paham", "iya", "ya",
    # Notification permission
    "not now", "nanti saja", "don't allow", "block",
    "no, thank you", "tidak terima kasih",
)
# CSS selectors that commonly identify modal/popup close buttons or overlays.
_POPUP_CLOSE_SELECTORS = (
    "[data-dismiss='modal']",
    "[aria-label='Close']",
    "[aria-label='close']",
    "[aria-label='Tutup']",
    "[aria-label='Dismiss']",
    ".modal-close",
    ".close-button",
    ".btn-close",
    ".cookie-close",
    ".popup-close",
    ".overlay-close",
    "[class*='close']",
    "[id*='close']",
    # Cookie banners
    "#onetrust-accept-btn-handler",
    "#accept-cookies",
    ".cc-accept",
    "[data-action='accept-cookies']",
    "[data-testid='cookie-accept']",
    "button[data-gdpr-action='accept']",
)

# Full-page content (get_full_content): auto-scroll settings
_FULL_PAGE_SCROLL_PX        = 600   # pixels per scroll step when sweeping the full page
_FULL_PAGE_SCROLL_WAIT_S    = 0.8   # seconds to wait after each scroll for lazy content to render (↑ from 0.5)
_MAX_FULL_PAGE_SCROLL_STEPS = 30    # safety cap – stops auto-scroll after this many steps
_MAX_FULL_PAGE_TEXT_CHARS   = 80_000  # higher text cap for full-page content extraction (↑ from 50K)

# ARIA roles treated as interactive – used to build the compact locators list in get_content.
# These match Playwright's get_by_role() expectations, so the LLM can use the "name" value
# directly in click/type params without reading any raw HTML.
_INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio",
    "combobox", "option", "tab", "menuitem", "searchbox",
    "switch", "treeitem", "spinbutton",
})
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

# ── JS stealth script injected before every page load ─────────────────────────
# This overrides browser automation fingerprints that are detectable by
# anti-bot systems even when using playwright-stealth.
_STEALTH_INIT_SCRIPT = """
// 1. Hide webdriver flag
Object.defineProperty(navigator, 'webdriver', {get: () => undefined, configurable: true});

// 2. Fake plugins array (real browsers have at least 3)
const _pluginData = [
  {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format'},
  {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: ''},
  {name: 'Native Client', filename: 'internal-nacl-plugin', description: ''},
];
Object.defineProperty(navigator, 'plugins', {
  get: () => {
    const arr = _pluginData.map(p => {
      const plugin = Object.create(Plugin.prototype);
      Object.defineProperty(plugin, 'name', {get: () => p.name});
      Object.defineProperty(plugin, 'filename', {get: () => p.filename});
      Object.defineProperty(plugin, 'description', {get: () => p.description});
      Object.defineProperty(plugin, 'length', {get: () => 0});
      return plugin;
    });
    arr.item = (i) => arr[i];
    arr.namedItem = (name) => arr.find(p => p.name === name) || null;
    Object.defineProperty(arr, 'length', {get: () => _pluginData.length});
    return arr;
  }, configurable: true
});

// 3. Fake mimeTypes
Object.defineProperty(navigator, 'mimeTypes', {get: () => ({length: 2}), configurable: true});

// 4. Languages
Object.defineProperty(navigator, 'languages', {get: () => ['id-ID', 'en-US', 'en'], configurable: true});

// 5. chrome.runtime must exist and not throw
if (!window.chrome) window.chrome = {};
if (!window.chrome.runtime) window.chrome.runtime = {sendMessage: () => {}, connect: () => ({})};

// 6. Notifications permission should not be 'denied' by default (bot heuristic)
try {
  const origQuery = window.Notification && window.Notification.permission !== undefined
    ? null : null;
  if (navigator.permissions && navigator.permissions.query) {
    const origFn = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (params) => {
      if (params && params.name === 'notifications') {
        return Promise.resolve({state: 'default', onchange: null});
      }
      return origFn(params);
    };
  }
} catch(e) {}

// 7. Hardware concurrency (bots often report 1)
if (navigator.hardwareConcurrency === 1) {
  Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 4, configurable: true});
}

// 8. DeviceMemory (bots often report 0.25)
try {
  if (navigator.deviceMemory !== undefined && navigator.deviceMemory < 2) {
    Object.defineProperty(navigator, 'deviceMemory', {get: () => 8, configurable: true});
  }
} catch(e) {}
"""  # noqa: E501


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
                                           "get_content" | "get_full_content" |
                                           "get_links" | "extract_data" |
                                           "save_session" | "select_option"
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
        task.metadata["option_text"]     = "Kopi & Teh"          # for select_option
        task.metadata["option_selector"] = "input[name='cat']"   # for select_option (optional CSS selector)
        # check_captcha and close_popup require no extra metadata keys.

    Result keys
    -----------
    ``action``        : str  – action that was performed
    ``success``       : bool – whether the action succeeded
    ``screenshot_b64``: str  – base64-encoded PNG (only for "screenshot")
    ``session_path``  : str  – path to saved session file (only for "save_session")
    ``page_text``     : str  – visible page text ("get_content" / "get_full_content")
    ``a11y_tree``     : list – accessibility tree nodes ("get_content" / "get_full_content")
    ``full_page``     : bool – True only for "get_full_content" (signals complete text)
    ``scroll_steps``  : int  – scroll iterations used (only for "get_full_content")
    ``items``         : list – extracted data items (only for "extract_data")
    ``message``       : str  – human-readable outcome description
    ``error``         : str  – present only on failure
    """

    name = "browser_navigator"
    description = (
        "Stateful headless browser that maintains an open Playwright session across multiple calls. "
        "Supports navigate, click, type, scroll, screenshot, get_content, get_full_content, "
        "get_links, extract_data, save_session, select_option, check_captcha, and close_popup actions. "
        "Use web_reader first to obtain page content, then use this tool for interactive steps."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "browser_action": {
                "type": "string",
                "enum": [
                    "navigate", "click", "type", "scroll", "screenshot",
                    "get_content", "get_full_content", "get_links",
                    "extract_data", "save_session", "select_option",
                    "check_captcha", "close_popup",
                ],
                "description": "The browser action to perform (set in task.metadata['browser_action']).",
            },
            "target_url":        {"type": "string", "description": "URL for 'navigate' action."},
            "click_text":        {"type": "string", "description": "Visible text of the element to click."},
            "type_selector":     {"type": "string", "description": "CSS selector of the input field to type into."},
            "type_label":        {"type": "string", "description": "Label/placeholder text of the input field."},
            "type_text":         {"type": "string", "description": "Text to type into the field."},
            "scroll_direction":  {"type": "string", "enum": ["down", "up"], "description": "Direction to scroll."},
            "session_url":       {"type": "string", "description": "Base URL used as key when saving a session."},
            "extract_selector":  {"type": "string", "description": "CSS selector for data extraction."},
            "extract_attribute": {"type": "string", "description": "Attribute to extract ('text' or 'href')."},
            "extract_limit":     {"type": "integer", "description": "Maximum number of items to extract."},
            "option_text":       {"type": "string", "description": "Visible text of the option to select."},
            "option_selector":   {"type": "string", "description": "CSS selector of the <select> element."},
        },
        "required": ["browser_action"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "action":         {"type": "string", "description": "Action that was performed."},
            "success":        {"type": "boolean", "description": "Whether the action succeeded."},
            "screenshot_b64": {"type": "string", "description": "Base64-encoded PNG screenshot (screenshot action only)."},
            "session_path":   {"type": "string", "description": "Path to saved session file (save_session only)."},
            "page_text":      {"type": "string", "description": "Visible page text (get_content / get_full_content)."},
            "a11y_tree":      {"type": "array",  "description": "Accessibility tree nodes."},
            "full_page":      {"type": "boolean", "description": "True only for get_full_content."},
            "scroll_steps":   {"type": "integer", "description": "Scroll iterations used (get_full_content only)."},
            "items":          {"type": "array",  "description": "Extracted data items (extract_data only)."},
            "message":        {"type": "string", "description": "Human-readable outcome description."},
            "error":          {"type": "string", "description": "Present only on failure."},
        },
    }

    def __init__(self) -> None:
        self._playwright:      Any            = None
        self._browser:         Any            = None
        self._context:         Any            = None
        # ── Multi-tab state ────────────────────────────────────────────────
        # Replaces the old single ``self._page`` attribute.  All existing code
        # that reads ``self._page`` goes through the property getter below.
        self._tabs:            dict[str, Any] = {}   # tab_id → Page
        self._active_tab_id:   str            = ""   # currently active tab
        self._tab_counter:     int            = 0    # monotonic id counter
        # ── SPA API response capture ───────────────────────────────────────
        # Populated by the response event listener in _ensure_browser so that
        # get_content / get_full_content can include JSON API data alongside
        # the rendered page text for SPA-heavy websites.
        self._api_responses:   list[dict[str, Any]] = []

    # ── Active-page property (multi-tab compatibility shim) ───────────────────

    @property
    def _page(self) -> Any:
        """Return the currently active tab's Playwright Page, or ``None``."""
        return self._tabs.get(self._active_tab_id)

    @_page.setter
    def _page(self, value: Any) -> None:
        """Assign a page to the active tab slot (used by tests and close())."""
        if value is None:
            self._tabs.clear()
            self._active_tab_id = ""
        else:
            if not self._active_tab_id:
                self._active_tab_id = "tab_1"
            self._tabs[self._active_tab_id] = value

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
            elif action == "get_full_content":
                return await self._action_get_full_content(task)
            elif action == "get_links":
                return await self._action_get_links(task)
            elif action == "extract_data":
                return await self._action_extract_data(task)
            elif action == "save_session":
                return await self._action_save_session(task)
            elif action == "select_option":
                return await self._action_select_option(task)
            elif action == "check_captcha":
                return await self._action_check_captcha(task)
            elif action == "close_popup":
                return await self._action_close_popup(task)
            # ── New actions (upgrade) ──────────────────────────────────────
            elif action == "open_tab":
                return await self._action_open_tab(task)
            elif action == "switch_tab":
                return await self._action_switch_tab(task)
            elif action == "close_tab":
                return await self._action_close_tab(task)
            elif action == "wait_for_element":
                return await self._action_wait_for_element(task)
            elif action == "wait_for_navigation":
                return await self._action_wait_for_navigation(task)
            elif action == "fill_form":
                return await self._action_fill_form(task)
            elif action == "hover":
                return await self._action_hover(task)
            elif action == "drag_drop":
                return await self._action_drag_drop(task)
            elif action == "press_key":
                return await self._action_press_key(task)
            elif action == "upload_file":
                return await self._action_upload_file(task)
            elif action == "scrape_table":
                return await self._action_scrape_table(task)
            elif action == "assert_text":
                return await self._action_assert_text(task)
            elif action == "load_session":
                return await self._action_load_session(task)
            elif action == "clear_session":
                return await self._action_clear_session(task)
            elif action == "explore_parallel":
                return await self._action_explore_parallel(task)
            elif action == "compare_screenshot":
                return await self._action_compare_screenshot(task)
            elif action == "solve_captcha":
                return await self._action_solve_captcha(task)
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
        """Launch browser + context + page if not already open.

        On first call the browser is launched with stealth configuration:
          * Chromium flags that disable automation-detection markers
          * A random realistic user-agent and viewport from curated pools
          * Indonesian locale / Jakarta timezone to match target audience
          * playwright-stealth applied to the page (if available)
          * Custom JS init-script to override remaining fingerprint leaks
          * Lightweight API-response interceptor for SPA data capture
        """
        if self._page is not None:
            return  # already open

        # If a context exists but all tabs were closed, create a new page
        # without re-launching the whole browser.
        if self._context is not None:
            self._tab_counter += 1
            tab_id   = f"tab_{self._tab_counter}"
            new_page = await self._context.new_page()
            new_page.set_default_timeout(_TIMEOUT_MS)
            new_page.set_default_navigation_timeout(_TIMEOUT_MS)
            # Apply stealth + init script to the new page as well
            await self._apply_stealth(new_page)
            self._tabs[tab_id]   = new_page
            self._active_tab_id  = tab_id
            return

        from playwright.async_api import async_playwright  # lazy import

        rng = random.SystemRandom()
        ua       = rng.choice(_UA_POOL)
        viewport = rng.choice(_VIEWPORT_POOL)

        self._playwright = await async_playwright().start()

        # ── Stealth launch args ───────────────────────────────────────────────
        # These Chromium flags suppress well-known automation fingerprints that
        # headless-detection scripts inspect (AutomationControlled feature,
        # Blink expose-intl property, etc.)
        stealth_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-service-autorun",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--allow-running-insecure-content",
            f"--window-size={viewport['width']},{viewport['height']}",
        ]

        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=stealth_args,
        )

        # ── Context fingerprint ───────────────────────────────────────────────
        context_kwargs: dict[str, Any] = {
            "java_script_enabled": True,
            "accept_downloads":    False,
            "user_agent":          ua,
            "viewport":            viewport,
            "locale":              "id-ID",
            "timezone_id":         "Asia/Jakarta",
            "color_scheme":        "light",
            "extra_http_headers": {
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        }
        if session_path and Path(session_path).exists():
            context_kwargs["storage_state"] = session_path
            logger.debug("BrowserNavigatorTool: loading session from %s", session_path)

        self._context = await self._browser.new_context(**context_kwargs)

        # ── Resource blocking (RAM conservation) ─────────────────────────────
        async def _block(route, request):  # noqa: ANN001
            if request.resource_type in _BLOCKED_RESOURCES:
                await route.abort()
            else:
                await route.continue_()

        await self._context.route("**/*", _block)

        # ── Intercept API responses for SPA data capture ──────────────────────
        # Store captured JSON API responses so _extract_page_text can return
        # them alongside the rendered page text.  This is especially useful for
        # SPAs that render content from XHR/fetch API calls.
        self._api_responses: list[dict[str, Any]] = []

        async def _capture_api_response(response: Any) -> None:  # noqa: ANN001
            try:
                ct = response.headers.get("content-type", "")
                if "application/json" in ct and response.status == 200:
                    body = await response.json()
                    if isinstance(body, (dict, list)):
                        self._api_responses.append({
                            "url":    response.url,
                            "status": response.status,
                            "data":   body,
                        })
                        # Keep only the 5 most recent API responses to limit memory
                        if len(self._api_responses) > 5:
                            self._api_responses = self._api_responses[-5:]
            except Exception:  # noqa: BLE001
                pass  # Silently ignore non-JSON or unreadable responses

        self._context.on("response", _capture_api_response)

        self._tab_counter += 1
        tab_id   = f"tab_{self._tab_counter}"
        new_page = await self._context.new_page()
        new_page.set_default_timeout(_TIMEOUT_MS)
        new_page.set_default_navigation_timeout(_TIMEOUT_MS)

        # ── Apply playwright-stealth + custom JS init script ──────────────────
        await self._apply_stealth(new_page)

        self._tabs[tab_id]   = new_page
        self._active_tab_id  = tab_id

    async def _apply_stealth(self, page: Any) -> None:
        """Apply all stealth measures to a Playwright Page object.

        Combines playwright-stealth (if installed) with our custom JS init script
        that overrides remaining detectable fingerprint leaks.

        Args:
            page: A Playwright ``Page`` object (newly created, before first goto).
        """
        # 1. playwright-stealth v2: uses Stealth class with apply_stealth_async()
        if _STEALTH_AVAILABLE and _STEALTH_INSTANCE is not None:
            try:
                await _STEALTH_INSTANCE.apply_stealth_async(page)
                logger.debug("BrowserNavigatorTool: playwright-stealth applied")
            except Exception as exc:  # noqa: BLE001
                logger.warning("BrowserNavigatorTool: playwright-stealth failed: %s", exc)

        # 2. Custom JS init script: additional overrides not covered by stealth lib
        try:
            await page.add_init_script(script=_STEALTH_INIT_SCRIPT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BrowserNavigatorTool: add_init_script failed: %s", exc)

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
            self._playwright    = None
            self._browser       = None
            self._context       = None
            self._tabs.clear()
            self._active_tab_id = ""
            self._tab_counter   = 0

    def has_active_page(self) -> bool:
        """Return True when the browser has an open, non-closed page.

        Used by the ``follow_parent`` feature to decide whether to reuse the
        existing browser session or start a new one.
        """
        if self._page is None:
            return False
        try:
            return not self._page.is_closed()
        except Exception:  # noqa: BLE001
            return False

    async def save_current_session(self, base_url: str) -> Optional[str]:
        """Persist the current browser context's cookies & storage keyed by *base_url*.

        This is called automatically at the end of every web automation task so
        that login state is preserved across ``/reset`` commands and future tasks.

        Args:
            base_url: The scheme+host URL to key the session under
                      (e.g. ``"https://example.com"``).

        Returns:
            The path where the session file was written, or ``None`` if the
            browser context is not open.
        """
        if self._context is None:
            return None
        from src.memory.state import BrowserSessionStore
        store = BrowserSessionStore()
        try:
            state: dict = await self._context.storage_state()
            path = store.save_session(base_url, state)
            logger.info(
                "BrowserNavigatorTool: auto-saved session for %s → %s", base_url, path
            )
            return str(path)
        except Exception as exc:
            logger.warning(
                "BrowserNavigatorTool: failed to auto-save session for %s: %s", base_url, exc
            )
            return None

    # ── Action implementations ────────────────────────────────────────────────

    async def _action_navigate(self, task: "AgentTask") -> dict[str, Any]:
        url: str = task.metadata.get("target_url", "").strip()
        if not url:
            return {"error": "target_url not provided", "success": False, "action": "navigate"}

        session_path = task.metadata.get("session_path")
        # Auto-load a previously saved login session for this domain so that
        # the agent resumes as a logged-in user after /reset or across tasks.
        if not session_path and self._page is None:
            # Only auto-load when starting a fresh browser context (self._page is None).
            # Mid-task navigations reuse the already-open context, so we skip the lookup.
            from src.memory.state import BrowserSessionStore
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            store = BrowserSessionStore()
            candidate = store.get_session_path(base_url)
            if candidate.exists():
                session_path = str(candidate)
                logger.debug(
                    "BrowserNavigatorTool: auto-loading session for %s from %s",
                    base_url, session_path,
                )
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

        # Eagerly capture a page text snippet so the LLM can see the landing
        # page content without requiring a separate get_content step.
        # Truncated to _CLICK_NAV_TEXT_CHARS (same as click-navigation results);
        # _compact_result will further cap to _REACT_RESULT_TEXT_CHARS for the
        # ReAct context while the summariser benefits from the larger snapshot.
        page_text = await self._extract_page_text()

        # Detect error pages immediately after navigation so the LLM knows
        # the page is broken and can re-plan without wasting extra steps.
        error_info = await self._detect_error_page()

        result: dict[str, Any] = {
            "action":    "navigate",
            "success":   True,
            "url":       self._page.url,
            "title":     title,
            "page_text": (page_text or "")[:_CLICK_NAV_TEXT_CHARS],
            "message":   f"Navigated to {self._page.url} – \"{title}\"",
        }
        if error_info:
            result["page_error"] = error_info
            logger.warning("navigate: error page detected at %s: %s", self._page.url, error_info)
        return result

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
            lambda: self._page.get_by_role("listitem", name=click_text, exact=False),
            lambda: self._page.get_by_role("radio",    name=click_text, exact=False),
            lambda: self._page.get_by_role("checkbox", name=click_text, exact=False),
            lambda: self._page.get_by_label(click_text, exact=False),
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

        # ── Force-click fallback: use Playwright force option on first text match ──
        # Handles elements obscured by an overlay (e.g. a modal backdrop) or
        # elements that are in a scrollable container and didn't pass visibility
        # checks in previous strategies.
        if not located:
            located = await self._click_by_force(click_text)
            if located:
                logger.debug("click: located '%s' via force click", click_text)

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

        # Shared JS helper: set a value on any form element using the correct
        # native prototype setter to avoid "TypeError: Illegal invocation".
        # Each element type has its own prototype chain:
        #   <input>    → HTMLInputElement.prototype.value  (or .checked for checkbox/radio)
        #   <textarea> → HTMLTextAreaElement.prototype.value
        #   <select>   → HTMLSelectElement.prototype.value
        _JS_SET_VALUE = """
            function setNativeValue(el, value) {
                const tag  = el.tagName;
                const type = (el.type || '').toLowerCase();
                if (tag === 'TEXTAREA') {
                    // Must use HTMLTextAreaElement setter — HTMLInputElement setter
                    // on a textarea throws "TypeError: Illegal invocation".
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    setter.call(el, value);
                    el.dispatchEvent(new Event('input',  {bubbles: true, cancelable: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                } else if (tag === 'SELECT') {
                    // HTMLSelectElement.prototype.value setter for <select> dropdowns.
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLSelectElement.prototype, 'value'
                    ).set;
                    setter.call(el, value);
                    el.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                } else if (type === 'checkbox' || type === 'radio') {
                    // Checkbox / radio: set .checked, not .value.
                    const truthful = /^(true|1|yes|on)$/i.test(value);
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'checked'
                    ).set;
                    setter.call(el, truthful);
                    el.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                } else {
                    // All other <input> types (text, email, number, password, …)
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value'
                    ).set;
                    setter.call(el, value);
                    el.dispatchEvent(new Event('input',  {bubbles: true, cancelable: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                }
                el.blur();
            }
        """

        # ── 1. CSS selector ───────────────────────────────────────────────────
        if selector and not filled:
            try:
                await self._page.fill(selector, text, timeout=_TIMEOUT_MS)
                filled = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("type via selector %r failed (will try JS setter): %s", selector, exc)
                # Playwright fill() doesn't support select/checkbox/radio; fall
                # back to the native setter which handles all element types.
                try:
                    await self._page.evaluate(
                        f"""(args) => {{
                            {_JS_SET_VALUE}
                            const el = document.querySelector(args.selector);
                            if (!el) throw new Error('Element not found: ' + args.selector);
                            el.focus();
                            setNativeValue(el, args.value);
                        }}""",
                        {"selector": selector, "value": text},
                    )
                    filled = True
                except Exception as exc2:  # noqa: BLE001
                    logger.debug("type via selector JS fallback %r failed: %s", selector, exc2)

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

            # If Playwright fill() failed for all locators, attempt JS-based
            # native setter via aria-label / placeholder matching.
            if not filled:
                try:
                    await self._page.evaluate(
                        f"""(args) => {{
                            {_JS_SET_VALUE}
                            const needle = args.label.toLowerCase();
                            const candidates = [
                                ...document.querySelectorAll(
                                    'input, textarea, select'
                                )
                            ].filter(el => {{
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden') return false;
                                if (el.disabled) return false;
                                // Match against placeholder, aria-label, name, id
                                const attrs = [
                                    el.placeholder, el.getAttribute('aria-label'),
                                    el.name, el.id
                                ];
                                if (attrs.some(a => a && a.toLowerCase().includes(needle))) return true;
                                // Also match via associated <label for="..."> element text
                                if (el.id) {{
                                    const lbl = document.querySelector('label[for="' + el.id + '"]');
                                    if (lbl) {{
                                        const lblText = (lbl.textContent || '').toLowerCase();
                                        if (lblText.includes(needle)) return true;
                                    }}
                                }}
                                return false;
                            }});
                            const el = candidates[0];
                            if (!el) throw new Error('No element matched label: ' + args.label);
                            el.focus();
                            setNativeValue(el, args.value);
                        }}""",
                        {"label": label, "value": text},
                    )
                    filled = True
                    used_target = f"label: {label}"
                except Exception as exc:  # noqa: BLE001
                    logger.debug("type via label JS fallback %r failed: %s", label, exc)

        # ── 3. Auto-detect: next unfilled visible form field ──────────────────
        if not filled:
            try:
                # Find the first visible, non-disabled, unfilled form field
                # (input, textarea, or select) and set its value using the
                # correct native prototype setter for each element type so that
                # React / Vue controlled-component listeners fire correctly
                # without "TypeError: Illegal invocation".
                await self._page.evaluate(
                    f"""(text) => {{
                        {_JS_SET_VALUE}
                        const SKIP_TYPES = new Set([
                            'hidden', 'submit', 'button',
                            'file', 'image', 'reset', 'range', 'color'
                        ]);
                        const fields = [...document.querySelectorAll(
                            'input, textarea, select'
                        )].filter(el => {{
                            if (SKIP_TYPES.has((el.type || '').toLowerCase())) return false;
                            const style = window.getComputedStyle(el);
                            return (
                                style.display !== 'none' &&
                                style.visibility !== 'hidden' &&
                                parseFloat(style.opacity) > 0 &&
                                !el.disabled &&
                                !el.readOnly
                            );
                        }});
                        // Prefer the first unfilled field; fall back to the first one
                        const el = fields.find(f => !f.value) || fields[0];
                        if (!el) return;
                        el.focus();
                        setNativeValue(el, text);
                    }}""",
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
        """Extract text content and accessibility tree from the current page.

        In addition to the raw ``page_text`` and full ``a11y_tree``, the result
        contains a ``locators`` field – a compact list of **interactive elements
        only** (buttons, links, inputs, checkboxes, etc.) extracted from the
        accessibility tree.  The LLM can use ``locators[n]["name"]`` directly as
        the ``text`` parameter of a ``click`` action or the ``label`` parameter
        of a ``type`` action, without ever having to parse raw HTML.

        This implements the token-efficient "Accessibility Tree" strategy:
        instead of reading the full page HTML the agent reads a structured
        snapshot of interactive elements and targets them by role + name.

        For SPA-heavy websites this action also includes any JSON API responses
        intercepted during navigation in ``api_data`` so the LLM can access
        the raw data powering the page without needing extra steps.
        """
        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "get_content",
            }

        title     = await self._page.title()
        # Use light_scroll=True to trigger Intersection Observer lazy loaders
        # that only fire on the first scroll – common in React/Vue apps.
        page_text = await self._extract_page_text(light_scroll=True)
        a11y_tree = await self._extract_page_a11y()

        # Build a compact locators list containing only interactive elements.
        # These role names match Playwright's get_by_role() expectations, so
        # the LLM can use the "name" value directly in click/type params.
        locators = [
            n for n in a11y_tree
            if n.get("role", "").lower() in _INTERACTIVE_ROLES and n.get("name")
        ][:_MAX_LOCATORS]

        result: dict[str, Any] = {
            "action":    "get_content",
            "success":   True,
            "title":     title,
            "url":       self._page.url,
            "page_text": page_text[:_MAX_PAGE_TEXT_CHARS],
            "a11y_tree": a11y_tree,
            "locators":  locators,
            "message":   f"Content extracted from {self._page.url} – \"{title}\"",
        }

        # Include recent intercepted API responses for SPA data capture
        if self._api_responses:
            # Summarise each response (truncate large payloads to stay token-safe)
            result["api_data"] = [
                {
                    "url":  r["url"],
                    "data": r["data"] if isinstance(r["data"], list) else
                            {k: v for k, v in list(r["data"].items())[:30]}
                            if isinstance(r["data"], dict) else r["data"],
                }
                for r in self._api_responses[-3:]  # last 3 API calls
            ]

        return result

    async def _action_get_full_content(self, task: "AgentTask") -> dict[str, Any]:
        """Extract the COMPLETE page content by auto-scrolling from top to bottom.

        When the client explicitly requests **all** content on a page (e.g.
        "tampilkan semua konten", "berikan seluruh isi halaman"), this action:

        1. Scrolls back to the top of the page.
        2. Iteratively scrolls down by ``_FULL_PAGE_SCROLL_PX`` pixels at a time,
           pausing ``_FULL_PAGE_SCROLL_WAIT_S`` seconds after each step so that
           lazy-loaded (infinite-scroll) content can render into the DOM.
        3. Stops when the scroll position no longer advances (bottom reached) or
           after at most ``_MAX_FULL_PAGE_SCROLL_STEPS`` iterations.
        4. Waits for ``networkidle`` once more so that any XHR/fetch calls
           triggered by scroll events finish before we extract the final text.
        5. Extracts the full visible text from the now-complete DOM and returns it
           without the usual ``_MAX_PAGE_TEXT_CHARS`` truncation (capped at the
           much larger ``_MAX_FULL_PAGE_TEXT_CHARS`` instead).

        The result carries ``"full_page": True`` so downstream code (the agent
        summariser) can recognise that this is a comprehensive snapshot and
        allocate a larger text budget when building the LLM summary.
        Also includes any intercepted API responses in ``api_data``.
        """
        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "get_full_content",
            }

        # ── 1. Scroll to the very top so we start consistently ────────────────
        await self._page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(_FULL_PAGE_SCROLL_WAIT_S)

        # ── 2. Incrementally scroll to the bottom ────────────────────────────
        scroll_steps = 0
        prev_scroll_y: float = -1.0

        for _ in range(_MAX_FULL_PAGE_SCROLL_STEPS):
            scroll_info: dict[str, float] = await self._page.evaluate(
                """() => ({
                    scrollY:      window.scrollY,
                    scrollHeight: document.body.scrollHeight,
                    innerHeight:  window.innerHeight
                })"""
            )
            scroll_y      = scroll_info["scrollY"]
            scroll_height = scroll_info["scrollHeight"]
            inner_height  = scroll_info["innerHeight"]

            # Bottom reached when the viewport touches the end of the document
            if scroll_y + inner_height >= scroll_height:
                break

            # Safety: stop if position did not advance (non-scrollable page)
            if scroll_y == prev_scroll_y:
                break

            prev_scroll_y = scroll_y
            await self._page.evaluate(f"window.scrollBy(0, {_FULL_PAGE_SCROLL_PX})")
            await asyncio.sleep(_FULL_PAGE_SCROLL_WAIT_S)
            scroll_steps += 1

        # ── 3. Wait for networkidle after scrolling so XHR responses finish ───
        # SPAs often fire fetch/XHR on scroll (infinite scroll, lazy sections);
        # waiting here ensures that data loaded by scroll events is in the DOM.
        try:
            await self._page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            logger.debug("get_full_content: post-scroll networkidle timed out")

        # ── 4. Extract full content from the now-complete DOM ─────────────────
        title     = await self._page.title()
        page_text = await self._extract_page_text()
        a11y_tree = await self._extract_page_a11y()

        locators = [
            n for n in a11y_tree
            if n.get("role", "").lower() in _INTERACTIVE_ROLES and n.get("name")
        ][:_MAX_LOCATORS]

        char_count = len(page_text)
        logger.info(
            "BrowserNavigatorTool: get_full_content – %d scroll steps, %d chars, url=%s",
            scroll_steps, char_count, self._page.url,
        )

        result: dict[str, Any] = {
            "action":       "get_full_content",
            "success":      True,
            "title":        title,
            "url":          self._page.url,
            "page_text":    page_text[:_MAX_FULL_PAGE_TEXT_CHARS],
            "a11y_tree":    a11y_tree,
            "locators":     locators,
            "full_page":    True,
            "scroll_steps": scroll_steps,
            "message": (
                f"Full content extracted from {self._page.url} – \"{title}\" "
                f"({scroll_steps} scroll steps, {char_count:,} chars)"
            ),
        }

        # Include recent intercepted API responses for SPA data capture
        if self._api_responses:
            result["api_data"] = [
                {
                    "url":  r["url"],
                    "data": r["data"] if isinstance(r["data"], list) else
                            {k: v for k, v in list(r["data"].items())[:30]}
                            if isinstance(r["data"], dict) else r["data"],
                }
                for r in self._api_responses[-3:]
            ]

        return result


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

    async def _action_get_links(self, task: "AgentTask") -> dict[str, Any]:
        """Extract all navigable links from the current page.

        Returns a structured list of links with their display text and target
        URL, intended for the LLM to analyse and select the most relevant one
        during exploration tasks (e.g. browsing documentation sites to find a
        specific topic).

        Up to 200 unique links are returned (de-duplicated by href).
        ``javascript:``, ``mailto:``, and ``tel:`` hrefs are excluded.
        """
        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "get_links",
            }

        try:
            links: list[dict[str, str]] = await self._page.evaluate(
                """() => {
                    const seen = new Set();
                    return Array.from(document.querySelectorAll('a[href]'))
                        .map(a => ({
                            text: (a.innerText || a.textContent || a.title || '')
                                      .replace(/\\s+/g, ' ').trim().substring(0, 120),
                            href: a.href,
                        }))
                        .filter(l =>
                            l.text &&
                            l.href &&
                            !l.href.startsWith('javascript:') &&
                            !l.href.startsWith('mailto:') &&
                            !l.href.startsWith('tel:') &&
                            !seen.has(l.href) &&
                            seen.add(l.href)
                        )
                        .slice(0, 200);
                }"""
            )
            return {
                "action":  "get_links",
                "success": True,
                "url":     self._page.url,
                "links":   links,
                "count":   len(links),
                "message": f"Extracted {len(links)} links from {self._page.url}",
            }
        except Exception as exc:
            return {
                "action":  "get_links",
                "success": False,
                "error":   str(exc),
                "links":   [],
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
            for _ in range(4):  # up to ~2 s of polling
                await asyncio.sleep(0.5)
                count: int = await self._page.evaluate(
                    "() => document.querySelectorAll('*').length"
                )
                if count == prev_count:
                    break
                prev_count = count
        except Exception:  # noqa: BLE001
            # If evaluation fails (e.g. page navigating), just sleep briefly
            await asyncio.sleep(0.5)

    async def _click_by_js_text(self, text: str) -> bool:
        """Click the first visible DOM element whose text content contains *text*.

        This is the most permissive fallback for SPA menu items rendered as
        ``<div>`` or ``<span>`` elements with JavaScript click handlers that
        cannot be reached by Playwright's ARIA-role locators.  It searches the
        entire DOM tree, skips script/style/input elements, and prefers the most
        specific (deepest) matching element to avoid accidentally clicking a
        container that holds multiple items.

        Elements inside scrollable modal containers may be below the visible
        fold but still present in the DOM; this method scrolls them into view
        before dispatching the click so that partially-clipped elements are
        handled correctly.

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
                        // Allow elements clipped by a scrollable modal container:
                        // only exclude elements that have truly zero layout dimensions
                        // (e.g. aria-hidden or CSS-collapsed elements), not those that
                        // are merely scrolled out of the visible viewport area.
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 && rect.height === 0) return false;
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
                    // Scroll the element into view within any parent scroll containers
                    // (handles elements below the fold inside modals/drawers).
                    target.scrollIntoView({block: 'nearest', inline: 'nearest'});
                    target.click();
                    return true;
                }""",
                [text, _JS_CLICK_MAX_TEXT_RATIO],
            )
            return bool(clicked)
        except Exception as exc:  # noqa: BLE001
            logger.debug("_click_by_js_text: failed for '%s': %s", text, exc)
            return False

    async def _click_by_force(self, text: str) -> bool:
        """Force-click the first Playwright locator matching *text* regardless of
        overlay or out-of-viewport state.

        This is a last-resort fallback for elements that are present and visible
        in the DOM (e.g. a category button inside a scrollable modal) but fail
        Playwright's normal interactability checks due to being partially clipped
        or overlapped by a semi-transparent backdrop.

        Uses Playwright's ``force=True`` option which bypasses actionability
        checks and dispatches the pointer event directly to the element.

        Returns ``True`` if the locator was found and clicked, ``False`` otherwise.
        """
        if self._page is None:
            return False
        try:
            loc = self._page.get_by_text(text, exact=False)
            count = await loc.count()
            if count == 0:
                return False
            await loc.first.scroll_into_view_if_needed(timeout=_CLICK_LOCATE_TIMEOUT_MS)
            await loc.first.click(force=True, timeout=_CLICK_LOCATE_TIMEOUT_MS)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("_click_by_force: failed for '%s': %s", text, exc)
            return False

    async def _select_native_auto(self, option_text: str) -> bool:
        """Scan every visible native ``<select>`` element on the page and select
        the first ``<option>`` whose display text or value matches *option_text*.

        This handles the common case where the caller did not supply an explicit
        CSS selector for the ``<select>`` element.  Because closed ``<option>``
        elements have zero bounding-box dimensions they are invisible to the
        normal JS DOM walk, so a dedicated ``querySelectorAll('select')`` pass is
        required.

        Uses the ``HTMLSelectElement.prototype.value`` native setter so that
        React / Vue controlled-component ``onChange`` listeners fire correctly.

        Returns ``True`` if a match was found and selected, ``False`` otherwise.
        """
        if self._page is None:
            return False
        try:
            selected: bool = await self._page.evaluate(
                """(needle) => {
                    const lower = needle.toLowerCase().trim();
                    for (const sel of document.querySelectorAll('select')) {
                        const style = window.getComputedStyle(sel);
                        if (style.display === 'none' || sel.disabled) continue;
                        // Find the first option whose visible text or value matches
                        const opt = [...sel.options].find(o => {
                            const optText  = (o.text  || '').toLowerCase().trim();
                            const optValue = (o.value || '').toLowerCase().trim();
                            return optText.includes(lower) || optValue.includes(lower);
                        });
                        if (!opt) continue;
                        // Use the native prototype setter to trigger framework listeners
                        const setter = Object.getOwnPropertyDescriptor(
                            window.HTMLSelectElement.prototype, 'value'
                        ).set;
                        setter.call(sel, opt.value);
                        sel.dispatchEvent(new Event('input',  {bubbles: true, cancelable: true}));
                        sel.dispatchEvent(new Event('change', {bubbles: true, cancelable: true}));
                        return true;
                    }
                    return false;
                }""",
                option_text,
            )
            return bool(selected)
        except Exception as exc:  # noqa: BLE001
            logger.debug("_select_native_auto: failed for '%s': %s", option_text, exc)
            return False

    async def _action_select_option(self, task: "AgentTask") -> dict[str, Any]:
        """Select a custom UI option (category button, radio-group item, etc.).

        Unlike the standard ``click`` action, this action is specialised for
        custom form widgets where the option is rendered as a clickable ``<div>``,
        ``<button>``, or ``<label>`` element rather than a native ``<select>``
        option.  It employs a tiered strategy that handles:

        1. Native ``<select>`` dropdowns – sets the value directly via JS and
           fires the ``change`` event so React/Vue controlled components update.
        2. Playwright ARIA locators – tries ``option``, ``radio``, ``checkbox``,
           ``button``, and ``listitem`` roles for accessible custom widgets.
        3. Text-based label lookup – clicks a ``<label>`` element whose text
           matches the option, which triggers the associated ``<input>``.
        4. JS DOM walk – scrolls the matching element into view inside any parent
           scrollable container (modal/drawer) and dispatches a native click.
        5. Force click – bypasses overlay/interactability checks as a last resort.

        Metadata keys:
            ``option_text``     – visible label text to select (required).
            ``option_selector`` – optional CSS selector for the ``<select>``
                                  element when a native dropdown is known.
        """
        option_text: str = task.metadata.get("option_text", "").strip()
        selector:    str = task.metadata.get("option_selector", "").strip()

        if not option_text:
            return {
                "error":   "option_text not provided",
                "success": False,
                "action":  "select_option",
            }

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        await self._wait_for_spa_stable()

        selected = False

        # ── 1. Native <select> via CSS selector ──────────────────────────────
        if selector and not selected:
            try:
                await self._page.select_option(selector, label=option_text, timeout=_TIMEOUT_MS)
                selected = True
                logger.debug("select_option: selected %r via native select %r", option_text, selector)
            except Exception:  # noqa: BLE001
                # Try setting by value as a fallback
                try:
                    await self._page.select_option(selector, value=option_text, timeout=_TIMEOUT_MS)
                    selected = True
                except Exception:  # noqa: BLE001
                    pass

        # ── 1b. Auto-detect native <select> without explicit CSS selector ────────
        if not selected:
            selected = await self._select_native_auto(option_text)
            if selected:
                logger.debug("select_option: selected %r via auto-detected native <select>", option_text)

        # ── 2. ARIA roles for custom option widgets ───────────────────────────
        if not selected:
            for role in ("option", "radio", "checkbox", "button", "listitem", "menuitem"):
                try:
                    loc = self._page.get_by_role(role, name=option_text, exact=False)  # type: ignore[arg-type]
                    await loc.first.wait_for(state="visible", timeout=_CLICK_LOCATE_TIMEOUT_MS)
                    await loc.first.click(timeout=_CLICK_LOCATE_TIMEOUT_MS)
                    selected = True
                    logger.debug("select_option: selected %r via role=%r", option_text, role)
                    break
                except Exception:  # noqa: BLE001
                    continue

        # ── 3. Label-based lookup (label → associated input) ─────────────────
        if not selected:
            try:
                loc = self._page.get_by_label(option_text, exact=False)
                await loc.first.wait_for(state="visible", timeout=_CLICK_LOCATE_TIMEOUT_MS)
                await loc.first.click(timeout=_CLICK_LOCATE_TIMEOUT_MS)
                selected = True
                logger.debug("select_option: selected %r via label", option_text)
            except Exception:  # noqa: BLE001
                pass

        # ── 4. JS DOM walk with scroll-into-view ─────────────────────────────
        if not selected:
            selected = await self._click_by_js_text(option_text)
            if selected:
                logger.debug("select_option: selected %r via JS text walk", option_text)

        # ── 5. Force click as last resort ────────────────────────────────────
        if not selected:
            selected = await self._click_by_force(option_text)
            if selected:
                logger.debug("select_option: selected %r via force click", option_text)

        if not selected:
            try:
                png_bytes = await self._page.screenshot(type="png", full_page=False)
                debug_b64 = base64.b64encode(png_bytes).decode()
            except Exception:  # noqa: BLE001
                debug_b64 = ""
            result: dict[str, Any] = {
                "error":   f"Option '{option_text}' not found in the page",
                "success": False,
                "action":  "select_option",
            }
            if debug_b64:
                result["screenshot_b64"] = debug_b64
            return result

        # Wait for React/SPA to process the selection before the next step
        await self._wait_for_spa_stable()
        await _random_delay(min_ms=200, max_ms=700)

        return {
            "action":      "select_option",
            "success":     True,
            "option_text": option_text,
            "message":     f"Selected option: \"{option_text}\"",
        }

    async def _detect_error_page(self) -> str:
        """Check whether the current page is displaying an error or bot-challenge state.

        Returns a human-readable description of the error if detected, or an
        empty string if the page appears normal.  Recognises:
          * Standard HTTP error pages (/500, /404, /error, etc.)
          * Known error text phrases (server error, CORS, etc.)
          * Cloudflare / DDoS Guard bot-challenge pages ("Just a moment...")
        """
        if self._page is None:
            return ""
        try:
            url = self._page.url
            url_lower = url.lower()
            title = await self._page.title()
            title_lower = title.lower()
            page_text_raw = await self._extract_page_text()
            snippet = page_text_raw[:300] if page_text_raw else ""

            # ── Bot / Cloudflare challenge detection ──────────────────────────
            if any(marker in title_lower for marker in _BOT_CHALLENGE_TITLES):
                return (
                    f"⚠️ Bot challenge page detected (title: {title!r}) at {url}. "
                    "Cloudflare or similar anti-bot system requires browser verification. "
                    "The agent cannot proceed automatically."
                )
            page_html_snippet = ""
            try:
                page_html_snippet = await self._page.evaluate(
                    "() => document.body ? document.body.innerHTML.slice(0, 1000) : ''"
                )
            except Exception:  # noqa: BLE001
                pass
            if any(marker in page_html_snippet for marker in _BOT_CHALLENGE_BODY_MARKERS):
                return (
                    f"⚠️ Bot challenge page detected (body marker) at {url}. "
                    "Anti-bot verification required. The agent cannot proceed automatically."
                )

            # ── URL-based detection (e.g. /500, /error, /not-found) ───────────
            if any(seg in url_lower for seg in _ERROR_URL_SEGMENTS):
                return f"Error page detected at {url} (title: {title!r}). {snippet}"

            # ── Content-based detection: look for known error phrases ─────────
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

    async def _extract_page_text(self, light_scroll: bool = False) -> str:
        """Extract visible text from the current page, with multi-layer fallback.

        Extraction strategy (tried in order, first non-empty result wins):
          1. Primary: ``innerText`` of a clean clone of ``<body>`` (fast, accurate
             for server-rendered HTML and React/Vue apps that write to the DOM).
          2. JS semantic extraction: collect ``textContent`` from semantic HTML tags
             (``p``, ``h1``–``h6``, ``li``, ``td``, ``blockquote``, ``article``,
             ``section``, etc.) and de-duplicate lines.  Works on SPAs that
             build content inside container divs without ``innerText`` support.
          3. Shadow DOM traversal: search ``shadowRoot`` of custom elements for
             text – handles websites built with Web Components (e.g. Lit, Stencil).
          4. ``<noscript>`` fallback: some SSR pages place the full text in
             ``<noscript>`` tags; extract when all other methods fail.

        Args:
            light_scroll: When True perform a tiny 200 px scroll before extracting
                          to trigger lazy-content loaders that activate on first
                          scroll (e.g. Intersection Observer-based widgets).
        """
        assert self._page is not None  # noqa: S101

        # Optional: tiny scroll to wake up Intersection Observer lazy loaders
        if light_scroll:
            try:
                await self._page.evaluate("window.scrollBy(0, 200)")
                await asyncio.sleep(0.3)
            except Exception:  # noqa: BLE001
                pass

        # ── Strategy 1: primary innerText extraction ──────────────────────────
        try:
            text: str = await self._page.evaluate(
                """() => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript, svg').forEach(el => el.remove());
                    return (clone.innerText || clone.textContent || '').replace(/\\s+/g, ' ').trim();
                }"""
            )
            if text and len(text.strip()) > 150:
                return text
        except Exception as exc:
            logger.debug("_extract_page_text: primary extraction failed: %s", exc)
            text = ""

        # ── Strategy 2: JS semantic tag extraction ────────────────────────────
        # Collects textContent from meaningful content tags, deduplicates lines.
        # Effective for React/Vue SPAs where innerText may return empty or minimal.
        try:
            semantic_text: str = await self._page.evaluate(
                """() => {
                    const tags = ['p','h1','h2','h3','h4','h5','h6','li','td','th',
                                  'blockquote','article','section','main','aside',
                                  'figcaption','caption','dt','dd','pre','code'];
                    const lines = new Set();
                    tags.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => {
                            const t = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                            if (t.length > 10) lines.add(t);
                        });
                    });
                    return Array.from(lines).join('\\n');
                }"""
            )
            if semantic_text and len(semantic_text.strip()) > 150:
                logger.debug("_extract_page_text: using semantic JS fallback")
                return semantic_text
        except Exception as exc:
            logger.debug("_extract_page_text: semantic extraction failed: %s", exc)

        # ── Strategy 3: Shadow DOM traversal ─────────────────────────────────
        # Websites built with Web Components (Lit, Stencil, FAST) render
        # content inside shadow roots invisible to normal DOM queries.
        try:
            shadow_text: str = await self._page.evaluate(
                """() => {
                    function extractFromShadow(root) {
                        const texts = [];
                        const walker = document.createTreeWalker(
                            root, NodeFilter.SHOW_TEXT,
                            { acceptNode: n => n.nodeValue.trim().length > 3
                              ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP }
                        );
                        let node;
                        while ((node = walker.nextNode())) texts.push(node.nodeValue.trim());
                        // Recurse into shadow roots
                        root.querySelectorAll('*').forEach(el => {
                            if (el.shadowRoot) texts.push(...extractFromShadow(el.shadowRoot).split('\\n'));
                        });
                        return texts.join('\\n');
                    }
                    return extractFromShadow(document.body || document.documentElement);
                }"""
            )
            if shadow_text and len(shadow_text.strip()) > 150:
                logger.debug("_extract_page_text: using Shadow DOM fallback")
                return shadow_text
        except Exception as exc:
            logger.debug("_extract_page_text: shadow DOM extraction failed: %s", exc)

        # ── Strategy 4: <noscript> tag fallback ───────────────────────────────
        # Some SSR frameworks (Next.js, Nuxt) embed full page content inside
        # <noscript> tags for non-JS environments; grab this as a last resort.
        try:
            noscript_text: str = await self._page.evaluate(
                """() => {
                    const ns = document.querySelectorAll('noscript');
                    return Array.from(ns).map(n => n.innerText || n.textContent || '').join('\\n').trim();
                }"""
            )
            if noscript_text and len(noscript_text.strip()) > 50:
                logger.debug("_extract_page_text: using noscript fallback")
                return noscript_text
        except Exception as exc:
            logger.debug("_extract_page_text: noscript extraction failed: %s", exc)

        # Return whatever the primary strategy produced (may be short/empty)
        return text or ""

    async def _extract_page_a11y(self) -> list[dict[str, Any]]:
        """Return a simplified accessibility tree from the current page.

        Each node contains at minimum ``role`` and ``name``.  For interactive
        elements the following optional fields are also included when present:

        * ``value``       – current value of an input or combobox.
        * ``checked``     – boolean checked state for checkboxes/radios.
        * ``disabled``    – True when the element is non-interactive.
        * ``description`` – accessible description (aria-describedby text).
        """
        assert self._page is not None  # noqa: S101
        try:
            snapshot = await self._page.accessibility.snapshot(interesting_only=True)
            if not snapshot:
                return []

            nodes: list[dict[str, Any]] = []

            def _walk(node: dict) -> None:
                role = node.get("role", "")
                name = (node.get("name") or "").strip()
                if role and name:
                    node_info: dict[str, Any] = {"role": role, "name": name}
                    # Include state info so the LLM can infer current element state
                    if node.get("checked") is not None:
                        node_info["checked"] = node["checked"]
                    if node.get("value"):
                        node_info["value"] = str(node["value"])[:80]
                    if node.get("disabled"):
                        node_info["disabled"] = True
                    if node.get("description"):
                        node_info["description"] = (node["description"] or "")[:80]
                    nodes.append(node_info)
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

    # ── Multi-tab management ──────────────────────────────────────────────────

    async def _action_open_tab(self, task: "AgentTask") -> dict[str, Any]:
        """Open a new browser tab, optionally navigating it to a URL immediately."""
        url: str = task.metadata.get("target_url", "").strip()

        await self._ensure_browser()
        assert self._context is not None  # noqa: S101

        self._tab_counter += 1
        tab_id   = f"tab_{self._tab_counter}"
        new_page = await self._context.new_page()
        new_page.set_default_timeout(_TIMEOUT_MS)
        new_page.set_default_navigation_timeout(_TIMEOUT_MS)
        self._tabs[tab_id]   = new_page
        self._active_tab_id  = tab_id

        result: dict[str, Any] = {
            "action":  "open_tab",
            "success": True,
            "tab_id":  tab_id,
            "tabs":    list(self._tabs.keys()),
            "message": f"Opened new tab '{tab_id}'",
        }

        if url:
            await new_page.goto(url, wait_until="domcontentloaded")
            try:
                await new_page.wait_for_load_state(
                    "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS
                )
            except Exception:  # noqa: BLE001
                pass
            await self._wait_for_spa_stable()
            await _random_delay(min_ms=200, max_ms=600)
            title = await new_page.title()
            result.update({
                "url":     new_page.url,
                "title":   title,
                "message": f"Opened new tab '{tab_id}' and navigated to {new_page.url}",
            })

        return result

    async def _action_switch_tab(self, task: "AgentTask") -> dict[str, Any]:
        """Switch the active tab to the given ``tab_id``."""
        tab_id: str = task.metadata.get("tab_id", "").strip()
        if not tab_id:
            return {"error": "tab_id not provided", "success": False, "action": "switch_tab"}
        if tab_id not in self._tabs:
            return {
                "error":   f"Tab '{tab_id}' does not exist. Available: {list(self._tabs.keys())}",
                "success": False,
                "action":  "switch_tab",
            }
        self._active_tab_id = tab_id
        page = self._tabs[tab_id]
        try:
            title = await page.title()
            url   = page.url
        except Exception:  # noqa: BLE001
            title, url = "", ""
        return {
            "action":  "switch_tab",
            "success": True,
            "tab_id":  tab_id,
            "tabs":    list(self._tabs.keys()),
            "url":     url,
            "title":   title,
            "message": f"Switched to tab '{tab_id}' – {url!r}",
        }

    async def _action_close_tab(self, task: "AgentTask") -> dict[str, Any]:
        """Close a tab by ID, or the active tab when no ID is given."""
        tab_id: str = task.metadata.get("tab_id", self._active_tab_id).strip()
        if not tab_id or tab_id not in self._tabs:
            return {
                "error":   f"Tab '{tab_id}' does not exist",
                "success": False,
                "action":  "close_tab",
            }
        try:
            await self._tabs[tab_id].close()
        except Exception:  # noqa: BLE001
            pass
        del self._tabs[tab_id]
        # Switch to the most recently opened remaining tab, if any
        if self._tabs:
            self._active_tab_id = list(self._tabs.keys())[-1]
        else:
            self._active_tab_id = ""
        return {
            "action":          "close_tab",
            "success":         True,
            "closed_tab_id":   tab_id,
            "active_tab_id":   self._active_tab_id,
            "tabs":            list(self._tabs.keys()),
            "message":         (
                f"Closed tab '{tab_id}'. "
                f"Active tab: '{self._active_tab_id}'" if self._active_tab_id
                else f"Closed tab '{tab_id}'. No tabs remaining."
            ),
        }

    # ── Wait actions ──────────────────────────────────────────────────────────

    async def _action_wait_for_element(self, task: "AgentTask") -> dict[str, Any]:
        """Wait until an element matching a text or CSS selector is visible.

        Metadata:
            ``wait_text``       – visible text to wait for (partial match)
            ``wait_selector``   – CSS selector to wait for
            ``wait_timeout_ms`` – max wait time in ms (default 10 000)
        """
        text:       str = task.metadata.get("wait_text",     "").strip()
        selector:   str = task.metadata.get("wait_selector", "").strip()
        timeout_ms: int = int(task.metadata.get("wait_timeout_ms", 10_000))

        if not text and not selector:
            return {
                "error":   "wait_text or wait_selector not provided",
                "success": False,
                "action":  "wait_for_element",
            }

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "wait_for_element",
            }

        try:
            if selector:
                await self._page.wait_for_selector(
                    selector, state="visible", timeout=timeout_ms
                )
                found_by = f"selector: {selector}"
            else:
                await (
                    self._page.get_by_text(text, exact=False)
                    .first.wait_for(state="visible", timeout=timeout_ms)
                )
                found_by = f"text: {text!r}"
            return {
                "action":   "wait_for_element",
                "success":  True,
                "found_by": found_by,
                "url":      self._page.url,
                "message":  f"Element visible: {found_by}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "wait_for_element",
                "success": False,
                "error":   f"Element not visible within {timeout_ms} ms: {exc}",
            }

    async def _action_wait_for_navigation(self, task: "AgentTask") -> dict[str, Any]:
        """Wait for the page to finish navigating, optionally matching a URL pattern.

        Metadata:
            ``wait_url_pattern`` – substring that must appear in the final URL
            ``wait_timeout_ms``  – max wait time in ms (default 15 000)
        """
        url_pattern: str = task.metadata.get("wait_url_pattern", "").strip()
        timeout_ms:  int = int(task.metadata.get("wait_timeout_ms", 15_000))

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "wait_for_navigation",
            }

        try:
            await self._page.wait_for_load_state(
                "domcontentloaded", timeout=timeout_ms
            )
            try:
                await self._page.wait_for_load_state(
                    "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS
                )
            except Exception:  # noqa: BLE001
                pass
            current_url = self._page.url
            if url_pattern and url_pattern.lower() not in current_url.lower():
                return {
                    "action":  "wait_for_navigation",
                    "success": False,
                    "error":   (
                        f"URL '{current_url}' does not contain pattern '{url_pattern}'"
                    ),
                    "url":     current_url,
                }
            return {
                "action":      "wait_for_navigation",
                "success":     True,
                "url":         current_url,
                "title":       await self._page.title(),
                "url_matched": url_pattern,
                "message":     f"Navigation completed to {current_url}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "wait_for_navigation",
                "success": False,
                "error":   f"Navigation wait failed: {exc}",
            }

    # ── Bulk form fill ────────────────────────────────────────────────────────

    async def _action_fill_form(self, task: "AgentTask") -> dict[str, Any]:
        """Fill multiple form fields in a single action step.

        Metadata:
            ``form_fields`` – JSON array (or Python list) of
                              ``{"label": "...", "selector": "...", "text": "..."}``
                              dicts.  ``label`` and ``selector`` are both optional
                              but at least one should be provided per field.
        """
        raw_fields = task.metadata.get("form_fields", [])
        if isinstance(raw_fields, str):
            try:
                raw_fields = json.loads(raw_fields)
            except Exception:
                return {
                    "error":   "form_fields must be a JSON array",
                    "success": False,
                    "action":  "fill_form",
                }

        if not raw_fields:
            return {
                "error":   "form_fields is empty",
                "success": False,
                "action":  "fill_form",
            }

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "fill_form",
            }

        from src.memory.state import AgentTask as _AgentTask  # local import avoids cycle

        results: list[dict[str, Any]] = []
        for field in raw_fields:
            label    = field.get("label",    "")
            selector = field.get("selector", "")
            text     = field.get("text",     "")

            sub_task = _AgentTask(
                user_input=task.user_input, session_id=task.session_id
            )
            sub_task.metadata.update({
                "browser_action": "type",
                "type_label":     label,
                "type_selector":  selector,
                "type_text":      text,
            })
            sub_result = await self._action_type(sub_task)
            results.append({
                "label":   label or selector or "(auto)",
                "success": sub_result.get("success", False),
                "message": sub_result.get("message", sub_result.get("error", "")),
            })
            await _random_delay(min_ms=100, max_ms=400)

        filled = sum(1 for r in results if r["success"])
        return {
            "action":  "fill_form",
            "success": all(r["success"] for r in results),
            "results": results,
            "filled":  filled,
            "total":   len(results),
            "url":     self._page.url,
            "message": f"Filled {filled}/{len(results)} form fields",
        }

    # ── Mouse actions ────────────────────────────────────────

    async def _action_hover(self, task: "AgentTask") -> dict[str, Any]:
        """Hover over an element to reveal dropdown menus or tooltips.

        Metadata:
            ``hover_text``     – visible text of the element to hover over
            ``hover_selector`` – CSS selector of the element to hover over
        """
        text:     str = task.metadata.get("hover_text",     "").strip()
        selector: str = task.metadata.get("hover_selector", "").strip()

        if not text and not selector:
            return {
                "error":   "hover_text or hover_selector not provided",
                "success": False,
                "action":  "hover",
            }

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        try:
            if selector:
                await self._page.hover(selector, timeout=_TIMEOUT_MS)
                target = f"selector: {selector}"
            else:
                await self._page.get_by_text(text, exact=False).first.hover(
                    timeout=_TIMEOUT_MS
                )
                target = f"text: {text!r}"
            await self._wait_for_spa_stable()
            await _random_delay(min_ms=200, max_ms=600)
            return {
                "action":  "hover",
                "success": True,
                "target":  target,
                "url":     self._page.url,
                "message": f"Hovered over {target}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "hover",
                "success": False,
                "error":   f"Hover failed for {text or selector!r}: {exc}",
            }

    async def _action_drag_drop(self, task: "AgentTask") -> dict[str, Any]:
        """Drag an element from one location and drop it onto another.

        Metadata:
            ``drag_selector`` – CSS selector of the element to drag (required)
            ``drop_selector`` – CSS selector of the drop target (required)
        """
        source: str = task.metadata.get("drag_selector", "").strip()
        target: str = task.metadata.get("drop_selector", "").strip()

        if not source or not target:
            return {
                "error":   "drag_selector and drop_selector are both required",
                "success": False,
                "action":  "drag_drop",
            }

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        try:
            await self._page.drag_and_drop(source, target, timeout=_TIMEOUT_MS)
            await self._wait_for_spa_stable()
            await _random_delay(min_ms=200, max_ms=600)
            return {
                "action":  "drag_drop",
                "success": True,
                "source":  source,
                "target":  target,
                "url":     self._page.url,
                "message": f"Dragged '{source}' onto '{target}'",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "drag_drop",
                "success": False,
                "error":   f"Drag-drop failed: {exc}",
            }

    # ── Keyboard ──────────────────────────────────────────────────────────────

    async def _action_press_key(self, task: "AgentTask") -> dict[str, Any]:
        """Press a keyboard key or combination (e.g. ``Enter``, ``Escape``, ``Control+A``).

        Metadata:
            ``key`` – Playwright key name (required), e.g. ``"Enter"``,
                      ``"Escape"``, ``"Tab"``, ``"ArrowDown"``, ``"Control+a"``
        """
        key: str = task.metadata.get("key", "").strip()
        if not key:
            return {"error": "key not provided", "success": False, "action": "press_key"}

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        try:
            await self._page.keyboard.press(key)
            await self._wait_for_spa_stable()
            await _random_delay(min_ms=100, max_ms=400)
            return {
                "action":  "press_key",
                "success": True,
                "key":     key,
                "url":     self._page.url,
                "message": f"Pressed key: {key!r}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "press_key",
                "success": False,
                "error":   f"Key press failed for {key!r}: {exc}",
            }

    # ── File upload ───────────────────────────────────────────────────────────

    async def _action_upload_file(self, task: "AgentTask") -> dict[str, Any]:
        """Set a file input to the specified local file path.

        Metadata:
            ``upload_filepath`` – absolute path to the file to upload (required)
            ``upload_selector`` – CSS selector for ``<input type="file">``
            ``upload_label``    – accessible label text for the upload field
        """
        selector: str = task.metadata.get("upload_selector", "").strip()
        label:    str = task.metadata.get("upload_label",    "").strip()
        filepath: str = task.metadata.get("upload_filepath", "").strip()

        if not filepath:
            return {
                "error":   "upload_filepath not provided",
                "success": False,
                "action":  "upload_file",
            }
        if not Path(filepath).exists():
            return {
                "error":   f"File not found: {filepath}",
                "success": False,
                "action":  "upload_file",
            }

        await self._ensure_browser()
        assert self._page is not None  # noqa: S101

        try:
            if selector:
                await self._page.set_input_files(selector, filepath, timeout=_TIMEOUT_MS)
                target = selector
            elif label:
                loc = self._page.get_by_label(label, exact=False)
                await loc.first.set_input_files(filepath)
                target = f"label: {label}"
            else:
                await self._page.set_input_files('input[type="file"]', filepath)
                target = 'input[type="file"] (auto-detected)'
            await _random_delay(min_ms=200, max_ms=600)
            return {
                "action":   "upload_file",
                "success":  True,
                "filepath": filepath,
                "target":   target,
                "url":      self._page.url,
                "message":  f"Uploaded '{Path(filepath).name}' to {target}",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "upload_file",
                "success": False,
                "error":   f"File upload failed: {exc}",
            }

    # ── Structured table extraction ───────────────────────────────────────────

    async def _action_scrape_table(self, task: "AgentTask") -> dict[str, Any]:
        """Extract an HTML table as a list of row-dicts keyed by column headers.

        Metadata:
            ``extract_selector`` – CSS selector for the table (default: ``"table"``)
            ``extract_limit``    – max rows to return (default: 100)
        """
        selector: str = (
            task.metadata.get("extract_selector", "table").strip() or "table"
        )
        limit: int = int(task.metadata.get("extract_limit", 100))

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "scrape_table",
            }

        try:
            rows: list[dict[str, str]] = await self._page.evaluate(
                """([sel, lim]) => {
                    const table = document.querySelector(sel);
                    if (!table) return [];
                    const headers = [];
                    const headRow = table.querySelector('thead tr');
                    if (headRow) {
                        headRow.querySelectorAll('th, td').forEach(
                            c => headers.push((c.innerText || c.textContent || '').trim())
                        );
                    }
                    const rows = [];
                    for (const tr of Array.from(
                            table.querySelectorAll('tbody tr, tr')
                        ).slice(0, lim)) {
                        const cells = tr.querySelectorAll('td, th');
                        if (cells.length === 0) continue;
                        const row = {};
                        cells.forEach((cell, i) => {
                            const key = (headers[i] !== undefined && headers[i] !== '')
                                ? headers[i]
                                : String(i);
                            row[key] = (cell.innerText || cell.textContent || '').trim();
                        });
                        rows.push(row);
                    }
                    return rows;
                }""",
                [selector, limit],
            )
            return {
                "action":   "scrape_table",
                "success":  True,
                "rows":     rows,
                "count":    len(rows),
                "selector": selector,
                "url":      self._page.url,
                "message":  f"Extracted {len(rows)} rows from table '{selector}'",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "scrape_table",
                "success": False,
                "error":   f"Table extraction failed: {exc}",
            }

    # ── Text assertion ────────────────────────────────────────────────────────

    async def _action_assert_text(self, task: "AgentTask") -> dict[str, Any]:
        """Assert that a text string is (or is not) present on the current page.

        Returns ``success: True`` when the assertion passes.  On failure the
        result contains ``"error"`` so the LLM can trigger re-planning.

        Metadata:
            ``assert_text_value``   – text to search for (required)
            ``assert_should_exist`` – ``"true"`` (default) or ``"false"``
        """
        text:         str  = task.metadata.get("assert_text_value", "").strip()
        should_exist: bool = (
            str(task.metadata.get("assert_should_exist", "true")).lower() != "false"
        )

        if not text:
            return {
                "error":   "assert_text_value not provided",
                "success": False,
                "action":  "assert_text",
            }

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "assert_text",
            }

        try:
            page_text = await self._extract_page_text()
            found     = text.lower() in page_text.lower()
            passed    = (should_exist and found) or (not should_exist and not found)
            result: dict[str, Any] = {
                "action":       "assert_text",
                "success":      passed,
                "text":         text,
                "should_exist": should_exist,
                "found":        found,
                "url":          self._page.url,
            }
            if passed:
                result["message"] = (
                    f"PASS: '{text}' {'found' if found else 'not found'} as expected"
                )
            else:
                msg = (
                    f"FAIL: '{text}' was {'NOT found' if should_exist else 'found'} "
                    f"on {self._page.url}"
                )
                result["message"] = msg
                result["error"]   = msg
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                "action":  "assert_text",
                "success": False,
                "error":   f"assert_text check failed: {exc}",
            }

    # ── Session management ────────────────────────────────────────────────────

    async def _action_load_session(self, task: "AgentTask") -> dict[str, Any]:
        """Explicitly load a saved browser session by URL key.

        Recreates the browser context using the saved cookies/storage so that
        the agent resumes as a logged-in user on the target website.

        Metadata:
            ``session_url`` – base URL key used when the session was saved
                              (e.g. ``"https://example.com"``).
        """
        from src.memory.state import BrowserSessionStore

        session_url: str = task.metadata.get("session_url", "").strip()
        if not session_url and self._page is not None:
            session_url = self._page.url
        if not session_url:
            return {
                "error":   "session_url not provided",
                "success": False,
                "action":  "load_session",
            }

        store = BrowserSessionStore()
        path  = store.get_session_path(session_url)
        if not path.exists():
            return {
                "action":      "load_session",
                "success":     False,
                "error":       f"No saved session found for '{session_url}'",
                "session_url": session_url,
            }

        # Close existing context (if any) and rebuild with the saved session
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:  # noqa: BLE001
                pass
            self._tabs.clear()
            self._active_tab_id = ""

        if self._browser is None:
            if self._playwright is None:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

        context_kwargs: dict[str, Any] = {
            "java_script_enabled": True,
            "accept_downloads":    False,
            "storage_state":       str(path),
        }
        self._context = await self._browser.new_context(**context_kwargs)

        async def _block(route, request):  # noqa: ANN001
            if request.resource_type in _BLOCKED_RESOURCES:
                await route.abort()
            else:
                await route.continue_()

        await self._context.route("**/*", _block)

        self._tab_counter += 1
        tab_id   = f"tab_{self._tab_counter}"
        new_page = await self._context.new_page()
        new_page.set_default_timeout(_TIMEOUT_MS)
        new_page.set_default_navigation_timeout(_TIMEOUT_MS)
        self._tabs[tab_id]   = new_page
        self._active_tab_id  = tab_id

        return {
            "action":       "load_session",
            "success":      True,
            "session_url":  session_url,
            "session_path": str(path),
            "tab_id":       tab_id,
            "message":      f"Session loaded for '{session_url}' from {path}",
        }

    async def _action_clear_session(self, task: "AgentTask") -> dict[str, Any]:
        """Delete the saved browser session for a domain.

        Metadata:
            ``session_url`` – base URL key of the session to delete
        """
        from src.memory.state import BrowserSessionStore

        session_url: str = task.metadata.get("session_url", "").strip()
        if not session_url:
            return {
                "error":   "session_url not provided",
                "success": False,
                "action":  "clear_session",
            }
        store = BrowserSessionStore()
        store.delete_session(session_url)
        return {
            "action":      "clear_session",
            "success":     True,
            "session_url": session_url,
            "message":     f"Session cleared for '{session_url}'",
        }

    # ── Parallel exploration ──────────────────────────────────────────────────

    async def _action_explore_parallel(self, task: "AgentTask") -> dict[str, Any]:
        """Open several URLs in parallel tabs and return a content summary of each.

        The LLM can use this to evaluate multiple candidate pages at once and
        choose the most relevant one without sequential navigation.

        Metadata:
            ``explore_urls``  – JSON array or comma-separated list of URLs
                                (maximum 5 to cap resource usage)
            ``explore_query`` – keyword describing what content is sought
        """
        raw_urls = task.metadata.get("explore_urls", [])
        if isinstance(raw_urls, str):
            try:
                raw_urls = json.loads(raw_urls)
            except Exception:
                raw_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        query: str = task.metadata.get("explore_query", "").strip()

        if not raw_urls:
            return {
                "error":   "explore_urls not provided",
                "success": False,
                "action":  "explore_parallel",
            }

        urls = list(raw_urls)[:5]   # cap at 5 tabs

        await self._ensure_browser()
        assert self._context is not None  # noqa: S101

        async def _fetch_url(url: str) -> dict[str, Any]:
            try:
                self._tab_counter += 1
                tid      = f"tab_{self._tab_counter}"
                pg       = await self._context.new_page()
                pg.set_default_timeout(_TIMEOUT_MS)
                pg.set_default_navigation_timeout(_TIMEOUT_MS)
                self._tabs[tid] = pg
                await pg.goto(url, wait_until="domcontentloaded")
                try:
                    await pg.wait_for_load_state(
                        "networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS
                    )
                except Exception:  # noqa: BLE001
                    pass
                title = await pg.title()
                text: str = await pg.evaluate(
                    """() => {
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll(
                            'script, style, noscript, svg'
                        ).forEach(el => el.remove());
                        return (clone.innerText || clone.textContent || '')
                            .replace(/\\s+/g, ' ').trim();
                    }"""
                )
                return {
                    "url":       pg.url,
                    "title":     title,
                    "tab_id":    tid,
                    "page_text": text[:1500],
                }
            except Exception as exc:  # noqa: BLE001
                return {"url": url, "error": str(exc)}

        summaries: list[dict[str, Any]] = list(
            await asyncio.gather(*(_fetch_url(u) for u in urls))
        )

        # Switch active tab to the first successfully loaded page
        for s in summaries:
            if not s.get("error") and s.get("tab_id") in self._tabs:
                self._active_tab_id = s["tab_id"]
                break

        return {
            "action":  "explore_parallel",
            "success": True,
            "query":   query,
            "results": summaries,
            "count":   len(summaries),
            "tabs":    list(self._tabs.keys()),
            "message": f"Explored {len(summaries)} URLs in parallel",
        }

    # ── Visual comparison ─────────────────────────────────────────────────────

    async def _action_compare_screenshot(self, task: "AgentTask") -> dict[str, Any]:
        """Capture a screenshot and compare it with a previously stored baseline.

        On the first call (no baseline exists) the screenshot is saved as the
        baseline.  On subsequent calls the pixel-level diff ratio is reported so
        the agent can alert the user when the page has visually changed.

        Metadata:
            ``compare_url``    – string key for storing the baseline
                                 (defaults to the current page URL)
            ``diff_threshold`` – float 0–1; changes above this value trigger
                                 ``"changed": true`` (default 0.02 = 2 %)
        """
        from src.tools.screenshot_store import ScreenshotStore

        compare_key: str   = task.metadata.get("compare_url", "").strip()
        threshold:   float = float(task.metadata.get("diff_threshold", 0.02))

        if self._page is None:
            return {
                "error":   "No page is open; navigate to a URL first",
                "success": False,
                "action":  "compare_screenshot",
            }
        if not compare_key:
            compare_key = self._page.url

        try:
            png_bytes: bytes = await self._page.screenshot(
                type="png", full_page=False
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "error":   f"Screenshot capture failed: {exc}",
                "success": False,
                "action":  "compare_screenshot",
            }

        store    = ScreenshotStore()
        baseline = store.load(compare_key)

        if baseline is None:
            store.save(compare_key, png_bytes)
            return {
                "action":        "compare_screenshot",
                "success":       True,
                "changed":       False,
                "baseline_set":  True,
                "compare_key":   compare_key,
                "url":           self._page.url,
                "message":       f"Baseline screenshot saved for '{compare_key}'",
            }

        diff_ratio = store.diff_ratio(baseline, png_bytes)
        changed    = diff_ratio > threshold
        b64        = base64.b64encode(png_bytes).decode()

        if changed:
            # Update the baseline to the new snapshot for next comparison
            store.save(compare_key, png_bytes)

        return {
            "action":         "compare_screenshot",
            "success":        True,
            "changed":        changed,
            "diff_ratio":     round(diff_ratio, 4),
            "threshold":      threshold,
            "compare_key":    compare_key,
            "url":            self._page.url,
            "screenshot_b64": b64,
            "message": (
                f"⚠️ Visual change detected! diff={diff_ratio:.1%} > threshold={threshold:.1%}"
                if changed
                else f"No significant change (diff={diff_ratio:.1%} ≤ {threshold:.1%})"
            ),
        }

    # ── Auto CAPTCHA solver ───────────────────────────────────────────────────

    async def _action_solve_captcha(self, task: "AgentTask") -> dict[str, Any]:
        """Attempt to automatically solve a CAPTCHA using an external solver API.

        Detects reCAPTCHA v2 or hCaptcha on the current page, extracts the site
        key, submits it to the configured solver service (2captcha / CapMonster),
        and injects the resulting token back into the page via JavaScript.

        Falls back to requesting manual assistance when:
          * No ``CAPTCHA_API_KEY`` env var is set.
          * The site key cannot be extracted from the DOM.
          * The solver service fails or times out.

        Result keys:
            ``solved``          – bool, True when token was injected
            ``captcha_type``    – ``"recaptcha_v2"`` | ``"hcaptcha"`` | ``"unknown"``
            ``fallback_manual`` – bool, True when human assistance is needed
        """
        from src.tools.captcha_solver import solve_recaptcha_v2, solve_hcaptcha

        if self._page is None:
            return {
                "error":           "No page is open; navigate to a URL first",
                "success":         False,
                "action":          "solve_captcha",
                "solved":          False,
                "fallback_manual": True,
            }

        # ── Detect CAPTCHA type and extract site key ──────────────────────
        captcha_info: dict[str, str] = await self._page.evaluate(
            """() => {
                // reCAPTCHA v2
                const rcEl = document.querySelector(
                    '.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"]'
                );
                if (rcEl) {
                    const sk = rcEl.dataset?.sitekey
                        || rcEl.getAttribute('data-sitekey')
                        || (rcEl.src?.match(/[?&]k=([^&]+)/) || [])[1]
                        || '';
                    return { type: 'recaptcha_v2', site_key: sk };
                }
                // hCaptcha
                const hcEl = document.querySelector(
                    '.h-captcha, [data-hcaptcha-sitekey], iframe[src*="hcaptcha"]'
                );
                if (hcEl) {
                    const sk = hcEl.dataset?.sitekey
                        || hcEl.getAttribute('data-hcaptcha-sitekey')
                        || '';
                    return { type: 'hcaptcha', site_key: sk };
                }
                return { type: '', site_key: '' };
            }"""
        )

        captcha_type = captcha_info.get("type", "")
        site_key     = captcha_info.get("site_key", "")
        site_url     = self._page.url

        if not captcha_type:
            detected, _desc = await self._detect_captcha()
            if not detected:
                return {
                    "action":           "solve_captcha",
                    "success":          True,
                    "solved":           False,
                    "captcha_detected":  False,
                    "fallback_manual":  False,
                    "message":          "No CAPTCHA detected on the current page.",
                }
            captcha_type = "unknown"

        if not site_key:
            return {
                "action":           "solve_captcha",
                "success":          True,
                "solved":           False,
                "captcha_type":     captcha_type,
                "captcha_detected": True,
                "fallback_manual":  True,
                "message": (
                    "⚠️ CAPTCHA terdeteksi tetapi site key tidak dapat diekstrak. "
                    "Mohon selesaikan CAPTCHA secara manual, lalu ulangi perintah."
                ),
            }

        # ── Submit to solver service ──────────────────────────────────────
        token: Optional[str] = None
        if captcha_type == "recaptcha_v2":
            token = await solve_recaptcha_v2(site_url, site_key)
        elif captcha_type == "hcaptcha":
            token = await solve_hcaptcha(site_url, site_key)

        if not token:
            return {
                "action":           "solve_captcha",
                "success":          True,
                "solved":           False,
                "captcha_type":     captcha_type,
                "captcha_detected": True,
                "fallback_manual":  True,
                "message": (
                    "⚠️ CAPTCHA terdeteksi. Solver otomatis tidak tersedia atau gagal. "
                    "Mohon selesaikan CAPTCHA secara manual, lalu ulangi perintah."
                ),
            }

        # ── Inject token ──────────────────────────────────────────────────
        token_js = json.dumps(token)
        if captcha_type == "hcaptcha":
            inject_js = f"""() => {{
                const el = document.querySelector(
                    'textarea[name="h-captcha-response"], [name="h-captcha-response"]'
                );
                if (el) el.value = {token_js};
            }}"""
        else:
            inject_js = f"""() => {{
                const el = document.querySelector(
                    '#g-recaptcha-response, [name="g-recaptcha-response"]'
                );
                if (el) {{ el.style.display = 'block'; el.value = {token_js}; }}
                // Trigger page callback if reCAPTCHA framework is present
                try {{
                    const cfg = window.___grecaptcha_cfg?.clients;
                    if (cfg) {{
                        const first = Object.values(cfg)[0];
                        if (first?.l?.l?.callback)
                            first.l.l.callback({token_js});
                    }}
                }} catch(e) {{}}
            }}"""

        try:
            await self._page.evaluate(inject_js)
        except Exception as exc:  # noqa: BLE001
            logger.warning("solve_captcha: token injection failed: %s", exc)
            return {
                "action":           "solve_captcha",
                "success":          False,
                "solved":           False,
                "captcha_type":     captcha_type,
                "captcha_detected": True,
                "fallback_manual":  True,
                "error":            f"Token injection failed: {exc}",
            }

        await _random_delay(min_ms=500, max_ms=1200)
        return {
            "action":           "solve_captcha",
            "success":          True,
            "solved":           True,
            "captcha_type":     captcha_type,
            "captcha_detected": True,
            "fallback_manual":  False,
            "url":              self._page.url,
            "message":          f"CAPTCHA ({captcha_type}) solved and token injected.",
        }

    # ── Captcha detection ─────────────────────────────────────────────────────    async def _detect_captcha(self) -> tuple[bool, str]:
        """Detect whether the current page presents a CAPTCHA challenge.

        Checks for:
        * Known captcha iframe sources (reCAPTCHA, hCaptcha, Turnstile, etc.)
        * Common CAPTCHA-related text phrases in the visible page content.

        Returns:
            ``(detected, description)`` where ``detected`` is a bool and
            ``description`` is a human-readable explanation (empty when not
            detected).
        """
        if self._page is None:
            return False, ""
        try:
            # Check iframes whose src attribute matches known captcha patterns
            iframe_src: str = await self._page.evaluate(
                """() => {
                    const iframes = [...document.querySelectorAll('iframe[src]')];
                    return iframes.map(f => f.src).join(' ');
                }"""
            )
            iframe_lower = iframe_src.lower()
            for pattern in _CAPTCHA_IFRAME_PATTERNS:
                if pattern in iframe_lower:
                    return True, f"CAPTCHA detected: '{pattern}' iframe found on page."

            # Also check page text for well-known CAPTCHA phrases
            page_text = await self._extract_page_text()
            page_lower = page_text.lower()
            for phrase in _CAPTCHA_TEXT_PHRASES:
                if phrase in page_lower:
                    return True, f"CAPTCHA detected: phrase '{phrase}' found in page content."
        except Exception as exc:  # noqa: BLE001
            logger.debug("_detect_captcha: check failed: %s", exc)
        return False, ""

    async def _action_check_captcha(self, task: "AgentTask") -> dict[str, Any]:
        """Check whether the current page shows a CAPTCHA challenge.

        This action is intended to be called by the agent after navigating or
        clicking when there is a reason to suspect a CAPTCHA might appear.  If
        a CAPTCHA is found, the result carries ``captcha_detected: True`` and a
        human-readable description so the LLM can report to the user that manual
        assistance is required.

        Result keys:
            ``captcha_detected`` – bool
            ``captcha_type``     – short identifier (e.g. ``"recaptcha"``)
            ``message``          – human-readable status
        """
        if self._page is None:
            return {
                "error":            "No page is open; navigate to a URL first",
                "success":          False,
                "action":           "check_captcha",
                "captcha_detected": False,
            }

        detected, description = await self._detect_captcha()
        captcha_type = ""
        if detected:
            for pattern in _CAPTCHA_IFRAME_PATTERNS:
                if pattern in description.lower():
                    captcha_type = pattern.split(".")[0]
                    break
            if not captcha_type:
                captcha_type = "unknown"

        return {
            "action":           "check_captcha",
            "success":          True,
            "captcha_detected": detected,
            "captcha_type":     captcha_type,
            "url":              self._page.url,
            "message": (
                description
                if detected
                else "No CAPTCHA detected on the current page."
            ),
        }

    # ── Popup / overlay dismissal ─────────────────────────────────────────────

    async def _dismiss_popup(self) -> tuple[bool, str]:
        """Attempt to close visible pop-ups, overlays, and cookie banners.

        Tries the following strategies in order:
        1. Click elements matching ``_POPUP_CLOSE_SELECTORS`` (common close
           button CSS patterns).
        2. Click elements whose visible text matches ``_POPUP_CLOSE_TEXTS``
           (e.g. "×", "Close", "Tutup").
        3. Press ``Escape`` to dismiss keyboard-dismissible modals.

        Returns:
            ``(dismissed, description)`` where ``dismissed`` is True if at
            least one close action succeeded, and ``description`` summarises
            what was closed.
        """
        if self._page is None:
            return False, ""

        # ── 1. CSS selector-based close buttons ───────────────────────────────
        for sel in _POPUP_CLOSE_SELECTORS:
            try:
                loc = self._page.locator(sel)
                count = await loc.count()
                if count > 0:
                    first_visible = loc.first
                    await first_visible.wait_for(state="visible", timeout=3_000)
                    await first_visible.click(timeout=3_000)
                    await asyncio.sleep(0.5)
                    logger.debug("_dismiss_popup: closed via selector %r", sel)
                    return True, f"Popup closed via selector '{sel}'."
            except Exception:  # noqa: BLE001
                continue

        # ── 2. Text-based close button search ────────────────────────────────
        for close_text in _POPUP_CLOSE_TEXTS:
            try:
                # Use get_by_role for × / button patterns
                for role in ("button", "link"):
                    loc = self._page.get_by_role(role, name=close_text, exact=False)  # type: ignore[arg-type]
                    count = await loc.count()
                    if count > 0:
                        await loc.first.wait_for(state="visible", timeout=3_000)
                        await loc.first.click(timeout=3_000)
                        await asyncio.sleep(0.5)
                        logger.debug("_dismiss_popup: closed via text %r role=%r", close_text, role)
                        return True, f"Popup closed via button text '{close_text}'."
            except Exception:  # noqa: BLE001
                continue

        # ── 3. Escape key as last resort ─────────────────────────────────────
        try:
            await self._page.keyboard.press("Escape")
            await asyncio.sleep(0.5)
            logger.debug("_dismiss_popup: sent Escape key")
            # Escape doesn't confirm dismissal, so check if any overlay disappeared
            overlay_gone: bool = await self._page.evaluate(
                """() => {
                    const overlays = document.querySelectorAll(
                        '.modal, .overlay, [role="dialog"], [aria-modal="true"]'
                    );
                    return overlays.length === 0;
                }"""
            )
            if overlay_gone:
                return True, "Popup dismissed via Escape key."
        except Exception:  # noqa: BLE001
            pass

        return False, ""

    async def _action_close_popup(self, task: "AgentTask") -> dict[str, Any]:
        """Close visible pop-ups, modal overlays, or cookie consent banners.

        Tries multiple strategies (CSS selector, text-based button search, Escape
        key) to dismiss any overlay that may be blocking form interaction.  The
        agent should call this action when it suspects a pop-up is preventing
        it from clicking or typing in a form field.

        Result keys:
            ``dismissed`` – bool, True if a popup was successfully closed
            ``message``   – human-readable outcome
        """
        if self._page is None:
            return {
                "error":     "No page is open; navigate to a URL first",
                "success":   False,
                "action":    "close_popup",
                "dismissed": False,
            }

        dismissed, description = await self._dismiss_popup()

        return {
            "action":    "close_popup",
            "success":   True,
            "dismissed": dismissed,
            "url":       self._page.url,
            "message":   description if dismissed else "No dismissible popup found on the current page.",
        }
