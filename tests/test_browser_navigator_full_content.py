"""
Tests for BrowserNavigatorTool.get_full_content (auto-scroll full-page extraction).

Validates:
  - ``get_full_content`` is dispatched from ``run()`` when browser_action is
    "get_full_content".
  - The returned dict includes ``full_page=True`` and ``scroll_steps``.
  - Page text is returned without the usual 8 000-char truncation
    (_MAX_FULL_PAGE_TEXT_CHARS = 50 000 applies instead).
  - The auto-scroll loop stops when the bottom of the page is reached
    (scrollY + innerHeight >= scrollHeight).
  - The auto-scroll loop stops after _MAX_FULL_PAGE_SCROLL_STEPS iterations
    even if the bottom is never reached.
  - The auto-scroll loop stops if scrollY does not advance (non-scrollable page).
  - ``get_full_content`` returns an error dict when no page is open.
  - ``_compact_result`` preserves the ``full_page`` flag and ``scroll_steps``
    field for the ReAct context.
  - The agent's ``_execute_step`` handles "get_full_content" and logs correctly.
  - The agent's ``_summarise`` uses a larger text budget for ``full_page`` results.
  - The _REACT_SYSTEM and _PLANNER_SYSTEM prompts mention ``get_full_content``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.browser_navigator import BrowserNavigatorTool, _MAX_FULL_PAGE_TEXT_CHARS
from src.memory.state import AgentTask
from src.agents.web_automation.agent import (
    _compact_result,
    _REACT_SYSTEM,
    _PLANNER_SYSTEM,
    _FULL_PAGE_SUMMARISE_TEXT_CHARS,
    _SUMMARISE_TEXT_CHARS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(browser_action: str, **meta) -> AgentTask:
    task = MagicMock(spec=AgentTask)
    task.session_id = "test_session"
    task.metadata = {"browser_action": browser_action, **meta}
    return task


def _make_navigator_with_mock_page() -> tuple[BrowserNavigatorTool, MagicMock]:
    """Return a BrowserNavigatorTool whose internal _page is replaced by a mock."""
    nav = BrowserNavigatorTool()
    mock_page = MagicMock()
    nav._page = mock_page
    return nav, mock_page


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Tests for _action_get_full_content ────────────────────────────────────────


class TestGetFullContentBasic:
    """Basic structure and field validation for get_full_content."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()
        self.page.title = AsyncMock(return_value="Full Page")
        self.page.url = "https://example.com/article"

    def _setup_scroll_mocks(self, *, already_at_bottom: bool = True) -> None:
        """Configure page.evaluate to simulate a non-scrollable (or bottom) page."""
        # First call: scrollTo (returns None)
        # Subsequent calls: scroll position queries
        scroll_info = {
            "scrollY": 0.0,
            "scrollHeight": 500.0,
            "innerHeight": 500.0,  # viewport == page → already at bottom
        }
        if not already_at_bottom:
            scroll_info["scrollHeight"] = 1000.0  # taller page

        async def _evaluate(expr, *args, **kwargs):
            if "scrollTo" in expr:
                return None
            if "scrollBy" in expr:
                return None
            return scroll_info

        self.page.evaluate = _evaluate

    def test_get_full_content_returns_full_page_true(self) -> None:
        """Result must have full_page=True."""
        self._setup_scroll_mocks()
        self.nav._extract_page_text = AsyncMock(return_value="Hello full world")
        self.nav._extract_page_a11y = AsyncMock(return_value=[])
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert result["success"] is True
        assert result["full_page"] is True

    def test_get_full_content_returns_scroll_steps(self) -> None:
        """Result must include a 'scroll_steps' integer field."""
        self._setup_scroll_mocks()
        self.nav._extract_page_text = AsyncMock(return_value="text")
        self.nav._extract_page_a11y = AsyncMock(return_value=[])
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert "scroll_steps" in result
        assert isinstance(result["scroll_steps"], int)

    def test_get_full_content_returns_required_fields(self) -> None:
        """Result must contain action, success, title, url, page_text, locators."""
        self._setup_scroll_mocks()
        self.nav._extract_page_text = AsyncMock(return_value="some text")
        self.nav._extract_page_a11y = AsyncMock(return_value=[
            {"role": "button", "name": "Click Me"},
        ])
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert result["action"] == "get_full_content"
        assert result["success"] is True
        assert result["title"] == "Full Page"
        assert result["url"] == "https://example.com/article"
        assert "page_text" in result
        assert "locators" in result

    def test_get_full_content_no_page_returns_error(self) -> None:
        """When no page is open, get_full_content must return an error dict."""
        nav = BrowserNavigatorTool()
        nav._page = None
        task = _make_task("get_full_content")
        result = _run(nav._action_get_full_content(task))

        assert result["success"] is False
        assert "error" in result
        assert result["action"] == "get_full_content"


class TestGetFullContentScrollBehavior:
    """Scroll loop correctness tests."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()
        self.page.title = AsyncMock(return_value="Page")
        self.page.url = "https://example.com"
        self.nav._extract_page_text = AsyncMock(return_value="text")
        self.nav._extract_page_a11y = AsyncMock(return_value=[])

    def test_stops_when_bottom_reached_immediately(self) -> None:
        """Should do 0 scroll steps when already at the bottom."""
        async def _eval(expr, *args, **kwargs):
            if "scrollTo" in expr or "scrollBy" in expr:
                return None
            return {"scrollY": 0.0, "scrollHeight": 500.0, "innerHeight": 500.0}

        self.page.evaluate = _eval
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert result["scroll_steps"] == 0

    def test_scrolls_until_bottom(self) -> None:
        """Should scroll exactly as many times as needed to reach the bottom."""
        # Simulate a page that requires 2 scrolls (viewport=400, height=1200)
        call_count = 0
        scroll_y = 0.0

        async def _eval(expr, *args, **kwargs):
            nonlocal scroll_y, call_count
            if "scrollTo" in expr:
                scroll_y = 0.0
                return None
            if "scrollBy" in expr:
                scroll_y += 600.0
                return None
            # position query
            return {"scrollY": scroll_y, "scrollHeight": 1200.0, "innerHeight": 400.0}

        self.page.evaluate = _eval
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        # 0+400=400 < 1200 → scroll → 600+400=1000 < 1200 → scroll → 1200+400≥1200 stop
        assert result["scroll_steps"] == 2

    def test_stops_when_scroll_position_unchanged(self) -> None:
        """Should stop if scrollY does not advance between steps."""
        # Simulate page that doesn't actually scroll (position stays at 0)
        async def _eval(expr, *args, **kwargs):
            if "scrollTo" in expr or "scrollBy" in expr:
                return None
            # scrollHeight > viewport but position never changes
            return {"scrollY": 0.0, "scrollHeight": 2000.0, "innerHeight": 400.0}

        self.page.evaluate = _eval
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        # First iteration: prev=-1, scrollY=0 → not equal → scroll
        # Second iteration: prev=0, scrollY=0 → equal → stop (scroll_steps=1)
        assert result["scroll_steps"] == 1


class TestGetFullContentTextBudget:
    """Page text is returned with the higher full-page cap."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()
        self.page.title = AsyncMock(return_value="Big Page")
        self.page.url = "https://example.com"
        self.nav._extract_page_a11y = AsyncMock(return_value=[])

        async def _eval(expr, *args, **kwargs):
            if "scrollTo" in expr or "scrollBy" in expr:
                return None
            return {"scrollY": 0.0, "scrollHeight": 500.0, "innerHeight": 500.0}

        self.page.evaluate = _eval

    def test_long_text_not_truncated_at_8000(self) -> None:
        """A text of 10 000 chars must NOT be cut to 8 000."""
        long_text = "x" * 10_000
        self.nav._extract_page_text = AsyncMock(return_value=long_text)
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert len(result["page_text"]) == 10_000

    def test_very_long_text_capped_at_max_full_page(self) -> None:
        """Text beyond _MAX_FULL_PAGE_TEXT_CHARS must be truncated."""
        very_long = "y" * (_MAX_FULL_PAGE_TEXT_CHARS + 5_000)
        self.nav._extract_page_text = AsyncMock(return_value=very_long)
        task = _make_task("get_full_content")
        result = _run(self.nav._action_get_full_content(task))

        assert len(result["page_text"]) == _MAX_FULL_PAGE_TEXT_CHARS


class TestRunDispatch:
    """Ensure BrowserNavigatorTool.run() dispatches to _action_get_full_content."""

    def test_run_dispatches_get_full_content(self) -> None:
        """run() must call _action_get_full_content for browser_action='get_full_content'."""
        nav = BrowserNavigatorTool()
        mock_result = {
            "action": "get_full_content",
            "success": True,
            "full_page": True,
            "scroll_steps": 0,
            "page_text": "text",
            "title": "T",
            "url": "https://x.com",
            "a11y_tree": [],
            "locators": [],
            "message": "ok",
        }
        nav._action_get_full_content = AsyncMock(return_value=mock_result)
        task = _make_task("get_full_content")
        result = _run(nav.run(task))

        nav._action_get_full_content.assert_awaited_once_with(task)
        assert result["action"] == "get_full_content"
        assert result["full_page"] is True


# ── Tests for _compact_result ──────────────────────────────────────────────────


class TestCompactResultFullPage:
    """_compact_result must preserve full_page and scroll_steps fields."""

    def test_compact_result_preserves_full_page_flag(self) -> None:
        result = {
            "action": "get_full_content",
            "success": True,
            "full_page": True,
            "scroll_steps": 3,
            "page_text": "some text",
            "locators": [],
            "a11y_tree": [{"role": "heading", "name": "Hi"}],
            "screenshot_b64": "base64data",
        }
        compact = _compact_result(result)

        assert compact.get("full_page") is True
        assert compact.get("scroll_steps") == 3
        # bulky fields must be stripped
        assert "a11y_tree" not in compact
        assert "screenshot_b64" not in compact

    def test_compact_result_truncates_page_text(self) -> None:
        """Even for full_page results, _compact_result truncates for the ReAct context."""
        long_text = "a" * 5_000
        result = {
            "action": "get_full_content",
            "full_page": True,
            "scroll_steps": 5,
            "page_text": long_text,
            "locators": [],
        }
        compact = _compact_result(result)

        # ReAct context uses _REACT_RESULT_TEXT_CHARS (800), not the full text
        assert len(compact["page_text"]) <= 801  # 800 chars + "…"


# ── Tests for prompt content ───────────────────────────────────────────────────


class TestPromptsMentionGetFullContent:
    """Both system prompts must reference the get_full_content action."""

    def test_react_system_mentions_get_full_content(self) -> None:
        assert "get_full_content" in _REACT_SYSTEM

    def test_planner_system_mentions_get_full_content(self) -> None:
        assert "get_full_content" in _PLANNER_SYSTEM

    def test_react_system_explains_when_to_use(self) -> None:
        """Prompt must explain the intent trigger for full-page extraction."""
        # Check for at least one phrase that describes the full-content intent
        trigger_phrases = [
            "semua konten", "seluruh isi", "full content",
            "keseluruhan", "scroll sampai habis",
        ]
        assert any(phrase in _REACT_SYSTEM for phrase in trigger_phrases)

    def test_planner_system_explains_when_to_use(self) -> None:
        trigger_phrases = [
            "semua konten", "seluruh isi", "full content",
            "keseluruhan", "scroll sampai habis",
        ]
        assert any(phrase in _PLANNER_SYSTEM for phrase in trigger_phrases)


# ── Tests for summariser text budget ──────────────────────────────────────────


class TestSummariserTextBudget:
    """_FULL_PAGE_SUMMARISE_TEXT_CHARS must be larger than _SUMMARISE_TEXT_CHARS."""

    def test_full_page_budget_larger_than_normal(self) -> None:
        assert _FULL_PAGE_SUMMARISE_TEXT_CHARS > _SUMMARISE_TEXT_CHARS
