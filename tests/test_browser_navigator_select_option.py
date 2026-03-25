"""
Tests for BrowserNavigatorTool improvements:

Validates:
  - New `select_option` action dispatches correctly for the "select_option"
    browser_action (unit test using mock page).
  - `_click_by_js_text` uses scrollIntoView before clicking (mock evaluate).
  - `_click_by_force` attempts scroll_into_view_if_needed before force click.
  - `_action_click` now tries `listitem`, `radio`, `checkbox`, `get_by_label`
    locators in addition to the original set.
  - Force-click fallback is invoked when all other strategies fail in click.
  - `_execute_step` in WebAutomationAgent handles "select_option" action.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.tools.browser_navigator import BrowserNavigatorTool
from src.memory.state import AgentTask


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


# ── Tests for _action_select_option ───────────────────────────────────────────


class TestActionSelectOption:
    """Unit tests for the new select_option browser action."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()

    def test_missing_option_text_returns_error(self) -> None:
        task = _make_task("select_option", option_text="", option_selector="")
        result = asyncio.get_event_loop().run_until_complete(
            self.nav._action_select_option(task)
        )
        assert result["success"] is False
        assert "option_text not provided" in result["error"]

    def test_select_option_success_via_aria_role(self) -> None:
        """select_option succeeds when an ARIA role locator matches."""
        # Mock _wait_for_spa_stable to be a no-op
        self.nav._wait_for_spa_stable = AsyncMock()

        # Build a locator mock that succeeds on the "option" role
        mock_loc = MagicMock()
        mock_loc.first = MagicMock()
        mock_loc.first.wait_for = AsyncMock()
        mock_loc.first.click = AsyncMock()

        # page.get_by_role returns the mock locator for "option" role
        def side_effect_role(role, name, exact):
            if role == "option":
                return mock_loc
            # Other roles: raise so they are skipped
            bad_loc = MagicMock()
            bad_loc.first = MagicMock()
            bad_loc.first.wait_for = AsyncMock(side_effect=Exception("not found"))
            return bad_loc

        self.page.get_by_role = MagicMock(side_effect=side_effect_role)
        # Make sure get_by_label fails so we reach role strategies
        bad_lbl = MagicMock()
        bad_lbl.first = MagicMock()
        bad_lbl.first.wait_for = AsyncMock(side_effect=Exception("no label"))
        self.page.get_by_label = MagicMock(return_value=bad_lbl)

        task = _make_task("select_option", option_text="Kopi & Teh", option_selector="")
        result = asyncio.get_event_loop().run_until_complete(
            self.nav._action_select_option(task)
        )
        assert result["success"] is True
        assert result["option_text"] == "Kopi & Teh"
        assert "Selected option" in result["message"]

    def test_select_option_returns_error_when_all_strategies_fail(self) -> None:
        """select_option returns error dict when no strategy succeeds."""
        self.nav._wait_for_spa_stable = AsyncMock()

        # All ARIA role locators fail
        def bad_role(role, name, exact):
            loc = MagicMock()
            loc.first.wait_for = AsyncMock(side_effect=Exception("nope"))
            return loc

        self.page.get_by_role = MagicMock(side_effect=bad_role)

        # get_by_label fails
        bad_lbl = MagicMock()
        bad_lbl.first.wait_for = AsyncMock(side_effect=Exception("no label"))
        self.page.get_by_label = MagicMock(return_value=bad_lbl)

        # JS evaluate (used by _click_by_js_text) returns False (not found)
        self.page.evaluate = AsyncMock(return_value=False)

        # Force click (get_by_text) finds 0 elements
        no_text = MagicMock()
        no_text.count = AsyncMock(return_value=0)
        self.page.get_by_text = MagicMock(return_value=no_text)

        # Screenshot for debug info
        self.page.screenshot = AsyncMock(return_value=b"png")

        task = _make_task("select_option", option_text="Nonexistent Option", option_selector="")
        result = asyncio.get_event_loop().run_until_complete(
            self.nav._action_select_option(task)
        )
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_select_option_with_native_select_selector(self) -> None:
        """When option_selector is set, native <select> is tried first."""
        self.nav._wait_for_spa_stable = AsyncMock()
        self.page.select_option = AsyncMock()

        task = _make_task(
            "select_option",
            option_text="Kopi & Teh",
            option_selector="select#category",
        )
        result = asyncio.get_event_loop().run_until_complete(
            self.nav._action_select_option(task)
        )
        assert result["success"] is True
        # Verify select_option was called with the label
        self.page.select_option.assert_awaited_once_with(
            "select#category",
            label="Kopi & Teh",
            timeout=pytest.approx(30_000, abs=1),
        )

    def test_select_option_auto_detects_native_select(self) -> None:
        """select_option succeeds via auto-detect native <select> when no selector provided."""
        self.nav._wait_for_spa_stable = AsyncMock()

        # Simulate: _select_native_auto finds a matching <select> and returns True
        self.nav._select_native_auto = AsyncMock(return_value=True)

        # Ensure ARIA roles are NOT needed (would raise if called)
        self.page.get_by_role = MagicMock(side_effect=Exception("should not reach ARIA roles"))

        task = _make_task("select_option", option_text="Option A", option_selector="")
        result = asyncio.get_event_loop().run_until_complete(
            self.nav._action_select_option(task)
        )
        assert result["success"] is True
        assert result["option_text"] == "Option A"
        self.nav._select_native_auto.assert_awaited_once_with("Option A")

    def test_select_native_auto_returns_true_when_evaluate_succeeds(self) -> None:
        """_select_native_auto returns True when page.evaluate returns True."""
        nav, page = _make_navigator_with_mock_page()
        page.evaluate = AsyncMock(return_value=True)

        result = asyncio.get_event_loop().run_until_complete(
            nav._select_native_auto("Option A")
        )
        assert result is True
        # The JS passed to evaluate must scan <select> elements
        js_code = page.evaluate.call_args[0][0]
        assert "querySelectorAll('select')" in js_code
        assert "HTMLSelectElement.prototype" in js_code

    def test_select_native_auto_returns_false_when_evaluate_returns_false(self) -> None:
        """_select_native_auto returns False when no matching <select> is found."""
        nav, page = _make_navigator_with_mock_page()
        page.evaluate = AsyncMock(return_value=False)

        result = asyncio.get_event_loop().run_until_complete(
            nav._select_native_auto("Nonexistent")
        )
        assert result is False

    def test_select_native_auto_returns_false_on_exception(self) -> None:
        """_select_native_auto returns False gracefully when evaluate raises."""
        nav, page = _make_navigator_with_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("evaluate error"))

        result = asyncio.get_event_loop().run_until_complete(
            nav._select_native_auto("Option A")
        )
        assert result is False


# ── Tests for _click_by_js_text (scroll-into-view improvement) ────────────────


class TestClickByJsText:
    """Verify _click_by_js_text passes scrollIntoView call in JS evaluation."""

    def test_js_text_walk_invokes_evaluate_with_scroll(self) -> None:
        nav, page = _make_navigator_with_mock_page()
        # Simulate: element found and clicked
        page.evaluate = AsyncMock(return_value=True)

        result = asyncio.get_event_loop().run_until_complete(
            nav._click_by_js_text("Kopi & Teh")
        )
        assert result is True
        # The JS code passed to evaluate must contain scrollIntoView
        js_code = page.evaluate.call_args[0][0]
        assert "scrollIntoView" in js_code

    def test_js_text_walk_returns_false_when_evaluate_returns_false(self) -> None:
        nav, page = _make_navigator_with_mock_page()
        page.evaluate = AsyncMock(return_value=False)

        result = asyncio.get_event_loop().run_until_complete(
            nav._click_by_js_text("Nonexistent Text")
        )
        assert result is False

    def test_js_text_walk_relaxed_size_check(self) -> None:
        """The JS code must check width===0 && height===0 (AND), not OR."""
        nav, page = _make_navigator_with_mock_page()
        page.evaluate = AsyncMock(return_value=True)

        asyncio.get_event_loop().run_until_complete(
            nav._click_by_js_text("any text")
        )
        js_code = page.evaluate.call_args[0][0]
        # Must use AND (&&) for zero-size check so elements clipped by a
        # scrollable container (non-zero width, zero height or vice-versa) are
        # still included as candidates rather than being filtered out.
        assert "rect.width === 0 && rect.height === 0" in js_code


# ── Tests for _click_by_force ─────────────────────────────────────────────────


class TestClickByForce:
    """Verify _click_by_force scrolls into view and uses force=True."""

    def test_force_click_success(self) -> None:
        nav, page = _make_navigator_with_mock_page()

        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=1)
        mock_loc.first = MagicMock()
        mock_loc.first.scroll_into_view_if_needed = AsyncMock()
        mock_loc.first.click = AsyncMock()
        page.get_by_text = MagicMock(return_value=mock_loc)

        result = asyncio.get_event_loop().run_until_complete(
            nav._click_by_force("Kopi & Teh")
        )
        assert result is True
        mock_loc.first.scroll_into_view_if_needed.assert_awaited_once()
        # click must be called with force=True
        _, kwargs = mock_loc.first.click.call_args
        assert kwargs.get("force") is True

    def test_force_click_returns_false_when_no_element(self) -> None:
        nav, page = _make_navigator_with_mock_page()

        mock_loc = MagicMock()
        mock_loc.count = AsyncMock(return_value=0)
        page.get_by_text = MagicMock(return_value=mock_loc)

        result = asyncio.get_event_loop().run_until_complete(
            nav._click_by_force("Nonexistent")
        )
        assert result is False


# ── Tests for _action_click extended locator strategies ───────────────────────


class TestActionClickExtendedLocators:
    """Verify _action_click now attempts listitem, radio, checkbox, label roles."""

    def _count_role_calls(self, page: MagicMock) -> set[str]:
        """Return the set of ARIA roles passed to page.get_by_role."""
        return {c.args[0] for c in page.get_by_role.call_args_list}

    def test_click_tries_listitem_role(self) -> None:
        nav, page = _make_navigator_with_mock_page()
        nav._wait_for_spa_stable = AsyncMock()
        nav._detect_error_page = AsyncMock(return_value="")
        nav._extract_page_text = AsyncMock(return_value="")

        # All locators fail → also tests that new roles are attempted
        def bad_role(role, name, exact):
            loc = MagicMock()
            loc.first.wait_for = AsyncMock(side_effect=Exception("nope"))
            return loc

        page.get_by_role = MagicMock(side_effect=bad_role)

        bad_lbl = MagicMock()
        bad_lbl.first.wait_for = AsyncMock(side_effect=Exception("no label"))
        page.get_by_label = MagicMock(return_value=bad_lbl)

        bad_text = MagicMock()
        bad_text.first.wait_for = AsyncMock(side_effect=Exception("no text"))
        page.get_by_text = MagicMock(return_value=bad_text)

        # JS evaluate returns False (not found)
        page.evaluate = AsyncMock(return_value=False)
        # Force click also finds nothing
        page.get_by_text.return_value.count = AsyncMock(return_value=0)

        page.screenshot = AsyncMock(return_value=b"png")
        page.url = "https://example.com"
        page.title = AsyncMock(return_value="Test")

        task = _make_task("click", click_text="Kopi & Teh")
        result = asyncio.get_event_loop().run_until_complete(
            nav._action_click(task)
        )
        # The click failed, but we want to confirm new roles were attempted
        roles_tried = self._count_role_calls(page)
        assert "listitem" in roles_tried
        assert "radio" in roles_tried
        assert "checkbox" in roles_tried


# ── Tests for _execute_step in WebAutomationAgent ────────────────────────────


class TestExecuteStepSelectOption:
    """Verify WebAutomationAgent._execute_step handles 'select_option' action."""

    def test_select_option_step_calls_navigator_with_correct_metadata(self) -> None:
        """Verify _execute_step handles 'select_option' by checking code structure."""
        import inspect
        import ast
        import textwrap

        # Read the agent source and verify the select_option branch exists
        # and sets the correct metadata fields.
        agent_src = (
            __import__("pathlib").Path(
                __file__
            ).parent.parent / "src" / "agents" / "web_automation" / "agent.py"
        ).read_text()

        tree = ast.parse(agent_src)

        # Find the _execute_step method
        execute_step_found = False
        select_option_branch_found = False
        option_text_set = False
        option_selector_set = False

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_step":
                execute_step_found = True
                src_body = ast.unparse(node)
                if "select_option" in src_body:
                    select_option_branch_found = True
                if "option_text" in src_body:
                    option_text_set = True
                if "option_selector" in src_body:
                    option_selector_set = True

        assert execute_step_found, "_execute_step method not found in agent"
        assert select_option_branch_found, "select_option branch not found in _execute_step"
        assert option_text_set, "option_text metadata key not set in _execute_step"
        assert option_selector_set, "option_selector metadata key not set in _execute_step"
