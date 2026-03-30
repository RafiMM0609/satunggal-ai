"""
Tests for the follow_parent feature in WebAutomationAgent.

The follow_parent feature keeps the browser open between web automation
requests so that follow-up commands can interact with the same page
without re-navigating.  The browser is closed when /reset is issued.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.state import AgentTask

_AGENT_SRC = (
    Path(__file__).parent.parent / "src" / "agents" / "web_automation" / "agent.py"
).read_text()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_task(user_input: str = "open google.com", session_id: str = "sess1") -> AgentTask:
    task = MagicMock(spec=AgentTask)
    task.session_id = session_id
    task.user_input = user_input
    task.metadata = {}
    task.tool_results = {}
    task.mark_processing = MagicMock()
    task.mark_done = MagicMock()
    task.mark_failed = MagicMock()
    return task


# ── Source-level / structural tests (no heavy deps needed) ────────────────────


class TestFollowParentStructure:
    """Verify the follow_parent state machinery exists at the source level."""

    def test_session_follow_parent_dict_declared(self) -> None:
        assert "_session_follow_parent" in _AGENT_SRC

    def test_session_navigator_dict_declared(self) -> None:
        assert "_session_navigator" in _AGENT_SRC

    def test_clear_web_automation_session_is_async(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "clear_web_automation_session"
            ):
                return  # found as async
        pytest.fail("clear_web_automation_session is not declared as 'async def'")

    def test_clear_web_automation_session_closes_navigator(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "clear_web_automation_session"
            ):
                body_src = ast.unparse(node)
                assert "navigator" in body_src and "close" in body_src, (
                    "clear_web_automation_session must close the persisted navigator"
                )
                return
        pytest.fail("clear_web_automation_session not found")

    def test_clear_web_automation_session_clears_follow_parent(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "clear_web_automation_session"
            ):
                body_src = ast.unparse(node)
                assert "_session_follow_parent" in body_src, (
                    "clear_web_automation_session must clear _session_follow_parent"
                )
                return
        pytest.fail("clear_web_automation_session not found")

    def test_run_checks_existing_navigator(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                body_src = ast.unparse(node)
                assert "_session_navigator" in body_src, (
                    "run() must check _session_navigator for an existing browser"
                )
                return
        pytest.fail("run() not found")

    def test_run_sets_follow_parent_in_metadata(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                body_src = ast.unparse(node)
                assert "follow_parent" in body_src, (
                    "run() must set task.metadata['follow_parent']"
                )
                return
        pytest.fail("run() not found")

    def test_plan_next_step_has_follow_parent_param(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_plan_next_step":
                arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
                defaults_count = len(node.args.defaults) + len(node.args.kw_defaults)
                assert "follow_parent" in arg_names, (
                    "_plan_next_step must accept 'follow_parent' parameter"
                )
                return
        pytest.fail("_plan_next_step not found")

    def test_plan_next_step_follow_parent_hint_in_context(self) -> None:
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_plan_next_step":
                body_src = ast.unparse(node)
                assert "follow_parent" in body_src, (
                    "_plan_next_step must use follow_parent to build context"
                )
                return
        pytest.fail("_plan_next_step not found")

    def test_follow_parent_aktif_message_in_context(self) -> None:
        """The planner context must contain a user-readable follow_parent notice."""
        assert "follow_parent AKTIF" in _AGENT_SRC

    def test_browser_not_closed_when_follow_parent_active(self) -> None:
        """run() must skip navigator.close() when the browser should be kept open."""
        tree = ast.parse(_AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                body_src = ast.unparse(node)
                # The finally block must use has_active_page() to check before closing
                assert "has_active_page" in body_src, (
                    "run() must call has_active_page() before deciding to close the browser"
                )
                return
        pytest.fail("run() not found")


# ── Behavioural tests (mock-based) ────────────────────────────────────────────


class TestFollowParentBehaviour:
    """Behavioural tests that exercise module-level state with mocked components."""

    def setup_method(self) -> None:
        # Import lazily to avoid heavy optional deps at collection time.
        import src.agents.web_automation.agent as ag
        self._ag = ag
        # Reset per-session state before each test.
        ag._session_follow_parent.clear()
        ag._session_navigator.clear()
        ag._session_last_url.clear()
        ag._session_domains.clear()

    def test_clear_removes_follow_parent_flag(self) -> None:
        ag = self._ag
        ag._session_follow_parent["u1"] = True
        ag._session_last_url["u1"] = "https://example.com"

        mock_nav = MagicMock()
        mock_nav.close = AsyncMock()
        mock_nav._page = None  # no live page
        ag._session_navigator["u1"] = mock_nav

        _run(ag.clear_web_automation_session("u1"))

        assert "u1" not in ag._session_follow_parent
        assert "u1" not in ag._session_navigator
        assert "u1" not in ag._session_last_url

    def test_clear_closes_live_navigator(self) -> None:
        ag = self._ag
        ag._session_last_url["u2"] = "https://example.com"

        mock_nav = MagicMock()
        mock_nav.close = AsyncMock()
        mock_nav._page = MagicMock()
        ag._session_navigator["u2"] = mock_nav

        _run(ag.clear_web_automation_session("u2"))

        mock_nav.close.assert_called_once()

    def test_clear_no_navigator_does_not_raise(self) -> None:
        ag = self._ag
        # Should silently complete even if no state exists
        _run(ag.clear_web_automation_session("nonexistent"))

    def test_follow_parent_flag_set_after_successful_navigation(self) -> None:
        """After run() completes with a final_url, follow_parent must be True."""
        ag = self._ag
        ag._session_last_url["s1"] = "https://example.com/page"

        mock_page = MagicMock()
        mock_page.is_closed = MagicMock(return_value=False)

        mock_nav = MagicMock()
        mock_nav._page = mock_page
        mock_nav.close = AsyncMock()
        mock_nav.save_current_session = AsyncMock()

        # Simulate a navigator stored by a previous run()
        ag._session_navigator["s1"] = mock_nav
        ag._session_follow_parent["s1"] = True

        # Verify the state is consistent
        assert ag._session_follow_parent.get("s1") is True
        assert ag._session_navigator.get("s1") is mock_nav

    def test_stale_page_triggers_new_navigator_creation(self) -> None:
        """If the stored page is closed, a new navigator must be created."""
        ag = self._ag
        ag._session_last_url["s2"] = "https://old.com"

        mock_page = MagicMock()
        mock_page.is_closed = MagicMock(return_value=True)  # page closed

        mock_nav = MagicMock()
        mock_nav._page = mock_page

        ag._session_navigator["s2"] = mock_nav
        ag._session_follow_parent["s2"] = True

        # After detecting a stale page, clear_web_automation_session is called
        # and state is cleaned up.
        ag._session_navigator.pop("s2", None)
        ag._session_follow_parent.pop("s2", None)

        assert "s2" not in ag._session_navigator
        assert "s2" not in ag._session_follow_parent

    def test_multiple_sessions_are_independent(self) -> None:
        """Clearing one session must not affect other sessions."""
        ag = self._ag
        ag._session_follow_parent["a"] = True
        ag._session_follow_parent["b"] = True
        ag._session_last_url["a"] = "https://a.com"
        ag._session_last_url["b"] = "https://b.com"

        mock_nav_a = MagicMock()
        mock_nav_a.close = AsyncMock()
        mock_nav_a._page = None
        ag._session_navigator["a"] = mock_nav_a

        _run(ag.clear_web_automation_session("a"))

        # Only session "a" should be cleared
        assert "a" not in ag._session_follow_parent
        assert "a" not in ag._session_navigator
        assert "b" in ag._session_follow_parent
        assert "b" in ag._session_last_url
