"""
Tests for BrowserNavigatorTool locator efficiency improvements.

Validates:
  - ``_action_get_content`` includes a ``locators`` field with interactive elements
    extracted from the accessibility tree.
  - ``locators`` contains only ARIA-interactive roles (button, link, textbox, etc.)
    and excludes non-interactive roles (heading, paragraph, etc.).
  - ``locators`` entries include optional state fields (checked, value, disabled,
    description) when the underlying a11y node provides them.
  - ``_extract_page_a11y`` enriches nodes with state fields from the a11y snapshot.
  - ``_compact_result`` in the agent preserves ``locators`` up to _REACT_LOCATORS_LIMIT
    while still stripping ``a11y_tree`` and ``screenshot_b64``.
  - Planner and ReAct system prompts mention locator-efficient strategy keywords.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

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


# ── Tests for _action_get_content locators field ──────────────────────────────


class TestGetContentLocators:
    """Unit tests verifying ``get_content`` returns a ``locators`` field."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()
        self.page.title = AsyncMock(return_value="Test Page")
        self.page.url = "https://example.com"

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_get_content_includes_locators_field(self) -> None:
        """Result must have a 'locators' key."""
        self.nav._extract_page_text = AsyncMock(return_value="Hello world")
        # Simulate a11y tree with interactive + non-interactive roles
        self.nav._extract_page_a11y = AsyncMock(return_value=[
            {"role": "button", "name": "Submit"},
            {"role": "link",   "name": "Home"},
            {"role": "heading", "name": "Welcome"},   # non-interactive
            {"role": "textbox", "name": "Email"},
        ])
        task = _make_task("get_content")
        result = self._run(self.nav._action_get_content(task))

        assert result["success"] is True
        assert "locators" in result

    def test_locators_contains_only_interactive_roles(self) -> None:
        """locators must exclude non-interactive roles (heading, paragraph, etc.)."""
        self.nav._extract_page_text = AsyncMock(return_value="")
        self.nav._extract_page_a11y = AsyncMock(return_value=[
            {"role": "button",  "name": "OK"},
            {"role": "heading", "name": "Welcome"},
            {"role": "textbox", "name": "Search"},
            {"role": "link",    "name": "About"},
            {"role": "generic", "name": "Container"},
        ])
        task = _make_task("get_content")
        result = self._run(self.nav._action_get_content(task))

        roles_in_locators = {n["role"] for n in result["locators"]}
        assert "button"  in roles_in_locators
        assert "textbox" in roles_in_locators
        assert "link"    in roles_in_locators
        assert "heading" not in roles_in_locators
        assert "generic" not in roles_in_locators

    def test_locators_limited_to_max_locators(self) -> None:
        """locators list must not exceed _MAX_LOCATORS (60) entries."""
        from src.tools.browser_navigator import _MAX_LOCATORS

        self.nav._extract_page_text = AsyncMock(return_value="")
        # Create 100 button nodes – well above the cap
        many_buttons = [{"role": "button", "name": f"Btn {i}"} for i in range(100)]
        self.nav._extract_page_a11y = AsyncMock(return_value=many_buttons)

        task = _make_task("get_content")
        result = self._run(self.nav._action_get_content(task))

        assert len(result["locators"]) <= _MAX_LOCATORS

    def test_locators_empty_when_no_interactive_elements(self) -> None:
        """locators is an empty list when the page has no interactive a11y nodes."""
        self.nav._extract_page_text = AsyncMock(return_value="")
        self.nav._extract_page_a11y = AsyncMock(return_value=[
            {"role": "heading", "name": "Title"},
            {"role": "img",     "name": "Logo"},
        ])
        task = _make_task("get_content")
        result = self._run(self.nav._action_get_content(task))

        assert result["locators"] == []

    def test_get_content_no_page_returns_error(self) -> None:
        """get_content without an open page returns an error dict."""
        nav = BrowserNavigatorTool()  # fresh, _page is None
        task = _make_task("get_content")
        result = self._run(nav._action_get_content(task))

        assert result["success"] is False
        assert "error" in result


# ── Tests for _extract_page_a11y state enrichment ────────────────────────────


class TestExtractPageA11yStateEnrichment:
    """Verify _extract_page_a11y enriches nodes with optional state fields."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_nav_with_snapshot(self, snapshot: dict) -> BrowserNavigatorTool:
        nav = BrowserNavigatorTool()
        mock_page = MagicMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=snapshot)
        nav._page = mock_page
        return nav

    def test_basic_role_and_name_captured(self) -> None:
        snapshot = {"role": "button", "name": "OK", "children": []}
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        assert {"role": "button", "name": "OK"} in nodes

    def test_checked_state_included(self) -> None:
        snapshot = {
            "role": "root",
            "name": "",
            "children": [
                {"role": "checkbox", "name": "Remember me", "checked": True, "children": []},
            ],
        }
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        cb_nodes = [n for n in nodes if n["role"] == "checkbox"]
        assert len(cb_nodes) == 1
        assert cb_nodes[0].get("checked") is True

    def test_value_field_included_and_truncated(self) -> None:
        long_value = "x" * 200
        snapshot = {
            "role": "root",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Email", "value": long_value, "children": []},
            ],
        }
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        tb_nodes = [n for n in nodes if n["role"] == "textbox"]
        assert len(tb_nodes) == 1
        assert len(tb_nodes[0].get("value", "")) <= 80

    def test_disabled_flag_included(self) -> None:
        snapshot = {
            "role": "root",
            "name": "",
            "children": [
                {"role": "button", "name": "Disabled Btn", "disabled": True, "children": []},
            ],
        }
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        btn_nodes = [n for n in nodes if n["role"] == "button"]
        assert btn_nodes[0].get("disabled") is True

    def test_description_field_included_and_truncated(self) -> None:
        long_desc = "d" * 200
        snapshot = {
            "role": "root",
            "name": "",
            "children": [
                {
                    "role": "button",
                    "name": "Info",
                    "description": long_desc,
                    "children": [],
                },
            ],
        }
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        btn_nodes = [n for n in nodes if n["role"] == "button"]
        assert "description" in btn_nodes[0]
        assert len(btn_nodes[0]["description"]) <= 80

    def test_nodes_without_name_excluded(self) -> None:
        """Nodes with empty name must be excluded from the result."""
        snapshot = {
            "role": "root",
            "name": "",
            "children": [
                {"role": "button", "name": "", "children": []},
                {"role": "link",   "name": "Home", "children": []},
            ],
        }
        nav = self._make_nav_with_snapshot(snapshot)
        nodes = self._run(nav._extract_page_a11y())
        assert all(n.get("name") for n in nodes)

    def test_empty_snapshot_returns_empty_list(self) -> None:
        nav = BrowserNavigatorTool()
        mock_page = MagicMock()
        mock_page.accessibility = MagicMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=None)
        nav._page = mock_page
        nodes = self._run(nav._extract_page_a11y())
        assert nodes == []


# ── Tests for _compact_result in agent ────────────────────────────────────────


class TestCompactResultLocators:
    """Verify _compact_result preserves 'locators' and strips bulky fields.

    Uses AST-based source inspection to avoid importing agent.py directly
    (agent.py has heavy optional dependencies like ollama that may not be
    installed in all test environments).
    """

    _AGENT_SRC = (
        __import__("pathlib").Path(__file__).parent.parent
        / "src" / "agents" / "web_automation" / "agent.py"
    ).read_text()

    def test_locators_not_in_skip_keys(self) -> None:
        """'locators' must NOT appear in _SKIP_KEYS so it reaches _compact_result."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        # Find _SKIP_KEYS assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_SKIP_KEYS":
                        src_repr = ast.unparse(node)
                        assert "locators" not in src_repr, (
                            "'locators' must not be in _SKIP_KEYS – it should be preserved"
                        )

    def test_compact_result_handles_locators(self) -> None:
        """_compact_result source must contain 'locators' handling logic."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_compact_result":
                body_src = ast.unparse(node)
                assert "locators" in body_src, (
                    "_compact_result must handle the 'locators' field"
                )
                return
        pytest.fail("_compact_result function not found in agent.py")

    def test_a11y_tree_in_skip_keys(self) -> None:
        """'a11y_tree' must remain in _SKIP_KEYS so full trees are stripped."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_SKIP_KEYS":
                        src_repr = ast.unparse(node)
                        assert "a11y_tree" in src_repr, (
                            "'a11y_tree' must stay in _SKIP_KEYS to strip full trees"
                        )

    def test_screenshot_b64_in_skip_keys(self) -> None:
        """'screenshot_b64' must remain in _SKIP_KEYS."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_SKIP_KEYS":
                        src_repr = ast.unparse(node)
                        assert "screenshot_b64" in src_repr, (
                            "'screenshot_b64' must stay in _SKIP_KEYS"
                        )

    def test_react_locators_limit_constant_defined(self) -> None:
        """_REACT_LOCATORS_LIMIT constant must be defined in agent.py."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "_REACT_LOCATORS_LIMIT" in names, (
            "_REACT_LOCATORS_LIMIT constant must be defined in agent.py"
        )


# ── Tests for system prompt keywords ──────────────────────────────────────────


class TestSystemPromptKeywords:
    """Smoke-test that system prompts contain efficient-locator guidance keywords.

    Reads agent.py source directly to avoid importing modules with optional
    heavy dependencies (ollama, httpx, etc.) that may not be available in all
    test environments.
    """

    _AGENT_SRC = (
        __import__("pathlib").Path(__file__).parent.parent
        / "src" / "agents" / "web_automation" / "agent.py"
    ).read_text()

    def _extract_string_constant(self, name: str) -> str:
        """Extract the content of a module-level string constant from agent.py."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return node.value.value
        # Not found as a simple constant; fall back to raw source search
        return ""

    def test_react_system_mentions_locators_strategy(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert react_src, "_REACT_SYSTEM not found as a string constant in agent.py"
        assert "locators" in react_src

    def test_planner_system_mentions_locators_strategy(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert planner_src, "_PLANNER_SYSTEM not found as a string constant in agent.py"
        assert "locators" in planner_src

    def test_react_system_discourages_unnecessary_get_content(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert react_src, "_REACT_SYSTEM not found as a string constant in agent.py"
        # Should contain guidance about NOT calling get_content unnecessarily
        assert "TIDAK PERLU" in react_src or "auto-wait" in react_src.lower()

    def test_planner_system_discourages_unnecessary_get_content(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert planner_src, "_PLANNER_SYSTEM not found as a string constant in agent.py"
        assert "TIDAK PERLU" in planner_src or "auto-wait" in planner_src.lower()
