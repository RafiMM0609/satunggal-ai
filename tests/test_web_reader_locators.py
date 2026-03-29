"""
Tests for WebReaderTool locators field.

Validates:
  - ``_extract_a11y`` returns the expected nodes.
  - ``locators`` filtering correctly keeps only interactive ARIA roles.
  - ``locators`` list is capped at ``_MAX_LOCATORS``.
  - The ``_INTERACTIVE_ROLES`` set in web_reader matches interactive ARIA roles.
  - System prompts in agent.py broaden registration trigger beyond "data random" only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.web_reader import WebReaderTool, _INTERACTIVE_ROLES, _MAX_LOCATORS

_OVERFLOW_COUNT = 20  # extra elements added beyond _MAX_LOCATORS in cap tests


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_page_with_snapshot(snapshot: dict) -> MagicMock:
    """Return a mock Playwright Page with an accessibility snapshot."""
    page = MagicMock()
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=snapshot)
    return page


# ── Tests for _extract_a11y (mirrors browser_navigator but in WebReaderTool) ──


class TestWebReaderExtractA11y:
    """Unit tests for WebReaderTool._extract_a11y static method."""

    def test_returns_role_and_name(self) -> None:
        snapshot = {
            "role": "root", "name": "",
            "children": [
                {"role": "button", "name": "Submit", "children": []},
            ],
        }
        page = _make_page_with_snapshot(snapshot)
        nodes = _run(WebReaderTool._extract_a11y(page))
        assert {"role": "button", "name": "Submit"} in nodes

    def test_textbox_captured(self) -> None:
        snapshot = {
            "role": "root", "name": "",
            "children": [
                {"role": "textbox", "name": "Email", "children": []},
                {"role": "textbox", "name": "Password", "children": []},
            ],
        }
        page = _make_page_with_snapshot(snapshot)
        nodes = _run(WebReaderTool._extract_a11y(page))
        names = [n["name"] for n in nodes]
        assert "Email"    in names
        assert "Password" in names

    def test_nodes_without_name_excluded(self) -> None:
        snapshot = {
            "role": "root", "name": "",
            "children": [
                {"role": "button", "name": "",     "children": []},
                {"role": "link",   "name": "Home", "children": []},
            ],
        }
        page = _make_page_with_snapshot(snapshot)
        nodes = _run(WebReaderTool._extract_a11y(page))
        assert all(n.get("name") for n in nodes)

    def test_empty_snapshot_returns_empty_list(self) -> None:
        page = MagicMock()
        page.accessibility = MagicMock()
        page.accessibility.snapshot = AsyncMock(return_value=None)
        nodes = _run(WebReaderTool._extract_a11y(page))
        assert nodes == []


# ── Tests for locators filtering logic ────────────────────────────────────────


class TestLocatorsFiltering:
    """Validate the locators filtering logic applied to a11y nodes in WebReaderTool.

    These tests exercise the same filter expression used inside _fetch:
        [n for n in a11y_tree if n.get("role","").lower() in _INTERACTIVE_ROLES
         and n.get("name")][:_MAX_LOCATORS]
    """

    def _build_locators(self, a11y_tree: list) -> list:
        """Apply the same locators filter as in WebReaderTool._fetch."""
        return [
            n for n in a11y_tree
            if n.get("role", "").lower() in _INTERACTIVE_ROLES and n.get("name")
        ][:_MAX_LOCATORS]

    def test_interactive_roles_kept(self) -> None:
        tree = [
            {"role": "textbox", "name": "Email"},
            {"role": "button",  "name": "Submit"},
            {"role": "radio",   "name": "Pembeli"},
            {"role": "link",    "name": "Home"},
        ]
        locators = self._build_locators(tree)
        roles = {n["role"] for n in locators}
        assert roles == {"textbox", "button", "radio", "link"}

    def test_non_interactive_roles_excluded(self) -> None:
        tree = [
            {"role": "heading",   "name": "Welcome"},
            {"role": "paragraph", "name": "Some text"},
            {"role": "img",       "name": "Logo"},
            {"role": "button",    "name": "OK"},
        ]
        locators = self._build_locators(tree)
        roles = {n["role"] for n in locators}
        assert roles == {"button"}

    def test_locators_capped_at_max(self) -> None:
        tree = [
            {"role": "button", "name": f"Btn {i}"}
            for i in range(_MAX_LOCATORS + _OVERFLOW_COUNT)
        ]
        locators = self._build_locators(tree)
        assert len(locators) <= _MAX_LOCATORS

    def test_empty_when_no_interactive_elements(self) -> None:
        tree = [
            {"role": "heading", "name": "Title"},
            {"role": "img",     "name": "Logo"},
        ]
        locators = self._build_locators(tree)
        assert locators == []

    def test_registration_form_fields_all_present(self) -> None:
        """A typical registration form page has its fields visible in locators."""
        tree = [
            {"role": "textbox",  "name": "Nama Lengkap"},
            {"role": "textbox",  "name": "Email"},
            {"role": "textbox",  "name": "Password"},
            {"role": "textbox",  "name": "Perusahaan"},
            {"role": "radio",    "name": "Pembeli"},
            {"role": "radio",    "name": "Eksportir"},
            {"role": "button",   "name": "Buat Akun"},
            {"role": "heading",  "name": "Daftar"},    # non-interactive, should be excluded
        ]
        locators = self._build_locators(tree)
        names = {n["name"] for n in locators}
        assert "Nama Lengkap" in names
        assert "Email"        in names
        assert "Password"     in names
        assert "Buat Akun"    in names
        assert "Daftar"       not in names  # heading excluded


# ── Tests for _INTERACTIVE_ROLES constant ────────────────────────────────────


class TestInteractiveRoles:
    """Verify _INTERACTIVE_ROLES contains the expected form-relevant ARIA roles."""

    def test_textbox_is_interactive(self) -> None:
        assert "textbox" in _INTERACTIVE_ROLES

    def test_button_is_interactive(self) -> None:
        assert "button" in _INTERACTIVE_ROLES

    def test_radio_is_interactive(self) -> None:
        assert "radio" in _INTERACTIVE_ROLES

    def test_checkbox_is_interactive(self) -> None:
        assert "checkbox" in _INTERACTIVE_ROLES

    def test_heading_is_not_interactive(self) -> None:
        assert "heading" not in _INTERACTIVE_ROLES

    def test_paragraph_is_not_interactive(self) -> None:
        assert "paragraph" not in _INTERACTIVE_ROLES


# ── Tests for broadened registration prompt guidance ─────────────────────────


class TestBroadenedRegistrationGuidance:
    """Verify the system prompts trigger registration flow for any register request.

    Reads agent.py source to avoid importing modules with heavy optional deps.
    """

    _AGENT_SRC = (
        Path(__file__).parent.parent
        / "src" / "agents" / "web_automation" / "agent.py"
    ).read_text()

    def _extract_string_constant(self, name: str) -> str:
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return node.value.value
        return ""

    def test_react_system_triggers_for_daftar_akun(self) -> None:
        """_REACT_SYSTEM registration guidance must apply to 'daftar akun' (not just 'data random')."""
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert react_src, "_REACT_SYSTEM not found"
        assert "daftar akun" in react_src.lower() or "sign up" in react_src.lower()

    def test_planner_system_triggers_for_daftar_akun(self) -> None:
        """_PLANNER_SYSTEM registration guidance must apply to 'daftar akun' (not just 'data random')."""
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert planner_src, "_PLANNER_SYSTEM not found"
        assert "daftar akun" in planner_src.lower() or "sign up" in planner_src.lower()

    def test_react_system_instructs_to_fill_form_after_locators(self) -> None:
        """_REACT_SYSTEM must instruct the LLM to use 'type' action to fill form fields
        when the previous step's locators show form fields (textbox, radio, etc.)."""
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert react_src, "_REACT_SYSTEM not found"
        # Prompt must reference locators AND type action in the registration guidance section
        assert (
            "locators" in react_src
            and '"type"' in react_src
        ), "Registration guidance must instruct LLM to use 'type' action based on locators"

    def test_planner_system_instructs_to_fill_form_after_locators(self) -> None:
        """_PLANNER_SYSTEM must instruct the LLM to use 'type' action to fill form fields
        when the previous step's locators show form fields (textbox, radio, etc.)."""
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert planner_src, "_PLANNER_SYSTEM not found"
        assert (
            "locators" in planner_src
            and '"type"' in planner_src
        ), "Registration guidance must instruct LLM to use 'type' action based on locators"

    def test_react_system_prohibits_early_done_during_registration(self) -> None:
        """_REACT_SYSTEM must warn the LLM not to output 'done' before the registration
        form has been filled and submitted."""
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert react_src, "_REACT_SYSTEM not found"
        # The prompt explicitly says not to output "done" before the form is submitted
        assert (
            "jangan" in react_src.lower() and "done" in react_src.lower()
            and ("form" in react_src.lower() or "submit" in react_src.lower())
        ), "_REACT_SYSTEM must prohibit outputting 'done' before form is filled and submitted"

    def test_planner_system_prohibits_early_done_during_registration(self) -> None:
        """_PLANNER_SYSTEM must warn the LLM not to output 'done' before the registration
        form has been filled and submitted."""
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert planner_src, "_PLANNER_SYSTEM not found"
        assert (
            "jangan" in planner_src.lower() and "done" in planner_src.lower()
            and ("form" in planner_src.lower() or "submit" in planner_src.lower())
        ), "_PLANNER_SYSTEM must prohibit outputting 'done' before form is filled and submitted"
