"""
Tests for the new web automation capabilities:
  - Identity generator (generate_identity)
  - Captcha detection (check_captcha action, _detect_captcha helper)
  - Popup dismissal (close_popup action, _dismiss_popup helper)
  - Agent pre-generates identity and includes it in planning context
  - New actions present in _REACT_SYSTEM and _PLANNER_SYSTEM
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.browser_navigator import (
    BrowserNavigatorTool,
    _CAPTCHA_IFRAME_PATTERNS,
    _CAPTCHA_TEXT_PHRASES,
    _POPUP_CLOSE_TEXTS,
)
from src.tools.identity_generator import generate_identity
from src.memory.state import AgentTask


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_task(browser_action: str, **meta) -> AgentTask:
    task = MagicMock(spec=AgentTask)
    task.session_id = "test_session"
    task.metadata = {"browser_action": browser_action, **meta}
    return task


def _make_navigator_with_mock_page() -> tuple[BrowserNavigatorTool, MagicMock]:
    nav = BrowserNavigatorTool()
    mock_page = MagicMock()
    nav._page = mock_page
    return nav, mock_page


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Identity Generator Tests ───────────────────────────────────────────────────


class TestIdentityGenerator:
    """Unit tests for generate_identity()."""

    def test_returns_all_required_keys(self) -> None:
        identity = generate_identity()
        required = {"first_name", "last_name", "full_name", "username", "email",
                    "password", "birthdate", "phone"}
        assert required == set(identity.keys())

    def test_seeded_results_are_reproducible(self) -> None:
        id1 = generate_identity(seed=42)
        id2 = generate_identity(seed=42)
        assert id1 == id2

    def test_different_seeds_produce_different_results(self) -> None:
        id1 = generate_identity(seed=1)
        id2 = generate_identity(seed=2)
        assert id1["email"] != id2["email"]

    def test_email_format_is_valid(self) -> None:
        identity = generate_identity(seed=10)
        email = identity["email"]
        assert "@" in email
        domain = email.split("@")[1]
        assert "." in domain

    def test_password_length_at_least_10(self) -> None:
        for seed in range(20):
            identity = generate_identity(seed=seed)
            assert len(identity["password"]) >= 10, (
                f"Password too short for seed={seed}: {identity['password']!r}"
            )

    def test_password_contains_uppercase(self) -> None:
        identity = generate_identity(seed=5)
        assert any(c.isupper() for c in identity["password"])

    def test_password_contains_lowercase(self) -> None:
        identity = generate_identity(seed=5)
        assert any(c.islower() for c in identity["password"])

    def test_password_contains_digit(self) -> None:
        identity = generate_identity(seed=5)
        assert any(c.isdigit() for c in identity["password"])

    def test_password_contains_symbol(self) -> None:
        symbols = set("!@#$%^&*")
        identity = generate_identity(seed=5)
        assert any(c in symbols for c in identity["password"])

    def test_phone_starts_with_08(self) -> None:
        for seed in range(10):
            identity = generate_identity(seed=seed)
            assert identity["phone"].startswith("08"), (
                f"Phone must start with '08', got {identity['phone']!r}"
            )

    def test_phone_is_11_digits(self) -> None:
        for seed in range(10):
            identity = generate_identity(seed=seed)
            assert identity["phone"].isdigit()
            assert len(identity["phone"]) == 11

    def test_birthdate_format_ddmmyyyy(self) -> None:
        identity = generate_identity(seed=7)
        parts = identity["birthdate"].split("/")
        assert len(parts) == 3
        day, month, year = parts
        assert day.isdigit() and 1 <= int(day) <= 28
        assert month.isdigit() and 1 <= int(month) <= 12
        assert year.isdigit() and 1988 <= int(year) <= 2002

    def test_full_name_is_first_and_last(self) -> None:
        identity = generate_identity(seed=3)
        assert identity["full_name"] == f"{identity['first_name']} {identity['last_name']}"

    def test_username_contains_first_and_last_lower(self) -> None:
        identity = generate_identity(seed=3)
        first_lower = identity["first_name"].lower()
        last_lower = identity["last_name"].lower()
        assert first_lower in identity["username"]
        assert last_lower in identity["username"]

    def test_all_values_are_non_empty_strings(self) -> None:
        identity = generate_identity(seed=99)
        for key, val in identity.items():
            assert isinstance(val, str) and val, f"Key {key!r} is empty or not a string"


# ── Captcha Detection Tests ────────────────────────────────────────────────────


class TestCaptchaDetection:
    """Unit tests for _detect_captcha and _action_check_captcha."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()

    def test_no_captcha_when_no_iframe_and_clean_text(self) -> None:
        self.page.evaluate = AsyncMock(return_value="")
        self.nav._extract_page_text = AsyncMock(return_value="Welcome to our site!")
        detected, desc = _run(self.nav._detect_captcha())
        assert detected is False
        assert desc == ""

    def test_recaptcha_iframe_detected(self) -> None:
        src = "https://www.google.com/recaptcha/api2/anchor"
        self.page.evaluate = AsyncMock(return_value=src)
        self.nav._extract_page_text = AsyncMock(return_value="")
        detected, desc = _run(self.nav._detect_captcha())
        assert detected is True
        assert "google.com/recaptcha" in desc

    def test_hcaptcha_iframe_detected(self) -> None:
        src = "https://newassets.hcaptcha.com/captcha/v1/abc/frame"
        self.page.evaluate = AsyncMock(return_value=src)
        self.nav._extract_page_text = AsyncMock(return_value="")
        detected, desc = _run(self.nav._detect_captcha())
        assert detected is True
        assert "hcaptcha" in desc.lower()

    def test_captcha_text_phrase_detected(self) -> None:
        self.page.evaluate = AsyncMock(return_value="")
        self.nav._extract_page_text = AsyncMock(
            return_value="Please verify you are human to continue."
        )
        detected, desc = _run(self.nav._detect_captcha())
        assert detected is True
        assert "verify you are human" in desc

    def test_saya_bukan_robot_detected(self) -> None:
        self.page.evaluate = AsyncMock(return_value="")
        self.nav._extract_page_text = AsyncMock(
            return_value="Centang kotak Saya bukan robot di bawah ini."
        )
        detected, desc = _run(self.nav._detect_captcha())
        assert detected is True

    def test_no_page_returns_false(self) -> None:
        nav = BrowserNavigatorTool()  # _page is None
        detected, desc = _run(nav._detect_captcha())
        assert detected is False
        assert desc == ""

    def test_action_check_captcha_no_captcha(self) -> None:
        self.page.evaluate = AsyncMock(return_value="")
        self.nav._extract_page_text = AsyncMock(return_value="Normal page content")
        self.page.url = "https://example.com"
        task = _make_task("check_captcha")
        result = _run(self.nav._action_check_captcha(task))
        assert result["success"] is True
        assert result["captcha_detected"] is False
        assert result["action"] == "check_captcha"

    def test_action_check_captcha_detected(self) -> None:
        src = "https://www.google.com/recaptcha/api2/bframe"
        self.page.evaluate = AsyncMock(return_value=src)
        self.nav._extract_page_text = AsyncMock(return_value="")
        self.page.url = "https://example.com/register"
        task = _make_task("check_captcha")
        result = _run(self.nav._action_check_captcha(task))
        assert result["success"] is True
        assert result["captcha_detected"] is True
        assert result["captcha_type"] != ""

    def test_action_check_captcha_no_page(self) -> None:
        nav = BrowserNavigatorTool()  # no page
        task = _make_task("check_captcha")
        result = _run(nav._action_check_captcha(task))
        assert result["success"] is False
        assert result["captcha_detected"] is False
        assert "error" in result

    def test_captcha_constants_cover_known_providers(self) -> None:
        """Known CAPTCHA providers must be covered by the detection patterns."""
        providers = [
            "google.com/recaptcha",
            "hcaptcha.com",
            "turnstile.cloudflare.com",
        ]
        for provider in providers:
            assert any(provider in p for p in _CAPTCHA_IFRAME_PATTERNS), (
                f"Provider {provider!r} not in _CAPTCHA_IFRAME_PATTERNS"
            )


# ── Popup Dismissal Tests ──────────────────────────────────────────────────────


class TestPopupDismissal:
    """Unit tests for _dismiss_popup and _action_close_popup."""

    def setup_method(self) -> None:
        self.nav, self.page = _make_navigator_with_mock_page()

    def test_action_close_popup_no_page(self) -> None:
        nav = BrowserNavigatorTool()  # no page
        task = _make_task("close_popup")
        result = _run(nav._action_close_popup(task))
        assert result["success"] is False
        assert result["dismissed"] is False
        assert "error" in result

    def test_action_close_popup_with_dismissible_popup(self) -> None:
        """When _dismiss_popup returns True, action result must show dismissed=True."""
        self.nav._dismiss_popup = AsyncMock(return_value=(True, "Popup closed via button text '×'."))
        self.page.url = "https://example.com"
        task = _make_task("close_popup")
        result = _run(self.nav._action_close_popup(task))
        assert result["success"] is True
        assert result["dismissed"] is True
        assert result["action"] == "close_popup"

    def test_action_close_popup_no_popup_found(self) -> None:
        """When no popup exists, action result must show dismissed=False."""
        self.nav._dismiss_popup = AsyncMock(return_value=(False, ""))
        self.page.url = "https://example.com"
        task = _make_task("close_popup")
        result = _run(self.nav._action_close_popup(task))
        assert result["success"] is True
        assert result["dismissed"] is False

    def test_popup_close_texts_include_common_variants(self) -> None:
        """Common close button labels must be in _POPUP_CLOSE_TEXTS."""
        for label in ("×", "close", "tutup", "dismiss"):
            assert label in _POPUP_CLOSE_TEXTS, (
                f"Close label {label!r} not found in _POPUP_CLOSE_TEXTS"
            )

    def test_dismiss_popup_no_page(self) -> None:
        nav = BrowserNavigatorTool()  # no page
        dismissed, desc = _run(nav._dismiss_popup())
        assert dismissed is False
        assert desc == ""

    def test_dismiss_popup_escape_key_fallback(self) -> None:
        """If CSS/text close fails, Escape key should be tried."""
        # Simulate all locators returning 0 count (no close buttons)
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=0)
        self.page.locator = MagicMock(return_value=mock_locator)

        mock_role_locator = MagicMock()
        mock_role_locator.count = AsyncMock(return_value=0)
        self.page.get_by_role = MagicMock(return_value=mock_role_locator)

        # Escape fires but overlay is still present
        self.page.keyboard = MagicMock()
        self.page.keyboard.press = AsyncMock()
        self.page.evaluate = AsyncMock(return_value=False)  # overlay still present

        dismissed, desc = _run(self.nav._dismiss_popup())
        # Escape was pressed even if overlay didn't disappear
        self.page.keyboard.press.assert_called_once_with("Escape")


# ── Browser Navigator Run Dispatch Tests ──────────────────────────────────────


class TestBrowserNavigatorNewActions:
    """Verify the run() dispatcher handles new action types."""

    def test_run_dispatches_check_captcha(self) -> None:
        nav = BrowserNavigatorTool()
        nav._action_check_captcha = AsyncMock(return_value={
            "action": "check_captcha", "success": True, "captcha_detected": False,
        })
        task = _make_task("check_captcha")
        result = _run(nav.run(task))
        nav._action_check_captcha.assert_called_once_with(task)
        assert result["action"] == "check_captcha"

    def test_run_dispatches_close_popup(self) -> None:
        nav = BrowserNavigatorTool()
        nav._action_close_popup = AsyncMock(return_value={
            "action": "close_popup", "success": True, "dismissed": False,
        })
        task = _make_task("close_popup")
        result = _run(nav.run(task))
        nav._action_close_popup.assert_called_once_with(task)
        assert result["action"] == "close_popup"

    def test_run_unknown_action_returns_error(self) -> None:
        nav = BrowserNavigatorTool()
        task = _make_task("nonexistent_action")
        result = _run(nav.run(task))
        assert result["success"] is False
        assert "error" in result


# ── Agent Identity & System Prompt Tests ──────────────────────────────────────


class TestAgentSystemPrompts:
    """Verify the agent system prompts include new captcha/popup guidance.

    Uses AST-based source inspection to avoid importing agent.py with heavy
    optional dependencies (ollama, httpx) that may not be installed.
    """

    _AGENT_SRC = (
        __import__("pathlib").Path(__file__).parent.parent
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

    # ── Captcha guidance ──────────────────────────────────────────────────────

    def test_react_system_mentions_check_captcha_action(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert "check_captcha" in react_src

    def test_planner_system_mentions_check_captcha_action(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert "check_captcha" in planner_src

    def test_react_system_instructs_captcha_human_help(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert "CAPTCHA" in react_src
        assert "bantuan manusia" in react_src or "human" in react_src.lower()

    def test_planner_system_instructs_captcha_human_help(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert "CAPTCHA" in planner_src
        assert "bantuan manusia" in planner_src or "human" in planner_src.lower()

    # ── Popup guidance ────────────────────────────────────────────────────────

    def test_react_system_mentions_close_popup_action(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert "close_popup" in react_src

    def test_planner_system_mentions_close_popup_action(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert "close_popup" in planner_src

    def test_react_system_re_planning_for_popup(self) -> None:
        """_REACT_SYSTEM must contain re-planning guidance for popup/overlay scenarios."""
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert "re-plan" in react_src.lower() or "re_plan" in react_src.lower() \
            or "RE-PLAN" in react_src or "overlay" in react_src.lower()

    # ── Identity data in context ──────────────────────────────────────────────

    def test_plan_next_step_accepts_identity_parameter(self) -> None:
        """_plan_next_step must have an 'identity' parameter."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_plan_next_step":
                arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
                assert "identity" in arg_names, (
                    "_plan_next_step must accept 'identity' parameter"
                )
                return
        pytest.fail("_plan_next_step not found in agent.py")

    def test_generate_identity_imported_in_agent(self) -> None:
        """agent.py must import generate_identity from the identity_generator module."""
        assert "generate_identity" in self._AGENT_SRC
        assert "identity_generator" in self._AGENT_SRC

    def test_agent_run_calls_generate_identity(self) -> None:
        """The run() method must call generate_identity()."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                body_src = ast.unparse(node)
                assert "generate_identity" in body_src, (
                    "run() must call generate_identity() to pre-generate identity data"
                )
                return
        pytest.fail("run() not found in agent.py")

    def test_identity_included_in_react_context(self) -> None:
        """_plan_next_step source must include identity data in context_parts."""
        import ast
        tree = ast.parse(self._AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_plan_next_step":
                body_src = ast.unparse(node)
                assert "identity" in body_src, (
                    "_plan_next_step must include identity data in context"
                )
                return
        pytest.fail("_plan_next_step not found in agent.py")

    # ── Existing guidance still present ──────────────────────────────────────

    def test_react_system_still_contains_registration_guidance(self) -> None:
        react_src = self._extract_string_constant("_REACT_SYSTEM")
        assert "Panduan pembuatan akun dan registrasi" in react_src

    def test_planner_system_still_contains_registration_guidance(self) -> None:
        planner_src = self._extract_string_constant("_PLANNER_SYSTEM")
        assert "Panduan pembuatan akun dan registrasi" in planner_src
