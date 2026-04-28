"""
Tests for Phase 3 collaboration:
  - DeveloperInspectorAgent.inspect_diff()
  - DeveloperAgent.inspector injection and post-sandbox review
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.developer_inspector.agent import (
    INSPECTOR_TEMPERATURE,
    INSPECTOR_TOP_P,
    DeveloperInspectorAgent,
)
from src.memory.state import AgentTask


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _make_inspector() -> DeveloperInspectorAgent:
    """Return a DeveloperInspectorAgent with a mocked LLM and dummy settings."""
    with patch("config.settings.get_settings") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            sandbox_repos_dir="/tmp/repos",
            github_pat="",
            gitlab_pat="",
            gitlab_hosts="",
        )
        agent = DeveloperInspectorAgent(llm=MagicMock())
    return agent


_SAMPLE_DIFF = """\
diff --git a/app.py b/app.py
index abc123..def456 100644
--- a/app.py
+++ b/app.py
@@ -10,6 +10,7 @@ def hello():
     return "hello"
 
+def goodbye():
+    return "goodbye"
"""


# ── DeveloperInspectorAgent.inspect_diff() tests ─────────────────────────────

class TestInspectDiff:

    def test_method_exists(self):
        """inspect_diff is defined on DeveloperInspectorAgent."""
        assert callable(getattr(DeveloperInspectorAgent, "inspect_diff", None))

    @pytest.mark.asyncio
    async def test_returns_review_on_success(self):
        """inspect_diff returns LLM output when diff is non-empty."""
        agent = _make_inspector()
        expected_review = "## 🔍 Code Review\n\n✅ Looks good!"
        agent._llm.chat = AsyncMock(return_value=expected_review)

        result = await agent.inspect_diff(_SAMPLE_DIFF, "Add goodbye function", "sess1")

        assert result == expected_review
        agent._llm.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_diff(self):
        """inspect_diff short-circuits and returns '' for empty diff."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock()

        result = await agent.inspect_diff("", "some task", "sess2")

        assert result == ""
        agent._llm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_empty_on_whitespace_diff(self):
        """inspect_diff short-circuits for whitespace-only diff."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock()

        result = await agent.inspect_diff("   \n\t  ", "task", "sess3")

        assert result == ""
        agent._llm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_correct_temperature(self):
        """inspect_diff passes INSPECTOR_TEMPERATURE and INSPECTOR_TOP_P to LLM."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock(return_value="OK")

        await agent.inspect_diff(_SAMPLE_DIFF, "Fix bug", "sess4")

        call_kwargs = agent._llm.chat.await_args.kwargs
        assert call_kwargs.get("temperature") == INSPECTOR_TEMPERATURE
        assert call_kwargs.get("top_p") == INSPECTOR_TOP_P

    @pytest.mark.asyncio
    async def test_diff_is_capped_at_max_chars(self):
        """inspect_diff truncates diffs longer than _MAX_DIFF_REVIEW_CHARS."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock(return_value="review")
        max_chars = agent._MAX_DIFF_REVIEW_CHARS

        long_diff = "+" + "x" * (max_chars + 1000)
        await agent.inspect_diff(long_diff, "task", "sess5")

        call_args = agent._llm.chat.await_args
        user_content = call_args.kwargs["messages"][-1]["content"]
        assert "dipotong" in user_content  # truncation notice appended

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_llm_failure(self):
        """inspect_diff is non-fatal: returns '' when LLM raises."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await agent.inspect_diff(_SAMPLE_DIFF, "task", "sess6")

        assert result == ""

    @pytest.mark.asyncio
    async def test_includes_task_description_in_prompt(self):
        """The task description is injected into the LLM user message."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock(return_value="OK")

        task_desc = "Implement feature XYZ in module ABC"
        await agent.inspect_diff(_SAMPLE_DIFF, task_desc, "sess7")

        call_args = agent._llm.chat.await_args
        user_content = call_args.kwargs["messages"][-1]["content"]
        assert task_desc in user_content

    @pytest.mark.asyncio
    async def test_strips_review_whitespace(self):
        """inspect_diff strips leading/trailing whitespace from the LLM output."""
        agent = _make_inspector()
        agent._llm.chat = AsyncMock(return_value="  \nreview text\n  ")

        result = await agent.inspect_diff(_SAMPLE_DIFF, "task", "sess8")

        assert result == "review text"


# ── DeveloperAgent inspector injection tests ─────────────────────────────────

class TestDeveloperAgentInspectorInjection:

    def _make_developer(self, inspector=None):
        """Return a DeveloperAgent with mocked settings and optional inspector."""
        from src.agents.developer.agent import DeveloperAgent

        with patch("config.settings.get_settings") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                sandbox_repos_dir="/tmp/repos",
                sandbox_python_image="python:3.11-slim",
                sandbox_timeout=60,
                sandbox_max_retries=3,
                github_pat="",
                gitlab_pat="",
                git_user_name="bot",
                git_user_email="bot@test",
            )
            agent = DeveloperAgent(llm=MagicMock(), inspector=inspector)
        return agent

    def test_accepts_inspector_kwarg(self):
        """DeveloperAgent.__init__ accepts inspector= keyword argument."""
        mock_inspector = MagicMock()
        agent = self._make_developer(inspector=mock_inspector)
        assert agent._inspector is mock_inspector

    def test_inspector_defaults_to_none(self):
        """DeveloperAgent._inspector is None when no inspector is injected."""
        agent = self._make_developer()
        assert agent._inspector is None

    def test_delegates_to_includes_developer_inspector(self):
        """DeveloperAgent.delegates_to lists 'developer_inspector'."""
        from src.agents.developer.agent import DeveloperAgent
        assert "developer_inspector" in DeveloperAgent.delegates_to
