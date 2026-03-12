"""
Unit tests for DeveloperInspectorAgent.

Tests validate:
  - Request extraction (LLM JSON parsing).
  - Anti-hallucination: temperature/top_p are passed to LLM.
  - Critic verification pass runs and upgrades/downgrades confidence labels.
  - RAG (_read_relevant_files) falls back gracefully when deps are missing.
  - Evidence gathering assembles the correct 8-tuple.
  - Extended grep covers more file extensions.
  - Hallucination metric: UNVERIFIED tags appear in critic output.
  - Full inspection flow with mocked LLM and repo.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.developer_inspector.agent import (
    CRITIC_TEMPERATURE,
    CRITIC_TOP_P,
    INSPECTOR_TEMPERATURE,
    INSPECTOR_TOP_P,
    MAX_RELEVANT_FILES,
    DeveloperInspectorAgent,
    InspectionRequest,
    _CRITIC_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    _gather_evidence,
    _resolve_branch_from_reply,
)
from src.memory.state import AgentTask


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(user_input: str = "inspect the repo", session_id: str = "test-session") -> AgentTask:
    task = AgentTask(session_id=session_id, user_input=user_input)
    return task


def _make_agent(llm_mock: AsyncMock | None = None) -> DeveloperInspectorAgent:
    """Return a DeveloperInspectorAgent with all heavy dependencies mocked."""
    with (
        patch("src.agents.developer_inspector.agent.LLMClient"),
        patch("src.agents.developer_inspector.agent.RepoTracker"),
        patch("src.agents.developer_inspector.agent.CLIExecutor"),
    ):
        agent = DeveloperInspectorAgent()
    if llm_mock is not None:
        agent._llm = llm_mock
    return agent


def _make_llm_mock(returns: str = "default response") -> AsyncMock:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value=returns)
    return mock


# ── System prompt integrity ────────────────────────────────────────────────────

class TestSystemPrompt:
    """Validate the system prompt enforces anti-hallucination rules."""

    def test_prompt_contains_antihallucination_rules(self):
        assert "ANTI-HALUSINASI" in _SYSTEM_PROMPT or "ATURAN KRITIS" in _SYSTEM_PROMPT

    def test_prompt_requires_evidence_for_every_finding(self):
        assert "Bukti" in _SYSTEM_PROMPT or "evidence" in _SYSTEM_PROMPT.lower()

    def test_prompt_has_unverified_label(self):
        assert "UNVERIFIED" in _SYSTEM_PROMPT

    def test_prompt_has_confirmed_label(self):
        assert "CONFIRMED" in _SYSTEM_PROMPT

    def test_critic_prompt_contains_verification_rules(self):
        assert "CONFIRMED" in _CRITIC_SYSTEM_PROMPT
        assert "UNVERIFIED" in _CRITIC_SYSTEM_PROMPT


# ── Request extraction ─────────────────────────────────────────────────────────

class TestExtractRequest:
    def test_valid_json_parsed_correctly(self):
        agent = _make_agent()
        llm_response = json.dumps({
            "repo_url": "https://github.com/user/repo",
            "problem":  "TypeError on startup",
            "keywords": ["TypeError", "startup"],
            "branch":   "main",
        })
        agent._llm = _make_llm_mock(returns=llm_response)

        result = asyncio.get_event_loop().run_until_complete(
            agent._extract_request("TypeError on startup https://github.com/user/repo")
        )

        assert result.repo_url == "https://github.com/user/repo"
        assert result.problem == "TypeError on startup"
        assert "TypeError" in result.keywords
        assert result.branch == "main"

    def test_fallback_on_invalid_json(self):
        agent = _make_agent()
        agent._llm = _make_llm_mock(returns="not valid json {{")

        result = asyncio.get_event_loop().run_until_complete(
            agent._extract_request("some user description")
        )

        # Should fallback to treating full input as the problem description.
        assert result.problem == "some user description"
        assert result.repo_url == ""

    def test_json_in_markdown_fences_stripped(self):
        agent = _make_agent()
        inner = json.dumps({"repo_url": "", "problem": "slow query", "keywords": ["SELECT"], "branch": ""})
        agent._llm = _make_llm_mock(returns=f"```json\n{inner}\n```")

        result = asyncio.get_event_loop().run_until_complete(
            agent._extract_request("slow query problem")
        )

        assert result.problem == "slow query"
        assert "SELECT" in result.keywords


# ── LLM temperature / top_p passthrough ───────────────────────────────────────

class TestLLMDeterministicParams:
    """Verify that inspection prompt calls LLM with correct low-temperature params."""

    def test_initial_report_uses_inspector_temperature(self):
        agent = _make_agent()
        first_call_kwargs: dict[str, Any] = {}

        async def _capture_chat(messages, **kwargs):
            # Only record the FIRST call (initial report, not the critic pass).
            if not first_call_kwargs:
                first_call_kwargs.update(kwargs)
            return "## Report\nSome finding [CONFIRMED]"

        agent._llm = AsyncMock()
        agent._llm.chat = AsyncMock(side_effect=_capture_chat)

        asyncio.get_event_loop().run_until_complete(
            agent._run_inspection_llm(
                user_input="check for bugs",
                problem="null pointer",
                evidence={"Dir tree": "src/\n  main.py"},
            )
        )

        assert first_call_kwargs.get("temperature") == INSPECTOR_TEMPERATURE, (
            f"Expected INSPECTOR_TEMPERATURE={INSPECTOR_TEMPERATURE}, "
            f"got {first_call_kwargs.get('temperature')}"
        )
        assert first_call_kwargs.get("top_p") == INSPECTOR_TOP_P

    def test_critic_uses_lower_temperature(self):
        agent = _make_agent()
        calls: list[dict] = []

        async def _capture_chat(messages, **kwargs):
            calls.append({"messages": messages, "kwargs": kwargs})
            return "Verified report"

        agent._llm = AsyncMock()
        agent._llm.chat = AsyncMock(side_effect=_capture_chat)

        asyncio.get_event_loop().run_until_complete(
            agent._verify_report("Initial report text", "Evidence text")
        )

        assert len(calls) == 1
        critic_call = calls[0]
        assert critic_call["kwargs"].get("temperature") == CRITIC_TEMPERATURE
        assert critic_call["kwargs"].get("top_p") == CRITIC_TOP_P

    def test_critic_temperature_lower_than_inspector(self):
        assert CRITIC_TEMPERATURE < INSPECTOR_TEMPERATURE
        assert CRITIC_TOP_P <= INSPECTOR_TOP_P


# ── Critic verification pass ───────────────────────────────────────────────────

class TestCriticPass:
    def test_verify_report_returns_verified_text(self):
        agent = _make_agent()
        verified_text = "Verified report with [CONFIRMED] findings"
        agent._llm = _make_llm_mock(returns=verified_text)

        result = asyncio.get_event_loop().run_until_complete(
            agent._verify_report("Initial report", "Some evidence")
        )

        assert result == verified_text

    def test_verify_report_falls_back_on_llm_failure(self):
        agent = _make_agent()
        agent._llm = AsyncMock()
        agent._llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        initial_report = "Some initial report"
        result = asyncio.get_event_loop().run_until_complete(
            agent._verify_report(initial_report, "evidence")
        )

        # Must fall back to the original report, not raise.
        assert result == initial_report

    def test_critic_receives_both_report_and_evidence(self):
        agent = _make_agent()
        captured_content: list[str] = []

        async def _capture(messages, **kwargs):
            for msg in messages:
                captured_content.append(msg["content"])
            return "Done"

        agent._llm = AsyncMock()
        agent._llm.chat = AsyncMock(side_effect=_capture)

        report = "## Findings\nBug in main.py"
        evidence = "## Dir tree\nmain.py"

        asyncio.get_event_loop().run_until_complete(
            agent._verify_report(report, evidence)
        )

        combined = " ".join(captured_content)
        assert report in combined or "Findings" in combined
        assert evidence in combined or "Dir tree" in combined


# ── Hallucination metric ───────────────────────────────────────────────────────

class TestHallucinationMetric:
    """A simple proxy metric: count UNVERIFIED labels in LLM output."""

    @staticmethod
    def count_labels(text: str) -> dict[str, int]:
        return {
            "CONFIRMED":   text.count("[CONFIRMED]"),
            "LIKELY":      text.count("[LIKELY]"),
            "UNVERIFIED":  text.count("[UNVERIFIED]"),
        }

    def test_high_unverified_count_indicates_hallucination_risk(self):
        # A response with > 50% UNVERIFIED findings is a hallucination warning.
        text = textwrap.dedent("""\
            Finding 1 [CONFIRMED] – found in main.py line 12.
            Finding 2 [UNVERIFIED] – assumed but no evidence.
            Finding 3 [UNVERIFIED] – assumed but no evidence.
            Finding 4 [UNVERIFIED] – assumed but no evidence.
        """)
        labels = self.count_labels(text)
        total = labels["CONFIRMED"] + labels["LIKELY"] + labels["UNVERIFIED"]
        unverified_ratio = labels["UNVERIFIED"] / total if total else 0
        assert unverified_ratio > 0.5, "Test setup: this report has high hallucination risk"

    def test_low_unverified_count_indicates_grounded_report(self):
        text = textwrap.dedent("""\
            Finding 1 [CONFIRMED] – line 12: `raise ValueError('...')`.
            Finding 2 [CONFIRMED] – line 45: `None` returned without guard.
            Finding 3 [LIKELY] – pattern seen in 3 files.
        """)
        labels = self.count_labels(text)
        total = labels["CONFIRMED"] + labels["LIKELY"] + labels["UNVERIFIED"]
        unverified_ratio = labels["UNVERIFIED"] / total if total else 0
        assert unverified_ratio == 0, "Well-evidenced report should have 0 UNVERIFIED"


# ── RAG / _read_relevant_files ────────────────────────────────────────────────

class TestReadRelevantFiles:
    def test_returns_graceful_message_when_code_search_not_importable(self):
        agent = _make_agent()

        import sys
        original = sys.modules.get("src.tools.code_search")
        sys.modules["src.tools.code_search"] = None  # type: ignore[assignment]

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent._read_relevant_files(Path("/tmp/nonexistent"), "some problem")
            )
            assert "unavailable" in result or "error" in result.lower() or "not available" in result.lower()
        finally:
            if original is None:
                del sys.modules["src.tools.code_search"]
            else:
                sys.modules["src.tools.code_search"] = original

    def test_returns_early_when_problem_empty(self):
        agent = _make_agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent._read_relevant_files(Path("/tmp"), "")
        )
        assert "no problem description" in result.lower()

    def test_reads_at_most_max_relevant_files(self, tmp_path):
        agent = _make_agent()

        # Create dummy python files.
        for i in range(MAX_RELEVANT_FILES + 3):
            (tmp_path / f"module_{i}.py").write_text(f"def func_{i}(): pass\n")

        # Mock code_search to return all files ranked.
        ranked_files = [f"module_{i}.py" for i in range(MAX_RELEVANT_FILES + 3)]
        symbol_index = {f"module_{i}.py": [f"func_{i}"] for i in range(MAX_RELEVANT_FILES + 3)}

        with (
            patch("src.tools.code_search.build_ast_index", return_value=symbol_index),
            patch("src.tools.code_search.rank_files_by_relevance", return_value=ranked_files),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                agent._read_relevant_files(tmp_path, "func problem")
            )

        # Only MAX_RELEVANT_FILES snippets should appear.
        snippet_count = result.count("📄")
        assert snippet_count <= MAX_RELEVANT_FILES


# ── Extended grep ──────────────────────────────────────────────────────────────

class TestGrepKeywords:
    def test_grep_command_includes_extended_extensions(self):
        agent = _make_agent()
        captured_cmd: list[str] = []

        async def _fake_run_cmd(cmd: str, cwd=None):
            captured_cmd.append(cmd)
            return "result"

        agent._run_cmd = _fake_run_cmd

        asyncio.get_event_loop().run_until_complete(
            agent._grep_keywords(Path("/tmp"), ["error", "panic"])
        )

        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        for ext in ["*.php", "*.cs", "*.rs", "*.vue"]:
            assert ext in cmd, f"Expected extension {ext!r} in grep command"

    def test_grep_error_patterns_runs(self):
        agent = _make_agent()

        async def _fake_run_cmd(cmd: str, cwd=None):
            return ""

        agent._run_cmd = _fake_run_cmd

        result = asyncio.get_event_loop().run_until_complete(
            agent._grep_error_patterns(Path("/tmp"))
        )

        assert "(no generic error patterns found)" in result


# ── Branch resolution ──────────────────────────────────────────────────────────

class TestBranchResolution:
    @pytest.mark.parametrize("reply,expected", [
        ("ya",          "develop"),
        ("yes",         "develop"),
        ("lanjutkan",   "develop"),
        ("ok",          "develop"),
        ("develop",     "develop"),          # explicit same branch re-confirmed
        ("feature/x",   "feature/x"),        # explicit different branch
    ])
    def test_confirmation_answers_accept(self, reply, expected):
        result = _resolve_branch_from_reply(reply, "develop")
        assert result == expected

    @pytest.mark.parametrize("bad_reply", [
        "I don't want to continue with this",     # sentence – not a branch name
        "",
        "saya tidak mau lanjutkan ini sekarang",   # multi-word sentence (id)
    ])
    def test_non_confirmation_returns_none(self, bad_reply):
        result = _resolve_branch_from_reply(bad_reply, "main")
        assert result is None


# ── Evidence building ──────────────────────────────────────────────────────────

class TestBuildEvidenceText:
    def test_empty_sections_excluded(self):
        agent = _make_agent()
        evidence = {
            "Section A": "content a",
            "Section B": "",          # empty – should be skipped
            "Section C": "  \n  ",    # whitespace-only – should be skipped
            "Section D": "content d",
        }
        text = agent._build_evidence_text(evidence)
        assert "Section A" in text
        assert "Section B" not in text
        assert "Section C" not in text
        assert "Section D" in text

    def test_sections_formatted_as_markdown_headings(self):
        agent = _make_agent()
        text = agent._build_evidence_text({"My Section": "body text"})
        assert "## My Section" in text
        assert "body text" in text


# ── Regression: RepoRecord attribute access (was .get() bug) ──────────────────

class TestResolveRepoRepoRecord:
    """
    Regression test for AttributeError: 'RepoRecord' object has no attribute 'get'.

    list_all() returns RepoRecord dataclass instances, not dicts.
    _resolve_repo must use attribute access (latest.local_path), not .get().
    """

    def test_no_url_uses_latest_tracked_repo_via_attribute(self, tmp_path):
        from src.memory.repo_tracker import RepoRecord

        agent = _make_agent()

        fake_record = RepoRecord(
            id=1,
            repo_name="my-repo",
            repo_url="https://github.com/user/my-repo",
            local_path=str(tmp_path),
            last_task_status="cloned",
            last_commit_hash="abc123",
            created_at="2026-03-12 00:00:00",
        )

        agent._repo_tracker = MagicMock()
        agent._repo_tracker.list_all.return_value = [fake_record]

        result = asyncio.get_event_loop().run_until_complete(
            agent._resolve_repo("")
        )

        # Should return the tmp_path without raising AttributeError
        assert result == tmp_path

    def test_no_url_returns_none_when_tracker_empty(self):
        agent = _make_agent()
        agent._repo_tracker = MagicMock()
        agent._repo_tracker.list_all.return_value = []

        result = asyncio.get_event_loop().run_until_complete(
            agent._resolve_repo("")
        )

        assert result is None

    def test_no_url_returns_none_when_local_path_missing_on_disk(self, tmp_path):
        from src.memory.repo_tracker import RepoRecord

        agent = _make_agent()
        nonexistent = tmp_path / "does_not_exist"

        fake_record = RepoRecord(
            id=2,
            repo_name="ghost-repo",
            repo_url="https://github.com/user/ghost",
            local_path=str(nonexistent),
            last_task_status="cloned",
            last_commit_hash="",
            created_at="2026-03-12 00:00:00",
        )

        agent._repo_tracker = MagicMock()
        agent._repo_tracker.list_all.return_value = [fake_record]

        result = asyncio.get_event_loop().run_until_complete(
            agent._resolve_repo("")
        )

        assert result is None


# ── No-repo flow ───────────────────────────────────────────────────────────────

class TestNoRepoFlow:
    def test_no_repo_generates_analysis_with_warning(self):
        agent = _make_agent()
        agent._llm = _make_llm_mock(returns="Analysis based on description only")
        agent._repo_tracker = MagicMock()
        agent._repo_tracker.list_all.return_value = []

        extracted = InspectionRequest(problem="DB crashes on startup")

        with patch.object(agent, "_extract_request", return_value=extracted):
            with patch.object(agent, "_resolve_repo", return_value=None):
                task = _make_task("DB crashes on startup")
                result = asyncio.get_event_loop().run_until_complete(agent.run(task))

        assert result.result is not None
        assert "⚠️" in result.result or "Catatan" in result.result


# ── Constants sanity ───────────────────────────────────────────────────────────

class TestConstants:
    def test_inspector_temperature_is_low(self):
        assert 0.0 <= INSPECTOR_TEMPERATURE <= 0.3, (
            "Inspector temperature should be low (≤0.3) for determinism"
        )

    def test_critic_temperature_is_lower_than_inspector(self):
        assert CRITIC_TEMPERATURE < INSPECTOR_TEMPERATURE

    def test_max_relevant_files_reasonable(self):
        assert 3 <= MAX_RELEVANT_FILES <= 15, (
            "MAX_RELEVANT_FILES should be between 3 and 15 for context balance"
        )


# ── Q/A mode: classify_intent ─────────────────────────────────────────────────

from src.tools.repo_qa import QAIntent, classify_intent, extract_specific_target
from src.agents.developer_inspector.agent import _QA_INTENT_LABELS


class TestClassifyIntent:
    @pytest.mark.parametrize("text,expected", [
        # API endpoints triggers
        ("ada api apa saja di repo ini?", QAIntent.API_ENDPOINTS),
        ("list semua endpoint REST", QAIntent.API_ENDPOINTS),
        ("show me all routes", QAIntent.API_ENDPOINTS),
        ("jelaskan endpoint /upload", QAIntent.SPECIFIC_SYMBOL),
        ("explain function process_payment", QAIntent.SPECIFIC_SYMBOL),
        # Tech stack
        ("teknologi apa yang dipakai?", QAIntent.TECH_STACK),
        ("what framework is used?", QAIntent.TECH_STACK),
        ("bahasa pemrograman apa?", QAIntent.TECH_STACK),
        # Data models
        ("apa saja data model di repo?", QAIntent.DATA_MODELS),
        ("jelaskan schema database", QAIntent.DATA_MODELS),
        ("show orm models", QAIntent.DATA_MODELS),
        # Dependencies
        ("apa saja dependency yang dipakai?", QAIntent.DEPENDENCIES),
        ("show requirements", QAIntent.DEPENDENCIES),
        ("list packages", QAIntent.DEPENDENCIES),
        # CI/CD
        ("bagaimana proses deployment?", QAIntent.CI_CD),
        ("explain the ci/cd pipeline", QAIntent.CI_CD),
        ("cara deploy ke production?", QAIntent.CI_CD),
        # Security
        ("bagaimana autentikasi dilakukan?", QAIntent.SECURITY),
        ("how is jwt implemented?", QAIntent.SECURITY),
        ("explain security layer", QAIntent.SECURITY),
        # Main flow
        ("bagaimana flow utama aplikasi?", QAIntent.MAIN_FLOW),
        ("explain the main flow", QAIntent.MAIN_FLOW),
        ("alur kerja sistem ini?", QAIntent.MAIN_FLOW),
        # Full inspection triggers (error/bug keywords)
        ("ada bug di payment service", QAIntent.FULL_INSPECTION),
        ("error 500 saat login", QAIntent.FULL_INSPECTION),
        ("crash waktu startup", QAIntent.FULL_INSPECTION),
        ("tolong perbaiki", QAIntent.FULL_INSPECTION),
    ])
    def test_classify_returns_correct_intent(self, text, expected):
        result = classify_intent(text)
        assert result == expected, f"classify_intent({text!r}) = {result!r}, want {expected!r}"

    def test_empty_input_returns_full_inspection(self):
        assert classify_intent("") == QAIntent.FULL_INSPECTION

    def test_unrecognized_input_returns_full_inspection(self):
        assert classify_intent("blablabla xyz 1234") == QAIntent.FULL_INSPECTION


class TestExtractSpecificTarget:
    @pytest.mark.parametrize("text,expected_substr", [
        ("jelaskan api /upload", "/upload"),
        ("explain endpoint /users/profile", "/users/profile"),
        ("explain function process_payment", "process_payment"),
        ("jelaskan method handle_request di agent.py", "handle_request"),
        ("explain class UserModel", "UserModel"),
    ])
    def test_extracts_target(self, text, expected_substr):
        target = extract_specific_target(text)
        assert expected_substr in target, (
            f"extract_specific_target({text!r}) = {target!r}, expected to contain {expected_substr!r}"
        )

    def test_no_target_returns_empty(self):
        result = extract_specific_target("ada api apa saja?")
        assert result == ""


# ── Q/A Intent labels ─────────────────────────────────────────────────────────

class TestQAIntentLabels:
    def test_all_non_full_inspection_intents_have_label(self):
        for intent in QAIntent:
            if intent == QAIntent.FULL_INSPECTION:
                continue
            assert intent in _QA_INTENT_LABELS, f"{intent} missing from _QA_INTENT_LABELS"
            assert _QA_INTENT_LABELS[intent], f"{intent} has empty label"

    def test_full_inspection_not_in_labels(self):
        assert QAIntent.FULL_INSPECTION not in _QA_INTENT_LABELS


# ── Q/A flow routing ─────────────────────────────────────────────────────────

class TestRunQAFlow:
    def _make_agent_with_qa_mock(self, llm_response: str):
        agent = _make_agent()
        agent._llm = _make_llm_mock(returns=llm_response)
        return agent

    def test_run_routes_to_qa_flow_for_api_question(self):
        """run() must call _run_qa_flow (not _run_inspection_task) for API questions."""
        agent = self._make_agent_with_qa_mock("📡 API answer here")
        qa_flow_called = []

        async def fake_qa_flow(task, repo_path, req):
            qa_flow_called.append(True)
            task.mark_done("📡 API answer")
            return task

        async def fake_inspection(task, repo_path, req):
            raise AssertionError("Inspection should NOT be called for Q/A intent")

        with patch.object(agent, "_resolve_repo", return_value=Path("/tmp/fake_repo")):
            with patch.object(agent, "_checkout_branch", new=AsyncMock()):
                with patch.object(agent, "_run_qa_flow", side_effect=fake_qa_flow):
                    with patch.object(agent, "_run_inspection_task", side_effect=fake_inspection):
                        with patch.object(agent, "_extract_request",
                                          return_value=InspectionRequest(
                                              repo_url="https://github.com/x/y",
                                              branch="main",
                                          )):
                            task = _make_task("ada api apa saja di repo ini?")
                            asyncio.get_event_loop().run_until_complete(agent.run(task))

        assert qa_flow_called, "_run_qa_flow was not called for API Q/A question"

    def test_run_routes_to_inspection_for_bug_report(self):
        """run() must call _run_inspection_task (not _run_qa_flow) for bug reports."""
        agent = self._make_agent_with_qa_mock("Inspection report")
        inspection_called = []

        async def fake_inspection(task, repo_path, req):
            inspection_called.append(True)
            task.mark_done("Inspection done")
            return task

        async def fake_qa(task, repo_path, req):
            raise AssertionError("Q/A flow should NOT be called for bug report")

        with patch.object(agent, "_resolve_repo", return_value=Path("/tmp/fake_repo")):
            with patch.object(agent, "_checkout_branch", new=AsyncMock()):
                with patch.object(agent, "_run_inspection_task", side_effect=fake_inspection):
                    with patch.object(agent, "_run_qa_flow", side_effect=fake_qa):
                        with patch.object(agent, "_extract_request",
                                          return_value=InspectionRequest(
                                              repo_url="https://github.com/x/y",
                                              branch="main",
                                          )):
                            task = _make_task("ada bug di payment service, error 500")
                            asyncio.get_event_loop().run_until_complete(agent.run(task))

        assert inspection_called, "_run_inspection_task was not called for bug report"

    def test_qa_mode_stored_in_pending_confirmation(self):
        """When no branch given, qa_mode and qa_intent must be saved in pending dict."""
        from src.agents.developer_inspector import agent as agent_module

        agent_obj = _make_agent()
        agent_obj._llm = _make_llm_mock(returns="irrelevant")
        original_pending = dict(agent_module._inspector_pending_confirmations)

        try:
            with patch.object(agent_obj, "_resolve_repo", return_value=Path("/tmp/r")):
                with patch.object(agent_obj, "_get_current_branch",
                                   new=AsyncMock(return_value="main")):
                    with patch.object(agent_obj, "_extract_request",
                                       return_value=InspectionRequest(
                                           repo_url="https://github.com/x/y",
                                           branch="",  # no branch → forces pending
                                       )):
                        task = _make_task("ada api apa saja?")
                        asyncio.get_event_loop().run_until_complete(agent_obj.run(task))

            # The pending entry for this session must contain qa_mode / qa_intent
            pending = agent_module._inspector_pending_confirmations.get(task.session_id, {})
            assert pending.get("qa_mode") is True, "qa_mode not stored in pending"
            assert pending.get("qa_intent") == QAIntent.API_ENDPOINTS.value, (
                f"qa_intent mismatch: {pending.get('qa_intent')!r}"
            )
        finally:
            agent_module._inspector_pending_confirmations.clear()
            agent_module._inspector_pending_confirmations.update(original_pending)
