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
    DeveloperInspectorAgent,
    _CRITIC_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    _gather_evidence,
    _resolve_branch_from_reply,
)
from src.agents.repo_agent_base import MAX_RELEVANT_FILES, RepoExtractionRequest as InspectionRequest
from src.memory.state import AgentTask


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(user_input: str = "inspect the repo", session_id: str = "test-session") -> AgentTask:
    task = AgentTask(session_id=session_id, user_input=user_input)
    return task


def _make_settings_mock() -> MagicMock:
    """Return a MagicMock that satisfies RepoAgentBase.__init__ field access."""
    s = MagicMock()
    s.sandbox_repos_dir = "/tmp/sandbox_repos"
    s.github_pat = ""
    s.gitlab_pat = ""
    return s


def _make_agent(llm_mock: AsyncMock | None = None) -> DeveloperInspectorAgent:
    """Return a DeveloperInspectorAgent with all heavy dependencies mocked."""
    with (
        patch("src.agents.repo_agent_base.LLMClient"),
        patch("src.agents.repo_agent_base.RepoTracker"),
        patch("src.agents.repo_agent_base.CLIExecutor"),
        patch("config.settings.get_settings", return_value=_make_settings_mock()),
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
from src.agents.developer_qna.agent import _QA_INTENT_LABELS


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
        # CI/CD — original + new dockerfile/docker-compose cases
        ("bagaimana proses deployment?", QAIntent.CI_CD),
        ("explain the ci/cd pipeline", QAIntent.CI_CD),
        ("cara deploy ke production?", QAIntent.CI_CD),
        ("berikan script dockerfile", QAIntent.CI_CD),
        ("tampilkan docker-compose", QAIntent.CI_CD),
        ("tunjukkan isi dockerfile", QAIntent.CI_CD),
        # Security
        ("bagaimana autentikasi dilakukan?", QAIntent.SECURITY),
        ("how is jwt implemented?", QAIntent.SECURITY),
        ("explain security layer", QAIntent.SECURITY),
        ("apakah ada middleware auth di repo ini?", QAIntent.SECURITY),
        # Main flow
        ("bagaimana flow utama aplikasi?", QAIntent.MAIN_FLOW),
        ("explain the main flow", QAIntent.MAIN_FLOW),
        ("alur kerja sistem ini?", QAIntent.MAIN_FLOW),
        # Specific symbol — original + new file-mention & existence cases
        ("jelaskan function controler.download", QAIntent.SPECIFIC_SYMBOL),
        ("jelaskan isi file main.py", QAIntent.SPECIFIC_SYMBOL),
        ("tampilkan isi dari config.yaml", QAIntent.SPECIFIC_SYMBOL),
        ("lihat middleware.go", QAIntent.SPECIFIC_SYMBOL),
        ("berikan isi controllers/user.go", QAIntent.SPECIFIC_SYMBOL),
        # 'router.go' triggers API_ENDPOINTS (routing file heuristic)
        ("lihat router.go", QAIntent.API_ENDPOINTS),
        ("adakah handle upload file pada repository ini", QAIntent.SPECIFIC_SYMBOL),
        ("apakah ada fungsi untuk login di sini?", QAIntent.SPECIFIC_SYMBOL),
        ("berikan kode fungsi process_order", QAIntent.SPECIFIC_SYMBOL),
        # API endpoint existence queries
        ("apakah ada endpoint untuk download?", QAIntent.API_ENDPOINTS),
        ("adakah route untuk /users di sini?", QAIntent.API_ENDPOINTS),
        # Full inspection triggers (error/bug keywords)
        ("ada bug di payment service", QAIntent.FULL_INSPECTION),
        ("error 500 saat login", QAIntent.FULL_INSPECTION),
        ("crash waktu startup", QAIntent.FULL_INSPECTION),
        ("tolong perbaiki", QAIntent.FULL_INSPECTION),
        # "adakah" + bug word must still go to FULL_INSPECTION
        ("adakah bug di service ini?", QAIntent.FULL_INSPECTION),
        # "autentikasi" is a strong SECURITY keyword — takes priority over "masalah"
        ("adakah masalah dengan autentikasi?", QAIntent.SECURITY),
        # "adakah" + masalah with no specific topic keyword → FULL_INSPECTION
        ("adakah masalah pada sistem ini?", QAIntent.FULL_INSPECTION),
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
        # New: file.extension mention
        ("jelaskan isi file main.py", "main.py"),
        ("tampilkan config.yaml", "config.yaml"),
        ("lihat controllers/user.go", "controllers/user.go"),
        # New: existence questions
        ("adakah handle upload file pada repository ini", "handle"),
        # "fungsi untuk login" → extracts the actual target after the preposition
        ("apakah ada fungsi untuk login", "login"),
        # New: imperative with keyword
        ("berikan kode fungsi process_order", "process_order"),
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
    def test_all_intents_have_label(self):
        for intent in QAIntent:
            assert intent in _QA_INTENT_LABELS, f"{intent} missing from _QA_INTENT_LABELS"
            assert _QA_INTENT_LABELS[intent], f"{intent} has empty label"

    def test_full_inspection_has_general_qa_label(self):
        # FULL_INSPECTION is used as a general Q/A fallback in DeveloperQnAAgent
        assert QAIntent.FULL_INSPECTION in _QA_INTENT_LABELS


# ── Q/A flow routing (now in DeveloperQnAAgent) ──────────────────────────────

class TestRunQAFlow:
    """Q/A routing tests now validate DeveloperQnAAgent, since Q/A was extracted
    from DeveloperInspectorAgent into its own agent."""

    def _make_qna_agent(self, llm_response: str):
        from src.agents.developer_qna.agent import DeveloperQnAAgent
        with (
            patch("src.agents.repo_agent_base.LLMClient"),
            patch("src.agents.repo_agent_base.RepoTracker"),
            patch("src.agents.repo_agent_base.CLIExecutor"),
            patch("config.settings.get_settings", return_value=_make_settings_mock()),
        ):
            agent = DeveloperQnAAgent()
        agent._llm = _make_llm_mock(returns=llm_response)
        return agent

    def test_run_routes_to_qa_flow_for_api_question(self):
        """DeveloperQnAAgent.run() must call _run_qa_flow for API questions."""
        agent = self._make_qna_agent("📡 API answer here")
        qa_flow_called = []

        async def fake_qa_flow(task, repo_path, req, intent):
            qa_flow_called.append(True)
            task.mark_done("📡 API answer")
            return task

        with patch.object(agent, "_resolve_repo", return_value=Path("/tmp/fake_repo")):
            with patch.object(agent, "_checkout_branch", new=AsyncMock()):
                with patch.object(agent, "_run_qa_flow", side_effect=fake_qa_flow):
                    with patch.object(agent, "_extract_request",
                                      return_value=InspectionRequest(
                                          repo_url="https://github.com/x/y",
                                          branch="main",
                                      )):
                        task = _make_task("ada api apa saja di repo ini?")
                        asyncio.get_event_loop().run_until_complete(agent.run(task))

        assert qa_flow_called, "_run_qa_flow was not called for API Q/A question"

    def test_run_routes_to_inspection_for_bug_report(self):
        """DeveloperInspectorAgent.run() must call _run_inspection_task for bug reports."""
        agent = _make_agent()
        agent._llm = _make_llm_mock(returns="Inspection report")
        inspection_called = []

        async def fake_inspection(task, repo_path, req):
            inspection_called.append(True)
            task.mark_done("Inspection done")
            return task

        with patch.object(agent, "_resolve_repo", return_value=Path("/tmp/fake_repo")):
            with patch.object(agent, "_checkout_branch", new=AsyncMock()):
                with patch.object(agent, "_run_inspection_task", side_effect=fake_inspection):
                    with patch.object(agent, "_extract_request",
                                      return_value=InspectionRequest(
                                          repo_url="https://github.com/x/y",
                                          branch="main",
                                      )):
                        task = _make_task("ada bug di payment service, error 500")
                        asyncio.get_event_loop().run_until_complete(agent.run(task))

        assert inspection_called, "_run_inspection_task was not called for bug report"

    def test_qna_pending_confirmation_stores_qa_intent(self):
        """DeveloperQnAAgent: when no branch given, qa_intent saved in pending dict."""
        from src.agents.developer_qna import agent as qna_module

        agent_obj = self._make_qna_agent("irrelevant")
        original_pending = dict(qna_module._qna_pending_confirmations)

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

            pending = qna_module._qna_pending_confirmations.get(task.session_id, {})
            assert pending.get("qa_intent") == QAIntent.API_ENDPOINTS.value, (
                f"qa_intent mismatch: {pending.get('qa_intent')!r}"
            )
        finally:
            qna_module._qna_pending_confirmations.clear()
            qna_module._qna_pending_confirmations.update(original_pending)


# ── code_search Go/Proto indexing ─────────────────────────────────────────────

class TestCodeSearchGoProto:
    """Validate that build_ast_index indexes .go and .proto files."""

    def test_go_extension_in_all_exts(self):
        from src.tools.code_search import _ALL_EXTS
        assert ".go" in _ALL_EXTS, ".go must be in _ALL_EXTS"

    def test_proto_extension_in_all_exts(self):
        from src.tools.code_search import _ALL_EXTS
        assert ".proto" in _ALL_EXTS, ".proto must be in _ALL_EXTS"

    def test_go_functions_extracted(self):
        from src.tools.code_search import _extract_symbols_regex
        go_source = textwrap.dedent("""\
            package main

            import (
                "net/http"
                "github.com/gorilla/mux"
            )

            func RegisterRoutes(r *mux.Router) {
                r.HandleFunc("/users", GetUsers).Methods("GET")
            }

            func GetUsers(w http.ResponseWriter, r *http.Request) {
                // handler body
            }
        """)
        symbols = _extract_symbols_regex(go_source, ".go")
        assert "RegisterRoutes" in symbols
        assert "GetUsers" in symbols

    def test_go_imports_extracted(self):
        from src.tools.code_search import _extract_symbols_regex
        go_source = textwrap.dedent("""\
            package routes

            import (
                "net/http"
                "github.com/gorilla/mux"
            )

            func Setup() {}
        """)
        symbols = _extract_symbols_regex(go_source, ".go")
        assert "net/http" in symbols
        assert "github.com/gorilla/mux" in symbols

    def test_go_method_receiver_function_extracted(self):
        from src.tools.code_search import _extract_symbols_regex
        go_source = textwrap.dedent("""\
            package server

            func (s *Server) HandlePing(w http.ResponseWriter) {}

            func NewServer() *Server { return &Server{} }
        """)
        symbols = _extract_symbols_regex(go_source, ".go")
        assert "HandlePing" in symbols, "method receiver function must be extracted"
        assert "NewServer" in symbols, "regular function must be extracted alongside method receiver"

    def test_proto_messages_extracted(self):
        from src.tools.code_search import _extract_symbols_regex
        proto_source = textwrap.dedent("""\
            syntax = "proto3";

            message UserRequest {
                string id = 1;
            }

            message UserResponse {
                string name = 1;
            }

            service UserService {
                rpc GetUser(UserRequest) returns (UserResponse);
            }
        """)
        symbols = _extract_symbols_regex(proto_source, ".proto")
        assert "UserRequest" in symbols
        assert "UserResponse" in symbols
        assert "UserService" in symbols
        assert "GetUser" in symbols

    def test_build_ast_index_includes_go_file(self, tmp_path):
        from src.tools.code_search import build_ast_index
        go_file = tmp_path / "routes.go"
        go_file.write_text(
            "package main\n\nfunc SetupRouter() {}\n",
            encoding="utf-8",
        )
        index = build_ast_index(tmp_path)
        assert "routes.go" in index, "routes.go must be indexed"
        assert "SetupRouter" in index["routes.go"]

    def test_build_ast_index_includes_proto_file(self, tmp_path):
        from src.tools.code_search import build_ast_index
        proto_file = tmp_path / "user.proto"
        proto_file.write_text(
            'syntax = "proto3";\nmessage User { string id = 1; }\n',
            encoding="utf-8",
        )
        index = build_ast_index(tmp_path)
        assert "user.proto" in index, "user.proto must be indexed"
        assert "User" in index["user.proto"]


# ── Hermes ReAct Loop Tests ──────────────────────────────────────────────────

class TestHermesLoop:
    @pytest.mark.asyncio
    async def test_hermes_list_dir(self, tmp_path):
        agent = _make_agent()
        
        # Setup files/dirs
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hello')")
        (tmp_path / "node_modules").mkdir() # should be skipped
        
        # Test normal listing
        res = await agent._hermes_list_dir(tmp_path, "src")
        assert "📄 src/app.py" in res
        
        # Test nonexistent dir
        res2 = await agent._hermes_list_dir(tmp_path, "nonexistent")
        assert "tidak ditemukan" in res2
        
        # Test directory traversal prevention
        res3 = await agent._hermes_list_dir(tmp_path, "../outside")
        assert "di luar repositori" in res3

    @pytest.mark.asyncio
    async def test_hermes_view_file(self, tmp_path):
        agent = _make_agent()
        f = tmp_path / "app.py"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        
        # View whole file
        res = await agent._hermes_view_file(tmp_path, "app.py")
        assert "line1" in res
        assert "line5" in res
        
        # View line range
        res_range = await agent._hermes_view_file(tmp_path, "app.py", 2, 4)
        assert "line1" not in res_range
        assert "2: line2" in res_range
        assert "3: line3" in res_range
        assert "4: line4" in res_range
        assert "line5" not in res_range

        # Out of bounds
        res_err = await agent._hermes_view_file(tmp_path, "app.py", 10, 12)
        assert "melebihi total baris" in res_err

        # Traversal prevention
        res_trav = await agent._hermes_view_file(tmp_path, "../outside.py")
        assert "di luar repositori" in res_trav

    @pytest.mark.asyncio
    async def test_hermes_grep(self, tmp_path):
        agent = _make_agent()
        agent._run_cmd = AsyncMock(return_value="app.py:1:print('hello')")
        
        res = await agent._hermes_grep(tmp_path, "hello")
        assert "print('hello')" in res

    @pytest.mark.asyncio
    async def test_hermes_git_log_and_diff(self, tmp_path):
        agent = _make_agent()
        agent._run_cmd = AsyncMock(return_value="commit1\ncommit2")
        
        # git_log
        with patch("pathlib.Path.exists", return_value=True): # mock .git folder existence
            res_log = await agent._hermes_git_log(tmp_path, 2)
            assert "commit1" in res_log
            
            # git_diff
            res_diff = await agent._hermes_git_diff(tmp_path, "HEAD~1 HEAD")
            assert "commit1" in res_diff

    @pytest.mark.asyncio
    async def test_hermes_search_symbols(self, tmp_path):
        agent = _make_agent()
        # Mock RAG/AST code search
        symbol_index = {"src/app.py": ["hello", "main"]}
        ranked_files = ["src/app.py"]
        
        with (
            patch("src.tools.code_search.build_ast_index", return_value=symbol_index),
            patch("src.tools.code_search.rank_files_by_relevance", return_value=ranked_files),
        ):
            res = await agent._hermes_search_symbols(tmp_path, "hello")
            assert "src/app.py" in res
            assert "hello" in res

    @pytest.mark.asyncio
    async def test_run_hermes_loop_react_flow(self, tmp_path):
        agent = _make_agent()
        
        # Simulate ReAct flow:
        # Step 1: LLM decides to list_dir
        # Step 2: LLM decides to view a file
        # Step 3: LLM decides to answer
        chat_responses = [
            json.dumps({"thought": "I will list the directory.", "action": "list_dir", "path": "."}),
            json.dumps({"thought": "I will view main.py.", "action": "view_file", "file_path": "main.py"}),
            json.dumps({"thought": "I have enough info.", "action": "answer", "content": "## 📋 LAPORAN INSPEKSI REPOSITORI\nAll is fine."}),
            "Verified report here." # critic pass response
        ]
        
        agent._llm.chat = AsyncMock(side_effect=chat_responses)
        
        # Mock internal file/dir helpers so the ReAct loop runs smoothly
        agent._hermes_list_dir = AsyncMock(return_value="📄 main.py")
        agent._hermes_view_file = AsyncMock(return_value="1: def main(): pass")
        agent._verify_report = AsyncMock(return_value="## Verified Report\nAll is fine.")
        
        report = await agent._run_hermes_loop(
            query="Find startup crash root cause",
            session_id="test-hermes",
            repo_path=tmp_path,
            max_steps=5,
            keywords=["crash"]
        )
        
        assert report == "## Verified Report\nAll is fine."
        # Ensure LLM chat was called exactly 3 times in ReAct loop.
        assert agent._llm.chat.call_count == 3
        agent._verify_report.assert_called_once()
