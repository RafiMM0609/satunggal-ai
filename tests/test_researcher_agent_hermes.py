from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.agents.researcher.agent import ResearcherAgent
from src.memory.state import AgentTask


# ── Fixtures & Mock Helpers ───────────────────────────────────────────────────

@pytest.fixture
def mock_history():
    history = MagicMock()
    history.get_as_llm_messages.return_value = []
    return history


@pytest.fixture
def mock_profile_store():
    store = MagicMock()
    store.get_all_preferences.return_value = {
        "preferred_name": "Boss",
        "explanation_style": "code_focused",
        "ignored_domains": '["medium.com"]',
        "trusted_domains": '["github.com"]'
    }
    return store


@pytest.fixture
def mock_llm():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestResearcherAgentHermes:

    @pytest.mark.asyncio
    @patch("src.agents.researcher.agent.get_user_profile_store")
    async def test_agent_get_current_time_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the researcher agent can execute the get_current_time tool and answer."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResearcherAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to call get_current_time then answer
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Need time to find year", "action": "get_current_time"}',
            '{"thought": "Now I can reply", "action": "answer", "content": "Riset berita terbaru selesai."}'
        ])

        task = AgentTask(session_id="sess123", user_input="riset berita terbaru")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert result_task.result == "Riset berita terbaru selesai."
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    @patch("src.agents.researcher.agent.get_user_profile_store")
    async def test_agent_get_user_profile_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the agent can read user profile and apply preferences."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResearcherAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to read user profile
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Load user profile", "action": "get_user_profile"}',
            '{"thought": "Profile loaded. Preferred name is Boss.", "action": "answer", "content": "Halo Boss! Ini hasil riset untukmu."}'
        ])

        task = AgentTask(session_id="sess123", user_input="halo")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "Boss" in result_task.result
        mock_profile_store.get_all_preferences.assert_called_once_with("sess123")

    @pytest.mark.asyncio
    @patch("src.agents.researcher.agent.get_user_profile_store")
    async def test_agent_update_user_profile_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the agent can update user profile preferences."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResearcherAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to update user profile
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "User wants concise explanations", "action": "update_user_profile", "profile_key": "explanation_style", "profile_value": "concise"}',
            '{"thought": "Updated. Proceeding to answer.", "action": "answer", "content": "Sip, sekarang preferensi gaya riset kamu diatur ke concise."}'
        ])

        task = AgentTask(session_id="sess123", user_input="Mulai sekarang buat penjelasan riset singkat saja")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "concise" in result_task.result
        mock_profile_store.set_preference.assert_called_once_with("sess123", "explanation_style", "concise")

    @pytest.mark.asyncio
    @patch("src.agents.researcher.agent.get_user_profile_store")
    async def test_delegation_mode_prompt_addendum(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that delegation mode appends the instruction warning against profile/time actions."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResearcherAgent(history=mock_history, llm=mock_llm)

        # We will capture the system prompt sent to LLM Client
        mock_llm.chat = AsyncMock(return_value='{"thought": "Quick research", "action": "answer", "content": "Fast summary"}')

        await agent.research_for_delegation(query="deep learning status", session_id="del123")

        # Verify chat arguments
        called_messages = mock_llm.chat.call_args[0][0]
        system_message = next(msg for msg in called_messages if msg["role"] == "system")
        
        # Check that CATATAN DELEGASI and constraint warning are present in system prompt
        assert "CATATAN DELEGASI" in system_message["content"]
        assert "Jangan gunakan tindakan 'get_user_profile', 'update_user_profile', atau 'get_current_time'" in system_message["content"]
