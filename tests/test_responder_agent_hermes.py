from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.agents.responder.agent import ResponderAgent
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
        "preferred_vibe": "genz"
    }
    return store


@pytest.fixture
def mock_llm():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestResponderAgentHermes:

    @pytest.mark.asyncio
    @patch("src.agents.responder.agent.get_user_profile_store")
    async def test_agent_get_current_time_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the responder agent can execute get_current_time and answer."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResponderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to get time then answer
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Need current time", "action": "get_current_time"}',
            '{"thought": "Now I can reply", "action": "answer", "content": "Sekarang jam 11 siang cuy!"}'
        ])

        task = AgentTask(session_id="sess123", user_input="jam berapa sekarang bro")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "11 siang" in result_task.result
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    @patch("src.agents.responder.agent.get_user_profile_store")
    async def test_agent_get_user_profile_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the agent can read user profile and address user correctly."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResponderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to get profile
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Check user profile", "action": "get_user_profile"}',
            '{"thought": "User preferred name is Boss. Reply casually.", "action": "answer", "content": "Halo Boss, ada apa cuy?"}'
        ])

        task = AgentTask(session_id="sess123", user_input="halo")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "Boss" in result_task.result
        mock_profile_store.get_all_preferences.assert_called_once_with("sess123")

    @pytest.mark.asyncio
    @patch("src.agents.responder.agent.get_user_profile_store")
    async def test_agent_update_user_profile_flow(self, mock_get_profile, mock_history, mock_profile_store, mock_llm):
        """Test that the agent can save new preferred vibe preferences."""
        mock_get_profile.return_value = mock_profile_store
        agent = ResponderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to update profile
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "User wants formal vibe", "action": "update_user_profile", "profile_key": "preferred_vibe", "profile_value": "formal"}',
            '{"thought": "Updated. Confirming to user.", "action": "answer", "content": "Baik, saya akan menggunakan gaya bahasa formal mulai sekarang."}'
        ])

        task = AgentTask(session_id="sess123", user_input="Tolong bicara formal saja")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "formal" in result_task.result
        mock_profile_store.set_preference.assert_called_once_with("sess123", "preferred_vibe", "formal")
