from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.agents.reminder_agent.agent import ReminderAgent
from src.memory.state import AgentTask
from src.tools.reminder_store import Reminder


# ── Fixtures & Mock Helpers ───────────────────────────────────────────────────

@pytest.fixture
def mock_history():
    history = MagicMock()
    # By default, history is empty
    history.get_as_llm_messages.return_value = []
    return history


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add.return_value = Reminder(
        id=42,
        chat_id="sess123",
        message="Meeting with Team",
        remind_at=datetime(2026, 6, 9, 3, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc),
        fired=False
    )
    store.list_pending.return_value = []
    store.delete.return_value = True
    return store


@pytest.fixture
def mock_llm():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestReminderAgentHermes:

    @pytest.mark.asyncio
    @patch("src.agents.reminder_agent.agent.get_reminder_store")
    async def test_agent_get_current_time_flow(self, mock_get_store, mock_history, mock_store, mock_llm):
        """Test that the agent can execute the get_current_time tool and answer."""
        mock_get_store.return_value = mock_store
        agent = ReminderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to call get_current_time then answer
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Need current time", "action": "get_current_time"}',
            '{"thought": "Now I can reply", "action": "answer", "content": "Halo! Waktu sekarang jam 10 pagi."}'
        ])

        task = AgentTask(session_id="sess123", user_input="Jam berapa sekarang?")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert result_task.result == "Halo! Waktu sekarang jam 10 pagi."
        assert mock_llm.chat.call_count == 2

    @pytest.mark.asyncio
    @patch("src.agents.reminder_agent.agent.get_reminder_store")
    @patch("src.agents.reminder_agent.scheduler.schedule_reminder")
    async def test_agent_add_reminder_flow(self, mock_schedule, mock_get_store, mock_history, mock_store, mock_llm):
        """Test that the agent can parse, save, and schedule a reminder."""
        mock_get_store.return_value = mock_store
        agent = ReminderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to add reminder
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "User wants a reminder", "action": "add_reminder", "message": "Meeting with Team", "remind_at_iso": "2026-06-09T03:00:00Z"}',
            '{"thought": "Created successfully", "action": "answer", "content": "Sip! Pengingat untuk Meeting with Team berhasil diset."}'
        ])

        task = AgentTask(session_id="sess123", user_input="ingetin meeting besok jam 10 pagi")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "Meeting with Team" in result_task.result
        mock_store.add.assert_called_once()
        mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.agents.reminder_agent.agent.get_reminder_store")
    async def test_agent_list_reminders_flow(self, mock_get_store, mock_history, mock_store, mock_llm):
        """Test that the agent can query the database for pending reminders."""
        mock_get_store.return_value = mock_store
        agent = ReminderAgent(history=mock_history, llm=mock_llm)

        # Set some active reminders on mock store
        mock_store.list_pending.return_value = [
            Reminder(
                id=1,
                chat_id="sess123",
                message="Beli susu",
                remind_at=datetime(2026, 6, 9, 1, 0, tzinfo=timezone.utc),
                created_at=datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc),
                fired=False
            )
        ]

        # Mock LLM to list reminders
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Checking reminders", "action": "list_reminders"}',
            '{"thought": "List shown", "action": "answer", "content": "Kamu punya 1 reminder: Beli susu jam 8 pagi."}'
        ])

        task = AgentTask(session_id="sess123", user_input="daftar pengingat saya")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "Beli susu" in result_task.result
        mock_store.list_pending.assert_called_once_with("sess123")

    @pytest.mark.asyncio
    @patch("src.agents.reminder_agent.agent.get_reminder_store")
    @patch("src.agents.reminder_agent.scheduler.cancel_scheduled_reminder")
    async def test_agent_cancel_reminder_flow(self, mock_cancel_sched, mock_get_store, mock_history, mock_store, mock_llm):
        """Test that the agent can delete a reminder from database and scheduler."""
        mock_get_store.return_value = mock_store
        agent = ReminderAgent(history=mock_history, llm=mock_llm)

        # Mock LLM to cancel reminder
        mock_llm.chat = AsyncMock(side_effect=[
            '{"thought": "Cancelling reminder #42", "action": "cancel_reminder", "reminder_id": 42}',
            '{"thought": "Cancelled successfully", "action": "answer", "content": "Reminder #42 berhasil dibatalkan."}'
        ])

        task = AgentTask(session_id="sess123", user_input="batalin pengingat nomor 42")
        result_task = await agent.run(task)

        assert result_task.status.value == "done"
        assert "42" in result_task.result
        mock_store.delete.assert_called_once_with(42, "sess123")
        mock_cancel_sched.assert_called_once_with(42)

    @pytest.mark.asyncio
    @patch("src.agents.reminder_agent.agent.get_reminder_store")
    async def test_agent_forces_answer_at_max_steps(self, mock_get_store, mock_history, mock_store, mock_llm):
        """Test that the agent generates a fallback response if it hits _MAX_HERMES_STEPS."""
        mock_get_store.return_value = mock_store
        agent = ReminderAgent(history=mock_history, llm=mock_llm)

        # Set max steps low for quick testing
        agent._MAX_HERMES_STEPS = 2

        # Mock LLM to keep calling get_current_time (stuck in loop)
        mock_llm.chat = AsyncMock(return_value='{"thought": "looping", "action": "get_current_time"}')

        task = AgentTask(session_id="sess123", user_input="ping")
        result_task = await agent.run(task)

        # The loop should stop, force prompt added, and fall back (we mock the forced chat call too)
        assert result_task.status.value == "done"
        # Since mock_llm.chat returned looping action each time, final force fallback message will be used.
        # But wait, our mock_llm.chat returned get_current_time, even on forced final chat.
        # So json.loads will see "action": "get_current_time" or it will raise.
        # If it returns "get_current_time" (which doesn't have "content" field), final_answer becomes empty,
        # triggering the ultimate text fallback.
        assert "Maaf, saya kesulitan" in result_task.result
