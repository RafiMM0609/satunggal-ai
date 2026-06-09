from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from telegram.constants import ParseMode

from src.proactive.daily_briefing import _send_long_message, _run_briefing


@pytest.mark.asyncio
async def test_send_long_message_short():
    """Test that a short message is sent as a single message with markdown ParseMode."""
    bot = MagicMock()
    bot.send_message = AsyncMock()
    
    text = "Short message"
    await _send_long_message(bot, "chat123", text)
    
    bot.send_message.assert_called_once_with(
        chat_id="chat123",
        text=text,
        parse_mode=ParseMode.MARKDOWN
    )


@pytest.mark.asyncio
async def test_send_long_message_long_split():
    """Test that a long message is split into paragraph-aligned chunks."""
    bot = MagicMock()
    bot.send_message = AsyncMock()

    # Create paragraphs that total > 4000 characters
    p1 = "A" * 2500
    p2 = "B" * 2000
    text = f"{p1}\n{p2}"
    
    await _send_long_message(bot, "chat123", text)
    
    assert bot.send_message.call_count == 2
    bot.send_message.assert_any_call(
        chat_id="chat123",
        text=p1,
        parse_mode=ParseMode.MARKDOWN
    )
    bot.send_message.assert_any_call(
        chat_id="chat123",
        text=p2,
        parse_mode=ParseMode.MARKDOWN
    )


@pytest.mark.asyncio
async def test_send_long_message_fallback_on_parse_error():
    """Test that if markdown send fails, it falls back to plain text."""
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=[Exception("Markdown error"), None])
    
    text = "Broken *markdown"
    await _send_long_message(bot, "chat123", text)
    
    assert bot.send_message.call_count == 2
    bot.send_message.assert_any_call(
        chat_id="chat123",
        text=text,
        parse_mode=ParseMode.MARKDOWN
    )
    bot.send_message.assert_any_call(
        chat_id="chat123",
        text=text
    )


@pytest.mark.asyncio
@patch("src.proactive._bot_ref.get_bot")
@patch("src.agents.researcher.agent.ResearcherAgent")
async def test_run_briefing_flow(mock_researcher_class, mock_get_bot):
    """Test that _run_briefing orchestrates the briefing flow correctly."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_get_bot.return_value = mock_bot
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.research_for_briefing = AsyncMock(side_effect=[
        "Summary for AI",
        "Summary for Tech"
    ])
    mock_researcher_class.return_value = mock_agent_instance
    
    topics = ["AI", "Tech"]
    chat_id = "12345"
    language = "id"
    
    await _run_briefing(chat_id, topics, language)
    
    # Verify research_for_briefing was called for each topic
    assert mock_agent_instance.research_for_briefing.call_count == 2
    mock_agent_instance.research_for_briefing.assert_any_call(
        topic="AI",
        language="id",
        session_id="proactive_briefing"
    )
    mock_agent_instance.research_for_briefing.assert_any_call(
        topic="Tech",
        language="id",
        session_id="proactive_briefing"
    )
    
    # Verify bot.send_message was called with compiled output
    assert mock_bot.send_message.call_count == 1
    call_args = mock_bot.send_message.call_args[1]
    assert call_args["chat_id"] == chat_id
    assert "Briefing Harian" in call_args["text"]
    assert "📌 *Ai*\nSummary for AI" in call_args["text"]
    assert "📌 *Tech*\nSummary for Tech" in call_args["text"]
