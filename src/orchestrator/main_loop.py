"""
Main orchestration loop – single entry point for ALL interfaces.

Flow:
    Interface (Telegram / REST API / CLI)
        │
        ▼
    process_message(session_id, user_text)
        │
        ├─► GatekeeperAgent  →  classify intent + select pre-agent tools
        │
        ├─► Pre-agent tool loop
        │       Run tools in intent_result.tools (e.g. tavily_search)
        │       Results stored in task.tool_results
        │
        ├─► AgentRouter  →  pick specialist agent by intent
        │
        ├─► Agent.run(task)
        │       Agent calls LLM, parses output, writes to task.metadata
        │       Agent appends tool names to task.pending_tools
        │
        ├─► Post-agent tool loop
        │       Run tools in task.pending_tools (e.g. wbs_generator)
        │       Results stored in task.tool_results + task.metadata
        │
        └─► return task  →  back to interface
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.memory.state import AgentTask


# ── Lazy singletons ────────────────────────────────────────────────────────────

_history    = None
_llm        = None
_agents     = None
_router     = None
_gatekeeper = None
_tools      = None   # dict[str, BaseTool]


def _get_pipeline():
    """Lazily create and cache all pipeline components."""
    global _history, _llm, _agents, _router, _gatekeeper, _tools

    if _gatekeeper is not None:
        return _history, _agents, _router, _gatekeeper, _tools

    from src.agents.content_creator.agent import ContentCreatorAgent
    from src.agents.gatekeeper.agent import GatekeeperAgent
    from src.agents.llm_client import LLMClient
    from src.agents.mandays_agent.agent import MandaysAgent
    from src.agents.researcher.agent import ResearcherAgent
    from src.agents.responder.agent import ResponderAgent
    from src.agents.wbs_agent.agent import WBSAgent
    from src.memory.history import ConversationHistory
    from src.orchestrator.router import AgentRouter
    from src.tools.mandays_generator import MandaysGeneratorTool
    from src.tools.wbs_generator import WBSGeneratorTool

    _history = ConversationHistory(max_messages=30)
    _llm     = LLMClient()

    # ── Tool registry ──────────────────────────────────────────────────────
    # Keyed by tool name (same string used in intent_result.tools and
    # task.pending_tools).  Excel-builder tools are pure/deterministic
    # and run post-agent via pending_tools; tavily runs pre-agent.
    _tools = {
        "wbs_generator":     WBSGeneratorTool(),
        "mandays_generator": MandaysGeneratorTool(),
    }

    # Tavily is optional – only registered when API key is available.
    try:
        from src.tools.tavily_search import TavilySearchTool
        _tavily = TavilySearchTool()
        _tools["tavily_search"] = _tavily
    except (ValueError, ImportError):
        _tavily = None
        logger.warning("TAVILY_API_KEY not configured – tavily_search tool disabled.")

    # ── Agent registry ─────────────────────────────────────────────────────
    _agents = {
        "responder":        ResponderAgent(_history, _llm),
        "researcher":       ResearcherAgent(_history, _llm),
        "content_creator":  ContentCreatorAgent(_history, _llm),
        "wbs_agent":        WBSAgent(_llm),
        "mandays_agent":    MandaysAgent(_llm),
    }
    _router     = AgentRouter(_agents)
    _gatekeeper = GatekeeperAgent()

    logger.info(
        "Pipeline initialised: %d agents, %d tools registered.",
        len(_agents), len(_tools),
    )
    return _history, _agents, _router, _gatekeeper, _tools


# ── Public API ────────────────────────────────────────────────────────────────

async def process_message(session_id: str, user_text: str) -> "AgentTask":
    """
    Core pipeline called by every interface.

    Args:
        session_id: Unique identifier for the conversation (e.g. Telegram user_id).
        user_text:  The raw text from the user.

    Returns:
        The completed AgentTask (task.result holds the reply text;
        task.metadata may contain extra data such as "excel_path").
    """
    from src.memory.state import AgentTask

    history, agents, router, gatekeeper, tools = _get_pipeline()

    # 1. Record the user turn in history
    history.add(session_id, "user", user_text)

    # 2. Build the task blackboard
    task = AgentTask(session_id=session_id, user_input=user_text)

    # 3. Classify intent (gatekeeper) – now also returns which tools to run
    intent_result = await gatekeeper.classify_intent(user_text, session_id=session_id)
    task.mark_routed(intent_result.intent.value)
    logger.info(
        "Intent: session=%s intent=%s confidence=%.2f tools=%s",
        session_id, intent_result.intent.value, intent_result.confidence,
        intent_result.tools,
    )

    # 4. Execute tools declared by gatekeeper (before calling specialist agent)
    for tool_name in intent_result.tools:
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("Tool '%s' requested by gatekeeper but not registered; skipping.", tool_name)
            continue
        try:
            logger.info("Executing tool '%s' for session=%s", tool_name, session_id)
            tool_output = await tool.run(task)
            task.tool_results[tool_name] = tool_output

            # Propagate critical keys to task.metadata so handlers can find them
            if "excel_path" in tool_output:
                task.metadata["excel_path"] = tool_output["excel_path"]

            logger.info(
                "Tool '%s' done for session=%s keys=%s",
                tool_name, session_id, list(tool_output.keys()),
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception: %s", tool_name, exc)
            task.tool_results[tool_name] = {"error": str(exc)}

    # 5. Route to specialist agent
    agent = router.resolve(task)
    task.mark_processing(agent.name)

    # 6. Execute agent (task.tool_results is already populated)
    task = await agent.run(task)
    logger.info(
        "Agent done: session=%s agent=%s pending_tools=%s",
        session_id, agent.name, task.pending_tools,
    )

    # 6b. Post-agent: drain pending_tools set by the agent during its run
    for tool_name in list(task.pending_tools):
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("pending_tools: '%s' not in registry; skipping.", tool_name)
            continue
        try:
            logger.info("Post-agent tool '%s' starting for session=%s", tool_name, session_id)
            tool_output = await tool.run(task)
            task.tool_results[tool_name] = tool_output
            if "excel_path" in tool_output:
                task.metadata["excel_path"] = tool_output["excel_path"]
            logger.info(
                "Post-agent tool '%s' done for session=%s keys=%s",
                tool_name, session_id, list(tool_output.keys()),
            )
        except Exception as exc:
            logger.exception("Post-agent tool '%s' raised: %s", tool_name, exc)
            task.tool_results[tool_name] = {"error": str(exc)}
    task.pending_tools.clear()

    # 7. Build final reply text (stored on task for callers)
    if not task.result:
        task.result = "Maaf, saya tidak dapat memproses permintaan Anda saat ini."

    # 8. Record assistant turn in history
    history.add(session_id, "assistant", task.result)

    logger.info(
        "pipeline done | session=%s intent=%s agent=%s status=%s",
        session_id, task.intent, agent.name, task.status,
    )
    return task


async def clear_session(session_id: str) -> None:
    """Wipe conversation history for a session (e.g. on /reset command)."""
    if _history is not None:
        _history.clear(session_id)
    logger.info("Session cleared: %s", session_id)

