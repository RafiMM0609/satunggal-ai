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
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type alias for the optional progress callback passed in by the interface layer.
# Signature: async callback(rendered_text: str) -> None
StatusCallback = Optional[Callable[[str], Awaitable[None]]]


async def _notify(cb: StatusCallback, text: str) -> None:
    """Fire the progress callback, swallowing any errors so the pipeline continues."""
    if cb is None:
        return
    try:
        await cb(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Progress callback raised: %s", exc)

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
    from src.agents.developer.agent import DeveloperAgent
    from src.agents.developer_inspector.agent import DeveloperInspectorAgent
    from src.agents.developer_qna.agent import DeveloperQnAAgent
    from src.agents.gatekeeper.agent import GatekeeperAgent
    from src.agents.llm_client import LLMClient
    from src.agents.log_viewer_agent.agent import LogViewerAgent
    from src.agents.mandays_agent.agent import MandaysAgent
    from src.agents.quiz_agent.agent import QuizAgent
    from src.agents.researcher.agent import ResearcherAgent
    from src.agents.responder.agent import ResponderAgent
    from src.agents.sysinfo_agent.agent import SysInfoAgent
    from src.agents.technical_writer.agent import TechnicalWriterAgent
    from src.agents.wbs_agent.agent import WBSAgent
    from src.agents.web_automation.agent import WebAutomationAgent
    from src.memory.history import ConversationHistory
    from src.orchestrator.router import AgentRouter
    from src.tools.browser_navigator import BrowserNavigatorTool
    from src.tools.diagram_renderer import DiagramRendererTool
    from src.tools.document_generator import DocumentGeneratorTool
    from src.tools.mandays_generator import MandaysGeneratorTool
    from src.tools.pdf_parser import PDFParserTool
    from src.tools.web_quiz_builder import WebQuizBuilderTool
    from src.tools.web_reader import WebReaderTool
    from src.tools.wbs_generator import WBSGeneratorTool

    _history = ConversationHistory(max_messages=30)
    _llm     = LLMClient()

    # ── Tool registry ──────────────────────────────────────────────────────
    # Keyed by tool name (same string used in intent_result.tools and
    # task.pending_tools).  Excel-builder tools are pure/deterministic
    # and run post-agent via pending_tools; tavily runs pre-agent.
    _tools = {
        "wbs_generator":      WBSGeneratorTool(),
        "mandays_generator":  MandaysGeneratorTool(),
        "diagram_renderer":   DiagramRendererTool(),
        "document_generator": DocumentGeneratorTool(),
        "pdf_parser":         PDFParserTool(),
        "web_quiz_builder":   WebQuizBuilderTool(),
        "web_reader":         WebReaderTool(),
        "browser_navigator":  BrowserNavigatorTool(),
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
        "responder":         ResponderAgent(_history, _llm),
        "researcher":        ResearcherAgent(_history, _llm),
        "content_creator":   ContentCreatorAgent(_history, _llm),
        "wbs_agent":         WBSAgent(_llm),
        "mandays_agent":     MandaysAgent(_llm),
        "developer":            DeveloperAgent(_llm),
        "developer_inspector": DeveloperInspectorAgent(llm=_llm, history=_history),
        "developer_qna":        DeveloperQnAAgent(llm=_llm, history=_history),
        "technical_writer":    TechnicalWriterAgent(_history, _llm),
        "sysinfo_agent":       SysInfoAgent(_history, _llm),
        "log_viewer_agent":    LogViewerAgent(_history, _llm),
        "quiz_agent":          QuizAgent(_llm),
        "web_automation":      WebAutomationAgent(_llm),
    }
    _router     = AgentRouter(_agents)
    _gatekeeper = GatekeeperAgent()

    logger.info(
        "Pipeline initialised: %d agents, %d tools registered.",
        len(_agents), len(_tools),
    )
    return _history, _agents, _router, _gatekeeper, _tools


# ── Public API ────────────────────────────────────────────────────────────────

async def process_message(
    session_id: str,
    user_text: str,
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Core pipeline called by every interface.

    Args:
        session_id:       Unique identifier for the conversation (e.g. Telegram user_id).
        user_text:        The raw text from the user.
        status_callback:  Optional async callable ``async (rendered_text: str) -> None``
                          invoked at every pipeline stage to update a live progress message.

    Returns:
        The completed AgentTask (task.result holds the reply text;
        task.metadata may contain extra data such as "excel_path").
    """
    from src.memory.state import AgentTask
    from src.tools.progress_tracker import ProgressTracker

    history, agents, router, gatekeeper, tools = _get_pipeline()

    # ── Bootstrap progress tracker ─────────────────────────────────────────
    tracker = ProgressTracker(title="⏳ Sedang memproses permintaan...")

    # 1. Record the user turn in history
    history.add(session_id, "user", user_text)

    # 2. Build the task blackboard
    task = AgentTask(session_id=session_id, user_input=user_text)

    # 3. Classify intent (gatekeeper) – now also returns which tools to run
    tracker.advance("gatekeeper")
    await _notify(status_callback, tracker.render())

    intent_result = await gatekeeper.classify_intent(user_text, session_id=session_id)
    task.mark_routed(intent_result.intent.value)
    logger.info(
        "Intent: session=%s intent=%s confidence=%.2f tools=%s",
        session_id, intent_result.intent.value, intent_result.confidence,
        intent_result.tools,
    )
    tracker.complete_current()

    # 4. Execute tools declared by gatekeeper (before calling specialist agent)
    for tool_name in intent_result.tools:
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("Tool '%s' requested by gatekeeper but not registered; skipping.", tool_name)
            continue

        tracker.advance(f"pre_tool:{tool_name}")
        await _notify(status_callback, tracker.render())

        try:
            logger.info("Executing tool '%s' for session=%s", tool_name, session_id)
            tool_output = await tool.run(task)
            task.tool_results[tool_name] = tool_output

            # Propagate critical keys to task.metadata so handlers can find them
            if "excel_path" in tool_output:
                task.metadata["excel_path"] = tool_output["excel_path"]
            if "document_path" in tool_output:
                task.metadata["document_path"] = tool_output["document_path"]

            logger.info(
                "Tool '%s' done for session=%s keys=%s",
                tool_name, session_id, list(tool_output.keys()),
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception: %s", tool_name, exc)
            task.tool_results[tool_name] = {"error": str(exc)}

        tracker.complete_current()

    # 5. Route to specialist agent
    agent = router.resolve(task)
    task.mark_processing(agent.name)

    # 6. Execute agent (task.tool_results is already populated)
    tracker.advance(f"agent:{agent.name}")
    await _notify(status_callback, tracker.render())

    task = await agent.run(task)
    logger.info(
        "Agent done: session=%s agent=%s pending_tools=%s",
        session_id, agent.name, task.pending_tools,
    )
    tracker.complete_current()

    # 6b. Post-agent: drain pending_tools set by the agent during its run
    for tool_name in list(task.pending_tools):
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("pending_tools: '%s' not in registry; skipping.", tool_name)
            continue

        tracker.advance(f"post_tool:{tool_name}")
        await _notify(status_callback, tracker.render())

        try:
            logger.info("Post-agent tool '%s' starting for session=%s", tool_name, session_id)
            tool_output = await tool.run(task)
            task.tool_results[tool_name] = tool_output
            if "excel_path" in tool_output:
                task.metadata["excel_path"] = tool_output["excel_path"]
            if "document_path" in tool_output:
                task.metadata["document_path"] = tool_output["document_path"]
            logger.info(
                "Post-agent tool '%s' done for session=%s keys=%s",
                tool_name, session_id, list(tool_output.keys()),
            )
        except Exception as exc:
            logger.exception("Post-agent tool '%s' raised: %s", tool_name, exc)
            task.tool_results[tool_name] = {"error": str(exc)}

        tracker.complete_current()

    task.pending_tools.clear()

    # 6c. Signal 100 % done before returning
    tracker.advance("done")
    await _notify(status_callback, tracker.render())

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


async def process_pdf_quiz(
    session_id: str,
    pdf_path: str,
    original_filename: str = "document.pdf",
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Dedicated pipeline for PDF → Interactive HTML Quiz generation.

    Bypasses the gatekeeper (intent is already known: quiz_generation).
    Flow:
        1. PDFParserTool   → extract text chunks from PDF
        2. QuizAgent       → generate questions batch by batch
        3. WebQuizBuilderTool → compile questions into single-file HTML

    Args:
        session_id:        Unique identifier for the conversation.
        pdf_path:          Absolute path to the downloaded PDF file.
        original_filename: Original filename shown to the user.
        status_callback:   Optional async callable for live progress updates.

    Returns:
        AgentTask with task.metadata["html_path"] set on success.
    """
    from src.memory.state import AgentTask

    _, agents, _, _, tools = _get_pipeline()

    task = AgentTask(
        session_id=session_id,
        user_input=f"Buat kuis dari PDF: {original_filename}",
        intent="quiz_generation",
    )
    task.metadata["pdf_path"]         = pdf_path
    task.metadata["quiz_title"]       = _make_quiz_title(original_filename)
    task.metadata["status_callback"]  = status_callback

    # ── Step 1: Parse PDF ──────────────────────────────────────────────────
    await _notify(status_callback, _quiz_status_msg(original_filename, phase="parsing"))
    pdf_tool = tools.get("pdf_parser")
    if pdf_tool is None:
        task.mark_failed("pdf_parser tool tidak terdaftar.")
        task.result = "❌ Sistem tidak dapat memproses PDF saat ini."
        return task

    parser_result = await pdf_tool.run(task)
    if "error" in parser_result:
        task.mark_failed(parser_result["error"])
        task.result = f"❌ Gagal membaca PDF: {parser_result['error']}"
        return task

    chunks: list[str] = parser_result["chunks"]
    task.metadata["pdf_chunks"] = chunks

    logger.info(
        "PDF parsed: session=%s pages=%d words=%d chunks=%d",
        session_id,
        parser_result.get("total_pages", 0),
        parser_result.get("total_words", 0),
        len(chunks),
    )

    # ── Step 2: Run QuizAgent (batch processing) ───────────────────────────
    quiz_agent = agents.get("quiz_agent")
    if quiz_agent is None:
        task.mark_failed("quiz_agent tidak terdaftar.")
        task.result = "❌ Quiz agent tidak tersedia."
        return task

    task.mark_processing("quiz_agent")
    task = await quiz_agent.run(task)

    # Free parsed chunks from memory
    task.metadata.pop("pdf_chunks", None)

    if task.status.value == "failed":
        return task

    # ── Step 3: Build HTML (via pending_tools) ─────────────────────────────
    await _notify(status_callback, _quiz_status_msg(original_filename, phase="building"))

    for tool_name in list(task.pending_tools):
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("process_pdf_quiz: tool '%s' not in registry; skipping.", tool_name)
            continue
        try:
            tool_output = await tool.run(task)
            task.tool_results[tool_name] = tool_output
            if "html_path" in tool_output:
                task.metadata["html_path"] = tool_output["html_path"]
            logger.info(
                "PDF quiz tool '%s' done for session=%s keys=%s",
                tool_name, session_id, list(tool_output.keys()),
            )
        except Exception as exc:
            logger.exception("PDF quiz tool '%s' raised: %s", tool_name, exc)
            task.tool_results[tool_name] = {"error": str(exc)}

    task.pending_tools.clear()

    await _notify(status_callback, _quiz_status_msg(original_filename, phase="done"))

    if not task.result:
        task.result = "✅ Kuis berhasil dibuat!"

    logger.info(
        "process_pdf_quiz done | session=%s html=%s",
        session_id, task.metadata.get("html_path", "MISSING"),
    )
    return task


def _make_quiz_title(filename: str) -> str:
    """Derive a human-friendly quiz title from the PDF filename."""
    name = filename.rsplit(".", 1)[0]  # strip extension
    # Replace underscores/hyphens with spaces, title-case
    name = name.replace("_", " ").replace("-", " ")
    return f"Kuis: {name.title()}"


def _quiz_status_msg(filename: str, phase: str) -> str:
    """Build a Markdown progress message for the PDF quiz pipeline."""
    pdf_icon   = "✅" if phase in ("generating", "building", "done") else "🔄"
    gen_icon   = "✅" if phase in ("building", "done") else ("🔄" if phase == "generating" else "⏳")
    build_icon = "✅" if phase == "done" else ("🔄" if phase == "building" else "⏳")
    final_icon = "✅" if phase == "done" else "⏳"

    return (
        f"⏳ *Proses Pembuatan Kuis Aktif*\n\n"
        f"📝 *{_make_quiz_title(filename)}*\n\n"
        f"  • 📄 Membaca PDF: {pdf_icon} {'Selesai' if pdf_icon == '✅' else 'Memproses...'}\n"
        f"  • 🧠 Menghasilkan Soal: {gen_icon} {'Selesai' if gen_icon == '✅' else ('Memproses...' if gen_icon == '🔄' else 'Menunggu')}\n"
        f"  • 🏗️ Membangun Website: {build_icon} {'Selesai' if build_icon == '✅' else ('Memproses...' if build_icon == '🔄' else 'Menunggu')}\n"
        f"  • 📦 Finalisasi File: {final_icon} {'Selesai' if final_icon == '✅' else 'Menunggu'}"
    )

