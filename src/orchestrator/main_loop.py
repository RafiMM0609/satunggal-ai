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


_RATE_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "ratelimit",
    "too many requests", "quota exceeded", "quota_exceeded",
    "limit exceeded", "limit_exceeded",
    "insufficient credits", "insufficient_credits",
    "credits", "billing", "payment required",
    "usage limit", "token limit",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a provider rate-limit / quota-exceeded error.

    Checks:
    * httpx.HTTPStatusError with status 429 (Too Many Requests) or 402 (Payment Required)
    * Any exception whose string representation contains known rate-limit keywords
    """
    import httpx  # local import to avoid top-level dependency

    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in (402, 429):
            return True

    exc_text = str(exc).lower()
    return any(kw in exc_text for kw in _RATE_LIMIT_KEYWORDS)


_RATE_LIMIT_REPLY = (
    "⚠️ Maaf, layanan LLM saat ini telah mencapai batas penggunaan (rate limit) dari provider.\n"
    "Silakan tunggu beberapa saat dan coba lagi, atau hubungi administrator untuk memeriksa kuota API."
)

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
    from src.agents.doc_agent.agent import DocAgent
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
    from src.tools.docx_parser import DocxParserTool
    from src.tools.docx_editor import DocxEditorTool
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
        "docx_parser":        DocxParserTool(),
        "docx_editor":        DocxEditorTool(),
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
        "web_automation":      WebAutomationAgent(_llm, history=_history),
        "doc_agent":           DocAgent(_history, _llm),
        # Legacy aliases – kept so any hardcoded name still resolves
        "doc_auditor":         None,
        "doc_editor":          None,
    }
    # Point legacy names to the same instance
    _agents["doc_auditor"] = _agents["doc_agent"]
    _agents["doc_editor"]  = _agents["doc_agent"]
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

    # Pass recent conversation history (excluding the just-added current message)
    # so the gatekeeper can correctly classify follow-up commands (e.g.
    # "berikan screenshot" after a previous web_automation turn).
    # Pass None (not an empty list) when there are no prior messages so the
    # gatekeeper skips injecting an empty history section into its prompt.
    recent_history = history.get_as_llm_messages(session_id)[:-1][-4:] or None
    try:
        intent_result = await gatekeeper.classify_intent(
            user_text, session_id=session_id, history=recent_history
        )
    except Exception as gk_exc:
        if _is_rate_limit_error(gk_exc):
            logger.warning(
                "Gatekeeper rate-limit hit: session=%s error=%s", session_id, gk_exc
            )
            task.result = _RATE_LIMIT_REPLY
            history.add(session_id, "assistant", task.result)
            tracker.advance("done")
            await _notify(status_callback, tracker.render())
            return task
        raise
    task.mark_routed(intent_result.intent.value)
    logger.info(
        "Intent: session=%s intent=%s confidence=%.2f tools=%s needs_clarification=%s",
        session_id, intent_result.intent.value, intent_result.confidence,
        intent_result.tools, intent_result.needs_clarification,
    )
    tracker.complete_current()

    # 3b. Self-Correction: if the gatekeeper is unsure, ask the user back
    #     instead of forwarding to a specialist agent that might misfire.
    if intent_result.needs_clarification:
        clarification = (
            intent_result.clarification_question
            or "Maaf, saya belum memahami permintaan Anda. Boleh dijelaskan lebih detail?"
        )
        task.result = clarification
        history.add(session_id, "assistant", task.result)
        logger.info(
            "Self-correction triggered: session=%s → returning clarification question",
            session_id,
        )
        tracker.advance("done")
        await _notify(status_callback, tracker.render())
        return task

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

    try:
        task = await agent.run(task)
    except Exception as agent_exc:
        if _is_rate_limit_error(agent_exc):
            logger.warning(
                "Agent rate-limit hit: session=%s agent=%s error=%s",
                session_id, agent.name, agent_exc,
            )
            task.result = _RATE_LIMIT_REPLY
            history.add(session_id, "assistant", task.result)
            tracker.advance("done")
            await _notify(status_callback, tracker.render())
            return task
        raise
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
    """Wipe conversation history and document index for a session (e.g. on /reset command)."""
    if _history is not None:
        _history.clear(session_id)
    # Also clear any indexed documents for this session
    try:
        from src.memory.doc_index import get_doc_index
        get_doc_index().clear_session(session_id)
    except (ImportError, AttributeError, Exception) as exc:
        logger.warning("Failed to clear doc index for session=%s: %s", session_id, exc)
    # Clear web-automation state: last-visited URL + saved browser sessions
    try:
        from src.agents.web_automation.agent import clear_web_automation_session
        clear_web_automation_session(session_id)
    except (ImportError, AttributeError, Exception) as exc:
        logger.warning("Failed to clear web automation session for session=%s: %s", session_id, exc)
    logger.info("Session cleared: %s", session_id)


async def _run_pdf_quiz_pipeline(
    task: "AgentTask",
    agents: dict,
    tools: dict,
    session_id: str,
    original_filename: str,
    status_callback: "StatusCallback",
    history,
) -> "AgentTask":
    """
    Inner pipeline: PDF → Interactive HTML Quiz.
    Called by process_pdf after gatekeeper confirms quiz_generation intent.
    Runs pdf_parser (full document), QuizAgent, then WebQuizBuilderTool.
    """
    # ── Full Parse ─────────────────────────────────────────────────────────
    await _notify(status_callback, _quiz_status_msg(original_filename, phase="parsing"))
    pdf_tool = tools.get("pdf_parser")
    if pdf_tool is None:
        task.mark_failed("pdf_parser tool tidak terdaftar.")
        task.result = "❌ Sistem tidak dapat memproses PDF saat ini."
        history.add(session_id, "assistant", task.result)
        return task

    parser_result = await pdf_tool.run(task)
    if "error" in parser_result:
        task.mark_failed(parser_result["error"])
        task.result = f"❌ Gagal membaca PDF: {parser_result['error']}"
        history.add(session_id, "assistant", task.result)
        return task

    chunks: list[str] = parser_result["chunks"]
    task.metadata["pdf_chunks"] = chunks

    logger.info(
        "PDF full-parse: session=%s pages=%d words=%d chunks=%d",
        session_id,
        parser_result.get("total_pages", 0),
        parser_result.get("total_words", 0),
        len(chunks),
    )

    # ── QuizAgent ──────────────────────────────────────────────────────────
    quiz_agent = agents.get("quiz_agent")
    if quiz_agent is None:
        task.mark_failed("quiz_agent tidak terdaftar.")
        task.result = "❌ Quiz agent tidak tersedia."
        history.add(session_id, "assistant", task.result)
        return task

    task.mark_processing("quiz_agent")
    task = await quiz_agent.run(task)

    # Free parsed chunks from memory
    task.metadata.pop("pdf_chunks", None)

    if task.status.value == "failed":
        history.add(session_id, "assistant", task.result or "")
        return task

    # ── Build HTML (via pending_tools) ─────────────────────────────────────
    await _notify(status_callback, _quiz_status_msg(original_filename, phase="building"))

    for tool_name in list(task.pending_tools):
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("_run_pdf_quiz_pipeline: tool '%s' not in registry; skipping.", tool_name)
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

    history.add(session_id, "assistant", task.result)
    logger.info(
        "_run_pdf_quiz_pipeline done | session=%s html=%s",
        session_id, task.metadata.get("html_path", "MISSING"),
    )
    return task


async def process_pdf(
    session_id: str,
    pdf_path: str,
    original_filename: str = "document.pdf",
    user_caption: str = "",
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Intent-aware pipeline for any PDF document.

    Replaces the hardcoded process_pdf_quiz with a 3-step approach:
        1. Quick Peek  – parse only page 1 to get a document preview (fast, low RAM)
        2. Gatekeeper  – LLM classifies intent from user_caption + document preview
        3. Full Process – run the matched pipeline (currently: quiz_generation)

    Args:
        session_id:        Unique identifier for the conversation.
        pdf_path:          Absolute path to the downloaded PDF file.
        original_filename: Original filename shown to the user.
        user_caption:      Text the user sent together with the PDF (may be empty).
        status_callback:   Optional async callable for live progress updates.

    Returns:
        AgentTask with appropriate metadata set on success.
    """
    from src.memory.state import AgentTask

    history, agents, _, gatekeeper, tools = _get_pipeline()

    task = AgentTask(
        session_id=session_id,
        user_input=user_caption.strip() or f"[PDF: {original_filename}]",
    )
    task.metadata["pdf_path"]        = pdf_path
    task.metadata["status_callback"] = status_callback

    # Record the user turn in history
    history.add(
        session_id, "user",
        user_caption.strip() or f"[PDF dikirim: {original_filename}]",
    )

    # ── Step 1: Quick Peek (halaman pertama saja) ──────────────────────────
    await _notify(status_callback, _pdf_scanning_msg(original_filename, phase="scanning"))

    pdf_tool = tools.get("pdf_parser")
    if pdf_tool is None:
        task.mark_failed("pdf_parser tool tidak terdaftar.")
        task.result = "❌ Sistem tidak dapat memproses PDF saat ini."
        history.add(session_id, "assistant", task.result)
        return task

    task.metadata["pdf_max_pages"] = 1
    peek_result = await pdf_tool.run(task)
    task.metadata.pop("pdf_max_pages", None)

    if "error" in peek_result:
        task.mark_failed(peek_result["error"])
        task.result = f"❌ Gagal membaca PDF: {peek_result['error']}"
        history.add(session_id, "assistant", task.result)
        return task

    preview_chunks = peek_result.get("chunks", [])
    preview_text   = preview_chunks[0][:1500] if preview_chunks else ""

    logger.info(
        "PDF quick-peek: session=%s total_pages=%d preview_chars=%d",
        session_id, peek_result.get("total_pages", 0), len(preview_text),
    )

    # ── Step 2: Gatekeeper Decision ────────────────────────────────────────
    await _notify(status_callback, _pdf_scanning_msg(original_filename, phase="analyzing"))

    caption_part  = user_caption.strip() or "(tidak ada pesan dari pengguna)"
    enriched_text = f"[Pesan user: {caption_part}]\n[Preview dokumen: {preview_text}]"
    intent_result = await gatekeeper.classify_intent(enriched_text, session_id=session_id)
    task.mark_routed(intent_result.intent.value)

    logger.info(
        "PDF intent: session=%s intent=%s confidence=%.2f needs_clarification=%s",
        session_id, intent_result.intent.value, intent_result.confidence,
        intent_result.needs_clarification,
    )

    await _notify(status_callback, _pdf_scanning_msg(original_filename, phase="routing"))

    # ── Step 2b: Gatekeeper needs clarification ────────────────────────────
    if intent_result.needs_clarification:
        clarification = (
            intent_result.clarification_question
            or "Mau diapakan dokumen ini? Misalnya: buat kuis interaktif, ringkasan, atau keperluan lain?"
        )
        task.result = clarification
        history.add(session_id, "assistant", task.result)
        logger.info("PDF intent clarification requested: session=%s", session_id)
        return task

    # ── Step 3: Route to appropriate pipeline ─────────────────────────────
    intent_value = intent_result.intent.value

    if intent_value == "quiz_generation":
        task.metadata["quiz_title"] = _make_quiz_title(original_filename)
        # Try caption first, then recent conversation history (user may have typed
        # "buat kuis 30 soal" as a text message before sending the PDF).
        question_count = _extract_question_count(user_caption)
        if not question_count:
            for msg in reversed(history.get(session_id)[-6:]):
                if msg.role == "user":
                    question_count = _extract_question_count(msg.content)
                    if question_count:
                        break
        if question_count:
            task.metadata["quiz_question_count"] = question_count
        return await _run_pdf_quiz_pipeline(
            task, agents, tools, session_id, original_filename, status_callback, history,
        )

    # Unsupported intent – friendly explanation
    task.result = (
        f"ℹ️ Saya menerima PDF *{original_filename}* dan mendeteksi permintaan: *{intent_value}*.\n\n"
        f"Saat ini PDF hanya mendukung pembuatan *kuis interaktif*.\n"
        f"Kirim ulang PDF dengan pesan _\"buat kuis dari dokumen ini\"_ untuk memulai. 🎯"
    )
    logger.info("PDF intent '%s' not yet supported: session=%s", intent_value, session_id)
    history.add(session_id, "assistant", task.result)
    return task


async def process_pdf_quiz(
    session_id: str,
    pdf_path: str,
    original_filename: str = "document.pdf",
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Backward-compatible alias for process_pdf.
    Always routes to quiz_generation by passing a fixed caption.
    Existing callers continue to work without modification.
    """
    return await process_pdf(
        session_id=session_id,
        pdf_path=pdf_path,
        original_filename=original_filename,
        user_caption="buat kuis",
        status_callback=status_callback,
    )


def _make_quiz_title(filename: str) -> str:
    """Derive a human-friendly quiz title from the PDF filename."""
    name = filename.rsplit(".", 1)[0]  # strip extension
    # Replace underscores/hyphens with spaces, title-case
    name = name.replace("_", " ").replace("-", " ")
    return f"Kuis: {name.title()}"


def _extract_question_count(text: str) -> int | None:
    """
    Extract desired quiz question count from user caption.

    Examples:
        "buat 30 soal dari PDF ini" → 30
        "buat kuis 50 pertanyaan"   → 50
        "buat kuis"                 → None (auto)
    """
    import re as _re
    match = _re.search(r"(\d+)\s*(?:soal|pertanyaan|question|soals?)", text, _re.IGNORECASE)
    if match:
        count = int(match.group(1))
        return max(5, min(count, 150))  # clamp to a sane range
    return None


def _pdf_scanning_msg(filename: str, phase: str) -> str:
    """Build a Markdown progress message for the Quick Peek + Gatekeeper phase."""
    scan_icon    = "✅" if phase in ("analyzing", "routing", "done") else "🔄"
    analyze_icon = "✅" if phase in ("routing", "done") else ("🔄" if phase == "analyzing" else "⏳")
    route_icon   = "✅" if phase == "done" else ("🔄" if phase == "routing" else "⏳")

    return (
        f"⏳ *Memproses PDF...*\n\n"
        f"📄 *{filename}*\n\n"
        f"  • 🔍 Memindai dokumen:       {scan_icon} {'Selesai' if scan_icon == '✅' else 'Memproses...'}\n"
        f"  • 🧭 Menganalisis permintaan: {analyze_icon} {'Selesai' if analyze_icon == '✅' else ('Memproses...' if analyze_icon == '🔄' else 'Menunggu')}\n"
        f"  • 🚦 Mengarahkan ke pipeline: {route_icon} {'Selesai' if route_icon == '✅' else ('Memproses...' if route_icon == '🔄' else 'Menunggu')}"
    )


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


# ── Edit intent detection ──────────────────────────────────────────────────────

# Kata kunci yang menandakan pengguna ingin mengedit dokumen (Bahasa Indonesia + English)
_EDIT_KEYWORDS = frozenset({
    # Indonesian
    "edit", "ubah", "ganti", "tambah", "hapus", "perbaiki", "revisi",
    "modifikasi", "perbarui", "update", "koreksi", "perbaikan", "replace",
    "rubah", "tukar", "sesuaikan", "sisipkan", "insert", "buang", "delete",
    "hilangkan", "timpa", "overwrite", "pertegas", "perjelas", "lengkapi",
    # Action verbs for implementing suggestions
    "implementasikan", "implementasi", "terapkan", "laksanakan",
    "eksekusi", "aplikasikan",
    # English
    "change", "modify", "remove", "correct", "fix", "rewrite", "revise",
    "append", "add", "update", "alter",
})


def is_edit_intent(user_caption: str) -> bool:
    """
    Deteksi apakah caption mengandung instruksi untuk mengedit dokumen.

    Mengembalikan True jika caption mengandung kata kunci edit.
    """
    if not user_caption or not user_caption.strip():
        return False
    words = user_caption.lower().split()
    return any(
        w.strip(",.!?;:\"'-()/\\") in _EDIT_KEYWORDS for w in words
    )


async def process_doc_session_message(
    session_id: str,
    user_text: str,
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Pipeline khusus untuk sesi dokumen aktif.

    Membypass gatekeeper dan langsung memanggil DocAgent, sehingga
    instruksi edit / pertanyaan tidak bisa salah diklasifikasikan sebagai
    DOCUMENT_CREATION (→ technical_writer → document_generator → WeasyPrint
    error) oleh gatekeeper.
    """
    from src.memory.state import AgentTask

    history, agents, *_ = _get_pipeline()

    history.add(session_id, "user", user_text)

    task = AgentTask(session_id=session_id, user_input=user_text)
    task.mark_routed("doc_audit")
    task.mark_processing("doc_agent")

    doc_agent = agents.get("doc_agent")
    if doc_agent is None:
        task.mark_failed("doc_agent tidak terdaftar.")
        task.result = "❌ Doc agent tidak tersedia."
        history.add(session_id, "assistant", task.result)
        return task

    task.metadata["status_callback"] = status_callback

    try:
        task = await doc_agent.run(task)
    except Exception as exc:
        logger.exception("process_doc_session_message: doc_agent.run failed session=%s: %s", session_id, exc)
        task.mark_failed(str(exc))
        task.result = "❌ Terjadi kesalahan saat memproses. Silakan coba lagi."

    if not task.result:
        task.result = "Maaf, tidak ada respons yang dihasilkan."

    history.add(session_id, "assistant", task.result)
    return task


async def process_docx(
    session_id: str,
    docx_path: str,
    original_filename: str = "document.docx",
    user_caption: str = "",
    status_callback: "StatusCallback" = None,
) -> "AgentTask":
    """
    Pipeline untuk file .docx – memparse lalu menganalisis ATAU mengedit.

    Alur (Mode Analisis – default):
        1. Jalankan DocxParserTool → dapatkan seksi-seksi dokumen
        2. Serahkan ke DocAuditorAgent (step-planned):
           a. Ringkas setiap bab (LLM)
           b. Simpan ke SQLite (DocIndex)
           c. Kirim laporan (judul + daftar isi + ringkasan)

    Alur (Mode Edit – ketika caption mengandung instruksi edit):
        1. Jalankan DocxParserTool → dapatkan seksi-seksi + metadata dokumen
        2. Serahkan ke DocEditorAgent:
           a. Baca peta paragraf (indeks, style, teks)
           b. Kirim instruksi + peta ke LLM → hasilkan operasi edit JSON
           c. Terapkan edit via DocxEditorTool (XML-level precision)
           d. Kembalikan file .docx yang sudah diedit

    Args:
        session_id:        Identifier sesi (Telegram user_id).
        docx_path:         Path absolut ke file .docx yang sudah diunduh.
        original_filename: Nama file asli.
        user_caption:      Pesan yang dikirim bersama file (boleh kosong).
        status_callback:   Callable async untuk update progres.

    Returns:
        AgentTask dengan task.result berisi laporan teks.
        Jika mode edit: task.metadata["document_path"] berisi path file hasil edit.
    """
    from src.memory.state import AgentTask

    history, agents, _, _, tools = _get_pipeline()

    user_input = user_caption.strip() or f"[DOCX: {original_filename}]"
    task = AgentTask(session_id=session_id, user_input=user_input)
    task.metadata["docx_path"]        = docx_path
    task.metadata["original_filename"] = original_filename
    task.metadata["status_callback"]  = status_callback

    # Catat turn user ke histori
    history.add(
        session_id, "user",
        user_caption.strip() or f"[Dokumen dikirim: {original_filename}]",
    )

    # ── Deteksi mode: edit vs analisis ────────────────────────────────────
    edit_mode = is_edit_intent(user_caption)
    logger.info(
        "process_docx: session=%s file=%r edit_mode=%s caption=%r",
        session_id, original_filename, edit_mode, user_caption[:80],
    )

    # ── Langkah 1: Parse DOCX ─────────────────────────────────────────────
    await _notify(
        status_callback,
        f"⏳ *Membaca dokumen...*\n📄 _{original_filename}_\n\n🔄 Memindai struktur dokumen...",
    )

    docx_tool = tools.get("docx_parser")
    if docx_tool is None:
        task.mark_failed("docx_parser tool tidak terdaftar.")
        task.result = "❌ Sistem tidak dapat memproses file DOCX saat ini."
        history.add(session_id, "assistant", task.result)
        return task

    parse_result = await docx_tool.run(task)
    if "error" in parse_result:
        task.mark_failed(parse_result["error"])
        task.result = f"❌ Gagal membaca file .docx: {parse_result['error']}"
        history.add(session_id, "assistant", task.result)
        return task

    sections: list[dict] = parse_result.get("sections", [])
    doc_title: str       = parse_result.get("doc_title", original_filename)
    total_words: int     = parse_result.get("total_words", 0)
    total_sections: int  = parse_result.get("total_sections", len(sections))
    detection_method: str = parse_result.get("detection_method", "formal")

    logger.info(
        "process_docx: parsed session=%s file=%r sections=%d words=%d method=%s",
        session_id, original_filename, total_sections, total_words, detection_method,
    )

    # Masukkan hasil parse ke metadata task
    task.metadata["docx_sections"]    = sections
    task.metadata["doc_title"]         = doc_title
    task.metadata["docx_file_id"]      = original_filename
    task.metadata["total_words"]       = total_words
    task.metadata["detection_method"]  = detection_method

    # ── Langkah 2: Routing ke agent yang sesuai ───────────────────────────
    if edit_mode:
        # ── MODE EDIT: DocAgent ────────────────────────────────────────────
        task.mark_routed("doc_agent")
        doc_agent = agents.get("doc_agent")
        if doc_agent is None:
            task.mark_failed("doc_agent tidak terdaftar.")
            task.result = "❌ Doc agent tidak tersedia."
            history.add(session_id, "assistant", task.result)
            return task

        task.mark_processing("doc_agent")
        task = await doc_agent.run(task)
        task.metadata.pop("docx_sections", None)

        if not task.result:
            task.result = "✅ Dokumen berhasil diedit."

    else:
        # ── MODE ANALISIS: DocAgent ────────────────────────────────────────
        task.mark_routed("doc_audit")
        doc_agent = agents.get("doc_agent")
        if doc_agent is None:
            task.mark_failed("doc_agent tidak terdaftar.")
            task.result = "❌ Doc agent tidak tersedia."
            history.add(session_id, "assistant", task.result)
            return task

        task.mark_processing("doc_agent")
        task = await doc_agent.run(task)
        task.metadata.pop("docx_sections", None)

        if not task.result:
            task.result = "✅ Analisis dokumen selesai."

    history.add(session_id, "assistant", task.result)
    logger.info(
        "process_docx done | session=%s status=%s mode=%s",
        session_id, task.status, "edit" if edit_mode else "analyze",
    )
    return task

