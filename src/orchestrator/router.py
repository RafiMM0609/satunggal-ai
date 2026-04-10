"""
AgentRouter – maps intent categories to specialist agents.

Add new intents here as you add more agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.gatekeeper.schemas import IntentCategory
from src.memory.state import AgentTask

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── Mode registry ─────────────────────────────────────────────────────────────
#
# Each entry defines:
#   label          – human-readable display name (used in /mode keyboard)
#   allowed_agents – list of agent names permitted in this mode;
#                    None means ALL agents are permitted (full-orchestrator behaviour)
#   system_prefix  – optional context injected into the Gatekeeper system prompt
#
# The "all" mode is the default and preserves full backward-compatibility.
#
MODE_MAP: dict[str, dict] = {
    "all": {
        "label": "🎯 Mode All (Default)",
        "allowed_agents": None,
        "system_prefix": None,
    },
    "dev": {
        "label": "💻 Mode Developer",
        "allowed_agents": [
            "developer", "developer_inspector", "developer_qna",
            "code_fix", "sysinfo_agent", "log_viewer_agent",
            "responder",
        ],
        "system_prefix": (
            "User sedang dalam Mode Developer. "
            "Fokus pada pengembangan software, inspeksi kode, debugging, dan operasi sistem."
        ),
    },
    "writer": {
        "label": "✍️ Mode Writer",
        "allowed_agents": [
            "researcher", "content_creator", "technical_writer", "responder",
        ],
        "system_prefix": (
            "User sedang dalam Mode Writer. "
            "Fokus pada pembuatan konten, riset, penulisan, dan dokumentasi teknis."
        ),
    },
    "office": {
        "label": "📋 Mode Office",
        "allowed_agents": [
            "wbs_agent", "mandays_agent", "reminder_agent", "researcher", "responder",
        ],
        "system_prefix": (
            "User sedang dalam Mode Office. "
            "Fokus pada manajemen proyek, perencanaan WBS, estimasi mandays, pengingat, dan riset. "
            "Gunakan bahasa yang santai, gaul, dan akrab — sapa user sebagai 'boss', "
            "pakai kata-kata seperti 'nih', 'dong', 'sih', 'yuk', 'mantap', 'siap', 'gaskeun'. "
            "Tetap informatif tapi buat percakapan terasa ringan dan menyenangkan."
        ),
    },
    "media": {
        "label": "📄 Mode Media",
        "allowed_agents": [
            "pdf_summarizer", "quiz_agent", "tg_quiz_agent",
            "doc_agent", "analysis_diagram", "responder",
        ],
        "system_prefix": (
            "User sedang dalam Mode Media. "
            "Fokus pada pemrosesan PDF, pembuatan kuis, audit dokumen, dan pembuatan diagram."
        ),
    },
    "web": {
        "label": "🌐 Mode Web",
        "allowed_agents": [
            "web_automation", "responder",
        ],
        "system_prefix": (
            "User sedang dalam Mode Web. "
            "Fokus pada otomasi web dan penjelajahan browser."
        ),
    },
}

# ── Intent → agent-name mapping ───────────────────────────────────────────────
#
# To add a new agent:
#   1. Create src/agents/<your_agent>/agent.py
#   2. Register it in main_loop.py's _build_agents()
#   3. Map its intents here
#
INTENT_AGENT_MAP: dict[IntentCategory, str] = {
    IntentCategory.GENERAL_INQUIRY:   "responder",
    IntentCategory.PRODUCT_QUESTION:  "responder",
    IntentCategory.COMPLAINT:         "responder",
    IntentCategory.ORDER_STATUS:      "responder",
    IntentCategory.BILLING:           "responder",
    IntentCategory.UNKNOWN:           "responder",
    IntentCategory.TECHNICAL_SUPPORT: "responder",
    IntentCategory.IMAGE_QUERY:       "responder",
    IntentCategory.RESEARCH:          "researcher",       #← only intent with live Tavily access
    IntentCategory.CONTENT_CREATION:  "content_creator",
    IntentCategory.WBS_PLANNING:      "wbs_agent",
    IntentCategory.MANDAYS_PLANNING:  "mandays_agent",
    IntentCategory.CODE_DEVELOPMENT:   "developer",           #← clone / AI-edit / sandbox
    IntentCategory.CODE_INSPECTION:    "developer_inspector",  #← read-only inspect + root cause
    IntentCategory.CODE_UNDERSTANDING: "developer_qna",        #← Q/A tentang isi repo
    IntentCategory.CODE_FIX:           "code_fix",             #← pipeline: inspect → auto-fix
    IntentCategory.DOCUMENT_CREATION:  "technical_writer",     #← generate PDF/Word dari repo/topik
    IntentCategory.SYSTEM_INFO:       "sysinfo_agent",        #← CPU / RAM / storage host info
    IntentCategory.LOG_VIEWER:        "log_viewer_agent",     #← tampilkan log bot untuk debugging
    IntentCategory.QUIZ_GENERATION:   "quiz_agent",           #← konversi PDF → kuis HTML interaktif
    IntentCategory.TELEGRAM_QUIZ:     "tg_quiz_agent",        #← konversi PDF → kuis polling Telegram (sendPoll)
    IntentCategory.PDF_SUMMARIZATION: "pdf_summarizer",       #← ringkas / QnA isi dokumen PDF
    IntentCategory.WEB_AUTOMATION:    "web_automation",       #← autonomous browsing & web interaction
    IntentCategory.DOC_AUDIT:         "doc_agent",            #← analisis + Q&A + edit interaktif .docx
    IntentCategory.DIAGRAM_FROM_ANALYSIS: "analysis_diagram", #← buat diagram dari hasil analisa & QnA sesi aktif
    IntentCategory.REMINDER:          "reminder_agent",       #← set / list / cancel timed reminders
}


def get_allowed_intents(mode: str) -> list[str] | None:
    """Return the list of allowed IntentCategory *values* for a given mode.

    Returns ``None`` when the mode is ``"all"`` (no restriction).
    For any other mode, derives the allowed intents from the allowed_agents
    list by doing a reverse lookup on INTENT_AGENT_MAP.
    Always includes the responder-backed generic intents so the bot can
    still handle greetings / clarifications regardless of mode.
    """
    mode_cfg = MODE_MAP.get(mode, MODE_MAP["all"])
    allowed_agents = mode_cfg.get("allowed_agents")
    if allowed_agents is None:
        return None  # "all" mode – no restriction

    allowed_set = set(allowed_agents)
    return [
        intent.value
        for intent, agent_name in INTENT_AGENT_MAP.items()
        if agent_name in allowed_set
    ]


def get_mode_system_prefix(mode: str) -> str | None:
    """Return the system prompt prefix for the given mode, or None."""
    return MODE_MAP.get(mode, MODE_MAP["all"]).get("system_prefix")


class AgentRouter:
    """Resolves an AgentTask to the correct BaseAgent instance."""

    def __init__(self, agents: dict[str, "BaseAgent"]) -> None:
        self._agents = agents

    def resolve(self, task: AgentTask) -> "BaseAgent":
        """
        Pick the agent for task.intent.

        Falls back to 'responder' if the intent has no mapping
        or the mapped agent is not registered.
        """
        intent = IntentCategory(task.intent or IntentCategory.UNKNOWN.value)
        agent_name = INTENT_AGENT_MAP.get(intent, "responder")
        agent = self._agents.get(agent_name) or self._agents["responder"]

        logger.info(
            "Router: session=%s intent=%s → agent=%s",
            task.session_id,
            intent.value,
            agent.name,
        )
        return agent

    def is_agent_allowed(self, agent_name: str, mode: str) -> bool:
        """Return True if *agent_name* is permitted in the given *mode*.

        Always returns True for mode ``"all"`` (no restriction).
        """
        mode_cfg = MODE_MAP.get(mode, MODE_MAP["all"])
        allowed_agents = mode_cfg.get("allowed_agents")
        if allowed_agents is None:
            return True
        return agent_name in allowed_agents
