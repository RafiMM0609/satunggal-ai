"""
ResponderAgent – general-purpose conversational agent.

Handles:  general_inquiry, product_question, complaint, order_status,
          technical_support, billing, image_query, unknown
Uses:     Conversation history + LLM to produce a contextual reply.

Note: This agent does NOT use Tavily web search. Live web search is
exclusively available to ResearcherAgent (triggered by the 'research' intent).
"""

from __future__ import annotations

import logging
import re

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask
from src.tools.telegram_formatter import sanitize_for_telegram

logger = logging.getLogger(__name__)

# ── Slang / Gen Z marker sets ─────────────────────────────────────────────────
_GENZ_FIRST_PERSON  = re.compile(r'\b(gue|gw|aku|w)\b', re.IGNORECASE)
_GENZ_SECOND_PERSON = re.compile(r'\b(lo|lu|kamu|elo)\b', re.IGNORECASE)
_GENZ_VOCAB         = re.compile(
    r'\b(cuy|bro|sis|gas|gaskeun|sikat|mantap|gercep|anjir|wkwk|hehe|btw|anw|'
    r'blunder|red\s*flag|spill|mager|kepo|kepoin|intip|teropong|cooking|'
    r'sat[- ]?set|ngl|fr|lol|omg|btw|fyi)\b',
    re.IGNORECASE,
)
_PANIC_MARKERS      = re.compile(r'\b(gawat|darurat|urgent|asap|crash)\b', re.IGNORECASE)
_PANIC_CAPS         = re.compile(r'\b[A-Z]{5,}\b')  # 5+ caps = likely screaming, not an acronym

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FORMAL = """\
Kamu adalah asisten AI yang ramah, profesional, dan membantu.
Jawab pertanyaan pengguna secara jelas dan ringkas.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
Jika kamu tidak tahu jawabannya, katakan dengan jujur.
"""

_SYSTEM_PROMPT_OFFICE = """\
Kamu adalah asisten AI yang gaul, santai, dan asik diajak ngobrol — khusus buat urusan kantor dan kerjaan.
Jawab pertanyaan pengguna secara jelas tapi tetap santai dan informal.
Sapa user sebagai "boss" sesekali biar lebih akrab.
Pakai kata-kata gaul Indonesia yang wajar: "nih", "dong", "sih", "yuk", "mantap", "siap", "gaskeun", "okee", dll.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris), tapi tetap dengan nada santai.
Jika kamu tidak tahu jawabannya, bilang aja jujur dengan cara yang santai.
"""

_SYSTEM_PROMPT_GENZ = """\
Lo adalah asisten AI yang gaul, adaptive, dan asik — kayak temen nongkrong yang juga jago teknis.
Balas pake gaya bahasa yang sama kayak user: kalau dia pake "Gue/Lo", lo pake "Gue/Lo" juga.
Boleh pakai slang yang wajar: "cuy", "bro", "gas", "sat-set", "mantap", "nih", "dong", "gercep", dll.
Jawab substansinya dengan benar dan lengkap — gaya santai bukan alasan buat jawaban yang asal.
Kalau lo nggak tau, bilang jujur dengan cara yang asik, jangan drama.
Gunakan bahasa Indonesia santai (atau Inggris kalau user pakai Inggris).
"""

_SYSTEM_PROMPT_GENZ_PANIC = """\
Lo adalah asisten AI yang gaul, adaptive, dan asik — kayak temen nongkrong yang juga jago teknis.
User lagi panik atau butuh bantuan cepat. Responmu harus:
1. Menenangkan tapi langsung ke poin — jangan terlalu banyak basa-basi.
2. Tunjukin bahwa lo langsung handle situasi ini: "Tenang cuy, gue bantuin beresin ini sekarang, sat-set!"
3. Gunakan gaya bahasa yang sama kayak user: kalau dia pake "Gue/Lo", lo pake "Gue/Lo" juga.
Jawab substansinya dengan benar dan lengkap.
"""

# Keep old name as alias so existing imports still work
_SYSTEM_PROMPT = _SYSTEM_PROMPT_FORMAL


def _detect_vibe(text: str) -> str:
    """Detect the communication style / vibe of the user's message.

    Returns one of:
        "genz_panic" – slang with panic/urgency markers
        "genz"       – clear Gen Z / slang vocabulary
        "office"     – mode handled separately by caller
        "formal"     – default professional style
    """
    has_genz = bool(
        _GENZ_FIRST_PERSON.search(text)
        or _GENZ_SECOND_PERSON.search(text)
        or _GENZ_VOCAB.search(text)
    )
    has_panic = bool(_PANIC_MARKERS.search(text) or _PANIC_CAPS.search(text))

    if has_genz and has_panic:
        return "genz_panic"
    if has_genz:
        return "genz"
    return "formal"


class ResponderAgent(BaseAgent):
    """Generates a conversational reply using history + LLM.

    Does NOT use Tavily web search. For live-search-backed answers,
    the orchestrator should route to ResearcherAgent via the 'research' intent.

    The system prompt is chosen dynamically based on the detected communication
    style (vibe) of the user's message, enabling "language mirroring".
    """

    name = "responder"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)

            # Pick system prompt: office mode always uses office prompt;
            # other modes use dynamic vibe detection for language mirroring.
            if task.current_mode == "office":
                system_prompt = _SYSTEM_PROMPT_OFFICE
                vibe = "office"
            else:
                vibe = _detect_vibe(task.user_input)
                if vibe == "genz_panic":
                    system_prompt = _SYSTEM_PROMPT_GENZ_PANIC
                elif vibe == "genz":
                    system_prompt = _SYSTEM_PROMPT_GENZ
                else:
                    system_prompt = _SYSTEM_PROMPT_FORMAL

            logger.debug(
                "ResponderAgent vibe=%s mode=%s session=%s",
                vibe, task.current_mode, task.session_id,
            )

            # ── Build message list ─────────────────────────────────────────
            messages = [{"role": "system", "content": system_prompt}]
            # Include at most last 10 messages for context
            messages.extend(history_messages[-10:])
            # Ensure the latest user message is in the list
            if not history_messages or history_messages[-1]["content"] != task.user_input:
                messages.append({"role": "user", "content": task.user_input})

            reply = await self._llm.chat(messages)
            task.mark_done(sanitize_for_telegram(reply))
            logger.info("Responder done for session=%s vibe=%s", task.session_id, vibe)
        except Exception as exc:
            logger.exception("ResponderAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan. Silakan coba lagi."

        return task
