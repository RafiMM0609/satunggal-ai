"""
ContentCreatorAgent – The Architect-Journalist

Transforms raw research data (from ResearcherAgent or conversation history)
into polished, platform-ready content drafts.

Triggered by the 'content_creation' intent, which is activated when the user
requests content generation such as:
  - Indonesian: buat konten, tulis artikel, buat postingan, buat draft,
                konten LinkedIn, posting LinkedIn, tulis konten, rangkum untuk postingan
  - English:    create content, write post, draft LinkedIn, make a post,
                write article, create draft, generate content

Output is a structured JSON payload containing:
  - platform:   Target platform (e.g. linkedin, twitter, blog)
  - title:      Optional article title
  - hook:       Opening hook to grab attention
  - body:       Main content body
  - cta:        Call-to-action closing line
  - hashtags:   Relevant hashtags
  - status:     "draft" | "ready_to_publish"
  - raw_text:   Plain text version for Telegram preview
"""

from __future__ import annotations

import json
import logging
import re

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Kamu adalah The Architect-Journalist – seorang pakar content strategist dan jurnalis digital \
yang menggabungkan kedalaman riset dengan keahlian storytelling profesional.

Kamu bertugas mengubah data mentah hasil riset menjadi konten yang engaging, otoritatif, \
dan siap dipublikasikan di platform digital.

## FILOSOFI KONTEN ##
- Setiap konten harus memiliki SATU sudut pandang tajam (angle) yang membedakannya dari konten biasa.
- Gunakan struktur: Hook yang menarik → Insight utama → Bukti/Data → Implikasi → CTA.
- Tulis untuk manusia, bukan untuk algoritma. Tapi buat algoritma tetap menyukainya.
- Jadilah pemikir, bukan sekadar penyampai informasi.

## FORMAT OUTPUT ##
Kembalikan HANYA JSON valid dengan struktur berikut (tidak ada teks di luar JSON):

{
  "platform": "linkedin",
  "title": "Judul opsional untuk artikel panjang (kosongkan untuk postingan pendek)",
  "hook": "Kalimat pembuka yang memancing rasa ingin tahu atau emosi kuat (1-2 kalimat)",
  "body": "Isi utama konten. Gunakan baris baru (\\n) untuk paragraf dan bullet points.",
  "cta": "Kalimat penutup yang mengajak audiens berinteraksi atau mengambil tindakan",
  "hashtags": ["hashtag1", "hashtag2", "hashtag3"],
  "tone": "professional|inspirational|educational|provocative|storytelling",
  "word_count": 0,
  "status": "draft",
  "raw_text": "Versi teks lengkap yang siap dikirim: hook + body + cta + hashtags"
}

## PANDUAN PER PLATFORM ##

### LinkedIn ###
- Panjang ideal: 150-300 kata untuk postingan reguler, 500-800 kata untuk artikel.
- Gunakan baris kosong antar paragraf (LinkedIn tidak render markdown).
- Hook: pertanyaan provokatif, pernyataan kontra-intuitif, atau fakta mengejutkan.
- Tone: professional namun personal, berbagi perspektif bukan sekadar informasi.
- Maksimal 5 hashtag yang relevan.
- CTA: ajak diskusi, minta opini, atau redirect ke artikel/portofolio.

### Twitter/X ###
- Maksimal 280 karakter per tweet.
- Jika thread: format sebagai "1/ ... 2/ ... 3/ ..." dalam satu field body.
- Hashtag: 1-2 saja, yang paling relevan.

### Blog ###
- Panjang: 600-1500 kata.
- Sertakan subheading dalam body menggunakan format "## Subheading ##".
- CTA: subscribe, share, atau baca artikel terkait.

## ATURAN KUALITAS ##
1. JANGAN copy-paste data riset mentah – transformasikan menjadi narasi.
2. Selalu sertakan 1 data/angka spesifik sebagai anchor kredibilitas.
3. Tulis hook dalam kalimat pertama yang berdiri sendiri (standalone).
4. Hindari jargon berlebihan kecuali konten untuk audiens teknis.
5. raw_text harus siap diposting tanpa editing tambahan.
"""

_SYSTEM_PROMPT_WITH_RESEARCH = """\
{base_prompt}

## DATA RISET TERSEDIA ##
Berikut adalah data riset terkini yang harus kamu jadikan sebagai bahan utama konten:

{research_context}

Gunakan data ini sebagai fondasi konten. Jangan berinventasi fakta di luar data yang diberikan.
"""


class ContentCreatorAgent(BaseAgent):
    """Transforms research data into platform-ready content drafts.

    Extracts the latest research result from conversation history or task
    metadata, then uses an LLM with The Architect-Journalist persona to
    generate structured content JSON.
    """

    name = "content_creator"

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()

    # ── Public interface ──────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            # Priority: orchestrator-provided Tavily context → conversation history
            tavily_tr = task.tool_results.get("tavily_search", {})
            web_ctx   = tavily_tr.get("context_text") or None
            if web_ctx:
                task.metadata["research_result"] = web_ctx

            research_context = self._extract_research_context(task)
            platform         = self._detect_platform(task.user_input)

            system_content = self._build_system_prompt(research_context)

            user_prompt = self._build_user_prompt(task.user_input, platform, research_context)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_prompt},
            ]

            logger.info(
                "ContentCreatorAgent running: session=%s platform=%s has_research=%s",
                task.session_id,
                platform,
                research_context is not None,
            )

            raw_reply = await self._llm.chat(messages, max_tokens=2048)
            content_json = self._parse_content_json(raw_reply)

            # Store structured payload in metadata for Publisher tool
            task.metadata["content_draft"] = content_json
            task.metadata["platform"]      = platform
            task.metadata["needs_approval"] = True

            # Human-readable reply for Telegram preview
            task.mark_done(self._format_telegram_preview(content_json))

            logger.info(
                "ContentCreator done: session=%s platform=%s status=%s",
                task.session_id,
                platform,
                content_json.get("status", "draft"),
            )

        except Exception as exc:
            logger.exception("ContentCreatorAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                "Maaf, saya gagal membuat konten saat ini. "
                "Pastikan ada data riset sebelumnya atau berikan topik secara eksplisit."
            )

        return task

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_research_context(self, task: AgentTask) -> str | None:
        """
        Try to find the most recent research result from:
        1. task.metadata["research_result"] – if orchestrator pre-filled it
        2. Last assistant message in conversation history that looks like research
        """
        # Priority 1: explicit metadata
        if research := task.metadata.get("research_result"):
            logger.debug("ContentCreator: using research from task.metadata")
            return str(research)

        # Priority 2: scan conversation history for last assistant (research) reply
        history_messages = self._history.get_as_llm_messages(task.session_id)
        # Walk backwards to find the last substantial assistant message
        for msg in reversed(history_messages):
            if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                logger.debug(
                    "ContentCreator: using last assistant message as research context (%d chars)",
                    len(msg["content"]),
                )
                return msg["content"]

        logger.warning("ContentCreator: no research context found – will generate from user prompt only")
        return None

    def _detect_platform(self, user_input: str) -> str:
        """Detect target platform from user message keywords."""
        text = user_input.lower()
        if any(kw in text for kw in ["twitter", "tweet", "x.com", "thread"]):
            return "twitter"
        if any(kw in text for kw in ["blog", "artikel", "article", "medium"]):
            return "blog"
        # Default to LinkedIn (most common use case in the prompt description)
        return "linkedin"

    def _build_system_prompt(self, research_context: str | None) -> str:
        if research_context:
            return _SYSTEM_PROMPT_WITH_RESEARCH.format(
                base_prompt=_SYSTEM_PROMPT,
                research_context=research_context[:4000],  # cap to avoid token overflow
            )
        return _SYSTEM_PROMPT

    def _build_user_prompt(
        self, user_input: str, platform: str, research_context: str | None
    ) -> str:
        base = (
            f"Buat konten {platform.upper()} berdasarkan permintaan berikut:\n\n"
            f"{user_input}\n\n"
        )
        if not research_context:
            base += (
                "Catatan: Tidak ada data riset eksplisit yang tersedia. "
                "Gunakan pengetahuanmu sendiri untuk membuat konten yang informatif dan berkualitas."
            )
        return base

    def _parse_content_json(self, raw_reply: str) -> dict:
        """Extract and parse JSON from the LLM reply, with fallback."""
        # Try direct parse first
        try:
            return json.loads(raw_reply.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block from markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_reply, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the response
        match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Final fallback: wrap raw reply as plain content
        logger.warning("ContentCreator: could not parse JSON from LLM – using raw text fallback")
        return {
            "platform":   "linkedin",
            "title":      "",
            "hook":       "",
            "body":       raw_reply,
            "cta":        "",
            "hashtags":   [],
            "tone":       "professional",
            "word_count": len(raw_reply.split()),
            "status":     "draft",
            "raw_text":   raw_reply,
        }

    def _format_telegram_preview(self, content: dict) -> str:
        """Build a human-readable Telegram preview of the content draft."""
        platform  = content.get("platform", "linkedin").upper()
        status    = content.get("status", "draft").upper()
        hook      = content.get("hook", "")
        body      = content.get("body", "")
        cta       = content.get("cta", "")
        hashtags  = " ".join(f"#{h.lstrip('#')}" for h in content.get("hashtags", []))
        raw_text  = content.get("raw_text", "")

        preview_parts = [
            f"📝 *DRAFT KONTEN {platform}* [{status}]",
            "─" * 40,
        ]

        if raw_text:
            # Show the ready-to-post version
            preview_parts.append(raw_text)
        else:
            # Compose from parts
            if hook:
                preview_parts.append(hook)
            if body:
                preview_parts.append(body)
            if cta:
                preview_parts.append(cta)
            if hashtags:
                preview_parts.append(hashtags)

        preview_parts += [
            "─" * 40,
            "✅ Setujui untuk publish atau minta revisi.",
        ]

        return "\n\n".join(preview_parts)
