"""
LLM client for the GatekeeperAgent.

Sends user text to the active LLM provider and parses the intent JSON response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from config.settings import Settings
from src.agents.gatekeeper.schemas import IntentCategory
from src.agents.llm_client import LLMClient
from src.memory.key_store import (
    PROVIDER_OLLAMA,
    effective_ollama_model,
    effective_openrouter_model,
    get_active_provider,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an intent classifier for a customer service AI routing system.

Classify the user's PRIMARY intent into EXACTLY ONE of:
- general_inquiry
- product_question
- complaint
- order_status
- technical_support
- billing
- data_analysis      (user wants a WBS / Work Breakdown Structure for a project)
- mandays_planning   (user wants a mandays plan, effort estimation, or resource allocation without a full WBS)
- image_query
- research           (user explicitly requests deep research or investigation using live data)
- content_creation   (user wants to create, write, or draft content for a platform such as LinkedIn, Twitter, blog, etc.)
- code_development   (user wants to clone a repo, edit/fix code using AI CLI, or run code in a Docker sandbox)
- code_inspection    (user wants to INSPECT a repo, find bugs/issues/root causes, review code quality — read-only, NO code changes)
- code_understanding (user wants to UNDERSTAND or EXPLORE a repo: what APIs exist, what tech stack is used, what are the data models, dependencies, CI/CD setup, main flow, or what a specific function/class does)
- document_creation  (user wants to generate a technical document, PDF, or Word file — from a GitHub repo, topic, or data such as WBS/mandays output)
- system_info        (user asks about server/host resource status: CPU usage, RAM, memory, storage, disk space, hardware info of this machine)
- log_viewer         (user wants to see, inspect, or debug the bot's recent application logs)
- web_automation     (user wants the bot to autonomously browse a website, click buttons, fill forms, take screenshots, read page content, or interact with a web page)
- quiz_generation    (user wants to convert a PDF into an interactive HTML quiz or a set of MCQ questions from educational/study material)
- pdf_summarization  (user wants to summarize, ask questions about, or understand the content of a PDF document)
- doc_audit          (user wants to ask questions about, explore, or get details about a .docx document that was previously uploaded and analyzed in this session)
- reminder           (user wants to set a timed reminder/alarm, list their reminders, or cancel/delete a reminder)
- unknown

Pre-agent tools the orchestrator can execute before the specialist agent:
- "tavily_search" : performs live web search and injects context (use for research)

Rules:
1. Reply with a JSON object ONLY – no markdown, no explanation.
2. Schema: {"intent": "<category>", "confidence": <float 0.0–1.0>, "tools": [<tool_name>, ...], "needs_clarification": <bool>, "clarification_question": "<question or null>"}
3. "tools" must be a list; only include "tavily_search" when intent is "research".
4. Use "unknown" when the intent is genuinely unclear.
4a. Self-Correction – when you set intent to "unknown" OR confidence < 0.50, you MUST also set:
    - "needs_clarification": true
    - "clarification_question": a concise, helpful question in the SAME language the user used (Indonesian or English)
      that will help disambiguate exactly what they want.
    Example Indonesian clarification questions:
      - "Maksud Anda ingin membuat WBS, estimasi mandays, atau ada kebutuhan lain?" 
      - "Boleh saya tahu lebih detail? Apakah Anda ingin membuat dokumen, mencari informasi, atau ada permintaan teknis tertentu?"
      - "Apakah Anda ingin saya mencari info di internet, atau langsung menjawab berdasarkan pengetahuan yang saya miliki?"
    Example English clarification questions:
      - "Could you clarify what you'd like me to do — research a topic, generate a document, or something else?"
      - "Could you give me more detail about what you need?"
    When needs_clarification is true the orchestrator will return your clarification_question directly to the user
    WITHOUT running any tools or specialist agent. Do NOT invent a clarification_question when confidence ≥ 0.50.
    When needs_clarification is false, set clarification_question to null.
5. Use "mandays_planning" when the user asks about mandays, effort, person-days, or resource estimation specifically.
6. Use "data_analysis" when the user explicitly asks for a WBS or project breakdown structure.
7. Use "research" ONLY when the user uses explicit investigative/research keywords such as:
   - Indonesian: teliti, riset, selidiki, telusuri, cari tahu secara mendalam, analisis mendalam, kaji, pelajari secara mendalam
   - English: research, investigate, deep dive, thoroughly analyze, look into in depth, study in depth
   - Or phrases like: "teliti dengan baik", "lakukan riset tentang", "berikan analisis mendalam", "selidiki kondisi"
   A question that is simply asking for information (without explicit research keywords) is NOT "research".
8. Use "technical_support" for technical troubleshooting questions that do NOT use explicit research/investigation keywords.
9. Use "content_creation" when the user asks to create, write, draft, or generate content for social media or publishing, such as:
   - Indonesian: buat konten, tulis artikel, buat postingan, buat draft, konten LinkedIn, posting LinkedIn, tulis konten, rangkum untuk postingan
   - English: create content, write post, draft LinkedIn, make a post, write article, create draft, generate content
10. Use "code_development" when the user mentions cloning a repo/GitHub URL, fixing/editing code with AI CLI (Copilot, Claude), running code in Docker/sandbox, or listing cloned repos:
    - Indonesian: clone repo, kloning, perbaiki kode di repo, jalankan di sandbox, daftar repo, edit kode, tambah fitur ke repo
    - English: clone repo, fix code in repo, run in docker sandbox, list cloned repos, edit this repo, add feature to repo
11. Use "code_inspection" when the user wants to INSPECT, REVIEW, or DIAGNOSE code/repo without making changes:
    - Indonesian: inspeksi repo, periksa kode, cari bug, temukan masalah, audit kode, review kode, analisa bug, cari penyebab error,
      diagnosis masalah, lacak bug, inspektor, investigasi kode, apa yang salah di repo, kenapa error, selidiki bug
    - English: inspect repo, review code, find bug, audit code, analyze error, diagnose issue, trace bug, what is wrong in repo,
      why is it failing, code review, root cause analysis, check the code, look at the repo for issues
    - Key differentiator: user wants FINDINGS about PROBLEMS and RECOMMENDATIONS to fix them.
      If user says "perbaiki" / "fix" → code_development. If user says "cari bug" / "apa yang salah" / "kenapa error" → code_inspection.
12. Use "code_understanding" when the user wants to LEARN or EXPLORE what is INSIDE a repo — not to find bugs:
    - Indonesian: ada api apa, tech stack apa, model data apa, dependency apa, alur utama bagaimana, fungsi X itu apa,
      class apa saja, endpoint apa, teknologi apa yang dipakai, library apa, struktur repo, jelaskan repo ini,
      apa yang ada di repo, bagaimana cara kerja, kenalkan isi repo, explorasi repo, pelajari repo
    - English: what APIs are there, what tech stack, what data models, what dependencies, explain the main flow,
      what does function X do, what is class Y, list all endpoints, what technology is used, explain this repo,
      explore the repo, what is in this repo, how does it work, walk me through the codebase
    - Key differentiator: user wants to UNDERSTAND the content/structure of the repo, not diagnose a problem.
      If user asks "ada API apa?" / "tech stack apa?" / "jelaskan fungsi X" → code_understanding.
      If user asks "kenapa error?" / "ada bug apa?" → code_inspection.
13. Use "document_creation" when the user asks to generate, create, or compile a technical document in PDF or Word format:
    - Indonesian: buat dokumen, buat dokumen teknis, generate PDF, buat PDF, buat Word, buat laporan teknis, dokumentasikan, buatkan dokumentasi, buat doc
    - English: generate document, create technical doc, make a PDF, create Word document, document this repo, write technical documentation
    - Even if a repo URL is mentioned, if the primary intent is to produce a document (not to fix/edit code), use "document_creation"
    - Pre-agent tools: include "tavily_search" only if no repo URL is present and additional context from the web would help
14. Use "system_info" when the user asks about the current resource usage or hardware specs of THIS running server/machine:
    - Indonesian: berapa CPU, cek RAM, info memori, cek storage, disk penuh, berapa sisa disk, status server, resource server,
      info hardware server, penggunaan CPU, penggunaan memori, berapa banyak RAM, lihat storage, cek resource
    - English: check CPU, how much RAM, disk space, storage info, server resource, memory usage, CPU usage, hardware info server
    - Key differentiator: user wants LIVE metrics of THIS machine. If user asks about cloud billing or a remote server by URL → general_inquiry.
15. Use "log_viewer" when the user wants to see, read, or debug the bot's recent log output:
    - Indonesian: lihat log, tampilkan log, cek log, log bot, log error, log terbaru, debug log, tampilkan 20 log, log terakhir,
      lihat log terakhir, beri tahu log, tunjukkan log, log aplikasi, lihat catatan log
    - English: show log, view log, check log, bot log, recent log, last log lines, debug log, show me the logs, display logs,
      application log, log output, what does the log say, show last 10 lines of log
16. Use "web_automation" when the user wants the bot to open, navigate, interact with, or extract information from a website:
    - Indonesian: buka website, kunjungi URL, klik tombol di website, isi form, screenshot website, ambil isi halaman,
      buka link, navigasi ke halaman, login ke website, klik menu, scraping, cek halaman, buka browser, akses URL,
      pergi ke website, daftarkan akun di, isi formulir di, klik daftar, ambil konten dari, buka url ini
    - English: open website, visit URL, click button on website, fill form, take screenshot, get page content,
      navigate to page, login to website, click menu, scrape website, check page, open browser, access URL,
      go to website, register account at, fill out form at, click sign up, get content from
    - Key differentiator: user wants the bot to actually BROWSE and INTERACT with a live website, not just search for info.
      If user says "buka website X dan klik tombol Y" / "login ke situs Z" / "isi form di URL ini" → web_automation.
      If user just wants information found via web search → research.
17. When the input starts with "[Pesan user:" it means a PDF document was uploaded by the user.
    "[Pesan user: ...]" contains the caption that user typed when sending the PDF
    (or "(tidak ada pesan dari pengguna)" when no caption was provided).
    "[Preview dokumen: ...]" contains the beginning of the document's actual text content.
    Classify based on BOTH the caption intent AND the document preview:
    - Caption contains "kuis" / "soal" / "quiz" / "pertanyaan" / "latihan" OR preview looks like structured educational/study material (chapters, definitions, numbered items) → quiz_generation
    - Caption contains "ringkas" / "rangkum" / "summarize" / "apa isi" / "ceritakan" / "jelaskan" / "apa yang ada" / "kesimpulan" → pdf_summarization
    - Caption is "(tidak ada pesan dari pengguna)" or vague and document type is unclear → needs_clarification asking what they want done with the PDF
    Examples:
      Input has "[Pesan user: buat kuis dari ini]" → {"intent": "quiz_generation", "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null}
      Input has "[Pesan user: ringkas dokumen ini]" → {"intent": "pdf_summarization", "confidence": 0.93, "tools": [], "needs_clarification": false, "clarification_question": null}
      Input has "[Pesan user: (tidak ada pesan dari pengguna)]" → {"intent": "unknown", "confidence": 0.20, "tools": [], "needs_clarification": true, "clarification_question": "Mau diapakan dokumen ini? Misalnya: buat kuis interaktif, ringkasan, atau ada keperluan lain?"}

18. When "Riwayat percakapan terakhir" is present in the context, use it to detect follow-up commands:
    - If the most recent [Asisten] response clearly involved web browsing, clicking, form filling, login,
      screenshot, or navigation (i.e., the previous intent was web_automation), AND the current user message
      is a short follow-up that does NOT mention a completely different topic (e.g. "berikan screenshot",
      "klik tombol X", "scroll ke bawah", "ambil foto halaman", "tangkap layar", "lanjutkan",
      "isi form", "klik menu", "screenshot dong", "screenshoot", "foto halaman"), classify the current
      message as "web_automation" with high confidence.
    - Similarly, if the most recent [Asisten] response was a research/code task and the current message
      is clearly a follow-up to that task, keep the same intent classification.
    - If the most recent [Asisten] response contains a document analysis report (indicated by phrases such as
      "Laporan Analisis Dokumen", "Daftar Isi", "Ringkasan per Bab", or "Tip: Balas pesan ini untuk bertanya"),
      AND the current user message is a question or request about the document content (e.g. "jelaskan bab 3",
      "apa maksud X di bab 4", "detail tentang bagian ini", "cek konsistensi"), classify as "doc_audit".

19. Use "doc_audit" when the user asks follow-up questions about a .docx document that was previously analyzed:
    - Indonesian: jelaskan bab ini, detail tentang bagian X, apa yang dibahas di bab Y, cek konsistensi,
      bandingkan bab, apa maksud X, detail bab, ringkasan ulang bab, pertanyaan tentang dokumen
    - English: explain chapter X, what does section Y say, detail about part Z, check consistency,
      compare chapters, what is discussed in chapter N, questions about the document
    - This intent is only valid when there is a previously analyzed document in the session.

20. Use "reminder" when the user wants to set, view, or cancel a timed reminder/alarm:
    - Indonesian: ingatkan saya, set reminder, buat pengingat, jadwalkan pengingat, daftar reminder,
      lihat reminder, tampilkan reminder, hapus reminder, batalkan reminder, cancel reminder,
      remind me, alarm, pengingat, set alarm, atur pengingat
    - English: remind me, set a reminder, create reminder, schedule reminder, list reminders,
      show reminders, delete reminder, cancel reminder, set alarm
    - Examples: "ingatkan saya untuk checkin jam 07:59", "remind me to take medicine tomorrow at 8am",
      "lihat daftar reminderku", "hapus reminder #3"

Example responses:
  {"intent": "data_analysis",      "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "mandays_planning",   "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "research",           "confidence": 0.91, "tools": ["tavily_search"], "needs_clarification": false, "clarification_question": null}
  {"intent": "code_development",   "confidence": 0.96, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "code_inspection",    "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "code_understanding", "confidence": 0.94, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "document_creation",  "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "system_info",        "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "log_viewer",         "confidence": 0.98, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "web_automation",     "confidence": 0.96, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "general_inquiry",    "confidence": 0.88, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "unknown",            "confidence": 0.30, "tools": [], "needs_clarification": true,  "clarification_question": "Boleh saya tahu lebih detail tentang apa yang ingin Anda lakukan? Apakah Anda ingin membuat dokumen, mencari informasi, atau ada kebutuhan teknis lainnya?"}
  {"intent": "quiz_generation",    "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "pdf_summarization",  "confidence": 0.93, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "doc_audit",          "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null}
  {"intent": "reminder",           "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null}
"""


@dataclass(frozen=True, slots=True)
class LLMIntentResponse:
    intent:                 IntentCategory
    confidence:             float
    model_used:             str
    tools:                  tuple[str, ...] = ()
    needs_clarification:    bool            = False
    clarification_question: str | None      = None


class GatekeeperLLMClient:
    """Async LLM client for intent classification, routed through the active provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = LLMClient(settings)

    async def classify_intent(
        self,
        user_text: str,
        history: "list[dict] | None" = None,
    ) -> LLMIntentResponse:
        # Inject recent conversation history into system prompt so the LLM can
        # correctly classify follow-up commands (e.g. "berikan screenshot" after
        # a previous web_automation turn).
        if history:
            lines = []
            for msg in history:
                role = "Pengguna" if msg.get("role") == "user" else "Asisten"
                # Truncate to 300 chars to keep the prompt within max_tokens budget
                # while still providing enough context for intent disambiguation.
                content = msg.get("content", "")[:300]
                lines.append(f"[{role}]: {content}")
            system_content = (
                _SYSTEM_PROMPT
                + "\n\nRiwayat percakapan terakhir:\n"
                + "\n".join(lines)
            )
        else:
            system_content = _SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_text},
        ]
        provider = get_active_provider()
        if provider == PROVIDER_OLLAMA:
            model_used = effective_ollama_model(self._settings.ollama_model)
        else:
            model_used = effective_openrouter_model(self._settings.openrouter_model)
        logger.debug(
            "GatekeeperLLMClient.classify_intent → provider=%s model=%s",
            provider, model_used,
        )
        raw = await self._llm.chat(messages, max_tokens=128, json_mode=True)
        return self._parse(raw, model_used=model_used)

    async def aclose(self) -> None:
        await self._llm.aclose()

    async def __aenter__(self) -> "GatekeeperLLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── private ───────────────────────────────────────────────────────────────

    def _parse(self, raw_content: str, *, model_used: str = "") -> LLMIntentResponse:
        if not raw_content:
            logger.warning("LLM returned empty content for intent classification")
            return LLMIntentResponse(
                intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                model_used=model_used,
            )

        try:
            parsed = json.loads(raw_content.strip())
            intent_str = parsed.get("intent", "unknown")
            confidence = float(parsed.get("confidence", 0.5))
            intent = IntentCategory(intent_str)
            raw_tools = parsed.get("tools", [])
            tools = tuple(t for t in raw_tools if isinstance(t, str))
            needs_clarification    = bool(parsed.get("needs_clarification", False))
            clarification_question = parsed.get("clarification_question") or None
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse LLM response (%s): %r", exc, raw_content)
            intent                 = IntentCategory.UNKNOWN
            confidence             = 0.0
            tools                  = ()
            needs_clarification    = True
            clarification_question = None

        return LLMIntentResponse(
            intent=intent,
            confidence=confidence,
            model_used=model_used,
            tools=tools,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
        )


# Keep backward-compatible alias so any code still importing OpenRouterClient
# continues to work without modification.
OpenRouterClient = GatekeeperLLMClient
