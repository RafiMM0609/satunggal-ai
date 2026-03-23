"""
LLM client for the GatekeeperAgent.

Sends user text to OpenRouter and parses the intent JSON response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from config.settings import Settings
from src.agents.gatekeeper.schemas import IntentCategory

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
"""


@dataclass(frozen=True, slots=True)
class LLMIntentResponse:
    intent:                 IntentCategory
    confidence:             float
    model_used:             str
    tools:                  tuple[str, ...] = ()
    needs_clarification:    bool            = False
    clarification_question: str | None      = None


class OpenRouterClient:
    """Thin async HTTP wrapper around the OpenRouter chat-completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout,
            headers=settings.openrouter_headers,
        )

    async def classify_intent(self, user_text: str) -> LLMIntentResponse:
        payload = {
            "model": self._settings.openrouter_model,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
        }
        logger.debug("OpenRouter classify → model=%s", self._settings.openrouter_model)
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return self._parse(response.json())

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── private ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict) -> LLMIntentResponse:
        model_used = data.get("model", self._settings.openrouter_model)
        content = data["choices"][0]["message"].get("content")

        if content is None:
            logger.warning("LLM returned null content: %r", data)
            return LLMIntentResponse(
                intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                model_used=model_used,
            )

        raw_content = content.strip()

        try:
            parsed = json.loads(raw_content)
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
