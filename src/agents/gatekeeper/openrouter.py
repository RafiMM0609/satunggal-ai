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
- document_creation  (user wants to generate a technical document, PDF, or Word file — from a GitHub repo, topic, or data such as WBS/mandays output)
- system_info        (user asks about server/host resource status: CPU usage, RAM, memory, storage, disk space, hardware info of this machine)
- unknown

Pre-agent tools the orchestrator can execute before the specialist agent:
- "tavily_search" : performs live web search and injects context (use for research)

Rules:
1. Reply with a JSON object ONLY – no markdown, no explanation.
2. Schema: {"intent": "<category>", "confidence": <float 0.0–1.0>, "tools": [<tool_name>, ...]}
3. "tools" must be a list; only include "tavily_search" when intent is "research".
4. Use "unknown" when the intent is genuinely unclear.
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
    - Key differentiator: user wants FINDINGS and RECOMMENDATIONS (not actual code changes).
      If user says "perbaiki" / "fix" → code_development. If user says "cari tahu" / "periksa" / "apa yang salah" → code_inspection.
12. Use "document_creation" when the user asks to generate, create, or compile a technical document in PDF or Word format:
    - Indonesian: buat dokumen, buat dokumen teknis, generate PDF, buat PDF, buat Word, buat laporan teknis, dokumentasikan, buatkan dokumentasi, buat doc
    - English: generate document, create technical doc, make a PDF, create Word document, document this repo, write technical documentation
    - Even if a repo URL is mentioned, if the primary intent is to produce a document (not to fix/edit code), use "document_creation"
    - Pre-agent tools: include "tavily_search" only if no repo URL is present and additional context from the web would help
13. Use "system_info" when the user asks about the current resource usage or hardware specs of THIS running server/machine:
    - Indonesian: berapa CPU, cek RAM, info memori, cek storage, disk penuh, berapa sisa disk, status server, resource server,
      info hardware server, penggunaan CPU, penggunaan memori, berapa banyak RAM, lihat storage, cek resource
    - English: check CPU, how much RAM, disk space, storage info, server resource, memory usage, CPU usage, hardware info server
    - Key differentiator: user wants LIVE metrics of THIS machine. If user asks about cloud billing or a remote server by URL → general_inquiry.

Example responses:
  {"intent": "data_analysis",    "confidence": 0.97, "tools": []}
  {"intent": "mandays_planning", "confidence": 0.95, "tools": []}
  {"intent": "research",         "confidence": 0.91, "tools": ["tavily_search"]}
  {"intent": "code_development",  "confidence": 0.96, "tools": []}
  {"intent": "code_inspection",   "confidence": 0.95, "tools": []}
  {"intent": "document_creation", "confidence": 0.95, "tools": []}
  {"intent": "system_info",       "confidence": 0.97, "tools": []}
  {"intent": "general_inquiry",   "confidence": 0.88, "tools": []}
"""


@dataclass(frozen=True, slots=True)
class LLMIntentResponse:
    intent:     IntentCategory
    confidence: float
    model_used: str
    tools:      tuple[str, ...] = ()


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
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse LLM response (%s): %r", exc, raw_content)
            intent     = IntentCategory.UNKNOWN
            confidence = 0.0
            tools      = ()

        return LLMIntentResponse(
            intent=intent,
            confidence=confidence,
            model_used=model_used,
            tools=tools,
        )
