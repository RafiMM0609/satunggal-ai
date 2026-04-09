"""
LLM client for the GatekeeperAgent.

Sends user text to the active LLM provider and parses the intent JSON response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

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

# ── Prompt building blocks ─────────────────────────────────────────────────────
#
# The gatekeeper prompt is assembled dynamically at call time.
# When a mode is active, only the intent descriptions and rules that are
# relevant to the allowed agents are included, shrinking the LLM's option
# space and significantly reducing intent misclassification.
#
# Structure:
#   _PREAMBLE            – system role + "Classify into EXACTLY ONE of:" header
#   _INTENT_DESCRIPTIONS – per-intent description lines (ordered dict)
#   _RESPONDER_INTENTS   – intents always present regardless of mode (fallback to responder)
#   _PRE_AGENT_TOOLS     – tools section (unchanged)
#   _SCHEMA_RULES        – rules 1–4a (always included)
#   _INTENT_RULES        – list of (rule_text, applicable_intents | None)
#                          None = always include; frozenset = include when any intent overlaps
#   _EXAMPLES            – example JSON responses (filtered per mode)
#
# _build_system_prompt(allowed_intents) assembles the final prompt.
# When allowed_intents is None (mode="all") it returns the full prompt.

_PREAMBLE = "You are an intent classifier for a customer service AI routing system.\n\nClassify the user's PRIMARY intent into EXACTLY ONE of:\n"

# Ordered so the final prompt reads logically (generic → specialised).
_INTENT_DESCRIPTIONS: dict[str, str] = {
    "general_inquiry":        "- general_inquiry",
    "product_question":       "- product_question",
    "complaint":              "- complaint",
    "order_status":           "- order_status",
    "technical_support":      "- technical_support",
    "billing":                "- billing",
    "wbs_planning":           '- wbs_planning        (user wants a WBS / Work Breakdown Structure for a project)',
    "mandays_planning":       '- mandays_planning   (user wants a mandays plan, effort estimation, or resource allocation without a full WBS)',
    "image_query":            "- image_query",
    "research":               "- research           (user explicitly requests deep research or investigation using live data)",
    "content_creation":       "- content_creation   (user wants to create, write, or draft content for a platform such as LinkedIn, Twitter, blog, etc.)",
    "code_development":       "- code_development   (user wants to clone a repo, edit/fix code using AI CLI, or run code in a Docker sandbox)",
    "code_inspection":        "- code_inspection    (user wants to INSPECT a repo, find bugs/issues/root causes, review code quality \u2014 read-only, NO code changes)",
    "code_understanding":     "- code_understanding (user wants to UNDERSTAND or EXPLORE a repo: what APIs exist, what tech stack is used, what are the data models, dependencies, CI/CD setup, main flow, or what a specific function/class does)",
    "code_fix":               "- code_fix           (user wants to AUTO-DETECT problems AND AUTO-FIX them in one go \u2014 combined inspect+edit pipeline)",
    "document_creation":      "- document_creation  (user wants to generate a technical document, PDF, or Word file \u2014 from a GitHub repo, topic, or data such as WBS/mandays output)",
    "system_info":            "- system_info        (user asks about server/host resource status: CPU usage, RAM, memory, storage, disk space, hardware info of this machine)",
    "log_viewer":             "- log_viewer         (user wants to see, inspect, or debug the bot's recent application logs)",
    "web_automation":         '- web_automation     (user wants the bot to autonomously browse a website, click buttons, fill forms, take screenshots, read page content, or interact with a web page)',
    "quiz_generation":        "- quiz_generation    (user wants to convert a PDF into an interactive HTML quiz or a set of MCQ questions from educational/study material)",
    "telegram_quiz":          '- telegram_quiz      (user explicitly wants quiz questions sent as interactive Telegram polls \u2014 keywords: "kuis telegram", "kirim polling", "kirim soal poll", "sendPoll", "kuis via telegram", "polling kuis", "quiz telegram")',
    "telegram_quiz_bank":     '- telegram_quiz_bank (user uploads a PDF that ALREADY CONTAINS a pre-existing collection of exam/quiz questions and wants them EXTRACTED (not generated) as Telegram polls \u2014 keywords: "bank soal", "kumpulan soal", "soal ujian", "soal latihan", "ekstrak soal", "import soal", "ambil soal dari pdf", "soal sudah ada", "pdf soal", "bank kuis"; KEY: user wants to EXTRACT existing questions, NOT generate new ones from study material)',
    "pdf_summarization":      "- pdf_summarization  (user wants to summarize, ask questions about, or understand the content of a PDF document)",
    "doc_audit":              "- doc_audit          (user wants to ask questions about, explore, or get details about a .docx document that was previously uploaded and analyzed in this session)",
    "diagram_from_analysis":  '- diagram_from_analysis (user wants to create a flow diagram or visual summary from the analysis and Q&A done in the current active document session \u2014 keywords: "buat diagram", "flow diagram", "gambarkan alur", "buat flowchart", "visualisasikan", "buat diagram dari diskusi", "generate diagram", "buat diagram dari analisa", "diagram dari hasil qna", "create diagram", "draw diagram", "diagram from analysis")',
    "reminder":               "- reminder           (user wants to set a timed reminder/alarm, list their reminders, or cancel/delete a reminder)",
    "unknown":                "- unknown",
}

# These intents route to the "responder" agent and are always included in every
# mode so the bot can still handle greetings / clarifications.
_RESPONDER_INTENTS: frozenset[str] = frozenset({
    "general_inquiry", "product_question", "complaint", "order_status",
    "billing", "unknown", "technical_support", "image_query",
})

_PRE_AGENT_TOOLS = """\

Pre-agent tools the orchestrator can execute before the specialist agent:
- "tavily_search" : performs live web search and injects context (use for research)
"""

_SCHEMA_RULES = """\
Rules:
1. Reply with a JSON object ONLY \u2013 no markdown, no explanation.
2. Schema: {"intent": "<category>", "confidence": <float 0.0\u20131.0>, "tools": [<tool_name>, ...], "needs_clarification": <bool>, "clarification_question": "<question or null>", "sub_intent": "<sub-intent or null>"}
3. "tools" must be a list; only include "tavily_search" when intent is "research".
4. Use "unknown" when the intent is genuinely unclear.
4a. Self-Correction \u2013 when you set intent to "unknown" OR confidence < 0.50, you MUST also set:
    - "needs_clarification": true
    - "clarification_question": a concise, helpful question in the SAME language the user used (Indonesian or English)
      that will help disambiguate exactly what they want.
    Example Indonesian clarification questions:
      - "Maksud Anda ingin membuat WBS, estimasi mandays, atau ada kebutuhan lain?"
      - "Boleh saya tahu lebih detail? Apakah Anda ingin membuat dokumen, mencari informasi, atau ada permintaan teknis tertentu?"
      - "Apakah Anda ingin saya mencari info di internet, atau langsung menjawab berdasarkan pengetahuan yang saya miliki?"
    Example English clarification questions:
      - "Could you clarify what you'd like me to do \u2014 research a topic, generate a document, or something else?"
      - "Could you give me more detail about what you need?"
    When needs_clarification is true the orchestrator will return your clarification_question directly to the user
    WITHOUT running any tools or specialist agent. Do NOT invent a clarification_question when confidence \u2265 0.50.
    When needs_clarification is false, set clarification_question to null.
"""

# Each entry: (rule_text, frozenset_of_relevant_intents | None)
# None  = always include (context-awareness, schema rules already in _SCHEMA_RULES).
# frozenset = include only when at least one intent in the set is in active_intents.
_INTENT_RULES: list[tuple[str, "frozenset[str] | None"]] = [
    (
        '5. Use "mandays_planning" when the user asks about mandays, effort, person-days, or resource estimation specifically.',
        frozenset({"mandays_planning"}),
    ),
    (
        '6. Use "wbs_planning" when the user explicitly asks for a WBS or project breakdown structure.',
        frozenset({"wbs_planning"}),
    ),
    (
        '7. Use "research" ONLY when the user uses explicit investigative/research keywords such as:\n'
        '   - Indonesian: teliti, riset, selidiki, telusuri, cari tahu secara mendalam, analisis mendalam, kaji, pelajari secara mendalam\n'
        '   - English: research, investigate, deep dive, thoroughly analyze, look into in depth, study in depth\n'
        '   - Or phrases like: "teliti dengan baik", "lakukan riset tentang", "berikan analisis mendalam", "selidiki kondisi"\n'
        '   A question that is simply asking for information (without explicit research keywords) is NOT "research".',
        frozenset({"research"}),
    ),
    (
        '8. Use "technical_support" for technical troubleshooting questions that do NOT use explicit research/investigation keywords.',
        frozenset({"technical_support"}),
    ),
    (
        '9. Use "content_creation" when the user asks to create content, write content, draft content, or generate content for social media or publishing, such as:\n'
        '   - Indonesian: buat konten, tulis artikel, buat postingan, buat draft, konten LinkedIn, posting LinkedIn, tulis konten, rangkum untuk postingan\n'
        '   - English: create content, write post, draft LinkedIn, make a post, write article, create draft, generate content',
        frozenset({"content_creation"}),
    ),
    (
        '10. Use "code_development" when the user mentions cloning a repo/GitHub URL, fixing/editing code with AI CLI (Copilot, Claude), running code in Docker/sandbox, or listing cloned repos:\n'
        '    - Indonesian: clone repo, kloning, perbaiki kode di repo, jalankan di sandbox, daftar repo, edit kode, tambah fitur ke repo\n'
        '    - English: clone repo, fix code in repo, run in docker sandbox, list cloned repos, edit this repo, add feature to repo\n'
        '    - Positive examples (MUST be code_development):\n'
        '      "perbaiki bug di file main.py repo github.com/foo/bar"\n'
        '      "tambahkan fitur login ke repo github.com/foo/bar"\n'
        '      "clone repo ini dan edit config-nya"\n'
        '      "fix the null pointer error in this repo"',
        frozenset({"code_development"}),
    ),
    (
        '11. Use "code_inspection" when the user wants to INSPECT, REVIEW, or DIAGNOSE code/repo without making changes:\n'
        '    - Indonesian: inspeksi repo, periksa kode, cari bug, temukan masalah, audit kode, review kode, analisa bug, cari penyebab error,\n'
        '      diagnosis masalah, lacak bug, inspektor, investigasi kode, apa yang salah di repo, kenapa error, selidiki bug\n'
        '    - English: inspect repo, review code, find bug, audit code, analyze error, diagnose issue, trace bug, what is wrong in repo,\n'
        '      why is it failing, code review, root cause analysis, check the code, look at the repo for issues\n'
        '    - Key differentiator: user wants FINDINGS about PROBLEMS and RECOMMENDATIONS to fix them.\n'
        '      If user says "perbaiki" / "fix" \u2192 code_development. If user says "cari bug" / "apa yang salah" / "kenapa error" \u2192 code_inspection.\n'
        '    - Positive examples (MUST be code_inspection):\n'
        '      "cari penyebab error di repo github.com/foo/bar"\n'
        '      "apa yang salah di kode ini? repo: github.com/foo/bar"\n'
        '      "temukan bug di repo ini dan berikan rekomendasinya"\n'
        '      "kenapa aplikasi saya crash? tolong inspeksi repo ini"',
        frozenset({"code_inspection"}),
    ),
    (
        '12. Use "code_understanding" when the user wants to LEARN or EXPLORE what is INSIDE a repo \u2014 not to find bugs:\n'
        '    - Indonesian: ada api apa, tech stack apa, model data apa, dependency apa, alur utama bagaimana, fungsi X itu apa,\n'
        '      class apa saja, endpoint apa, teknologi apa yang dipakai, library apa, struktur repo, jelaskan repo ini,\n'
        '      apa yang ada di repo, bagaimana cara kerja, kenalkan isi repo, explorasi repo, pelajari repo\n'
        '    - English: what APIs are there, what tech stack, what data models, what dependencies, explain the main flow,\n'
        '      what does function X do, what is class Y, list all endpoints, what technology is used, explain this repo,\n'
        '      explore the repo, what is in this repo, how does it work, walk me through the codebase\n'
        '    - Key differentiator: user wants to UNDERSTAND the content/structure of the repo, not diagnose a problem.\n'
        '      If user asks "ada API apa?" / "tech stack apa?" / "jelaskan fungsi X" \u2192 code_understanding.\n'
        '      If user asks "kenapa error?" / "ada bug apa?" \u2192 code_inspection.\n'
        '    - Positive examples (MUST be code_understanding):\n'
        '      "ada API apa saja di repo github.com/foo/bar?"\n'
        '      "tech stack apa yang dipakai di repo ini?"\n'
        '      "jelaskan alur utama aplikasi di repo github.com/foo/bar"\n'
        '      "fungsi authenticate() itu ngapain di repo ini?"\n'
        '    - When intent is code_understanding, ALSO set "sub_intent" to the most specific value:\n'
        '      "api_endpoints"   \u2192 user asks about HTTP routes, REST API, endpoint list\n'
        '      "tech_stack"      \u2192 user asks about technologies, frameworks, languages, libraries\n'
        '      "data_models"     \u2192 user asks about database schema, ORM models, data structures\n'
        '      "dependencies"    \u2192 user asks about packages, requirements, modules\n'
        '      "ci_cd"           \u2192 user asks about deployment, Docker, CI/CD pipeline\n'
        '      "security"        \u2192 user asks about auth, authorization, JWT, security\n'
        '      "main_flow"       \u2192 user asks about architecture, business logic, system flow\n'
        '      "specific_symbol" \u2192 user asks about a specific file, function, class, or method\n'
        '      "full_inspection" \u2192 general Q&A that does not fit the above categories\n'
        '      Set "sub_intent": null for all other intents.',
        frozenset({"code_understanding"}),
    ),
    (
        '12a. CRITICAL \u2013 Developer intent disambiguation (read before classifying ANY repo-related message):\n'
        '    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n'
        '    \u2502 Intent           \u2502 Keyword signals                                 \u2502\n'
        '    \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n'
        '    \u2502 code_development \u2502 perbaiki, tambah, edit, ubah, update, fix,      \u2502\n'
        '    \u2502                  \u2502 implement, deploy, push, kode ulang              \u2502\n'
        '    \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n'
        '    \u2502 code_inspection  \u2502 cari bug, kenapa error, apa yang salah, temukan \u2502\n'
        '    \u2502                  \u2502 masalah, root cause, diagnosa, inspeksi, audit   \u2502\n'
        '    \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n'
        '    \u2502 code_understanding\u2502 ada apa, apa itu, jelaskan, tech stack, API apa,\u2502\n'
        '    \u2502                  \u2502 bagaimana cara kerja, model data, dependency     \u2502\n'
        '    \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n'
        '    \u2502 code_fix         \u2502 temukan DAN perbaiki, cari lalu fix, auto-fix,  \u2502\n'
        '    \u2502                  \u2502 otomatis perbaiki semua bug, diagnosa lalu edit  \u2502\n'
        '    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n'
        '    Negative examples to avoid misclassification:\n'
        '    \u2717 "ada apa di repo ini?"            \u2192 NEVER code_inspection \u2192 ALWAYS code_understanding\n'
        '    \u2717 "kenapa error?"                   \u2192 NEVER code_development \u2192 ALWAYS code_inspection\n'
        '    \u2717 "perbaiki error ini"              \u2192 NEVER code_inspection \u2192 ALWAYS code_development\n'
        '    \u2717 "apakah kode ini sudah bagus?"    \u2192 NEVER code_inspection \u2192 ALWAYS code_understanding\n'
        '    \u2717 "temukan dan fix semua bug"       \u2192 NEVER code_inspection alone \u2192 ALWAYS code_fix',
        frozenset({"code_development", "code_inspection", "code_understanding", "code_fix"}),
    ),
    (
        '14. Use "code_fix" when the user explicitly wants the system to BOTH identify problems AND automatically fix them:\n'
        '    - Indonesian: temukan dan perbaiki, cari dan fix, auto-fix semua bug, otomatis perbaiki masalah,\n'
        '      diagnosa lalu perbaiki, detect dan fix, cari masalah langsung perbaiki juga\n'
        '    - English: find and fix, detect and fix, auto-fix all bugs, automatically repair issues,\n'
        '      diagnose then fix, scan and fix, find problems and fix them automatically\n'
        '    - Key differentiator: user explicitly combines FINDING (inspect) + FIXING (develop) in one request.\n'
        '      If user asks only to "cari bug" \u2192 code_inspection.\n'
        '      If user asks only to "perbaiki bug" \u2192 code_development.\n'
        '      If user asks to "cari dan langsung perbaiki" \u2192 code_fix.',
        frozenset({"code_fix"}),
    ),
    (
        '15. Use "document_creation" when the user asks to generate, create, or compile a technical document in PDF or Word format:\n'
        '    - Indonesian: buat dokumen, buat dokumen teknis, generate PDF, buat PDF, buat Word, buat laporan teknis, dokumentasikan, buatkan dokumentasi, buat doc\n'
        '    - English: generate document, create technical doc, make a PDF, create Word document, document this repo, write technical documentation\n'
        '    - Even if a repo URL is mentioned, if the primary intent is to produce a document (not to fix/edit code), use "document_creation"\n'
        '    - Pre-agent tools: include "tavily_search" only if no repo URL is present and additional context from the web would help',
        frozenset({"document_creation"}),
    ),
    (
        '16. Use "system_info" when the user asks about the current resource usage or hardware specs of THIS running server/machine:\n'
        '    - Indonesian: berapa CPU, cek RAM, info memori, cek storage, disk penuh, berapa sisa disk, status server, resource server,\n'
        '      info hardware server, penggunaan CPU, penggunaan memori, berapa banyak RAM, lihat storage, cek resource\n'
        '    - English: check CPU, how much RAM, disk space, storage info, server resource, memory usage, CPU usage, hardware info server\n'
        '    - Key differentiator: user wants LIVE metrics of THIS machine. If user asks about cloud billing or a remote server by URL \u2192 general_inquiry.',
        frozenset({"system_info"}),
    ),
    (
        '17. Use "log_viewer" when the user wants to see, read, or debug the bot\u2019s recent log output:\n'
        '    - Indonesian: lihat log, tampilkan log, cek log, log bot, log error, log terbaru, debug log, tampilkan 20 log, log terakhir,\n'
        '      lihat log terakhir, beri tahu log, tunjukkan log, log aplikasi, lihat catatan log\n'
        '    - English: show log, view log, check log, bot log, recent log, last log lines, debug log, show me the logs, display logs,\n'
        '      application log, log output, what does the log say, show last 10 lines of log',
        frozenset({"log_viewer"}),
    ),
    (
        '18. Use "web_automation" when the user wants the bot to open, navigate, interact with, or extract information from a website:\n'
        '    - Indonesian: buka website, kunjungi URL, klik tombol di website, isi form, screenshot website, ambil isi halaman,\n'
        '      buka link, navigasi ke halaman, login ke website, klik menu, scraping, cek halaman, buka browser, akses URL,\n'
        '      pergi ke website, daftarkan akun di, isi formulir di, klik daftar, ambil konten dari, buka url ini\n'
        '    - English: open website, visit URL, click button on website, fill form, take screenshot, get page content,\n'
        '      navigate to page, login to website, click menu, scrape website, check page, open browser, access URL,\n'
        '      go to website, register account at, fill out form at, click sign up, get content from\n'
        '    - Key differentiator: user wants the bot to actually BROWSE and INTERACT with a live website, not just search for info.\n'
        '      If user says "buka website X dan klik tombol Y" / "login ke situs Z" / "isi form di URL ini" \u2192 web_automation.\n'
        '      If user just wants information found via web search \u2192 research.',
        frozenset({"web_automation"}),
    ),
    (
        '19. When the input starts with "[Pesan user:" it means a PDF document was uploaded by the user.\n'
        '    "[Pesan user: ...]" contains the caption that user typed when sending the PDF\n'
        '    (or "(tidak ada pesan dari pengguna)" when no caption was provided).\n'
        '    "[Preview dokumen: ...]" contains the beginning of the document\'s actual text content.\n'
        '    Classify based on BOTH the caption intent AND the document preview:\n'
        '    - Caption contains "kuis" / "soal" / "quiz" / "pertanyaan" / "latihan" OR preview looks like structured educational/study material (chapters, definitions, numbered items) \u2192 quiz_generation\n'
        '    - Caption contains "kuis telegram" / "polling kuis" / "kirim poll" / "kirim soal poll" / "quiz telegram" / "soal telegram" \u2192 telegram_quiz\n'
        '    - Caption contains "bank soal" / "kumpulan soal" / "soal ujian" / "soal latihan" / "ekstrak soal" / "import soal" / "ambil soal" / "pdf soal" / "bank kuis" OR preview looks like a question bank (numbered questions with A/B/C/D options, "kunci jawaban", answer keys) \u2192 telegram_quiz_bank\n'
        '    - Caption contains "ringkas" / "rangkum" / "summarize" / "apa isi" / "ceritakan" / "jelaskan" / "apa yang ada" / "kesimpulan" \u2192 pdf_summarization\n'
        '    - Caption is "(tidak ada pesan dari pengguna)" or vague and document type is unclear \u2192 needs_clarification asking what they want done with the PDF\n'
        '    Examples:\n'
        '      Input has "[Pesan user: buat kuis dari ini]" \u2192 {"intent": "quiz_generation", "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}\n'
        '      Input has "[Pesan user: buat kuis telegram]" \u2192 {"intent": "telegram_quiz", "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}\n'
        '      Input has "[Pesan user: bank soal ini ekstrak jadi kuis telegram]" \u2192 {"intent": "telegram_quiz_bank", "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}\n'
        '      Input has "[Pesan user: ringkas dokumen ini]" \u2192 {"intent": "pdf_summarization", "confidence": 0.93, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}\n'
        '      Input has "[Pesan user: (tidak ada pesan dari pengguna)]" \u2192 {"intent": "unknown", "confidence": 0.20, "tools": [], "needs_clarification": true, "clarification_question": "Mau diapakan dokumen ini? Misalnya: buat kuis interaktif, kuis telegram, ringkasan, atau ada keperluan lain?", "sub_intent": null}',
        frozenset({"quiz_generation", "telegram_quiz", "telegram_quiz_bank", "pdf_summarization"}),
    ),
    (
        '20. When "Riwayat percakapan terakhir" is present in the context, use it to detect follow-up commands:\n'
        '    - If the most recent [Asisten] response clearly involved web browsing, clicking, form filling, login,\n'
        '      screenshot, or navigation (i.e., the previous intent was web_automation), AND the current user message\n'
        '      is a short follow-up that does NOT mention a completely different topic (e.g. "berikan screenshot",\n'
        '      "klik tombol X", "scroll ke bawah", "ambil foto halaman", "tangkap layar", "lanjutkan",\n'
        '      "isi form", "klik menu", "screenshot dong", "screenshoot", "foto halaman"), classify the current\n'
        '      message as "web_automation" with high confidence.\n'
        '    - Similarly, if the most recent [Asisten] response was a research/code task and the current message\n'
        '      is clearly a follow-up to that task, keep the same intent classification.\n'
        '    - If the most recent [Asisten] response contains a document analysis report (indicated by phrases such as\n'
        '      "Laporan Analisis Dokumen", "Daftar Isi", "Ringkasan per Bab", or "Tip: Balas pesan ini untuk bertanya"),\n'
        '      AND the current user message is a question or request about the document content (e.g. "jelaskan bab 3",\n'
        '      "apa maksud X di bab 4", "detail tentang bagian ini", "cek konsistensi"), classify as "doc_audit".',
        None,  # Always include – context-awareness is universal
    ),
    (
        '21. Use "doc_audit" when the user asks follow-up questions about a .docx document that was previously analyzed:\n'
        '    - Indonesian: jelaskan bab ini, detail tentang bagian X, apa yang dibahas di bab Y, cek konsistensi,\n'
        '      bandingkan bab, apa maksud X, detail bab, ringkasan ulang bab, pertanyaan tentang dokumen\n'
        '    - English: explain chapter X, what does section Y say, detail about part Z, check consistency,\n'
        '      compare chapters, what is discussed in chapter N, questions about the document\n'
        '    - This intent is only valid when there is a previously analyzed document in the session.',
        frozenset({"doc_audit"}),
    ),
    (
        '22. Use "reminder" when the user wants to set, view, or cancel a timed reminder/alarm:\n'
        '    - Indonesian: ingatkan saya, set reminder, buat pengingat, jadwalkan pengingat, daftar reminder,\n'
        '      lihat reminder, tampilkan reminder, hapus reminder, batalkan reminder, cancel reminder,\n'
        '      remind me, alarm, pengingat, set alarm, atur pengingat\n'
        '    - English: remind me, set a reminder, create reminder, schedule reminder, list reminders,\n'
        '      show reminders, delete reminder, cancel reminder, set alarm\n'
        '    - Examples: "ingatkan saya untuk checkin jam 07:59", "remind me to take medicine tomorrow at 8am",\n'
        '      "lihat daftar reminderku", "hapus reminder #3"',
        frozenset({"reminder"}),
    ),
    (
        '23. Use "telegram_quiz_bank" when the user uploads a PDF that ALREADY CONTAINS a collection of exam/quiz\n'
        '    questions (bank soal) and wants those questions EXTRACTED and sent as Telegram polls \u2014 NOT generating\n'
        '    new questions from study material:\n'
        '    - Indonesian: bank soal, kumpulan soal, soal ujian, soal latihan, ekstrak soal, import soal,\n'
        '      ambil soal dari pdf, soal sudah ada, pdf soal, bank kuis, soal-soal, ekstrak kuis,\n'
        '      jadikan polling soal-soal ini, kirim soal dari pdf ini, ambilkan soal dari sini\n'
        '    - English: question bank, exam questions, extract questions, import questions, get questions from pdf,\n'
        '      questions already in pdf, bank of questions, existing questions\n'
        '    - Key differentiator: user wants to EXTRACT pre-existing questions, NOT generate new ones from study text.\n'
        '      If caption says "buat soal" / "generate soal" / "buat kuis dari materi ini" \u2192 quiz_generation or telegram_quiz.\n'
        '      If caption says "bank soal" / "ekstrak soal" / "soal yang ada di pdf ini" / preview shows A/B/C/D options \u2192 telegram_quiz_bank.\n'
        '      NEVER classify as telegram_quiz_bank when the PDF contains study material (chapters, definitions) and user says "buat soal/kuis".',
        frozenset({"telegram_quiz_bank"}),
    ),
    (
        '24. Use "diagram_from_analysis" when the user wants to create a flow diagram, flowchart, or visual\n'
        '    summary BASED ON the analysis and Q&A done in the current active document session:\n'
        '    - Indonesian: buat diagram, buat flow diagram, gambarkan alur, buat flowchart, visualisasikan,\n'
        '      buat diagram dari diskusi, buat diagram dari analisa, diagram dari hasil qna, diagram dari tadi,\n'
        '      buat diagram alur, buat chart, buat visualisasi, gambarkanlah, flow chart dari dokumen ini,\n'
        '      diagram dari hasil analisa kita, buat diagram dari pembahasan kita\n'
        '    - English: create diagram, make flow diagram, draw diagram, generate diagram, visualize the flow,\n'
        '      create flowchart, diagram from our analysis, diagram from discussion, flow diagram from the document,\n'
        '      diagram from qna, make a chart of the analysis\n'
        '    - Key differentiator: user is in an active document session and wants a DIAGRAM OUTPUT, not more text answers.\n'
        '      If user says "buat diagram dari diskusi kita" / "gambarkan alur dari analisa" \u2192 diagram_from_analysis.\n'
        '      If user says "jelaskan" / "apa maksud" \u2192 doc_audit.',
        frozenset({"diagram_from_analysis"}),
    ),
]

# Example responses shown to the LLM; one entry per intent.
# The builder filters to only those intents active in the current mode.
_EXAMPLES: dict[str, str] = {
    "wbs_planning":           '  {"intent": "wbs_planning",           "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "mandays_planning":       '  {"intent": "mandays_planning",       "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "research":               '  {"intent": "research",               "confidence": 0.91, "tools": ["tavily_search"], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "code_development":       '  {"intent": "code_development",       "confidence": 0.96, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "code_inspection":        '  {"intent": "code_inspection",        "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "code_understanding":     '  {"intent": "code_understanding",     "confidence": 0.94, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": "api_endpoints"}',
    "code_fix":               '  {"intent": "code_fix",               "confidence": 0.94, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "document_creation":      '  {"intent": "document_creation",      "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "system_info":            '  {"intent": "system_info",            "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "log_viewer":             '  {"intent": "log_viewer",             "confidence": 0.98, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "web_automation":         '  {"intent": "web_automation",         "confidence": 0.96, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "general_inquiry":        '  {"intent": "general_inquiry",        "confidence": 0.88, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "unknown":                '  {"intent": "unknown",                "confidence": 0.30, "tools": [], "needs_clarification": true,  "clarification_question": "Boleh saya tahu lebih detail tentang apa yang ingin Anda lakukan? Apakah Anda ingin membuat dokumen, mencari informasi, atau ada kebutuhan teknis lainnya?", "sub_intent": null}',
    "quiz_generation":        '  {"intent": "quiz_generation",        "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "telegram_quiz":          '  {"intent": "telegram_quiz",          "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "telegram_quiz_bank":     '  {"intent": "telegram_quiz_bank",     "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "pdf_summarization":      '  {"intent": "pdf_summarization",      "confidence": 0.93, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "doc_audit":              '  {"intent": "doc_audit",              "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "diagram_from_analysis":  '  {"intent": "diagram_from_analysis",  "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "reminder":               '  {"intent": "reminder",               "confidence": 0.97, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
    "content_creation":       '  {"intent": "content_creation",       "confidence": 0.95, "tools": [], "needs_clarification": false, "clarification_question": null, "sub_intent": null}',
}


def _build_system_prompt(allowed_intents: "list[str] | None" = None) -> str:
    """Assemble a mode-aware gatekeeper system prompt.

    When *allowed_intents* is ``None`` (mode ``"all"``), the full prompt is
    returned.  For any other mode, only the intent descriptions and rules that
    are relevant to the allowed agent set are included, shrinking the LLM's
    option space and reducing intent misclassification.

    Responder-backed fallback intents (general_inquiry, unknown, etc.) are
    always included regardless of mode so the bot can handle greetings and
    clarifications in every mode.
    """
    if allowed_intents is None:
        active_intents = set(_INTENT_DESCRIPTIONS.keys())
    else:
        # Ensure responder fallback intents are always present.
        active_intents = set(allowed_intents) | _RESPONDER_INTENTS

    # 1. Intent list – preserve original ordering via dict insertion order.
    intent_lines = [
        desc
        for intent, desc in _INTENT_DESCRIPTIONS.items()
        if intent in active_intents
    ]

    # 2. Rules – include only rules relevant to at least one active intent.
    rule_parts: list[str] = []
    for rule_text, applicable in _INTENT_RULES:
        if applicable is None or (applicable & active_intents):
            rule_parts.append(rule_text)

    # 3. Examples – include only those for active intents.
    example_lines = [
        line
        for intent, line in _EXAMPLES.items()
        if intent in active_intents
    ]

    return (
        _PREAMBLE
        + "\n".join(intent_lines)
        + "\n"
        + _PRE_AGENT_TOOLS
        + _SCHEMA_RULES
        + "\n".join(rule_parts)
        + "\n\nExample responses:\n"
        + "\n".join(example_lines)
        + "\n"
    )


@dataclass(frozen=True, slots=True)
class LLMIntentResponse:
    intent:                 IntentCategory
    confidence:             float
    model_used:             str
    tools:                  tuple[str, ...] = ()
    needs_clarification:    bool            = False
    clarification_question: str | None      = None
    sub_intent:             str | None      = None


class GatekeeperLLMClient:
    """Async LLM client for intent classification, routed through the active provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = LLMClient(settings)

    async def classify_intent(
        self,
        user_text: str,
        history: "list[dict] | None" = None,
        allowed_intents: "list[str] | None" = None,
    ) -> LLMIntentResponse:
        # Inject current date/time (WIB, UTC+7) so the LLM is never anchored to
        # its training-data cutoff when answering time-sensitive questions.
        wib = timezone(timedelta(hours=7))
        now_str = datetime.now(tz=wib).strftime("%A, %d %B %Y %H:%M WIB")

        # Build a mode-scoped prompt: only the intent descriptions and rules
        # relevant to the active mode are sent, reducing the LLM option space
        # and minimising intent misclassification.
        system_content_base = _build_system_prompt(allowed_intents) + f"\n\nWaktu saat ini: {now_str}"

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
                system_content_base
                + "\n\nRiwayat percakapan terakhir:\n"
                + "\n".join(lines)
            )
        else:
            system_content = system_content_base

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
        raw = await self._llm.chat(messages, max_tokens=5012, json_mode=True)
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
            sub_intent             = parsed.get("sub_intent") or None
            if isinstance(sub_intent, str):
                sub_intent = sub_intent.strip() or None
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse LLM response (%s): %r", exc, raw_content)
            intent                 = IntentCategory.UNKNOWN
            confidence             = 0.0
            tools                  = ()
            needs_clarification    = True
            clarification_question = None
            sub_intent             = None

        return LLMIntentResponse(
            intent=intent,
            confidence=confidence,
            model_used=model_used,
            tools=tools,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            sub_intent=sub_intent,
        )


# Keep backward-compatible alias so any code still importing OpenRouterClient
# continues to work without modification.
OpenRouterClient = GatekeeperLLMClient
