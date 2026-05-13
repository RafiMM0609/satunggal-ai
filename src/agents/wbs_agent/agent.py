"""
WBSAgent – handles wbs_planning intent (WBS Gantt-chart generation).

Flow:
  1. Call LLM with WBS system prompt → receive structured JSON.
  2. Store JSON in task.metadata["wbs_json_data"].
  3. Append "wbs_generator" to task.pending_tools to signal the orchestrator
     that it should run WBSGeneratorTool next (pure Excel build, no LLM).
  4. Mark task done with a human-readable reply text.
  5. Orchestrator runs the tool post-agent, sets task.metadata["excel_path"],
     then delivers the file to the client.
"""

from __future__ import annotations

import json
import logging
import re

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Kamu adalah seorang ahli manajemen proyek dan WBS (Work Breakdown Structure).\n"
    "Pengguna akan meminta WBS proyek dengan Gantt chart / timeline per hari kerja.\n\n"
    "TUGAS KAMU:\n"
    "Buat WBS dalam format JSON berikut untuk di-render sebagai Gantt chart Excel.\n"
    "SELALU kembalikan HANYA JSON tanpa penjelasan atau markdown lain.\n\n"
    "SKEMA JSON (ikuti persis):\n"
    "{\n"
    '  "project_info": {\n'
    '    "project_name": "<nama proyek>",\n'
    '    "start_date": "<YYYY-MM-DD, contoh: 2026-03-10>",\n'
    '    "end_date":   "<YYYY-MM-DD, contoh: 2026-04-30>"\n'
    "  },\n"
    '  "timeline_config": {\n'
    '    "sprints": [\n'
    "      {\n"
    '        "sprint_name": "Sprint 1",\n'
    '        "days": [\n'
    '          {"date": 10, "month": "Maret",  "year": 2026, "is_weekday": true},\n'
    '          {"date": 11, "month": "Maret",  "year": 2026, "is_weekday": true}\n'
    "        ]\n"
    "      }\n"
    "    ]\n"
    "  },\n"
    '  "wbs_data": [\n'
    "    {\n"
    '      "category": "<nama kategori/fase, contoh: Requirement Gathering>",\n'
    '      "is_header": true,\n'
    '      "tasks": [\n'
    '        {"task_name": "<nama task>", "active_days": [10, 11]}\n'
    "      ]\n"
    "    },\n"
    "    {\n"
    '      "category": "<nama kategori dengan sub-kategori, contoh: Development>",\n'
    '      "is_header": true,\n'
    '      "sub_categories": [\n'
    "        {\n"
    '          "name": "<nama sub-kategori, contoh: Backend>",\n'
    '          "tasks": [\n'
    '            {"task_name": "<nama task>", "active_days": [12, 13, 14]}\n'
    "          ]\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "ATURAN PENTING:\n"
    "1. Gunakan HANYA hari kerja (Senin\u2013Jumat) di dalam 'days' tiap sprint.\n"
    "2. 'active_days' untuk setiap task berisi daftar angka tanggal (date) yang ada di sprint itu.\n"
    "3. Pastikan semua angka di 'active_days' cocok persis dengan field 'date' yang terdaftar di 'days' sprint yang relevan.\n"
    "4. Buat sprint 2-minggu (10 hari kerja) secara berurutan tanpa overlap.\n"
    "5. Gunakan is_header: true untuk setiap kategori utama.\n"
    "6. Gunakan 'tasks' langsung di bawah kategori jika tidak ada sub-kategori; gunakan 'sub_categories' jika ada.\n"
    "7. Tanggal mulai default: 10 Maret 2026.\n"
    "8. HANYA kembalikan JSON, tanpa teks lain.\n"
)


class WBSAgent(BaseAgent):
    """Generates WBS JSON via LLM and delegates Excel building to the orchestrator."""

    name = "wbs_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": task.user_input},
            ]

            reply = await self._llm.chat(messages)

            if not reply:
                task.mark_done("Maaf, LLM tidak memberikan respons. Coba lagi.")
                return task

            data = _extract_json(reply)
            if data is None:
                logger.warning("WBSAgent: no valid JSON in reply: %r", reply[:300])
                task.mark_done(
                    "Maaf, gagal mem-parse WBS dari LLM. Coba ulangi permintaan."
                )
                return task

            project_name = data.get("project_info", {}).get("project_name", "Proyek")

            # Store JSON for WBSGeneratorTool to consume
            task.metadata["wbs_json_data"] = data

            # Signal orchestrator to run Excel generator after this agent
            task.pending_tools.append("wbs_generator")

            task.metadata["has_wbs_json"] = True
            task.mark_done(
                f"WBS Gantt chart untuk *{project_name}* berhasil dibuat!\n"
                "File Excel sedang dikirim..."
            )
            logger.info(
                "WBSAgent: JSON ready, pending_tools=%s session=%s",
                task.pending_tools, task.session_id,
            )

        except Exception as exc:
            logger.exception("WBSAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses permintaan WBS Anda."

        return task


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Extract and parse the first JSON object from the LLM reply."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None



