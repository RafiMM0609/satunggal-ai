"""
MandaysAgent – handles mandays_planning intent (effort estimation).

Flow:
  1. Call LLM with Mandays system prompt → receive structured JSON.
  2. Store JSON in task.metadata["mandays_json_data"].
  3. Append "mandays_generator" to task.pending_tools to signal the orchestrator
     that it should run MandaysGeneratorTool next (pure Excel build, no LLM).
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
    "Kamu adalah seorang ahli manajemen proyek yang berspesialisasi dalam estimasi effort dan perencanaan mandays.\n"
    "Pengguna akan meminta estimasi mandays, alokasi sumber daya, atau rencana effort untuk suatu proyek atau fitur.\n\n"
    "TUGAS KAMU:\n"
    "Buat rencana mandays yang realistis dan terperinci dalam format JSON berikut.\n"
    "SELALU kembalikan HANYA JSON tanpa penjelasan atau markdown lain.\n\n"
    "SKEMA JSON (ikuti persis):\n"
    "{\n"
    '  "project_info": {\n'
    '    "name": "<nama proyek atau fitur>",\n'
    '    "start_date": "<tanggal mulai, contoh: 10 Maret 2026>",\n'
    '    "end_date":   "<tanggal selesai estimasi>"\n'
    "  },\n"
    '  "roles": ["SA","TL","BA","SM","UI","DBA","BE1","BE2","FE1","FE2","QA","DevOps","TW"],\n'
    '  "work_breakdown_structure": [\n'
    "    {\n"
    '      "sprint_name": "Sprint 1",\n'
    '      "period": "<contoh: 10 Mar - 23 Mar 2026>",\n'
    '      "total_mandays": {"SA":0,"TL":0,"BA":0,"SM":0,"UI":0,"DBA":0,"BE1":0,"BE2":0,"FE1":0,"FE2":0,"QA":0,"DevOps":0,"TW":0},\n'
    '      "features": [\n'
    "        {\n"
    '          "feature_group": "<nama fitur atau area kerja>",\n'
    '          "tasks": [\n'
    '            {"name": "<nama task>", "mandays": {"BE1": 2, "QA": 1}}\n'
    "          ]\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    '  "grand_total": {"SA":0,"TL":0,"BA":0,"SM":0,"UI":0,"DBA":0,"BE1":0,"BE2":0,"FE1":0,"FE2":0,"QA":0,"DevOps":0,"TW":0}\n'
    "}\n\n"
    "ATURAN:\n"
    "1. Fokus pada estimasi effort yang realistis per role per task.\n"
    "2. Isi 'total_mandays' tiap sprint dengan jumlah mandays dari semua tasks di sprint itu.\n"
    "3. Isi 'grand_total' dengan total keseluruhan per role.\n"
    "4. Gunakan hanya role yang relevan \u2013 jangan isi 0 untuk role yang tidak terlibat (biarkan key tetap ada dengan nilai 0).\n"
    "5. Bagi pekerjaan ke dalam sprint 2-minggu yang realistis.\n"
    "6. Tanggal mulai default: 10 Maret 2026.\n"
    "7. HANYA kembalikan JSON, tanpa teks lain.\n"
)


class MandaysAgent(BaseAgent):
    """Generates Mandays JSON via LLM and delegates Excel building to the orchestrator."""

    name = "mandays_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": task.user_input},
            ]

            reply = await self._llm.chat(messages, max_tokens=4096)

            if not reply:
                task.mark_done("Maaf, LLM tidak memberikan respons. Coba lagi.")
                return task

            data = _extract_json(reply)
            if data is None:
                logger.warning("MandaysAgent: no valid JSON in reply: %r", reply[:300])
                task.mark_done(
                    "Maaf, gagal mem-parse rencana mandays dari LLM. Coba ulangi permintaan."
                )
                return task

            project_name = data.get("project_info", {}).get("name", "Proyek")

            # Store JSON for MandaysGeneratorTool to consume
            task.metadata["mandays_json_data"] = data

            # Signal orchestrator to run Excel generator after this agent
            task.pending_tools.append("mandays_generator")

            task.metadata["has_mandays_json"] = True
            task.mark_done(
                f"Rencana mandays untuk *{project_name}* berhasil dibuat!\n"
                "File Excel sedang dikirim..."
            )
            logger.info(
                "MandaysAgent: JSON ready, pending_tools=%s session=%s",
                task.pending_tools, task.session_id,
            )

        except Exception as exc:
            logger.exception("MandaysAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses permintaan mandays Anda."

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
