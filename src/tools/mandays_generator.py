"""
MandaysGeneratorTool – pure Excel builder, NO LLM.

Responsibility (single):
  Read Mandays JSON from task.metadata["mandays_json_data"] and write it to an
  Excel file via generate_mandays.generate_excel.

Flow (called by orchestrator AFTER MandaysAgent finishes):
  1. MandaysAgent calls LLM → parses JSON → stores in task.metadata["mandays_json_data"]
  2. MandaysAgent appends "mandays_generator" to task.pending_tools
  3. Orchestrator sees pending_tools, calls MandaysGeneratorTool.run(task)
  4. This tool reads the JSON, generates Excel, returns path
  5. Orchestrator stores excel_path in task.metadata["excel_path"]
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)


class MandaysGeneratorTool(BaseTool):
    """Converts a Mandays JSON dict (already in task.metadata) into an Excel file."""

    name = "mandays_generator"
    description = (
        "Generate a Mandays estimation Excel spreadsheet from a structured "
        "JSON definition stored in task.metadata['mandays_json_data']. "
        "Called by the orchestrator after MandaysAgent has produced the mandays JSON."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "mandays_json_data": {
                "type": "object",
                "description": "Structured mandays estimation data (set in task.metadata['mandays_json_data'] by MandaysAgent).",
            },
        },
        "required": ["mandays_json_data"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "excel_path":   {"type": "string", "description": "Absolute path to the generated Excel file."},
            "project_name": {"type": "string", "description": "Name of the project."},
            "error":        {"type": "string", "description": "Present only on failure."},
        },
    }

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """
        Generate Mandays Excel from pre-parsed JSON in task.metadata["mandays_json_data"].

        Returns:
            { "excel_path": str, "project_name": str }
            or { "error": str }
        """
        data = task.metadata.get("mandays_json_data")
        if not data:
            logger.error(
                "MandaysGeneratorTool: task.metadata['mandays_json_data'] missing. session=%s",
                task.session_id,
            )
            return {"error": "mandays_json_data not found in task metadata"}

        excel_path = _make_output_path(task.session_id, prefix="mandays")
        try:
            from src.tools.mandays.generate_mandays import generate_excel  # noqa: PLC0415
            generate_excel(data, excel_path)
        except Exception as exc:
            logger.exception("MandaysGeneratorTool: generate_excel failed: %s", exc)
            return {"error": str(exc)}

        project_name = data.get("project_info", {}).get("name", "Proyek")
        logger.info(
            "MandaysGeneratorTool: Excel OK session=%s path=%s",
            task.session_id, excel_path,
        )
        return {"excel_path": excel_path, "project_name": project_name}


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_output_path(session_id: str, prefix: str) -> str:
    out_dir = os.path.join(tempfile.gettempdir(), "advance_ai_excel")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{prefix}_{session_id}_{ts}.xlsx")
