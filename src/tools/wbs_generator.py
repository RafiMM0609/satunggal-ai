"""
WBSGeneratorTool – pure Excel builder, NO LLM.

Responsibility (single):
  Read WBS JSON from task.metadata["wbs_json_data"] and write it to an
  Excel file via generate_wbs.generate_excel.

Flow (called by orchestrator AFTER WBSAgent finishes):
  1. WBSAgent calls LLM → parses JSON → stores in task.metadata["wbs_json_data"]
  2. WBSAgent appends "wbs_generator" to task.pending_tools
  3. Orchestrator sees pending_tools, calls WBSGeneratorTool.run(task)
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


class WBSGeneratorTool(BaseTool):
    """Converts a WBS JSON dict (already in task.metadata) into an Excel file."""

    name = "wbs_generator"

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """
        Generate WBS Excel from pre-parsed JSON in task.metadata["wbs_json_data"].

        Returns:
            { "excel_path": str, "project_name": str }
            or { "error": str }
        """
        data = task.metadata.get("wbs_json_data")
        if not data:
            logger.error(
                "WBSGeneratorTool: task.metadata['wbs_json_data'] missing. session=%s",
                task.session_id,
            )
            return {"error": "wbs_json_data not found in task metadata"}

        excel_path = _make_output_path(task.session_id, prefix="wbs")
        try:
            from src.tools.wbs.generate_wbs import generate_excel  # noqa: PLC0415
            generate_excel(data, excel_path)
        except Exception as exc:
            logger.exception("WBSGeneratorTool: generate_excel failed: %s", exc)
            return {"error": str(exc)}

        project_name = data.get("project_info", {}).get("project_name", "Proyek")
        logger.info(
            "WBSGeneratorTool: Excel OK session=%s path=%s",
            task.session_id, excel_path,
        )
        return {"excel_path": excel_path, "project_name": project_name}


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_output_path(session_id: str, prefix: str) -> str:
    out_dir = os.path.join(tempfile.gettempdir(), "advance_ai_excel")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{prefix}_{session_id}_{ts}.xlsx")
