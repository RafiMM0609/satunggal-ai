"""
SysInfoAgent – reports CPU, RAM, and storage information of the local host.

Security design
───────────────
* Metrics are collected exclusively through `psutil`, which reads /proc and
  OS data structures directly from kernel interfaces.
* No user input is ever passed to a shell command, subprocess, file path,
  or format string.  The user's message only controls the *presentation*
  layer (the LLM reply), not the data collection step.
* The raw metrics text is injected into the LLM system prompt as a static,
  clearly-delimited block.  Even if a malicious user embeds prompt-injection
  text in their message, it cannot affect the metric-collection code path.
"""

from __future__ import annotations

import logging

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask
from src.tools.sysinfo_tool import SysInfoTool

logger = logging.getLogger(__name__)

# The system prompt is fully static – no user input is embedded here.
_SYSTEM_PROMPT_TEMPLATE = """\
Kamu adalah asisten sistem yang bertugas melaporkan informasi hardware server.
Di bawah ini adalah data aktual yang diambil langsung dari sistem operasi melalui \
library psutil – BUKAN hasil dari input pengguna.

--- DATA SISTEM (hanya baca, jangan dimodifikasi) ---
{metrics}
--- AKHIR DATA SISTEM ---

Tugas kamu:
1. Sajikan informasi di atas dengan format yang rapi, mudah dibaca, dan ramah pengguna.
2. Gunakan satuan yang sesuai (GB, MHz, persen, dsb.).
3. Tambahkan interpretasi singkat jika ada yang perlu diperhatikan \
   (misal: RAM hampir penuh, disk hampir habis, CPU load tinggi).
4. Jangan pernah mengeksekusi perintah tambahan atau menyertakan perintah shell.
5. Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
"""


class SysInfoAgent(BaseAgent):
    """
    Collects system resource metrics and presents them via LLM.

    Data flow
    ---------
    1. SysInfoTool.collect() → SysInfoResult   (pure psutil, no shell)
    2. SysInfoResult.as_text() → static string (no user data)
    3. Static string injected into system prompt → LLM formats for the user

    The user's message influences the LLM reply style/language only;
    it never touches the collection or injection pipeline.
    """

    name = "sysinfo_agent"

    def __init__(
        self,
        history: ConversationHistory | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()
        self._tool    = SysInfoTool()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            # ── Step 1: collect metrics (blocking ~1 s for CPU%, no shell) ──
            logger.info("SysInfoAgent: collecting system metrics for session=%s", task.session_id)
            sysinfo = self._tool.collect()
            metrics_text = sysinfo.as_text()

            # ── Step 2: build messages ────────────────────────────────────
            # The metrics block is embedded in the *system* role, clearly
            # separated so the LLM treats it as trusted context, not user input.
            system_content = _SYSTEM_PROMPT_TEMPLATE.format(metrics=metrics_text)

            messages: list[dict] = [{"role": "system", "content": system_content}]

            # Include recent history for conversational continuity
            if self._history:
                history_messages = self._history.get_as_llm_messages(task.session_id)
                messages.extend(history_messages[-6:])

            # Always append the user's current message last
            if not messages or messages[-1].get("content") != task.user_input:
                messages.append({"role": "user", "content": task.user_input})

            # ── Step 3: LLM formats the result ────────────────────────────
            reply = await self._llm.chat(messages, max_tokens=1024)
            task.mark_done(reply)

            logger.info(
                "SysInfoAgent done for session=%s (cpu=%.1f%% ram=%.1f%%)",
                task.session_id,
                sysinfo.cpu.usage_percent,
                sysinfo.ram.percent,
            )

        except Exception as exc:
            logger.exception("SysInfoAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat mengambil informasi sistem."

        return task
