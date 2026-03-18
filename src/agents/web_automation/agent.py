"""
WebAutomationAgent – Autonomous Browsing & Web Interaction Agent.

This agent receives a natural-language web-automation request from the user
and orchestrates the WebReaderTool + BrowserNavigatorTool to carry it out.

Supported high-level tasks (intent: web_automation)
────────────────────────────────────────────────────
  • Open a URL and summarise its content.
  • Navigate to a page and describe the menu / UI elements.
  • Click a button or link identified by text.
  • Fill a form field with supplied data.
  • Scroll the page and report what appeared.
  • Take a screenshot and describe it (multimodal, when LLM supports vision).
  • Log in to a website and save the session for future reuse.

Workflow (Orchestrator-driven)
──────────────────────────────
  1. LLM decomposes the user's request into an ordered action plan.
  2. For each step the agent sets ``task.metadata["browser_action"]`` and
     calls the appropriate tool (web_reader or browser_navigator).
  3. Tool results are collected and fed back to the LLM to decide the next
     step (ReAct-style loop, max 10 iterations to prevent runaway loops).
  4. The LLM produces a final natural-language summary for the user.

VPS constraints honoured
─────────────────────────
  • One browser tab at a time.
  • Resources blocked (images/media/fonts).
  • 30-second timeout per action.
  • Browser is always closed after the task finishes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask
from src.tools.browser_navigator import BrowserNavigatorTool
from src.tools.web_reader import WebReaderTool

logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
Kamu adalah Web Automation Planner. Tugasmu adalah menguraikan permintaan \
pengguna menjadi serangkaian langkah browsing yang terurut dan dapat dieksekusi.

Setiap langkah HARUS berupa JSON object dengan field berikut:
  "action": satu dari ["read_url", "navigate", "click", "type", "scroll", "screenshot", "save_session", "done"]
  "params": object parameter yang sesuai dengan action:
    - read_url:    {"url": "..."}
    - navigate:    {"url": "..."}
    - click:       {"text": "..."}
    - type:        {"selector": "...", "text": "..."}   (selector boleh kosong)
    - scroll:      {"direction": "down"|"up"}
    - screenshot:  {}
    - save_session:{"url": "..."}
    - done:        {"summary": "ringkasan hasil untuk pengguna"}

Aturan:
1. Balas HANYA dengan JSON array dari langkah-langkah tersebut – tidak ada teks lain.
2. Selalu akhiri dengan langkah "done" yang berisi ringkasan apa yang sudah dilakukan.
3. Maksimal 8 langkah (tidak termasuk "done").
4. Gunakan bahasa yang sama dengan permintaan pengguna untuk field "summary".
"""

_SUMMARISER_SYSTEM = """\
Kamu adalah asisten yang merangkum hasil browsing web untuk pengguna.
Berdasarkan log aksi dan konten halaman yang diberikan, buat ringkasan yang:
  - Jelas dan mudah dipahami.
  - Menyebutkan URL yang dikunjungi dan judul halamannya.
  - Menjelaskan elemen-elemen penting yang ditemukan (menu, tombol, form, dll.).
  - Melaporkan status setiap aksi (berhasil / gagal).
  - Menggunakan bahasa yang sama dengan permintaan pengguna (Indonesia atau Inggris).
"""

_MAX_STEPS    = 8   # hard cap excluding the final "done" step
_MAX_TOKENS   = 2048


class WebAutomationAgent(BaseAgent):
    """
    Autonomous web browsing agent.

    Uses a ReAct-style loop:
      LLM plans → tool executes → result fed back → LLM plans next step.
    """

    name = "web_automation"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)
        navigator = BrowserNavigatorTool()
        reader    = WebReaderTool()

        try:
            steps = await self._plan_steps(task.user_input)
            logger.info(
                "WebAutomationAgent: %d steps planned for session=%s",
                len(steps), task.session_id,
            )

            action_log: list[str] = []

            for i, step in enumerate(steps[:_MAX_STEPS], start=1):
                action = step.get("action", "done")
                params = step.get("params", {})

                if action == "done":
                    summary = params.get("summary", "Selesai.")
                    action_log.append(f"[{i}] done → {summary}")
                    break

                log_entry, tool_result = await self._execute_step(
                    action=action,
                    params=params,
                    task=task,
                    reader=reader,
                    navigator=navigator,
                    step_num=i,
                )
                action_log.append(log_entry)
                task.tool_results[f"step_{i}_{action}"] = tool_result

                if tool_result.get("error"):
                    logger.warning(
                        "WebAutomationAgent: step %d/%d failed: %s",
                        i, len(steps), tool_result["error"],
                    )
                    action_log.append(f"  ⚠ Gagal: {tool_result['error']}")
                    # Continue to next step instead of aborting (best-effort)

            # Build final reply using the LLM summariser
            reply = await self._summarise(task.user_input, action_log, task.tool_results)
            task.mark_done(reply)

        except Exception as exc:
            logger.exception("WebAutomationAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                "Maaf, terjadi kesalahan saat menjalankan web automation. "
                f"Detail: {exc}"
            )
        finally:
            await navigator.close()

        return task

    # ── Step planner ─────────────────────────────────────────────────────────

    async def _plan_steps(self, user_input: str) -> list[dict[str, Any]]:
        """Ask the LLM to produce a structured action plan."""
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": user_input},
        ]
        try:
            raw = await self._llm.chat(messages, max_tokens=512)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            steps = json.loads(raw)
            if not isinstance(steps, list):
                raise ValueError("Expected a JSON array")
            return steps
        except Exception as exc:
            logger.warning("WebAutomationAgent: plan parsing failed (%s), using fallback", exc)
            # Fallback: treat the whole input as a read_url request if it looks like a URL
            url = self._extract_url(user_input)
            if url:
                return [
                    {"action": "read_url", "params": {"url": url}},
                    {"action": "done",     "params": {"summary": "Konten URL telah dibaca."}},
                ]
            return [{"action": "done", "params": {"summary": "Tidak dapat membuat rencana aksi."}}]

    # ── Step executor ────────────────────────────────────────────────────────

    async def _execute_step(
        self,
        *,
        action: str,
        params: dict[str, Any],
        task: AgentTask,
        reader: WebReaderTool,
        navigator: BrowserNavigatorTool,
        step_num: int,
    ) -> tuple[str, dict[str, Any]]:
        """Execute a single planned step and return (log_entry, tool_result)."""
        result: dict[str, Any] = {}

        if action == "read_url":
            url = params.get("url", "")
            task.metadata["target_url"] = url
            result = await reader.run(task)
            log = (
                f"[{step_num}] read_url {url} → "
                f"title={result.get('title', '?')!r} "
                f"nodes={len(result.get('a11y_tree', []))}"
            )

        elif action == "navigate":
            url = params.get("url", "")
            task.metadata.update({"browser_action": "navigate", "target_url": url})
            result = await navigator.run(task)
            log = f"[{step_num}] navigate → {result.get('message', result.get('error', '?'))}"

        elif action == "click":
            text = params.get("text", "")
            task.metadata.update({"browser_action": "click", "click_text": text})
            result = await navigator.run(task)
            log = f"[{step_num}] click '{text}' → {result.get('message', result.get('error', '?'))}"

        elif action == "type":
            task.metadata.update({
                "browser_action":  "type",
                "type_selector":   params.get("selector", ""),
                "type_text":       params.get("text", ""),
            })
            result = await navigator.run(task)
            log = f"[{step_num}] type → {result.get('message', result.get('error', '?'))}"

        elif action == "scroll":
            direction = params.get("direction", "down")
            task.metadata.update({"browser_action": "scroll", "scroll_direction": direction})
            result = await navigator.run(task)
            log = f"[{step_num}] scroll {direction} → {result.get('message', result.get('error', '?'))}"

        elif action == "screenshot":
            task.metadata["browser_action"] = "screenshot"
            result = await navigator.run(task)
            has_img = bool(result.get("screenshot_b64"))
            log = f"[{step_num}] screenshot → {'captured' if has_img else 'failed'}"

        elif action == "save_session":
            session_url = params.get("url", "")
            task.metadata.update({"browser_action": "save_session", "session_url": session_url})
            result = await navigator.run(task)
            log = f"[{step_num}] save_session → {result.get('message', result.get('error', '?'))}"

        else:
            result = {"error": f"Unknown action: {action}"}
            log    = f"[{step_num}] unknown action: {action}"

        return log, result

    # ── Summariser ────────────────────────────────────────────────────────────

    async def _summarise(
        self,
        user_input: str,
        action_log: list[str],
        tool_results: dict[str, Any],
    ) -> str:
        """Ask the LLM to produce a user-friendly summary of what happened."""
        log_text = "\n".join(action_log)

        # Build a short context snippet from the last read_url result
        page_snippet = ""
        for key, val in tool_results.items():
            if isinstance(val, dict) and "page_text" in val:
                page_snippet = val["page_text"][:2000]
                break

        context = f"Log aksi:\n{log_text}"
        if page_snippet:
            context += f"\n\nCuplikan konten halaman:\n{page_snippet}"

        messages = [
            {"role": "system", "content": _SUMMARISER_SYSTEM},
            {"role": "user",   "content": f"Permintaan pengguna:\n{user_input}\n\n{context}"},
        ]
        try:
            return await self._llm.chat(messages, max_tokens=_MAX_TOKENS)
        except Exception as exc:
            logger.warning("WebAutomationAgent: summariser failed: %s", exc)
            return f"Web automation selesai.\n\nLog:\n{log_text}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_url(text: str) -> str:
        """Extract the first http(s) URL from *text*, or return empty string."""
        import re
        m = re.search(r"https?://[^\s\"'<>]+", text)
        return m.group(0) if m else ""
