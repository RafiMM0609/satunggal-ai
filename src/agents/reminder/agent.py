"""
ReminderAgent – set, list, and cancel timed reminders delivered via Telegram.

Capabilities:
1. Set reminder  – parse natural-language time/date + message, store in SQLite,
                   schedule an APScheduler job to send via Telegram.
2. List reminders – return a formatted list of the user's pending reminders.
3. Cancel reminder – delete a reminder by its number/id.

Scheduling is done with APScheduler (AsyncIOScheduler).  A single process-wide
scheduler instance is managed in `src.agents.reminder.scheduler`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask
from src.tools.reminder_store import get_reminder_store, Reminder

logger = logging.getLogger(__name__)

# ── System prompt for LLM parsing ────────────────────────────────────────────

_PARSE_SYSTEM_PROMPT = """\
Kamu adalah parser reminder. Tugas kamu adalah mengekstrak informasi dari permintaan reminder pengguna.

Hari dan waktu saat ini (UTC+7 / WIB): {now_wib}

Ekstrak dari pesan pengguna:
1. "message" – apa yang harus diingatkan (singkat dan jelas)
2. "remind_at_iso" – kapan harus mengirim reminder, dalam format ISO-8601 UTC
   - Konversi waktu yang disebutkan dari WIB (UTC+7) ke UTC
   - Jika user menyebut "07:59", dan hari tidak disebutkan, gunakan hari ini (atau besok jika waktunya sudah lewat)
   - Jika user menyebut "besok pukul 08:00", gunakan tanggal besok
   - Jika user menyebut "Senin jam 09:00", hitung tanggal Senin berikutnya
3. "action" – salah satu dari: "set", "list", "cancel"
   - "set" = user ingin membuat reminder baru
   - "list" = user ingin melihat daftar reminder
   - "cancel" = user ingin membatalkan/menghapus reminder
4. "cancel_id" – integer id reminder yang akan dihapus (hanya untuk action "cancel"), atau null

Balas HANYA dengan JSON, tanpa markdown, tanpa penjelasan:
{
  "action": "set" | "list" | "cancel",
  "message": "<isi reminder atau null>",
  "remind_at_iso": "<ISO-8601 UTC datetime atau null>",
  "cancel_id": <integer atau null>
}

Contoh:
User: "ingatkan saya untuk checkin pada pukul 07:59"
→ {"action": "set", "message": "checkin", "remind_at_iso": "2025-01-15T00:59:00", "cancel_id": null}

User: "lihat daftar reminder saya"
→ {"action": "list", "message": null, "remind_at_iso": null, "cancel_id": null}

User: "hapus reminder nomor 3"
→ {"action": "cancel", "message": null, "remind_at_iso": null, "cancel_id": 3}
"""

_PARSE_USER_TEMPLATE = "Pesan pengguna: {user_input}"


class ReminderAgent(BaseAgent):
    """Manages timed reminders stored in SQLite and delivered via Telegram."""

    name = "reminder_agent"

    def __init__(
        self,
        history: ConversationHistory,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self._history = history
        self._llm = llm or LLMClient()
        self._store = get_reminder_store()

    # ── Main entry ────────────────────────────────────────────────────────────

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            parsed = await self._parse_intent(task.user_input)
            action = parsed.get("action", "set")

            if action == "list":
                reply = self._handle_list(task.session_id)
            elif action == "cancel":
                cancel_id = parsed.get("cancel_id")
                reply = self._handle_cancel(task.session_id, cancel_id)
            else:
                reply = await self._handle_set(task.session_id, parsed)

            task.mark_done(reply)
        except Exception as exc:
            logger.exception("ReminderAgent error: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses reminder. Silakan coba lagi."

        return task

    # ── Action handlers ───────────────────────────────────────────────────────

    async def _handle_set(self, chat_id: str, parsed: dict) -> str:
        message = parsed.get("message")
        remind_at_iso = parsed.get("remind_at_iso")

        if not message or not remind_at_iso:
            return (
                "⚠️ Saya tidak bisa memahami detail reminder Anda.\n"
                "Coba tulis seperti ini:\n"
                "• *ingatkan saya untuk meeting jam 14:00*\n"
                "• *remind me to take medicine at 08:00 tomorrow*"
            )

        try:
            remind_at = datetime.fromisoformat(remind_at_iso).replace(tzinfo=timezone.utc)
        except ValueError:
            return f"⚠️ Format waktu tidak valid: `{remind_at_iso}`"

        now_utc = datetime.now(timezone.utc)
        if remind_at <= now_utc:
            # If time already passed today, it might be intended for tomorrow
            remind_at += timedelta(days=1)
            if remind_at <= now_utc:
                return (
                    "⚠️ Waktu reminder sudah lewat. Silakan tentukan waktu di masa mendatang."
                )

        # Save to DB
        reminder = self._store.add(chat_id, message, remind_at)

        # Schedule the job
        from src.agents.reminder.scheduler import schedule_reminder
        await schedule_reminder(reminder)

        # Format display time in WIB (UTC+7)
        wib_time = remind_at + timedelta(hours=7)
        time_str = wib_time.strftime("%d %B %Y pukul %H:%M WIB")

        return (
            f"✅ Reminder berhasil dibuat!\n\n"
            f"📌 *#{reminder.id}* — {reminder.message}\n"
            f"⏰ {time_str}"
        )

    def _handle_list(self, chat_id: str) -> str:
        reminders = self._store.list_pending(chat_id)
        if not reminders:
            return "📭 Tidak ada reminder aktif."

        lines = ["📋 *Daftar Reminder Aktif:*\n"]
        for r in reminders:
            wib_time = r.remind_at + timedelta(hours=7)
            time_str = wib_time.strftime("%d %b %Y %H:%M WIB")
            lines.append(f"• *#{r.id}* — {r.message}\n  ⏰ {time_str}")

        lines.append(f"\n_Total: {len(reminders)} reminder aktif_")
        lines.append("_Untuk menghapus: \"hapus reminder #[id]\"_")
        return "\n".join(lines)

    def _handle_cancel(self, chat_id: str, cancel_id) -> str:
        if cancel_id is None:
            return (
                "⚠️ ID reminder tidak ditemukan. "
                "Gunakan `/list reminder` untuk melihat daftar, lalu "
                "tulis \"hapus reminder #[nomor]\"."
            )

        try:
            rid = int(cancel_id)
        except (TypeError, ValueError):
            return f"⚠️ ID reminder tidak valid: `{cancel_id}`"

        # Also remove the scheduler job
        from src.agents.reminder.scheduler import cancel_scheduled_reminder
        cancel_scheduled_reminder(rid)

        deleted = self._store.delete(rid, chat_id)
        if deleted:
            return f"🗑️ Reminder *#{rid}* berhasil dihapus."
        return (
            f"⚠️ Reminder *#{rid}* tidak ditemukan atau bukan milik Anda, "
            "atau sudah terkirim."
        )

    # ── LLM parsing ───────────────────────────────────────────────────────────

    async def _parse_intent(self, user_input: str) -> dict:
        """Use LLM to parse the user's reminder request into structured data."""
        now_utc = datetime.now(timezone.utc)
        now_wib = now_utc + timedelta(hours=7)
        now_wib_str = now_wib.strftime("%A, %d %B %Y %H:%M WIB")

        system_prompt = _PARSE_SYSTEM_PROMPT.format(now_wib=now_wib_str)
        user_prompt = _PARSE_USER_TEMPLATE.format(user_input=user_input)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = await self._llm.chat(messages)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract the first JSON object from the LLM response."""
        import json

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Find first { ... }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning("ReminderAgent: no JSON found in LLM response: %s", raw[:200])
            return {"action": "set", "message": None, "remind_at_iso": None, "cancel_id": None}

        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("ReminderAgent: JSON parse error: %s — raw: %s", exc, raw[:200])
            return {"action": "set", "message": None, "remind_at_iso": None, "cancel_id": None}
