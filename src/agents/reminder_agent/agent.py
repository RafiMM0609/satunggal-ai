"""
ReminderAgent – set, list, and cancel timed reminders delivered via Telegram.

Capabilities:
1. Set reminder  – parse natural-language time/date + message, store in SQLite,
                   schedule an APScheduler job to send via Telegram.
2. List reminders – return a formatted list of the user's pending reminders.
3. Cancel reminder – delete a reminder by its number/id.

Scheduling is done with APScheduler (AsyncIOScheduler).  A single process-wide
scheduler instance is managed in `src.agents.reminder_agent.scheduler`.
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
from src.tools.reminder_store import get_reminder_store

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
3. "action" – salah satu dari: "set", "list", "cancel", "clarify"
   - "set" = user ingin membuat reminder baru dengan detail yang cukup jelas
   - "list" = user ingin melihat daftar reminder
   - "cancel" = user ingin membatalkan/menghapus reminder
   - "clarify" = user meminta reminder tapi TIDAK menyebutkan: (a) apa yang harus diingatkan DAN (b) kapan waktunya — permintaan terlalu samar/tidak lengkap
4. "cancel_id" – integer id reminder yang akan dihapus (hanya untuk action "cancel"), atau null

Balas HANYA dengan JSON, tanpa markdown, tanpa penjelasan:
{{
  "action": "set" | "list" | "cancel" | "clarify",
  "message": "<isi reminder atau null>",
  "remind_at_iso": "<ISO-8601 UTC datetime atau null>",
  "cancel_id": <integer atau null>
}}

Contoh:
User: "ingatkan saya untuk checkin pada pukul 07:59"
→ {{"action": "set", "message": "checkin", "remind_at_iso": "2025-01-15T00:59:00", "cancel_id": null}}

User: "lihat daftar reminder saya"
→ {{"action": "list", "message": null, "remind_at_iso": null, "cancel_id": null}}

User: "hapus reminder nomor 3"
→ {{"action": "cancel", "message": null, "remind_at_iso": null, "cancel_id": 3}}

User: "tolong bikin reminder" (tanpa detail apa dan kapan)
→ {{"action": "clarify", "message": null, "remind_at_iso": null, "cancel_id": null}}

User: "set reminder dong" (terlalu samar)
→ {{"action": "clarify", "message": null, "remind_at_iso": null, "cancel_id": null}}
"""

_PARSE_USER_TEMPLATE = "Pesan pengguna: {user_input}"

# ── System prompt for elaborating a vague reminder request ───────────────────

_ELABORATE_SYSTEM_PROMPT = """\
Kamu adalah asisten yang membantu membuat detail reminder dari permintaan yang samar.
Berdasarkan konteks percakapan, buat detail reminder yang masuk akal.

Hari dan waktu saat ini (UTC+7 / WIB): {now_wib}

Tentukan:
1. "message" – isi reminder yang jelas dan spesifik (singkat, maks 20 kata)
2. "remind_at_iso" – waktu reminder dalam ISO-8601 UTC (pilih waktu yang masuk akal berdasarkan konteks)
3. "action" – selalu "set"

Balas HANYA dengan JSON, tanpa markdown, tanpa penjelasan:
{{
  "action": "set",
  "message": "<isi reminder>",
  "remind_at_iso": "<ISO-8601 UTC datetime>",
  "cancel_id": null
}}
"""

# ── Affirmative patterns ─────────────────────────────────────────────────────

_AFFIRMATIVE_RE = re.compile(
    r"\b(iya|ya|yes|ok|oke|okee|boleh|silakan|lanjut|benar|betul|tentu|sure|yep|yup|gas|gaskeun|iyaa+|yaa+)\b",
    re.IGNORECASE,
)

_CLARIFICATION_MARKER = "apakah perlu saya bantu detailkan"


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
            is_office = task.current_mode == "office"

            # Check if the user is responding "yes" to our previous clarification prompt
            if self._is_affirmative(task.user_input):
                original_request = self._get_pre_clarification_request(task.session_id)
                if original_request:
                    reply = await self._handle_clarify_and_set(
                        task.session_id, original_request, is_office
                    )
                    task.mark_done(reply)
                    return task

            parsed = await self._parse_intent(task.user_input)
            action = parsed.get("action", "set")

            if action == "list":
                reply = self._handle_list(task.session_id, is_office)
            elif action == "cancel":
                cancel_id = parsed.get("cancel_id")
                reply = self._handle_cancel(task.session_id, cancel_id, is_office)
            elif action == "clarify":
                reply = self._clarification_question(is_office)
            else:
                reply = await self._handle_set(task.session_id, parsed, is_office)

            task.mark_done(reply)
        except Exception as exc:
            logger.exception("ReminderAgent error: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                "Aduh boss, ada error nih pas proses remindernya. Coba lagi ya! 🙏"
                if task.current_mode == "office"
                else "Maaf, terjadi kesalahan saat memproses reminder. Silakan coba lagi."
            )

        return task

    # ── Clarification helpers ─────────────────────────────────────────────────

    @staticmethod
    def _clarification_question(is_office: bool) -> str:
        if is_office:
            return (
                "Hmm, kayaknya requestnya kurang lengkap nih boss 🤔\n"
                "Apakah perlu saya bantu detailkan task dan buatkan remindernya boss?"
            )
        return (
            "Permintaan reminder Anda kurang lengkap.\n"
            "Apakah perlu saya bantu detailkan task dan buatkan remindernya boss?"
        )

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        """Return True if the text looks like a 'yes' response."""
        return bool(_AFFIRMATIVE_RE.search(text.strip()))

    def _get_pre_clarification_request(self, session_id: str) -> str | None:
        """Return the user message that preceded our last clarification question.

        Searches backwards through history for the clarification marker in an
        assistant message, then returns the user message just before it.
        """
        messages = self._history.get_as_llm_messages(session_id)
        # messages[-1] is the current user message (just added by orchestrator)
        # Walk backwards to find the clarification assistant message
        for i in range(len(messages) - 2, -1, -1):
            msg = messages[i]
            if msg["role"] == "assistant" and _CLARIFICATION_MARKER in msg["content"].lower():
                # The user message before this assistant message is what we want
                if i > 0 and messages[i - 1]["role"] == "user":
                    return messages[i - 1]["content"]
        return None

    async def _handle_clarify_and_set(
        self, session_id: str, original_request: str, is_office: bool
    ) -> str:
        """Use LLM to elaborate on a vague request and create the reminder."""
        now_utc = datetime.now(timezone.utc)
        now_wib = now_utc + timedelta(hours=7)
        now_wib_str = now_wib.strftime("%A, %d %B %Y %H:%M WIB")

        system_prompt = _ELABORATE_SYSTEM_PROMPT.format(now_wib=now_wib_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Permintaan awal: {original_request}"},
        ]
        raw = await self._llm.chat(messages)
        parsed = self._extract_json(raw)
        return await self._handle_set(session_id, parsed, is_office)

    # ── Action handlers ───────────────────────────────────────────────────────

    async def _handle_set(self, chat_id: str, parsed: dict, is_office: bool = False) -> str:
        message = parsed.get("message")
        remind_at_iso = parsed.get("remind_at_iso")

        if not message or not remind_at_iso:
            if is_office:
                return (
                    "⚠️ Hmm boss, kurang jelas nih detail remindernya.\n"
                    "Coba tulis gini ya:\n"
                    "• *ingatkan saya meeting jam 14:00*\n"
                    "• *remind me to take medicine at 08:00 tomorrow*"
                )
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
                if is_office:
                    return "⚠️ Aduh boss, waktunya udah lewat nih. Pilih waktu yang akan datang ya!"
                return "⚠️ Waktu reminder sudah lewat. Silakan tentukan waktu di masa mendatang."

        # Save to DB
        reminder = self._store.add(chat_id, message, remind_at)

        # Schedule the job
        from src.agents.reminder_agent.scheduler import schedule_reminder
        await schedule_reminder(reminder)

        # Format display time in WIB (UTC+7)
        wib_time = remind_at + timedelta(hours=7)
        time_str = wib_time.strftime("%d %B %Y pukul %H:%M WIB")

        if is_office:
            return (
                f"✅ Siap boss! Remindernya udah dibuat nih 🎉\n\n"
                f"📌 *#{reminder.id}* — {reminder.message}\n"
                f"⏰ {time_str}"
            )
        return (
            f"✅ Reminder berhasil dibuat!\n\n"
            f"📌 *#{reminder.id}* — {reminder.message}\n"
            f"⏰ {time_str}"
        )

    def _handle_list(self, chat_id: str, is_office: bool = False) -> str:
        reminders = self._store.list_pending(chat_id)
        if not reminders:
            if is_office:
                return "📭 Kosong boss, belum ada reminder aktif nih."
            return "📭 Tidak ada reminder aktif."

        if is_office:
            lines = ["📋 *Reminder Aktif Lo Boss:*\n"]
        else:
            lines = ["📋 *Daftar Reminder Aktif:*\n"]

        for r in reminders:
            wib_time = r.remind_at + timedelta(hours=7)
            time_str = wib_time.strftime("%d %b %Y %H:%M WIB")
            lines.append(f"• *#{r.id}* — {r.message}\n  ⏰ {time_str}")

        lines.append(f"\n_Total: {len(reminders)} reminder aktif_")
        if is_office:
            lines.append("_Mau hapus? Ketik \"hapus reminder #[id]\" ya boss_")
        else:
            lines.append("_Untuk menghapus: \"hapus reminder #[id]\"_")
        return "\n".join(lines)

    def _handle_cancel(self, chat_id: str, cancel_id, is_office: bool = False) -> str:
        if cancel_id is None:
            if is_office:
                return (
                    "⚠️ ID remindernya mana boss? "
                    "Ketik `/list reminder` dulu buat liat daftarnya, "
                    "terus tulis \"hapus reminder #[nomor]\" ya."
                )
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
        from src.agents.reminder_agent.scheduler import cancel_scheduled_reminder
        cancel_scheduled_reminder(rid)

        deleted = self._store.delete(rid, chat_id)
        if deleted:
            if is_office:
                return f"🗑️ Beres boss! Reminder *#{rid}* udah dihapus."
            return f"🗑️ Reminder *#{rid}* berhasil dihapus."
        if is_office:
            return (
                f"⚠️ Reminder *#{rid}* gak ketemu boss, "
                "atau bukan punya lo, atau udah terkirim."
            )
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
