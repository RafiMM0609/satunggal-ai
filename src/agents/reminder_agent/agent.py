"""
ReminderAgent – set, list, and cancel timed reminders delivered via Telegram.

Capabilities:
1. Set reminder  – propose a detailed reminder suggestion (with conflict check),
                   wait for user confirmation, then store in SQLite and schedule
                   an APScheduler job to send via Telegram.
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
from src.tools.reminder_store import (
    get_reminder_store,
    set_pending_suggestion,
    get_pending_suggestion,
    clear_pending_suggestion,
)

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

# ── System prompt for generating a detailed reminder suggestion ───────────────

_SUGGEST_SYSTEM_PROMPT = """\
Kamu adalah asisten pintar yang membantu membuat reminder yang efektif dan terperinci.

Hari dan waktu saat ini (UTC+7 / WIB): {now_wib}

Reminder aktif pengguna saat ini:
{existing_reminders}

Berdasarkan permintaan pengguna, buatlah 1 hingga 3 SARAN reminder yang berbeda-beda sehingga pengguna bisa memilih yang paling sesuai.
Setiap saran harus memiliki variasi yang bermakna — misalnya: waktu berbeda, fokus berbeda, atau tingkat detail yang berbeda.

Untuk setiap saran:
1. Perjelas judul/isi reminder agar spesifik dan actionable (bukan hanya kata kunci)
2. Tentukan waktu yang masuk akal — konversi ke UTC untuk remind_at_iso
3. Tambahkan "notes" berisi tips atau catatan berguna jika relevan, atau null jika tidak perlu
4. Jika ini event penting (meeting, interview, presentasi, penerbangan, ujian, deadline), sertakan "prep_reminder" sekitar 30 menit sebelumnya; jika tidak relevan, set ke null

Selain itu:
5. Periksa reminder aktif — tandai di "conflicts" jika waktunya berdekatan (±1 jam dari salah satu saran baru)

Balas HANYA dengan JSON, tanpa markdown, tanpa penjelasan:
{{
  "suggestions": [
    {{
      "message": "<isi reminder yang jelas dan spesifik>",
      "remind_at_iso": "<ISO-8601 UTC datetime>",
      "notes": "<catatan berguna atau null>",
      "prep_reminder": {{
        "message": "<isi reminder persiapan atau null>",
        "remind_at_iso": "<ISO-8601 UTC datetime atau null>"
      }}
    }}
  ],
  "conflicts": [
    {{"id": <int>, "message": "<isi reminder>", "remind_at_wib": "<tanggal dan jam WIB>"}}
  ]
}}

Contoh untuk permintaan "ingatkan meeting besok":
{{
  "suggestions": [
    {{
      "message": "Meeting tim — persiapkan agenda dan materi",
      "remind_at_iso": "2026-04-11T02:00:00",
      "notes": "Pastikan koneksi internet stabil sebelum meeting",
      "prep_reminder": {{
        "message": "Persiapan meeting: cek agenda dan buka aplikasi video call",
        "remind_at_iso": "2026-04-11T01:30:00"
      }}
    }},
    {{
      "message": "Meeting tim besok pagi",
      "remind_at_iso": "2026-04-10T23:00:00",
      "notes": "Reminder malam hari agar bisa persiapan dari malam",
      "prep_reminder": null
    }},
    {{
      "message": "Meeting tim — 15 menit sebelum mulai",
      "remind_at_iso": "2026-04-11T01:45:00",
      "notes": null,
      "prep_reminder": null
    }}
  ],
  "conflicts": []
}}
"""

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

# ── Affirmative / negative patterns ──────────────────────────────────────────

_AFFIRMATIVE_RE = re.compile(
    r"\b(iya|ya|yes|ok|oke|okee|boleh|silakan|lanjut|benar|betul|tentu|sure|yep|yup|gas|gaskeun|iyaa+|yaa+|setuju|acc|konfirmasi|buat|buatkan)\b",
    re.IGNORECASE,
)

_NEGATIVE_RE = re.compile(
    r"\b(tidak|nggak|ngga|gak|ga|no|jangan|batal|cancel|batalkan|stop|tidak jadi|nope|ndak|nda)\b",
    re.IGNORECASE,
)

# Matches "pilih 1", "opsi 2", "option 3", "nomor 1", "no 2", or a bare digit word "1"/"2"/"3"
_SELECTION_RE = re.compile(
    r"(?:pilih|opsi|option|nomor|no\.?)\s*([1-9])|\b([1-9])\b",
    re.IGNORECASE,
)

# Matches "semua", "all", "semuanya" — confirm all suggestions
_ALL_RE = re.compile(r"\b(semua|semuanya|all)\b", re.IGNORECASE)

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
            user_input = task.user_input.strip()

            # ── Check for a pending confirmation (suggestion was already shown) ──
            pending = get_pending_suggestion(task.session_id)
            if pending:
                selected = self._parse_selection(user_input, len(pending.get("suggestions", [])))
                if selected is not None:
                    # selected is a list of 0-based indices
                    reply = await self._confirm_pending(
                        task.session_id, pending, is_office, selected
                    )
                    task.mark_done(reply)
                    return task
                if self._is_negative(user_input):
                    clear_pending_suggestion(task.session_id)
                    reply = (
                        "Oke boss, saran remindernya dibatalkan. "
                        "Kasih tau aja kalau mau buat reminder baru ya! 😊"
                        if is_office
                        else "Baik, saran reminder dibatalkan. "
                        "Silakan beri tahu jika ingin membuat reminder baru."
                    )
                    task.mark_done(reply)
                    return task
                # User sent something else — treat as a new/modified request.
                # Clear the stale pending suggestion so it doesn't interfere
                # with the fresh request that follows.
                clear_pending_suggestion(task.session_id)

            # ── Check for affirmative to the old clarification prompt ────────
            if self._is_affirmative(user_input):
                original_request = self._get_pre_clarification_request(task.session_id)
                if original_request:
                    reply = await self._handle_clarify_and_suggest(
                        task.session_id, original_request, is_office
                    )
                    task.mark_done(reply)
                    return task

            # ── Parse intent ─────────────────────────────────────────────────
            parsed = await self._parse_intent(user_input)
            action = parsed.get("action", "set")

            if action == "list":
                reply = self._handle_list(task.session_id, is_office)
            elif action == "cancel":
                cancel_id = parsed.get("cancel_id")
                reply = self._handle_cancel(task.session_id, cancel_id, is_office)
            elif action == "clarify":
                reply = self._clarification_question(is_office)
            else:
                # New flow: suggest details and ask for confirmation
                reply = await self._handle_suggest(task.session_id, parsed, is_office)

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
            "Apakah perlu saya bantu detailkan task dan buatkan remindernya?"
        )

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        return bool(_AFFIRMATIVE_RE.search(text.strip()))

    @staticmethod
    def _is_negative(text: str) -> bool:
        return bool(_NEGATIVE_RE.search(text.strip()))

    @staticmethod
    def _parse_selection(text: str, num_suggestions: int) -> list[int] | None:
        """Parse user input to determine which suggestion(s) to confirm.

        Returns a list of 0-based indices if the user made a selection or said
        "ya"/"semua".  Returns None if the input is not a recognisable selection.

        Rules:
        - "ya" / affirmative (single suggestion or "all") → [0] or all indices
        - "semua" / "all"  → all indices [0..n-1]
        - "pilih 1" / "1"  → [0]
        - "pilih 2,3"      → [1, 2]
        """
        stripped = text.strip()

        # "semua" / "all" → confirm every suggestion
        if _ALL_RE.search(stripped):
            return list(range(num_suggestions))

        # plain affirmative → confirm all (keeps backwards-compat for single suggestion)
        if _AFFIRMATIVE_RE.search(stripped):
            return list(range(num_suggestions))

        # numbered selection, e.g. "pilih 1", "opsi 2", bare "1"
        found = [int(m.group(1) or m.group(2)) for m in _SELECTION_RE.finditer(stripped)]
        if found:
            # convert 1-based to 0-based, clamp to valid range
            indices = sorted({n - 1 for n in found if 1 <= n <= num_suggestions})
            return indices if indices else None

        return None

    def _get_pre_clarification_request(self, session_id: str) -> str | None:
        """Return the user message that preceded our last clarification question.

        Searches backwards through history for the clarification marker in an
        assistant message, then returns the user message just before it.
        Skips the most-recent user message (the current affirmative reply).
        """
        messages = self._history.get_as_llm_messages(session_id)
        if not messages:
            return None

        # Skip the last message if it is the current user turn (the affirmative reply)
        search_end = len(messages)
        if messages[-1]["role"] == "user":
            search_end -= 1

        for i in range(search_end - 1, -1, -1):
            msg = messages[i]
            if msg["role"] == "assistant" and _CLARIFICATION_MARKER in msg["content"].lower():
                # The user message before this assistant message is what we want
                if i > 0 and messages[i - 1]["role"] == "user":
                    return messages[i - 1]["content"]
        return None

    # ── Suggestion flow ───────────────────────────────────────────────────────

    async def _handle_suggest(
        self, session_id: str, parsed: dict, is_office: bool
    ) -> str:
        """Ask LLM to generate 1-3 reminder suggestions, check conflicts,
        store them as pending, and return formatted options for user to choose."""
        now_utc = datetime.now(timezone.utc)
        now_wib = now_utc + timedelta(hours=7)
        now_wib_str = now_wib.strftime("%A, %d %B %Y %H:%M WIB")

        # Format existing reminders for LLM context
        existing = self._store.list_pending(session_id)
        if existing:
            existing_lines = []
            for r in existing:
                wib = r.remind_at + timedelta(hours=7)
                existing_lines.append(
                    f"- #{r.id}: \"{r.message}\" → {wib.strftime('%A, %d %b %Y %H:%M WIB')}"
                )
            existing_str = "\n".join(existing_lines)
        else:
            existing_str = "(tidak ada reminder aktif)"

        system_prompt = _SUGGEST_SYSTEM_PROMPT.format(
            now_wib=now_wib_str,
            existing_reminders=existing_str,
        )

        # Build a combined user request from parsed data
        parts = []
        if parsed.get("message"):
            parts.append(f"Isi: {parsed['message']}")
        if parsed.get("remind_at_iso"):
            parts.append(f"Waktu (UTC): {parsed['remind_at_iso']}")
        user_content = "; ".join(parts) if parts else "Buat reminder baru"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = await self._llm.chat(messages)
        result = self._extract_json(raw)

        suggestions_raw = result.get("suggestions") or []
        conflicts = result.get("conflicts") or []

        # Validate and normalise each suggestion; skip invalid ones
        valid_suggestions: list[dict] = []
        for s in suggestions_raw:
            msg = s.get("message")
            iso = s.get("remind_at_iso")
            if not msg or not iso:
                continue
            try:
                dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if dt <= now_utc:
                dt += timedelta(days=1)
                if dt <= now_utc:
                    continue
                iso = dt.replace(tzinfo=None).isoformat()

            # Validate prep reminder if present
            prep = s.get("prep_reminder") or {}
            valid_prep: dict | None = None
            if prep.get("message") and prep.get("remind_at_iso"):
                try:
                    prep_dt = datetime.fromisoformat(prep["remind_at_iso"]).replace(
                        tzinfo=timezone.utc
                    )
                    if prep_dt > now_utc:
                        valid_prep = {
                            "message": prep["message"],
                            "remind_at_iso": prep["remind_at_iso"],
                        }
                except (ValueError, TypeError):
                    pass

            valid_suggestions.append({
                "message": msg,
                "remind_at_iso": iso,
                "notes": s.get("notes"),
                "prep": valid_prep,
            })

        if not valid_suggestions:
            if is_office:
                return (
                    "⚠️ Hmm boss, saya kesulitan memahami detail remindernya.\n"
                    "Coba tulis lebih spesifik ya, contoh:\n"
                    "• *ingatkan saya meeting besok jam 10:00*\n"
                    "• *reminder minum obat setiap hari jam 08:00*"
                )
            return (
                "⚠️ Saya kesulitan memahami detail reminder Anda.\n"
                "Coba tulis lebih spesifik, contoh:\n"
                "• *ingatkan saya untuk meeting besok jam 10:00*\n"
                "• *reminder minum obat setiap hari jam 08:00*"
            )

        # Build response lines
        lines: list[str] = []
        is_multi = len(valid_suggestions) > 1
        if is_office:
            header = (
                "🔔 *Berikut beberapa saran reminder boss, pilih yang paling cocok:*\n"
                if is_multi
                else "🔔 *Berikut saran reminder yang saya siapkan boss:*\n"
            )
        else:
            header = (
                "🔔 *Berikut beberapa saran reminder, pilih yang paling sesuai:*\n"
                if is_multi
                else "🔔 *Berikut saran reminder dari saya:*\n"
            )
        lines.append(header)

        for idx, s in enumerate(valid_suggestions, start=1):
            dt = datetime.fromisoformat(s["remind_at_iso"]).replace(tzinfo=timezone.utc)
            wib = dt + timedelta(hours=7)
            time_str = wib.strftime("%A, %d %B %Y pukul %H:%M WIB")

            if is_multi:
                lines.append(f"*Opsi {idx}*")
            lines.append(f"📌 *Judul:* {s['message']}")
            lines.append(f"📅 *Waktu:* {time_str}")
            if s.get("notes"):
                lines.append(f"📝 *Catatan:* {s['notes']}")
            if s.get("prep"):
                prep_dt = datetime.fromisoformat(s["prep"]["remind_at_iso"]).replace(
                    tzinfo=timezone.utc
                )
                prep_wib = prep_dt + timedelta(hours=7)
                prep_time_str = prep_wib.strftime("%A, %d %B %Y pukul %H:%M WIB")
                lines.append(
                    f"💡 *Reminder Persiapan:* \"{s['prep']['message']}\"\n"
                    f"   ⏰ {prep_time_str}"
                )
            if is_multi and idx < len(valid_suggestions):
                lines.append("")  # blank line between options

        # Conflict warnings
        if conflicts:
            lines.append("")
            for c in conflicts:
                lines.append(
                    f"⚠️ *Jadwal bertabrakan:* #{c.get('id')} \"{c.get('message')}\" "
                    f"— {c.get('remind_at_wib')}"
                )

        lines.append("")
        if is_multi:
            if is_office:
                lines.append(
                    "Balas *pilih 1*, *pilih 2*, dll untuk memilih satu opsi, "
                    "atau *semua* untuk buat semua reminder sekaligus boss 😊\n"
                    "Ketik *batal* untuk membatalkan."
                )
            else:
                lines.append(
                    "Balas *pilih 1*, *pilih 2*, dst untuk memilih opsi tertentu, "
                    "atau *semua* untuk membuat semua reminder.\n"
                    "Ketik *batal* untuk membatalkan."
                )
        else:
            if is_office:
                lines.append("Balas *ya* untuk membuat reminder ini, atau jelaskan perubahannya boss 😊")
            else:
                lines.append(
                    "Balas *ya* untuk membuat reminder ini, "
                    "atau jelaskan perubahan yang diinginkan."
                )

        # Persist suggestions as pending
        set_pending_suggestion(session_id, {"suggestions": valid_suggestions})

        return "\n".join(lines)

    async def _confirm_pending(
        self,
        session_id: str,
        pending: dict,
        is_office: bool,
        selected_indices: list[int] | None = None,
    ) -> str:
        """Create reminder(s) for the selected suggestion indices.

        selected_indices: 0-based list; None or empty = all suggestions.
        """
        clear_pending_suggestion(session_id)

        suggestions = pending.get("suggestions", [])
        if selected_indices is None:
            selected_indices = list(range(len(suggestions)))

        reply_parts: list[str] = []
        for idx in selected_indices:
            if idx < 0 or idx >= len(suggestions):
                continue
            s = suggestions[idx]
            primary_reply = await self._handle_set(session_id, s, is_office)
            reply_parts.append(primary_reply)
            if s.get("prep"):
                prep_reply = await self._handle_set(
                    session_id, s["prep"], is_office, is_prep=True
                )
                reply_parts.append(prep_reply)

        return "\n\n".join(reply_parts) if reply_parts else (
            "⚠️ Tidak ada reminder yang dibuat." if not is_office
            else "⚠️ Gak ada reminder yang dibuat boss."
        )

    async def _handle_clarify_and_suggest(
        self, session_id: str, original_request: str, is_office: bool
    ) -> str:
        """Use LLM to elaborate on a vague request, then present as a suggestion."""
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
        return await self._handle_suggest(session_id, parsed, is_office)

    # ── Action handlers ───────────────────────────────────────────────────────

    async def _handle_set(
        self,
        chat_id: str,
        parsed: dict,
        is_office: bool = False,
        is_prep: bool = False,
    ) -> str:
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

        label = "Reminder persiapan" if is_prep else "Reminder"
        if is_office:
            return (
                f"✅ Siap boss! {label}nya udah dibuat nih 🎉\n\n"
                f"📌 *#{reminder.id}* — {reminder.message}\n"
                f"⏰ {time_str}"
            )
        return (
            f"✅ {label} berhasil dibuat!\n\n"
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
