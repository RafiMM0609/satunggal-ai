"""
ReminderAgent – set, list, and cancel timed reminders delivered via Telegram.

Using the Hermes ReAct loop pattern, this agent can inspect active reminders,
detect schedule conflicts, interact with the user via a natural conversational loop,
and manage permanent user preferences using a Long-Term User Profile Store.
"""

from __future__ import annotations

import logging
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.history import ConversationHistory
from src.memory.state import AgentTask
from src.tools.reminder_store import get_reminder_store
from src.tools.user_profile_store import get_user_profile_store

logger = logging.getLogger(__name__)

# Matches <think>…</think> or <thinking>…</thinking> blocks produced by reasoning models.
_THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)

_HERMES_REMINDER_DECISION_PROMPT = """\
Kamu adalah **Asisten Pengingat Pribadi Mandiri (Hermes Reminder Agent)** yang chill, asik, dan sangat fleksibel.
Tugas kamu adalah membantu pengguna mengelola pengingat (reminder) mereka secara alami melalui percakapan, layaknya asisten manusia asli.

Kamu memiliki akses ke database pengingat aktif, riwayat chat, dan profil preferensi jangka panjang pengguna. Kamu memiliki beberapa alat (tools) berikut:
1. `get_current_time`: Mendapatkan waktu dan hari saat ini (dalam format WIB dan UTC). Gunakan ini sebagai referensi utama untuk menghitung waktu relatif seperti "besok", "lusa", "Senin jam 10 pagi", "minggu depan", dll.
   Parameter: Tidak ada.
2. `list_reminders`: Mengembalikan daftar pengingat aktif/belum terkirim milik pengguna ini. Gunakan ini untuk melihat jadwal mereka atau mendeteksi tabrakan waktu.
   Parameter: Tidak ada.
3. `add_reminder`: Membuat pengingat baru di waktu tertentu.
   Parameter:
     - `message` (string, isi pengingat yang spesifik dan jelas)
     - `remind_at_iso` (string, format ISO-8601 UTC datetime string)
4. `cancel_reminder`: Membatalkan/menghapus pengingat aktif berdasarkan ID reminder.
   Parameter:
     - `reminder_id` (integer)
5. `get_user_profile`: Membaca seluruh informasi profil, nama panggilan, aturan, atau preferensi penting pengguna yang tersimpan secara jangka panjang (permanen).
   Parameter: Tidak ada.
6. `update_user_profile`: Menyimpan atau memperbarui informasi profil, aturan khusus, atau fakta jangka panjang penting tentang pengguna.
   Parameter:
     - `profile_key` (string, kategori informasi, misal: `preferred_name`, `rules`, `routines`, `relationships`, `locations`)
     - `profile_value` (string, nilai/detail preferensi dalam format teks biasa atau JSON)
7. `answer`: Memberikan tanggapan akhir kepada pengguna. Gunakan tindakan ini untuk menyapa, bertanya balik, mengonfirmasi, memberikan saran, menjelaskan konflik jadwal, atau membalas secara ramah dan santai (sesuai Contextual Vibe Awareness).
   Parameter:
     - `content` (string, pesan yang akan dikirim ke pengguna)

Format Output Wajib:
Kamu harus membalas dalam format JSON yang valid. Jangan sertakan markdown atau penjelasan di luar JSON tersebut.
Struktur JSON:
{
  "thought": "Pemikiranmu tentang apa yang diinginkan user, analisis waktu saat ini, konflik jadwal, atau rencana langkah selanjutnya.",
  "action": "Nama tindakan yang dipilih ('get_current_time', 'list_reminders', 'add_reminder', 'cancel_reminder', 'get_user_profile', 'update_user_profile', atau 'answer').",
  "message": "Pesan reminder (hanya diisi jika action adalah 'add_reminder').",
  "remind_at_iso": "Waktu UTC ISO-8601 (hanya diisi jika action adalah 'add_reminder').",
  "reminder_id": <integer atau null jika action adalah 'cancel_reminder'>,
  "profile_key": "Kunci profil (hanya diisi jika action adalah 'update_user_profile')",
  "profile_value": "Nilai profil (hanya diisi jika action adalah 'update_user_profile')",
  "content": "Pesan balasan ke user dalam bahasa santai/chill (hanya diisi jika action adalah 'answer')."
}

PANDUAN INTERAKSI & MEMORI:
- **Langkah Pertama:** Pada langkah pertama percakapan (Step 1), jika kamu belum memuat profil pengguna, gunakan `get_user_profile` terlebih dahulu agar kamu tahu nama panggilan kesukaan mereka dan aturan khusus yang mereka tetapkan.
- **Deteksi Preferensi Baru:** Jika pengguna memberikan instruksi khusus atau fakta baru yang penting untuk diingat selamanya (misal: "Mulai sekarang panggil gue Boss", "Selalu buat reminder persiapan 30 menit sebelum meeting"), simpan segera menggunakan `update_user_profile` sebelum membuat pengingat atau menjawab.
- **Deteksi Konflik:** Jika pengguna meminta reminder di waktu yang berdekatan dengan reminder yang sudah ada (selisih kurang dari 1 jam), beri tahu mereka secara ramah dan tanyakan apakah ingin tetap diset atau disesuaikan.
- **Konfirmasi Otomatis vs Tanya Balik:**
  - Jika detailnya sudah sangat jelas (misal: "ingetin meeting besok jam 10 pagi"), buatlah reminder-nya (`add_reminder`), lalu gunakan `answer` untuk mengonfirmasi bahwa sudah berhasil dibuat.
  - Jika ada yang kurang jelas atau tidak lengkap, tanyakan balik secara asik.
  - Jika ada konfirmasi (misal: "oke set aja", "cancel yang tadi"), periksa riwayat percakapan untuk menentukan reminder mana yang dimaksud.
"""


class ReminderAgent(BaseAgent):
    """Manages timed reminders stored in SQLite and scheduled via APScheduler using a Hermes ReAct loop."""

    name = "reminder_agent"

    role = "Asisten Pengingat Pribadi (Reminder Assistant)"
    goal = "Membantu pengguna mengelola, menjadwalkan, dan membatalkan pengingat secara alami, asik, dan fleksibel."
    backstory = (
        "Kamu adalah asisten pengingat yang sangat ramah, santai, dan pengertian. "
        "Kamu berbicara dengan gaya bahasa santai/casual (slang/Gen Z atau gabungan Indonesia-Inggris yang asik). "
        "Tugas utama kamu adalah memastikan pengguna tidak melewatkan jadwal mereka, "
        "sembari memastikan jadwal yang dibuat bebas dari bentrokan/konflik."
    )

    _MAX_HERMES_STEPS = 5

    def __init__(
        self,
        history: ConversationHistory,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self._history = history
        self._llm = llm or LLMClient()
        self._store = get_reminder_store()
        self._profile_store = get_user_profile_store()

    async def run(self, task: AgentTask) -> AgentTask:
        try:
            history_messages = self._history.get_as_llm_messages(task.session_id)
            reply = await self._run_hermes_loop(
                query=task.user_input,
                session_id=task.session_id,
                history_messages=history_messages
            )
            task.mark_done(reply)
        except Exception as exc:
            logger.exception("ReminderAgent error: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan saat memproses reminder. Silakan coba lagi."

        return task

    async def _run_hermes_loop(
        self,
        query: str,
        session_id: str,
        history_messages: list[dict] | None = None
    ) -> str:
        """Run the Hermes ReAct loop to dynamically process and manage reminders."""
        persona = self.get_persona_prompt()
        system_prompt = _HERMES_REMINDER_DECISION_PROMPT
        if persona:
            system_prompt = persona + "\n\n" + system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            # Keep last 8 turns for history context to prevent bloating
            messages.extend(history_messages[-8:])

        # If history is empty or the last message doesn't match current query, append it
        if not history_messages or history_messages[-1]["content"] != query:
            messages.append({"role": "user", "content": query})

        step = 0
        final_answer = ""

        while step < self._MAX_HERMES_STEPS:
            step += 1
            logger.info(
                "ReminderAgent Hermes Loop: Step %d/%d for session=%s",
                step, self._MAX_HERMES_STEPS, session_id
            )

            try:
                raw_response = await self._llm.chat(
                    messages,
                    max_tokens=2048,
                    json_mode=True
                )

                cleaned_response = _THINK_TAG_RE.sub("", raw_response).strip()

                try:
                    action_data = json.loads(cleaned_response)
                except Exception as exc:
                    logger.warning("Failed to parse JSON: %s. Raw: %r", exc, cleaned_response)
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', cleaned_response)
                    action = action_match.group(1) if action_match else "answer"

                    content_match = re.search(r'"content"\s*:\s*"(.*)"', cleaned_response, re.DOTALL)
                    content = content_match.group(1) if content_match else ""

                    action_data = {
                        "thought": "Failed to parse JSON cleanly.",
                        "action": action,
                        "content": content
                    }

                thought = action_data.get("thought", "")
                action = action_data.get("action", "answer")
                logger.info(
                    "ReminderAgent Step %d: Thought: %s | Action: %s",
                    step, thought, action
                )

                messages.append({"role": "assistant", "content": raw_response})

                if action == "answer":
                    final_answer = action_data.get("content", "")
                    break

                elif action == "get_current_time":
                    now_utc = datetime.now(timezone.utc)
                    now_wib = now_utc + timedelta(hours=7)
                    tool_output = (
                        f"Current Date/Time:\n"
                        f"- UTC: {now_utc.isoformat(timespec='seconds')}\n"
                        f"- WIB (Local): {now_wib.strftime('%A, %d %B %Y %H:%M:%S WIB')}"
                    )
                    messages.append({
                        "role": "user",
                        "content": f"[Hasil get_current_time]\n{tool_output}"
                    })

                elif action == "list_reminders":
                    reminders = self._store.list_pending(session_id)
                    if not reminders:
                        tool_output = "No pending reminders."
                    else:
                        lines = []
                        for r in reminders:
                            wib_time = r.remind_at + timedelta(hours=7)
                            lines.append(
                                f"- ID #{r.id}: \"{r.message}\" at {wib_time.strftime('%Y-%m-%d %H:%M')} WIB"
                            )
                        tool_output = "\n".join(lines)
                    messages.append({
                        "role": "user",
                        "content": f"[Hasil list_reminders]\n{tool_output}"
                    })

                elif action == "add_reminder":
                    message_text = action_data.get("message")
                    remind_at_iso = action_data.get("remind_at_iso")

                    if not message_text or not remind_at_iso:
                        tool_output = "Error: parameters 'message' and 'remind_at_iso' are required."
                    else:
                        try:
                            # Normalize ISO format
                            iso_str = remind_at_iso.strip()
                            if iso_str.endswith('Z'):
                                iso_str = iso_str[:-1] + '+00:00'
                            dt = datetime.fromisoformat(iso_str)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            else:
                                dt = dt.astimezone(timezone.utc)

                            # Save to DB
                            reminder = self._store.add(session_id, message_text, dt)

                            # Register scheduler job
                            from src.agents.reminder_agent.scheduler import schedule_reminder
                            await schedule_reminder(reminder)

                            wib_time = dt + timedelta(hours=7)
                            tool_output = (
                                f"Success: Reminder ID #{reminder.id} created for "
                                f"\"{reminder.message}\" at {wib_time.strftime('%A, %d %B %Y %H:%M WIB')}."
                            )
                        except Exception as exc:
                            tool_output = f"Error adding reminder: {exc}"

                    messages.append({
                        "role": "user",
                        "content": f"[Hasil add_reminder]\n{tool_output}"
                    })

                elif action == "cancel_reminder":
                    reminder_id_val = action_data.get("reminder_id")
                    if reminder_id_val is None:
                        tool_output = "Error: parameter 'reminder_id' is required."
                    else:
                        try:
                            rid = int(reminder_id_val)
                            from src.agents.reminder_agent.scheduler import cancel_scheduled_reminder
                            cancel_scheduled_reminder(rid)
                            deleted = self._store.delete(rid, session_id)
                            if deleted:
                                tool_output = f"Success: Reminder ID #{rid} cancelled."
                            else:
                                tool_output = f"Error: Reminder ID #{rid} not found or not owned by you."
                        except Exception as exc:
                            tool_output = f"Error cancelling reminder: {exc}"

                    messages.append({
                        "role": "user",
                        "content": f"[Hasil cancel_reminder]\n{tool_output}"
                    })

                elif action == "get_user_profile":
                    prefs = self._profile_store.get_all_preferences(session_id)
                    tool_output = json.dumps(prefs, indent=2)
                    messages.append({
                        "role": "user",
                        "content": f"[Hasil get_user_profile]\n{tool_output}"
                    })

                elif action == "update_user_profile":
                    key = action_data.get("profile_key")
                    val = action_data.get("profile_value")
                    if not key or val is None:
                        tool_output = "Error: parameters 'profile_key' and 'profile_value' are required."
                    else:
                        self._profile_store.set_preference(session_id, key, val)
                        tool_output = f"Success: Updated user profile preference '{key}'."
                    messages.append({
                        "role": "user",
                        "content": f"[Hasil update_user_profile]\n{tool_output}"
                    })

                else:
                    messages.append({
                        "role": "user",
                        "content": f"Error: action '{action}' is invalid. Choose from: get_current_time, list_reminders, add_reminder, cancel_reminder, get_user_profile, update_user_profile, answer."
                    })
            except Exception as exc:
                logger.error("Error in ReminderAgent Hermes step: %s", exc)
                messages.append({
                    "role": "user",
                    "content": f"Terjadi kesalahan internal: {exc}. Silakan selesaikan dengan action 'answer'."
                })
                if step >= self._MAX_HERMES_STEPS - 1:
                    break

        if not final_answer:
            logger.info("ReminderAgent: forcing final answer generation")
            force_prompt = (
                "Langkah maksimum ReAct loop telah tercapai. Kamu harus segera memberikan jawaban akhir "
                "kepada pengguna menggunakan tindakan 'answer' dan parameter 'content'."
            )
            messages.append({"role": "user", "content": force_prompt})
            try:
                raw_response = await self._llm.chat(
                    messages,
                    max_tokens=1024,
                    json_mode=True
                )
                cleaned = _THINK_TAG_RE.sub("", raw_response).strip()
                action_data = json.loads(cleaned)
                final_answer = action_data.get("content", "")
                if not final_answer:
                    final_answer = "Maaf, saya kesulitan memproses permintaan reminder Anda saat ini."
            except Exception as exc:
                logger.error("Failed to generate forced final answer: %s", exc)
                final_answer = "Maaf, saya kesulitan memproses permintaan reminder Anda saat ini."

        return final_answer
