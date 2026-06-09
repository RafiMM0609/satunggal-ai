"""
ResponderAgent – general-purpose conversational agent.

Handles:  general_inquiry, product_question, complaint, order_status,
          technical_support, billing, image_query, unknown
Uses:     Conversation history + LLM to produce a contextual reply.

Note: This agent does NOT use Tavily web search. Live web search is
exclusively available to ResearcherAgent (triggered by the 'research' intent).
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
from src.tools.telegram_formatter import sanitize_for_telegram

# NOTE ON SHARED DATABASE (reminders.db):
# ResponderAgent utilizes the UserProfileStore which persists data in 'reminders.db'.
# We share the 'reminders.db' SQLite file with ReminderAgent and ResearcherAgent to maintain a unified user profile 
# (e.g. 'preferred_name', 'preferred_vibe', timezone offset) across all agents in the platform.
# This prevents database fragmentation, reduces DB file management, and ensures a consistent 
# user experience across the different conversational agents.
from src.tools.user_profile_store import get_user_profile_store

logger = logging.getLogger(__name__)

# ── Slang / Gen Z marker sets ─────────────────────────────────────────────────
_GENZ_FIRST_PERSON  = re.compile(r'\b(gue|gw|aku|w)\b', re.IGNORECASE)
_GENZ_SECOND_PERSON = re.compile(r'\b(lo|lu|kamu|elo)\b', re.IGNORECASE)
_GENZ_VOCAB         = re.compile(
    r'\b(cuy|bro|sis|gas|gaskeun|sikat|mantap|gercep|anjir|wkwk|hehe|btw|anw|'
    r'blunder|red\s*flag|spill|mager|kepo|kepoin|intip|teropong|cooking|'
    r'sat[- ]?set|ngl|fr|lol|omg|btw|fyi)\b',
    re.IGNORECASE,
)
_PANIC_MARKERS      = re.compile(r'\b(gawat|darurat|urgent|asap|crash)\b', re.IGNORECASE)
_PANIC_CAPS         = re.compile(r'\b[A-Z]{5,}\b')  # 5+ caps = likely screaming, not an acronym

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FORMAL = """\
Kamu adalah asisten AI yang ramah, profesional, dan membantu.
Jawab pertanyaan pengguna secara jelas dan ringkas.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris).
Jika kamu tidak tahu jawabannya, katakan dengan jujur.
"""

_SYSTEM_PROMPT_OFFICE = """\
Kamu adalah asisten AI yang gaul, santai, dan asik diajak ngobrol — khusus buat urusan kantor dan kerjaan.
Jawab pertanyaan pengguna secara jelas tapi tetap santai dan informal.
Sapa user sebagai "boss" sesekali biar lebih akrab.
Pakai kata-kata gaul Indonesia yang wajar: "nih", "dong", "sih", "yuk", "mantap", "siap", "gaskeun", "okee", dll.
Gunakan bahasa yang sama dengan pengguna (Indonesia atau Inggris), tapi tetap dengan nada santai.
Jika kamu tidak tahu jawabannya, bilang aja jujur dengan cara yang santai.
"""

_SYSTEM_PROMPT_GENZ = """\
Lo adalah asisten AI yang gaul, adaptive, dan asik — kayak temen nongkrong yang juga jago teknis.
Balas pake gaya bahasa yang sama kayak user: kalau dia pake "Gue/Lo", lo pake "Gue/Lo" juga.
Boleh pakai slang yang wajar: "cuy", "bro", "gas", "sat-set", "mantap", "nih", "dong", "gercep", dll.
Jawab substansinya dengan benar dan lengkap — gaya santai bukan alasan buat jawaban yang asal.
Kalau lo nggak tau, bilang jujur dengan cara yang asik, jangan drama.
Gunakan bahasa Indonesia santai (atau Inggris kalau user pakai Inggris).
"""

_SYSTEM_PROMPT_GENZ_PANIC = """\
Lo adalah asisten AI yang gaul, adaptive, dan asik — kayak temen nongkrong yang juga jago teknis.
User lagi panik atau butuh bantuan cepat. Responmu harus:
1. Menenangkan tapi langsung ke poin — jangan terlalu banyak basa-basi.
2. Tunjukin bahwa lo langsung handle situasi ini: "Tenang cuy, gue bantuin beresin ini sekarang, sat-set!"
3. Gunakan gaya bahasa yang sama kayak user: kalau dia pake "Gue/Lo", lo pake "Gue/Lo" juga.
Jawab substansinya dengan benar dan lengkap.
"""

# Keep old name as alias so existing imports still work
_SYSTEM_PROMPT = _SYSTEM_PROMPT_FORMAL

# Matches <think>…</think> or <thinking>…</thinking> blocks produced by reasoning models.
_THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    flags=re.DOTALL | re.IGNORECASE,
)

_HERMES_RESPONDER_DECISION_PROMPT = """\
Kamu adalah **Asisten Obrolan Pribadi Mandiri (Hermes Responder Agent)**. Tugas kamu adalah membalas obrolan pengguna secara ramah, kontekstual, dan dengan gaya bahasa (vibe) yang tepat.

Kamu memiliki akses ke database preferensi profil pengguna (Long-Term User Profile Store).

### GAYA BAHASA / PERSONA YANG TERSEDIA:

1. **FORMAL (Default / Professional)**
   - Gaya: Ramah, profesional, membantu, sopan.
   - Sapaan/Nada: Jelas, ringkas, menggunakan bahasa Indonesia baku/formal yang santai.

2. **OFFICE (Office/Work Mode)**
   - Gaya: Gaul kantoran, santai, asik diajak ngobrol untuk urusan kerjaan.
   - Sapaan/Nada: Sapa pengguna sebagai "boss" sesekali. Gunakan kata-kata: "nih", "dong", "sih", "siap", "gaskeun", "okee".

3. **GENZ (Gen Z / Slang Mode)**
   - Gaya: Gaul, adaptif, asik seperti teman nongkrong yang jago teknis.
   - Sapaan/Nada: Cerminkan panggilan pengguna. Jika pengguna memakai "Gue/Lo", gunakan "Gue/Lo". Gunakan slang wajar: "cuy", "bro", "gas", "sat-set", "mantap", "gercep".

4. **GENZ_PANIC (Gen Z Panic/Urgent Mode)**
   - Gaya: Tenang, sigap, solutif, langsung ke inti masalah tanpa banyak basa-basi.
   - Sapaan/Nada: "Tenang cuy, gue bantuin beresin ini sekarang, sat-set!". Cerminkan panggilan "Gue/Lo" jika cocok.

### ATURAN PEMILIHAN GAYA BAHASA (VIBE):
- Muat profil pengguna di Step 1 menggunakan tool `get_user_profile`.
- Periksa nilai preferensi `"preferred_vibe"` dari profil pengguna:
  - Jika `"preferred_vibe"` berharga `"formal"`, gunakan gaya **FORMAL**.
  - Jika `"preferred_vibe"` berharga `"office"`, gunakan gaya **OFFICE**.
  - Jika `"preferred_vibe"` berharga `"genz"`, gunakan gaya **GENZ** (atau **GENZ_PANIC** jika pengguna terdeteksi panik/urgent).
  - Jika `"preferred_vibe"` berharga `"auto"` (default), lakukan deteksi dinamis terhadap pesan terakhir pengguna:
    * Jika pengguna tampak panik/terburu-buru/menggunakan huruf kapital berlebihan/kata urgent: gunakan gaya **GENZ_PANIC**.
    * Jika pengguna menggunakan kata slang/Gen Z (seperti gue, lo, bro, cuy, wkwk, mager, kepo): gunakan gaya **GENZ**.
    * Jika tidak keduanya: gunakan gaya **FORMAL**.

### ALAT (TOOLS) YANG TERSEDIA:
1. `get_current_time`: Mendapatkan waktu dan hari saat ini (UTC & WIB). Gunakan ini jika pengguna menanyakan hari/waktu saat ini.
   Parameter: Tidak ada.
2. `get_user_profile`: Membaca seluruh preferensi profil jangka panjang pengguna.
   Parameter: Tidak ada.
3. `update_user_profile`: Memperbarui preferensi profil pengguna secara permanen.
   Parameter:
     - `profile_key` (string, kategori preferensi, misal: `preferred_name`, `preferred_vibe`)
     - `profile_value` (string, nilai baru, misal: untuk `preferred_vibe` isinya `formal`, `office`, `genz`, atau `auto`)
4. `answer`: Memberikan balasan akhir ke pengguna dengan gaya bahasa (vibe) yang telah ditentukan.
   Parameter:
     - `content` (string, isi balasan akhir)

Format Output Wajib:
Kamu harus membalas dalam format JSON yang valid. Jangan sertakan markdown/teks di luar JSON tersebut.
Struktur JSON:
{
  "thought": "Pemikiranmu tentang deteksi vibe, apa yang diinginkan user, pemuatan profil, atau rencana langkah selanjutnya.",
  "action": "Nama tindakan yang dipilih ('get_current_time', 'get_user_profile', 'update_user_profile', atau 'answer').",
  "profile_key": "Kunci profil (hanya diisi jika action adalah 'update_user_profile')",
  "profile_value": "Nilai profil (hanya diisi jika action adalah 'update_user_profile')",
  "content": "Isi pesan balasan akhir dengan gaya bahasa yang sesuai (hanya diisi jika action adalah 'answer')."
}

PENTING:
- **Langkah Pertama:** Pada langkah pertama percakapan (Step 1), jika kamu belum memuat profil pengguna, gunakan `get_user_profile` terlebih dahulu.
- **Deteksi Preferensi Baru:** Jika pengguna secara eksplisit meminta perubahan sapaan atau gaya bahasa (misal: "Mulai sekarang panggil gue Boss", "Pake gaya formal dong"), segera simpan menggunakan `update_user_profile` sebelum memberikan jawaban akhir.
- ReAct loop dibatasi maksimal 3 langkah. Pastikan kamu sudah menjawab (`action`: `answer`) pada langkah ke-3.
"""


def _detect_vibe(text: str) -> str:
    """Detect the communication style / vibe of the user's message.

    Returns one of:
        "genz_panic" – slang with panic/urgency markers
        "genz"       – clear Gen Z / slang vocabulary
        "formal"     – default professional style
    """
    has_genz = bool(
        _GENZ_FIRST_PERSON.search(text)
        or _GENZ_SECOND_PERSON.search(text)
        or _GENZ_VOCAB.search(text)
    )
    has_panic = bool(_PANIC_MARKERS.search(text) or _PANIC_CAPS.search(text))

    if has_genz and has_panic:
        return "genz_panic"
    if has_genz:
        return "genz"
    return "formal"


class ResponderAgent(BaseAgent):
    """Generates a conversational reply using history + LLM.

    Does NOT use Tavily web search. For live-search-backed answers,
    the orchestrator should route to ResearcherAgent via the 'research' intent.

    The system prompt is chosen dynamically based on the detected communication
    style (vibe) of the user's message, enabling "language mirroring".
    """

    name = "responder"

    _MAX_HERMES_STEPS = 3

    def __init__(
        self,
        history: ConversationHistory,
        llm: LLMClient | None = None,
    ) -> None:
        self._history = history
        self._llm     = llm or LLMClient()
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
            logger.info("Responder done for session=%s", task.session_id)
        except Exception as exc:
            logger.exception("ResponderAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = "Maaf, terjadi kesalahan. Silakan coba lagi."

        return task

    async def _run_hermes_loop(
        self,
        query: str,
        session_id: str,
        history_messages: list[dict] | None = None
    ) -> str:
        """Run the lightweight ReAct loop for ResponderAgent."""
        # Dynamic vibe detection context (used when preferred_vibe is 'auto')
        detected_vibe = _detect_vibe(query)
        vibe_context = f"\n[Dynamic Vibe Detection: User message vibe is detected as '{detected_vibe}']"

        system_prompt = _HERMES_RESPONDER_DECISION_PROMPT + vibe_context
        persona = self.get_persona_prompt()
        if persona:
            system_prompt = persona + "\n\n" + system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            # Include last 10 messages for conversational context
            messages.extend(history_messages[-10:])

        if not history_messages or history_messages[-1]["content"] != query:
            messages.append({"role": "user", "content": query})

        step = 0
        final_answer = ""

        while step < self._MAX_HERMES_STEPS:
            step += 1
            logger.info(
                "ResponderAgent Hermes Loop: Step %d/%d for session=%s",
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

                    key_match = re.search(r'"profile_key"\s*:\s*"([^"]+)"', cleaned_response)
                    sk = key_match.group(1) if key_match else ""

                    val_match = re.search(r'"profile_value"\s*:\s*"([^"]+)"', cleaned_response)
                    sv = val_match.group(1) if val_match else ""

                    content_match = re.search(r'"content"\s*:\s*"(.*)"', cleaned_response, re.DOTALL)
                    sc = content_match.group(1) if content_match else ""

                    action_data = {
                        "thought": "Failed to parse JSON cleanly.",
                        "action": action,
                        "profile_key": sk,
                        "profile_value": sv,
                        "content": sc
                    }

                thought = action_data.get("thought", "")
                action = action_data.get("action", "answer")
                logger.info(
                    "ResponderAgent Step %d: Thought: %s | Action: %s",
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
                        "content": f"Error: tindakan '{action}' tidak valid. Silakan pilih 'get_current_time', 'get_user_profile', 'update_user_profile', atau 'answer'."
                    })

            except Exception as exc:
                logger.error("Error in ResponderAgent Hermes step: %s", exc)
                messages.append({
                    "role": "user",
                    "content": f"Terjadi kesalahan internal: {exc}. Silakan selesaikan dengan action 'answer'."
                })
                if step >= self._MAX_HERMES_STEPS - 1:
                    break

        if not final_answer:
            logger.info("ResponderAgent: forcing final answer generation")
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
                    final_answer = "Maaf, terjadi kesalahan saat memproses obrolan Anda."
            except Exception as exc:
                logger.error("Failed to generate forced final answer: %s", exc)
                final_answer = "Maaf, terjadi kesalahan saat memproses obrolan Anda."

        return sanitize_for_telegram(final_answer)
