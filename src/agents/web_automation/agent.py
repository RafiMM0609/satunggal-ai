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
  • Read the current page content after navigation or clicks (get_content).
  • Extract structured list data from a page using CSS selectors (extract_data).
  • Explore a page autonomously to find content related to a specific topic
    (e.g. "explore https://docs.example.com, find content about full text search").

Workflow (true ReAct loop)
──────────────────────────
  1. The LLM decides the SINGLE NEXT action based on the user's request and all
     previous steps + their results (full ReAct: Reason → Act → Observe → repeat).
  2. The chosen action is executed (web_reader or browser_navigator).
  3. The result is appended to the accumulated context in compact form.
  4. Steps 1–3 repeat until the LLM outputs ``done`` or max steps is reached.
  5. The LLM produces a final natural-language summary for the user.

  Key exploration actions
  -----------------------
  • ``get_links``        – extracts all navigable links from the current page so the
                           LLM can pick the most relevant one and follow it.
  • ``navigate``         – opens a URL (reuses the existing browser context).
  • ``get_content``      – reads the visible text of the current page (fast, partial).
  • ``get_full_content`` – auto-scrolls the full page then returns complete text;
                           use when the user explicitly requests ALL page content.

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
from typing import TYPE_CHECKING, Any

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask
from src.tools.browser_navigator import BrowserNavigatorTool
from src.tools.web_reader import WebReaderTool

if TYPE_CHECKING:
    from src.memory.history import ConversationHistory

logger = logging.getLogger(__name__)

# ── Per-session last-visited URL store ────────────────────────────────────────
# Tracks the last successfully navigated URL for each session so that
# follow-up commands (e.g. "Klik tombol Sign in") can resume browsing
# on the correct page without the user having to repeat the URL.
_session_last_url: dict[str, str] = {}

# ── Per-session visited-domain store ─────────────────────────────────────────
# Tracks which base URLs (scheme+host) were visited per session so that
# clear_web_automation_session() can delete the right browser session files.
_session_domains: dict[str, set[str]] = {}


def clear_web_automation_session(session_id: str) -> None:
    """Remove all web-automation state for *session_id*.

    Clears:
    * The last-visited URL entry for this session.
    * All saved Playwright browser session files (cookies/localStorage) for
      every domain visited during this session.

    Called by the orchestrator's ``clear_session()`` when the user runs /reset.
    """
    from src.memory.state import BrowserSessionStore

    # Remove the last-visited URL for this session
    _session_last_url.pop(session_id, None)

    # Delete browser session files for all domains visited in this session
    domains = _session_domains.pop(session_id, set())
    if domains:
        store = BrowserSessionStore()
        for base_url in domains:
            store.delete_session(base_url)
        logger.info(
            "clear_web_automation_session: deleted %d browser session(s) for session=%s",
            len(domains), session_id,
        )
    else:
        logger.info("clear_web_automation_session: no browser sessions to delete for session=%s", session_id)

# ── System prompts ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
Kamu adalah Web Automation Planner. Tugasmu adalah menguraikan permintaan \
pengguna menjadi serangkaian langkah browsing yang terurut dan dapat dieksekusi.

Setiap langkah HARUS berupa JSON object dengan field berikut:
  "action": satu dari ["read_url", "navigate", "click", "type", "scroll", \
"screenshot", "get_content", "get_full_content", "get_links", "extract_data", \
"save_session", "select_option", "done"]
  "params": object parameter yang sesuai dengan action:
    - read_url:          {"url": "..."}
    - navigate:          {"url": "..."}
    - click:             {"text": "..."}
    - type:              {"selector": "...", "label": "...", "text": "..."}
                         (selector = CSS selector jika diketahui; label = teks label/placeholder
                          field tersebut agar dapat ditemukan secara akurat; keduanya boleh kosong
                          tapi SANGAT DIANJURKAN untuk mengisi setidaknya salah satu agar field
                          yang tepat dapat diidentifikasi, terutama saat mengisi lebih dari 1 field)
    - scroll:            {"direction": "down"|"up"}
    - screenshot:        {}
    - get_content:       {}
    - get_full_content:  {}
    - get_links:         {}
    - extract_data:      {"selector": "...", "attribute": "text"|"href", "limit": 50}
    - save_session:      {"url": "..."}
    - select_option:     {"text": "...", "selector": "..."}
                         (text = teks opsi yang ingin dipilih, contoh: "Kopi & Teh";
                          selector = CSS selector elemen <select> opsional jika diketahui;
                          GUNAKAN ini untuk memilih kategori, radio button, atau toggle button
                          yang merupakan elemen kustom dalam form)
    - done:              {"summary": "ringkasan hasil untuk pengguna"}

Kapan menggunakan "get_content" vs "get_full_content":
  • "get_content"       – gunakan untuk membaca sebagian konten halaman (default, cepat).
  • "get_full_content"  – gunakan HANYA ketika pengguna secara EKSPLISIT meminta
    SELURUH isi halaman, misalnya: "tampilkan semua konten", "berikan seluruh isi
    halaman", "scroll sampai habis", "ambil semua teks di halaman ini", "baca
    keseluruhan artikel", "full page content". Action ini akan auto-scroll dari
    atas ke bawah agar konten lazy-load termuat sepenuhnya, lalu mengembalikan
    teks lengkap tanpa pemotongan.

Strategi Locator Efisien (Hemat Token – WAJIB DIIKUTI):
  • JANGAN memasukkan langkah "get_content" sebelum setiap klik hanya untuk membaca HTML.
    Playwright menemukan elemen secara otomatis menggunakan strategi bertingkat:
    get_by_role → get_by_label → get_by_text → JS DOM walk.
  • Jika teks tombol/link sudah diketahui (misalnya "Login", "Submit", "Masuk"),
    langsung buat langkah "click" dengan teks tersebut TANPA "get_content" terlebih dahulu.
  • Playwright menunggu otomatis (auto-wait) – TIDAK PERLU menambahkan langkah "tunggu"
    atau "screenshot" hanya untuk memastikan elemen sudah muncul sebelum diklik.
  • Gunakan "get_content" hanya ketika:
      (a) Halaman baru dibuka dan kamu belum tahu elemen apa yang tersedia, atau
      (b) Kamu perlu membaca ISI KONTEN halaman untuk menjawab pertanyaan pengguna.
  • Hasil "get_content" menyertakan field "locators" – daftar elemen interaktif
    ({role, name}) dari Accessibility Tree. Gunakan "name" sebagai "text" pada "click"
    atau "label" pada "type". Ini jauh lebih ringkas daripada membaca seluruh page_text.

Panduan penggunaan action:
  • Gunakan "navigate" untuk membuka URL, lalu "click" untuk berinteraksi,
    kemudian "get_content" untuk membaca konten halaman terkini.
  • Gunakan "get_content" setelah "navigate" atau "click" agar konten halaman
    yang sudah diperbarui dapat dibaca (bukan "read_url" yang membuka tab baru).
  • Gunakan "get_links" untuk mengekstrak semua link navigasi dari halaman saat ini;
    sangat berguna saat mengeksplor halaman dokumentasi untuk menemukan topik yang relevan.
  • Gunakan "extract_data" dengan selector CSS untuk mengambil daftar item
    terstruktur (misalnya: daftar repositori, berita, produk, baris tabel).
    Jika selector tidak diketahui, kosongkan dan biarkan auto-detect bekerja.
  • Gunakan "read_url" hanya jika tidak perlu interaksi klik/scroll sebelumnya.

Panduan pengisian form (type):
  • Saat mengisi lebih dari satu field, SELALU gunakan parameter "label" yang berisi
    teks label atau placeholder field tersebut (contoh: "Email", "Nomor Ponsel",
    "Password", "PIN") agar setiap langkah menargetkan field yang berbeda.
  • Jika terdapat field "Nomor Ponsel atau Email", gunakan label: "Nomor Ponsel atau Email"
    atau label: "Email".
  • Untuk field PIN gunakan label: "PIN" atau label: "Kode PIN".
  • Jangan membiarkan "label" dan "selector" keduanya kosong saat ada beberapa field
    yang harus diisi; hal ini dapat menyebabkan semua input masuk ke field yang sama.

Panduan pemilihan kategori dan opsi form kustom (select_option):
  • Untuk memilih kategori produk (contoh: "Kopi & Teh"), radio button, atau opsi
    dalam widget form kustom (bukan elemen <select> standar), SELALU gunakan action
    "select_option" dengan parameter {"text": "<teks opsi>"}.
  • Jangan menggunakan "click" untuk elemen kategori karena elemen tersebut mungkin
    dirender sebagai div/button/label kustom yang memerlukan strategi pencarian
    khusus (scroll ke dalam modal, force click, dll.).
  • Jika "select_option" juga gagal, coba "scroll" ke bawah terlebih dahulu agar
    elemen masuk ke viewport, kemudian ulangi "select_option".

Penanganan login dan navigasi pasca-klik:
  • Setelah mengklik tombol submit login (contoh: "Masuk", "Login", "Sign In",
    "Submit"), sistem akan otomatis menunggu navigasi/redirect selesai dan
    menangkap konten halaman yang baru. Tambahkan langkah "get_content"
    SETELAH tombol submit diklik agar detail halaman pasca-login dapat ditampilkan.
  • Jika halaman melakukan redirect setelah login, langkah "get_content"
    berikutnya akan membaca konten dari halaman tujuan redirect tersebut.
  • Tambahkan langkah "screenshot" setelah login untuk memvisualisasikan
    tampilan halaman pasca-login (opsional namun dianjurkan).

Penanganan menu SPA (Single Page Application):
  • Setelah login berhasil ke aplikasi berbasis SPA (React/Vue/Angular), menu
    mungkin dirender secara dinamis. Gunakan langkah "get_content" terlebih dahulu
    untuk memastikan menu sudah dimuat sebelum mengklik.
  • SELALU tambahkan langkah "get_content" SETELAH mengklik item menu SPA untuk
    memverifikasi bahwa halaman berhasil dimuat (bukan menampilkan error 500).
  • Jika hasil klik menunjukkan error halaman (misalnya pesan error sistem, halaman
    500, atau CORS error), laporkan dalam summary dan jangan coba klik ulang.
  • Untuk klik pada item menu navigasi dalam SPA, gunakan teks yang TEPAT sesuai
    tampilan di halaman (persis seperti yang terlihat, termasuk kapitalisasi).

Penanganan perintah lanjutan (follow-up):
  • Jika konteks percakapan atau metadata menunjukkan URL terakhir yang dikunjungi,
    dan perintah saat ini TIDAK menyebutkan URL baru (misalnya perintah seperti
    "klik tombol X", "scroll ke bawah", "isi form", dll.), tambahkan langkah
    "navigate" ke URL tersebut sebagai langkah PERTAMA sebelum aksi lainnya.
  • Pencocokan teks elemen (click) bersifat case-insensitive dan partial match,
    sehingga "sign in", "Sign In", maupun "SIGN IN" akan mencocokkan tombol
    yang sama.

Aturan:
1. Balas HANYA dengan JSON array dari langkah-langkah tersebut – tidak ada teks lain.
2. Selalu akhiri dengan langkah "done" yang berisi ringkasan apa yang sudah dilakukan.
3. Maksimal 15 langkah (tidak termasuk "done").
4. Gunakan bahasa yang sama dengan permintaan pengguna untuk field "summary".
"""

_SUMMARISER_SYSTEM = """\
Kamu adalah asisten yang merangkum hasil browsing web untuk pengguna.
Berdasarkan log aksi dan konten halaman yang diberikan, buat ringkasan yang:
  - SELALU awali jawaban dengan baris "📍 URL Aktif: <url>" menggunakan nilai
    "URL aktif saat ini" dari konteks. Baris ini WAJIB ada di bagian paling atas
    jawaban agar pengguna tahu halaman mana yang sedang aktif dan perintah
    lanjutan dapat menggunakan konteks URL tersebut secara otomatis.
  - Jelas dan mudah dipahami.
  - Menyebutkan URL yang dikunjungi dan judul halamannya.
  - Menjelaskan elemen-elemen penting yang ditemukan (menu, tombol, form, dll.).
  - Jika ada daftar item yang diekstrak (repositori, berita, produk, dll.),
    tampilkan dalam format daftar yang terstruktur dan mudah dibaca.
  - Jika task melibatkan eksplorasi dokumen/halaman untuk mencari topik tertentu,
    rangkumkan ISI KONTEN yang ditemukan secara lengkap dan informatif, bukan
    hanya mencantumkan link atau nama halaman saja.
  - Melaporkan status setiap aksi (berhasil / gagal).
  - Jika terjadi navigasi/redirect setelah login (ditandai dengan "navigated: true"
    dalam log), tampilkan informasi halaman tujuan redirect (URL, judul, konten).
  - Jika terdapat "page_error" dalam hasil klik (ditandai dengan kunci "page_error"
    dalam log), laporkan dengan jelas bahwa halaman menampilkan error setelah klik
    tersebut, beserta detail pesan error yang ditemukan.
  - Menggunakan bahasa yang sama dengan permintaan pengguna (Indonesia atau Inggris).
"""

_REACT_SYSTEM = """\
Kamu adalah Web Automation Decision Maker. Berdasarkan permintaan pengguna dan \
riwayat langkah yang sudah dilakukan, tentukan SATU langkah berikutnya yang perlu diambil.

Balas HANYA dengan SATU JSON object (bukan array) dengan field berikut:
  "action": satu dari ["read_url", "navigate", "click", "type", "scroll",
            "screenshot", "get_content", "get_full_content", "get_links",
            "extract_data", "save_session", "select_option", "done"]
  "params": parameter yang sesuai dengan action:
    - read_url:          {"url": "..."}
    - navigate:          {"url": "..."}
    - click:             {"text": "..."}
    - type:              {"selector": "...", "label": "...", "text": "..."}
    - scroll:            {"direction": "down"|"up"}
    - screenshot:        {}
    - get_content:       {}
    - get_full_content:  {}
    - get_links:         {}
    - extract_data:      {"selector": "...", "attribute": "text"|"href", "limit": 50}
    - save_session:      {"url": "..."}
    - select_option:     {"text": "...", "selector": "..."}
                         (text = teks opsi yang ingin dipilih, contoh: "Kopi & Teh";
                          selector = CSS selector elemen <select> opsional;
                          GUNAKAN untuk memilih kategori, radio button, toggle button kustom)
    - done:              {"summary": "ringkasan lengkap hasil untuk pengguna"}
  "reasoning": penjelasan singkat mengapa langkah ini dipilih (1-2 kalimat)

Kapan menggunakan "get_content" vs "get_full_content":
  • "get_content"       – gunakan untuk membaca sebagian konten halaman (default, hemat token).
  • "get_full_content"  – gunakan HANYA ketika pengguna secara EKSPLISIT meminta
    SELURUH isi halaman, misalnya: "tampilkan semua konten", "berikan seluruh isi
    halaman", "scroll sampai habis dan ambil semua", "full content", "baca semua
    artikel ini", "keseluruhan konten". Action ini auto-scroll dari atas ke bawah
    agar konten lazy-load termuat, lalu mengembalikan teks lengkap halaman.
    JANGAN gunakan jika pengguna hanya ingin berinteraksi dengan halaman.

Strategi Locator Efisien (Hemat Token – WAJIB DIIKUTI):
  • JANGAN membaca seluruh HTML sebelum klik. Playwright mencari elemen secara otomatis
    menggunakan teks/label – kamu TIDAK perlu memanggil "get_content" sebelum setiap klik.
  • "click" dan "type" menggunakan strategi bertingkat: get_by_role → get_by_label →
    get_by_text → JS DOM walk. Cukup berikan teks tombol/link yang terlihat di halaman.
  • Jika teks tombol/link sudah diketahui (dari riwayat atau konteks), langsung gunakan
    "click" dengan teks tersebut TANPA memanggil "get_content" terlebih dahulu.
  • Playwright menunggu otomatis (auto-wait) hingga elemen muncul – TIDAK PERLU menambahkan
    langkah "tunggu" atau "get_content" hanya untuk memverifikasi elemen ada.
  • Gunakan "get_content" hanya ketika:
      (a) Kamu belum tahu elemen apa yang ada di halaman (halaman baru dibuka), atau
      (b) Kamu perlu membaca ISI KONTEN halaman (bukan hanya berinteraksi dengannya).
  • Setelah "get_content", gunakan field "locators" dari hasilnya – daftar elemen interaktif
    ({role, name}) dari Accessibility Tree. Gunakan "name" sebagai "text" pada "click"
    atau "label" pada "type" untuk langkah berikutnya. JANGAN baca page_text hanya untuk
    mencari nama tombol; gunakan "locators" yang jauh lebih ringkas.

Panduan eksplorasi halaman dokumentasi / pencarian konten:
  • Saat diminta mengeksplor, mencari, atau menemukan konten tertentu dalam sebuah halaman:
    1. Mulai dengan "navigate" atau "read_url" ke URL yang diberikan
    2. Gunakan "get_links" untuk melihat semua link navigasi yang tersedia
    3. Analisis teks link dan pilih yang paling relevan dengan query pengguna
    4. Gunakan "navigate" ke URL link tersebut
    5. Gunakan "get_content" untuk membaca konten halaman yang baru dibuka
    6. Jika konten relevan ditemukan, output "done" dengan ringkasan lengkap isi konten
    7. Jika belum cukup, gunakan "get_links" lagi untuk menelusuri sub-navigasi lebih dalam
  • Jangan berulang kali mengunjungi URL yang sama jika tidak ada perubahan.
  • Prioritaskan link yang teks-nya paling relevan dengan topik yang dicari.
  • Setelah menemukan halaman konten yang relevan, baca penuh dengan "get_content" sebelum "done".

Panduan umum:
  • Gunakan "navigate" lalu "get_content" (bukan "read_url") saat sudah ada browser terbuka.
  • Gunakan "read_url" hanya jika ini adalah langkah pertama dan belum ada browser.
  • Setelah "click" yang menyebabkan navigasi, gunakan "get_content" untuk membaca halaman baru.
  • Untuk field teks form: gunakan "type" dengan "label" yang sesuai placeholder/label field.
  • Untuk kategori, radio button, atau opsi kustom dalam form: SELALU gunakan "select_option"
    (bukan "click") agar elemen dapat ditemukan bahkan jika berada di luar viewport modal.

Gunakan action "done" dengan ringkasan komprehensif ketika:
  - Konten yang relevan sudah ditemukan dan kamu memiliki cukup informasi untuk menjawab query
  - Tidak ada lagi link relevan untuk diikuti
  - Langkah-langkah sebelumnya gagal dan sudah ada informasi yang cukup untuk dilaporkan
  - Mendekati batas langkah maksimum

Aturan:
  1. Balas HANYA dengan SATU JSON object – tidak ada teks lain di luar JSON.
  2. Jangan mengulangi langkah yang persis sama jika sudah dilakukan dan gagal.
  3. Selalu akhiri dengan "done" yang berisi ringkasan lengkap.
"""

_MAX_REACT_STEPS       = 20    # max number of tool-execution steps in the ReAct loop
_MAX_TOKENS            = 2048
_SUMMARISE_TEXT_CHARS  = 2000  # page text characters included per result in summariser
_FULL_PAGE_SUMMARISE_TEXT_CHARS = 8_000  # higher budget for get_full_content results
_SUMMARISE_ITEMS_LIMIT = 50    # max extracted items shown in summariser
_HISTORY_MSG_CHARS     = 500   # max characters per message included in planner context
_MAX_ERROR_MSG_CHARS   = 200   # max error message characters included in action log entries
_REACT_RESULT_TEXT_CHARS = 800  # max page_text chars kept in each compact ReAct step result
_REACT_LINKS_LIMIT     = 60    # max links kept per step in compact ReAct context
_REACT_LOCATORS_LIMIT  = 30    # max interactive element locators kept per step in compact context

_SKIP_KEYS = frozenset({"screenshot_b64", "a11y_tree"})


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact version of a tool result for the ReAct context.

    Strips bulky fields (base64 screenshots, full accessibility trees) and
    truncates large text to keep per-step token usage within a safe budget.
    The resulting dict is serialised as JSON and appended to the accumulated
    steps context that the LLM reads on every planning call.

    The ``locators`` field (interactive elements from the a11y tree) is
    preserved but limited to ``_REACT_LOCATORS_LIMIT`` entries so the LLM
    can use element names directly in subsequent ``click``/``type`` actions
    without having to parse the full page HTML.
    """
    compact = {k: v for k, v in result.items() if k not in _SKIP_KEYS}
    # Truncate long page text so it doesn't dominate the context
    if "page_text" in compact and isinstance(compact["page_text"], str):
        if len(compact["page_text"]) > _REACT_RESULT_TEXT_CHARS:
            compact["page_text"] = compact["page_text"][:_REACT_RESULT_TEXT_CHARS] + "…"
    # Limit the number of links to keep the context manageable
    if "links" in compact and isinstance(compact["links"], list):
        compact["links"] = compact["links"][:_REACT_LINKS_LIMIT]
    # Limit extracted items list
    if "items" in compact and isinstance(compact["items"], list):
        compact["items"] = compact["items"][:30]
    # Limit interactive locators – the LLM uses these to target click/type actions
    if "locators" in compact and isinstance(compact["locators"], list):
        compact["locators"] = compact["locators"][:_REACT_LOCATORS_LIMIT]
    return compact


class WebAutomationAgent(BaseAgent):
    """
    Autonomous web browsing agent.

    Uses a ReAct-style loop:
      LLM plans → tool executes → result fed back → LLM plans next step.
    """

    name = "web_automation"

    def __init__(
        self,
        llm: LLMClient | None = None,
        history: "ConversationHistory | None" = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._history = history

    async def run(self, task: AgentTask) -> AgentTask:
        task.mark_processing(self.name)
        navigator = BrowserNavigatorTool()
        reader    = WebReaderTool()

        try:
            action_log: list[str] = []
            # Accumulated context for the ReAct loop: each entry holds the
            # action name, its params, and a compact version of the result
            # so the LLM can decide the next step with full history.
            steps_done: list[dict[str, Any]] = []

            for i in range(1, _MAX_REACT_STEPS + 1):
                # Plan the single next step given everything done so far
                next_step = await self._plan_next_step(
                    task.user_input, task.session_id, steps_done
                )
                action = next_step.get("action", "done")
                params = next_step.get("params", {})

                logger.info(
                    "WebAutomationAgent: step %d action=%r session=%s",
                    i, action, task.session_id,
                )

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

                # Track the last navigated URL per session for follow-up commands.
                # get_content, get_full_content, and get_links also return the current
                # page URL, which keeps the session URL accurate even when no navigation occurred.
                if action in ("navigate", "read_url", "get_content", "get_full_content", "get_links"):
                    visited_url = tool_result.get("url") or params.get("url", "")
                    if visited_url and not tool_result.get("error"):
                        _session_last_url[task.session_id] = visited_url

                if tool_result.get("error"):
                    logger.warning(
                        "WebAutomationAgent: step %d failed: %s",
                        i, tool_result["error"],
                    )
                    action_log.append(f"  ⚠ Gagal: {tool_result['error']}")
                    # Continue to next step instead of aborting (best-effort)

                # Build a compact result for the ReAct context (strips screenshots
                # and large a11y trees to keep token usage manageable)
                steps_done.append({
                    "step":   action,
                    "params": params,
                    "result": _compact_result(tool_result),
                })
            else:
                # Loop exhausted without a "done" action – log that max steps was reached
                action_log.append(f"[{_MAX_REACT_STEPS + 1}] done (batas langkah maksimum tercapai)")
                logger.warning(
                    "WebAutomationAgent: max ReAct steps (%d) reached for session=%s",
                    _MAX_REACT_STEPS, task.session_id,
                )

            # Collect screenshots captured during the session so the interface
            # layer (e.g. Telegram handler) can forward them to the user.
            screenshots: list[str] = [
                val["screenshot_b64"]
                for val in task.tool_results.values()
                if isinstance(val, dict)
                and val.get("action") == "screenshot"
                and val.get("screenshot_b64")
            ]
            if screenshots:
                task.metadata["screenshots"] = screenshots

            # Build final reply using the LLM summariser
            final_url = _session_last_url.get(task.session_id, "")
            reply = await self._summarise(task.user_input, action_log, task.tool_results, final_url)
            task.mark_done(reply)

        except Exception as exc:
            logger.exception("WebAutomationAgent failed: %s", exc)
            task.mark_failed(str(exc))
            task.result = (
                "Maaf, terjadi kesalahan saat menjalankan web automation. "
                f"Detail: {exc}"
            )
        finally:
            # Auto-save the browser session (cookies & storage) before closing so
            # that login state is preserved across /reset commands and future tasks.
            final_url = _session_last_url.get(task.session_id, "")
            if final_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(final_url)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    await navigator.save_current_session(base_url)
                    # Register the domain so /reset can clean up its session file
                    _session_domains.setdefault(task.session_id, set()).add(base_url)
                except Exception as exc:
                    logger.warning(
                        "WebAutomationAgent: failed to auto-save session for %s: %s",
                        final_url, exc,
                    )
            await navigator.close()

        return task

    # ── ReAct step planner ────────────────────────────────────────────────────

    async def _plan_next_step(
        self,
        user_input: str,
        session_id: str,
        steps_done: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask the LLM to decide the SINGLE NEXT action to take.

        Includes the full accumulated browsing history (actions + compact
        results) so the LLM can make an informed, adaptive decision rather
        than committing to a fixed plan created before any page was seen.
        """
        context_parts: list[str] = []

        # Include the last visited URL for follow-up command context
        last_url = _session_last_url.get(session_id, "")
        if last_url:
            context_parts.append(f"URL terakhir yang dikunjungi: {last_url}")

        # Include recent conversation history for multi-turn awareness
        if self._history and session_id:
            recent = self._history.get(session_id)
            prev_messages = recent[:-1][-4:] if len(recent) > 1 else []
            if prev_messages:
                lines = "\n".join(
                    f"[{m.role.upper()}]: {m.content[:_HISTORY_MSG_CHARS]}"
                    for m in prev_messages
                )
                context_parts.append(f"Riwayat percakapan:\n{lines}")

        # Include all steps done so far with their compact results
        if steps_done:
            steps_text = json.dumps(steps_done, ensure_ascii=False, indent=2)
            context_parts.append(f"Langkah yang sudah dilakukan:\n{steps_text}")

        system_content = (
            _REACT_SYSTEM
            + "\n\nKonteks tambahan:\n"
            + "\n\n".join(context_parts)
            if context_parts
            else _REACT_SYSTEM
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_input},
        ]
        try:
            raw = await self._llm.chat(messages, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            step = json.loads(raw)
            if not isinstance(step, dict):
                raise ValueError("Expected a JSON object")
            return step
        except Exception as exc:
            logger.warning("WebAutomationAgent: next-step planning failed (%s), using fallback", exc)
            # Fallback: if this is the first step and the input contains a URL, read it
            if not steps_done:
                url = self._extract_url(user_input)
                if url:
                    return {"action": "read_url", "params": {"url": url}}
                if last_url:
                    return {"action": "navigate", "params": {"url": last_url}}
            return {"action": "done", "params": {"summary": "Tidak dapat merencanakan langkah selanjutnya."}}

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
            if result.get("navigated"):
                # Navigation occurred (e.g. login redirect): include destination info in log
                page_error_note = ""
                if result.get("page_error"):
                    page_error_note = f" ⚠ PAGE_ERROR: {result['page_error'][:_MAX_ERROR_MSG_CHARS]}"
                log = (
                    f"[{step_num}] click '{text}' → navigated to {result.get('url', '?')} "
                    f"title={result.get('title', '?')!r} "
                    f"chars={len(result.get('page_text', ''))}"
                    f"{page_error_note}"
                )
                # Update the last-visited URL so follow-up commands target the new page
                if result.get("url") and not result.get("error"):
                    _session_last_url[task.session_id] = result["url"]
            elif result.get("page_error"):
                log = (
                    f"[{step_num}] click '{text}' → {result.get('message', '?')} "
                    f"⚠ PAGE_ERROR: {result['page_error'][:_MAX_ERROR_MSG_CHARS]}"
                )
            else:
                log = f"[{step_num}] click '{text}' → {result.get('message', result.get('error', '?'))}"

        elif action == "type":
            task.metadata.update({
                "browser_action":  "type",
                "type_selector":   params.get("selector", ""),
                "type_label":      params.get("label", ""),
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

        elif action == "get_content":
            task.metadata["browser_action"] = "get_content"
            result = await navigator.run(task)
            if result.get("url") and not result.get("error"):
                _session_last_url[task.session_id] = result["url"]
            log = (
                f"[{step_num}] get_content → "
                f"title={result.get('title', '?')!r} "
                f"chars={len(result.get('page_text', ''))} "
                f"nodes={len(result.get('a11y_tree', []))}"
            )

        elif action == "get_full_content":
            task.metadata["browser_action"] = "get_full_content"
            result = await navigator.run(task)
            if result.get("url") and not result.get("error"):
                _session_last_url[task.session_id] = result["url"]
            log = (
                f"[{step_num}] get_full_content → "
                f"title={result.get('title', '?')!r} "
                f"chars={len(result.get('page_text', ''))} "
                f"scroll_steps={result.get('scroll_steps', 0)} "
                f"nodes={len(result.get('a11y_tree', []))}"
            )

        elif action == "get_links":
            task.metadata["browser_action"] = "get_links"
            result = await navigator.run(task)
            if result.get("url") and not result.get("error"):
                _session_last_url[task.session_id] = result["url"]
            log = (
                f"[{step_num}] get_links → "
                f"{result.get('count', 0)} links from {result.get('url', '?')}"
            )

        elif action == "extract_data":
            task.metadata.update({
                "browser_action":    "extract_data",
                "extract_selector":  params.get("selector",  ""),
                "extract_attribute": params.get("attribute", "text"),
                "extract_limit":     params.get("limit",     50),
            })
            result = await navigator.run(task)
            log = (
                f"[{step_num}] extract_data selector={params.get('selector', 'auto')!r} → "
                f"{result.get('count', 0)} items"
            )

        elif action == "select_option":
            option_text = params.get("text", "")
            task.metadata.update({
                "browser_action":  "select_option",
                "option_text":     option_text,
                "option_selector": params.get("selector", ""),
            })
            result = await navigator.run(task)
            log = (
                f"[{step_num}] select_option '{option_text}' → "
                f"{result.get('message', result.get('error', '?'))}"
            )

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
        current_url: str = "",
    ) -> str:
        """Ask the LLM to produce a user-friendly summary of what happened."""
        log_text = "\n".join(action_log)

        # Collect page content from read_url and get_content results
        page_snippets: list[str] = []
        extracted_data: list[str] = []
        link_sections: list[str] = []

        for key, val in tool_results.items():
            if not isinstance(val, dict):
                continue
            # Page text from read_url, get_content, or get_full_content.
            # Use a larger text budget for get_full_content so the LLM can
            # present the complete page content requested by the user.
            if val.get("page_text"):
                title = val.get("title", "")
                url   = val.get("url",   "")
                header = f"[{key}] {title} ({url})" if (title or url) else f"[{key}]"
                text_limit = (
                    _FULL_PAGE_SUMMARISE_TEXT_CHARS
                    if val.get("full_page")
                    else _SUMMARISE_TEXT_CHARS
                )
                page_snippets.append(f"{header}:\n{val['page_text'][:text_limit]}")
            # Error page info from click results
            if val.get("page_error"):
                page_snippets.append(f"[{key}] PAGE_ERROR: {val['page_error'][:_SUMMARISE_TEXT_CHARS]}")
            # Structured items from extract_data
            if val.get("items"):
                items_text = "\n".join(f"  - {item}" for item in val["items"][:_SUMMARISE_ITEMS_LIMIT])
                extracted_data.append(
                    f"[{key}] {val.get('count', len(val['items']))} items "
                    f"dari {val.get('url', '')}:\n{items_text}"
                )
            # Links extracted from get_links (show a representative sample)
            if val.get("action") == "get_links" and val.get("links"):
                links_sample = val["links"][:20]
                links_text = "\n".join(
                    f"  - {lnk.get('text', '')}: {lnk.get('href', '')}"
                    for lnk in links_sample
                )
                link_sections.append(
                    f"[{key}] {val.get('count', len(val['links']))} links "
                    f"dari {val.get('url', '')}:\n{links_text}"
                )

        context = f"Log aksi:\n{log_text}"
        if current_url:
            context = f"URL aktif saat ini: {current_url}\n\n" + context
        if page_snippets:
            context += "\n\nKonten halaman:\n" + "\n\n".join(page_snippets)
        if extracted_data:
            context += "\n\nData yang diekstrak:\n" + "\n\n".join(extracted_data)
        if link_sections:
            context += "\n\nLink navigasi yang ditemukan:\n" + "\n\n".join(link_sections)

        messages = [
            {"role": "system", "content": _SUMMARISER_SYSTEM},
            {"role": "user",   "content": f"Permintaan pengguna:\n{user_input}\n\n{context}"},
        ]
        try:
            return await self._llm.chat(messages, max_tokens=_MAX_TOKENS)
        except Exception as exc:
            logger.warning("WebAutomationAgent: summariser failed: %s", exc)
            prefix = f"📍 URL Aktif: {current_url}\n\n" if current_url else ""
            return f"{prefix}Web automation selesai.\n\nLog:\n{log_text}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_url(text: str) -> str:
        """Extract the first http(s) URL from *text*, or return empty string."""
        import re
        m = re.search(r"https?://[^\s\"'<>]+", text)
        return m.group(0) if m else ""
