# APP_LOGIC_WORKFLOW.md
# Panduan Teknis – Alur Kerja & Pengembangan

> Dokumen ini adalah **referensi utama developer** untuk memahami alur kerja internal sistem  
> dan cara menambah/memodifikasi komponen (agent, tool, intent) secara benar.  
> Untuk daftar kemampuan & roadmap fitur, lihat [CAPABILITIES.md](CAPABILITIES.md).

---

## Daftar Isi

1. [Konsep Inti](#1-konsep-inti)
2. [Alur Request End-to-End](#2-alur-request-end-to-end)
3. [AgentTask – Blackboard Pattern](#3-agenttask--blackboard-pattern)
4. [Dua Fase Eksekusi Tool](#4-dua-fase-eksekusi-tool)
5. [Struktur Folder](#5-struktur-folder)
6. [Cara Menambah Agent Baru](#6-cara-menambah-agent-baru)
7. [Cara Menambah Tool Baru](#7-cara-menambah-tool-baru)
8. [Cara Menambah Intent Baru](#8-cara-menambah-intent-baru)
9. [Kontrak Antar Komponen](#9-kontrak-antar-komponen)
10. [Aturan Utama yang Tidak Boleh Dilanggar](#10-aturan-utama-yang-tidak-boleh-dilanggar)
11. [Logging & Debugging](#11-logging--debugging)

---

## 1. Konsep Inti

Sistem ini dibangun di atas tiga prinsip:

| Prinsip | Penjelasan |
|---------|------------|
| **Single Entry Point** | Semua interface (Telegram, REST API) memanggil satu fungsi: `process_message()` di `src/orchestrator/main_loop.py` |
| **Blackboard Pattern** | Satu objek `AgentTask` dibawa dari awal ke akhir pipeline. Semua komponen baca/tulis ke objek yang sama, tidak ada passing parameter manual |
| **Orchestrator sebagai Controller** | Hanya orchestrator yang memanggil tools. Agent **tidak** diizinkan memanggil tool secara langsung — agent hanya **memberi sinyal** via `task.pending_tools` |

---

## 2. Alur Request End-to-End

```
┌─────────────────────────────────────────────────────────────────────┐
│  Interface (Telegram / REST API / CLI)                              │
│  Memanggil: process_message(session_id, user_text)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  AgentTask      │  ← blackboard dibuat kosong
                    │  (state baru)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────────────────────────────┐
                    │  GatekeeperAgent.classify_intent()              │
                    │  → intent: "mandays_planning"                   │
                    │  → confidence: 0.98                             │
                    │  → tools: []   (pre-agent tools)                │
                    └────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────────────┐
              │  Pre-agent Tool Loop  (jika tools != [])            │
              │  Contoh: tools=["tavily_search"]                    │
              │  → tool.run(task)                                   │
              │  → task.tool_results["tavily_search"] = {...}       │
              └──────────────┬──────────────────────────────────────┘
                             │
                    ┌────────▼────────────────────────────┐
                    │  AgentRouter.resolve(task)           │
                    │  intent → agent name                │
                    │  "mandays_planning" → MandaysAgent  │
                    └────────┬────────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────────────┐
              │  MandaysAgent.run(task)                             │
              │  1. Panggil LLM → terima JSON mandays               │
              │  2. task.metadata["mandays_json_data"] = json_data  │
              │  3. task.pending_tools.append("mandays_generator")  │
              │  4. task.mark_done("Rencana mandays dibuat...")     │
              └──────────────┬──────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────────────┐
              │  Post-agent Tool Loop  (dari task.pending_tools)    │
              │  → MandaysGeneratorTool.run(task)                   │
              │    · baca task.metadata["mandays_json_data"]        │
              │    · build Excel file                               │
              │    · return {"excel_path": "/tmp/...", ...}         │
              │  → task.metadata["excel_path"] = excel_path         │
              │  → task.pending_tools.clear()                       │
              └──────────────┬──────────────────────────────────────┘
                             │
                    ┌────────▼───────────────────────────────┐
                    │  task.result  →  teks reply ke user    │
                    │  task.metadata["excel_path"]  →  file  │
                    └────────┬───────────────────────────────┘
                             │
                    ┌────────▼───────────────────────────────┐
                    │  Interface: kirim teks + file          │
                    └────────────────────────────────────────┘
```

---

## 3. AgentTask – Blackboard Pattern

**File:** `src/memory/state.py`

```python
@dataclass
class AgentTask:
    session_id:    str               # ID sesi pengguna
    user_input:    str               # Teks asli dari pengguna
    intent:        str | None        # Diisi oleh GatekeeperAgent
    status:        TaskStatus        # PENDING → ROUTING → PROCESSING → DONE/FAILED
    result:        str | None        # Teks reply final untuk pengguna
    metadata:      dict[str, Any]    # Data bebas antar komponen (excel_path, json_data, dll)
    agent_trace:   list[str]         # Jejak perjalanan task (untuk debug)
    tool_results:  dict[str, Any]    # Output dari setiap tool yang sudah dijalankan
    pending_tools: list[str]         # Tools yang diminta agent untuk dijalankan post-agent
```

### Konvensi Penggunaan `metadata`

| Key | Diisi Oleh | Dibaca Oleh | Isi |
|-----|-----------|-------------|-----|
| `wbs_json_data` | WBSAgent | WBSGeneratorTool | Dict JSON WBS |
| `mandays_json_data` | MandaysAgent | MandaysGeneratorTool | Dict JSON Mandays |
| `excel_path` | Orchestrator (dari tool output) | Interface handler | Path file Excel temporer |
| `has_wbs_json` | WBSAgent | Interface handler | Boolean flag |
| `has_mandays_json` | MandaysAgent | Interface handler | Boolean flag |
| `error` | `task.mark_failed()` | Interface handler | Pesan error |

> **⚠️ Pengecualian DeveloperAgent:** DeveloperAgent **tidak** menyimpan state di `task.metadata` dan **tidak** menggunakan `task.pending_tools`. Agent ini mengelola seluruh workflow-nya secara internal melalui `CLIExecutor`, `SandboxRunner`, dan `GitManager` — bukan melalui pipeline orchestrator. Ini adalah satu-satunya pengecualian dari pola blackboard standar, karena tools tersebut bersifat stateful, sequential, dan berbagi konteks (repo path, credentials) yang tidak dapat dilewatkan melalui antarmuka tool biasa.

---

## 4. Dua Fase Eksekusi Tool

### Fase 1 – Pre-agent (sebelum agent dipanggil)

- **Dipicu oleh:** `intent_result.tools` (list dari GatekeeperAgent)
- **Kapan digunakan:** Tool yang mengumpulkan **konteks eksternal** untuk diberikan ke agent
- **Contoh:** `tavily_search` → hasilnya masuk `task.tool_results["tavily_search"]`, agent membaca ini saat LLM call

```
Gatekeeper: tools=["tavily_search"]
    → Orchestrator jalankan TavilySearchTool sebelum agent
    → ResearcherAgent membaca task.tool_results["tavily_search"]["context_text"]
```

### Fase 2 – Post-agent (setelah agent selesai)

- **Dipicu oleh:** `task.pending_tools` (list yang ditambahkan agent selama `run()`)
- **Kapan digunakan:** Tool yang melakukan **aksi deterministik** berdasarkan output LLM
- **Contoh:** `wbs_generator` / `mandays_generator` → membaca JSON dari `task.metadata`, build Excel

```
WBSAgent: task.pending_tools.append("wbs_generator")
    → Orchestrator jalankan WBSGeneratorTool setelah agent
    → WBSGeneratorTool membaca task.metadata["wbs_json_data"]
    → Menulis Excel, return {"excel_path": "..."}
    → Orchestrator: task.metadata["excel_path"] = excel_path
```

### Kapan Pakai Fase 1 vs Fase 2?

| Kriteria | Pre-agent (Fase 1) | Post-agent (Fase 2) |
|----------|-------------------|---------------------|
| Tool butuh output LLM? | Tidak | Ya |
| Tool memberi input ke LLM? | Ya | Tidak |
| Tool melakukan aksi file/IO? | Tidak (biasanya) | Ya |
| Siapa yang menentukan? | Gatekeeper (LLM) | Agent (kode) |

---

## 5. Struktur Folder

```
src/
├── agents/
│   ├── base_agent.py            ← ABC: semua agent wajib extends ini
│   ├── llm_client.py            ← Wrapper HTTP ke OpenRouter (OpenAI-compatible)
│   ├── repo_agent_base.py       ← Base class untuk agent berbasis repo (clone/pull/RAG/Tavily)
│   ├── gatekeeper/
│   │   ├── agent.py             ← GatekeeperAgent.classify_intent()
│   │   ├── openrouter.py        ← HTTP call + parse JSON intent dari LLM
│   │   └── schemas.py           ← IntentCategory enum + IntentResult model
│   ├── responder/agent.py       ← Percakapan umum, support, billing
│   ├── researcher/agent.py      ← Riset mendalam; baca task.tool_results["tavily_search"]
│   ├── content_creator/agent.py ← Konten platform digital
│   ├── wbs_agent/agent.py       ← LLM → JSON → task.pending_tools["wbs_generator"]
│   ├── mandays_agent/agent.py   ← LLM → JSON → task.pending_tools["mandays_generator"]
│   ├── developer/agent.py       ← Clone → edit → Docker sandbox → push (self-contained)
│   ├── developer_inspector/agent.py ← Read-only: inspeksi repo, root cause analysis
│   ├── developer_qna/
│   │   ├── agent.py             ← Q&A factual tentang isi codebase
│   │   └── HOW_IT_WORKS.md
│   ├── technical_writer/agent.py← Dokumen teknis chunked → PDF/DOCX
│   ├── doc_agent/agent.py       ← Analisis, Q&A, edit .docx (4 mode)
│   ├── quiz_agent/
│   │   ├── agent.py             ← PDF → soal kuis → WebQuizBuilderTool
│   │   └── HOW_IT_WORKS.md
│   ├── sysinfo_agent/agent.py   ← CPU/RAM/disk via psutil
│   ├── log_viewer_agent/agent.py← Tampilkan log dari ring buffer
│   ├── web_automation/agent.py  ← ReAct loop: navigate/click/fill/screenshot
│   └── reminder/
│       ├── agent.py             ← Set/list/cancel timed reminders
│       └── scheduler.py         ← APScheduler instance (process-wide)
│
├── orchestrator/
│   ├── main_loop.py             ← process_message() – controller utama
│   └── router.py                ← INTENT_AGENT_MAP: intent → agent name
│
├── tools/
│   ├── base_tool.py             ← ABC: semua tool wajib extends ini
│   ├── tavily_search.py         ← Pre-agent: live web search
│   ├── wbs_generator.py         ← Post-agent: build WBS Excel Gantt
│   ├── mandays_generator.py     ← Post-agent: build Mandays Excel
│   ├── diagram_renderer.py      ← Post-agent: render Mermaid → PNG via mmdc
│   ├── document_generator.py    ← Post-agent: Markdown → DOCX/PDF (Pandoc/WeasyPrint)
│   ├── web_quiz_builder.py      ← Post-agent: soal JSON → HTML kuis interaktif
│   ├── browser_navigator.py     ← Playwright controller (klik/type/scroll/screenshot)
│   ├── web_reader.py            ← HTTP page reader + a11y locators
│   ├── pdf_parser.py            ← Ekstraksi teks PDF via PyMuPDF (chunked, anti-OOM)
│   ├── docx_parser.py           ← Ekstraksi seksi/bab dari .docx
│   ├── docx_editor.py           ← Edit .docx via XML operations
│   ├── cli_executor.py          ← Async non-interactive shell runner (timeout 5 mnt)
│   ├── sandbox_runner.py        ← Docker build/run + traceback detection + fallback
│   ├── git_manager.py           ← git add -A → commit → push dengan PAT auth
│   ├── sysinfo_tool.py          ← Metrik CPU/RAM/disk via psutil
│   ├── reminder_store.py        ← SQLite store untuk reminder jobs
│   ├── log_buffer.py            ← In-memory ring buffer log bot
│   ├── repo_qa.py               ← Engine Q&A: extractor API, model, tech stack, dll.
│   ├── identity_generator.py    ← Pembuat identitas acak (untuk web automation)
│   ├── progress_tracker.py      ← Live-update progress message di Telegram
│   ├── wbs/
│   │   ├── generate_wbs.py      ← Core Excel rendering logic (WBS)
│   │   └── extract_wbs.py       ← Excel → JSON (standalone utility)
│   └── mandays/
│       ├── generate_mandays.py  ← Core Excel rendering logic (Mandays)
│       └── extract_mandays.py   ← Excel → JSON (standalone utility)
│
├── memory/
│   ├── state.py                 ← AgentTask dataclass (blackboard)
│   ├── history.py               ← ConversationHistory (in-memory)
│   └── repo_tracker.py          ← RepoTracker: SQLite registry repo yang pernah di-clone
│
├── interfaces/
│   ├── telegram_bot.py          ← Telegram Application builder
│   ├── rest_api.py              ← FastAPI: /chat, /health, /session/{id}
│   └── webhook.py               ← Webhook runner (uvicorn)
│
└── handlers/
    ├── message.py               ← Routing pesan Telegram → process_message()
    └── command.py               ← Handler /start /help /ping /reset
```

---

## 6. Cara Menambah Agent Baru

Contoh: menambah `DocumentAgent` untuk intent `document_analysis`.

### Langkah 1 – Buat file agent

```python
# src/agents/document_agent/agent.py
from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.memory.state import AgentTask

class DocumentAgent(BaseAgent):
    name = "document_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    async def run(self, task: AgentTask) -> AgentTask:
        # Logika agent di sini
        task.mark_done("Dokumen berhasil dianalisis.")
        return task
```

```python
# src/agents/document_agent/__init__.py
# (kosong)
```

### Langkah 2 – Tambah intent ke enum

```python
# src/agents/gatekeeper/schemas.py
class IntentCategory(str, Enum):
    ...
    DOCUMENT_ANALYSIS = "document_analysis"   # ← tambahkan ini
```

### Langkah 3 – Map intent ke agent

```python
# src/orchestrator/router.py
INTENT_AGENT_MAP: dict[IntentCategory, str] = {
    ...
    IntentCategory.DOCUMENT_ANALYSIS: "document_agent",   # ← tambahkan ini
}
```

### Langkah 4 – Register agent di pipeline

```python
# src/orchestrator/main_loop.py  →  fungsi _get_pipeline()
from src.agents.document_agent.agent import DocumentAgent

_agents = {
    ...
    "document_agent": DocumentAgent(_llm),   # ← tambahkan ini
}
```

### Langkah 5 – Update prompt Gatekeeper

```python
# src/agents/gatekeeper/openrouter.py  →  _SYSTEM_PROMPT
# Tambahkan deskripsi intent baru ke daftar intent yang bisa dipilih LLM:
# - document_analysis  (user ingin menganalisis atau merangkum dokumen)
```

---

## 7. Cara Menambah Tool Baru

### Tool Post-agent (dipicu oleh agent via `pending_tools`)

Contoh: `PDFExportTool` yang mengubah WBS JSON menjadi PDF.

```python
# src/tools/pdf_exporter.py
from src.tools.base_tool import BaseTool
from src.memory.state import AgentTask

class PDFExporterTool(BaseTool):
    name = "pdf_exporter"

    async def run(self, task: AgentTask) -> dict:
        data = task.metadata.get("wbs_json_data")
        if not data:
            return {"error": "wbs_json_data tidak ditemukan"}
        
        # ... logika build PDF ...
        pdf_path = "/tmp/wbs_output.pdf"
        return {"pdf_path": pdf_path}
```

Register di orchestrator:

```python
# src/orchestrator/main_loop.py  →  _get_pipeline()
from src.tools.pdf_exporter import PDFExporterTool

_tools = {
    ...
    "pdf_exporter": PDFExporterTool(),   # ← tambahkan ini
}
```

Agent yang ingin memicu tool ini cukup:

```python
task.pending_tools.append("pdf_exporter")
```

### Tool Pre-agent (diputuskan Gatekeeper via `intent_result.tools`)

Gatekeeper menentukan pre-agent tools via LLM. Tidak ada hardcode di kode Python —  
cukup **update prompt Gatekeeper** agar LLM tahu nama tool baru dan kapan menggunakannya.

```python
# src/agents/gatekeeper/openrouter.py  →  _SYSTEM_PROMPT
# Tambahkan ke daftar "Pre-agent tools":
# - "new_tool_name" : deskripsi kapan tool ini digunakan
```

---

## 8. Cara Menambah Intent Baru

Ringkasan 5 langkah (sama seperti di bagian Agent):

| # | File | Aksi |
|---|------|------|
| 1 | `src/agents/gatekeeper/schemas.py` | Tambah value ke `IntentCategory` enum |
| 2 | `src/agents/gatekeeper/openrouter.py` | Tambah deskripsi intent ke `_SYSTEM_PROMPT` |
| 3 | `src/orchestrator/router.py` | Map `IntentCategory.NEW_INTENT` → `"agent_name"` di `INTENT_AGENT_MAP` |
| 4 | `src/agents/<new_agent>/agent.py` | Buat agent baru (extends `BaseAgent`) |
| 5 | `src/orchestrator/main_loop.py` | Register agent baru di `_agents` dict |

---

## 9. Kontrak Antar Komponen

### BaseAgent (src/agents/base_agent.py)

```python
class BaseAgent(ABC):
    name: str                                      # slug unik, wajib di-set
    async def run(task: AgentTask) -> AgentTask:   # wajib implement
```

**Kewajiban implementasi:**
- Selalu panggil `task.mark_done(result_text)` atau `task.mark_failed(reason)` sebelum `return task`
- Simpan data untuk tool di `task.metadata[key]` (bukan langsung tulis file)
- Sinyal tool via `task.pending_tools.append("tool_name")` (bukan panggil tool langsung)

### BaseTool (src/tools/base_tool.py)

```python
class BaseTool(ABC):
    name: str                                          # slug unik, harus sama dengan key di _tools dict
    async def run(task: AgentTask) -> dict[str, Any]:  # wajib implement
```

**Kewajiban implementasi:**
- Selalu return `dict` (tidak boleh raise exception ke atas — tangkap dan return `{"error": "..."}`)
- Baca input dari `task.metadata` atau `task.tool_results`
- Tidak memanggil LLM di dalam tool — LLM hanya di dalam agent
- Tidak memodifikasi `task.result` — itu hak agent

### Gatekeeper → Orchestrator

```python
IntentResult:
  intent:     IntentCategory  # intent tunggal
  confidence: float           # 0.0 – 1.0
  tools:      list[str]       # nama tool pre-agent (boleh kosong [])
```

### Orchestrator → Interface

```python
AgentTask (setelah pipeline selesai):
  task.result              → str, teks reply untuk pengguna
  task.metadata["excel_path"] → str | None, path file Excel jika ada
  task.status              → TaskStatus.DONE atau FAILED
```

---

## 10. Aturan Utama yang Tidak Boleh Dilanggar

```
✅ Orchestrator adalah SATU-SATUNYA yang memanggil tool.run()
✅ Agent hanya menulis task.metadata + task.pending_tools, tidak pernah memanggil tool langsung
✅ Tool tidak boleh memanggil LLM — semua LLM call ada di agent
✅ Tool harus pure/deterministik — input sama → output sama
✅ Semua data antar komponen lewat AgentTask — tidak ada global state / singleton data
✅ Setiap agent baru harus extends BaseAgent
✅ Setiap tool baru harus extends BaseTool
✅ Nama tool di kode (class.name) harus identik dengan key di _tools dict di main_loop.py
✅ Nama tool di pending_tools / intent_result.tools harus identik dengan key di _tools dict
```

> **⚠️ Pengecualian DeveloperAgent:** Aturan "orchestrator sebagai controller" **tidak berlaku** untuk DeveloperAgent. Agent ini memiliki internal tools sendiri (`CLIExecutor`, `SandboxRunner`, `GitManager`) yang dipanggil langsung di dalam `agent.run()` karena eksekusinya bersifat stateful dan sequential (clone → edit → sandbox → push), bukan stateless/parallel seperti tool standar. Ini adalah desain yang disengaja, bukan pelanggaran pola.

---

## 11. Logging & Debugging

### Format Log

Sistem menggunakan `logging` standard Python. Format per komponen:

```
# Gatekeeper → intent
Intent: session=<id> intent=<intent> confidence=<float> tools=<list>

# Agent selesai → pending_tools
Agent done: session=<id> agent=<name> pending_tools=<list>

# Pre-agent tool
Executing tool '<name>' for session=<id>
Tool '<name>' done for session=<id> keys=<list>

# Post-agent tool
Post-agent tool '<name>' starting for session=<id>
Post-agent tool '<name>' done for session=<id> keys=<list>

# Pipeline selesai
pipeline done | session=<id> intent=<intent> agent=<name> status=<status>
```

### Contoh Log Skenario Mandays

```
INFO  Intent: session=6478491074 intent=mandays_planning confidence=0.98 tools=[]
INFO  Router: session=6478491074 intent=mandays_planning → agent=mandays_agent
INFO  Agent done: session=6478491074 agent=mandays_agent pending_tools=['mandays_generator']
INFO  Post-agent tool 'mandays_generator' starting for session=6478491074
INFO  Post-agent tool 'mandays_generator' done for session=6478491074 keys=['excel_path', 'project_name']
INFO  pipeline done | session=6478491074 intent=mandays_planning agent=mandays_agent status=done
```

### Contoh Log Skenario Research (dengan Tavily)

```
INFO  Intent: session=6478491074 intent=research confidence=0.93 tools=['tavily_search']
INFO  Executing tool 'tavily_search' for session=6478491074
INFO  Tool 'tavily_search' done for session=6478491074 keys=['context_text', 'results_count']
INFO  Router: session=6478491074 intent=research → agent=researcher
INFO  Agent done: session=6478491074 agent=researcher pending_tools=[]
INFO  pipeline done | session=6478491074 intent=research agent=researcher status=done
```

### Contoh Log Skenario Code Development (DeveloperAgent)

```
INFO  Intent: session=6478491074 intent=code_development confidence=0.96 tools=[]
INFO  Router: session=6478491074 intent=code_development → agent=developer
INFO  DeveloperAgent: LLM-direct mode active (no claude CLI found; using OpenRouter)
INFO  DeveloperAgent: cloning https://github.com/owner/repo.git → /home/user/sandbox_repos/owner-repo
INFO  DeveloperAgent: edit attempt 1/3
INFO  DeveloperAgent: LLM-direct applied 2 file patch(es)
INFO  CLIExecutor.run cwd=/home/user/sandbox_repos/owner-repo cmd=docker compose up --build --abort-on-container-exit
INFO  DeveloperAgent: sandbox green on attempt 1
INFO  GitManager: pushed commit=a3f1c9b to https://***@github.com/owner/repo.git
INFO  pipeline done | session=6478491074 intent=code_development agent=developer status=done
```

### Tool Tidak Terdaftar

Jika `pending_tools` berisi nama yang tidak ada di `_tools` dict:
```
WARNING  pending_tools: 'unknown_tool' not in registry; skipping.
```

Artinya: nama di `task.pending_tools.append(...)` tidak cocok dengan key di `_tools` dict di `main_loop.py`.
