# Tools API Reference

Dokumen ini adalah acuan wajib sebelum menggunakan internal tools dalam pengembangan agent baru.
Bertujuan mencegah kesalahan parameter seperti `unexpected keyword argument`.

---

## `CLIExecutor` — `src/tools/cli_executor.py`

Async wrapper untuk menjalankan shell command.

### Constructor

```python
CLIExecutor(
    work_dir: Path | str | None = None,  # default CWD untuk semua perintah
    timeout:  int               = 300,   # detik; batas waktu kill proses
    auto_yes: bool              = True,  # auto-answer Y/N prompts via stdin
)
```

> ⚠️ **`timeout` hanya bisa di-set di constructor, BUKAN di `.run()`.**

### `.run()` method

```python
await executor.run(
    command:  str | Sequence[str],      # perintah shell atau argv list
    *,
    work_dir: Path | str | None = None, # override CWD khusus pemanggilan ini
    env:      dict[str, str] | None = None,
) -> CommandResult
```

> ⚠️ **Parameter CWD bernama `work_dir`, BUKAN `cwd`.**

### Contoh benar

```python
# Konstruksi dengan timeout pendek
cli = CLIExecutor(timeout=30)

# Override work_dir per pemanggilan
result = await cli.run("git log --oneline -n 10", work_dir=repo_path)
result = await cli.run("ls -la",                  work_dir=Path("/tmp/myrepo"))
```

### Contoh salah (akan raise TypeError)

```python
# ❌ 'cwd' tidak dikenal
await cli.run("git status", cwd=repo_path)

# ❌ 'timeout' bukan parameter .run()
await cli.run("git clone ...", timeout=60)
```

### `CommandResult` properties

| Property | Type | Keterangan |
|---|---|---|
| `command` | `str` | Perintah yang dijalankan |
| `returncode` | `int` | Exit code (0 = sukses) |
| `stdout` | `str` | Standard output |
| `stderr` | `str` | Standard error |
| `timed_out` | `bool` | True jika proses di-kill karena timeout |
| `succeeded` | `bool` | `returncode == 0 and not timed_out` |
| `combined_output` | `str` | stdout + stderr digabung |

---

## `GitManager` — `src/tools/git_manager.py`

High-level git operations (commit + push). Untuk **write** operations saja.

### Constructor

```python
GitManager(repo_path: Path)  # path lokal repo
```

### `.commit_and_push()` method

```python
result: GitPushResult = await gm.commit_and_push(commit_message: str)
```

### `GitPushResult` properties

| Property | Type | Keterangan |
|---|---|---|
| `committed` | `bool` | True jika `git commit` sukses |
| `pushed` | `bool` | True jika `git push` sukses |
| `commit_hash` | `str` | Short hash commit baru |
| `remote_url` | `str` | Remote URL (PAT diredact) |
| `error` | `str` | Pesan error jika gagal |
| `succeeded` | `bool` | `committed and pushed` |

> ⚠️ **`DeveloperInspectorAgent` TIDAK BOLEH menggunakan `GitManager`.**
> GitManager hanya untuk `DeveloperAgent` (write operations).

---

## `RepoTracker` — `src/memory/repo_tracker.py`

Menyimpan mapping `repo_url → local_path` ke disk (JSON).

```python
tracker = RepoTracker()
tracker.upsert(repo_url: str, local_path: str)
repos = tracker.list_repos()  # -> list[dict]
# dict keys: "repo_url", "local_path", "updated_at"
```

---

## `LLMClient` — `src/agents/llm_client.py`

Thin wrapper untuk memanggil LLM via OpenRouter.

```python
llm = LLMClient()

response: str = await llm.complete(
    messages: list[dict],   # [{"role": "system"|"user"|"assistant", "content": "..."}]
)
```

---

## `TavilySearchTool` — `src/tools/tavily_search.py`

Live web search. Hanya tersedia jika `TAVILY_API_KEY` di-set.

```python
tool     = TavilySearchTool()
response = await tool.search(query: str)
context  = response.as_context_text()  # str siap inject ke LLM
```

> Prefer menggunakan hasil pre-fetch dari `task.tool_results["tavily_search"]["context_text"]`
> ketimbang memanggil langsung (lihat `ResearcherAgent` sebagai contoh).

---

## `AgentTask` — `src/memory/state.py`

Blackboard yang mengalir antar komponen pipeline.

```python
task.mark_done(result: str)    # set status DONE + isi task.result
task.mark_failed(reason: str)  # set status FAILED + isi task.result
task.mark_routed(intent: str)  # dipanggil oleh orchestrator, bukan agent

task.user_input    # str — input asli user
task.session_id    # str
task.intent        # str — nilai IntentCategory
task.result        # str — output final yang dikirim ke user
task.tool_results  # dict[str, dict] — hasil tools
task.pending_tools # list[str] — tools yang dieksekusi post-agent
task.metadata      # dict — data bebas antar komponen
```

---

## Checklist Pembuatan Agent Baru

Gunakan checklist ini setiap kali membuat agent:

- [ ] Buat `src/agents/<nama>/agent.py` + `__init__.py`
- [ ] Inherit dari `BaseAgent`, implementasikan `async def run(self, task: AgentTask) -> AgentTask`
- [ ] Set `name = "<nama>"` sebagai class attribute
- [ ] Selalu panggil `task.mark_done(...)` atau `task.mark_failed(...)` sebelum `return task`
- [ ] Tambahkan `IntentCategory.<NAMA> = "<nama>"` di `src/agents/gatekeeper/schemas.py`
- [ ] Daftarkan di `INTENT_AGENT_MAP` di `src/orchestrator/router.py`
- [ ] Import & daftarkan di `_agents` dict di `src/orchestrator/main_loop.py`
- [ ] Update system prompt gatekeeper di `src/agents/gatekeeper/openrouter.py` (intent list + rules)
- [ ] Verifikasi parameter API tool sesuai dokumen ini sebelum commit

---

## Aturan Khusus: `DeveloperInspectorAgent`

Agent ini adalah **READ-ONLY**. Larangan keras:

| Larangan | Alasan |
|---|---|
| `GitManager.commit_and_push()` | Akan menulis ke repo |
| `SandboxRunner` | Eksekusi kode |
| Perintah write: `echo > file`, `sed -i`, dll | Mengubah state repo |
| `git add`, `git commit`, `git push` | Write to VCS |

Perintah yang **boleh** digunakan:
`git log`, `git diff`, `git show`, `git status`, `find`, `grep`, `cat`, `head`, `tail`, `ls`, `wc`

## New Tools & Updates (2026-03-12)

- **ProgressTracker** — `src/tools/progress_tracker.py`: tool ringan untuk melacak progres tugas panjang; API ringkas: buat instance, lalu `update(step: str, percent: int)` untuk mengirim status.
- **SandboxRunner (update):** peningkatan deteksi traceback, dukungan retry otomatis, dan batasan resource dasar (memori/CPU) untuk container sandbox.
- **RepoTracker (runtime):** storage default dimigrasi ke SQLite untuk keandalan; metode publik tetap `upsert()` dan `list_repos()`.
- **GitManager (update):** penambahan dukungan pembuatan branch dan perbaikan redaksi PAT saat mengembalikan `remote_url` di hasil operasi.
- **TavilySearchTool (update):** hasil pencarian sekarang dicache dan menyertakan `context_text`; prefer menggunakan `task.tool_results` yang sudah di-prefetch.
