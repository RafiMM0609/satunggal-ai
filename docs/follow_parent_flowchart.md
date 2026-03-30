# follow_parent — Flowchart & Feature Description

## Deskripsi

`follow_parent` adalah fitur khusus untuk **Web Automation Agent**.
Setelah sebuah tugas web automation selesai, browser **tidak ditutup** — halaman
terakhir yang dibuka tetap aktif di memori.  Permintaan berikutnya dari sesi yang
sama langsung berinteraksi dengan halaman yang sudah terbuka tanpa perlu navigasi
ulang.  Fitur ini dinonaktifkan (browser ditutup) ketika pengguna mengirim perintah
`/reset` di Telegram.

---

## Flowchart — Alur follow_parent

```
┌─────────────────────────────────────────────────────────────────┐
│                     TELEGRAM USER                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ Kirim pesan (teks)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  echo_text handler                              │
│           (src/handlers/message.py)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ process_message(session_id, user_text)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               GatekeeperAgent – Intent Classification           │
│           Apakah intent == WEB_AUTOMATION ?                     │
└──────────────┬──────────────────────────────────────────────────┘
               │ Ya → intent = WEB_AUTOMATION
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AgentRouter                                  │
│         Pilih WebAutomationAgent                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebAutomationAgent.run(task)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Cek _session_navigator[session_id]                   │
│       Apakah ada browser persisten yang masih terbuka?          │
└──────────────┬──────────────────────┬──────────────────────────┘
               │ TIDAK                │ YA (follow_parent aktif)
               ▼                      ▼
      BrowserNavigatorTool()   Reuse navigator yang ada
      (browser baru)           (browser sudah di halaman terakhir)
               │                      │
               └──────────┬───────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ReAct Loop (maks 20 langkah)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  _plan_next_step(user_input, session_id, steps_done,     │   │
│  │                  follow_parent=True/False)                │   │
│  │                                                          │   │
│  │  Konteks yang dikirim ke LLM:                            │   │
│  │  • follow_parent=True  → "Browser sudah terbuka di       │   │
│  │    {last_url}, langsung interaksi tanpa navigate ulang"  │   │
│  │  • follow_parent=False → "URL terakhir: {last_url}"      │   │
│  └───────────────────────────┬──────────────────────────────┘   │
│                              │ JSON action (navigate/click/…)   │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              _execute_step(action, params)               │   │
│  │   BrowserNavigatorTool / WebReaderTool                   │   │
│  └───────────────────────────┬──────────────────────────────┘   │
│                              │ result                           │
│         ┌────────────────────┤                                  │
│         │ action == "done"?  │                                  │
│         ▼ Ya                 ▼ Tidak → lanjut loop              │
│       break                                                     │
└──────────────┬──────────────────────────────────────────────────┘
               │ Setelah loop selesai
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              _summarise() – ringkasan untuk user                │
│              task.mark_done(reply)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │ finally block
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Auto-save session (cookies/localStorage ke disk)       │
│                                                                 │
│          Apakah final_url ada & halaman masih terbuka?          │
└──────────────┬──────────────────────────────────────────────────┘
               │ YA
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  SET follow_parent = True                                       │
│  _session_navigator[session_id] = navigator  (simpan instance)  │
│  _session_follow_parent[session_id] = True                      │
│  task.metadata["follow_parent"] = True                          │
│                                                                 │
│  ⚠ Browser TIDAK ditutup — tetap terbuka di halaman terakhir   │
└────────────────────────────┬────────────────────────────────────┘
                             │ return task
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Kirim hasil ke Telegram (teks + screenshot)            │
│                                                                 │
│  Permintaan berikutnya dari sesi ini akan langsung masuk        │
│  ke browser yang sudah terbuka (follow_parent loop di atas)     │
└────────────────────────────┬────────────────────────────────────┘
                             │ User kirim /reset
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  /reset command handler                                         │
│  await clear_session(session_id)                                │
│    └─► await clear_web_automation_session(session_id)           │
│          • await navigator.close()  ← TUTUP BROWSER            │
│          • _session_navigator.pop(session_id)                   │
│          • _session_follow_parent.pop(session_id)               │
│          • _session_last_url.pop(session_id)                    │
│          • Hapus file sesi browser dari disk                    │
│  clear_doc_session(session_id)                                  │
│  history.clear(session_id)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ringkasan State yang Dikelola

| Variabel modul                  | Keterangan                                                   | Dibersihkan saat |
|---------------------------------|--------------------------------------------------------------|------------------|
| `_session_follow_parent`        | Dict `{session_id: True}` jika follow_parent aktif          | `/reset`         |
| `_session_navigator`            | Dict `{session_id: BrowserNavigatorTool}` instance hidup    | `/reset`         |
| `_session_last_url`             | Dict `{session_id: url}` URL terakhir yang dikunjungi       | `/reset`         |
| `_session_domains`              | Dict `{session_id: set(base_url)}` domain yang dikunjungi   | `/reset`         |

---

## Catatan Implementasi

- `clear_web_automation_session()` dijadikan **async** agar dapat menutup browser
  dengan `await navigator.close()`.
- `clear_session()` di `main_loop.py` kini menggunakan `await` saat memanggil
  `clear_web_automation_session()`.
- Fitur ini **hanya aktif untuk WebAutomationAgent** — agent lain tidak terpengaruh.
- `task.metadata["follow_parent"] = True` dikirimkan di setiap respons agar
  lapisan interface dapat mengetahui status browser jika diperlukan.
- Jika browser mengalami crash atau halaman tertutup secara tidak terduga,
  `page.is_closed()` mendeteksi kondisi ini dan memulai browser baru secara otomatis.
