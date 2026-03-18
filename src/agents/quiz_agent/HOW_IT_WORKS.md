# 📚 PDF-to-Web Quiz Generator — HOW IT WORKS

> **Agen**: `quiz_agent` | **Fitur**: PDF → Kuis HTML Interaktif Single-File

---

## 🏗️ Apa yang Sudah Dibangun

### Komponen Baru

| File | Fungsi |
|---|---|
| `src/agents/quiz_agent/__init__.py` | Package init |
| `src/agents/quiz_agent/agent.py` | QuizAgent – batch LLM processing |
| `src/tools/pdf_parser.py` | PDFParserTool – ekstraksi teks via PyMuPDF |
| `src/tools/web_quiz_builder.py` | WebQuizBuilderTool – generator HTML quiz |

### Komponen yang Dimodifikasi

| File | Perubahan |
|---|---|
| `src/agents/gatekeeper/schemas.py` | Tambah `QUIZ_GENERATION` ke `IntentCategory` |
| `src/orchestrator/router.py` | Mapping `quiz_generation` → `quiz_agent` |
| `src/orchestrator/main_loop.py` | Registrasi agent/tools + fungsi `process_pdf_quiz()` |
| `src/tools/progress_tracker.py` | Tambah stage label khusus kuis |
| `src/handlers/message.py` | Tambah `handle_pdf_document()` handler |
| `src/handlers/__init__.py` | Export `handle_pdf_document` |
| `src/interfaces/telegram_bot.py` | Registrasi handler `filters.Document.PDF` |
| `requirements.txt` | Tambah `PyMuPDF>=1.24.0` |

---

## 🔄 Alur Kerja Teknis

```
User kirim PDF via Telegram
        │
        ▼
handle_pdf_document() [handlers/message.py]
  • Download PDF ke /tmp/advance_ai_pdf_uploads/
  • Tampilkan progress message (live editing)
        │
        ▼
process_pdf_quiz() [orchestrator/main_loop.py]
  • Bypass gatekeeper (intent sudah jelas: quiz_generation)
        │
        ├─► PDFParserTool [tools/pdf_parser.py]
        │     • Buka PDF dengan PyMuPDF (fitz)
        │     • Baca 10 halaman per iterasi (anti-OOM)
        │     • Bersihkan teks dari artefak OCR
        │     • Bagi menjadi chunk ~2.000 kata
        │     → task.metadata["pdf_chunks"] = [chunk1, chunk2, ...]
        │
        ├─► QuizAgent [agents/quiz_agent/agent.py]
        │     • Loop setiap chunk secara SEQUENTIAL
        │     • Panggil LLM (temperature=0.3) untuk 10-15 soal per batch
        │     • Akumulasi soal di task.metadata["quiz_questions"]
        │     • Update progress via status_callback
        │     • Hapus chunk dari memori setelah diproses
        │     → task.pending_tools = ["web_quiz_builder"]
        │
        └─► WebQuizBuilderTool [tools/web_quiz_builder.py]
              • Inject soal JSON ke HTML template
              • Template: Tailwind CSS (CDN) + Alpine.js v3 (CDN)
              • Tulis ke /tmp/advance_ai_quiz/quiz_<session>_<ts>.html
              → task.metadata["html_path"] = "/tmp/.../quiz.html"

        │
        ▼
handle_pdf_document() mengirim file .html ke Telegram user
```

---

## 📦 Output HTML

File HTML yang dihasilkan bersifat **self-contained** (satu file) dan memiliki fitur:

| Fitur | Detail |
|---|---|
| ✅ Responsive Design | Optimal di HP dan laptop (via Tailwind CSS) |
| ✅ Dark Mode | Toggle manual + deteksi preferensi sistem |
| ✅ Instant Feedback | Hijau (benar) / Merah (salah) langsung setelah memilih |
| ✅ Penjelasan Jawaban | Muncul otomatis setelah menjawab setiap soal |
| ✅ Final Scoreboard | Nilai persentase + pesan motivasi + ringkasan benar/salah |
| ✅ Review Mode | Lihat semua jawaban setelah kuis selesai |
| ✅ Progress Bar | Persentase soal yang sudah dijawab di header |
| ✅ Ulangi Kuis | Tombol restart tanpa reload halaman |

---

## 🧠 System Prompt QuizAgent

QuizAgent menggunakan system prompt ketat yang memastikan:

- **Format JSON valid**: Output HANYA berupa array JSON, tidak ada teks tambahan
- **4 pilihan jawaban**: Setiap soal WAJIB punya tepat 4 opsi (A, B, C, D)
- **Kunci jawaban akurat**: Field `correct` adalah index integer (0-3), divalidasi ulang
- **Distribusi kesulitan**: 30% mudah, 50% sedang, 20% sulit
- **Anti-hallucination**: Temperature rendah (0.3) untuk konsistensi format
- **Penjelasan wajib**: Setiap soal disertai field `explanation`

---

## 💾 Strategi Anti-Crash (VPS 2 GB RAM)

| Strategi | Implementasi |
|---|---|
| **Sequential Batch** | Setiap chunk diproses satu per satu, tidak paralel |
| **Cache Clearing** | `chunks[batch_idx - 1] = ""` setelah chunk selesai diproses |
| **Page Grouping** | PyMuPDF membaca max. 10 halaman per iterasi |
| **File Cleanup** | PDF sumber dihapus segera setelah parsing selesai |
| **HTML Cleanup** | File HTML dihapus dari server setelah berhasil dikirim ke user |
| **Size Guard** | Tolak PDF > 20 MB sebelum diproses |

---

## 📊 Format Soal JSON

```json
[
  {
    "id": 1,
    "question": "Apa yang dimaksud dengan ...",
    "options": [
      "A. Pilihan pertama",
      "B. Pilihan kedua",
      "C. Pilihan ketiga",
      "D. Pilihan keempat"
    ],
    "correct": 2,
    "explanation": "Penjelasan singkat mengapa C adalah jawaban benar."
  }
]
```

---

## 🚀 Cara Menggunakan

1. Buka bot di Telegram
2. Kirim file PDF (maks. 20 MB) langsung ke bot — **tidak perlu ketik perintah apapun**
3. Bot akan menampilkan progress real-time yang diperbarui setiap batch
4. Setelah selesai, bot mengirimkan file `.html` yang bisa langsung dibuka di browser

---

## ⚠️ Keterbatasan & Yang Perlu Ditambahkan

### 🔴 Kritis (Perlu Segera)

1. **PDF Berbasis Gambar (Scan)**: PDFParserTool saat ini **tidak mendukung** PDF scan.
   - **Solusi yang diperlukan**: Integrasi OCR seperti `pytesseract` atau Tesseract CLI
   - Atau layanan cloud OCR (Google Vision API, AWS Textract)

2. **Batas Token LLM**: Untuk PDF sangat panjang, chunk besar mungkin melebihi context window LLM.
   - **Solusi**: Tambah validasi ukuran chunk berdasarkan model yang digunakan

3. **Keamanan File PDF**: Tidak ada sanitasi konten PDF untuk mencegah injection attacks.
   - **Solusi**: Validasi MIME type secara server-side, bukan hanya dari Telegram filter

### 🟡 Penting (Untuk Produksi)

4. **Persistensi State**: Jika bot restart saat memproses PDF besar, progress hilang.
   - **Solusi**: Simpan `quiz_questions` yang sudah terkumpul ke SQLite setelah setiap batch
   - Implementasi: Update `src/memory/state.py` atau buat `src/memory/quiz_store.py`

5. **Rate Limiting**: Tidak ada pembatasan jumlah PDF yang bisa dikirim satu user.
   - **Solusi**: Tambah rate limiter per user_id di handler

6. **Konfigurasi Chunk Size**: `_WORDS_PER_CHUNK = 2000` hardcoded.
   - **Solusi**: Ekspos ke `config/settings.py` sebagai `QUIZ_WORDS_PER_CHUNK`

7. **Validasi PDF Terenkripsi**: PDF dengan password akan gagal tanpa pesan error yang jelas.
   - **Solusi**: Cek `doc.is_encrypted` dan berikan pesan yang informatif

### 🟢 Nice to Have (Enhancement)

8. **Pemilihan Jumlah Soal**: User tidak bisa memilih berapa soal yang dibuat.
   - **Solusi**: Parse pesan caption PDF untuk opsi seperti "50 soal" atau "100 soal"

9. **Pilihan Bahasa**: Soal selalu dalam Bahasa Indonesia.
   - **Solusi**: Deteksi bahasa PDF dan sesuaikan system prompt

10. **Export ke Format Lain**: Soal hanya bisa diakses via HTML.
    - **Solusi**: Tambah opsi export ke PDF ujian atau format GIFT (Moodle)

11. **Kategorisasi Soal**: Semua soal flat tanpa kategorisasi per topik/bab.
    - **Solusi**: Modifikasi prompt untuk menambah field `"topic"` dan filter soal per topik di UI

12. **Timer Kuis**: Tidak ada fitur waktu pengerjaan.
    - **Solusi**: Tambah Alpine.js countdown timer ke template HTML

13. **True Offline (No CDN)**: Template HTML masih menggunakan CDN untuk Tailwind dan Alpine.js.
    - **Solusi**: Bundle Tailwind CSS yang diperlukan secara lokal (PurgeCSS) dan inline Alpine.js minified

14. **Analytics**: Tidak ada tracking waktu pengerjaan per soal.
    - **Solusi**: Tambah logging di Alpine.js dan kirim summary ke server

---

## 🔧 Dependensi Baru

```
PyMuPDF>=1.24.0  # pip install PyMuPDF
```

Di luar Python, tidak ada dependensi sistem tambahan yang diperlukan untuk fitur dasar ini.

---

## 🧪 Testing

Untuk menguji fitur ini secara lokal:

```bash
# 1. Install dependensi baru
pip install PyMuPDF

# 2. Jalankan test unit (jika ada)
pytest tests/

# 3. Test manual via Telegram:
#    - Jalankan bot: python main.py
#    - Kirim file PDF ke bot
#    - Periksa apakah HTML quiz terkirim kembali
```

Test unit yang sudah ada di `tests/` masih berjalan tanpa perubahan karena fitur baru ini
tidak memodifikasi agent yang sudah ada, hanya menambah komponen baru.
