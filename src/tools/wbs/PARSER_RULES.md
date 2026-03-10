# Parser rules — Excel → JSON

Dokumentasi ini menjelaskan aturan-aturan yang saat ini diterapkan oleh skrip `excel_to_json.py` untuk mengonversi file Excel proyek (contoh: `contoh.xlsx`) menjadi format JSON. File ini menjadi sumber kebenaran ketika melakukan perubahan agar kita bisa kembali jika perkembangan berikutnya membuat hasil lebih buruk.

## Ringkasan output
- `project_info`: `project_name`, `start_date`, `end_date` (YYYY-MM-DD bila dapat dikonversi).
- `timeline_config`: `month`, `sprint_name`, `dates` (list angka hari dari header tanggal).
- `wbs_data`: list kategori. Setiap kategori memiliki `is_header: true` dan salah satu dari:
  - `tasks`: list tugas (jika kategori tanpa sub-categories)
  - `sub_categories`: list objek `{ "name": ..., "tasks": [...] }`

Contoh struktur akhir (dipertahankan):

```json
{
  "project_info": { ... },
  "timeline_config": { ... },
  "wbs_data": [ /* categories ... */ ]
}
```

## Lokasi default dalam sheet
- `wbs_start_row` default: baris 6 (baris awal WBS / daftar kategori dan tugas).
- `date_header_row` default: baris 5 (baris yang berisi angka tanggal seperti 18,19,20,...).
- Kolom tanggal biasanya dimulai dari kolom B (index 2) dan dapat meluas ke kanan.

Jika tata letak berbeda, parameter di fungsi `parse_excel_to_json()` dapat disesuaikan.

## Deteksi metadata project
- Pencarian `project_name`, `start_date`, `end_date` dilakukan pada baris 1..9 kolom A/B.
- Jika sel berformat `Label : Value` (mis. `Nama Project : WATI CR`), parser membaca di sel yang sama.
- Untuk tanggal berbentuk teks dengan nama bulan Indonesia (contoh: `18 Februari 2026`) parser mencoba mengubah ke format `YYYY-MM-DD`.

## Deteksi header tanggal / timeline
- Parser mencari baris yang berisi banyak angka 1..31 (atau nilai tanggal Excel) untuk menentukan kolom-kolom tanggal.
- Setelah ditemukan, `dates` adalah daftar nilai hari (integer) yang berkorespondensi dengan kolom-kolom tersebut.
- `month` dicari dengan memindai baris-baris di atas baris header tanggal untuk menemukan nama bulan Indonesia (Januari..Desember). Jika ditemukan, `timeline_config.month` diisi.
- `sprint_name` dicari di sekitar area header (baris atas timeline), tetapi jika tidak ditemukan persis, nilainya bisa `null`.

## Pembacaan WBS (Category / Sub-Category / Tasks)
- Parser membaca kolom A (seluruhnya) dan menggunakan aturan warna untuk membedakan baris:
  - Category: sel kolom A berisi warna hijau `#92D050` (di workbook ini muncul sebagai `FF92D050` atau encoded index). Ketika terdeteksi, buat entri kategori baru.
  - Sub-Category: sel kolom A berwarna biru muda `#B7DEE8` (muncul sebagai `FFB7DEE8` atau encoding terkait). Ketika terdeteksi, buat sub-category di dalam kategori saat ini.
  - Task: baris-barus berikutnya (kolom A berisi teks) sampai bertemu sub-category/category berikutnya dianggap task. Jika tidak ada sub-category aktif, task dimasukkan pada sub-category bernama `General` di dalam kategori saat ini.

## Deteksi active_days
- Untuk setiap task, parser memeriksa tiap kolom tanggal (kolom yang diidentifikasi pada header).
- Jika sel pada baris task tersebut memiliki fill berwarna biru aktif, tanggal dari header diambil dan dimasukkan ke `active_days` untuk task tersebut.

Warna biru aktif yang dikenali (karena variasi encoding Excel / tema) meliputi:
- literal RGB/HEX: `#31869B` (juga `FF31869B`) — warna biru gelap yang umum.
- beberapa sel di workbook contoh muncul sebagai `#00B0F0` (juga `FF00B0F0`) atau encoded sebagai indexed/theme value (mis. index `7`).

Parser saat ini mencocokkan beberapa representasi (rgb string berakhiran `31869B` atau `00B0F0`, `FF`-prefixed, atau index==7) supaya mendukung encoding workbook berbeda.

Jika Anda ingin pembatasan ketat (hanya pas hex tertentu), ubah fungsi-fungsi deteksi warna di `excel_to_json.py`.

## Kekhususan implementasi saat ini
- Semua pembacaan warna menggunakan `openpyxl` `cell.fill.start_color` dan `cell.fill.fgColor` dan mencoba beberapa properti: `rgb`, `index`, `theme`.
- Task name diambil dari kolom A (kolom WBS); kolom tanggal dipetakan menggunakan baris header yang berisi angka-angka hari.
- Jika header tanggal memiliki nilai berupa `datetime`, parser mengambil `day` (angka).

## Debug & troubleshooting
- Jika `active_days` tidak terisi:
  - Periksa apakah warna sel di Excel menggunakan theme/indexed color. Jalankan skrip inspeksi (sudah tersedia di repo) untuk menampilkan `start_color.rgb` / `start_color.index` dari beberapa sel contoh.
  - Sesuaikan fungsi `is_active_blue()` di `excel_to_json.py` untuk menambahkan encoding warna yang tergolong di file Anda.
- Jika `month` atau `sprint_name` `null`, periksa posisi label ‘Februari’ / ‘Sprint 1’ di sheet — parser mencari di area header; jika label berada di sel yang berbeda, sesuaikan pemindaian.

## Konfigurasi & eksekusi
- Dependencies: `openpyxl` (lihat `requirements.txt`).
- Jalankan:

```bash
python3 excel_to_json.py path/to/contoh.xlsx -o output.json
```

atau gunakan fungsi langsung dari Python:

```py
from excel_to_json import parse_excel_to_json
res = parse_excel_to_json('contoh.xlsx')
```

## Catatan versi / revert
- Setiap perubahan pada file `excel_to_json.py` harus diuji dengan file contoh (`contoh.xlsx`) dan dibandingkan dengan `PARSER_RULES.md`. Jika perubahan membuat keluaran menurun, gunakan perubahan di file ini untuk mengembalikan logika deteksi warna atau posisi kolom.

---
File ini dibuat sebagai checkpoint dokumentasi aturan parsing pada saat pengembangan ini.
