Panduan Menjalankan skrip

File penting
- `extract_wbs.py` : Parser yang membaca file Excel WBS dan menghasilkan `output.json`.
- `generate_wbs.py` : Generator yang membuat file Excel (`.xlsx`) dari `output.json`.

Prasyarat
- Python 3.8+ (disarankan environment virtual)
- Dependensi: lihat `requirements.txt` (mengandung `openpyxl`)

Instalasi (virtualenv/venv contoh)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Menjalankan parser (`extract_wbs.py`)

1. Pastikan file Excel tersedia (mis. `contoh.xlsx`).
2. Jalankan:

```bash
python extract_wbs.py contoh.xlsx -o output.json
```

- Output: `output.json` berisi struktur JSON (`project_info`, `timeline_config`, `wbs_data`).

Menjalankan generator (`generate_wbs.py`)

1. Pastikan `output.json` (format seperti contoh) tersedia.
2. Jalankan:

```bash
python generate_wbs.py output.json wbs_output.xlsx
```

-- Output: `wbs_output.xlsx` (format Excel WBS). Skrip ini menempatkan timeline mulai dari kolom D sehingga `extract_wbs.py` dapat mendeteksinya.

Validasi roundtrip (disarankan)

Untuk memastikan hasil generator sesuai input, jalankan parser pada file hasil generator:

```bash
python extract_wbs.py wbs_output.xlsx -o roundtrip.json
```

Lalu bandingkan `roundtrip.json` dengan `output.json` (misal menggunakan `diff` atau script Python). Jika identik, roundtrip berhasil.

Masalah umum & troubleshooting

- Parser tidak menemukan header tanggal:
  - Pastikan baris "Bulan / Sprint / Tanggal" berada pada baris atas dan kolom tanggal dimulai di kolom D.
  - Pastikan tanggal pada baris header adalah angka (1..31).

- `end_date` menjadi `null` di output:
  - Parser mencari dua entri bertanda `Start Date` pada kolom A (baris 1..9). Jika generator menulis label `End Date`, parser bisa melewatkannya. Panduan ini membuat generator menulis label `Start Date` pada A3 untuk kompatibilitas.

- Warna sel tidak terdeteksi:
  - Pastikan fill color dalam Excel adalah solid fill dengan kode warna yang dipakai:
    - Category header: `#92D050` (hijau)
    - Sub-category header: `#B7DEE8` (biru muda)
    - Active timeline cell: `#31869B` (biru tua)

Tips
- Untuk debugging cepat, buka `roundtrip.json` dan bandingkan terhadap `output.json`.
-- Jika timeline Anda mulai di kolom selain D, sesuaikan `DATA_COL_OFFSET` pada `generate_wbs.py` atau kode `find_timeline_header` pada `extract_wbs.py`.

Kontak / Notes
-- File generator: [generate_wbs.py](generate_wbs.py)
-- File parser: [extract_wbs.py](extract_wbs.py)

Selesai.