# Excel to JSON

Skrip ini membaca file Excel (.xlsx) dan mengubahnya menjadi format JSON seperti contoh yang diberikan.

Asumsi tata letak (default):
- Kolom A: `Category`
- Kolom B: `Sub Category` (opsional)
- Kolom C: `Task Name`
- Kolom D ke kanan: header berisi tanggal (angka 1-31 atau tanggal Excel). Sel yang berwarna (tidak putih) pada baris `Task Name` menandakan `active_days`.
- Informasi project (opsional) dapat ditaruh pada kolom A/B di baris atas, misal `Project Name`, `Start Date`, `End Date`.

Cara pakai:

```bash
python3 excel_to_json.py path/to/input.xlsx -o output.json
```

Opsi:
- `-s/--sheet`: nama sheet jika bukan sheet pertama
- `-o/--output`: file JSON output; kalau tidak diberikan, hasil dicetak ke stdout

Catatan:
- Deteksi warna menggunakan `openpyxl` — beberapa format warna (tema/indexed) tetap dianggap berwarna.
- Jika struktur file Anda berbeda, sesuaikan kolom `category_col`, `sub_col`, `task_col` di file `excel_to_json.py`.
