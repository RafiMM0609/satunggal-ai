# WBS JSON <-> Excel (Generator)

This repository contains scripts to extract a Work Breakdown Structure (WBS) from an Excel file into JSON, and to generate a styled Excel WBS from a JSON file.

Files
- `extract_wbs.py` — (existing) extracts WBS from an Excel file to JSON.
- `generate_wbs.py` — generates `WBS_Generated.xlsx` from `output.json`.
- `wbs_template.json` — example/template JSON you can edit.
- `output.json` — default input JSON read by `generate_wbs.py`.

Prerequisites
- Python 3.8+ (this workspace uses a venv at `.venv`).
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Or, if you use the repo venv provided here, run:

```bash
"/home/anton/Koding/ngawur/excel/Untitled Folder/.venv/bin/python" -m pip install -r requirements.txt
```

How to extract WBS from Excel to JSON

1. Prepare your Excel file (example: `contoh.xlsx`).
2. Run the extractor:

```bash
# Generic
python extract_wbs.py <input.xlsx> <output.json>

# Using workspace venv (example)
"/home/anton/Koding/ngawur/excel/Untitled Folder/.venv/bin/python" extract_wbs.py contoh.xlsx output.json
```

After running, `output.json` will contain the WBS JSON used by the generator.

How to generate the styled Excel WBS from JSON

1. Ensure `output.json` exists (you can also copy `wbs_template.json` to `output.json`):

```bash
cp wbs_template.json output.json
```

2. Run the generator:

```bash
# Generic
python generate_wbs.py

# Using workspace venv (example)
"/home/anton/Koding/ngawur/excel/Untitled Folder/.venv/bin/python" generate_wbs.py
```

Output: `WBS_Generated.xlsx` will be created in the current folder.

Notes & tips
- Edit `wbs_template.json` to match your project details and roles, then copy to `output.json` and run the generator.
- The generator uses `openpyxl` for styling. If you see errors about missing packages, install them via `pip install -r requirements.txt`.
- Column widths and colors are set to match the project formatting; adjust `generate_wbs.py` if you need different colors/widths.

Troubleshooting
- If `generate_wbs.py` fails with JSON decoding errors, validate `output.json` with an online JSON validator or `jq`.
- If the extractor script expects specific Excel layout, open `extract_wbs.py` to confirm input format.

Contact
- If you want me to extend the generator (e.g., add formulas, CSV export, or different color themes), tell me what to add.