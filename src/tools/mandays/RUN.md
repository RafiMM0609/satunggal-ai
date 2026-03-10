# Running the WBS scripts (use the project `venv`)

This file explains how to run the extractor and generator using a Python virtual environment (venv).

Working directory (example):

```
/home/anton/Koding/ngawur/excel/Untitled Folder
```

1) Ensure you have a venv (recommended name: `.venv`)

- If the repository already contains `.venv`, skip this step.
- To create a new venv:

```bash
python3 -m venv .venv
```

2) Activate the venv (optional) or call the venv `python` directly

- To activate (bash/zsh):

```bash
source .venv/bin/activate
```

- Or call the venv `python` directly without activating:

```bash
"$(pwd)/.venv/bin/python" <script.py> [args]
```

3) Install dependencies

```bash
# If activated
pip install -r requirements.txt

# Or using venv python explicitly
"$(pwd)/.venv/bin/python" -m pip install -r requirements.txt
```

4) Extract WBS from Excel to JSON

```bash
# Example: extract_mandays.py <input.xlsx> <output.json>
# Using venv python explicitly:
"$(pwd)/.venv/bin/python" extract_mandays.py contoh.xlsx output.json
```

After running, `output.json` will contain the WBS JSON used by the generator.

5) Generate styled Excel from JSON

```bash
# Ensure output.json exists (or copy template):
cp wbs_template.json output.json

# Run generator (creates WBS_Generated.xlsx):
"$(pwd)/.venv/bin/python" generate_mandays.py
```

6) Notes & troubleshooting

- If you prefer not to create `.venv` in the repo, create a system venv elsewhere and use its `python` path instead.
- If `pip install -r requirements.txt` fails, try upgrading pip: `"$(pwd)/.venv/bin/python" -m pip install --upgrade pip`
- To validate JSON syntax quickly: `python -m json.tool output.json` (use venv python if needed)

That's it — use the `.venv` python for reliable, isolated runs.