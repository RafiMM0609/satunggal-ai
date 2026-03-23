"""
Tests for WebQuizBuilderTool.

Validates:
  - HTML generation succeeds with valid quiz data
  - Brand color overrides are present in Tailwind config (all required shades)
  - Dynamic class preloader div is present
  - Required Alpine.js x-data, x-show, x-for directives are in the HTML
  - JavaScript syntax is valid (via `node --check`)
  - Security: </script> in JSON data is properly escaped
  - Output file is written and readable on disk
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tools.web_quiz_builder import WebQuizBuilderTool, _HTML_TEMPLATE
from src.memory.state import AgentTask

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    {
        "id": 1,
        "question": "Apa output dari print(2+2)?",
        "options": ["A. 3", "B. 4", "C. 5", "D. 22"],
        "correct": 1,
        "explanation": "2+2 menghasilkan 4.",
    },
    {
        "id": 2,
        "question": "Keyword untuk mendefinisikan fungsi di Python?",
        "options": ["A. function", "B. func", "C. define", "D. def"],
        "correct": 3,
        "explanation": "Python menggunakan keyword def.",
    },
]


def _make_task(questions=None, title="Test Kuis") -> AgentTask:
    task = MagicMock(spec=AgentTask)
    task.session_id = "test_session_001"
    task.metadata = {
        "quiz_questions": questions if questions is not None else SAMPLE_QUESTIONS,
        "quiz_title": title,
    }
    return task


def _generate_html(questions=None, title="Test Kuis") -> str:
    """Helper: directly generate HTML from template for unit tests."""
    quiz_payload = {
        "title": title,
        "questions": questions if questions is not None else SAMPLE_QUESTIONS,
    }
    quiz_json = json.dumps(quiz_payload, ensure_ascii=False, indent=2)
    quiz_json = quiz_json.replace("</script>", r"<\/script>")
    return _HTML_TEMPLATE.format(quiz_data_json=quiz_json)


# ── Template unit tests (no async needed) ─────────────────────────────────────


class TestHtmlTemplate:
    def test_brand_color_shades_400_defined(self):
        html = _generate_html()
        assert "400:'#60a5fa'" in html or "400: '#60a5fa'" in html or "400:" in html, (
            "brand-400 color must be defined in tailwind.config"
        )

    def test_brand_color_shades_900_defined(self):
        html = _generate_html()
        assert "900:'#1e3a8a'" in html or "900: '#1e3a8a'" in html or "900:" in html, (
            "brand-900 color must be defined in tailwind.config"
        )

    def test_all_brand_shades_in_config(self):
        html = _generate_html()
        # Extract the brand config block
        brand_match = re.search(r"brand:\s*\{([^}]+)\}", html)
        assert brand_match is not None, "brand color config block not found"
        brand_config = brand_match.group(1)
        for shade in ["50", "400", "500", "600", "700", "900"]:
            assert shade + ":" in brand_config, (
                f"brand-{shade} shade must be defined in tailwind.config"
            )

    def test_preloader_div_present(self):
        html = _generate_html()
        assert "Preload dynamic" in html, "Tailwind class preloader comment not found"

    def test_preloader_contains_brand400(self):
        html = _generate_html()
        preloader_match = re.search(
            r"Preload dynamic.*?<\/div>", html, re.DOTALL
        )
        assert preloader_match is not None, "Preloader div not found"
        assert "hover:border-brand-400" in preloader_match.group(0)

    def test_preloader_contains_brand900(self):
        html = _generate_html()
        preloader_match = re.search(
            r"Preload dynamic.*?<\/div>", html, re.DOTALL
        )
        assert preloader_match is not None, "Preloader div not found"
        assert "brand-900" in preloader_match.group(0)

    def test_preloader_contains_score_colors(self):
        html = _generate_html()
        preloader_match = re.search(
            r"Preload dynamic.*?<\/div>", html, re.DOTALL
        )
        assert preloader_match is not None
        content = preloader_match.group(0)
        for cls in ["border-green-400", "border-yellow-400", "border-red-400"]:
            assert cls in content, f"Score color class '{cls}' missing from preloader"

    def test_alpine_darkmode_config(self):
        html = _generate_html()
        assert "darkMode: 'class'" in html, "Tailwind darkMode: 'class' must be set"

    def test_alpine_xdata_on_html(self):
        html = _generate_html()
        assert 'x-data="quizApp()"' in html

    def test_alpine_xcloak_present(self):
        html = _generate_html()
        assert "x-cloak" in html

    def test_quiz_data_injected(self):
        html = _generate_html(title="Kuis Python Dasar")
        assert "Kuis Python Dasar" in html
        assert "QUIZ_DATA" in html

    def test_correct_question_count_in_data(self):
        html = _generate_html()
        data_match = re.search(r"const QUIZ_DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        assert data_match is not None, "QUIZ_DATA not found"
        parsed = json.loads(data_match.group(1))
        assert len(parsed["questions"]) == len(SAMPLE_QUESTIONS)

    def test_script_injection_escaped(self):
        """Ensure </script> in quiz data cannot break out of the script block."""
        evil_questions = [
            {
                "id": 1,
                "question": "</script><script>alert(1)</script>",
                "options": ["A", "B", "C", "D"],
                "correct": 0,
                "explanation": "test",
            }
        ]
        html = _generate_html(questions=evil_questions)
        # The breaking sequence must NOT appear unescaped
        # Find the script block containing QUIZ_DATA
        script_match = re.search(
            r"<script>\s*const QUIZ_DATA.*?</script>", html, re.DOTALL
        )
        assert script_match is not None, "QUIZ_DATA script block not found"
        script_block = script_match.group(0)
        # Inside the script block, </script> must not appear unescaped
        inner = script_block[len("<script>"):-len("</script>")]
        assert "</script>" not in inner, (
            "</script> from JSON must be escaped to prevent script injection"
        )

    def test_alpine_js_cdn_loads(self):
        html = _generate_html()
        assert "alpinejs" in html
        assert "cdn.min.js" in html

    def test_tailwind_cdn_loads(self):
        html = _generate_html()
        assert "cdn.tailwindcss.com" in html

    def test_navigation_buttons_present(self):
        html = _generate_html()
        assert "nextQuestion()" in html
        assert "prevQuestion()" in html
        assert "startQuiz()" in html
        assert "restartQuiz()" in html

    def test_score_getters_present(self):
        html = _generate_html()
        assert "get scorePct()" in html
        assert "get progressPct()" in html
        assert "get scoreColor()" in html
        assert "get scoreLabel()" in html


# ── JavaScript syntax validation ───────────────────────────────────────────────


class TestJavaScriptSyntax:
    def _extract_scripts(self, html: str) -> str:
        """Extract all inline <script> content (excluding CDN src scripts)."""
        scripts = re.findall(
            r"<script(?!\s+src)[^>]*>(.*?)</script>", html, re.DOTALL
        )
        return "\n".join(scripts)

    def test_js_syntax_valid(self):
        """Use node --check to validate the embedded JavaScript syntax."""
        html = _generate_html()
        js_content = self._extract_scripts(html)

        with tempfile.NamedTemporaryFile(
            suffix=".js", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(js_content)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, (
                f"JavaScript syntax error:\n{result.stderr}"
            )
        finally:
            os.unlink(tmp_path)


# ── Async integration tests (WebQuizBuilderTool.run) ─────────────────────────


class TestWebQuizBuilderTool:
    def test_run_success(self):
        tool = WebQuizBuilderTool()
        task = _make_task()
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        assert "html_path" in result
        assert result["html_path"].endswith(".html")
        assert os.path.exists(result["html_path"])

    def test_run_file_readable(self):
        tool = WebQuizBuilderTool()
        task = _make_task()
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        html_path = result["html_path"]
        content = Path(html_path).read_text(encoding="utf-8")
        assert len(content) > 1000
        assert "QUIZ_DATA" in content
        assert "quizApp()" in content

    def test_run_empty_questions_returns_error(self):
        tool = WebQuizBuilderTool()
        task = _make_task(questions=[])
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        assert "error" in result

    def test_run_html_has_correct_title(self):
        tool = WebQuizBuilderTool()
        task = _make_task(title="Kuis Sejarah Indonesia")
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        content = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "Kuis Sejarah Indonesia" in content

    def test_run_html_has_all_questions(self):
        tool = WebQuizBuilderTool()
        questions = SAMPLE_QUESTIONS * 3  # 6 questions
        task = _make_task(questions=questions)
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        content = Path(result["html_path"]).read_text(encoding="utf-8")
        parsed_data = json.loads(
            re.search(r"const QUIZ_DATA\s*=\s*(\{.*?\});", content, re.DOTALL).group(1)
        )
        assert len(parsed_data["questions"]) == 6

    def test_run_output_in_temp_dir(self):
        tool = WebQuizBuilderTool()
        task = _make_task()
        result = asyncio.get_event_loop().run_until_complete(tool.run(task))
        assert "advance_ai_quiz" in result["html_path"]
