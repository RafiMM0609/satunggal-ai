"""
WebQuizBuilderTool – Membangun Single-File HTML Quiz Interaktif.

Menerima daftar soal JSON dari task.metadata["quiz_questions"] dan
menyuntikkannya ke dalam template HTML yang sudah berisi:
  - Tailwind CSS (Play CDN) untuk styling responsif
  - Alpine.js v3 (CDN) untuk logika interaktif di sisi browser
  - Dark mode, instant feedback, progress bar, dan scoreboard

Output: file .html yang bisa langsung dibuka di browser tanpa server.

Tool ini dipanggil oleh orchestrator SETELAH QuizAgent selesai mengumpulkan soal.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, TYPE_CHECKING

from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id" x-data="quizApp()" :class="darkMode ? 'dark' : ''">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title x-text="quizTitle"></title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            brand: {{ 50:'#eff6ff', 100:'#dbeafe', 200:'#bfdbfe', 300:'#93c5fd', 400:'#60a5fa', 500:'#3b82f6', 600:'#2563eb', 700:'#1d4ed8', 800:'#1e40af', 900:'#1e3a8a' }}
          }}
        }}
      }}
    }}
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  <style>
    [x-cloak] {{ display: none !important; }}
    .option-btn {{ transition: all 0.2s ease; }}
    .option-btn:disabled {{ cursor: default; }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .fade-in {{ animation: fadeIn 0.3s ease forwards; }}
  </style>
</head>
<body class="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans">

  <!-- Preload dynamic Tailwind classes used by Alpine.js at runtime -->
  <div class="hidden
    border-green-400 bg-green-50 text-green-600 text-green-700 text-green-800 dark:bg-green-900/30 dark:text-green-200 dark:text-green-400
    border-red-400 bg-red-50 text-red-500 text-red-700 dark:bg-red-900/30 dark:text-red-300 dark:text-red-400
    border-yellow-400 text-yellow-500 text-yellow-600 dark:text-yellow-400
    border-green-300 border-red-300 dark:border-green-700 dark:border-red-700
    hover:border-brand-400 hover:bg-brand-50 dark:hover:bg-brand-900/30 dark:text-brand-400
    opacity-60"></div>

  <!-- ── Header ──────────────────────────────────────────────────────── -->
  <header class="sticky top-0 z-10 bg-white/80 dark:bg-slate-800/80 backdrop-blur border-b border-slate-200 dark:border-slate-700 shadow-sm">
    <div class="max-w-3xl mx-auto px-4 py-3 flex items-center justify-between">
      <div>
        <h1 class="text-lg font-bold text-brand-600 dark:text-brand-400" x-text="quizTitle"></h1>
        <p class="text-xs text-slate-500 dark:text-slate-400" x-show="!finished">
          Soal <span x-text="currentIndex + 1"></span> dari <span x-text="questions.length"></span>
        </p>
      </div>
      <div class="flex items-center gap-3">
        <!-- Score badge -->
        <span x-show="!finished" class="hidden sm:inline-flex items-center gap-1 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded-full text-xs font-semibold">
          ✅ <span x-text="score"></span>/<span x-text="answered"></span>
        </span>
        <!-- Dark mode toggle -->
        <button @click="darkMode = !darkMode"
                class="p-2 rounded-lg bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                :aria-label="darkMode ? 'Mode Terang' : 'Mode Gelap'">
          <span x-show="!darkMode">🌙</span>
          <span x-show="darkMode">☀️</span>
        </button>
      </div>
    </div>
    <!-- Progress bar -->
    <div x-show="!finished" class="h-1 bg-slate-200 dark:bg-slate-700">
      <div class="h-1 bg-brand-500 transition-all duration-500"
           :style="`width: ${{progressPct}}%`"></div>
    </div>
  </header>

  <!-- ── Main Container ──────────────────────────────────────────────── -->
  <main class="max-w-3xl mx-auto px-4 py-8">

    <!-- ── Splash Screen ───────────────────────────────────────────── -->
    <div x-show="!started && !finished" x-cloak class="fade-in text-center space-y-6">
      <div class="text-6xl">📚</div>
      <h2 class="text-2xl font-bold" x-text="quizTitle"></h2>
      <p class="text-slate-500 dark:text-slate-400">
        Kuis ini berisi <strong x-text="questions.length"></strong> soal pilihan ganda.
        Jawab semua soal dan lihat skormu di akhir!
      </p>
      <div class="flex flex-col sm:flex-row gap-3 justify-center">
        <button @click="startQuiz()"
                class="px-8 py-3 bg-brand-600 hover:bg-brand-700 text-white font-semibold rounded-xl shadow-md hover:shadow-lg transition-all">
          🚀 Mulai Kuis
        </button>
      </div>
      <p class="text-xs text-slate-400">Dibuat dengan AdvanceAI • PDF-to-Quiz Generator</p>
    </div>

    <!-- ── Question Card ───────────────────────────────────────────── -->
    <div x-show="started && !finished" x-cloak class="fade-in space-y-6">

      <!-- Question -->
      <div class="bg-white dark:bg-slate-800 rounded-2xl shadow-md p-6 border border-slate-100 dark:border-slate-700">
        <div class="flex items-start gap-3">
          <span class="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-full bg-brand-500 text-white text-sm font-bold"
                x-text="currentIndex + 1"></span>
          <p class="text-base font-medium leading-relaxed" x-text="currentQuestion.question"></p>
        </div>
      </div>

      <!-- Options -->
      <div class="space-y-3">
        <template x-for="(opt, idx) in currentQuestion.options" :key="idx">
          <button
            class="option-btn w-full text-left px-5 py-4 rounded-xl border-2 font-medium transition-all"
            :class="optionClass(idx)"
            :disabled="answered > currentIndex"
            @click="selectAnswer(idx)">
            <span x-text="opt"></span>
            <!-- Feedback icon -->
            <span x-show="answered > currentIndex && idx === currentQuestion.correct"
                  class="float-right text-green-600 dark:text-green-400">✓</span>
            <span x-show="answered > currentIndex && idx === userAnswers[currentIndex] && idx !== currentQuestion.correct"
                  class="float-right text-red-500 dark:text-red-400">✗</span>
          </button>
        </template>
      </div>

      <!-- Explanation -->
      <div x-show="answered > currentIndex && currentQuestion.explanation"
           class="fade-in bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-xl p-4 text-sm text-blue-800 dark:text-blue-200">
        <strong>💡 Penjelasan:</strong>
        <span x-text="currentQuestion.explanation"></span>
      </div>

      <!-- Navigation -->
      <div class="flex justify-between items-center pt-2">
        <button @click="prevQuestion()"
                x-show="currentIndex > 0"
                class="px-5 py-2 rounded-lg border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors text-sm font-medium">
          ← Sebelumnya
        </button>
        <div class="flex-1"></div>
        <button @click="nextQuestion()"
                x-show="answered > currentIndex"
                class="px-6 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white font-semibold transition-all shadow-sm">
          <span x-text="currentIndex < questions.length - 1 ? 'Lanjut →' : '🏁 Lihat Hasil'"></span>
        </button>
      </div>

    </div>

    <!-- ── Scoreboard ──────────────────────────────────────────────── -->
    <div x-show="finished" x-cloak class="fade-in space-y-6 text-center">

      <!-- Score circle -->
      <div class="inline-flex flex-col items-center justify-center w-40 h-40 rounded-full border-8 mx-auto"
           :class="scoreColor">
        <span class="text-4xl font-black" x-text="score"></span>
        <span class="text-sm text-slate-500 dark:text-slate-400">dari <span x-text="questions.length"></span></span>
      </div>

      <div>
        <h2 class="text-2xl font-bold" x-text="scoreLabel"></h2>
        <p class="text-slate-500 dark:text-slate-400 mt-1" x-text="`Skor kamu: ${{scorePct}}%`"></p>
      </div>

      <!-- Action buttons -->
      <div class="flex flex-col sm:flex-row gap-3 justify-center">
        <button @click="reviewMode = true; finished = false; currentIndex = 0"
                class="px-6 py-3 rounded-xl border-2 border-brand-500 text-brand-600 dark:text-brand-400 font-semibold hover:bg-brand-50 dark:hover:bg-brand-900/30 transition-colors">
          🔍 Review Jawaban
        </button>
        <button @click="restartQuiz()"
                class="px-6 py-3 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-semibold shadow-md transition-all">
          🔄 Ulangi Kuis
        </button>
      </div>

      <!-- Stats grid -->
      <div class="grid grid-cols-3 gap-4 mt-4">
        <div class="bg-green-50 dark:bg-green-900/20 rounded-xl p-4">
          <div class="text-2xl font-bold text-green-600 dark:text-green-400" x-text="score"></div>
          <div class="text-xs text-slate-500 mt-1">Benar</div>
        </div>
        <div class="bg-red-50 dark:bg-red-900/20 rounded-xl p-4">
          <div class="text-2xl font-bold text-red-500 dark:text-red-400" x-text="questions.length - score"></div>
          <div class="text-xs text-slate-500 mt-1">Salah</div>
        </div>
        <div class="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
          <div class="text-2xl font-bold text-blue-600 dark:text-blue-400" x-text="scorePct + '%'"></div>
          <div class="text-xs text-slate-500 mt-1">Nilai</div>
        </div>
      </div>

      <!-- Review list -->
      <div x-show="reviewMode" class="text-left space-y-4 mt-4">
        <h3 class="font-bold text-lg">📋 Review Jawaban</h3>
        <template x-for="(q, idx) in questions" :key="q.id">
          <div class="bg-white dark:bg-slate-800 rounded-xl border p-4 space-y-2"
               :class="userAnswers[idx] === q.correct ? 'border-green-300 dark:border-green-700' : 'border-red-300 dark:border-red-700'">
            <div class="flex gap-2 items-start">
              <span class="text-sm font-bold" x-text="idx + 1 + '.'"></span>
              <p class="text-sm font-medium" x-text="q.question"></p>
            </div>
            <div class="text-sm pl-5 space-y-1">
              <template x-for="(opt, oi) in q.options" :key="oi">
                <div :class="{{
                       'text-green-700 dark:text-green-400 font-semibold': oi === q.correct,
                       'text-red-500 dark:text-red-400 line-through': oi === userAnswers[idx] && oi !== q.correct,
                       'text-slate-500': oi !== q.correct && oi !== userAnswers[idx]
                     }}" x-text="opt"></div>
              </template>
              <p x-show="q.explanation" class="text-xs text-blue-600 dark:text-blue-400 mt-1 italic"
                 x-text="'💡 ' + q.explanation"></p>
            </div>
          </div>
        </template>
      </div>
    </div>

  </main>

  <!-- ── Alpine.js App Logic ────────────────────────────────────────────────── -->
  <script>
    const QUIZ_DATA = {quiz_data_json};

    function quizApp() {{
      return {{
        quizTitle:    QUIZ_DATA.title || 'Kuis Interaktif',
        questions:    QUIZ_DATA.questions || [],
        darkMode:     window.matchMedia('(prefers-color-scheme: dark)').matches,
        started:      false,
        finished:     false,
        reviewMode:   false,
        currentIndex: 0,
        userAnswers:  [],
        score:        0,
        answered:     0,

        get currentQuestion() {{
          return this.questions[this.currentIndex] || {{}};
        }},

        get progressPct() {{
          return this.questions.length
            ? Math.round((this.answered / this.questions.length) * 100)
            : 0;
        }},

        get scorePct() {{
          return this.questions.length
            ? Math.round((this.score / this.questions.length) * 100)
            : 0;
        }},

        get scoreLabel() {{
          const p = this.scorePct;
          if (p >= 90) return '🏆 Luar Biasa! Kamu Menguasai Materi Ini!';
          if (p >= 75) return '🥇 Bagus Sekali! Hampir Sempurna!';
          if (p >= 60) return '👍 Cukup Baik! Terus Berlatih!';
          if (p >= 40) return '📖 Perlu Belajar Lebih Giat!';
          return '💪 Jangan Menyerah! Coba Lagi!';
        }},

        get scoreColor() {{
          const p = this.scorePct;
          if (p >= 75) return 'border-green-400 text-green-600 dark:text-green-400';
          if (p >= 50) return 'border-yellow-400 text-yellow-600 dark:text-yellow-400';
          return 'border-red-400 text-red-500 dark:text-red-400';
        }},

        startQuiz() {{
          this.started = true;
          this.userAnswers = new Array(this.questions.length).fill(-1);
        }},

        restartQuiz() {{
          this.currentIndex = 0;
          this.score = 0;
          this.answered = 0;
          this.finished = false;
          this.reviewMode = false;
          this.userAnswers = new Array(this.questions.length).fill(-1);
        }},

        selectAnswer(idx) {{
          if (this.answered > this.currentIndex) return; // already answered
          this.userAnswers[this.currentIndex] = idx;
          this.answered++;
          if (idx === this.currentQuestion.correct) this.score++;
        }},

        optionClass(idx) {{
          const base = 'border-2 ';
          const notAnswered = 'border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-brand-400 hover:bg-brand-50 dark:hover:bg-slate-700';
          if (this.answered <= this.currentIndex) return base + notAnswered;

          if (idx === this.currentQuestion.correct)
            return base + 'border-green-400 bg-green-50 dark:bg-green-900/30 text-green-800 dark:text-green-200';
          if (idx === this.userAnswers[this.currentIndex])
            return base + 'border-red-400 bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300';
          return base + 'border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-800 opacity-60';
        }},

        prevQuestion() {{
          if (this.currentIndex > 0) this.currentIndex--;
        }},

        nextQuestion() {{
          if (this.currentIndex < this.questions.length - 1) {{
            this.currentIndex++;
          }} else {{
            this.finished = true;
          }}
        }},
      }};
    }}
  </script>

</body>
</html>
"""


# ── WebQuizBuilderTool ────────────────────────────────────────────────────────

class WebQuizBuilderTool(BaseTool):
    """
    Converts accumulated quiz questions into a single-file interactive HTML.

    Input (from task.metadata):
        "quiz_questions": list[dict]  – validated question objects from QuizAgent
        "quiz_title":     str         – optional quiz title

    Output:
        { "html_path": str }  – path to the generated .html file
    """

    name = "web_quiz_builder"

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        questions: list[dict] = task.metadata.get("quiz_questions", [])
        if not questions:
            logger.error(
                "WebQuizBuilderTool: no quiz_questions in metadata. session=%s",
                task.session_id,
            )
            return {"error": "Tidak ada soal yang tersedia untuk dibangun."}

        quiz_title = task.metadata.get("quiz_title", "Kuis Interaktif")

        # Prepare JSON payload for the template
        quiz_payload = {
            "title":     quiz_title,
            "questions": questions,
        }
        quiz_json = json.dumps(quiz_payload, ensure_ascii=False, indent=2)
        # Prevent </script> in JSON from breaking out of the <script> block
        quiz_json = quiz_json.replace("</script>", "<\/script>")

        # Inject data into template
        # Use .format() so that {{ }} escapes in the template resolve to literal { }
        html_content = _HTML_TEMPLATE.format(quiz_data_json=quiz_json)

        # Write to temp file
        html_path = _make_html_path(task.session_id)
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except OSError as exc:
            logger.exception("WebQuizBuilderTool: failed to write HTML: %s", exc)
            return {"error": f"Gagal menulis file HTML: {exc}"}

        logger.info(
            "WebQuizBuilderTool: HTML OK – questions=%d path=%s session=%s",
            len(questions), html_path, task.session_id,
        )
        return {"html_path": html_path}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_html_path(session_id: str) -> str:
    out_dir = os.path.join(tempfile.gettempdir(), "advance_ai_quiz")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"quiz_{session_id}_{ts}.html")
