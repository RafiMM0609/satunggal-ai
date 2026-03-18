"""
Progress Tracker – renders a Telegram-ready progress bar text.

Usage (in orchestrator):

    tracker = ProgressTracker()
    tracker.advance("Menganalisis permintaan...", pct=15)
    await status_callback(tracker.render())

    tracker.complete_current()
    tracker.advance("Riset sedang berlangsung...", pct=50)
    await status_callback(tracker.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────────────

_BAR_LEN = 10  # total blocks in the progress bar

# Map (stage_key) → (percent, user-friendly label)
# Used by the orchestrator to announce each pipeline stage.
STAGE_MAP: dict[str, tuple[int, str]] = {
    # Gatekeeper
    "gatekeeper":                  (15, "🔍 Menganalisis permintaan..."),
    # Pre-agent tools
    "pre_tool:tavily_search":      (30, "🌐 Mencari referensi web..."),
    "pre_tool:wbs_generator":      (35, "📐 Menyiapkan tool WBS..."),
    "pre_tool:mandays_generator":  (35, "📊 Menyiapkan tool Man-Days..."),
    "pre_tool:diagram_renderer":   (35, "🖼️ Menyiapkan renderer diagram..."),
    "pre_tool:document_generator": (35, "📝 Menyiapkan generator dokumen..."),
    # Agent
    "agent:responder":             (55, "💬 Menyusun jawaban..."),
    "agent:researcher":            (55, "🔬 Riset sedang berlangsung..."),
    "agent:content_creator":       (55, "✍️ Membuat konten..."),
    "agent:wbs_agent":             (55, "📐 Menyusun WBS..."),
    "agent:mandays_agent":         (55, "📊 Menghitung Man-Days..."),
    "agent:developer":             (55, "💻 Menganalisis kode..."),
    "agent:developer_inspector":   (55, "🔎 Inspeksi kode..."),
    "agent:technical_writer":      (55, "📖 Menyusun dokumen teknis..."),
    # Post-agent tools
    "post_tool:wbs_generator":      (75, "📐 Generate file WBS..."),
    "post_tool:mandays_generator":  (80, "📊 Generate file Man-Days..."),
    "post_tool:diagram_renderer":   (82, "🖼️ Render diagram..."),
    "post_tool:document_generator": (85, "📝 Generate dokumen..."),
    # Done
    "done":                        (100, "✅ Selesai!"),
    # Quiz pipeline stages
    "agent:quiz_agent":            (55, "🧠 Menghasilkan soal kuis..."),
    "post_tool:web_quiz_builder":  (90, "🏗️ Membangun website kuis..."),
    "post_tool:pdf_parser":        (25, "📄 Membaca dan memparse PDF..."),
}


# ── Data helpers ───────────────────────────────────────────────────────────────

@dataclass
class _Step:
    label: str
    done: bool = False
    active: bool = True


# ── Main class ─────────────────────────────────────────────────────────────────

class ProgressTracker:
    """Stateful progress tracker that renders a compact Telegram message."""

    def __init__(self, title: str = "⏳ Sedang memproses...") -> None:
        self._title = title
        self._pct: int = 5
        self._steps: list[_Step] = []

    # ── Mutation helpers ───────────────────────────────────────────────────

    def advance(self, stage_key: str) -> None:
        """
        Register the next stage by its key from ``STAGE_MAP``.

        Marks the *previous* active step as done, then adds the new step.
        If *stage_key* is not in STAGE_MAP, the call is silently skipped.
        """
        entry = STAGE_MAP.get(stage_key)
        if entry is None:
            return
        pct, label = entry
        self._complete_current()
        self._pct = pct
        self._steps.append(_Step(label=label))

    def complete_current(self) -> None:
        """Mark the currently active step as done (call after the stage finishes)."""
        self._complete_current()

    def _complete_current(self) -> None:
        for s in reversed(self._steps):
            if s.active and not s.done:
                s.done = True
                s.active = False
                break

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return the full formatted progress message (plain text + Markdown)."""
        bar = _build_bar(self._pct)
        lines: list[str] = [
            f"*{self._title}*",
            f"`{bar}` *{self._pct}%*",
        ]

        if self._steps:
            lines.append("")
            lines.append("*Log:*")
            for s in self._steps:
                if s.done:
                    lines.append(f"✅ {s.label}")
                elif s.active:
                    lines.append(f"🚧 {s.label}")
                else:
                    lines.append(f"⏸ {s.label}")

        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_bar(pct: int) -> str:
    filled = round(_BAR_LEN * pct / 100)
    return "[" + "█" * filled + "░" * (_BAR_LEN - filled) + "]"
