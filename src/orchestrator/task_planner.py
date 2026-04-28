"""
TaskPlanner – decomposes complex multi-step user requests into sequential agent plans.
ConsistencyChecker – reviews the final answer quality before it reaches the user.

Phase 2 of the Autonomous Multi-Agent Workforce transformation.

Flow
----
1. ``TaskPlanner.should_plan(user_text)`` uses heuristic patterns to detect
   compound / multi-step requests without spending any LLM tokens.
2. ``TaskPlanner.plan(user_text, session_id)`` asks the LLM to produce an
   ordered JSON execution plan that maps steps to registered agent names.
3. ``main_loop._execute_plan(plan, task, ...)`` runs each step sequentially,
   carrying the previous step's output as context into the next.
4. ``ConsistencyChecker.check(user_input, result, session_id)`` does a
   lightweight LLM review of the composed answer and returns a corrected
   version only when an obvious problem is detected.

On any error the system degrades gracefully:
* ``plan()``  → returns a single-step plan (normal routing takes over).
* ``check()`` → returns the original result unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Chainable agent catalogue ─────────────────────────────────────────────────
# Maps agent_name → default short description used when the agent has no persona.
# Only agents suitable for sequential multi-step chaining are listed here.

CHAINABLE_AGENTS: dict[str, str] = {
    "researcher":          "Web Research Specialist – searches the web for up-to-date information",
    "content_creator":     "Content Writer – writes articles, blog posts, social-media copy",
    "technical_writer":    "Technical Documentation Writer – creates structured technical docs (PDF/Word)",
    "developer":           "Code Developer – clones repos, applies code changes, runs sandbox verification",
    "developer_inspector": "Code Inspector – analyses a codebase and identifies bugs/issues (read-only)",
    "code_fix":            "Code Bug Fixer – detects and automatically fixes bugs in a repo",
    "wbs_agent":           "WBS Planner – creates Work Breakdown Structure plans in Excel",
    "mandays_agent":       "Mandays Estimator – estimates work effort in mandays",
    "pdf_summarizer":      "PDF Summarizer – summarises PDF documents and answers questions about them",
    "responder":           "General Assistant – answers general questions and handles simple requests",
}


# ── Compound-request detection patterns ───────────────────────────────────────
# Conservative list – only match clearly sequential/compound phrasings.
# False negatives (missing a compound request) are preferred over false positives
# (splitting a simple request into unnecessary steps).

_COMPOUND_PATTERNS: list[re.Pattern[str]] = [
    # Indonesian: "pertama/langkah pertama X ... kemudian/lalu/setelah itu Y"
    re.compile(
        r"\b(?:pertama|langkah\s+pertama|step\s+1|tahap\s+pertama)\b.{10,}"
        r"\b(?:kemudian|lalu|setelah\s+itu|berikutnya|selanjutnya|kedua|step\s+2)\b",
        re.I | re.S,
    ),
    # "riset/cari/cek X lalu/kemudian/dan buat/tulis/generate Y"
    re.compile(
        r"\b(?:riset|cari|cek\s+(?:kode|repo)|analisa|inspect)\b.{5,}"
        r"\b(?:lalu|kemudian|dan\s+(?:buat|tulis|generate|write|create))\b",
        re.I | re.S,
    ),
    # "setelah X, buat/tulis Y"
    re.compile(
        r"\b(?:setelah|sesudah)\b.{5,}"
        r"\b(?:lalu|kemudian|buat|generate|tulis|write|create)\b",
        re.I | re.S,
    ),
    # English: "first ... then/after that ..."
    re.compile(
        r"\b(?:first|step\s+1|to\s+start)\b.{10,}"
        r"\b(?:then|after\s+that|next|second|step\s+2)\b",
        re.I | re.S,
    ),
    # "research X then/and write/create/build Y"
    re.compile(
        r"\b(?:research|check|inspect|analyse?)\b.{5,}"
        r"\b(?:then|and)\s+(?:write|create|generate|build|make)\b",
        re.I | re.S,
    ),
]


# ── LLM prompts ───────────────────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """\
You are a task decomposition expert for an AI multi-agent system.

Available agents:
{agent_catalogue}

Your task:
Given a user request, decide if it requires multiple agents working sequentially.

Output ONLY valid JSON in one of these two exact forms:

Single task (no decomposition needed):
{{"mode": "single", "steps": []}}

Multiple sequential tasks:
{{"mode": "sequential", "steps": [
  {{"step_id": 1, "agent_name": "<name>", "description": "<what this step does>", "instruction": "<exact instruction for this agent in the same language as the user>"}},
  {{"step_id": 2, "agent_name": "<name>", "description": "<what this step does>", "instruction": "<exact instruction; mention that Step 1 result will be provided as context>"}},
  ...]}}

Rules:
- Only use agent names from the catalogue above.
- Minimum 2 steps if mode is sequential. Maximum 4 steps.
- Each instruction must be self-contained and written in the same language as the user request.
- For steps after step 1, begin the instruction with "Berdasarkan hasil langkah sebelumnya:"
  (Indonesian) or "Based on the previous step results:" (English).
- If you are unsure whether decomposition is needed, return single mode.
- Return ONLY the JSON. No explanations, no markdown fences.
"""

_CHECKER_SYSTEM_PROMPT = """\
You are a quality-control reviewer for an AI assistant system.

User's original request:
{user_input}

The assistant's response:
{result}

Does the response fully and correctly answer the user's request?

If YES → reply with exactly: OK
If NO (off-topic, obviously truncated, clear factual error, or misses the main point)
→ provide an improved version of the response only.

Rules:
- Do NOT add preambles like "Here is the improved version:".
- Do NOT change a correct answer just for style.
- Match the language of the user request.
- Reply with "OK" when in doubt.
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in a multi-agent execution plan."""

    step_id:     int
    agent_name:  str
    description: str   # human-readable label for status updates
    instruction: str   # exact prompt to send to the agent as task.user_input


@dataclass
class TaskPlan:
    """An ordered execution plan produced by TaskPlanner."""

    original_request: str
    mode:  str                    # "single" | "sequential"
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def is_multi_step(self) -> bool:
        """Return True when this plan has ≥ 2 sequential steps to execute."""
        return self.mode == "sequential" and len(self.steps) >= 2


# ── TaskPlanner ───────────────────────────────────────────────────────────────

class TaskPlanner:
    """
    Decomposes complex multi-step user requests into sequential agent plans.

    Usage::

        planner = TaskPlanner(agents=agents_dict, llm=llm_client)

        if planner.should_plan(user_text):
            plan = await planner.plan(user_text, session_id=session_id)
            if plan.is_multi_step:
                # run plan via main_loop._execute_plan()
                ...
    """

    def __init__(
        self,
        agents: dict[str, "BaseAgent"],
        llm:    "LLMClient",
    ) -> None:
        self._agents = agents
        self._llm    = llm

    # ── Public API ────────────────────────────────────────────────────────────

    def should_plan(self, user_text: str) -> bool:
        """Heuristic check: does the message contain compound/sequential task language?

        Uses pre-compiled regex patterns.  Returns ``True`` when at least one
        compound pattern matches.  Intentionally conservative to avoid false
        positives that would add LLM latency to simple requests.
        """
        for pat in _COMPOUND_PATTERNS:
            if pat.search(user_text):
                logger.debug(
                    "TaskPlanner.should_plan: compound pattern matched → planning triggered"
                )
                return True
        return False

    async def plan(
        self,
        user_text:  str,
        session_id: str = "unknown",
    ) -> TaskPlan:
        """Ask the LLM to decompose the request into a sequential execution plan.

        Falls back to a single-step plan (normal routing takes over) on any
        error so the pipeline never breaks even if the planner call fails.
        """
        catalogue = self._build_agent_catalogue()
        system    = _PLANNER_SYSTEM_PROMPT.format(agent_catalogue=catalogue)
        messages  = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ]

        try:
            raw  = await self._llm.chat(messages, max_tokens=512)
            plan = self._parse_plan(raw, user_text)
            plan = self._validate_plan(plan)
            logger.info(
                "TaskPlanner: plan created mode=%s steps=%d session=%s",
                plan.mode, len(plan.steps), session_id,
            )
            return plan
        except Exception as exc:
            logger.warning(
                "TaskPlanner.plan failed (falling back to single): %s session=%s",
                exc, session_id,
            )
            return TaskPlan(original_request=user_text, mode="single", steps=[])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_agent_catalogue(self) -> str:
        """Build a numbered list of available chainable agents for the planner prompt."""
        lines: list[str] = []
        for agent_name, default_desc in CHAINABLE_AGENTS.items():
            agent = self._agents.get(agent_name)
            if agent is None:
                continue
            # Prefer the agent's own persona role if set in Phase 1
            desc = getattr(agent, "role", "") or default_desc
            lines.append(f"- {agent_name}: {desc}")
        return "\n".join(lines) if lines else "No agents available."

    @staticmethod
    def _parse_plan(raw: str, original_request: str) -> TaskPlan:
        """Parse the LLM's JSON output into a ``TaskPlan``."""
        # Strip markdown fences that some models add despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M).strip()
        data    = json.loads(cleaned)

        mode      = str(data.get("mode", "single"))
        steps_raw = data.get("steps", [])

        steps: list[PlanStep] = []
        for s in steps_raw:
            steps.append(PlanStep(
                step_id     = int(s["step_id"]),
                agent_name  = str(s["agent_name"]),
                description = str(s.get("description", f"Step {s['step_id']}")),
                instruction = str(s["instruction"]),
            ))

        return TaskPlan(original_request=original_request, mode=mode, steps=steps)

    def _validate_plan(self, plan: TaskPlan) -> TaskPlan:
        """Remove steps whose agent_name is not in the registry.

        Downgrades to single mode when fewer than 2 valid steps remain.
        """
        if plan.mode == "single" or not plan.steps:
            return plan

        valid: list[PlanStep] = [
            s for s in plan.steps
            if self._agents.get(s.agent_name) is not None
        ]
        if len(valid) < 2:
            logger.debug(
                "TaskPlanner._validate_plan: only %d valid steps → downgrading to single",
                len(valid),
            )
            return TaskPlan(original_request=plan.original_request, mode="single", steps=[])

        return TaskPlan(original_request=plan.original_request, mode="sequential", steps=valid)


# ── ConsistencyChecker ────────────────────────────────────────────────────────

class ConsistencyChecker:
    """
    Reviews the final agent response for quality before it is returned to the user.

    Used by the orchestrator after multi-step plan execution to verify that the
    composed answer is coherent and fully addresses the original request.

    The LLM is instructed to return "OK" when the answer is acceptable, and to
    return an improved version only when there is an obvious problem.  On any
    error the original result is returned unchanged (fail-safe).
    """

    #: Cap context lengths to keep review calls cheap.
    MAX_INPUT_CHARS:  int = 800
    MAX_RESULT_CHARS: int = 2000

    def __init__(self, llm: "LLMClient") -> None:
        self._llm = llm

    async def check(
        self,
        user_input: str,
        result:     str,
        session_id: str = "unknown",
    ) -> str:
        """Review *result* against *user_input* and return the (possibly improved) text.

        Returns *result* unchanged when:
        * The LLM confirms the quality ("OK").
        * *result* is empty or whitespace-only.
        * The review call itself fails.
        """
        if not result or not result.strip():
            return result

        system = _CHECKER_SYSTEM_PROMPT.format(
            user_input=user_input[: self.MAX_INPUT_CHARS],
            result=result[: self.MAX_RESULT_CHARS],
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": "Lakukan review."},
        ]

        try:
            review = await self._llm.chat(messages, max_tokens=1024)
            review = review.strip()

            if review.upper().startswith("OK"):
                logger.debug(
                    "ConsistencyChecker: result approved for session=%s", session_id
                )
                return result

            # LLM returned a corrected version
            logger.info(
                "ConsistencyChecker: result corrected (%d → %d chars) session=%s",
                len(result), len(review), session_id,
            )
            return review

        except Exception as exc:
            logger.warning(
                "ConsistencyChecker failed (returning original): %s session=%s",
                exc, session_id,
            )
            return result
