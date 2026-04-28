"""Abstract base class that every agent must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.memory.state import AgentTask


class BaseAgent(ABC):
    """
    All agents inherit from this class.

    Contract:
    - Implement `run(task)` – mutate and return the task.
    - Set `name` class attribute to a unique slug.
    - Optionally set `role`, `goal`, `backstory`, and `delegates_to` for
      persona-aware prompting and inter-agent delegation.
    """

    name: str = "base"

    # ── Persona attributes ────────────────────────────────────────────────────
    # Sub-classes should override these to give the agent a distinct identity
    # that gets injected into the LLM system prompt via get_persona_prompt().

    #: Short job title / role description (e.g. "Senior Developer Orchestrator").
    role: str = ""

    #: Primary objective this agent is trying to achieve.
    goal: str = ""

    #: Narrative backstory that shapes the agent's reasoning style.
    backstory: str = ""

    # ── Delegation registry ───────────────────────────────────────────────────
    # List of agent *names* that this agent is allowed to consult.
    # Populated by sub-classes; read by the orchestrator delegation loop.
    delegates_to: list[str] = []

    # ── Persona helpers ───────────────────────────────────────────────────────

    def get_persona_prompt(self) -> str:
        """Return a formatted persona block for use in LLM system prompts.

        Returns an empty string when none of role/goal/backstory are set so
        that agents that have not yet defined a persona produce no extra tokens.
        """
        parts: list[str] = []
        if self.role:
            parts.append(f"**Role:** {self.role}")
        if self.goal:
            parts.append(f"**Goal:** {self.goal}")
        if self.backstory:
            parts.append(f"**Backstory:** {self.backstory}")
        if not parts:
            return ""
        # Append the universal vibe awareness note so every persona-enabled agent
        # inherits adaptive language style by default.
        parts.append(
            "\n**Contextual Vibe Awareness:**\n"
            "Lo bakal nerima input yang bervariasi, dari formal sampai bahasa tongkrongan "
            "(slang/Gen Z). Tugas lo adalah nangkep substansi tugasnya tanpa peduli seberapa "
            "santai bahasanya. Kalau user nanya pake 'Gue/Lo', bales dengan gaya yang sama. "
            "Kalau user formal, lo boleh sedikit lebih rapi tapi tetep asik (jangan kaku)."
        )
        return "\n".join(parts)

    @abstractmethod
    async def run(self, task: AgentTask) -> AgentTask:
        """Execute the agent's logic. Must call task.mark_done() or task.mark_failed()."""
        ...

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
