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
    """

    name: str = "base"

    @abstractmethod
    async def run(self, task: AgentTask) -> AgentTask:
        """Execute the agent's logic. Must call task.mark_done() or task.mark_failed()."""
        ...

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
