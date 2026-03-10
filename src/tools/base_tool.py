"""Abstract base class for all orchesrated tools.

A Tool is a self-contained, async unit of work that can be scheduled by the
orchestrator BEFORE routing to a specialist agent.  The tool's output is
stored in `task.tool_results[tool.name]` and is available to the agent via
that same dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.state import AgentTask


class BaseTool(ABC):
    """All tools must inherit from this class."""

    name: str = "base_tool"

    @abstractmethod
    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """Execute the tool for *task* and return a result dict.

        The orchestrator merges the returned dict into
        ``task.tool_results[self.name]``.
        """
        ...

    def __repr__(self) -> str:
        return f"<Tool:{self.name}>"
