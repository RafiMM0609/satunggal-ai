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

    # ── Self-describing schema ────────────────────────────────────────────────
    # Sub-classes should override these so LLMs can reason about when and how
    # to call the tool via Tool Calling / Function Calling.

    #: Human-readable explanation of what this tool does.
    description: str = ""

    #: JSON Schema dict describing the inputs the tool reads from AgentTask.
    #: Use ``{"type": "object", "properties": {...}, "required": [...]}`` format.
    input_schema: dict[str, Any] = {}

    #: JSON Schema dict describing the dict this tool returns from ``run()``.
    output_schema: dict[str, Any] = {}

    # ── Schema helper ─────────────────────────────────────────────────────────

    def get_tool_schema(self) -> dict[str, Any]:
        """Return the full OpenAI-style function-calling schema for this tool.

        The returned dict is compatible with the ``tools`` parameter accepted
        by OpenAI chat-completion and OpenRouter endpoints::

            {
              "type": "function",
              "function": {
                "name": "<tool_name>",
                "description": "<description>",
                "parameters": { ... input_schema ... }
              }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"Tool: {self.name}",
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    @abstractmethod
    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """Execute the tool for *task* and return a result dict.

        The orchestrator merges the returned dict into
        ``task.tool_results[self.name]``.
        """
        ...

    def __repr__(self) -> str:
        return f"<Tool:{self.name}>"
