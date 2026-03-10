"""
Blackboard state – a single task that travels between agents.

Every request from any interface (Telegram, REST API, CLI) is wrapped
in an AgentTask so all agents speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    PENDING    = "pending"
    ROUTING    = "routing"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class AgentTask:
    """A unit of work passed through the orchestration pipeline."""

    session_id:    str
    user_input:    str
    intent:        Optional[str]      = None
    status:        TaskStatus         = TaskStatus.PENDING
    result:        Optional[str]      = None
    metadata:      dict[str, Any]     = field(default_factory=dict)
    agent_trace:   list[str]          = field(default_factory=list)
    tool_results:  dict[str, Any]     = field(default_factory=dict)

    # Tools requested by an agent to be executed by the orchestrator AFTER
    # the agent finishes.  Agent sets this list; orchestrator drains it.
    pending_tools: list[str]          = field(default_factory=list)

    def mark_routed(self, intent: str) -> None:
        self.intent = intent
        self.status = TaskStatus.ROUTING
        self.agent_trace.append(f"router → {intent}")

    def mark_processing(self, agent_name: str) -> None:
        self.status = TaskStatus.PROCESSING
        self.agent_trace.append(f"processing by {agent_name}")

    def mark_done(self, result: str) -> None:
        self.status = TaskStatus.DONE
        self.result = result

    def mark_failed(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.metadata["error"] = reason
        self.result = None
