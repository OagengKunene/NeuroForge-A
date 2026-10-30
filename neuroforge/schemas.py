"""Typed data contracts for agent requests, actions and results."""
from typing import Any, Literal
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)
    max_steps: int = Field(default=4, ge=1, le=8)


class ToolCall(BaseModel):
    name: Literal["calculator", "unit_convert", "text_analyze"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    thought: str = Field(default="", max_length=1200)
    action: Literal["tool", "final"]
    tool_call: ToolCall | None = None
    answer: str | None = Field(default=None, max_length=6000)


class TraceEvent(BaseModel):
    step: int
    event: str
    detail: str


class AgentResponse(BaseModel):
    goal: str
    answer: str
    steps: int
    trace: list[TraceEvent]
    run_id: int | None = None
