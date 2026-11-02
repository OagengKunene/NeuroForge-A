"""Validated requests, model decisions, and observable execution events."""
from typing import Any, Literal, Self
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator

class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    goal: str = Field(min_length=3, max_length=2000)
    max_steps: int = Field(default=3, ge=1, le=6)

class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["calculator", "unit_convert", "text_analyze"]
    arguments: dict[str, Any]

class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool", "final"]
    tool_call: ToolCall | None = None
    answer: str | None = Field(default=None, max_length=6000)

    @model_validator(mode="after")
    def consistent_action(self) -> Self:
        if self.action == "tool" and (self.tool_call is None or self.answer is not None):
            raise ValueError("A tool action needs tool_call and must not include answer.")
        if self.action == "final" and (
            self.tool_call is not None or not self.answer or not self.answer.strip()
        ):
            raise ValueError("A final action needs a nonempty answer and no tool_call.")
        return self

class TraceEvent(BaseModel):
    step: int
    event: Literal["tool", "tool_error", "model_error", "final", "step_limit"]
    detail: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

class AgentResponse(BaseModel):
    goal: str
    answer: str
    steps: int
    status: Literal["completed", "step_limit", "failed"]
    trace: list[TraceEvent]
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    elapsed_seconds: float = 0

