"""Bounded autonomous agent loop with explicit tool validation."""
import json
import os
from .llm import HuggingFaceModel
from .memory import save_run
from .schemas import AgentDecision, AgentResponse, TraceEvent
from .tools import execute_tool

SYSTEM_PROMPT = """You are NeuroForge, a careful technical problem-solving agent.
You may choose only these tools:
1. calculator(expression: string) — safe arithmetic, e.g. "(0.22 * 5.5 ** 2) / 2"
2. unit_convert(value: number, from_unit: string, to_unit: string) — supported units include m, km, cm, mm, s, min, h, m/s, km/h, kg, g, Pa, kPa, bar, J, kJ, W, kW.
3. text_analyze(text: string) — basic text statistics.
Do not claim to have run a tool unless its result appears in the tool results.
Do not invent measurements, sources, tests, or external actions. If information is missing, state assumptions.
Return ONLY one JSON object matching one of these forms:
{"thought":"brief rationale","action":"tool","tool_call":{"name":"calculator","arguments":{"expression":"2+2"}}}
{"thought":"brief rationale","action":"final","answer":"answer to user"}
The thought field must be concise. Never request tools outside the list."""


class NeuroForgeAgent:
    def __init__(self, model=None, max_steps=None):
        self.model = model or HuggingFaceModel()
        self.max_steps = max(1, min(int(max_steps or os.getenv("MAX_AGENT_STEPS", "4")), 8))

    def run(self, goal: str) -> AgentResponse:
        trace: list[TraceEvent] = []
        observations: list[dict] = []
        answer = ""
        completed_steps = 0
        for step in range(1, self.max_steps + 1):
            prompt = json.dumps({
                "goal": goal,
                "previous_observations": observations,
                "instruction": "Choose one tool if it is necessary; otherwise return a final answer.",
            }, ensure_ascii=False)
            raw = self.model.decide(SYSTEM_PROMPT, prompt)
            decision = AgentDecision.model_validate(raw)
            completed_steps = step
            if decision.thought:
                trace.append(TraceEvent(step=step, event="reasoning", detail=decision.thought))
            if decision.action == "final":
                answer = decision.answer or "The model did not provide a final answer."
                trace.append(TraceEvent(step=step, event="final", detail="Final response generated."))
                break
            if decision.tool_call is None:
                raise ValueError("Model selected a tool action without a tool call.")
            try:
                result = execute_tool(decision.tool_call.name, decision.tool_call.arguments)
                observations.append({"tool": decision.tool_call.name, "result": result})
                trace.append(TraceEvent(
                    step=step, event="tool",
                    detail=f"{decision.tool_call.name} returned {json.dumps(result, ensure_ascii=False)}",
                ))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                observations.append({"tool": decision.tool_call.name, "error": error})
                trace.append(TraceEvent(step=step, event="tool_error", detail=error))
        if not answer:
            prompt = json.dumps({"goal": goal, "observations": observations,
                                 "instruction": "Summarise the evidence now. Do not call tools."})
            final = self.model.decide(SYSTEM_PROMPT, prompt)
            validated = AgentDecision.model_validate(final)
            answer = validated.answer or "The agent reached its step limit without a final answer."
            trace.append(TraceEvent(step=completed_steps, event="step_limit", detail="Final synthesis requested after the step limit."))
        serialised = [event.model_dump() for event in trace]
        run_id = save_run(goal, answer, completed_steps, serialised)
        return AgentResponse(goal=goal, answer=answer, steps=completed_steps, trace=trace, run_id=run_id)
