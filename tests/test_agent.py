import pytest
from neuroforge.agent import NeuroForgeAgent
from neuroforge.schemas import AgentDecision

TOOL = {"action": "tool", "tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}}}
FINAL = {"action": "final", "answer": "The result is 4."}

class SequenceModel:
    def __init__(self, values):
        self.values = values
        self.calls = 0
        self.prompts = []
    def decide(self, system, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        result = self.values[min(self.calls - 1, len(self.values) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

def test_agent_executes_tool_and_finishes():
    model = SequenceModel([TOOL, FINAL])
    result = NeuroForgeAgent(model, max_steps=3).run("What is two plus two?")
    assert result.answer == "The result is 4."
    assert result.status == "completed"
    assert result.steps == model.calls == 2
    assert result.trace[0].result["result"] == 4
    assert '"result": 4.0' in model.prompts[1]

def test_step_budget_makes_no_hidden_synthesis_call():
    model = SequenceModel([TOOL])
    result = NeuroForgeAgent(model, max_steps=1).run("Add two and two")
    assert model.calls == result.steps == 1
    assert result.status == "step_limit"
    assert "**4**" in result.answer
    assert result.trace[-1].event == "step_limit"

def test_malformed_decision_can_recover_within_budget():
    model = SequenceModel([{"action": "unknown"}, TOOL, FINAL])
    result = NeuroForgeAgent(model, max_steps=3).run("Add two and two")
    assert result.status == "completed"
    assert result.steps == 3
    assert result.trace[0].event == "model_error"

def test_invalid_json_output_can_recover():
    model = SequenceModel([ValueError("bad JSON"), FINAL])
    result = NeuroForgeAgent(model, max_steps=2).run("Explain the result")
    assert result.status == "completed"
    assert model.calls == 2

def test_invalid_output_never_executes_a_tool():
    model = SequenceModel([{"action": "tool", "tool_call": {"name": "shell", "arguments": {"cmd": "anything"}}}])
    result = NeuroForgeAgent(model, max_steps=2).run("Try an unavailable tool")
    assert result.status == "failed"
    assert model.calls == 2
    assert not any(event.event == "tool" for event in result.trace)

def test_tool_failure_is_observed_and_recoverable():
    broken = {"action": "tool", "tool_call": {"name": "calculator", "arguments": {"expression": "1/0"}}}
    model = SequenceModel([broken, TOOL, FINAL])
    result = NeuroForgeAgent(model, max_steps=3).run("Check a calculation")
    assert result.status == "completed"
    assert result.trace[0].event == "tool_error"
    assert '"error":' in model.prompts[1]

def test_model_failure_returns_recordable_result():
    result = NeuroForgeAgent(SequenceModel([RuntimeError("backend unavailable")])).run("A technical task")
    assert result.status == "failed"
    assert result.steps == 1
    assert result.run_id
    assert result.trace[0].event == "model_error"

@pytest.mark.parametrize("goal", ["", "  ", "a", "x" * 2001])
def test_bad_goals_do_not_consume_model_calls(goal):
    model = SequenceModel([FINAL])
    with pytest.raises(ValueError):
        NeuroForgeAgent(model).run(goal)
    assert model.calls == 0

@pytest.mark.parametrize("decision", [
    {"action": "tool"},
    {"action": "final"},
    {"action": "final", "answer": " "},
    {"action": "final", "answer": "ok", "tool_call": TOOL["tool_call"]},
    {"action": "tool", "tool_call": TOOL["tool_call"], "answer": "claimed result"},
])
def test_schema_enforces_consistent_actions(decision):
    with pytest.raises(ValueError):
        AgentDecision.model_validate(decision)

