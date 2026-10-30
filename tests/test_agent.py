from neuroforge.agent import NeuroForgeAgent


class FakeModel:
    def __init__(self):
        self.calls = 0

    def decide(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "Compute the requested value.",
                "action": "tool",
                "tool_call": {
                    "name": "calculator",
                    "arguments": {"expression": "2 + 2"},
                },
            }
        return {
            "thought": "The calculation is complete.",
            "action": "final",
            "answer": "The result is 4.",
        }


def test_agent_uses_tool_and_returns_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "agent.db"))
    result = NeuroForgeAgent(model=FakeModel(), max_steps=3).run("What is 2 + 2?")
    assert result.answer == "The result is 4."
    assert result.steps == 2
    assert any(event.event == "tool" for event in result.trace)
    assert result.run_id is not None
