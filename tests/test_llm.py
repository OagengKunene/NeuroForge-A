import pytest
from neuroforge.llm import parse_model_json, build_messages, decision_from_generation
from neuroforge.schemas import AgentDecision
import json

def test_parses_object_and_fenced_object():
    assert parse_model_json('{"action":"final","answer":"done"}')["answer"] == "done"
    fence = chr(96) * 3
    assert parse_model_json(fence + 'json\n{"action":"final","answer":"done"}\n' + fence)["action"] == "final"

@pytest.mark.parametrize("content", ["", "not JSON", "[]", "null", '{"a": 1} {"b": 2}', '{"value": NaN}', '{"value": Infinity}', '{"value": 1e999}'])
def test_rejects_unsupported_or_nonfinite_model_output(content):
    with pytest.raises(ValueError):
        parse_model_json(content)


def test_native_qwen_call_becomes_a_validated_tool_decision():
    output = '<tool_call>{"name":"unit_convert","arguments":{"value":72,"from_unit":"km/h","to_unit":"m/s"}}</tool_call>'
    decision = AgentDecision.model_validate(decision_from_generation(output))
    assert decision.action == "tool"
    assert decision.tool_call.arguments["value"] == 72


def test_native_answer_becomes_a_final_response():
    decision = AgentDecision.model_validate(decision_from_generation("72 km/h is 20 m/s."))
    assert decision.answer == "72 km/h is 20 m/s."


@pytest.mark.parametrize("output", ['<tool_call>{"name":"calculator"}', '<tool_call>{}</tool_call><tool_call>{}</tool_call>'])
def test_rejects_incomplete_or_multiple_calls(output):
    with pytest.raises(ValueError):
        decision_from_generation(output)


def test_native_messages_keep_real_function_arguments_and_result():
    prompt = json.dumps({"goal":"Convert a speed", "observations":[{"tool":"unit_convert", "arguments":{"value":72,"from_unit":"km/h","to_unit":"m/s"}, "result":{"result":20.0}}]})
    messages = build_messages("Use tools", prompt)
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["value"] == 72
    assert json.loads(messages[3]["content"])["result"]["result"] == 20.0
