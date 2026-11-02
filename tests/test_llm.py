import pytest
from neuroforge.llm import parse_model_json

def test_parses_object_and_fenced_object():
    assert parse_model_json('{"action":"final","answer":"done"}')["answer"] == "done"
    fence = chr(96) * 3
    assert parse_model_json(fence + 'json\n{"action":"final","answer":"done"}\n' + fence)["action"] == "final"

@pytest.mark.parametrize("content", ["", "not JSON", "[]", "null", '{"a": 1} {"b": 2}', '{"value": NaN}', '{"value": Infinity}', '{"value": 1e999}'])
def test_rejects_unsupported_or_nonfinite_model_output(content):
    with pytest.raises(ValueError):
        parse_model_json(content)
