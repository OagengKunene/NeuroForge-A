import pytest
from neuroforge.tools import calculator, unit_convert, text_analyze, execute_tool


def test_calculator():
    assert calculator("(0.22 * 5.5 ** 2) / 2")["result"] == pytest.approx(3.3275)


def test_calculator_rejects_code():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo unsafe')")


def test_unit_conversion():
    result = unit_convert(72, "km/h", "m/s")
    assert result["result"] == pytest.approx(20)


def test_unit_conversion_rejects_dimensions():
    with pytest.raises(ValueError):
        unit_convert(1, "km", "kg")


def test_text_analysis():
    result = text_analyze("NeuroForge builds useful tools. Tests matter!")
    assert result["words"] == 6
    assert result["sentences"] == 2


def test_tool_arguments_are_exact():
    with pytest.raises(ValueError):
        execute_tool("calculator", {"expression": "1+1", "extra": "no"})


@pytest.mark.parametrize("expression", ["1e999", "1e999-1e999", "(-1)**0.5", "1/0", "10**9", "1e15*2"])
def test_calculator_rejects_nonfinite_complex_or_oversized_results(expression):
    with pytest.raises(ValueError):
        calculator(expression)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "12", 1e13])
def test_unit_conversion_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError):
        unit_convert(value, "km", "m")


def test_unit_conversion_rejects_invalid_unit_types():
    with pytest.raises(ValueError):
        unit_convert(1, None, "m")

