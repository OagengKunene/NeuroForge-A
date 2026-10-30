"""Small allow-listed tools. No arbitrary code or shell execution."""
import ast
import operator
import re
from typing import Any

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent magnitude is limited to 8.")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        if abs(result) > 1e15:
            raise ValueError("Result exceeds the permitted magnitude.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numeric constants and +, -, *, /, %, **, and parentheses are allowed.")


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a safe arithmetic expression."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must contain 1–200 characters.")
    if not re.fullmatch(r"[0-9eE+*/().%\-\s]+", expression):
        raise ValueError("Expression contains unsupported characters.")
    value = _eval_node(ast.parse(expression, mode="eval"))
    return {"expression": expression, "result": value}


_UNIT_TABLE = {
    "m": ("length", 1.0), "km": ("length", 1000.0),
    "cm": ("length", 0.01), "mm": ("length", 0.001),
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "m/s": ("speed", 1.0), "km/h": ("speed", 1 / 3.6),
    "kg": ("mass", 1.0), "g": ("mass", 0.001),
    "pa": ("pressure", 1.0), "kpa": ("pressure", 1000.0),
    "bar": ("pressure", 100000.0),
    "j": ("energy", 1.0), "kj": ("energy", 1000.0),
    "w": ("power", 1.0), "kw": ("power", 1000.0),
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert between supported units in the same physical dimension."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or abs(value) > 1e12:
        raise ValueError("Value must be numeric and within ±1e12.")
    source, target = from_unit.lower().strip(), to_unit.lower().strip()
    if source not in _UNIT_TABLE or target not in _UNIT_TABLE:
        raise ValueError("Unsupported unit. Supported units: " + ", ".join(_UNIT_TABLE))
    source_dim, source_factor = _UNIT_TABLE[source]
    target_dim, target_factor = _UNIT_TABLE[target]
    if source_dim != target_dim:
        raise ValueError("Units belong to different physical dimensions.")
    converted = float(value) * source_factor / target_factor
    return {"value": value, "from": source, "to": target, "result": converted}


def text_analyze(text: str) -> dict[str, Any]:
    """Return simple deterministic statistics for supplied text."""
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError("Text must be a string of at most 12,000 characters.")
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "average_word_length": round(sum(map(len, words)) / len(words), 2) if words else 0,
    }


TOOL_REGISTRY = {
    "calculator": calculator,
    "unit_convert": unit_convert,
    "text_analyze": text_analyze,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and execute one registered tool."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' is not available.")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    required = {
        "calculator": {"expression"},
        "unit_convert": {"value", "from_unit", "to_unit"},
        "text_analyze": {"text"},
    }[name]
    if set(arguments) != required:
        raise ValueError(f"{name} requires exactly these arguments: {sorted(required)}")
    return TOOL_REGISTRY[name](**arguments)
