"""Local Transformers inference. Load weights before ZeroGPU requests."""
import json
import math
import os
import re
import threading

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "calculator", "description": "Calculate a finite real arithmetic expression. Use this for numerical calculations.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Arithmetic using +, -, *, /, ** and parentheses."}}, "required": ["expression"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "unit_convert", "description": "Convert engineering units of the same dimension. Use this for unit conversion requests.", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "text_analyze", "description": "Count words, sentences, and characters in supplied text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}},
]


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    request = json.loads(user_prompt)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": request["goal"]}]
    for observation in request.get("observations", []):
        if "tool" in observation:
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": observation["tool"],
                 "arguments": observation.get("arguments", {})}}
            ]})
            body = {"result": observation["result"]} if "result" in observation else {"error": observation["error"]}
            messages.append({"role": "tool", "name": observation["tool"], "content": json.dumps(body)})
        else:
            messages.append({"role": "user", "content": "The previous response could not be used. Select one available function if needed, or give a concise final answer. Do not invent tool results."})
    return messages


def decision_from_generation(content: str) -> dict:
    """Translate Qwen's native tool format into the validated internal contract."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response.")
    content = content.strip()
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if calls:
        if len(calls) != 1:
            raise ValueError("Select one function at a time.")
        return {"action": "tool", "tool_call": parse_model_json(calls[0])}
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("The function call was incomplete.")
    # Also accept a bare structured call or the internal decision format.
    if content.startswith("{") or content.startswith(chr(96) * 3):
        try:
            parsed = parse_model_json(content)
        except ValueError:
            parsed = None
        if parsed is not None and "action" in parsed:
            return parsed
        if parsed is not None and set(parsed) == {"name", "arguments"}:
            return {"action": "tool", "tool_call": parsed}
    return {"action": "final", "answer": content}

def parse_model_json(content: str) -> dict:
    """Accept one JSON object, optionally enclosed in a Markdown code fence."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = content.strip()
    fence = chr(96) * 3
    fenced = re.fullmatch(fence + r"(?:json)?\s*\n?(.*?)\s*" + fence, cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value is not allowed: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON numeric value is outside the finite range.")
        return result

    try:
        parsed = json.loads(cleaned, parse_constant=reject_constant, parse_float=finite_float)
    except json.JSONDecodeError as exc:
        raise ValueError("The model response was not a single valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")
    return parsed

class LocalModel:
    """A small instruction model genuinely executed on the allocated GPU."""

    def __init__(self):
        # Lazy imports keep unit tests independent of CUDA and model downloads.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = os.getenv("NEUROFORGE_MODEL", DEFAULT_MODEL)
        self.revision = os.getenv("NEUROFORGE_MODEL_REVISION") or (
            DEFAULT_REVISION if self.model_id == DEFAULT_MODEL else "main"
        )
        self.device = "cuda" if os.getenv("SPACE_ID") or torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.lock = threading.Lock()
        options = {"revision": self.revision, "trust_remote_code": False, "token": False}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa",
            **options,
        ).to(self.device).eval()
        print(f"NeuroForge model loaded: {self.model_id} on {self.device}", flush=True)

    def decide(self, system_prompt: str, user_prompt: str) -> dict:
        messages = build_messages(system_prompt, user_prompt)
        with self.lock, self.torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096,
                return_token_type_ids=False,
            ).to(self.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            content = self.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        return decision_from_generation(content)

