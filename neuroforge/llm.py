"""Local Transformers inference. Load weights before ZeroGPU requests."""
import json
import math
import os
import re
import threading

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with self.lock, self.torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096
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
        return parse_model_json(content)

