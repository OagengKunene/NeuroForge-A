"""Hugging Face Inference Providers client."""
import json
import os
from huggingface_hub import InferenceClient


class HuggingFaceModel:
    def __init__(self, model: str | None = None):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is missing. Add it to your environment or .env file.")
        self.model = model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.client = InferenceClient(token=token, model=self.model, timeout=90)

    def decide(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The Hugging Face model returned an empty response.")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Model response must be a JSON object.")
        return parsed
