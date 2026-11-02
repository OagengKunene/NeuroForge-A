"""Bounded copies of a visitor's Gradio session history; no global database."""
from copy import deepcopy
from datetime import datetime, timezone
from .schemas import AgentResponse

HISTORY_LIMIT = 10

def add_run(history: list | None, result: AgentResponse) -> list[dict]:
    entry = {
        "run_id": result.run_id,
        "goal": result.goal,
        "answer": result.answer,
        "status": result.status,
        "model_calls": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return [entry, *deepcopy(history or [])][:HISTORY_LIMIT]

