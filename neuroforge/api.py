"""FastAPI interface for NeuroForge."""
from fastapi import FastAPI, HTTPException
from .agent import NeuroForgeAgent
from .memory import recent_runs
from .schemas import AgentRequest, AgentResponse

app = FastAPI(title="NeuroForge AI", version="0.1.0",
              description="A bounded autonomous agent powered by Hugging Face models.")


@app.get("/health")
def health():
    return {"status": "ok", "service": "NeuroForge AI"}


@app.get("/runs")
def runs():
    return {"runs": recent_runs()}


@app.post("/run", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    try:
        return NeuroForgeAgent(max_steps=request.max_steps).run(request.goal)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc
