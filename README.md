---
title: NeuroForge AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# NeuroForge AI

**An open-source, tool-using autonomous AI agent powered by Hugging Face models.**

NeuroForge turns a user goal into a structured plan, asks a Hugging Face instruction model to select from a controlled toolset, executes validated tools, records outcomes in SQLite, and produces a final answer with an execution trace.

> This is a research and engineering assistant prototype. It does not execute arbitrary model-generated Python, run shell commands, or take consequential external actions.

## Architecture

```text
User goal
   |
   v
FastAPI / Streamlit interface
   |
   v
Agent orchestrator ---- SQLite memory
   |
   +---- Hugging Face InferenceClient (planning / tool selection)
   |
   +---- Policy and schema validation
   |
   +---- Allow-listed tools (calculator, unit conversion, text analysis)
   |
   v
Evidence-backed result + execution trace
```

## Quick start

1. Create a Python 3.11+ environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `HF_TOKEN`.
4. Run the UI: `streamlit run app.py`
5. Or run the API: `uvicorn neuroforge.api:app --reload`

The default model is configurable with `HF_MODEL`. The selected model must be available through your Hugging Face Inference Provider account. Provider availability, model routing and usage limits can change; check the Hugging Face model page and account settings.

## Example tasks

- "Calculate the energy stored in a 0.22 F capacitor charged to 5.5 V."
- "Convert 72 km/h to m/s and explain the result."
- "Compare these two short technical descriptions and identify the key differences."

## Configuration

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access token (keep private) |
| `HF_MODEL` | Model ID, default `Qwen/Qwen2.5-7B-Instruct` |
| `MAX_AGENT_STEPS` | Maximum tool-selection rounds (default 4) |
| `DATABASE_PATH` | SQLite database path (default `neuroforge.db`) |

## Project structure

- `neuroforge/llm.py`: Hugging Face chat-completion client
- `neuroforge/agent.py`: bounded agent loop and trace
- `neuroforge/tools.py`: allow-listed, validated tools
- `neuroforge/memory.py`: SQLite run history
- `neuroforge/schemas.py`: typed request and response models
- `neuroforge/api.py`: FastAPI service
- `app.py`: Streamlit interface
- `tests/`: automated tests

## Security principles

- Model output is untrusted input and is parsed/validated before use.
- Only registered tools can be called.
- No arbitrary Python, shell, filesystem, browser or network execution is exposed to the model.
- Tool inputs have explicit schemas and size/range limits.
- Secrets are loaded from environment variables and must never be committed.
- The agent has a strict step limit and records tool errors.

## Development

```bash
pytest
```

## Roadmap

- [x] Hugging Face model integration
- [x] Bounded tool-using agent loop
- [x] Validated tools and SQLite run history
- [x] API and interactive UI
- [ ] Model and prompt benchmark suite
- [ ] Retrieval-augmented technical document analysis
- [ ] Human approval workflow for external actions
- [ ] Hugging Face Spaces deployment

## License

MIT. See [LICENSE](LICENSE).
