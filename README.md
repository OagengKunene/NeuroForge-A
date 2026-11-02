---
title: NeuroForge AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12.12"
suggested_hardware: zero-a10g
pinned: false
short_description: GPU engineering assistant with verified tool outputs
models:
  - Qwen/Qwen2.5-1.5B-Instruct
tags:
  - agents
  - engineering
  - zerogpu
---

# NeuroForge AI

A technical problem-solving workspace for Kunene. A small instruction model runs directly on Hugging Face ZeroGPU, selects validated tools, and explains the results with an observable execution trace.

**[Open the app](https://huggingface.co/spaces/Oageng/NeuroForge-AI)**

## What it does

- Calculate engineering expressions, including capacitor energy.
- Convert supported units with physical-dimension checks.
- Count words, sentences, and characters in short passages.
- Show actual tool inputs, outputs, and errors separately from the model's explanation.
- Keep up to ten recent runs in the current visitor's session.

The default model is **Qwen/Qwen2.5-1.5B-Instruct**, pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It is a pretrained model, not a model trained by this project.

## Architecture

Gradio task → ZeroGPU allocation → local Qwen inference → validated decision → registered tool → observation → final response and verified evidence.

The GPU handler uses `@spaces.GPU` with a duration calculated from the call budget: 30–90 seconds, with 45 seconds for the default three calls. Model weights are loaded and moved to CUDA at application startup, before requests are served, following the [ZeroGPU model-loading guidance](https://huggingface.co/docs/hub/spaces-zerogpu).

The model uses Qwen's native function-calling chat template. The adapter converts function calls into the project's validated internal schema, and final explanations use natural language. This avoids requiring a small language model to imitate a custom response protocol.

The app downloads the public model and runs it locally in the Space. It does not call a paid Inference Provider and needs no inference API token. The GitHub deployment token is used only by the publishing workflow.

## Run on Hugging Face

The existing Space must use **Gradio + ZeroGPU**. CPU Basic is not selected by the workflow. Free ZeroGPU hosting depends on Hugging Face account eligibility; queue waits and daily quotas apply.

The Space card pins a supported Python runtime, Gradio, and the runtime dependencies. The first start downloads model weights and can take several minutes.

The GitHub sync workflow:
1. Runs the CPU-only unit and UI construction checks.
2. Confirms the Space is configured for ZeroGPU.
3. Uploads this repository through the existing GitHub `HF_TOKEN` secret.
4. Waits for the Space to start and reports a build/runtime failure.

The publishing token must have write access to `Oageng/NeuroForge-AI`. Keep it in GitHub secrets; never commit its value.

## Run locally

Use Python 3.12:

```bash
pip install -r requirements.txt
python app.py
```

A CUDA GPU is used when available. Outside Spaces, a machine without CUDA can run the model on CPU, although generation will be slower and needs enough RAM for the model.

Optional environment settings:

| Variable | Purpose |
|---|---|
| `NEUROFORGE_MODEL` | Public Transformers causal-language-model ID |
| `NEUROFORGE_MODEL_REVISION` | Optional checkpoint revision for a model override |
| `PORT` | Local server port, default 7860 |

Model overrides never enable remote custom model code. The default public model is downloaded without an account token.

## API

Gradio exposes the same ZeroGPU handler through its API. Open **Use via API** in the app for the generated client instructions. The endpoint is `/run_task`, with task and maximum model calls as inputs. Gradio manages session state for each client.

The earlier global SQLite history endpoint and separate FastAPI/Render deployment have been retired. This app has one Gradio interface and API, with session history.

## Bounds and reliability

- Tasks: 3–2,000 characters.
- Model-call budget: 1–6, including malformed-response recovery. No hidden final call is added.
- Each generation: at most 320 new tokens.
- GPU allocation: at most 90 seconds per run.
- Calculator: 200-character expressions, bounded exponents, finite real inputs/results.
- Unit conversion: supported dimensions only; finite values within the stated tool bounds.
- Registered tools only; no arbitrary Python, shell, filesystem, or network actions are offered to the model.

If the model exhausts its call budget after a successful tool call, the app returns the verified tool results. If it fails without evidence, the run is marked failed. A model's explanation can still be wrong: compare it with the tool evidence.

The trace contains observable actions and results, not hidden chain-of-thought.

## Session history

Run history is held in Gradio session state, is limited to ten entries, expires after one hour, and starts empty after a page refresh. Visitors do not receive a global list of other visitors' tasks. Clear this session resets the current workspace. The app does not write tasks to SQLite or enable Gradio's persistent run-history feature.

## Development

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

Tests use fake model decisions and do not download model weights or allocate a GPU. They cover tool calculations, malformed/nonfinite output, recovery, exact model-call budgets, failed runs, isolated session histories, and construction of the pinned Gradio UI. Live GPU inference is a separate deployment check.

After successful deployment, **Check NeuroForge GPU examples** submits two fixed public examples through the authenticated Gradio API and verifies their tool results. This consumes the publisher account's ZeroGPU allowance. The check uses only its own example responses; it does not collect private Space runtime logs or other visitors' histories. It can also be run manually from GitHub Actions.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Gradio workspace and actual GPU inference handler |
| `assets/style.css` | Interface styling |
| `neuroforge/llm.py` | Local model loading and strict JSON parsing |
| `neuroforge/agent.py` | Bounded agent loop and evidence summary |
| `neuroforge/tools.py` | Calculator, conversions, and text statistics |
| `neuroforge/memory.py` | Bounded session-history copies |
| `neuroforge/schemas.py` | Validated model decisions and results |
| `tests/` | Offline tests |
| `.github/workflows/` | Reusable tests and gated Hugging Face deployment |

## License

MIT. See [LICENSE](LICENSE). Model weights are distributed under the model publisher's license.

