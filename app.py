"""NeuroForge AI: a genuine ZeroGPU model with a session-scoped workspace."""
import os
from functools import wraps
from pathlib import Path

# Import before Torch/Transformers so ZeroGPU can install its CUDA hooks.
import spaces
import gradio as gr

from neuroforge.agent import NeuroForgeAgent, evidence_summary
from neuroforge.llm import LocalModel, DEFAULT_MODEL
from neuroforge.memory import add_run
from neuroforge.schemas import AgentResponse, TraceEvent

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Executed at startup, before Gradio registers GPU handlers. Unit tests import
# the module without loading several GB of model weights.
_BACKEND = LocalModel() if __name__ == "__main__" else None


def present_run(goal, max_steps, history, model):
    history = history or []
    try:
        result = NeuroForgeAgent(model=model, max_steps=max_steps).run(goal)
    except ValueError:
        return (
            "Enter a task of 3–2,000 characters and choose 1–6 model calls.",
            "No tool has run.", "Not started", [], history, history,
        )
    trace = [event.model_dump(exclude_none=True) for event in result.trace]
    observations = [
        {"tool": event.tool, "result": event.result}
        for event in result.trace if event.event == "tool"
    ]
    verified = evidence_summary(observations) if observations else "No tools were executed in this run."
    updated = add_run(history, result)
    labels = {"completed": "Completed", "step_limit": "Call limit reached", "failed": "Failed"}
    status = (
        f"{labels[result.status]} · {result.steps} model calls · "
        f"{len(observations)} verified tool results · {result.elapsed_seconds:g}s"
    )
    return result.answer, verified, status, trace, updated, updated


@spaces.GPU(duration=90)
def _gpu_task(goal: str, max_steps: int, history: list):
    """Run the local language model and validated tools on shared ZeroGPU."""
    if _BACKEND is None:
        raise RuntimeError("Start app.py to load the local model before inference.")
    return present_run(goal, max_steps, history, _BACKEND)


def runtime_error_category(error):
    """Map only known SDK conditions to fixed public messages; never return logs."""
    text = (str(getattr(error, "title", "")) + " " + str(error)).lower()
    if "quota" in text or "runs limit" in text or "gpu limit" in text:
        return "GPU_QUOTA", "The shared GPU allowance is currently exhausted. Try after your Hugging Face quota resets."
    if "illegal duration" in text or "requested duration" in text:
        return "GPU_DURATION", "The requested GPU duration exceeds the allowance available for this run."
    if "internal gradio" in text:
        return "GPU_REQUEST_CONTEXT", "The GPU service could not read this Gradio request."
    if "proxy token" in text or "unauthorized" in text or "authentication" in text:
        return "GPU_AUTH", "Hugging Face could not authorise this GPU allocation. Refresh the page and try again."
    if "queue" in text or "timeout" in text:
        return "GPU_QUEUE", "A shared GPU did not become available in time. Try again later."
    if "typeerror" in text or isinstance(error, TypeError):
        return "GPU_WORKER_TYPE", "The GPU worker encountered a runtime compatibility error."
    if "modulenotfound" in text or "importerror" in text:
        return "GPU_WORKER_IMPORT", "The GPU worker could not load a runtime dependency."
    if "cuda" in text:
        return "GPU_CUDA", "The allocated GPU could not initialise model inference."
    if "aborted" in text:
        return "GPU_ABORTED", "The GPU task was interrupted before it completed."
    if "schedule" in text:
        return "GPU_SCHEDULER", "Hugging Face could not schedule this GPU request."
    if "attributeerror" in text or isinstance(error, AttributeError):
        return "GPU_WORKER_ATTRIBUTE", "The GPU worker encountered a runtime compatibility error."
    if "worker error" in text:
        return "GPU_WORKER", "The GPU worker could not initialise the model request."
    return "GPU_UNAVAILABLE", "The GPU request could not start. Try again later."


@wraps(_gpu_task)
def run_task(goal: str, max_steps: int, history: list):
    # wraps preserves the SDK's ZeroGPU marker on the registered event handler.
    # Successful requests still execute the genuine GPU-decorated model task.
    try:
        return _gpu_task(goal, max_steps, history)
    except Exception as error:
        code, message = runtime_error_category(error)
        result = AgentResponse(
            goal=str(goal)[:2000], answer=message, status="failed", steps=0,
            trace=[TraceEvent(step=0, event="model_error", detail=code)],
        )
        updated = add_run(history, result)
        return message, "No tool ran.", f"GPU unavailable · {code}", [result.trace[0].model_dump(exclude_none=True)], updated, updated


def create_demo():
    with gr.Blocks(title="NeuroForge AI", analytics_enabled=False, delete_cache=(300, 3600)) as demo:
        session_history = gr.State(value=list, time_to_live=3600)
        gr.HTML("""
        <section id="nf-hero">
          <div class="nf-eyebrow">KUNENE / AI ENGINEERING LAB</div>
          <h1>NeuroForge AI<span>.</span></h1>
          <p>Think it through. Check the numbers.</p>
          <div class="nf-badges"><span>Local model inference</span><span>Shared ZeroGPU</span><span>Observable tool use</span></div>
        </section>
        """)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.Markdown("### What are we working on?")
                goal = gr.Textbox(
                    label="Your task",
                    placeholder="Calculate capacitor energy, convert engineering units, or analyse a short passage.",
                    lines=5, max_lines=10, max_length=2000,
                )
                with gr.Row():
                    max_steps = gr.Slider(
                        minimum=1, maximum=6, value=3, step=1, precision=0,
                        label="Maximum model calls",
                        info="Three calls usually allow a calculation and an explanation.",
                    )
                run_button = gr.Button("Run NeuroForge →", variant="primary", elem_id="nf-run")
                gr.Examples(
                    examples=[
                        ["Calculate the energy stored in a 0.22 F capacitor charged to 5.5 V. Use E = 0.5 * C * V^2 and report joules.", 3],
                        ["Convert 72 km/h to m/s and explain the result.", 3],
                        ["Count the words and sentences in: NeuroForge builds useful tools. Tests matter!", 3],
                    ],
                    inputs=[goal, max_steps], cache_examples=False, label="Start with an example",
                )
            with gr.Column(scale=2):
                gr.HTML("""
                <aside id="nf-toolkit">
                  <div class="nf-eyebrow">YOUR TOOLKIT</div>
                  <h2>A result you can inspect.</h2>
                  <div class="nf-tool"><strong>01 / Calculator</strong><p>Bounded arithmetic with finite, real results.</p></div>
                  <div class="nf-tool"><strong>02 / Unit conversion</strong><p>Engineering units with dimension checks.</p></div>
                  <div class="nf-tool"><strong>03 / Text analysis</strong><p>Word, sentence, and character statistics.</p></div>
                </aside>
                """)
                model_name = os.getenv("NEUROFORGE_MODEL", DEFAULT_MODEL)
                gr.Markdown(f"**Model** · {model_name}")
                gr.Markdown(
                    "Runs use a shared GPU. Queue waits and daily Hugging Face quotas apply. "
                    "A run can use up to 90 GPU seconds."
                )
        status = gr.Textbox(label="Run status", value="Ready", interactive=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.Markdown("### Model response")
                answer = gr.Markdown("Your response will appear here after a run.", elem_id="nf-answer")
            with gr.Column(scale=2):
                gr.Markdown("### Verified tool evidence")
                evidence = gr.Markdown("No tool has run.", elem_id="nf-evidence")
        with gr.Accordion("Execution trace · tool inputs, outputs, and errors", open=False):
            trace = gr.JSON(value=[], label="Observable events")
        with gr.Accordion("Recent runs · this session only", open=False):
            history_view = gr.JSON(value=[], label="Up to 10 recent runs")
            clear = gr.Button("Clear this session", variant="secondary")
        gr.Markdown(
            "History is kept in this browser session's server state, expires after one hour, "
            "and resets on refresh. Technical answers can be wrong; compare the explanation "
            "with the verified tool outputs."
        )
        inputs = [goal, max_steps, session_history]
        outputs = [answer, evidence, status, trace, session_history, history_view]
        run_button.click(
            fn=run_task, inputs=inputs, outputs=outputs,
            api_name="run_task", concurrency_limit=1, concurrency_id="nf-gpu",
        )
        goal.submit(
            fn=run_task, inputs=inputs, outputs=outputs,
            api_name="run_task_enter", concurrency_limit=1, concurrency_id="nf-gpu",
        )
        clear.click(
            fn=lambda: ("Your response will appear here after a run.", "No tool has run.", "Ready", [], [], []),
            inputs=[], outputs=outputs, queue=False, api_visibility="private",
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(max_size=12, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css_paths=Path(__file__).parent / "assets" / "style.css",
        ssr_mode=False,
        run_history=False,
        footer_links=["api"],
        show_error=False,
    )

