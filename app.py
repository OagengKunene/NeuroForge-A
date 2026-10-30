"""Gradio front end for NeuroForge AI."""
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def run_task(goal: str, max_steps: int):
    if not goal or len(goal.strip()) < 3:
        return "Please enter a task with at least three characters.", [], "Not started"
    try:
        from neuroforge.agent import NeuroForgeAgent

        result = NeuroForgeAgent(max_steps=int(max_steps)).run(goal.strip())
        trace = [event.model_dump() for event in result.trace]
        return result.answer, trace, f"Completed · Run #{result.run_id} · {result.steps} agent steps"
    except Exception as exc:
        return f"**Execution failed:** {type(exc).__name__}: {exc}", [], "Failed"


def load_history():
    try:
        from neuroforge.memory import recent_runs
        return recent_runs()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


with gr.Blocks(title="NeuroForge AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 NeuroForge AI\n"
        "### Autonomous technical problem-solving powered by Hugging Face models\n"
        "Describe a task. NeuroForge can select from a small, validated toolset, "
        "execute calculations, and show its execution trace."
    )

    with gr.Row():
        with gr.Column(scale=3):
            goal = gr.Textbox(
                label="Your task",
                placeholder=(
                    "Example: Calculate the energy stored in a 0.22 F capacitor "
                    "charged to 5.5 V, and explain the engineering significance."
                ),
                lines=5,
            )
            max_steps = gr.Slider(
                minimum=1, maximum=8, value=4, step=1,
                label="Maximum agent steps",
            )
            run_button = gr.Button("Run NeuroForge", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("### Model")
            gr.Markdown(f"`{MODEL_NAME}`")
            gr.Markdown(
                "**Available tools**\n"
                "- Calculator\n"
                "- Unit conversion\n"
                "- Text analysis\n\n"
                "**Safety**\n"
                "The model cannot execute arbitrary Python, shell commands, "
                "or unrestricted external actions."
            )

    answer = gr.Markdown(label="Agent response")
    status = gr.Textbox(label="Run status", interactive=False)
    with gr.Accordion("Execution trace", open=True):
        trace = gr.JSON(label="Trace events")

    run_button.click(
        fn=run_task,
        inputs=[goal, max_steps],
        outputs=[answer, trace, status],
    )
    goal.submit(
        fn=run_task,
        inputs=[goal, max_steps],
        outputs=[answer, trace, status],
    )

    with gr.Accordion("Recent run history", open=False):
        refresh = gr.Button("Refresh history")
        history = gr.JSON(label="Recent runs")
        refresh.click(fn=load_history, inputs=[], outputs=[history])

    gr.Markdown(
        "---\n"
        "*NeuroForge is a research prototype. Verify technical outputs "
        "before relying on them. Run history is stored in the Space's local "
        "SQLite filesystem and may not survive a Space restart.*"
    )


if __name__ == "__main__":
    demo.launch()
