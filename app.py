"""Streamlit front end for NeuroForge AI."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="NeuroForge AI", page_icon="🧠", layout="wide")
st.title("🧠 NeuroForge AI")
st.caption("Autonomous technical problem-solving with Hugging Face models")

with st.sidebar:
    st.subheader("Model configuration")
    st.code(os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    st.caption("Set HF_TOKEN in your local .env file. Never paste secrets into the app or commit them.")
    max_steps = st.slider("Maximum agent steps", min_value=1, max_value=8, value=4)
    st.divider()
    st.caption("Available tools: calculator, unit conversion, text analysis.")

goal = st.text_area(
    "Describe the task",
    placeholder="Example: Calculate the energy stored in a 0.22 F capacitor charged to 5.5 V, and explain the engineering significance.",
    height=130,
)
if st.button("Run NeuroForge", type="primary", disabled=len(goal.strip()) < 3):
    from neuroforge.agent import NeuroForgeAgent
    with st.spinner("Reasoning and executing validated tools..."):
        try:
            result = NeuroForgeAgent(max_steps=max_steps).run(goal.strip())
            st.subheader("Result")
            st.write(result.answer)
            st.caption(f"Run #{result.run_id} · {result.steps} agent steps")
            with st.expander("Execution trace", expanded=True):
                for event in result.trace:
                    st.markdown(f"**Step {event.step} · {event.event}**")
                    st.write(event.detail)
        except Exception as exc:
            st.error(f"Could not complete the run: {exc}")

st.divider()
st.caption("NeuroForge is a research prototype. Verify technical outputs before relying on them.")
