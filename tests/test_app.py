import app
from tests.test_agent import SequenceModel, TOOL, FINAL

def test_ui_constructs_without_loading_a_model():
    assert app._BACKEND is None
    demo = app.create_demo()
    assert demo.title == "NeuroForge AI"
    names = {fn.api_name for fn in demo.fns.values()}
    assert "run_task" in names

def test_presentation_exposes_verified_evidence_and_session_history():
    output = app.present_run("What is two plus two?", 3, [], SequenceModel([TOOL, FINAL]))
    answer, evidence, status, trace, state, visible = output
    assert answer == "The result is 4."
    assert "**4**" in evidence
    assert "Completed" in status
    assert trace[0]["result"]["result"] == 4
    assert state == visible
    assert state[0]["goal"] == "What is two plus two?"

def test_bad_input_does_not_change_history_or_run_model():
    model = SequenceModel([FINAL])
    history = [{"goal": "Existing task"}]
    output = app.present_run("a", 3, history, model)
    assert output[2] == "Not started"
    assert output[4] == history
    assert model.calls == 0
