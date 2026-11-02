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


def test_gpu_failure_returns_fixed_category_without_private_details(monkeypatch):
    secret = "private-token-or-another-visitors-input"
    def unavailable(*args):
        raise RuntimeError("GPU quota exceeded " + secret)
    monkeypatch.setattr(app, "_gpu_task", unavailable)
    output = app.run_task("Convert a speed", 3, [])
    assert "GPU_QUOTA" in output[2]
    assert secret not in str(output)
    assert output[4][0]["status"] == "failed"


def test_successful_gpu_request_keeps_the_real_task_path(monkeypatch):
    expected = ("answer", "evidence", "status", [], [], [])
    calls = []
    def gpu(*args):
        calls.append(args)
        return expected
    monkeypatch.setattr(app, "_gpu_task", gpu)
    assert app.run_task("A technical task", 3, []) == expected
    assert len(calls) == 1
