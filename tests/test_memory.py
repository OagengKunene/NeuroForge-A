import os
from neuroforge.memory import save_run, recent_runs


def test_run_history_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    run_id = save_run("test goal", "test answer", 1, [])
    assert run_id > 0
    runs = recent_runs()
    assert runs[0]["goal"] == "test goal"
