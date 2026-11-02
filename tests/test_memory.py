from neuroforge.memory import add_run, HISTORY_LIMIT
from neuroforge.schemas import AgentResponse

def response(goal="First visitor's task", status="completed"):
    return AgentResponse(goal=goal, answer="A result", steps=1, status=status, trace=[])

def test_session_histories_are_isolated():
    first = add_run([], response())
    second = add_run([], response("Second visitor's task"))
    assert first[0]["goal"] == "First visitor's task"
    assert second[0]["goal"] == "Second visitor's task"
    assert first[0]["run_id"] != second[0]["run_id"]

def test_history_is_bounded_and_does_not_mutate_earlier_state():
    history = []
    for i in range(HISTORY_LIMIT + 4):
        previous = history
        history = add_run(history, response(str(i)))
        assert history is not previous
    assert len(history) == HISTORY_LIMIT
    assert history[0]["goal"] == str(HISTORY_LIMIT + 3)

def test_failed_runs_are_preserved_in_the_visitors_history():
    history = add_run([], response(status="failed"))
    assert history[0]["status"] == "failed"

