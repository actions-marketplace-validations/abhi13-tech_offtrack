"""pytest plugin behavior via pytester (real nested pytest runs)."""

from __future__ import annotations

import json
from pathlib import Path

pytest_plugins = ["pytester"]

AGENT_SNIPPET = """
import json, os
from datetime import datetime, timezone
from pathlib import Path

def emit(events):
    d = Path(os.environ["OFFTRACK_TRACE_DIR"])
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    with (d / "agent-x.jsonl").open("a") as f:
        for e in events:
            e.setdefault("t0", now); e.setdefault("t1", now)
            f.write(json.dumps(e) + "\\n")

def good_agent():
    emit([
        {"ev": "step", "v": 1, "type": "tool_call", "name": "lookup", "args": {"id": 1},
         "result": {"ok": True}, "status": "ok"},
        {"ev": "step", "v": 1, "type": "final_answer", "name": "final", "result": "done",
         "status": "ok"},
        {"ev": "end", "status": "complete"},
    ])

def bad_agent():
    emit([
        {"ev": "step", "v": 1, "type": "tool_call", "name": "delete_everything",
         "args": {"id": 1}, "result": {"ok": True}, "status": "ok"},
        {"ev": "step", "v": 1, "type": "final_answer", "name": "final", "result": "done",
         "status": "ok"},
        {"ev": "end", "status": "complete"},
    ])
"""


def make_baseline(dir: Path, task="suite/task") -> None:
    """Write a baseline JSON matching good_agent's trajectory shape."""
    from offtrack.ingest import build_trajectory

    events = [
        {
            "ev": "step",
            "v": 1,
            "type": "tool_call",
            "name": "lookup",
            "args": {"id": 1},
            "result": {"ok": True},
            "status": "ok",
        },
        {
            "ev": "step",
            "v": 1,
            "type": "final_answer",
            "name": "final",
            "result": "done",
            "status": "ok",
        },
        {"ev": "end", "status": "complete"},
    ]
    trajs = [build_trajectory(events, task, "baseline", i, "test").trajectory for i in range(3)]
    doc = {
        "offtrack_schema": 1,
        "baseline": {
            "baseline_id": "B1",
            "task_key": task,
            "label": "default",
            "created_at": "2026-08-04",
            "git_ref": None,
            "model": None,
            "config_hash": "x",
        },
        "trajectories": [json.loads(t.model_dump_json()) for t in trajs],
    }
    suite, _, name = task.partition("/")
    out = dir / "baselines" / suite
    out.mkdir(parents=True)
    (out / f"{name}.json").write_text(json.dumps(doc))


def test_matching_run_passes(pytester):
    make_baseline(Path(pytester.path))
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import good_agent

        @pytest.mark.offtrack(task="suite/task")
        def test_agent(offtrack):
            with offtrack.record():
                good_agent()
            offtrack.assert_matches_baseline()
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_divergent_run_fails_with_step_pointer(pytester):
    make_baseline(Path(pytester.path))
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import bad_agent

        @pytest.mark.offtrack(task="suite/task")
        def test_agent(offtrack):
            with offtrack.record():
                bad_agent()
            offtrack.assert_matches_baseline()
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*diverged from baseline at step*"])


def test_no_baseline_warns_and_passes(pytester):
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import good_agent

        @pytest.mark.offtrack(task="suite/unknown")
        def test_agent(offtrack):
            with offtrack.record():
                good_agent()
            offtrack.assert_matches_baseline()
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1, warnings=1)


def test_no_baseline_strict_fails(pytester):
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import good_agent

        @pytest.mark.offtrack(task="suite/unknown")
        def test_agent(offtrack):
            with offtrack.record():
                good_agent()
            offtrack.assert_matches_baseline()
        """
    )
    result = pytester.runpytest("--offtrack-require-baseline")
    result.assert_outcomes(failed=1)


def test_empty_capture_fails_actionably(pytester):
    make_baseline(Path(pytester.path))
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.offtrack(task="suite/task")
        def test_agent(offtrack):
            with offtrack.record():
                pass  # agent never emits
            offtrack.assert_matches_baseline()
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*no trace events captured*"])


def test_cost_and_steps_assertions(pytester):
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import good_agent

        @pytest.mark.offtrack(task="suite/task")
        def test_limits(offtrack):
            with offtrack.record():
                good_agent()
            offtrack.assert_max_steps(10)

        @pytest.mark.offtrack(task="suite/task")
        def test_too_many_steps(offtrack):
            with offtrack.record():
                good_agent()
            offtrack.assert_max_steps(1)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1, failed=1)


def test_spill_and_merge_into_db(pytester):
    make_baseline(Path(pytester.path))
    pytester.makepyfile(agent=AGENT_SNIPPET)
    pytester.makepyfile(
        """
        import pytest
        from agent import good_agent

        @pytest.mark.offtrack(task="suite/task")
        def test_agent(offtrack):
            with offtrack.record():
                good_agent()
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    # Session finish merged the spill into the DB and cleaned pending/.
    from offtrack.store import Store

    db = Path(pytester.path) / ".offtrack" / "offtrack.db"
    assert db.exists()
    store = Store(db)
    n = store.conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    assert n == 1
    assert not list((Path(pytester.path) / ".offtrack" / "pending").glob("*.jsonl"))
