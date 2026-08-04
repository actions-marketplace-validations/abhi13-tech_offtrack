"""TrajectoryBuilder: events → Trajectory, including degradation paths."""

from __future__ import annotations

import json
from pathlib import Path

from offtrack.ingest import build_from_trace_dir, build_trajectory
from offtrack.model import INLINE_MAX, StepType, TrajStatus, is_truncated_stub


def ev_step(type="tool_call", name="lookup", args=None, result=None, **kw):
    e = {"ev": "step", "v": 1, "type": type, "name": name, "args": args, "result": result}
    e.update(kw)
    return e


END = {"ev": "end", "status": "complete"}


class TestBuild:
    def test_happy_path(self):
        events = [
            ev_step(
                "llm_call",
                "fake-careful",
                model="fake-careful",
                usage={"tokens_in": 220, "tokens_out": 18},
                t0="2026-08-04T10:00:00+00:00",
                t1="2026-08-04T10:00:01+00:00",
            ),
            ev_step(
                "tool_call",
                "lookup_order",
                args={"order_id": "TEST-1"},
                result={"total_usd": 842.0},
                t0="2026-08-04T10:00:01+00:00",
                t1="2026-08-04T10:00:02+00:00",
            ),
            ev_step(
                "final_answer",
                "final",
                result="done",
                t0="2026-08-04T10:00:02+00:00",
                t1="2026-08-04T10:00:03+00:00",
            ),
            END,
        ]
        r = build_trajectory(events, "s/t", "baseline", 0, "test")
        t = r.trajectory
        assert t.status == TrajStatus.COMPLETE
        assert [s.type for s in t.steps] == [
            StepType.LLM_CALL,
            StepType.TOOL_CALL,
            StepType.FINAL_ANSWER,
        ]
        assert t.tokens_in == 220 and t.wall_ms == 3000
        assert not r.warnings

    def test_no_end_event_is_partial(self):
        r = build_trajectory([ev_step()], "s/t", "baseline", 0, "test")
        assert r.trajectory.status == TrajStatus.PARTIAL
        assert any("no end event" in w for w in r.warnings)

    def test_empty_events_is_empty(self):
        r = build_trajectory([], "s/t", "baseline", 0, "test")
        assert r.trajectory.status == TrajStatus.EMPTY

    def test_unclassifiable_type_skipped_with_warning(self):
        r = build_trajectory([ev_step(type="weird"), ev_step(), END], "s/t", "baseline", 0, "t")
        assert len(r.trajectory.steps) == 1
        assert any("unclassifiable" in w for w in r.warnings)

    def test_big_payload_truncated_and_blob_kept(self):
        big = {"data": "x" * (INLINE_MAX + 50)}
        r = build_trajectory([ev_step(result=big), END], "s/t", "baseline", 0, "t")
        step = r.trajectory.steps[0]
        assert is_truncated_stub(step.result)
        assert step.result_blob in r.blobs

    def test_steps_sorted_by_timestamp(self):
        events = [
            ev_step(name="second", t0="2026-08-04T10:00:05+00:00"),
            ev_step(name="first", t0="2026-08-04T10:00:01+00:00"),
            END,
        ]
        r = build_trajectory(events, "s/t", "baseline", 0, "t")
        assert [s.name for s in r.trajectory.steps] == ["first", "second"]

    def test_untimestamped_keep_source_order(self):
        r = build_trajectory([ev_step(name="a"), ev_step(name="b"), END], "s/t", "baseline", 0, "t")
        assert [s.name for s in r.trajectory.steps] == ["a", "b"]

    def test_error_end_status(self):
        r = build_trajectory(
            [ev_step(), {"ev": "end", "status": "error"}], "s/t", "baseline", 0, "t"
        )
        assert r.trajectory.status == TrajStatus.ERROR


class TestTraceDir:
    def write(self, path: Path, name: str, events):
        (path / name).write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_merges_files_and_worst_end_wins(self, tmp_path: Path):
        self.write(
            tmp_path, "agent-1.jsonl", [ev_step(name="a", t0="2026-08-04T10:00:00+00:00"), END]
        )
        self.write(
            tmp_path,
            "shim-2.jsonl",
            [ev_step(name="b", t0="2026-08-04T10:00:01+00:00"), {"ev": "end", "status": "error"}],
        )
        r = build_from_trace_dir(tmp_path, "s/t", "candidate", 0)
        assert [s.name for s in r.trajectory.steps] == ["a", "b"]
        assert r.trajectory.status == TrajStatus.ERROR

    def test_torn_final_line_dropped(self, tmp_path: Path):
        content = json.dumps(ev_step()) + "\n" + json.dumps(END) + "\n" + '{"ev": "st'
        (tmp_path / "agent-1.jsonl").write_text(content)
        r = build_from_trace_dir(tmp_path, "s/t", "baseline", 0)
        assert len(r.trajectory.steps) == 1
        assert any("torn final line" in w for w in r.warnings)

    def test_empty_dir_is_empty_trajectory(self, tmp_path: Path):
        r = build_from_trace_dir(tmp_path, "s/t", "baseline", 0)
        assert r.trajectory.status == TrajStatus.EMPTY

    def test_demo_agent_events_round_trip(self, tmp_path: Path):
        """The real demo agent's emitted events build a complete trajectory."""
        import subprocess
        import sys

        agent = Path(__file__).parents[2] / "examples" / "refund-agent" / "agent.py"
        env = {"OFFTRACK_TRACE_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
        subprocess.run(
            [sys.executable, str(agent), "--model", "fake-careful"],
            env=env,
            check=True,
            capture_output=True,
        )
        r = build_from_trace_dir(tmp_path, "refund/happy", "baseline", 0)
        t = r.trajectory
        assert t.status == TrajStatus.COMPLETE
        tool_names = [s.name for s in t.steps if s.type == StepType.TOOL_CALL]
        assert tool_names == ["lookup_order", "check_refund_policy", "escalate"]
        assert t.steps[-1].type == StepType.FINAL_ANSWER
        assert t.tokens_in and t.tokens_in > 0
