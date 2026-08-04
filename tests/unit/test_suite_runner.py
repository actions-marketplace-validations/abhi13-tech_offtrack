"""Suite loading/resolution and the runner scenario matrix."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from offtrack.model import TrajStatus
from offtrack.runner import run_task
from offtrack.suite import SuiteConfigError, load_suite_file, resolve_tasks

DEMO = Path(__file__).parents[2] / "examples" / "refund-agent"


def write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "offtrack.yaml"
    p.write_text(textwrap.dedent(body))
    return p


BASIC = """
version: 1
suites:
  - name: refund
    tasks:
      - id: happy
        run: {command: ["python", "agent.py"]}
        input: {order_id: "TEST-1"}
"""


class TestSuiteLoading:
    def test_basic_load(self, tmp_path):
        sf = load_suite_file(write_yaml(tmp_path, BASIC))
        assert sf.suites[0].tasks[0].id == "happy"
        assert sf.config.repetitions == 5

    def test_missing_file_actionable_error(self, tmp_path):
        with pytest.raises(SuiteConfigError, match="offtrack init"):
            load_suite_file(tmp_path / "offtrack.yaml")

    def test_invalid_yaml_rejected(self, tmp_path):
        with pytest.raises(SuiteConfigError, match="not valid YAML"):
            load_suite_file(write_yaml(tmp_path, "suites: [unclosed"))

    def test_task_without_run_mode_rejected(self, tmp_path):
        bad = """
        suites:
          - name: s
            tasks:
              - id: t
                run: {}
        """
        with pytest.raises(SuiteConfigError, match=r"command.*entrypoint|entrypoint.*command"):
            load_suite_file(write_yaml(tmp_path, bad))

    def test_bad_mask_surfaces_at_load(self, tmp_path):
        bad = """
        config:
          mask:
            rules:
              - path: "not-a-path"
        suites: []
        """
        with pytest.raises(Exception, match="must start"):
            load_suite_file(write_yaml(tmp_path, bad))


class TestResolution:
    def test_task_key_format(self, tmp_path):
        sf = load_suite_file(write_yaml(tmp_path, BASIC))
        [rt] = resolve_tasks(sf)
        assert rt.task_key == "refund/happy"

    def test_matrix_expansion(self, tmp_path):
        body = """
        suites:
          - name: s
            tasks:
              - id: t
                run: {command: ["x"]}
                matrix: {model: [a, b]}
                env: {AGENT_MODEL: "${matrix.model}"}
        """
        sf = load_suite_file(write_yaml(tmp_path, body))
        tasks = resolve_tasks(sf)
        assert [t.task_key for t in tasks] == ["s/t#model=a", "s/t#model=b"]
        assert tasks[0].task.env["AGENT_MODEL"] == "a"

    def test_undefined_env_var_listed(self, tmp_path):
        body = """
        suites:
          - name: s
            tasks:
              - id: t
                run: {command: ["x"]}
                env: {K: "${OFFTRACK_TEST_UNDEFINED_VAR}"}
        """
        sf = load_suite_file(write_yaml(tmp_path, body))
        with pytest.raises(SuiteConfigError, match="OFFTRACK_TEST_UNDEFINED_VAR"):
            resolve_tasks(sf)

    def test_only_filter(self, tmp_path):
        sf = load_suite_file(write_yaml(tmp_path, BASIC))
        assert resolve_tasks(sf, only=["happy"])
        assert resolve_tasks(sf, only=["refund/happy"])
        with pytest.raises(SuiteConfigError, match="no tasks match"):
            resolve_tasks(sf, only=["nope"])

    def test_config_hash_stable(self, tmp_path):
        sf = load_suite_file(write_yaml(tmp_path, BASIC))
        [a] = resolve_tasks(sf)
        [b] = resolve_tasks(load_suite_file(tmp_path / "offtrack.yaml"))
        assert a.config_hash == b.config_hash

    def test_repetitions_override(self, tmp_path):
        body = """
        config: {repetitions: 3}
        suites:
          - name: s
            tasks:
              - id: t
                run: {command: ["x"]}
                repetitions: 7
        """
        sf = load_suite_file(write_yaml(tmp_path, body))
        assert resolve_tasks(sf)[0].repetitions == 7


def demo_task(tmp_path: Path, model: str = "fake-careful", command=None) -> object:
    cmd = command or f'["{sys.executable}", "{DEMO}/agent.py", "--model", "{model}"]'
    body = f"""
    suites:
      - name: refund
        tasks:
          - id: happy
            run: {{command: {cmd}}}
            input: {{order_id: "TEST-1"}}
    """
    sf = load_suite_file(write_yaml(tmp_path, body))
    return resolve_tasks(sf)[0]


class TestRunner:
    def test_demo_agent_runs_n_times(self, tmp_path):
        rt = demo_task(tmp_path)
        result = run_task(rt, "baseline", tmp_path / "work", "run1", repetitions=2)
        assert len(result.outcomes) == 2
        for o in result.outcomes:
            assert o.result.trajectory.status == TrajStatus.COMPLETE
            assert o.exit_code == 0
            tools = [s.name for s in o.result.trajectory.steps if s.type.value == "tool_call"]
            assert tools == ["lookup_order", "check_refund_policy", "escalate"]

    def test_crash_recorded_as_error_not_divergence(self, tmp_path):
        rt = demo_task(tmp_path, command=f'["{sys.executable}", "-c", "import sys; sys.exit(3)"]')
        result = run_task(rt, "candidate", tmp_path / "work", "run1", repetitions=1)
        [o] = result.outcomes
        assert o.exit_code == 3
        # No traces emitted → EMPTY, an ERROR-sample, never PASS/FAIL material.
        assert o.result.trajectory.status == TrajStatus.EMPTY

    def test_timeout_kills_and_records(self, tmp_path):
        rt = demo_task(
            tmp_path,
            command=f'["{sys.executable}", "-c", "import time; time.sleep(30)"]',
        )
        object.__setattr__(rt, "timeout_s", 1) if hasattr(rt, "__dataclass_fields__") else setattr(
            rt, "timeout_s", 1
        )
        result = run_task(rt, "candidate", tmp_path / "work", "run1", repetitions=1)
        [o] = result.outcomes
        assert o.timed_out
        assert o.result.trajectory.status in (TrajStatus.TIMEOUT, TrajStatus.EMPTY)

    def test_missing_command_actionable(self, tmp_path):
        rt = demo_task(tmp_path, command='["definitely-not-a-real-binary-xyz"]')
        result = run_task(rt, "candidate", tmp_path / "work", "run1", repetitions=1)
        [o] = result.outcomes
        assert any("not found" in w for w in o.warnings)

    def test_sloppy_persona_produces_different_trajectory(self, tmp_path):
        careful = (
            run_task(demo_task(tmp_path), "baseline", tmp_path / "w1", "r1", repetitions=1)
            .outcomes[0]
            .result.trajectory
        )
        sloppy = (
            run_task(
                demo_task(tmp_path, model="fake-sloppy"),
                "candidate",
                tmp_path / "w2",
                "r2",
                repetitions=1,
            )
            .outcomes[0]
            .result.trajectory
        )
        assert careful.content_hash != sloppy.content_hash
        sloppy_tools = [s.name for s in sloppy.steps if s.type.value == "tool_call"]
        assert sloppy_tools == ["lookup_order", "issue_refund"]


class TestEntrypoint:
    def test_entrypoint_runs_in_child(self, tmp_path):
        mod = tmp_path / "myagent.py"
        mod.write_text(
            "def run(task_input):\n    return f\"handled {task_input.get('order_id')}\"\n"
        )
        body = """
        suites:
          - name: s
            tasks:
              - id: t
                run: {entrypoint: "myagent:run"}
                input: {order_id: "X1"}
        """
        sf = load_suite_file(write_yaml(tmp_path, body))
        [rt] = resolve_tasks(sf)
        import os

        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(tmp_path) + (os.pathsep + old if old else "")
        try:
            result = run_task(rt, "baseline", tmp_path / "work", "r1", repetitions=1)
        finally:
            if old is None:
                del os.environ["PYTHONPATH"]
            else:
                os.environ["PYTHONPATH"] = old
        [o] = result.outcomes
        assert o.exit_code == 0
        finals = [s for s in o.result.trajectory.steps if s.type.value == "final_answer"]
        assert finals and finals[0].result == "handled X1"

    def test_entrypoint_exception_contained(self, tmp_path):
        mod = tmp_path / "badagent.py"
        mod.write_text("def run(task_input):\n    raise RuntimeError('boom')\n")
        body = """
        suites:
          - name: s
            tasks:
              - id: t
                run: {entrypoint: "badagent:run"}
        """
        sf = load_suite_file(write_yaml(tmp_path, body))
        [rt] = resolve_tasks(sf)
        import os

        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(tmp_path) + (os.pathsep + old if old else "")
        try:
            result = run_task(rt, "candidate", tmp_path / "work", "r1", repetitions=1)
        finally:
            if old is None:
                del os.environ["PYTHONPATH"]
            else:
                os.environ["PYTHONPATH"] = old
        [o] = result.outcomes
        assert o.exit_code == 1
        assert "boom" in o.stderr_tail
