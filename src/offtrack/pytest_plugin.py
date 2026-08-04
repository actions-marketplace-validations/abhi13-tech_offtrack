"""pytest plugin: record trajectories inside tests and assert against baselines.

    @pytest.mark.offtrack(task="refund/over-limit")
    def test_refund(offtrack):
        with offtrack.record() as trace:
            run_my_agent(...)
        trace.assert_matches_baseline()

Verdict semantics: FAIL → test failure; INCONCLUSIVE → pytest.skip (flip with
--offtrack-inconclusive=fail); no baseline → record + warn + pass (flip with
--offtrack-require-baseline for CI). Baselines are read from baselines/*.json
(read-only, xdist/fork-safe); recorded runs spill to
.offtrack/pending/<worker>.jsonl and merge into the DB at session finish.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings as _warnings
from pathlib import Path
from typing import Any

try:
    import pytest
except ImportError:  # pragma: no cover - entry point never loads without pytest
    raise

from offtrack.align import best_variant_match
from offtrack.align.engine import VariantMatch
from offtrack.compare import _prepare
from offtrack.ingest import build_from_trace_dir
from offtrack.mask import parse_rules
from offtrack.model import Trajectory, TrajStatus


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("offtrack")
    group.addoption(
        "--offtrack-inconclusive",
        default="skip",
        choices=("skip", "fail"),
        help="INCONCLUSIVE verdicts skip (default) or fail the test",
    )
    group.addoption(
        "--offtrack-require-baseline",
        action="store_true",
        help="fail when a task has no baseline instead of recording one",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "offtrack(task=...): bind this test to an offtrack task key")


def _project_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "offtrack.yaml").exists() or (candidate / "baselines").is_dir():
            return candidate
    return cur


class TraceHandle:
    """What the `offtrack` fixture yields: capture window + assertions."""

    def __init__(self, task_key: str, root: Path, request: Any):
        self.task_key = task_key
        self.root = root
        self._request = request
        self.trajectory: Trajectory | None = None
        self._trace_dir: Path | None = None

    # -- capture --------------------------------------------------------------

    def record(self) -> TraceHandle:
        return self

    def __enter__(self) -> TraceHandle:
        self._trace_dir = Path(tempfile.mkdtemp(prefix="offtrack-pytest-"))
        self._old_env = os.environ.get("OFFTRACK_TRACE_DIR")
        os.environ["OFFTRACK_TRACE_DIR"] = str(self._trace_dir)
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._trace_dir is not None
        os.environ.pop("OFFTRACK_TRACE_DIR", None)
        if self._old_env is not None:
            os.environ["OFFTRACK_TRACE_DIR"] = self._old_env
        result = build_from_trace_dir(self._trace_dir, self.task_key, "candidate", 0)
        self.trajectory = result.trajectory
        self.trajectory.source = "pytest"
        self.trajectory.meta["nodeid"] = self._request.node.nodeid
        _spill(self.root, self.trajectory)

    # -- baseline access ------------------------------------------------------

    def _baselines(self) -> list[Trajectory] | None:
        suite, _, task = self.task_key.partition("/")
        path = self.root / "baselines" / suite / f"{task.replace('/', '_')}.json"
        if not path.exists():
            return None
        doc = json.loads(path.read_text())
        return [Trajectory.model_validate(t) for t in doc["trajectories"]]

    def _match(self) -> VariantMatch | None:
        assert self.trajectory is not None, "assert called outside/before record() block"
        baselines = self._baselines()
        if baselines is None:
            return None
        rules = parse_rules(None)  # builtins; suite masks need offtrack.yaml
        masked = [b.model_copy(update={"steps": _prepare(b, rules, [])}) for b in baselines]
        probe = self.trajectory.model_copy(update={"steps": _prepare(self.trajectory, rules, [])})
        base_models = {s.model for b in baselines for s in b.steps if s.model}
        cand_models = {s.model for s in self.trajectory.steps if s.model}
        exempt = bool(base_models) and bool(cand_models) and base_models != cand_models
        return best_variant_match(masked, probe, model_exempt=exempt)

    # -- assertions -----------------------------------------------------------

    @property
    def first_divergence(self) -> int | None:
        match = self._match()
        if match is None:
            return None
        return match.alignment.first_divergence

    def assert_matches_baseline(self) -> None:
        assert self.trajectory is not None
        if self.trajectory.status == TrajStatus.EMPTY:
            pytest.fail(
                f"offtrack: no trace events captured for {self.task_key} — is the "
                "agent emitting to $OFFTRACK_TRACE_DIR inside the record() block?"
            )
        match = self._match()
        if match is None:
            if self._request.config.getoption("--offtrack-require-baseline"):
                pytest.fail(
                    f"offtrack: no baseline for {self.task_key} "
                    "(--offtrack-require-baseline is set). Record one: offtrack record"
                )
            _warnings.warn(
                f"offtrack: no baseline for {self.task_key} — recorded this run; "
                "promote it with `offtrack record` to start gating",
                stacklevel=2,
            )
            return
        al = match.alignment
        if not al.is_divergent:
            return
        op = al.ops[al.first_divergence or 0]
        detail = f"kind={al.divergence_kind}, op={al.first_divergence}"
        if self._request.config.getoption("--offtrack-inconclusive") == "skip" and (
            self.trajectory.status != TrajStatus.COMPLETE
        ):
            pytest.skip(f"offtrack: run was {self.trajectory.status.value} — inconclusive")
        pytest.fail(
            f"offtrack: trajectory diverged from baseline at step "
            f"{op.a_idx if op.a_idx is not None else op.b_idx} ({detail}). "
            f"Inspect: offtrack show {self.trajectory.trajectory_id}"
        )

    def assert_no_divergence(self, before_step: int) -> None:
        match = self._match()
        if match is None:
            return
        al = match.alignment
        if al.is_divergent:
            op = al.ops[al.first_divergence or 0]
            where = op.a_idx if op.a_idx is not None else op.b_idx
            if where is not None and where < before_step:
                pytest.fail(
                    f"offtrack: divergence at step {where}, before the allowed step {before_step}"
                )

    def assert_cost_under(self, usd: float) -> None:
        assert self.trajectory is not None
        cost = self.trajectory.cost_usd
        if cost is not None and cost > usd:
            pytest.fail(f"offtrack: run cost ${cost:.4f} exceeds budget ${usd:.4f}")

    def assert_max_steps(self, n: int) -> None:
        assert self.trajectory is not None
        if len(self.trajectory.steps) > n:
            pytest.fail(f"offtrack: {len(self.trajectory.steps)} steps exceeds max {n}")


def _worker_id(config: Any) -> str:
    return str(getattr(config, "workerinput", {}).get("workerid", "main"))


def _spill(root: Path, traj: Trajectory) -> None:
    pending = root / ".offtrack" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with (pending / f"{worker}.jsonl").open("a") as f:
        f.write(traj.model_dump_json() + "\n")


@pytest.fixture
def offtrack(request: Any) -> TraceHandle:
    marker = request.node.get_closest_marker("offtrack")
    task = marker.kwargs.get("task") if marker else None
    if task is None:
        task = f"pytest/{request.node.name}"
    root = _project_root(Path(request.config.rootpath))
    return TraceHandle(task, root, request)


def pytest_sessionfinish(session: Any) -> None:
    """Controller merges spilled trajectories into the DB (spill-and-merge)."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return  # workers spill; only the controller merges
    root = _project_root(Path(session.config.rootpath))
    pending = root / ".offtrack" / "pending"
    if not pending.exists():
        return
    files = list(pending.glob("*.jsonl"))
    if not files:
        return
    from offtrack.store import Store

    store = Store(root / ".offtrack" / "offtrack.db")
    merged = 0
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                traj = Trajectory.model_validate_json(line)
            except ValueError:
                continue
            if store.load_trajectory(traj.trajectory_id) is None:
                store.save_trajectory(traj)
                merged += 1
        f.unlink()
    store.close()
