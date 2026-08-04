"""Task runner: invoke the agent N times, one trace dir per attempt.

Directory-per-attempt is the trace-association mechanism — everything an
attempt writes into its OFFTRACK_TRACE_DIR belongs to that attempt. Crashes,
timeouts, and empty runs are recorded as statuses, never conflated with
behavioral divergence.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from offtrack.ingest import BuildResult, build_from_trace_dir
from offtrack.model import TrajStatus
from offtrack.suite import ResolvedTask


@dataclass
class AttemptOutcome:
    attempt: int
    result: BuildResult
    exit_code: int | None
    timed_out: bool
    stderr_tail: str = ""
    warnings: list[str] = field(default_factory=list)


def run_attempt(
    rt: ResolvedTask,
    attempt: int,
    kind: str,
    work_root: Path,
    run_id: str,
    cwd: Path | None = None,
) -> AttemptOutcome:
    trace_dir = work_root / rt.task_key.replace("/", "_") / str(attempt)
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True)

    env = dict(os.environ)
    env.update(rt.task.env)
    env.update(
        (rt.matrix_env and {f"OFFTRACK_MATRIX_{k.upper()}": v for k, v in rt.matrix_env.items()})
        or {}
    )
    env.update(
        {
            "OFFTRACK_RUN_ID": run_id,
            "OFFTRACK_TASK_KEY": rt.task_key,
            "OFFTRACK_ATTEMPT": str(attempt),
            "OFFTRACK_TASK_INPUT": json.dumps(rt.task.input),
            "OFFTRACK_TRACE_DIR": str(trace_dir),
        }
    )

    if rt.task.run.command:
        argv = rt.task.run.command
    else:
        argv = [sys.executable, "-m", "offtrack._child", str(rt.task.run.entrypoint)]

    timed_out = False
    exit_code: int | None = None
    stderr_tail = ""
    try:
        proc = subprocess.run(
            argv,
            env=env,
            timeout=rt.timeout_s,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        exit_code = proc.returncode
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stderr_tail = ((e.stderr or b"").decode(errors="replace") if e.stderr else "")[-2000:]
    except FileNotFoundError as e:
        return AttemptOutcome(
            attempt,
            build_from_trace_dir(trace_dir, rt.task_key, kind, attempt),
            None,
            False,
            stderr_tail=str(e),
            warnings=[f"command not found: {argv[0]} — check run.command in offtrack.yaml"],
        )

    result = build_from_trace_dir(trace_dir, rt.task_key, kind, attempt)
    traj = result.trajectory

    if timed_out:
        traj.status = TrajStatus.TIMEOUT
        result.warnings.append(f"attempt {attempt}: timed out after {rt.timeout_s}s")
    elif exit_code not in (0, None) and traj.status != TrajStatus.EMPTY:
        traj.status = TrajStatus.ERROR
        result.warnings.append(
            f"attempt {attempt}: agent exited {exit_code} after step "
            f"{len(traj.steps)} — counted per on_crash policy"
        )
    traj.finalize()
    return AttemptOutcome(attempt, result, exit_code, timed_out, stderr_tail, result.warnings)


@dataclass
class TaskRunResult:
    task_key: str
    outcomes: list[AttemptOutcome]

    @property
    def usable(self) -> list[AttemptOutcome]:
        return [o for o in self.outcomes if o.result.trajectory.status != TrajStatus.EMPTY]

    @property
    def empty_count(self) -> int:
        return len(self.outcomes) - len(self.usable)


def run_task(
    rt: ResolvedTask,
    kind: str,
    work_root: Path,
    run_id: str,
    repetitions: int | None = None,
    cwd: Path | None = None,
) -> TaskRunResult:
    n = repetitions or rt.repetitions
    outcomes = [run_attempt(rt, i, kind, work_root, run_id, cwd=cwd) for i in range(n)]
    return TaskRunResult(rt.task_key, outcomes)
