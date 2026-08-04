"""TrajectoryBuilder: capture events → finalized Trajectory.

Every adapter (shims, OTel, LangGraph, Claude Code) translates its input into
capture-event dicts; ALL policy — ordering, parallel grouping, truncation,
hashing, totals — lives here so adapters stay translation-only.

Capture-event JSONL format (one JSON object per line):
  {"ev": "step", "v": 1, "type": "...", "name": "...", "args": ..., "result": ...,
   "usage": {"tokens_in": N, "tokens_out": N}, "model": "...",
   "t0": iso8601, "t1": iso8601, "group": "...", "status": "ok"}
  {"ev": "end", "status": "complete"}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from offtrack.model import (
    Step,
    StepStatus,
    StepType,
    Trajectory,
    TrajStatus,
    truncate_payload,
)


class BuildResult:
    def __init__(self, trajectory: Trajectory, blobs: dict[str, bytes], warnings: list[str]):
        self.trajectory = trajectory
        self.blobs = blobs
        self.warnings = warnings


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def build_trajectory(
    events: list[dict[str, Any]],
    task_key: str,
    kind: str,
    attempt: int,
    source: str,
) -> BuildResult:
    """Assemble capture events into a finalized, truncated, hashed Trajectory."""
    warnings: list[str] = []
    blobs: dict[str, bytes] = {}
    steps: list[Step] = []
    end_status: str | None = None
    dropped_payloads = 0

    for i, ev in enumerate(events):
        if ev.get("ev") == "end":
            end_status = ev.get("status")
            continue
        if ev.get("ev") != "step":
            warnings.append(f"event {i}: unknown ev {ev.get('ev')!r} skipped")
            continue
        try:
            step_type = StepType(ev["type"])
        except (KeyError, ValueError):
            warnings.append(f"event {i}: unclassifiable type {ev.get('type')!r} skipped")
            continue

        args_res = truncate_payload(ev.get("args"))
        result_res = truncate_payload(ev.get("result"))
        for res in (args_res, result_res):
            if res.blob is not None and res.blob_sha256 is not None:
                blobs[res.blob_sha256] = res.blob
            if res.dropped:
                dropped_payloads += 1

        usage = ev.get("usage") or {}
        raw_status = ev.get("status", "ok")
        try:
            status = StepStatus(raw_status)
        except ValueError:
            status = StepStatus.UNKNOWN
            warnings.append(f"event {i}: unknown status {raw_status!r} recorded as 'unknown'")

        steps.append(
            Step(
                idx=len(steps),
                type=step_type,
                name=str(ev.get("name") or step_type.value),
                args=args_res.inline,
                result=result_res.inline,
                status=status,
                model=ev.get("model"),
                tokens_in=usage.get("tokens_in"),
                tokens_out=usage.get("tokens_out"),
                cost_usd=ev.get("cost_usd"),
                started_at=_parse_ts(ev.get("t0")),
                ended_at=_parse_ts(ev.get("t1")),
                parallel_group=ev.get("group"),
                args_blob=args_res.blob_sha256,
                result_blob=result_res.blob_sha256,
            )
        )

    if dropped_payloads:
        warnings.append(
            f"{dropped_payloads} payload(s) over 512 KiB stored as structure-only stubs"
        )

    # Stable order: by start time when present, tie-break by source sequence.
    order = sorted(range(len(steps)), key=lambda i: (steps[i].started_at or datetime.max, i))
    steps = [steps[i] for i in order]

    if end_status is None and steps:
        traj_status = TrajStatus.PARTIAL
        warnings.append("no end event — trajectory recorded as partial")
    elif not steps:
        traj_status = TrajStatus.EMPTY
    else:
        try:
            traj_status = TrajStatus(end_status)
        except ValueError:
            traj_status = TrajStatus.PARTIAL
            warnings.append(f"unknown end status {end_status!r} recorded as partial")

    traj = Trajectory(
        task_key=task_key,
        kind=kind,  # type: ignore[arg-type]
        attempt=attempt,
        status=traj_status,
        source=source,
        steps=steps,
        meta={"warnings": warnings} if warnings else {},
    ).finalize()
    return BuildResult(traj, blobs, warnings)


def read_capture_jsonl(path: Path, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    """Read capture-event JSONL, dropping a torn final line (crash-safe writes)."""
    events: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                if warnings is not None:
                    warnings.append(f"{path.name}: torn final line dropped (partial write)")
            elif warnings is not None:
                warnings.append(f"{path.name}: unparseable line {i + 1} dropped")
    return events


def build_from_trace_dir(
    trace_dir: Path,
    task_key: str,
    kind: str,
    attempt: int,
) -> BuildResult:
    """Assemble a trajectory from every capture file in an attempt's trace dir.

    Directory-per-attempt is the association mechanism: everything in the dir
    belongs to this attempt. Multiple capture files (shim + agent, or multiple
    processes) merge by timestamp.
    """
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    files = sorted(trace_dir.glob("*.jsonl"))
    for f in files:
        events.extend(read_capture_jsonl(f, warnings))

    if len(files) > 1:
        # Multiple emitters: keep at most one end event (the "worst" status wins).
        ends = [e for e in events if e.get("ev") == "end"]
        if len(ends) > 1:
            severity = {"complete": 0, "partial": 1, "timeout": 2, "error": 3}
            worst = max(ends, key=lambda e: severity.get(e.get("status", "partial"), 1))
            events = [e for e in events if e.get("ev") != "end"] + [worst]

    result = build_trajectory(events, task_key, kind, attempt, source=_infer_source(files))
    result.warnings[:0] = warnings
    if warnings:
        result.trajectory.meta.setdefault("warnings", [])
        for w in warnings:
            if w not in result.trajectory.meta["warnings"]:
                result.trajectory.meta["warnings"].insert(0, w)
    return result


def _infer_source(files: list[Path]) -> str:
    names = {f.name.split("-")[0] for f in files}
    if names == {"shim"}:
        return "shim"
    if len(names) == 1:
        return names.pop() or "capture"
    return "mixed"
