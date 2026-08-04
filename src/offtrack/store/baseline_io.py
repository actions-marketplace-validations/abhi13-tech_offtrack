"""Baseline export/import: baselines/<suite>/<task_id>.json.

The JSON files are the source of truth in CI (committed, human-reviewable);
the DB is a cache. Canonical serialization gives stable git diffs. Import is
idempotent by trajectory content_hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offtrack.model import Trajectory, canonical_json, is_truncated_stub
from offtrack.store.db import Store

FORMAT_VERSION = 1
_READABLE = {FORMAT_VERSION, FORMAT_VERSION - 1}


class BaselineFormatTooNewError(RuntimeError):
    pass


def _task_path(baselines_dir: Path, task_key: str) -> Path:
    # task_key "<suite>/<task_id>[#matrix]" → baselines/<suite>/<task_id>[#matrix].json
    suite, _, task = task_key.partition("/")
    safe_task = task.replace("/", "_")
    return baselines_dir / suite / f"{safe_task}.json"


def export_baseline(
    store: Store,
    baseline_id: str,
    baselines_dir: Path,
    with_payloads: bool = False,
) -> Path:
    row = store.conn.execute(
        "SELECT * FROM baselines WHERE baseline_id = ?", (baseline_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no baseline {baseline_id}")
    trajectories = store.baseline_trajectories(baseline_id)

    doc: dict[str, Any] = {
        "offtrack_schema": FORMAT_VERSION,
        "baseline": {
            "baseline_id": row["baseline_id"],
            "task_key": row["task_key"],
            "label": row["label"],
            "created_at": row["created_at"],
            "git_ref": row["git_ref"],
            "model": row["model"],
            "config_hash": row["config_hash"],
        },
        "trajectories": [_strip(t.model_dump(mode="json"), with_payloads) for t in trajectories],
    }
    path = _task_path(baselines_dir, row["task_key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(canonical_json(doc)), indent=1, sort_keys=True) + "\n")
    return path


def _strip(traj: dict[str, Any], with_payloads: bool) -> dict[str, Any]:
    """Drop oversized inline payloads from exports unless explicitly kept."""
    if with_payloads:
        return traj
    for step in traj["steps"]:
        for key in ("args", "result"):
            if is_truncated_stub(step.get(key)):
                stub = dict(step[key])
                stub.pop("head", None)  # keep sha/size/shape, drop bulk
                step[key] = stub
    return traj


def import_baseline(store: Store, path: Path) -> str:
    doc = json.loads(path.read_text())
    schema = doc.get("offtrack_schema")
    if not isinstance(schema, int) or schema > FORMAT_VERSION:
        raise BaselineFormatTooNewError(
            f"{path} uses baseline format v{schema}; this offtrack reads ≤v{FORMAT_VERSION}. "
            "Upgrade offtrack (pip install -U offtrack). Your file is unaffected."
        )
    meta = doc["baseline"]
    existing = store.conn.execute(
        "SELECT baseline_id FROM baselines WHERE baseline_id = ?", (meta["baseline_id"],)
    ).fetchone()
    if existing:
        return str(meta["baseline_id"])

    with store._write() as c:
        task_key = meta["task_key"]
        c.execute(
            "INSERT OR IGNORE INTO tasks(task_key, suite, task_id, config_json, config_hash, "
            "updated_at) VALUES (?,?,?,?,?,?)",
            (
                task_key,
                task_key.split("/")[0],
                task_key.split("/", 1)[-1],
                "{}",
                meta["config_hash"],
                meta["created_at"],
            ),
        )
        c.execute(
            "UPDATE baselines SET active = 0 WHERE task_key = ? AND label = ?",
            (task_key, meta["label"]),
        )
        c.execute(
            "INSERT INTO baselines(baseline_id, task_key, label, created_at, git_ref, model, "
            "config_hash, active) VALUES (?,?,?,?,?,?,?,1)",
            (
                meta["baseline_id"],
                task_key,
                meta["label"],
                meta["created_at"],
                meta["git_ref"],
                meta["model"],
                meta["config_hash"],
            ),
        )

    # Idempotency key is trajectory_id (preserved in exports) — NOT content_hash:
    # identical baseline recordings legitimately share a hash, and collapsing
    # them would silently shrink the stats layer's sample count.
    seen = {
        r["trajectory_id"]
        for r in store.conn.execute(
            "SELECT trajectory_id FROM trajectories WHERE baseline_id = ?",
            (meta["baseline_id"],),
        )
    }
    for tdoc in doc["trajectories"]:
        traj = Trajectory.model_validate(tdoc)
        if traj.trajectory_id in seen:
            continue
        store.save_trajectory(traj, baseline_id=meta["baseline_id"])
        seen.add(traj.trajectory_id)
    return str(meta["baseline_id"])


def import_all(store: Store, baselines_dir: Path) -> list[str]:
    ids = []
    for path in sorted(baselines_dir.rglob("*.json")):
        ids.append(import_baseline(store, path))
    return ids
