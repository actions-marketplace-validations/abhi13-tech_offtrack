"""Store: the single write path to .offtrack/offtrack.db.

WAL journal, 10s busy timeout, short BEGIN IMMEDIATE transactions with a
jittered retry loop. CI is read-mostly by design: baselines come from
committed JSON, each job writes its own local runs DB.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offtrack.model import Step, StepStatus, Trajectory, TrajStatus, new_ulid
from offtrack.store.schema import migrate

BUSY_RETRIES = 3


class StoreBusyError(RuntimeError):
    pass


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="milliseconds") if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _connect(db_path)
        migrate(self.conn, db_path)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Short immediate transaction with jittered retry on SQLITE_BUSY."""
        for attempt in range(BUSY_RETRIES + 1):
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                if attempt == BUSY_RETRIES:
                    raise StoreBusyError(
                        "database locked — is another offtrack run active? "
                        "Nothing was written; committed baselines/ are unaffected. "
                        "For parallel CI jobs use one DB per job (the default in isolated "
                        "workspaces) and share baselines via exported JSON."
                    ) from e
                time.sleep(0.1 * (attempt + 1) + random.random() * 0.1)
        try:
            yield self.conn
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    # --- runs ---------------------------------------------------------------

    def create_run(self, argv: str, meta: dict[str, Any] | None = None) -> str:
        from offtrack import __version__

        run_id = new_ulid()
        with self._write() as c:
            c.execute(
                "INSERT INTO runs(run_id, created_at, argv, offtrack_version, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    _iso(datetime.now(timezone.utc)),
                    argv,
                    __version__,
                    json.dumps(meta or {}),
                ),
            )
        return run_id

    # --- trajectories -------------------------------------------------------

    def save_trajectory(
        self,
        traj: Trajectory,
        run_id: str | None = None,
        baseline_id: str | None = None,
        blobs: dict[str, bytes] | None = None,
    ) -> None:
        with self._write() as c:
            c.execute(
                "INSERT INTO trajectories(trajectory_id, task_key, kind, baseline_id, run_id, "
                "attempt, status, source, content_hash, step_count, tokens_in, tokens_out, "
                "cost_usd, wall_ms, started_at, ended_at, meta_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    traj.trajectory_id,
                    traj.task_key,
                    traj.kind,
                    baseline_id,
                    run_id,
                    traj.attempt,
                    traj.status.value,
                    traj.source,
                    traj.content_hash,
                    len(traj.steps),
                    traj.tokens_in,
                    traj.tokens_out,
                    traj.cost_usd,
                    traj.wall_ms,
                    _iso(traj.started_at),
                    _iso(traj.ended_at),
                    json.dumps(traj.meta),
                ),
            )
            c.executemany(
                "INSERT INTO steps(trajectory_id, idx, type, name, args_json, result_json, "
                "args_blob, result_blob, status, model, tokens_in, tokens_out, cost_usd, "
                "started_at, ended_at, latency_ms, parallel_group, content_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        traj.trajectory_id,
                        s.idx,
                        s.type.value,
                        s.name,
                        json.dumps(s.args) if s.args is not None else None,
                        json.dumps(s.result) if s.result is not None else None,
                        s.args_blob,
                        s.result_blob,
                        s.status.value,
                        s.model,
                        s.tokens_in,
                        s.tokens_out,
                        s.cost_usd,
                        _iso(s.started_at),
                        _iso(s.ended_at),
                        s.latency_ms,
                        s.parallel_group,
                        s.content_hash,
                    )
                    for s in traj.steps
                ],
            )
            for sha, data in (blobs or {}).items():
                c.execute(
                    "INSERT OR IGNORE INTO blobs(sha256, size, encoding, data) VALUES (?,?,?,?)",
                    (sha, len(data), "zlib", data),
                )

    def load_trajectory(self, trajectory_id: str) -> Trajectory | None:
        row = self.conn.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_trajectory(row)

    def _row_to_trajectory(self, row: sqlite3.Row) -> Trajectory:
        steps = [
            Step(
                idx=s["idx"],
                type=s["type"],
                name=s["name"],
                args=json.loads(s["args_json"]) if s["args_json"] else None,
                result=json.loads(s["result_json"]) if s["result_json"] else None,
                status=StepStatus(s["status"] or "unknown"),
                model=s["model"],
                tokens_in=s["tokens_in"],
                tokens_out=s["tokens_out"],
                cost_usd=s["cost_usd"],
                started_at=_parse_dt(s["started_at"]),
                ended_at=_parse_dt(s["ended_at"]),
                latency_ms=s["latency_ms"],
                parallel_group=s["parallel_group"],
                content_hash=s["content_hash"],
                args_blob=s["args_blob"],
                result_blob=s["result_blob"],
            )
            for s in self.conn.execute(
                "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY idx",
                (row["trajectory_id"],),
            )
        ]
        return Trajectory(
            trajectory_id=row["trajectory_id"],
            task_key=row["task_key"],
            kind=row["kind"],
            attempt=row["attempt"],
            status=TrajStatus(row["status"]),
            source=row["source"] or "unknown",
            steps=steps,
            content_hash=row["content_hash"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            cost_usd=row["cost_usd"],
            wall_ms=row["wall_ms"],
            started_at=_parse_dt(row["started_at"]),
            ended_at=_parse_dt(row["ended_at"]),
            meta=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    def trajectories_for(self, task_key: str, kind: str) -> list[Trajectory]:
        rows = self.conn.execute(
            "SELECT * FROM trajectories WHERE task_key = ? AND kind = ? ORDER BY trajectory_id",
            (task_key, kind),
        ).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def get_blob(self, sha256: str) -> bytes | None:
        row = self.conn.execute("SELECT data FROM blobs WHERE sha256 = ?", (sha256,)).fetchone()
        return row["data"] if row else None

    # --- baselines ----------------------------------------------------------

    def create_baseline(
        self,
        task_key: str,
        config_hash: str,
        label: str = "default",
        model: str | None = None,
        git_ref: str | None = None,
    ) -> str:
        baseline_id = new_ulid()
        with self._write() as c:
            c.execute(
                "INSERT OR IGNORE INTO tasks(task_key, suite, task_id, config_json, "
                "config_hash, updated_at) VALUES (?,?,?,?,?,?)",
                (
                    task_key,
                    task_key.split("/")[0],
                    task_key.split("/", 1)[-1],
                    "{}",
                    config_hash,
                    _iso(datetime.now(timezone.utc)),
                ),
            )
            # One active baseline per (task, label): deactivate predecessors.
            c.execute(
                "UPDATE baselines SET active = 0 WHERE task_key = ? AND label = ?",
                (task_key, label),
            )
            c.execute(
                "INSERT INTO baselines(baseline_id, task_key, label, created_at, git_ref, "
                "model, config_hash, active) VALUES (?,?,?,?,?,?,?,1)",
                (
                    baseline_id,
                    task_key,
                    label,
                    _iso(datetime.now(timezone.utc)),
                    git_ref,
                    model,
                    config_hash,
                ),
            )
        return baseline_id

    def active_baseline(self, task_key: str, label: str = "default") -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM baselines WHERE task_key = ? AND label = ? AND active = 1 "
            "ORDER BY created_at DESC LIMIT 1",
            (task_key, label),
        ).fetchone()
        return row

    def baseline_trajectories(self, baseline_id: str) -> list[Trajectory]:
        rows = self.conn.execute(
            "SELECT * FROM trajectories WHERE baseline_id = ? ORDER BY attempt",
            (baseline_id,),
        ).fetchall()
        return [self._row_to_trajectory(r) for r in rows]
