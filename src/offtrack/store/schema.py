"""SQLite DDL and the migration runner.

Migrations are ordered functions; v1 ships as migration #1 so the runner is
exercised from the first release (no "initial schema" special case). A DB
written by a newer offtrack is a hard error — never forward-compat reads.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_V1 = """
CREATE TABLE schema_version(
  version INTEGER NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE TABLE tasks(
  task_key TEXT PRIMARY KEY,
  suite TEXT NOT NULL,
  task_id TEXT NOT NULL,
  config_json TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE baselines(
  baseline_id TEXT PRIMARY KEY,
  task_key TEXT NOT NULL REFERENCES tasks(task_key),
  label TEXT NOT NULL DEFAULT 'default',
  created_at TEXT,
  git_ref TEXT,
  model TEXT,
  notes TEXT,
  config_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE runs(
  run_id TEXT PRIMARY KEY,
  created_at TEXT,
  git_ref TEXT,
  argv TEXT,
  offtrack_version TEXT,
  pricing_version TEXT,
  config_hash TEXT,
  meta_json TEXT
);

CREATE TABLE trajectories(
  trajectory_id TEXT PRIMARY KEY,
  task_key TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('baseline','candidate')),
  baseline_id TEXT REFERENCES baselines(baseline_id),
  run_id TEXT REFERENCES runs(run_id),
  attempt INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('complete','partial','error','timeout','empty')),
  source TEXT,
  content_hash TEXT NOT NULL,
  step_count INTEGER,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd REAL,
  wall_ms INTEGER,
  started_at TEXT,
  ended_at TEXT,
  meta_json TEXT
);
CREATE INDEX ix_traj_task ON trajectories(task_key, kind);
CREATE INDEX ix_traj_run ON trajectories(run_id);

CREATE TABLE steps(
  trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  args_json TEXT,
  result_json TEXT,
  args_blob TEXT,
  result_blob TEXT,
  status TEXT,
  model TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd REAL,
  started_at TEXT,
  ended_at TEXT,
  latency_ms INTEGER,
  parallel_group TEXT,
  content_hash TEXT NOT NULL,
  PRIMARY KEY(trajectory_id, idx)
);

CREATE TABLE blobs(
  sha256 TEXT PRIMARY KEY,
  size INTEGER,
  encoding TEXT,
  data BLOB
);

CREATE TABLE alignments(
  candidate_id TEXT NOT NULL,
  baseline_traj_id TEXT NOT NULL,
  mask_hash TEXT NOT NULL,
  score REAL,
  norm_score REAL,
  first_div_idx INTEGER,
  ops_json TEXT,
  PRIMARY KEY(candidate_id, baseline_traj_id, mask_hash)
);

CREATE TABLE verdicts(
  verdict_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  task_key TEXT NOT NULL,
  baseline_id TEXT,
  verdict TEXT NOT NULL CHECK(verdict IN ('PASS','FAIL','INCONCLUSIVE','ERROR')),
  div_rate_baseline REAL,
  div_rate_candidate REAL,
  p_value REAL,
  first_div_json TEXT,
  metrics_json TEXT,
  warnings_json TEXT,
  mask_hash TEXT,
  created_at TEXT
);
"""


def _migration_1(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_V1)


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_migration_1]

CURRENT_VERSION = len(MIGRATIONS)


class SchemaTooNewError(RuntimeError):
    """The DB was written by a newer offtrack."""


def _db_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0)
    except sqlite3.OperationalError:
        return 0  # no schema_version table → fresh DB


def migrate(conn: sqlite3.Connection, db_path: Path | None = None) -> int:
    """Bring the DB to CURRENT_VERSION. Returns the version migrated from."""
    version = _db_version(conn)
    if version > CURRENT_VERSION:
        raise SchemaTooNewError(
            f"database written by newer offtrack (schema v{version}, this build reads "
            f"≤v{CURRENT_VERSION}). Upgrade offtrack, or delete .offtrack/offtrack.db "
            "to start fresh (committed baselines/ are unaffected)."
        )
    if version == CURRENT_VERSION:
        return version

    if db_path is not None and version > 0:
        backup_dir = db_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(db_path, backup_dir / f"{db_path.name}.pre-v{CURRENT_VERSION}")

    for target in range(version + 1, CURRENT_VERSION + 1):
        with conn:  # one transaction per migration
            MIGRATIONS[target - 1](conn)
            conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (target, datetime.now(timezone.utc).isoformat(timespec="milliseconds")),
            )
            conn.execute(f"PRAGMA user_version = {target}")
    return version
