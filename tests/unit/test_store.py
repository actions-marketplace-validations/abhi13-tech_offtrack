"""Store: schema/migrations, trajectory round-trip, baseline export/import."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from offtrack.model import Step, StepType, Trajectory
from offtrack.store import CURRENT_VERSION, SchemaTooNewError, Store
from offtrack.store.baseline_io import (
    BaselineFormatTooNewError,
    export_baseline,
    import_all,
    import_baseline,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / ".offtrack" / "offtrack.db")


def make_traj(task_key="suite/task", kind="baseline", attempt=0, n=3) -> Trajectory:
    steps = [
        Step(
            idx=i,
            type=StepType.TOOL_CALL,
            name=f"tool{i}",
            args={"i": i},
            result={"ok": True},
            tokens_in=10,
            tokens_out=5,
        )
        for i in range(n)
    ]
    return Trajectory(task_key=task_key, kind=kind, attempt=attempt, steps=steps).finalize()


class TestMigrations:
    def test_fresh_db_at_current_version(self, store: Store):
        row = store.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == CURRENT_VERSION

    def test_reopen_is_noop(self, tmp_path: Path):
        path = tmp_path / "db.sqlite"
        Store(path).close()
        Store(path).close()  # no error, no double-apply

    def test_newer_schema_hard_error(self, tmp_path: Path):
        path = tmp_path / "db.sqlite"
        s = Store(path)
        s.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, '2099-01-01')",
            (CURRENT_VERSION + 5,),
        )
        s.conn.commit()
        s.close()
        with pytest.raises(SchemaTooNewError, match="newer offtrack"):
            Store(path)

    def test_wal_mode_active(self, store: Store):
        assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


class TestTrajectoryRoundTrip:
    def test_save_and_load_identical(self, store: Store):
        t = make_traj()
        store.save_trajectory(t)
        loaded = store.load_trajectory(t.trajectory_id)
        assert loaded is not None
        assert loaded.content_hash == t.content_hash
        assert loaded.model_dump(mode="json") == t.model_dump(mode="json")

    def test_load_missing_returns_none(self, store: Store):
        assert store.load_trajectory("nope") is None

    def test_blobs_stored_and_fetched(self, store: Store):
        t = make_traj()
        store.save_trajectory(t, blobs={"abc123": b"compressed"})
        assert store.get_blob("abc123") == b"compressed"
        assert store.get_blob("missing") is None

    def test_trajectories_for_filters_kind(self, store: Store):
        store.save_trajectory(make_traj(kind="baseline", attempt=0))
        store.save_trajectory(make_traj(kind="candidate", attempt=0))
        assert len(store.trajectories_for("suite/task", "baseline")) == 1

    def test_run_linked(self, store: Store):
        run_id = store.create_run("offtrack check")
        t = make_traj(kind="candidate")
        store.save_trajectory(t, run_id=run_id)
        row = store.conn.execute(
            "SELECT run_id FROM trajectories WHERE trajectory_id = ?", (t.trajectory_id,)
        ).fetchone()
        assert row["run_id"] == run_id


class TestBaselines:
    def test_create_and_lookup_active(self, store: Store):
        bid = store.create_baseline("suite/task", "cfg1", model="fake-careful")
        row = store.active_baseline("suite/task")
        assert row is not None and row["baseline_id"] == bid

    def test_new_baseline_deactivates_old(self, store: Store):
        store.create_baseline("suite/task", "cfg1")
        bid2 = store.create_baseline("suite/task", "cfg2")
        assert store.active_baseline("suite/task")["baseline_id"] == bid2
        active = store.conn.execute(
            "SELECT COUNT(*) FROM baselines WHERE task_key='suite/task' AND active=1"
        ).fetchone()[0]
        assert active == 1

    def test_labels_independent(self, store: Store):
        a = store.create_baseline("suite/task", "cfg", label="default")
        b = store.create_baseline("suite/task", "cfg", label="gpt5")
        assert store.active_baseline("suite/task", "default")["baseline_id"] == a
        assert store.active_baseline("suite/task", "gpt5")["baseline_id"] == b


class TestBaselineExportImport:
    def test_round_trip(self, store: Store, tmp_path: Path):
        bid = store.create_baseline("suite/task", "cfg1", model="fake-careful")
        for i in range(3):
            store.save_trajectory(make_traj(attempt=i), baseline_id=bid)
        out = tmp_path / "baselines"
        path = export_baseline(store, bid, out)
        assert path == out / "suite" / "task.json"

        fresh = Store(tmp_path / "other" / "db.sqlite")
        imported = import_baseline(fresh, path)
        assert imported == bid
        trajs = fresh.baseline_trajectories(bid)
        assert len(trajs) == 3
        assert {t.content_hash for t in trajs} == {
            t.content_hash for t in store.baseline_trajectories(bid)
        }

    def test_import_idempotent(self, store: Store, tmp_path: Path):
        bid = store.create_baseline("suite/task", "cfg1")
        store.save_trajectory(make_traj(), baseline_id=bid)
        path = export_baseline(store, bid, tmp_path / "b")
        fresh = Store(tmp_path / "f" / "db.sqlite")
        import_baseline(fresh, path)
        import_baseline(fresh, path)
        assert len(fresh.baseline_trajectories(bid)) == 1

    def test_export_stable_git_diffs(self, store: Store, tmp_path: Path):
        bid = store.create_baseline("suite/task", "cfg1")
        store.save_trajectory(make_traj(), baseline_id=bid)
        p1 = export_baseline(store, bid, tmp_path / "b")
        first = p1.read_text()
        p2 = export_baseline(store, bid, tmp_path / "b")
        assert p2.read_text() == first

    def test_newer_format_rejected(self, store: Store, tmp_path: Path):
        doc = {"offtrack_schema": 99, "baseline": {}, "trajectories": []}
        p = tmp_path / "x.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(BaselineFormatTooNewError, match="Upgrade offtrack"):
            import_baseline(store, p)

    def test_import_all_walks_tree(self, store: Store, tmp_path: Path):
        b1 = store.create_baseline("s1/t1", "c")
        store.save_trajectory(make_traj(task_key="s1/t1"), baseline_id=b1)
        b2 = store.create_baseline("s2/t2", "c")
        store.save_trajectory(make_traj(task_key="s2/t2"), baseline_id=b2)
        out = tmp_path / "baselines"
        export_baseline(store, b1, out)
        export_baseline(store, b2, out)
        fresh = Store(tmp_path / "f" / "db.sqlite")
        assert sorted(import_all(fresh, out)) == sorted([b1, b2])


class TestConcurrency:
    def test_second_writer_gets_clean_error_when_locked(self, tmp_path: Path):
        path = tmp_path / "db.sqlite"
        a = Store(path)
        b = Store(path)
        # Hold a write txn open on A, then try to write on B.
        a.conn.execute("BEGIN IMMEDIATE")
        b.conn.execute("PRAGMA busy_timeout=50")
        import offtrack.store.db as dbmod

        orig_sleep = dbmod.time.sleep
        dbmod.time.sleep = lambda _t: None  # fast test
        try:
            with pytest.raises(dbmod.StoreBusyError, match="another offtrack run"), b._write():
                pass
        finally:
            dbmod.time.sleep = orig_sleep
            a.conn.rollback()

    def test_error_in_write_rolls_back(self, store: Store):
        t = make_traj()
        with pytest.raises(sqlite3.IntegrityError), store._write() as c:
            c.execute(
                "INSERT INTO trajectories(trajectory_id, task_key, kind, attempt, status, "
                "content_hash) VALUES (?,?,?,?,?,?)",
                (t.trajectory_id, t.task_key, "bad-kind", 0, "complete", "h"),
            )
        assert store.load_trajectory(t.trajectory_id) is None
