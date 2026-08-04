"""Failure-UX matrix: every error names what broke, what's safe, and the fix.

Uses Typer's CliRunner where possible for speed; the flows that need real
subprocesses live in tests/integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from offtrack.cli.app import app
from offtrack.store import Store

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def offtrack(*args: str):
    result = runner.invoke(app, list(args))
    # Rich wraps lines to terminal width; normalize for phrase assertions.
    result.flat = " ".join(result.output.split())  # type: ignore[attr-defined]
    return result


class TestInit:
    def test_scaffolds_and_is_idempotent(self, project: Path):
        r1 = offtrack("init")
        assert r1.exit_code == 0
        assert (project / ".offtrack" / ".gitignore").read_text().startswith("*")
        assert (project / "offtrack.yaml").exists()
        assert (project / "baselines" / "README.md").exists()

        r2 = offtrack("init")
        assert r2.exit_code == 0
        assert r2.output.count("skipped") == 3  # nothing overwritten

    def test_scaffold_yaml_is_valid(self, project: Path):
        offtrack("init")
        r = offtrack("list", "tasks")
        assert r.exit_code == 0
        assert "my-agent/example-task" in r.flat


class TestMissingConfig:
    def test_check_without_config_exits_4_with_fix(self, project: Path):
        r = offtrack("check")
        assert r.exit_code == 4
        assert "offtrack init" in r.flat

    def test_record_without_config_exits_4(self, project: Path):
        r = offtrack("record")
        assert r.exit_code == 4


class TestNoBaseline:
    def test_check_with_config_but_no_baseline(self, project: Path):
        offtrack("init")
        # Point the scaffold task at a real command so loading succeeds.
        r = offtrack("check")
        assert r.exit_code == 4
        assert "setup error" in r.flat
        assert "offtrack record" in r.flat


class TestCorruptedDb:
    def test_doctor_detects_and_repairs(self, project: Path):
        offtrack("init")
        db = project / ".offtrack" / "offtrack.db"
        Store(db).close()  # create valid db
        db.write_bytes(b"this is not a sqlite database at all")

        r = offtrack("doctor")
        assert r.exit_code == 4
        assert "integrity" in r.flat
        assert "baselines/ are unaffected" in r.flat
        assert "--repair" in r.flat

        r2 = offtrack("doctor", "--repair")
        assert r2.exit_code == 0
        assert "rebuilt" in r2.flat
        # Old file preserved, new DB valid.
        assert list(project.glob(".offtrack/offtrack.db.corrupt-*"))
        r3 = offtrack("doctor")
        assert r3.exit_code == 0

    def test_doctor_clean_project(self, project: Path):
        offtrack("init")
        r = offtrack("doctor")
        assert r.exit_code == 0
        assert "valid" in r.flat


class TestBaselineFormatGuard:
    def test_newer_format_rejected_with_upgrade_hint(self, project: Path):
        offtrack("init")
        bad = project / "baselines" / "s"
        bad.mkdir(parents=True)
        (bad / "t.json").write_text(
            json.dumps({"offtrack_schema": 99, "baseline": {}, "trajectories": []})
        )
        r = offtrack("baseline", "import", str(bad / "t.json"))
        assert r.exit_code != 0
        # error propagates with the upgrade message
        assert "Upgrade offtrack" in str(r.output) + str(r.exception)


class TestShowDiff:
    def test_show_missing_trajectory(self, project: Path):
        offtrack("init")
        r = offtrack("show", "NOPE123")
        assert r.exit_code == 4
        assert "list runs" in r.flat

    def test_unknown_list_target(self, project: Path):
        offtrack("init")
        r = offtrack("list", "nonsense")
        assert r.exit_code == 4
        assert "tasks, runs, or baselines" in r.flat
