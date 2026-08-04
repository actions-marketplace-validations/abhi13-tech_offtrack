"""The product promise, end to end: the gate catches the sloppy persona.

Runs the real CLI against the real demo agent in fake mode ($0, offline):
record golden baselines with the careful persona, verify the careful persona
PASSes (exit 0), then verify the sloppy persona FAILs (exit 1) with the first
divergence at the skipped policy check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DEMO = REPO / "examples" / "refund-agent"

CONFIG = """\
version: 1
config:
  repetitions: 5
  timeout_s: 60
suites:
  - name: refund
    tasks:
      - id: over-limit
        run: {command: ["@@PY@@", "@@AGENT@@", "--model", "${AGENT_MODEL}"]}
        input: {order_id: "TEST-1"}
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A fresh project dir using the demo agent."""
    for name in ("agent.py", "tools.py", "providers.py"):
        shutil.copy(DEMO / name, tmp_path / name)
    shutil.copytree(DEMO / "fake_scripts", tmp_path / "fake_scripts")
    config = CONFIG.replace("@@PY@@", sys.executable).replace(
        "@@AGENT@@", str(tmp_path / "agent.py")
    )
    (tmp_path / "offtrack.yaml").write_text(config)
    return tmp_path


def offtrack(project: Path, *args: str, model: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, AGENT_MODEL=model)
    env.pop("SSL_CERT_FILE", None)
    return subprocess.run(
        [sys.executable, "-m", "offtrack.cli.app", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_gate_catches_the_sloppy_persona(project: Path) -> None:
    rec = offtrack(project, "record", model="fake-careful")
    assert rec.returncode == 0, rec.stderr
    assert (project / "baselines" / "refund" / "over-limit.json").exists()

    ok = offtrack(project, "check", model="fake-careful")
    assert ok.returncode == 0, f"careful persona must PASS:\n{ok.stdout}\n{ok.stderr}"
    assert "PASS" in ok.stdout

    bad = offtrack(
        project,
        "check",
        "--report",
        "terminal",
        "--report",
        "json",
        "--report-out",
        str(project / "report.json"),
        model="fake-sloppy",
    )
    assert bad.returncode == 1, f"sloppy persona must FAIL:\n{bad.stdout}\n{bad.stderr}"
    assert "first divergence" in bad.stdout

    # The JSON report localizes the divergence at the skipped policy check.
    report = json.loads((project / "report.json").read_text())
    assert report["verdict"] == "FAIL"
    [task] = report["tasks"]
    div = task["first_divergence"]
    assert div is not None
    named = json.dumps(div)
    assert "check_refund_policy" in named
    assert task["behavioral"]["rate_candidate"] == 1.0
    assert task["behavioral"]["rate_baseline"] == 0.0


def test_no_baseline_is_setup_error_not_failure(project: Path) -> None:
    result = offtrack(project, "check", model="fake-careful")
    assert result.returncode == 4
    err = result.stdout + result.stderr
    assert "setup error" in err and "offtrack record" in err
