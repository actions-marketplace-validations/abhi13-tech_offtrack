"""offtrack bisect finds the exact commit where behavior regressed.

Builds a real git repo: 6 commits, the demo agent flips from the careful to
the sloppy persona at commit index 3 (0-based, within the good..bad range).
Baselines are recorded at the good commit. Fully offline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from offtrack.bisect import BisectError, bisect, commit_range, load_baselines_at

REPO_ROOT = Path(__file__).parents[2]
DEMO = REPO_ROOT / "examples" / "refund-agent"

CONFIG = """\
version: 1
config:
  repetitions: 2
  timeout_s: 60
suites:
  - name: refund
    tasks:
      - id: over-limit
        run: {command: ["@@PY@@", "agent_entry.py"]}
        input: {order_id: "TEST-1"}
"""

ENTRY = """\
import subprocess, sys, pathlib
model = (pathlib.Path(__file__).parent / "MODEL.txt").read_text().strip()
sys.exit(subprocess.run(
    [sys.executable, str(pathlib.Path(__file__).parent / "agent.py"), "--model", model]
).returncode)
"""


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture
def bisect_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")

    for name in ("agent.py", "tools.py", "providers.py"):
        shutil.copy(DEMO / name, repo / name)
    shutil.copytree(DEMO / "fake_scripts", repo / "fake_scripts")
    (repo / "offtrack.yaml").write_text(CONFIG.replace("@@PY@@", sys.executable))
    (repo / "agent_entry.py").write_text(ENTRY)
    (repo / "MODEL.txt").write_text("fake-careful\n")

    # good commit: record + commit baselines with the careful persona.
    env_python = sys.executable
    proc = subprocess.run(
        [env_python, "-m", "offtrack.cli.app", "record"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "good: baselines recorded")

    shas = [git(repo, "rev-parse", "HEAD")]  # index 0 = good
    for i in range(1, 6):
        if i == 3:  # THE regression commit
            (repo / "MODEL.txt").write_text("fake-sloppy\n")
            msg = f"commit {i}: swap model (the regression)"
        else:
            (repo / f"note{i}.txt").write_text(f"innocuous change {i}\n")
            msg = f"commit {i}: innocuous"
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", msg)
        shas.append(git(repo, "rev-parse", "HEAD"))
    return repo, shas


class TestBisect:
    def test_finds_the_regression_commit(self, bisect_repo):
        repo, shas = bisect_repo
        outcome = bisect(repo, good=shas[0], bad=shas[5], runs=1)
        assert outcome.first_bad == shas[3]
        # The reported divergence points at the skipped policy check.
        bad_probe = next(p for p in outcome.probes if p.commit == shas[3] and p.bad)
        assert "check_refund_policy" in json.dumps(bad_probe.detail)

    def test_wrong_endpoints_fail_loudly(self, bisect_repo):
        repo, shas = bisect_repo
        with pytest.raises(BisectError, match="does not diverge"):
            bisect(repo, good=shas[0], bad=shas[1], runs=1)  # bad is actually good

    def test_same_commit_rejected(self, bisect_repo):
        repo, shas = bisect_repo
        with pytest.raises(BisectError, match="same commit"):
            bisect(repo, good=shas[0], bad=shas[0])

    def test_commit_range_oldest_first(self, bisect_repo):
        repo, shas = bisect_repo
        commits = commit_range(repo, shas[0], shas[5])
        assert commits == shas[1:]

    def test_baselines_loaded_from_good_ref(self, bisect_repo):
        repo, shas = bisect_repo
        baselines = load_baselines_at(repo, shas[0])
        assert "refund/over-limit" in baselines
        assert len(baselines["refund/over-limit"]) == 2

    def test_missing_baselines_actionable(self, tmp_path):
        repo = tmp_path / "empty"
        repo.mkdir()
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.email", "t@t")
        git(repo, "config", "user.name", "t")
        (repo / "x.txt").write_text("x")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "no baselines here")
        with pytest.raises(BisectError, match="committed golden baselines"):
            load_baselines_at(repo, "HEAD")
