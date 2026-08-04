"""offtrack bisect: find the commit that introduced a trajectory divergence.

Semantics mirror git bisect, but the predicate is behavioral: at each probe
commit the agent code from THAT commit runs in an isolated git worktree,
while the golden baselines stay pinned from the --good ref (the last point
where behavior was known-good). A probe is BAD when a majority of its runs
diverge on any task. Binary search assumes the usual bisect monotonicity;
endpoints are verified first so a wrong --good/--bad claim fails loudly
instead of producing a confident wrong answer.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from offtrack.compare import compare_task
from offtrack.model import Trajectory
from offtrack.runner import run_task
from offtrack.suite import SuiteConfigError, load_suite_file, resolve_tasks


class BisectError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise BisectError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def commit_range(repo: Path, good: str, bad: str) -> list[str]:
    """Commits after good, up to and including bad — oldest first."""
    out = _git(repo, "rev-list", "--reverse", f"{good}..{bad}")
    return [line for line in out.splitlines() if line]


def load_baselines_at(repo: Path, ref: str) -> dict[str, list[Trajectory]]:
    """Baseline trajectories per task_key, read from a ref's baselines/ tree."""
    try:
        listing = _git(repo, "ls-tree", "-r", "--name-only", ref, "--", "baselines")
    except BisectError:
        listing = ""
    result: dict[str, list[Trajectory]] = {}
    for path in listing.splitlines():
        if not path.endswith(".json") or path.endswith("README.md"):
            continue
        try:
            doc = json.loads(_git(repo, "show", f"{ref}:{path}"))
        except (BisectError, json.JSONDecodeError):
            continue
        if "trajectories" not in doc or "baseline" not in doc:
            continue
        task_key = doc["baseline"]["task_key"]
        result[task_key] = [Trajectory.model_validate(t) for t in doc["trajectories"]]
    if not result:
        raise BisectError(
            f"no baselines found at {ref}:baselines/ — bisect needs committed "
            "golden baselines at the --good ref. Record and commit them first."
        )
    return result


@dataclass
class ProbeResult:
    commit: str
    bad: bool
    rates: dict[str, float] = field(default_factory=dict)
    detail: dict[str, object] | None = None  # first_divergence of the worst task


@dataclass
class BisectOutcome:
    first_bad: str | None
    probes: list[ProbeResult]
    log: list[str] = field(default_factory=list)


def probe_commit(
    repo: Path,
    commit: str,
    baselines: dict[str, list[Trajectory]],
    runs: int,
    only: list[str] | None,
) -> ProbeResult:
    """Run the suite as of `commit` in a temp worktree; compare to baselines."""
    with tempfile.TemporaryDirectory(prefix="offtrack-bisect-") as tmp:
        worktree = Path(tmp) / "wt"
        _git(repo, "worktree", "add", "--detach", str(worktree), commit)
        try:
            config_path = worktree / "offtrack.yaml"
            try:
                sf = load_suite_file(config_path)
                resolved = resolve_tasks(sf, only=only)
            except SuiteConfigError as e:
                raise BisectError(f"commit {commit[:10]}: cannot load offtrack.yaml — {e}") from e

            rates: dict[str, float] = {}
            worst_rate = -1.0
            detail = None
            for rt in resolved:
                if rt.task_key not in baselines:
                    continue
                run_result = run_task(
                    rt,
                    "candidate",
                    worktree / ".offtrack" / "bisect",
                    run_id=f"bisect-{commit[:10]}",
                    repetitions=runs,
                    cwd=worktree,  # relative run.command paths resolve per-commit
                )
                candidates = [o.result.trajectory for o in run_result.outcomes]
                report = compare_task(
                    rt,
                    sf.config,
                    baselines[rt.task_key],
                    candidates,
                    allow_stale=True,  # config drift across commits is expected
                )
                rate = report["behavioral"]["rate_candidate"]
                if rate is None:
                    raise BisectError(
                        f"commit {commit[:10]}: task {rt.task_key} produced no "
                        f"usable runs ({report['behavioral']['reason']}) — cannot "
                        "classify this probe; fix the agent invocation first"
                    )
                rates[rt.task_key] = rate
                if rate > worst_rate:
                    worst_rate = rate
                    detail = report.get("first_divergence")
            if not rates:
                raise BisectError(
                    f"commit {commit[:10]}: no tasks match the baselines at the "
                    "good ref — check task keys or pass explicit task filters"
                )
            bad = any(rate > 0.5 for rate in rates.values())
            return ProbeResult(commit=commit, bad=bad, rates=rates, detail=detail)
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
            )


def bisect(
    repo: Path,
    good: str,
    bad: str = "HEAD",
    runs: int = 3,
    only: list[str] | None = None,
    verify_endpoints: bool = True,
    progress: object = None,
) -> BisectOutcome:
    log: list[str] = []
    probes: list[ProbeResult] = []

    def note(msg: str) -> None:
        log.append(msg)
        if progress is not None:
            progress(msg)  # type: ignore[operator]

    good_sha = _git(repo, "rev-parse", good)
    bad_sha = _git(repo, "rev-parse", bad)
    if good_sha == bad_sha:
        raise BisectError("--good and --bad are the same commit")

    baselines = load_baselines_at(repo, good_sha)
    note(f"baselines pinned from {good_sha[:10]} ({len(baselines)} task(s))")

    commits = commit_range(repo, good_sha, bad_sha)
    if not commits:
        raise BisectError(f"{bad} is not a descendant of {good} — bisect needs a linear range")
    note(f"searching {len(commits)} commit(s), ~{max(1, len(commits)).bit_length()} probes")

    if verify_endpoints:
        note(f"verifying endpoints (good={good_sha[:10]}, bad={bad_sha[:10]})…")
        good_probe = probe_commit(repo, good_sha, baselines, runs, only)
        probes.append(good_probe)
        if good_probe.bad:
            raise BisectError(
                f"--good {good_sha[:10]} already diverges from its own baselines "
                f"(rates: {good_probe.rates}). Endpoints are wrong; nothing to bisect."
            )
        bad_probe = probe_commit(repo, bad_sha, baselines, runs, only)
        probes.append(bad_probe)
        if not bad_probe.bad:
            raise BisectError(
                f"--bad {bad_sha[:10]} does not diverge (rates: {bad_probe.rates}). "
                "Endpoints are wrong; nothing to bisect."
            )

    # Standard first-bad binary search over the good..bad range.
    lo, hi = 0, len(commits) - 1  # invariant: first bad is in commits[lo..hi]
    while lo < hi:
        mid = (lo + hi) // 2
        commit = commits[mid]
        note(f"probing {commit[:10]} ({hi - lo + 1} candidate commit(s) left)…")
        result = probe_commit(repo, commit, baselines, runs, only)
        probes.append(result)
        note(f"  → {'BAD' if result.bad else 'good'} {result.rates}")
        if result.bad:
            hi = mid
        else:
            lo = mid + 1

    first_bad = commits[lo]
    note(f"first bad commit: {first_bad[:10]}")
    return BisectOutcome(first_bad=first_bad, probes=probes, log=log)
