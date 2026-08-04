"""offtrack CLI.

Exit codes (D1): 0 PASS · 1 FAIL · 3 INCONCLUSIVE · 4 setup/environment error.
Exit 2 is never used — Typer/Click reserve it for usage errors, and a typo'd
flag must never look like a verdict. Infra problems never masquerade as FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from offtrack import __version__
from offtrack.compare import build_report, compare_task
from offtrack.render import render_markdown, render_terminal
from offtrack.runner import run_task
from offtrack.store import SchemaTooNewError, Store
from offtrack.store.baseline_io import export_baseline, import_all
from offtrack.suite import (
    ResolvedTask,
    SuiteConfigError,
    SuiteFile,
    load_suite_file,
    resolve_tasks,
)

app = typer.Typer(
    name="offtrack",
    help="git diff for AI agent runs — record golden trajectories, find the first "
    "divergent step, gate CI on real regressions.",
    no_args_is_help=True,
    add_completion=False,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INCONCLUSIVE = 3
EXIT_SETUP = 4

console = Console()
err_console = Console(stderr=True)


def _fail_setup(message: str) -> None:
    err_console.print(f"[red]✗[/] {message}", highlight=False)
    raise typer.Exit(EXIT_SETUP)


def _project_root(config: Path) -> Path:
    return config.resolve().parent


def _open_store(root: Path) -> Store:
    try:
        return Store(root / ".offtrack" / "offtrack.db")
    except SchemaTooNewError as e:
        _fail_setup(str(e))
        raise  # unreachable


def _load(config: Path, only: list[str] | None) -> tuple[SuiteFile, list[ResolvedTask]]:
    try:
        sf = load_suite_file(config)
        return sf, resolve_tasks(sf, only=only)
    except SuiteConfigError as e:
        _fail_setup(str(e))
        raise  # unreachable


@app.callback()
def _main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("offtrack.yaml"), "--config", help="Suite config file"),
) -> None:
    ctx.obj = {"config": config}


@app.command()
def version() -> None:
    """Print the offtrack version."""
    console.print(__version__)


from offtrack.cli.bisect_cmd import bisect_cmd  # noqa: E402
from offtrack.cli.ingest_cmds import ingest_app  # noqa: E402
from offtrack.cli.inspect_cmds import baseline_app, diff, doctor, show  # noqa: E402

app.command()(show)
app.command()(diff)
app.command()(doctor)
app.command("bisect")(bisect_cmd)
app.add_typer(baseline_app, name="baseline")
app.add_typer(ingest_app, name="ingest")


@app.command()
def record(
    ctx: typer.Context,
    tasks: list[str] | None = typer.Argument(None, help="Task keys/ids to record (default all)"),
    runs: int | None = typer.Option(None, "--runs", help="Recordings per task (default: config)"),
    label: str = typer.Option("default", "--label", help="Baseline label"),
) -> None:
    """Run the suite N times and store golden baselines (auto-exported to baselines/)."""
    config: Path = ctx.obj["config"]
    sf, resolved = _load(config, tasks)
    root = _project_root(config)
    store = _open_store(root)
    run_id = store.create_run("offtrack record")
    work = root / ".offtrack" / "tmp" / run_id
    baselines_dir = root / "baselines"

    for rt in resolved:
        n = runs or rt.repetitions
        console.print(f"  recording [bold]{rt.task_key}[/] × {n}…", highlight=False)
        result = run_task(rt, "baseline", work, run_id, repetitions=n)
        usable = result.usable
        if not usable:
            _fail_setup(
                f"task {rt.task_key}: all {n} attempts produced no traces. "
                "Committed baselines are unaffected. Is the agent emitting capture "
                "events to $OFFTRACK_TRACE_DIR? Run with a working task or check "
                "run.command in offtrack.yaml."
            )
        baseline_id = store.create_baseline(rt.task_key, rt.config_hash, label=label)
        for outcome in usable:
            store.save_trajectory(
                outcome.result.trajectory,
                run_id=run_id,
                baseline_id=baseline_id,
                blobs=outcome.result.blobs,
            )
        path = export_baseline(store, baseline_id, baselines_dir)
        skipped = result.empty_count
        note = f" ({skipped} empty attempt(s) skipped)" if skipped else ""
        console.print(
            f"  [green]✓[/] {rt.task_key}: {len(usable)} recording(s) → "
            f"{path.relative_to(root)}{note}",
            highlight=False,
        )

        # Volatile-field hint: masks derived from baseline self-variance.
        from offtrack.mask import parse_rules, suggest_masks

        rules = parse_rules(rt.mask_config)
        from offtrack.compare import _prepare  # reuse ignore+mask pipeline

        prepared = [_prepare(o.result.trajectory, rules, sf.config.ignore_steps) for o in usable]
        suggestions = suggest_masks(prepared)
        if suggestions:
            console.print(
                "  [yellow]hint:[/] these fields varied across recordings — consider masking:",
                highlight=False,
            )
            for s in suggestions[:5]:
                console.print(
                    f'    - step: {s["step"]}\n      path: "{s["path"]}"', highlight=False
                )

    console.print(
        f"\n  [green]done[/] — baselines committed-ready in {baselines_dir.relative_to(root)}/"
    )


@app.command()
def check(
    ctx: typer.Context,
    tasks: list[str] | None = typer.Argument(None, help="Task keys/ids to check (default all)"),
    runs: int | None = typer.Option(None, "--runs", help="Candidate runs per task"),
    report: list[str] = typer.Option(
        ["terminal"], "--report", help="Output format(s): terminal, md, json, github"
    ),
    report_out: Path | None = typer.Option(None, "--report-out", help="Write md/json here"),
    inconclusive_as: str = typer.Option(
        "inconclusive", "--inconclusive-as", help="Map INCONCLUSIVE to: pass, fail, inconclusive"
    ),
    allow_stale: bool = typer.Option(
        False, "--allow-stale", help="Compare against stale baselines"
    ),
) -> None:
    """Re-run the suite and compare against baselines. The CI entrypoint."""
    config: Path = ctx.obj["config"]
    sf, resolved = _load(config, tasks)
    root = _project_root(config)
    store = _open_store(root)

    baselines_dir = root / "baselines"
    if baselines_dir.exists():
        import_all(store, baselines_dir)

    run_id = store.create_run("offtrack check")
    work = root / ".offtrack" / "tmp" / run_id

    task_reports = []
    for rt in resolved:
        row = store.active_baseline(rt.task_key)
        if row is None:
            _fail_setup(
                f"no baseline for {rt.task_key}.\n"
                f"  Looked in: {baselines_dir}/ and .offtrack/offtrack.db\n"
                "  This is a setup error, not a test failure. To create baselines:\n"
                "    offtrack record\n"
                "  then re-run: offtrack check"
            )
            raise AssertionError  # unreachable: _fail_setup raises
        baselines = store.baseline_trajectories(row["baseline_id"])
        n = runs or rt.repetitions
        console.print(f"  checking [bold]{rt.task_key}[/] × {n}…", highlight=False)
        result = run_task(rt, "candidate", work, run_id, repetitions=n)
        for outcome in result.outcomes:
            store.save_trajectory(
                outcome.result.trajectory, run_id=run_id, blobs=outcome.result.blobs
            )
        candidates = [o.result.trajectory for o in result.outcomes]
        task_reports.append(
            compare_task(
                rt,
                sf.config,
                baselines,
                candidates,
                baseline_config_hash=row["config_hash"],
                allow_stale=allow_stale,
            )
        )

    doc = build_report(run_id, task_reports)

    if "terminal" in report:
        render_terminal(doc, console)
    if "md" in report or "github" in report:
        md = render_markdown(doc)
        if report_out:
            report_out.write_text(md)
        if "github" in report:
            import os

            summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_path:
                with open(summary_path, "a") as f:
                    f.write(md + "\n")
        if not report_out and "md" in report:
            console.print(md, highlight=False)
    if "json" in report:
        out = json.dumps(doc, indent=2, default=str)
        if report_out and "md" not in report:
            report_out.write_text(out)
        else:
            print(out)

    verdict = doc["verdict"]
    if verdict == "INCONCLUSIVE" and inconclusive_as == "pass":
        raise typer.Exit(EXIT_PASS)
    if verdict == "INCONCLUSIVE" and inconclusive_as == "fail":
        raise typer.Exit(EXIT_FAIL)
    raise typer.Exit(
        {
            "PASS": EXIT_PASS,
            "FAIL": EXIT_FAIL,
            "INCONCLUSIVE": EXIT_INCONCLUSIVE,
            "ERROR": EXIT_SETUP,
        }[verdict]
    )


@app.command()
def init(ctx: typer.Context) -> None:
    """Scaffold offtrack.yaml, .offtrack/, and baselines/ in this project."""
    config: Path = ctx.obj["config"]
    root = Path.cwd()

    ot_dir = root / ".offtrack"
    if ot_dir.exists():
        console.print("  .offtrack/ exists, skipped", highlight=False)
    else:
        ot_dir.mkdir()
        (ot_dir / ".gitignore").write_text("*\n!.gitignore\n")
        console.print("  [green]created[/] .offtrack/ (self-gitignoring)", highlight=False)

    cfg = root / config.name
    if cfg.exists():
        console.print(f"  {cfg.name} exists, skipped", highlight=False)
    else:
        cfg.write_text(SCAFFOLD_YAML)
        console.print(f"  [green]created[/] {cfg.name}", highlight=False)

    bdir = root / "baselines"
    if bdir.exists():
        console.print("  baselines/ exists, skipped", highlight=False)
    else:
        bdir.mkdir()
        (bdir / "README.md").write_text(
            "Golden trajectories. Commit these — changing agent behavior becomes a reviewed act.\n"
        )
        console.print("  [green]created[/] baselines/ (commit these)", highlight=False)

    console.print(
        "\n  next steps:\n"
        "    1. edit offtrack.yaml — point run.command at your agent\n"
        "    2. offtrack record     — capture golden trajectories\n"
        "    3. change something, then: offtrack check",
        highlight=False,
    )


@app.command("list")
def list_cmd(
    ctx: typer.Context,
    what: str = typer.Argument("tasks", help="tasks | runs | baselines"),
) -> None:
    """List tasks, runs, or baselines."""
    config: Path = ctx.obj["config"]
    root = _project_root(config)
    if what == "tasks":
        _, resolved = _load(config, None)
        for rt in resolved:
            console.print(f"  {rt.task_key}  (×{rt.repetitions}, {rt.timeout_s}s timeout)")
        return
    store = _open_store(root)
    if what == "runs":
        rows = store.conn.execute(
            "SELECT run_id, created_at, argv FROM runs ORDER BY run_id DESC LIMIT 20"
        ).fetchall()
        for r in rows:
            console.print(f"  {r['run_id']}  {r['created_at']}  {r['argv']}")
    elif what == "baselines":
        rows = store.conn.execute(
            "SELECT baseline_id, task_key, label, created_at, active FROM baselines "
            "ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        for r in rows:
            flag = "[green]active[/]" if r["active"] else "[dim]inactive[/]"
            console.print(f"  {r['baseline_id']}  {r['task_key']}  [{r['label']}]  {flag}")
    else:
        _fail_setup(f"unknown list target {what!r} — use tasks, runs, or baselines")


SCAFFOLD_YAML = """\
version: 1

config:
  repetitions: 5        # recordings/checks per task — stats need >=3
  timeout_s: 300
  # mask:               # volatile-field masking (uuids/timestamps ON by default)
  #   rules:
  #     - field: request_id

suites:
  - name: my-agent
    tasks:
      - id: example-task
        # Your agent, run as a subprocess. It receives:
        #   OFFTRACK_TASK_INPUT  (JSON of `input` below)
        #   OFFTRACK_TRACE_DIR   (write capture-event JSONL here)
        run: {command: ["python", "my_agent.py"]}
        input: {question: "What is our refund policy?"}
        # Or run an importable entrypoint in a contained child process:
        # run: {entrypoint: "my_agent:run"}
"""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
