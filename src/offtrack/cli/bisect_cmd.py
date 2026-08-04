"""`offtrack bisect` CLI command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console

console = Console()
err_console = Console(stderr=True)

EXIT_SETUP = 4


def bisect_cmd(
    good: str = typer.Option(..., "--good", help="Last known-good ref (baselines pinned here)"),
    bad: str = typer.Option("HEAD", "--bad", help="Known-bad ref"),
    runs: int = typer.Option(3, "--runs", help="Runs per probe commit"),
    tasks: list[str] | None = typer.Argument(None, help="Task keys/ids to probe (default all)"),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip endpoint verification (2 probes faster, riskier)"
    ),
) -> None:
    """Binary-search the commit that made your agent's behavior diverge."""
    from offtrack.bisect import BisectError, bisect

    repo = Path.cwd()
    try:
        outcome = bisect(
            repo,
            good=good,
            bad=bad,
            runs=runs,
            only=tasks or None,
            verify_endpoints=not no_verify,
            progress=lambda msg: console.print(f"  [dim]{msg}[/]", highlight=False),
        )
    except BisectError as e:
        err_console.print(f"[red]✗[/] {e}", highlight=False)
        raise typer.Exit(EXIT_SETUP) from None

    assert outcome.first_bad is not None
    show = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "show",
            "--no-patch",
            "--format=%h %an %ad%n  %s",
            outcome.first_bad,
        ],
        capture_output=True,
        text=True,
    )
    console.print(f"\n  [red bold]first bad commit:[/] {outcome.first_bad}", highlight=False)
    if show.returncode == 0:
        console.print(f"  {show.stdout.strip()}", highlight=False)

    last_bad = next((p for p in reversed(outcome.probes) if p.bad), None)
    if last_bad and last_bad.detail:
        div = last_bad.detail
        base = div.get("baseline_step")
        cand = div.get("candidate_step")
        if isinstance(base, dict) or isinstance(cand, dict):
            console.print("\n  divergence at that commit:", highlight=False)
            if isinstance(base, dict):
                console.print(f"    [dim]baseline[/]  {base['name']}", highlight=False)
            if isinstance(cand, dict):
                console.print(f"    [red]this run[/]  {cand['name']}", highlight=False)
    raise typer.Exit(0)
