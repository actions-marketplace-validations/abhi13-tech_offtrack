"""`offtrack ingest` — trace-format debugging and one-off imports."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()
err_console = Console(stderr=True)

ingest_app = typer.Typer(help="Inspect and import traces.", no_args_is_help=True)


@ingest_app.command("explain")
def explain(
    path: Path = typer.Argument(..., help="OTel trace file (OTLP JSON or collector JSONL)"),
) -> None:
    """Print per-span classification decisions for an OTel GenAI trace file."""
    from offtrack.ingest.otel import read_otel_file

    if not path.exists():
        err_console.print(f"[red]✗[/] {path} does not exist", highlight=False)
        raise typer.Exit(4)
    events, decisions, warnings = read_otel_file(path)
    for d in decisions:
        color = "green" if d.step_type else "yellow"
        console.print(f"  [{color}]{d!r}[/]", highlight=False)
    steps = len([e for e in events if e.get("ev") == "step"])
    console.print(f"\n  {steps} step(s) from {len(decisions)} span(s)", highlight=False)
    for w in warnings:
        console.print(f"  [yellow]·[/] {w}", highlight=False)


@ingest_app.command("claude-code")
def claude_code(
    session: Path = typer.Argument(..., help="Claude Code session JSONL file"),
    task: str = typer.Option("claude-code/session", "--task", help="Task key to file under"),
) -> None:
    """Import a Claude Code session as a candidate trajectory."""
    from offtrack.ingest.builder import build_trajectory
    from offtrack.integrations.claude_code import read_claude_code_session
    from offtrack.store import Store

    if not session.exists():
        err_console.print(f"[red]✗[/] {session} does not exist", highlight=False)
        raise typer.Exit(4)
    events, warnings = read_claude_code_session(session)
    result = build_trajectory(events, task, "candidate", 0, source="claude-code")
    store = Store(Path.cwd() / ".offtrack" / "offtrack.db")
    store.save_trajectory(result.trajectory, blobs=result.blobs)
    t = result.trajectory
    console.print(
        f"  [green]✓[/] imported {len(t.steps)} step(s) as {t.trajectory_id} "
        f"(status={t.status.value})",
        highlight=False,
    )
    console.print(f"    inspect: offtrack show {t.trajectory_id}", highlight=False)
    for w in warnings + result.warnings:
        console.print(f"  [yellow]·[/] {w}", highlight=False)
