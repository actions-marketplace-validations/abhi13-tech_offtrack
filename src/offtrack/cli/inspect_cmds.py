"""Inspection and maintenance commands: diff, show, baseline, doctor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer
from rich.console import Console

from offtrack.align import best_variant_match
from offtrack.model import Trajectory
from offtrack.store import Store
from offtrack.store.baseline_io import export_baseline, import_all, import_baseline

console = Console()
err_console = Console(stderr=True)

EXIT_SETUP = 4

baseline_app = typer.Typer(help="Manage golden baselines.", no_args_is_help=True)


def _fail(message: str) -> None:
    err_console.print(f"[red]✗[/] {message}", highlight=False)
    raise typer.Exit(EXIT_SETUP)


def _store(root: Path) -> Store:
    return Store(root / ".offtrack" / "offtrack.db")


def _load_traj(store: Store, ref: str) -> Trajectory:
    t = store.load_trajectory(ref)
    if t is None:
        # Prefix match for convenience (ULIDs are long).
        row = store.conn.execute(
            "SELECT trajectory_id FROM trajectories WHERE trajectory_id LIKE ? "
            "ORDER BY trajectory_id DESC LIMIT 2",
            (ref + "%",),
        ).fetchall()
        if len(row) == 1:
            loaded = store.load_trajectory(row[0]["trajectory_id"])
            assert loaded is not None
            return loaded
        if len(row) > 1:
            _fail(f"trajectory prefix {ref!r} is ambiguous — give more characters")
        _fail(
            f"no trajectory {ref!r} in .offtrack/offtrack.db. "
            "Use `offtrack list runs` to find run ids, nothing else is affected."
        )
    assert t is not None
    return t


def show(
    ref: str = typer.Argument(..., help="Trajectory id (or unique prefix)"),
    page: int = typer.Option(1, "--page", help="Page of steps (25/page)"),
    raw: bool = typer.Option(False, "--raw", help="Full payloads, no truncation"),
) -> None:
    """Inspect one trajectory step by step."""
    root = Path.cwd()
    store = _store(root)
    t = _load_traj(store, ref)
    console.print(
        f"\n  [bold]{t.trajectory_id}[/]  {t.task_key}  [{t.kind}]  "
        f"status={t.status.value}  steps={len(t.steps)}",
        highlight=False,
    )
    if t.tokens_in is not None:
        lb = "≥" if t.totals_are_lower_bound else ""
        console.print(
            f"  tokens {lb}{t.tokens_in}→{t.tokens_out}  "
            f"cost {t.cost_usd if t.cost_usd is not None else 'n/a'}  "
            f"wall {t.wall_ms}ms",
            highlight=False,
        )
    per_page = 25
    start = (page - 1) * per_page
    window = t.steps[start : start + per_page]
    if not window and t.steps:
        _fail(f"page {page} is past the end ({len(t.steps)} steps, 25/page)")
    for s in window:
        args = json.dumps(s.args, default=str) if s.args is not None else ""
        if not raw and len(args) > 90:
            args = args[:89] + "…"
        console.print(f"  {s.idx:3}  {s.type.value:13} {s.name}({args})", highlight=False)
        if raw and s.result is not None:
            console.print(f"       → {json.dumps(s.result, default=str)}", highlight=False)
    total_pages = (len(t.steps) + per_page - 1) // per_page
    if total_pages > 1:
        console.print(f"\n  page {page}/{total_pages} — use --page N", highlight=False)


def diff(
    ref_a: str = typer.Argument(..., help="Baseline trajectory id (or prefix)"),
    ref_b: str = typer.Argument(..., help="Candidate trajectory id (or prefix)"),
    context: int = typer.Option(2, "--context", help="Steps shown around divergence"),
    full: bool = typer.Option(False, "--full", help="Show every aligned step"),
) -> None:
    """Trajectory diff between two stored runs."""
    root = Path.cwd()
    store = _store(root)
    a, b = _load_traj(store, ref_a), _load_traj(store, ref_b)
    match = best_variant_match([a], b)
    al = match.alignment

    if not al.is_divergent:
        console.print(
            f"\n  [green]✓ no divergence[/] — {len(al.ops)} steps aligned "
            f"(norm score {al.norm_score:.2f})",
            highlight=False,
        )
        raise typer.Exit(0)

    console.print(
        f"\n  [red]✗ diverged[/] at op {al.first_divergence} ({al.divergence_kind})",
        highlight=False,
    )
    fd = al.first_divergence or 0
    for i, op in enumerate(al.ops):
        near = abs(i - fd) <= context
        if not (full or near):
            continue
        sa = a.steps[op.a_idx] if op.a_idx is not None else None
        sb = b.steps[op.b_idx] if op.b_idx is not None else None
        if op.kind == "pair":
            marker = "=" if (op.sim or 0) >= 0.85 else "~"
            style = "dim" if marker == "=" else "yellow"
            assert sa is not None
            console.print(
                f"  [{style}]{marker}  {sa.idx:3} {sa.type.value:13} {sa.name}[/]",
                highlight=False,
            )
        elif op.kind == "missing_step":
            assert sa is not None
            console.print(
                f"  [red]-  {sa.idx:3} {sa.type.value:13} {sa.name}  (baseline only)[/]",
                highlight=False,
            )
        else:
            assert sb is not None
            console.print(
                f"  [red]+  {sb.idx:3} {sb.type.value:13} {sb.name}  (this run only)[/]",
                highlight=False,
            )
        if i == fd:
            console.print("     [red bold]^ first divergence[/]", highlight=False)
    raise typer.Exit(1)


@baseline_app.command("export")
def baseline_export(
    out: Path = typer.Option(Path("baselines"), "--out", help="Export directory"),
    with_payloads: bool = typer.Option(False, "--with-payloads", help="Keep truncated heads"),
) -> None:
    """Export all active baselines to committable JSON."""
    store = _store(Path.cwd())
    rows = store.conn.execute(
        "SELECT baseline_id, task_key FROM baselines WHERE active=1"
    ).fetchall()
    if not rows:
        _fail("no active baselines in .offtrack/offtrack.db — run `offtrack record` first")
    for r in rows:
        path = export_baseline(store, r["baseline_id"], out, with_payloads=with_payloads)
        console.print(f"  [green]✓[/] {r['task_key']} → {path}", highlight=False)


@baseline_app.command("import")
def baseline_import(
    path: Path = typer.Argument(..., help="Baseline JSON file or directory"),
) -> None:
    """Import baseline JSON (idempotent)."""
    store = _store(Path.cwd())
    if not path.exists():
        _fail(f"{path} does not exist — nothing was imported")
    if path.is_dir():
        ids = import_all(store, path)
        console.print(f"  [green]✓[/] imported {len(ids)} baseline(s)", highlight=False)
    else:
        bid = import_baseline(store, path)
        console.print(f"  [green]✓[/] imported {bid}", highlight=False)


@baseline_app.command("list")
def baseline_list() -> None:
    """List baselines with active flags."""
    store = _store(Path.cwd())
    rows = store.conn.execute(
        "SELECT baseline_id, task_key, label, created_at, active, "
        "(SELECT COUNT(*) FROM trajectories t WHERE t.baseline_id = b.baseline_id) AS n "
        "FROM baselines b ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    for r in rows:
        flag = "[green]active[/]" if r["active"] else "[dim]inactive[/]"
        console.print(
            f"  {r['baseline_id'][:12]}…  {r['task_key']}  [{r['label']}]  "
            f"{r['n']} recording(s)  {flag}",
            highlight=False,
        )


def doctor(
    repair: bool = typer.Option(False, "--repair", help="Rebuild a corrupted DB from baselines/"),
) -> None:
    """Validate the local setup: DB integrity, baselines, config."""
    root = Path.cwd()
    db_path = root / ".offtrack" / "offtrack.db"
    problems = 0

    config = root / "offtrack.yaml"
    if config.exists():
        try:
            from offtrack.suite import load_suite_file, resolve_tasks

            sf = load_suite_file(config)
            tasks = resolve_tasks(sf)
            console.print(
                f"  [green]✓[/] offtrack.yaml valid — {len(tasks)} task(s)", highlight=False
            )
        except Exception as e:
            problems += 1
            console.print(f"  [red]✗[/] offtrack.yaml: {e}", highlight=False)
    else:
        console.print("  [yellow]·[/] no offtrack.yaml (run `offtrack init`)", highlight=False)

    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            conn.close()
            if ok == "ok":
                console.print("  [green]✓[/] offtrack.db integrity ok", highlight=False)
            else:
                raise sqlite3.DatabaseError(ok)
        except sqlite3.DatabaseError as e:
            problems += 1
            console.print(
                f"  [red]✗[/] offtrack.db failed integrity check ({e}). "
                "Your committed baselines/ are unaffected.",
                highlight=False,
            )
            if repair:
                import time

                backup = db_path.with_name(f"offtrack.db.corrupt-{int(time.time())}")
                db_path.rename(backup)
                store = Store(db_path)
                bdir = root / "baselines"
                if bdir.exists():
                    ids = import_all(store, bdir)
                    console.print(
                        f"  [green]✓[/] rebuilt DB, re-imported {len(ids)} baseline(s); "
                        f"old file kept at {backup.name}",
                        highlight=False,
                    )
                else:
                    console.print(
                        f"  [green]✓[/] rebuilt empty DB; old file kept at {backup.name}",
                        highlight=False,
                    )
                problems -= 1
            else:
                console.print(
                    "      fix: offtrack doctor --repair  "
                    "(rebuilds from baselines/, keeps the old file)",
                    highlight=False,
                )
    else:
        console.print(
            "  [yellow]·[/] no local DB yet (created on first record/check)", highlight=False
        )

    bdir = root / "baselines"
    if bdir.exists():
        n = len(list(bdir.rglob("*.json")))
        console.print(f"  [green]✓[/] baselines/ present — {n} file(s)", highlight=False)
    else:
        console.print(
            "  [yellow]·[/] no baselines/ dir (created by `offtrack record`)", highlight=False
        )

    raise typer.Exit(EXIT_SETUP if problems else 0)
