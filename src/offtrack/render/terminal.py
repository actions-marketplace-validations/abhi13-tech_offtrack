"""Terminal rendering of the verdict document (rich)."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.text import Text

BADGE = {
    "PASS": ("✓", "green"),
    "FAIL": ("✗", "red"),
    "INCONCLUSIVE": ("~", "yellow"),
    "ERROR": ("!", "magenta"),
}


def _fmt_args(args: Any, limit: int = 70) -> str:
    if args is None:
        return ""
    s = json.dumps(args, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def render_terminal(report: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    for task in report["tasks"]:
        _render_task(task, console)
    verdict = report["verdict"]
    _, color = BADGE[verdict]
    counts: dict[str, int] = {}
    for t in report["tasks"]:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    summary = ", ".join(f"{v} {k.lower()}" for k, v in sorted(counts.items()))
    console.print(f"\n  [{color} bold]{verdict}[/] — {summary}", highlight=False)


def _render_task(task: dict[str, Any], console: Console) -> None:
    verdict = task["verdict"]
    symbol, color = BADGE[verdict]
    b = task["behavioral"]
    rate = ""
    if b["rate_baseline"] is not None and b["rate_candidate"] is not None:
        clean = b["n_candidate"] - b["k_candidate"]
        rate = f"{clean}/{b['n_candidate']} aligned"
    console.print(f"  [{color}]{symbol}[/] [bold]{task['task_key']}[/]  {rate}", highlight=False)

    div = task.get("first_divergence")
    if div and verdict in ("FAIL", "INCONCLUSIVE"):
        _render_divergence(div, console)

    if verdict != "PASS" and b["reason"]:
        console.print(f"      [dim]{b['reason']}[/]", highlight=False)
    if b.get("prescription"):
        console.print(
            f"      [yellow]→ run {b['prescription']} more repetition(s): "
            f"offtrack check --runs {b['n_candidate'] + b['prescription']}[/]",
            highlight=False,
        )

    deltas = []
    for m in task["metrics"]:
        if m["change"] is not None:
            mark = {"FAIL": "[red]", "WARN": "[yellow]"}.get(m["verdict"], "[dim]")
            deltas.append(f"{mark}Δ{m['name']} {m['change']:+.0%}[/]")
    if deltas:
        console.print("      " + "  ".join(deltas), highlight=False)

    for m in task["metrics"]:
        if m["verdict"] in ("FAIL", "WARN"):
            console.print(f"      [dim]{m['name']}: {m['reason']}[/]", highlight=False)

    if task["warnings"]:
        console.print("      [yellow]caveats:[/]", highlight=False)
        seen = set()
        for w in task["warnings"]:
            if w not in seen:
                seen.add(w)
                console.print(f"        [yellow]·[/] [dim]{w}[/]", highlight=False)


def _render_divergence(div: dict[str, Any], console: Console) -> None:
    for ctx in div.get("context_before") or []:
        if ctx:
            console.print(
                f"      [dim]=  step {ctx['idx']}  {ctx['type']:12} "
                f"{ctx['name']}({_fmt_args(ctx['args'], 48)})[/]",
                highlight=False,
            )
    base, cand = div.get("baseline_step"), div.get("candidate_step")
    kind = div.get("kind")
    line = Text("      ▲ first divergence — ", style="red bold")
    if kind == "missing_step" and base:
        line.append(f"missing step: expected {base['name']}({_fmt_args(base['args'], 40)})")
    elif kind == "extra_step" and cand:
        line.append(f"extra step: {cand['name']}({_fmt_args(cand['args'], 40)})")
    elif base and cand:
        line.append(f"expected {base['name']}({_fmt_args(base['args'], 30)}), ")
        line.append(f"got {cand['name']}({_fmt_args(cand['args'], 30)})")
    console.print(line)
    if base and kind == "changed_step":
        console.print(
            f"        [dim]baseline[/]  {base['name']}({_fmt_args(base['args'])})",
            highlight=False,
        )
    if cand and kind == "changed_step":
        console.print(
            f"        [red]this run[/]  {cand['name']}({_fmt_args(cand['args'])})",
            highlight=False,
        )
    v = div.get("variant") or {}
    if v.get("count", 1) > 1:
        console.print(
            f"        [dim]closest baseline variant {v['index'] + 1}/{v['count']} "
            f"(seen {v['seen']}× during recording)[/]",
            highlight=False,
        )
    if div.get("resynced"):
        console.print("        [dim]trajectories resynced after the divergence[/]", highlight=False)
