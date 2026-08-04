"""Markdown rendering for PR comments and $GITHUB_STEP_SUMMARY."""

from __future__ import annotations

import json
from typing import Any

MARKER = "<!-- offtrack-report -->"
MAX_BYTES = 60_000

EMOJI = {"PASS": "✅", "FAIL": "❌", "INCONCLUSIVE": "🟡", "ERROR": "🟣"}


def _fmt_args(args: Any, limit: int = 60) -> str:
    if args is None:
        return ""
    s = json.dumps(args, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _first_div_cell(task: dict[str, Any]) -> str:
    div = task.get("first_divergence")
    if not div:
        return "—"
    base, cand = div.get("baseline_step"), div.get("candidate_step")
    kind = div.get("kind")
    where = base or cand
    step_no = where["idx"] if where else "?"
    if kind == "missing_step" and base:
        return f"**step {step_no}** — expected `{base['name']}`, not called"
    if kind == "extra_step" and cand:
        return f"**step {step_no}** — extra `{cand['name']}`"
    if base and cand:
        return f"**step {step_no}** — expected `{base['name']}`, got `{cand['name']}`"
    return f"step {step_no}"


def _metric_cell(task: dict[str, Any], name: str) -> str:
    for m in task["metrics"]:
        if m["name"] == name and m["change"] is not None:
            return f"{m['change']:+.0%}"
    return "—"


def render_markdown(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    counts: dict[str, int] = {}
    for t in report["tasks"]:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    n_bad = counts.get("FAIL", 0)
    total = len(report["tasks"])

    lines = [
        MARKER,
        f"## offtrack · {EMOJI[verdict]} {verdict}"
        + (f" — {n_bad}/{total} task(s) diverged from baseline" if n_bad else ""),
        "",
        "| Task | Verdict | First divergence | Δcost | Δtokens | Δlatency |",
        "|---|---|---|---|---|---|",
    ]
    for t in report["tasks"]:
        lines.append(
            f"| {t['task_key']} | {EMOJI[t['verdict']]} {t['verdict'].lower()} "
            f"| {_first_div_cell(t)} | {_metric_cell(t, 'cost')} "
            f"| {_metric_cell(t, 'tokens')} | {_metric_cell(t, 'latency')} |"
        )
    lines.append("")

    shown = 0
    for t in report["tasks"]:
        div = t.get("first_divergence")
        if not div or t["verdict"] == "PASS":
            continue
        lines.append(f"<details><summary><b>{t['task_key']}</b> — divergence detail</summary>")
        lines.append("")
        lines.append("```")
        for ctx in div.get("context_before") or []:
            if ctx:
                lines.append(
                    f"step {ctx['idx']}  {ctx['type']:12} "
                    f"{ctx['name']}({_fmt_args(ctx['args'], 45)})   (both)"
                )
        base, cand = div.get("baseline_step"), div.get("candidate_step")
        if base:
            lines.append(f"baseline: {base['name']}({_fmt_args(base['args'])})")
        if cand:
            lines.append(
                f"this run: {cand['name']}({_fmt_args(cand['args'])})   ← first divergence"
            )
        if not cand and div.get("kind") == "missing_step":
            lines.append("this run: (step not taken)                ← first divergence")
        lines.append("```")
        if t["behavioral"]["reason"]:
            lines.append(f"\n{t['behavioral']['reason']}")
        lines.append("</details>")
        lines.append("")
        shown += 1

    truncated_note = None
    body = "\n".join(lines)
    if len(body.encode()) > MAX_BYTES:
        while len(body.encode()) > MAX_BYTES and lines:
            lines = lines[:-1]
            body = "\n".join(lines)
        truncated_note = "\n\n_…report truncated — run `offtrack check` locally for the full diff._"
        body += truncated_note

    body += f"\n\n_Reproduce locally: `offtrack check`_ · run `{report['run_id']}`"
    return body
