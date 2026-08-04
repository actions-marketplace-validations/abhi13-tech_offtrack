"""Claude Code session ingest: session JSONL → capture events.

    offtrack ingest claude-code ~/.claude/projects/<proj>/<session>.jsonl

Assistant `tool_use` blocks become tool_call steps (paired with the user
`tool_result` that follows); assistant turns become llm_call steps; the last
text-only assistant message becomes the final_answer. Task-subagent
sidechains collapse to a single `handoff` step so sequences stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def read_claude_code_session(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse one Claude Code session JSONL into capture events."""
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    pending_tools: dict[str, dict[str, Any]] = {}
    last_text: str | None = None
    last_ts: str | None = None

    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                warnings.append(f"{path.name}: torn final line dropped")
            continue
        if not isinstance(record, dict):
            continue
        # Sidechain records belong to subagents — collapsed via the Task tool result.
        if record.get("isSidechain"):
            continue
        rtype = record.get("type")
        message = record.get("message") or {}
        ts = record.get("timestamp")
        last_ts = ts or last_ts

        if rtype == "assistant":
            blocks = _blocks(message)
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            model = message.get("model")
            usage = message.get("usage") or {}
            events.append(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "llm_call",
                    "name": model or "claude",
                    "model": model,
                    "args": {"tool_intent": sorted(b.get("name", "") for b in tool_uses)},
                    "result": {"final": not tool_uses},
                    "usage": {
                        "tokens_in": usage.get("input_tokens"),
                        "tokens_out": usage.get("output_tokens"),
                    },
                    "t0": ts,
                    "t1": ts,
                    "status": "ok",
                }
            )
            group = f"g{i}" if len(tool_uses) > 1 else None
            for block in tool_uses:
                pending_tools[str(block.get("id"))] = {
                    "name": block.get("name", "tool"),
                    "args": block.get("input"),
                    "t0": ts,
                    "group": group,
                }
            if texts and not tool_uses:
                last_text = "\n".join(t for t in texts if t)

        elif rtype == "user":
            for block in _blocks(message):
                if block.get("type") != "tool_result":
                    continue
                call_id = str(block.get("tool_use_id"))
                pending = pending_tools.pop(call_id, None)
                if pending is None:
                    warnings.append(
                        f"{path.name}: tool_result {call_id[:12]} without a "
                        "matching tool_use — skipped"
                    )
                    continue
                name = pending["name"]
                step_type = "handoff" if name == "Task" else "tool_call"
                events.append(
                    {
                        "ev": "step",
                        "v": 1,
                        "type": step_type,
                        "name": (
                            (pending["args"] or {}).get("subagent_type", "subagent")
                            if step_type == "handoff"
                            else name
                        ),
                        "args": pending["args"],
                        "result": _result_content(block),
                        "t0": pending["t0"],
                        "t1": ts,
                        "group": pending.get("group"),
                        "status": "error" if block.get("is_error") else "ok",
                    }
                )

    if pending_tools:
        warnings.append(
            f"{path.name}: {len(pending_tools)} tool call(s) never got results "
            "(session may have been interrupted)"
        )
    if last_text:
        events.append(
            {
                "ev": "step",
                "v": 1,
                "type": "final_answer",
                "name": "final",
                "result": last_text,
                "t0": last_ts,
                "t1": last_ts,
                "status": "ok",
            }
        )
        events.append({"ev": "end", "status": "complete"})
    elif events:
        events.append({"ev": "end", "status": "partial"})
        warnings.append(f"{path.name}: session has no final text answer — partial")
    return events, warnings


def _result_content(block: dict[str, Any]) -> Any:
    content = block.get("content")
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        joined = "\n".join(t for t in texts if t)
        return joined or content
    return content
