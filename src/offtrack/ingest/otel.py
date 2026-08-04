"""OTel GenAI trace importer — the vendor-neutral ingest wedge.

Accepts OTLP/JSON (resourceSpans tree) and collector file-exporter JSONL
(sniffed). The GenAI semantic conventions are pre-stable with two attribute
generations in the wild, plus the OpenInference flavor; fields are coalesced
per-span (first non-null wins, newest generation first) so mixed traces
just work. `offtrack ingest --explain` prints every classification decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical field ← [newest semconv, old semconv, OpenInference]
COALESCE: dict[str, list[str]] = {
    "provider": ["gen_ai.provider.name", "gen_ai.system", "llm.provider"],
    "request_model": ["gen_ai.request.model", "llm.model_name"],
    "response_model": ["gen_ai.response.model"],
    "tokens_in": [
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.prompt_tokens",
        "llm.token_count.prompt",
    ],
    "tokens_out": [
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.completion_tokens",
        "llm.token_count.completion",
    ],
    "operation": ["gen_ai.operation.name"],
    "oi_kind": ["openinference.span.kind"],
    "tool_name": ["gen_ai.tool.name", "tool.name"],
    "tool_args": ["gen_ai.tool.call.arguments", "input.value"],
    "tool_result": ["gen_ai.tool.call.result", "output.value"],
    "agent_name": ["gen_ai.agent.name"],
}

CHAT_OPS = {"chat", "text_completion", "generate_content"}
AGENT_OPS = {"invoke_agent", "create_agent"}


class SpanDecision:
    """One span's classification, for --explain output."""

    def __init__(self, span_id: str, name: str):
        self.span_id = span_id
        self.name = name
        self.step_type: str | None = None
        self.rule: str = "unclassified"
        self.fields_from: dict[str, str] = {}

    def __repr__(self) -> str:
        out = f"{self.name} [{self.span_id[:8]}] → {self.step_type or 'SKIPPED'} ({self.rule})"
        if self.fields_from:
            src = ", ".join(f"{k}←{v}" for k, v in self.fields_from.items())
            out += f" | {src}"
        return out


def _attr_value(v: Any) -> Any:
    """Unwrap an OTLP AnyValue ({stringValue: ...} etc.) or pass through."""
    if isinstance(v, dict):
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if key in v:
                raw = v[key]
                if key == "intValue" and isinstance(raw, str):
                    return int(raw)
                return raw
        if "arrayValue" in v:
            return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    return v


def _attrs_of(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes")
    if isinstance(raw, dict):  # simplified/file-exporter form
        return {k: _attr_value(v) for k, v in raw.items()}
    out: dict[str, Any] = {}
    for item in raw or []:
        key = item.get("key")
        if key is not None:
            out[key] = _attr_value(item.get("value"))
    return out


def _coalesce(attrs: dict[str, Any], decision: SpanDecision) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for canon, sources in COALESCE.items():
        for src in sources:
            if src in attrs and attrs[src] is not None:
                fields[canon] = attrs[src]
                decision.fields_from[canon] = src
                break
    return fields


def _ns_to_iso(ns: Any) -> str | None:
    if ns in (None, "", 0):
        return None
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
    except (ValueError, OverflowError):
        return None


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return {"__raw__": value}
    return value


def classify_span(span: dict[str, Any]) -> tuple[dict[str, Any] | None, SpanDecision]:
    """Classify one span into a capture event (or None for structural spans)."""
    span_id = str(span.get("spanId") or span.get("span_id") or "")
    name = str(span.get("name") or "")
    decision = SpanDecision(span_id, name)
    attrs = _attrs_of(span)
    fields = _coalesce(attrs, decision)

    op = fields.get("operation")
    oi = str(fields.get("oi_kind") or "").upper()
    step_type: str | None = None
    step_name = name

    if op == "execute_tool" or fields.get("tool_name") or oi == "TOOL":
        step_type = "tool_call"
        step_name = str(fields.get("tool_name") or name.removeprefix("execute_tool "))
        decision.rule = (
            "operation=execute_tool"
            if op == "execute_tool"
            else "tool_name attr"
            if fields.get("tool_name")
            else "openinference TOOL"
        )
    elif op in CHAT_OPS or fields.get("request_model") or oi == "LLM":
        step_type = "llm_call"
        step_name = str(fields.get("response_model") or fields.get("request_model") or name)
        decision.rule = (
            f"operation={op}"
            if op in CHAT_OPS
            else "request_model attr"
            if fields.get("request_model")
            else "openinference LLM"
        )
    elif op in AGENT_OPS or oi == "AGENT":
        step_type = "handoff"
        step_name = str(fields.get("agent_name") or name)
        decision.rule = f"operation={op}" if op in AGENT_OPS else "openinference AGENT"
    else:
        decision.rule = f"no genai signals (op={op!r}, oi={oi or None!r})"

    decision.step_type = step_type
    if step_type is None:
        return None, decision

    status = "ok"
    status_obj = span.get("status") or {}
    if str(status_obj.get("code", "")).upper() in ("STATUS_CODE_ERROR", "ERROR", "2"):
        status = "error"

    event: dict[str, Any] = {
        "ev": "step",
        "v": 1,
        "type": step_type,
        "name": step_name,
        "status": status,
        "t0": _ns_to_iso(span.get("startTimeUnixNano") or span.get("start_time_unix_nano")),
        "t1": _ns_to_iso(span.get("endTimeUnixNano") or span.get("end_time_unix_nano")),
    }
    if step_type == "llm_call":
        event["model"] = fields.get("response_model") or fields.get("request_model")
        usage = {}
        if fields.get("tokens_in") is not None:
            usage["tokens_in"] = int(fields["tokens_in"])
        if fields.get("tokens_out") is not None:
            usage["tokens_out"] = int(fields["tokens_out"])
        if usage:
            event["usage"] = usage
    if step_type == "tool_call":
        if "tool_args" in fields:
            event["args"] = _maybe_json(fields["tool_args"])
        if "tool_result" in fields:
            event["result"] = _maybe_json(fields["tool_result"])
    return event, decision


def _iter_spans(doc: Any) -> list[dict[str, Any]]:
    """Yield spans from OTLP/JSON tree or a bare span list."""
    spans: list[dict[str, Any]] = []
    if isinstance(doc, dict) and "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            for ss in rs.get("scopeSpans", rs.get("instrumentationLibrarySpans", [])):
                spans.extend(ss.get("spans", []))
    elif isinstance(doc, dict) and ("spanId" in doc or "span_id" in doc):
        spans.append(doc)
    elif isinstance(doc, list):
        for item in doc:
            spans.extend(_iter_spans(item))
    return spans


def read_otel_file(path: Path) -> tuple[list[dict[str, Any]], list[SpanDecision], list[str]]:
    """Read OTLP/JSON or collector JSONL; returns (events, decisions, warnings)."""
    warnings: list[str] = []
    text = path.read_text()
    spans: list[dict[str, Any]] = []
    stripped = text.lstrip()
    if stripped.startswith("{") and '"resourceSpans"' in text.split("\n", 1)[0]:
        # Could still be JSONL of OTLP export lines — try whole-file JSON first.
        try:
            spans = _iter_spans(json.loads(text))
        except json.JSONDecodeError:
            spans = []
    if not spans:
        for i, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                spans.extend(_iter_spans(json.loads(line)))
            except json.JSONDecodeError:
                if i == len(text.splitlines()) - 1:
                    warnings.append(f"{path.name}: torn final line dropped")
                else:
                    warnings.append(f"{path.name}: unparseable line {i + 1} dropped")

    # Dedup by span id (re-exports), keep last.
    by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for s in spans:
        sid = str(s.get("spanId") or s.get("span_id") or "")
        if sid:
            by_id[sid] = s
        else:
            anonymous.append(s)
    unique = list(by_id.values()) + anonymous

    # Group by trace: pick the largest trace if several (warn).
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in unique:
        tid = str(s.get("traceId") or s.get("trace_id") or "?")
        by_trace.setdefault(tid, []).append(s)
    if len(by_trace) > 1:
        sizes = {t: len(v) for t, v in by_trace.items()}
        chosen = max(sizes, key=lambda t: sizes[t])
        warnings.append(
            f"{path.name}: {len(by_trace)} traces found; selected the largest "
            f"({sizes[chosen]} spans). Pin one trace per attempt to silence."
        )
        unique = by_trace[chosen]

    events: list[dict[str, Any]] = []
    decisions: list[SpanDecision] = []
    skipped = 0
    for span in unique:
        event, decision = classify_span(span)
        decisions.append(decision)
        if event is None:
            skipped += 1
        else:
            events.append(event)
    if skipped:
        warnings.append(
            f"{path.name}: {skipped} of {len(unique)} spans not classifiable as steps "
            "— run `offtrack ingest --explain` to see per-span decisions"
        )
    if events:
        events.append({"ev": "end", "status": "complete"})
    return events, decisions, warnings
