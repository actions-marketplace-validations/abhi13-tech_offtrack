"""Zero-code capture shims for the OpenAI and Anthropic Python SDKs.

    import offtrack.capture
    offtrack.capture.install()          # patches whichever SDKs are importable

Every chat/messages call is recorded as an llm_call step. Tool steps are
reconstructed by MESSAGE DELTA: tool calls emitted in response N are paired
(by tool_call_id / tool_use_id) with the tool-result messages newly present
in request N+1 — so the shim sees real args and results without touching the
agent's code. Events flush to $OFFTRACK_TRACE_DIR/shim-<pid>.jsonl per call
(crash-safe: whatever was flushed survives SIGKILL).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_INSTALLED: set[str] = set()

# tool_call_id → {name, args, t} emitted by an earlier assistant response,
# awaiting a result message in a later request.
_PENDING_TOOLS: dict[str, dict[str, Any]] = {}
_SEEN_RESULTS: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _emit(event: dict[str, Any]) -> None:
    trace_dir = os.environ.get("OFFTRACK_TRACE_DIR")
    if not trace_dir:
        return
    path = Path(trace_dir) / f"shim-{os.getpid()}.jsonl"
    line = json.dumps(event, default=str) + "\n"
    with _LOCK, path.open("a") as f:
        f.write(line)  # single write per event: crash-safe


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {"__raw__": value}
    return value


def _flush_tool_results(messages: list[Any]) -> None:
    """Pair tool-result messages in this request with pending tool calls."""
    for m in messages or []:
        role = _get(m, "role")
        # OpenAI: {"role": "tool", "tool_call_id": ..., "content": ...}
        if role == "tool":
            call_id = _get(m, "tool_call_id")
            if call_id and call_id in _PENDING_TOOLS and call_id not in _SEEN_RESULTS:
                _SEEN_RESULTS.add(call_id)
                pending = _PENDING_TOOLS.pop(call_id)
                _emit(
                    {
                        "ev": "step",
                        "v": 1,
                        "type": "tool_call",
                        "name": pending["name"],
                        "args": pending["args"],
                        "result": _maybe_json(_get(m, "content")),
                        "t0": pending["t"],
                        "t1": _now(),
                        "group": pending.get("group"),
                        "status": "ok",
                    }
                )
        # Anthropic: {"role": "user", "content": [{"type": "tool_result", ...}]}
        elif role == "user":
            content = _get(m, "content")
            if isinstance(content, list):
                for block in content:
                    if _get(block, "type") == "tool_result":
                        call_id = _get(block, "tool_use_id")
                        if call_id and call_id in _PENDING_TOOLS and call_id not in _SEEN_RESULTS:
                            _SEEN_RESULTS.add(call_id)
                            pending = _PENDING_TOOLS.pop(call_id)
                            _emit(
                                {
                                    "ev": "step",
                                    "v": 1,
                                    "type": "tool_call",
                                    "name": pending["name"],
                                    "args": pending["args"],
                                    "result": _maybe_json(_get(block, "content")),
                                    "t0": pending["t"],
                                    "t1": _now(),
                                    "group": pending.get("group"),
                                    "status": "ok",
                                }
                            )


def _get(obj: Any, attr: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _record_openai_response(response: Any, t0: str) -> None:
    model = _get(response, "model") or "openai"
    usage = _get(response, "usage")
    choices = _get(response, "choices") or []
    message = _get(choices[0], "message") if choices else None
    tool_calls = _get(message, "tool_calls") or []

    group = f"g-{_get(response, 'id')}" if len(tool_calls) > 1 else None
    intents = []
    for tc in tool_calls:
        fn = _get(tc, "function")
        name = _get(fn, "name") or "tool"
        intents.append(name)
        _PENDING_TOOLS[_get(tc, "id")] = {
            "name": name,
            "args": _maybe_json(_get(fn, "arguments")),
            "t": _now(),
            "group": group,
        }

    _emit(
        {
            "ev": "step",
            "v": 1,
            "type": "llm_call",
            "name": model,
            "model": model,
            "args": {"tool_intent": intents},
            "result": {"final": not tool_calls},
            "usage": {
                "tokens_in": _get(usage, "prompt_tokens"),
                "tokens_out": _get(usage, "completion_tokens"),
            },
            "t0": t0,
            "t1": _now(),
            "status": "ok",
        }
    )
    if not tool_calls:
        content = _get(message, "content")
        if content:
            _emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "final_answer",
                    "name": "final",
                    "result": content,
                    "t0": _now(),
                    "t1": _now(),
                    "status": "ok",
                }
            )
        _emit({"ev": "end", "status": "complete"})


def _record_anthropic_response(response: Any, t0: str) -> None:
    model = _get(response, "model") or "anthropic"
    usage = _get(response, "usage")
    content = _get(response, "content") or []

    tool_uses = [b for b in content if _get(b, "type") == "tool_use"]
    texts = [b for b in content if _get(b, "type") == "text"]
    group = f"g-{_get(response, 'id')}" if len(tool_uses) > 1 else None

    intents = []
    for block in tool_uses:
        name = _get(block, "name") or "tool"
        intents.append(name)
        _PENDING_TOOLS[_get(block, "id")] = {
            "name": name,
            "args": _get(block, "input"),
            "t": _now(),
            "group": group,
        }

    _emit(
        {
            "ev": "step",
            "v": 1,
            "type": "llm_call",
            "name": model,
            "model": model,
            "args": {"tool_intent": intents},
            "result": {"final": not tool_uses},
            "usage": {
                "tokens_in": _get(usage, "input_tokens"),
                "tokens_out": _get(usage, "output_tokens"),
            },
            "t0": t0,
            "t1": _now(),
            "status": "ok",
        }
    )
    if not tool_uses:
        final_text = "".join(str(_get(b, "text") or "") for b in texts)
        if final_text:
            _emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "final_answer",
                    "name": "final",
                    "result": final_text,
                    "t0": _now(),
                    "t1": _now(),
                    "status": "ok",
                }
            )
        _emit({"ev": "end", "status": "complete"})


def install_openai() -> bool:
    """Patch openai's chat.completions.create (sync). Returns True if patched."""
    if "openai" in _INSTALLED:
        return True
    try:
        from openai.resources.chat import completions as mod
    except ImportError:
        return False

    original = mod.Completions.create

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        _flush_tool_results(kwargs.get("messages") or [])
        t0 = _now()
        response = original(self, *args, **kwargs)
        with contextlib.suppress(Exception):  # capture must never break the agent
            _record_openai_response(response, t0)
        return response

    mod.Completions.create = create  # type: ignore[method-assign]
    _INSTALLED.add("openai")
    return True


def install_anthropic() -> bool:
    """Patch anthropic's messages.create (sync). Returns True if patched."""
    if "anthropic" in _INSTALLED:
        return True
    try:
        from anthropic.resources import messages as mod
    except ImportError:
        return False

    original = mod.Messages.create

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        _flush_tool_results(kwargs.get("messages") or [])
        t0 = _now()
        response = original(self, *args, **kwargs)
        with contextlib.suppress(Exception):  # capture must never break the agent
            _record_anthropic_response(response, t0)
        return response

    mod.Messages.create = create  # type: ignore[method-assign]
    _INSTALLED.add("anthropic")
    return True


def install() -> list[str]:
    """Patch every importable provider SDK; returns the list patched."""
    patched = []
    if install_openai():
        patched.append("openai")
    if install_anthropic():
        patched.append("anthropic")
    return patched
