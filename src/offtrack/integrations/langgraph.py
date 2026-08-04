"""LangGraph / LangChain callback adapter.

    from offtrack.integrations.langgraph import OfftrackCallbackHandler
    graph.invoke(inputs, config={"callbacks": [OfftrackCallbackHandler()]})

Unlike the SDK shims, the callback sees REAL executed tool args and results
(not message-delta reconstruction). Requires langchain-core; the class is
defined lazily so importing this module without it gives an actionable error.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _Emitter:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        trace_dir = os.environ.get("OFFTRACK_TRACE_DIR")
        if not trace_dir:
            return
        path = Path(trace_dir) / f"langgraph-{os.getpid()}.jsonl"
        with self._lock, path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__raw__": str(value)}


def _build_handler_class() -> type:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as e:
        raise ImportError(
            "The LangGraph adapter needs langchain-core. "
            'Install it with: pip install "offtrack[langgraph]"'
        ) from e

    class OfftrackCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
        """Records llm_call / tool_call / final_answer capture events."""

        def __init__(self) -> None:
            self._emitter = _Emitter()
            self._starts: dict[str, str] = {}  # run_id → t0
            self._tool_meta: dict[str, dict[str, Any]] = {}
            self._last_text: str | None = None

        # -- LLM ------------------------------------------------------------

        def on_chat_model_start(
            self, serialized: dict[str, Any], messages: Any, *, run_id: Any, **kw: Any
        ) -> None:
            self._starts[str(run_id)] = _now()

        def on_llm_start(
            self, serialized: dict[str, Any], prompts: Any, *, run_id: Any, **kw: Any
        ) -> None:
            self._starts[str(run_id)] = _now()

        def on_llm_end(self, response: Any, *, run_id: Any, **kw: Any) -> None:
            t0 = self._starts.pop(str(run_id), _now())
            model = None
            tokens_in = tokens_out = None
            intents: list[str] = []
            text: str | None = None
            try:
                llm_output = getattr(response, "llm_output", None) or {}
                model = llm_output.get("model_name") or llm_output.get("model")
                usage = llm_output.get("token_usage") or {}
                tokens_in = usage.get("prompt_tokens")
                tokens_out = usage.get("completion_tokens")
                gen = response.generations[0][0]
                message = getattr(gen, "message", None)
                if message is not None:
                    model = model or (getattr(message, "response_metadata", {}) or {}).get(
                        "model_name"
                    )
                    for tc in getattr(message, "tool_calls", None) or []:
                        name = tc.get("name") if isinstance(tc, dict) else None
                        if name:
                            intents.append(name)
                    text = getattr(message, "content", None) or None
                else:
                    text = getattr(gen, "text", None) or None
            except (AttributeError, IndexError, KeyError, TypeError):
                pass
            if text and not intents:
                self._last_text = str(text)
            self._emitter.emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "llm_call",
                    "name": model or "llm",
                    "model": model,
                    "args": {"tool_intent": sorted(intents)},
                    "result": {"final": not intents},
                    "usage": {"tokens_in": tokens_in, "tokens_out": tokens_out},
                    "t0": t0,
                    "t1": _now(),
                    "status": "ok",
                }
            )

        def on_llm_error(self, error: BaseException, *, run_id: Any, **kw: Any) -> None:
            t0 = self._starts.pop(str(run_id), _now())
            self._emitter.emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "llm_call",
                    "name": "llm",
                    "result": {"error": str(error)[:500]},
                    "t0": t0,
                    "t1": _now(),
                    "status": "error",
                }
            )

        # -- Tools ----------------------------------------------------------

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            inputs: dict[str, Any] | None = None,
            **kw: Any,
        ) -> None:
            rid = str(run_id)
            self._starts[rid] = _now()
            self._tool_meta[rid] = {
                "name": (serialized or {}).get("name") or "tool",
                "args": _jsonable(inputs if inputs is not None else input_str),
            }

        def on_tool_end(self, output: Any, *, run_id: Any, **kw: Any) -> None:
            rid = str(run_id)
            t0 = self._starts.pop(rid, _now())
            meta = self._tool_meta.pop(rid, {"name": "tool", "args": None})
            content = getattr(output, "content", output)
            self._emitter.emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "tool_call",
                    "name": meta["name"],
                    "args": meta["args"],
                    "result": _jsonable(content),
                    "t0": t0,
                    "t1": _now(),
                    "status": "ok",
                }
            )

        def on_tool_error(self, error: BaseException, *, run_id: Any, **kw: Any) -> None:
            rid = str(run_id)
            t0 = self._starts.pop(rid, _now())
            meta = self._tool_meta.pop(rid, {"name": "tool", "args": None})
            self._emitter.emit(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "tool_call",
                    "name": meta["name"],
                    "args": meta["args"],
                    "result": {"error": str(error)[:500]},
                    "t0": t0,
                    "t1": _now(),
                    "status": "error",
                }
            )

        # -- Completion ------------------------------------------------------

        def finish(self, final_answer: str | None = None) -> None:
            """Emit the final answer + end event. Call after graph.invoke().

            If final_answer is omitted, the last text-only LLM response is used.
            """
            answer = final_answer if final_answer is not None else self._last_text
            if answer:
                self._emitter.emit(
                    {
                        "ev": "step",
                        "v": 1,
                        "type": "final_answer",
                        "name": "final",
                        "result": answer,
                        "t0": _now(),
                        "t1": _now(),
                        "status": "ok",
                    }
                )
            self._emitter.emit({"ev": "end", "status": "complete"})

    return OfftrackCallbackHandler


_handler_class: type | None = None


def __getattr__(name: str) -> Any:
    global _handler_class
    if name == "OfftrackCallbackHandler":
        if _handler_class is None:
            _handler_class = _build_handler_class()
        return _handler_class
    raise AttributeError(name)
