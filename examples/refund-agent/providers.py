"""LLM providers for the refund-agent demo.

``FakeLLM`` is a *scripted LLM*, not a mocked test: it walks a persona script
(careful.json / sloppy.json) and emits tool calls / final answers through the
same agent loop a real provider would drive. The entire offtrack pipeline —
capture, ingest, alignment, stats — executes identically in fake and real mode.

Real mode: set OPENAI_API_KEY or ANTHROPIC_API_KEY and pass a real model name.
Fake mode:  model names "fake-careful" / "fake-sloppy" (no key, no network).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).parent / "fake_scripts"


@dataclass
class LLMResponse:
    """Uniform response shape for the agent loop."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    model: str = "unknown"
    tokens_in: int = 0
    tokens_out: int = 0


class FakeLLM:
    def __init__(self, persona: str):
        path = SCRIPTS_DIR / f"{persona}.json"
        self.script = json.loads(path.read_text())
        self.model = self.script["model_name"]

    def respond(self, user_input: str, history: list[dict[str, Any]], ctx: dict[str, Any]) -> LLMResponse:
        last_tool = next(
            (h["name"] for h in reversed(history) if h["role"] == "tool"),
            None,
        )
        last_result = next(
            (h["content"] for h in reversed(history) if h["role"] == "tool"),
            None,
        )
        for turn in self.script["turns"]:
            if "match_input_contains" in turn:
                if last_tool is None and turn["match_input_contains"] in user_input.lower():
                    return self._emit(turn, ctx)
                continue
            if turn.get("after_tool") != last_tool:
                continue
            want = turn.get("when_result_contains")
            if want and want not in json.dumps(last_result or {}):
                continue
            return self._emit(turn, ctx)
        return LLMResponse(final_answer="I'm not sure how to help with that.", model=self.model)

    def _emit(self, turn: dict[str, Any], ctx: dict[str, Any]) -> LLMResponse:
        subst = _Substituter(ctx)
        calls = [
            {"name": c["name"], "args": {k: subst(v) for k, v in c["args"].items()}}
            for c in turn.get("tool_calls", [])
        ]
        answer = turn.get("final_answer")
        return LLMResponse(
            tool_calls=calls,
            final_answer=subst(answer) if answer else None,
            model=self.model,
            tokens_in=turn.get("tokens_in", 0),
            tokens_out=turn.get("tokens_out", 0),
        )


class _Substituter:
    """Replace ${var} placeholders from the run context; numbers stay numbers."""

    def __init__(self, ctx: dict[str, Any]):
        self.ctx = ctx

    def __call__(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
            return self.ctx.get(value[2:-1], value)
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: str(self.ctx.get(m.group(1), m.group(0))),
            value,
        )


def get_llm(model: str) -> Any:
    if model.startswith("fake-"):
        return FakeLLM(model.removeprefix("fake-"))
    raise NotImplementedError(
        f"Real provider mode for {model!r} lands with the SDK shims; "
        "use fake-careful / fake-sloppy for now."
    )
