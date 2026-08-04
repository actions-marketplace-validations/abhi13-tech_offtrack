"""SDK shims: message-delta tool reconstruction, tested against SDK-shaped stubs.

The recorder functions are exercised directly with objects duck-typing the
OpenAI/Anthropic response shapes — no network, no SDK install needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from offtrack.capture import (
    _PENDING_TOOLS,
    _SEEN_RESULTS,
    _flush_tool_results,
    _now,
    _record_anthropic_response,
    _record_openai_response,
)
from offtrack.ingest import build_from_trace_dir
from offtrack.model import StepType, TrajStatus


@pytest.fixture(autouse=True)
def trace_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OFFTRACK_TRACE_DIR", str(tmp_path))
    _PENDING_TOOLS.clear()
    _SEEN_RESULTS.clear()
    return tmp_path


def read_events(tmp_path: Path):
    events = []
    for f in tmp_path.glob("shim-*.jsonl"):
        events += [json.loads(line) for line in f.read_text().splitlines()]
    return events


def openai_response(tool_calls=None, content=None, model="gpt-5"):
    return NS(
        id="resp-1",
        model=model,
        usage=NS(prompt_tokens=100, completion_tokens=20),
        choices=[NS(message=NS(tool_calls=tool_calls, content=content))],
    )


def openai_tool_call(call_id, name, args):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


class TestOpenAI:
    def test_llm_call_with_tool_intent(self, trace_dir):
        resp = openai_response(
            tool_calls=[openai_tool_call("c1", "lookup_order", {"order_id": "T1"})]
        )
        _record_openai_response(resp, "2026-08-05T10:00:00+00:00")
        [ev] = read_events(trace_dir)
        assert ev["type"] == "llm_call" and ev["model"] == "gpt-5"
        assert ev["args"]["tool_intent"] == ["lookup_order"]
        assert ev["usage"] == {"tokens_in": 100, "tokens_out": 20}
        assert "c1" in _PENDING_TOOLS

    def test_tool_result_pairing_by_delta(self, trace_dir):
        _record_openai_response(
            openai_response(tool_calls=[openai_tool_call("c1", "lookup", {"id": 4})]),
            "2026-08-05T10:00:00+00:00",
        )
        # Next request carries the tool result message.
        _flush_tool_results([{"role": "tool", "tool_call_id": "c1", "content": '{"total": 842}'}])
        events = read_events(trace_dir)
        tool = next(e for e in events if e["type"] == "tool_call")
        assert tool["name"] == "lookup"
        assert tool["args"] == {"id": 4}
        assert tool["result"] == {"total": 842}
        assert not _PENDING_TOOLS

    def test_duplicate_results_recorded_once(self, trace_dir):
        _record_openai_response(openai_response(tool_calls=[openai_tool_call("c1", "t", {})]), "t0")
        msg = [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]
        _flush_tool_results(msg)
        _flush_tool_results(msg)  # same history resent next turn
        events = [e for e in read_events(trace_dir) if e.get("type") == "tool_call"]
        assert len(events) == 1

    def test_final_answer_and_end(self, trace_dir):
        _record_openai_response(openai_response(content="all done"), "t0")
        events = read_events(trace_dir)
        assert [e.get("type", e["ev"]) for e in events] == [
            "llm_call",
            "final_answer",
            "end",
        ]

    def test_parallel_calls_share_group(self, trace_dir):
        resp = openai_response(
            tool_calls=[
                openai_tool_call("c1", "a", {}),
                openai_tool_call("c2", "b", {}),
            ]
        )
        _record_openai_response(resp, "t0")
        _flush_tool_results(
            [
                {"role": "tool", "tool_call_id": "c1", "content": "1"},
                {"role": "tool", "tool_call_id": "c2", "content": "2"},
            ]
        )
        tools = [e for e in read_events(trace_dir) if e.get("type") == "tool_call"]
        assert tools[0]["group"] == tools[1]["group"] is not None


class TestAnthropic:
    def anthropic_response(self, blocks, model="claude-fable-5"):
        return NS(
            id="msg-1",
            model=model,
            usage=NS(input_tokens=90, output_tokens=15),
            content=blocks,
        )

    def test_tool_use_flow(self, trace_dir):
        resp = self.anthropic_response(
            [NS(type="tool_use", id="tu1", name="search", input={"q": "x"})]
        )
        _record_anthropic_response(resp, "t0")
        _flush_tool_results(
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "hits"}],
                }
            ]
        )
        events = read_events(trace_dir)
        llm = next(e for e in events if e["type"] == "llm_call")
        tool = next(e for e in events if e["type"] == "tool_call")
        assert llm["model"] == "claude-fable-5"
        assert llm["usage"] == {"tokens_in": 90, "tokens_out": 15}
        assert tool["name"] == "search" and tool["args"] == {"q": "x"}

    def test_text_only_is_final(self, trace_dir):
        resp = self.anthropic_response([NS(type="text", text="the answer")])
        _record_anthropic_response(resp, "t0")
        events = read_events(trace_dir)
        assert events[-1] == {"ev": "end", "status": "complete"}
        final = next(e for e in events if e.get("type") == "final_answer")
        assert final["result"] == "the answer"


class TestEndToEndBuild:
    def test_shim_events_build_complete_trajectory(self, trace_dir):
        """A full simulated agent turn builds a COMPLETE trajectory."""
        _record_openai_response(
            openai_response(tool_calls=[openai_tool_call("c1", "lookup", {"id": 1})]),
            _now(),
        )
        _flush_tool_results([{"role": "tool", "tool_call_id": "c1", "content": "found"}])
        _record_openai_response(openai_response(content="done"), _now())

        result = build_from_trace_dir(trace_dir, "s/t", "candidate", 0)
        t = result.trajectory
        assert t.status == TrajStatus.COMPLETE
        assert [s.type for s in t.steps] == [
            StepType.LLM_CALL,
            StepType.TOOL_CALL,
            StepType.LLM_CALL,
            StepType.FINAL_ANSWER,
        ]
        assert t.source == "shim"
