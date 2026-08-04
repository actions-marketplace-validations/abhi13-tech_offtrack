"""LangGraph callback adapter (real langchain-core BaseCallbackHandler)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from offtrack.integrations.langgraph import OfftrackCallbackHandler


@pytest.fixture(autouse=True)
def trace_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OFFTRACK_TRACE_DIR", str(tmp_path))
    return tmp_path


def read_events(tmp_path: Path):
    events = []
    for f in tmp_path.glob("langgraph-*.jsonl"):
        events += [json.loads(line) for line in f.read_text().splitlines()]
    return events


def llm_response(tool_calls=None, text="", model="gpt-5"):
    message = NS(
        tool_calls=tool_calls or [],
        content=text,
        response_metadata={"model_name": model},
    )
    return NS(
        llm_output={
            "model_name": model,
            "token_usage": {"prompt_tokens": 80, "completion_tokens": 12},
        },
        generations=[[NS(message=message)]],
    )


class TestHandler:
    def test_is_real_callback_handler(self):
        from langchain_core.callbacks import BaseCallbackHandler

        assert issubclass(OfftrackCallbackHandler, BaseCallbackHandler)

    def test_llm_flow(self, trace_dir):
        h = OfftrackCallbackHandler()
        h.on_chat_model_start({}, [], run_id="r1")
        h.on_llm_end(llm_response(tool_calls=[{"name": "search", "args": {}}]), run_id="r1")
        [ev] = read_events(trace_dir)
        assert ev["type"] == "llm_call"
        assert ev["args"]["tool_intent"] == ["search"]
        assert ev["usage"] == {"tokens_in": 80, "tokens_out": 12}
        assert ev["model"] == "gpt-5"

    def test_tool_flow_records_real_args_and_results(self, trace_dir):
        h = OfftrackCallbackHandler()
        h.on_tool_start({"name": "lookup_order"}, "", run_id="r2", inputs={"order_id": "T1"})
        h.on_tool_end(NS(content={"total": 842}), run_id="r2")
        [ev] = read_events(trace_dir)
        assert ev["type"] == "tool_call" and ev["name"] == "lookup_order"
        assert ev["args"] == {"order_id": "T1"}
        assert ev["result"] == {"total": 842}

    def test_tool_error(self, trace_dir):
        h = OfftrackCallbackHandler()
        h.on_tool_start({"name": "flaky"}, "", run_id="r3", inputs={})
        h.on_tool_error(RuntimeError("boom"), run_id="r3")
        [ev] = read_events(trace_dir)
        assert ev["status"] == "error" and "boom" in json.dumps(ev["result"])

    def test_finish_emits_final_and_end(self, trace_dir):
        h = OfftrackCallbackHandler()
        h.on_chat_model_start({}, [], run_id="r4")
        h.on_llm_end(llm_response(text="the answer"), run_id="r4")
        h.finish()
        events = read_events(trace_dir)
        assert events[-1] == {"ev": "end", "status": "complete"}
        final = next(e for e in events if e.get("type") == "final_answer")
        assert final["result"] == "the answer"

    def test_explicit_final_answer_wins(self, trace_dir):
        h = OfftrackCallbackHandler()
        h.finish(final_answer="explicit")
        final = next(e for e in read_events(trace_dir) if e.get("type") == "final_answer")
        assert final["result"] == "explicit"
