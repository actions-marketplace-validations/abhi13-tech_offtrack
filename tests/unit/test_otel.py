"""OTel GenAI importer: dual semconv generations, OpenInference, OTLP shapes."""

from __future__ import annotations

import json
from pathlib import Path

from offtrack.ingest.otel import classify_span, read_otel_file


def otlp_span(name: str, attrs: dict, span_id="a1b2c3", trace_id="t1", **kw):
    return {
        "spanId": span_id,
        "traceId": trace_id,
        "name": name,
        "startTimeUnixNano": str(1_754_300_000_000_000_000),
        "endTimeUnixNano": str(1_754_300_001_000_000_000),
        "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in attrs.items()],
        **kw,
    }


class TestClassification:
    def test_new_semconv_llm(self):
        ev, d = classify_span(
            otlp_span(
                "chat gpt-5",
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "gpt-5",
                    "gen_ai.provider.name": "openai",
                },
            )
        )
        assert ev is not None and ev["type"] == "llm_call" and ev["model"] == "gpt-5"
        assert d.fields_from["provider"] == "gen_ai.provider.name"

    def test_old_semconv_llm(self):
        ev, d = classify_span(
            otlp_span("chat", {"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"})
        )
        assert ev is not None and ev["type"] == "llm_call"
        assert d.fields_from["provider"] == "gen_ai.system"

    def test_mixed_generations_new_wins(self):
        _ev, d = classify_span(
            otlp_span(
                "chat",
                {
                    "gen_ai.system": "old-provider",
                    "gen_ai.provider.name": "new-provider",
                    "gen_ai.request.model": "m",
                },
            )
        )
        assert d.fields_from["provider"] == "gen_ai.provider.name"

    def test_execute_tool(self):
        ev, _ = classify_span(
            otlp_span(
                "execute_tool lookup",
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.arguments": '{"id": 4}',
                },
            )
        )
        assert ev is not None and ev["type"] == "tool_call"
        assert ev["name"] == "lookup" and ev["args"] == {"id": 4}

    def test_openinference_flavor(self):
        ev, _ = classify_span(
            otlp_span(
                "my_llm",
                {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "claude-x",
                    "llm.token_count.prompt": "120",
                },
            )
        )
        assert ev is not None and ev["type"] == "llm_call"
        assert ev["usage"]["tokens_in"] == 120

    def test_openinference_tool(self):
        ev, _ = classify_span(
            otlp_span(
                "t",
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "search",
                    "input.value": "plain string arg",
                },
            )
        )
        assert ev is not None and ev["type"] == "tool_call"
        assert ev["args"] == {"__raw__": "plain string arg"}

    def test_agent_handoff(self):
        ev, _ = classify_span(
            otlp_span(
                "invoke_agent researcher",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "researcher"},
            )
        )
        assert ev is not None and ev["type"] == "handoff" and ev["name"] == "researcher"

    def test_structural_span_skipped_with_reason(self):
        ev, d = classify_span(otlp_span("internal.chain", {"some.attr": "x"}))
        assert ev is None
        assert "no genai signals" in d.rule

    def test_error_status(self):
        ev, _ = classify_span(
            otlp_span("chat", {"gen_ai.request.model": "m"}, status={"code": "STATUS_CODE_ERROR"})
        )
        assert ev is not None and ev["status"] == "error"

    def test_timestamps_converted(self):
        ev, _ = classify_span(otlp_span("chat", {"gen_ai.request.model": "m"}))
        assert ev is not None and ev["t0"] == "2025-08-04T09:33:20.000+00:00"


class TestFileReading:
    def test_otlp_json_tree(self, tmp_path: Path):
        doc = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="s1"),
                                otlp_span(
                                    "execute_tool x", {"gen_ai.tool.name": "x"}, span_id="s2"
                                ),
                            ]
                        }
                    ]
                }
            ]
        }
        p = tmp_path / "trace.json"
        p.write_text(json.dumps(doc))
        events, _decisions, _warnings = read_otel_file(p)
        assert [e["type"] for e in events if e["ev"] == "step"] == ["llm_call", "tool_call"]
        assert events[-1] == {"ev": "end", "status": "complete"}

    def test_jsonl_lines(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        lines = [
            json.dumps(otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="s1")),
            json.dumps(otlp_span("execute_tool y", {"gen_ai.tool.name": "y"}, span_id="s2")),
        ]
        p.write_text("\n".join(lines))
        events, _, _ = read_otel_file(p)
        assert len([e for e in events if e["ev"] == "step"]) == 2

    def test_duplicate_span_ids_deduped(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        s = otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="dup")
        p.write_text(json.dumps(s) + "\n" + json.dumps(s))
        events, _, _ = read_otel_file(p)
        assert len([e for e in events if e["ev"] == "step"]) == 1

    def test_multiple_traces_largest_selected_with_warning(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        lines = [
            json.dumps(
                otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="a", trace_id="t1")
            ),
            json.dumps(
                otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="b", trace_id="t2")
            ),
            json.dumps(
                otlp_span("execute_tool z", {"gen_ai.tool.name": "z"}, span_id="c", trace_id="t2")
            ),
        ]
        p.write_text("\n".join(lines))
        events, _, warnings = read_otel_file(p)
        assert len([e for e in events if e["ev"] == "step"]) == 2  # t2 wins
        assert any("traces found" in w for w in warnings)

    def test_unclassifiable_spans_warned(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        lines = [
            json.dumps(otlp_span("chain.internal", {"x": "y"}, span_id="a")),
            json.dumps(otlp_span("chat", {"gen_ai.request.model": "m"}, span_id="b")),
        ]
        p.write_text("\n".join(lines))
        _events, _decisions, warnings = read_otel_file(p)
        assert any("not classifiable" in w for w in warnings)
        assert any("--explain" in w for w in warnings)

    def test_torn_final_line(self, tmp_path: Path):
        p = tmp_path / "spans.jsonl"
        p.write_text(json.dumps(otlp_span("chat", {"gen_ai.request.model": "m"})) + '\n{"broken')
        events, _, warnings = read_otel_file(p)
        assert len([e for e in events if e["ev"] == "step"]) == 1
        assert any("torn final line" in w for w in warnings)
