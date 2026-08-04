"""Claude Code session reader."""

from __future__ import annotations

import json
from pathlib import Path

from offtrack.integrations.claude_code import read_claude_code_session


def write_session(tmp_path: Path, records) -> Path:
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def assistant(blocks, model="claude-fable-5", usage=None):
    return {
        "type": "assistant",
        "timestamp": "2026-08-05T10:00:00.000Z",
        "message": {
            "model": model,
            "usage": usage or {"input_tokens": 50, "output_tokens": 9},
            "content": blocks,
        },
    }


def tool_result(call_id, content, is_error=False):
    return {
        "type": "user",
        "timestamp": "2026-08-05T10:00:01.000Z",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }


class TestReader:
    def test_full_session(self, tmp_path):
        records = [
            assistant(
                [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]
            ),
            tool_result("t1", "file.txt"),
            assistant([{"type": "text", "text": "Done, one file."}]),
        ]
        events, warnings = read_claude_code_session(write_session(tmp_path, records))
        types = [e.get("type", e["ev"]) for e in events]
        assert types == ["llm_call", "tool_call", "llm_call", "final_answer", "end"]
        tool = events[1]
        assert tool["name"] == "Bash" and tool["args"] == {"command": "ls"}
        assert tool["result"] == "file.txt"
        assert not warnings

    def test_task_subagent_becomes_handoff(self, tmp_path):
        records = [
            assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Task",
                        "input": {"subagent_type": "Explore", "prompt": "look around"},
                    }
                ]
            ),
            {
                "type": "assistant",
                "isSidechain": True,
                "timestamp": "x",
                "message": {"content": [{"type": "text", "text": "subagent internals"}]},
            },
            tool_result("t1", "explored: 3 files"),
            assistant([{"type": "text", "text": "done"}]),
        ]
        events, _ = read_claude_code_session(write_session(tmp_path, records))
        handoff = next(e for e in events if e.get("type") == "handoff")
        assert handoff["name"] == "Explore"
        assert handoff["result"] == "explored: 3 files"
        # Sidechain records never became steps.
        assert all("subagent internals" not in json.dumps(e) for e in events)

    def test_orphan_result_warns(self, tmp_path):
        _events, warnings = read_claude_code_session(
            write_session(
                tmp_path, [tool_result("ghost", "x"), assistant([{"type": "text", "text": "hi"}])]
            )
        )
        assert any("without a matching tool_use" in w for w in warnings)

    def test_interrupted_session_partial(self, tmp_path):
        records = [
            assistant(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "sleep 999"},
                    }
                ]
            ),
        ]
        events, warnings = read_claude_code_session(write_session(tmp_path, records))
        assert events[-1] == {"ev": "end", "status": "partial"}
        assert any("never got results" in w for w in warnings)

    def test_error_tool_result(self, tmp_path):
        records = [
            assistant(
                [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "bad"}}]
            ),
            tool_result("t1", "command not found", is_error=True),
            assistant([{"type": "text", "text": "that failed"}]),
        ]
        events, _ = read_claude_code_session(write_session(tmp_path, records))
        tool = next(e for e in events if e.get("type") == "tool_call")
        assert tool["status"] == "error"
