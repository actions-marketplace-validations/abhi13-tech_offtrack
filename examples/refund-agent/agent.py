"""The refund-agent demo: a small tool-using support agent.

Policy (POLICY.md in prose): refunds ≤ $500 may be auto-approved; anything
larger must go through check_refund_policy and then escalate. The careful
persona follows this; the sloppy persona (a stand-in for a cheaper model)
skips the policy check and refunds directly — the marquee regression.

Run:  python agent.py --task refund --order TEST-1 --model fake-careful
Env:  OFFTRACK_TASK_INPUT overrides --order/--task (JSON), as injected by
      the offtrack runner. Emits capture events to OFFTRACK_TRACE_DIR when
      offtrack's shims are installed; also runs standalone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from providers import get_llm  # noqa: E402
from tools import TOOLS  # noqa: E402

MAX_TURNS = 10


def emit_event(event: dict[str, Any]) -> None:
    """Write a capture event to the offtrack trace dir, if one is set."""
    trace_dir = os.environ.get("OFFTRACK_TRACE_DIR")
    if not trace_dir:
        return
    path = Path(trace_dir) / f"agent-{os.getpid()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_agent(task_input: dict[str, Any], model: str) -> str:
    llm = get_llm(model)
    order_id = task_input.get("order_id", "TEST-1")
    user_input = task_input.get(
        "message", f"Customer wants a refund on order {order_id}."
    )
    ctx: dict[str, Any] = {"order_id": order_id}
    history: list[dict[str, Any]] = [{"role": "user", "content": user_input}]

    for _ in range(MAX_TURNS):
        t0 = now()
        resp = llm.respond(user_input, history, ctx)
        emit_event(
            {
                "ev": "step",
                "v": 1,
                "type": "llm_call",
                "name": resp.model,
                "model": resp.model,
                "args": {"tool_intent": [c["name"] for c in resp.tool_calls]},
                "result": {"final": resp.final_answer is not None},
                "usage": {"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out},
                "t0": t0,
                "t1": now(),
                "status": "ok",
            }
        )
        if resp.final_answer is not None:
            emit_event(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "final_answer",
                    "name": "final",
                    "result": resp.final_answer,
                    "t0": now(),
                    "t1": now(),
                    "status": "ok",
                }
            )
            emit_event({"ev": "end", "status": "complete"})
            return resp.final_answer

        group = f"g{len(history)}" if len(resp.tool_calls) > 1 else None
        for call in resp.tool_calls:
            fn = TOOLS[call["name"]]
            t0 = now()
            time.sleep(0.01)  # visible nonzero latency in demos
            result = fn(**call["args"])
            emit_event(
                {
                    "ev": "step",
                    "v": 1,
                    "type": "tool_call",
                    "name": call["name"],
                    "args": call["args"],
                    "result": result,
                    "t0": t0,
                    "t1": now(),
                    "group": group,
                    "status": "ok",
                }
            )
            history.append({"role": "tool", "name": call["name"], "content": result})
            # Feed context the personas' ${...} placeholders read from.
            if call["name"] == "lookup_order" and "total_usd" in result:
                ctx["order_total"] = result["total_usd"]
            if call["name"] == "issue_refund":
                ctx["last_refund_id"] = result.get("refund_id")
            if call["name"] == "escalate":
                ctx["last_ticket_id"] = result.get("ticket_id")

    emit_event({"ev": "end", "status": "partial"})
    return "Reached max turns without resolution."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="refund")
    parser.add_argument("--order", default="TEST-1")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "fake-careful"))
    args = parser.parse_args()

    task_input: dict[str, Any] = {"order_id": args.order, "task": args.task}
    if os.environ.get("OFFTRACK_TASK_INPUT"):
        task_input.update(json.loads(os.environ["OFFTRACK_TASK_INPUT"]))

    answer = run_agent(task_input, args.model)
    print(answer)


if __name__ == "__main__":
    main()
