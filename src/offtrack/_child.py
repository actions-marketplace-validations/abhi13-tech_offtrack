"""Child-process entrypoint runner: `python -m offtrack._child module:function`.

Runs the user's entrypoint in a contained process so crashes and timeouts
never take down the offtrack runner. The function receives the task input
dict; its return value becomes a fallback final_answer if the trace lacks one.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    spec = sys.argv[1]
    module_name, _, func_name = spec.partition(":")
    task_input = json.loads(os.environ.get("OFFTRACK_TASK_INPUT", "{}"))
    trace_dir = os.environ.get("OFFTRACK_TRACE_DIR")
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        result = func(task_input)
    except Exception:
        if trace_dir:
            (Path(trace_dir) / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1

    if trace_dir and result is not None:
        # Fallback final_answer if the entrypoint's trace didn't emit one.
        path = Path(trace_dir) / f"entrypoint-{os.getpid()}.jsonl"
        has_final = False
        for f in Path(trace_dir).glob("*.jsonl"):
            try:
                for line in f.read_text().splitlines():
                    if '"final_answer"' in line:
                        has_final = True
                        break
            except OSError:
                pass
        if not has_final:
            now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            with path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ev": "step",
                            "v": 1,
                            "type": "final_answer",
                            "name": "final",
                            "result": result
                            if isinstance(result, str | dict | list)
                            else str(result),
                            "t0": now,
                            "t1": now,
                            "status": "ok",
                        }
                    )
                    + "\n"
                )
                fh.write(json.dumps({"ev": "end", "status": "complete"}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
