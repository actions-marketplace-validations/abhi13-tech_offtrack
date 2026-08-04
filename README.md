# offtrack

**git diff for AI agent runs.**

Record golden trajectories. Re-run after any change. See the first step where your agent went off track — and gate CI on it.

> ⚠️ Under active development toward v0.1.0. Full README, docs, and demo GIF landing with the release.

## Why

You bumped a model or tweaked a prompt. Every eval score still looks fine. But your agent now refunds $842 without checking policy. Scores tell you *whether* behavior changed — offtrack shows you *where*: the first divergent step, with cost and latency deltas, from traces stored locally in SQLite. Statistical PASS/FAIL/INCONCLUSIVE verdicts gate CI so stochastic LLM variance doesn't cause false alarms.

## Quickstart (target)

```bash
pip install offtrack
offtrack init
offtrack record --promote     # capture golden trajectories
# ...change your model/prompt...
offtrack check                # first divergent step + verdict, exit code for CI
```

MIT licensed. Local-first: SQLite store, no server, no account. Vendor-neutral: OTel GenAI ingest plus OpenAI/Anthropic/LangGraph/Claude Code adapters.
