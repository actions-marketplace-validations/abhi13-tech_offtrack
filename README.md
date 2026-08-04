# offtrack

**git diff for AI agent runs.**

Record golden trajectories. Re-run after any change. See the first step where your agent went off track — and gate CI on it.

[![CI](https://github.com/abhi13-tech/offtrack/actions/workflows/ci.yml/badge.svg)](https://github.com/abhi13-tech/offtrack/actions/workflows/ci.yml)
[![gated by offtrack](https://img.shields.io/badge/gated%20by-offtrack-orange)](https://github.com/abhi13-tech/offtrack/actions/workflows/dogfood.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/offtrack/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

![offtrack demo: record golden trajectories, then catch the model that skips the refund policy check](docs/demo.gif)

You bumped a model. Or tweaked a prompt. Or upgraded your agent framework.

Every eval score still looks fine. But your agent now refunds $842 **without checking the refund policy** — cheaper, faster, and catastrophically wrong:

```
$ offtrack check

  ✗ refund/over-limit  0/5 aligned
      =  step 0  llm_call     (both runs identical)
      =  step 1  tool_call    lookup_order({"order_id": "TEST-1"})
      ▲ first divergence — expected check_refund_policy, got issue_refund
        baseline  check_refund_policy({"amount_usd": 842.0})
        this run  issue_refund({"order_id": "TEST-1", "amount_usd": 842.0})
      divergence rate rose 0% → 100% (Fisher exact p=0.004, effect +100%)
      Δtokens -46%  Δlatency -30%

  FAIL — exit code 1
```

Eval scores tell you **whether** behavior changed. offtrack shows you **where**: the first divergent step, with cost and latency deltas, from traces stored locally in SQLite. Statistical verdicts (`PASS` / `FAIL` / `INCONCLUSIVE`) gate CI so stochastic LLM variance doesn't cause false alarms — and honest uncertainty doesn't hide behind a green check.

## Quickstart

```bash
pip install offtrack
offtrack init                 # scaffold offtrack.yaml + baselines/
offtrack record               # capture golden trajectories (N runs per task)
# ...change your model / prompt / framework...
offtrack check                # first divergent step + verdict + exit code
```

## Capturing traces — three ways

**Zero-code (OpenAI / Anthropic SDKs)** — two lines at your agent's entrypoint:

```python
import offtrack.capture
offtrack.capture.install()   # patches whichever SDKs are importable
```

Every LLM call is recorded, and tool calls are reconstructed by message delta — real names, args, and results, no changes to your agent loop.

**LangGraph / LangChain** — pass the callback:

```python
from offtrack.integrations.langgraph import OfftrackCallbackHandler

handler = OfftrackCallbackHandler()
graph.invoke(inputs, config={"callbacks": [handler]})
handler.finish()
```

**Anything else** — write capture events (JSONL) to `$OFFTRACK_TRACE_DIR`, or export OpenTelemetry GenAI traces there: offtrack ingests both semconv generations and the OpenInference flavor. Claude Code sessions import directly with `offtrack ingest claude-code <session.jsonl>`.

## How it works

1. **Record** — `offtrack record` runs each task N times (default 5) and stores the trajectories: every LLM call, tool call, argument, token count, cost, latency. Baselines auto-export to `baselines/*.json` — **committed to git**, so changing golden behavior is a reviewed act in PRs.
2. **Align** — new runs are aligned against baselines with Needleman-Wunsch sequence alignment over steps (tool-name gating + structural argument similarity). Reordered parallel calls, retries, and volatile fields (UUIDs, timestamps — masked by default) don't cause false alarms.
3. **Localize** — the report points at the **first divergent step**: missing, extra, or changed — with the argument-level diff and whether the trajectories resynced afterward.
4. **Verdict** — baselines are recorded N times so run-to-run variance is *measured, not assumed*. FAIL requires an exact Fisher test plus a minimum effect size; PASS requires an exact upper confidence bound. Anything else is INCONCLUSIVE with a prescription: "run 2 more repetitions."

## CI

```yaml
- uses: abhi13-tech/offtrack@v0
  with:
    suite: offtrack.yaml
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

FAIL fails the check and posts a PR comment with the divergence table. INCONCLUSIVE warns without blocking (strictness is opt-in via `inconclusive-as: fail`). Exit codes: `0` pass, `1` fail, `3` inconclusive, `4` setup error — infra problems never masquerade as regressions.

**This repo gates itself:** every PR runs offtrack against its own demo agent ([dogfood.yml](.github/workflows/dogfood.yml)), and the gate's failure path is itself under test.

## pytest

```python
@pytest.mark.offtrack(task="refund/over-limit")
def test_refund_flow(offtrack):
    with offtrack.record():
        run_my_agent("refund order TEST-1")
    offtrack.assert_matches_baseline()
    offtrack.assert_cost_under(usd=0.05)
    offtrack.assert_max_steps(30)
```

`pytest -n auto` (xdist) is fully supported — workers spill trajectories to append-only files; the controller merges at session end.

## What offtrack is not

These are complements, not competitors — offtrack is the trajectory-diff layer:

| | offtrack | promptfoo | LangSmith | EvalView |
|---|---|---|---|---|
| Unit of comparison | **step-level trajectory** | prompt/output pairs | hosted traces + evals | eval-run snapshots |
| First-divergence localization | **✓** | — | — | — |
| Semantic alignment (retries, reordering, masks) | **✓** | — | — | exact-match only |
| Statistical verdict (variance-aware) | **✓** | — | — | pass@k |
| Local-first, no account | **✓** SQLite | ✓ | hosted | ✓ |
| Vendor-neutral ingest (OTel GenAI, both semconv generations) | **✓** | — | — | — |

offtrack doesn't score answer *quality* — pair it with an eval framework for that. It catches the thing eval scores structurally miss: **procedural regressions** in how the agent got there.

## Demo

`examples/refund-agent/` is a complete offline demo — a small support agent with a written refund policy and two scripted personas: `fake-careful` (follows policy) and `fake-sloppy` (a stand-in for a cheaper model that skips the policy check). No API key, no network, $0:

```bash
cd examples/refund-agent
AGENT_MODEL=fake-careful offtrack record    # golden: lookup → policy check → escalate
AGENT_MODEL=fake-sloppy  offtrack check     # FAIL: skips policy, refunds $842 directly
```

The fake mode is a scripted LLM, not a mocked test — the entire real pipeline (capture, ingest, alignment, stats) executes identically in fake and real mode.

## Roadmap

- ~~LangGraph callback + OpenAI/Anthropic SDK capture shims~~ — shipped in 0.2.0 (`offtrack.capture.install()`)
- ~~Claude Code session ingest~~ — shipped in 0.2.0 (`offtrack ingest claude-code`)
- Semantic matchers (embedding / LLM-judge) via the `Matcher` protocol — v1 is deterministic-only by design
- `offtrack bisect` — find the commit that introduced a divergence
- CrewAI / Pydantic-AI adapters · HTML report viewer

## Development

```bash
git clone https://github.com/abhi13-tech/offtrack && cd offtrack
uv sync --all-extras --group dev
uv run pytest          # 200 tests, all offline
uv run mypy            # strict
just dogfood           # run the gate against the demo locally
```

MIT licensed.
