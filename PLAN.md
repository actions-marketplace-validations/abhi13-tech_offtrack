# offtrack — Complete Implementation Plan

## Context

Abhishek is targeting Agentic AI engineer roles (e.g. Palo Alto Networks Enterprise AI: LLM APIs, RAG, orchestration, agent/tool calling, prompt design, **evaluation frameworks**, **AI observability**). His CV is strong on RAG/orchestration but thin on evals + observability, and these JDs explicitly ask for "evidence of applied AI work (GitHub, writing, demos)."

A 13-agent deep-research workflow (7 ecosystem sweeps, 3 idea generators, 3 adversarial judges, ~250 web searches, July 2026 data) selected this project: **offtrack — "git diff for AI agent runs."** Ranked #1 by all judges (29.7/40); the only concept where no judge found an existing tool that does it. Verified open lane: EvalView (~125★) does exact-match snapshots only with documented baseline-churn complaints; first-divergence reporting was requested on `openai-agents-python` #3447 (May 2026) and closed unimplemented; promptfoo's OpenAI acquisition (Mar 2026) created a vendor-neutrality vacuum. Two of three idea generators independently converged on this concept.

**What it is:** record "golden" agent trajectories → change something (model bump, prompt tweak, framework upgrade) → re-run → semantically align new runs against baselines → report the **first divergent step** → gate CI with statistical verdicts (PASS/FAIL/INCONCLUSIVE) so stochastic LLM variance doesn't cause false alarms. Local-first (SQLite, no server), vendor-neutral (OTel GenAI ingest), MIT.

**Constraints:** solo build, ~2 weeks of daily Claude-assisted development, Python, pip-installable, demo-able in a README GIF, deep enough to discuss in interviews. Repo: `/Users/abhishekreddy/offtrack`, GitHub `abhi13-tech/offtrack`.

**User approvals so far:** flagship OSS tool (vs portfolio spread) ✔ · trajectory-diff concept ✔ · name `offtrack` (PyPI free, no GitHub collisions) ✔ · design part 1 (architecture, OTel-first ingest, local SQLite) ✔.

---

## Locked decisions (conflicts between design passes resolved)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Exit codes: 0 PASS · 1 FAIL · 3 INCONCLUSIVE · 4 setup/env ERROR** | Typer/Click reserve exit 2 for usage errors; a typo'd flag must never look like a verdict. `--inconclusive-as {pass,fail,inconclusive}` for strict teams. Infra problems are always 4 and never masquerade as FAIL. |
| D2 | **One config file: `offtrack.yaml`** (config + mask rules + task suites together) | One discovery path; masks/thresholds live beside the tasks they govern. |
| D3 | **Store split: `.offtrack/offtrack.db` (SQLite WAL, gitignored) + `baselines/*.json` at repo root (committed)** | DB = local ergonomics/cache; JSON = source of truth for CI, human-reviewable — changing golden behavior becomes a reviewed act in PRs. `.offtrack/` self-gitignores. |
| D4 | **Masks apply at compare time, raw data stored unmasked** | Users tune masks against already-recorded data; `mask_hash` stamps every comparison. |
| D5 | **v1 alignment is deterministic/structural only** (no LLM-judge calls) | Zero API spend for comparisons, reproducible; `Matcher` protocol is the extension point for semantic/embedding matching later. |
| D6 | **Everything must run offline** via the demo's scripted FakeLLM | Powers the GIF, the dogfood CI, fork-safe PR gates — all at $0. |
| D7 | **Dependency budget: 4** — typer, rich, pydantic v2, pyyaml | Stats are exact/stdlib (`math.comb`, bisection — no scipy/numpy); stdlib `sqlite3`, no ORM; offtrack itself never makes network calls. |
| D8 | **Python ≥3.10**, src layout, hatchling, py.typed, uv-first dev | CI matrix 3.10 / 3.12 / 3.14. |
| D9 | Verdict vocabulary is fixed strings `PASS/FAIL/INCONCLUSIVE/ERROR` everywhere | CLI, JSON report, pytest, Action outputs. |

---

## Architecture (7 units)

```
offtrack.yaml ──► suite loader ──► runner ──► trace dirs ──► ingest adapters ──► TrajectoryBuilder
                                                                                      │
        renderer (terminal/md/json) ◄── stats/verdicts ◄── alignment engine ◄── SQLite store
                                                                                      ▲
        surfaces: typer CLI · pytest plugin · GitHub Action          baselines/*.json ┘
```

### Module layout — `/Users/abhishekreddy/offtrack/src/offtrack/`
`model.py` · `suite.py` · `runner.py` · `store/` (`db.py`, `schema.py`, `migrations.py`, `baseline_io.py`) · `ingest/` (`builder.py`, `otel.py`, `shims.py`, `langgraph.py`, `claude_code.py`) · `align/` (`engine.py`, `similarity.py`, `matchers.py`) · `mask.py` · `stats.py` · `pricing/` (`pricing.json`, `resolve.py`) · `render/` (`terminal.py`, `markdown.py`, `jsonout.py`) · `cli/` (`app.py`, `commands/*.py`) · `pytest_plugin.py` · `py.typed`. Repo root: `action.yml`, `examples/refund-agent/`, `baselines/`, `tests/`, `docs/`, `.github/workflows/`.

---

## 1. Trace model (`model.py`)

- `Step`: `idx, type{llm_call|tool_call|handoff|final_answer}, name, args, result, status{ok|error|timeout|unknown}, model, tokens_in/out, cost_usd, started_at/ended_at (UTC ms), latency_ms, parallel_group, content_hash, args_blob/result_blob` (pydantic v2).
- `Trajectory`: ULID id (time-sortable), `task_key` (`<suite>/<task_id>[#matrix-slug]`), `kind{baseline|candidate}`, `attempt`, `status{complete|partial|error|timeout|empty}`, `source` (adapter), steps, totals, meta (git_ref, warnings, ambiguity flags).
- **Two-tier hashing:** `content_hash` (stored; sha256 of canonical JSON, unmasked — dedup/variants/idempotent import) vs `compare_hash` (computed lazily under active mask, keyed by `mask_hash` — changing masks never requires re-ingesting). Canonical JSON: sorted keys, NFC strings, floats via `repr`, NaN/Inf → `"__nonfinite__"` + warning.
- **Payload truncation:** ≤16 KiB inline · 16–512 KiB → stub {sha256, size, head 2 KiB, key-shape} + zstd blob in content-addressed `blobs` table · >512 KiB → stub only + one warning per run. Diffs on truncated payloads say "compared by structure; payload truncated" — never pretend.
- **Cost:** bundled `pricing.json` (versioned) → `~/.offtrack/pricing.json` → project → inline yaml overrides (later wins); alias longest-prefix resolution for date-suffixed model ids; `offtrack pricing update` pulls LiteLLM's community price registry (never auto-fetched). Provider-reported cost in trace wins over computed. Unknown model → cost NULL, excluded from gating (a missing price can never cause FAIL or PASS). Runs snapshot the resolved prices used, so history never mutates.

## 2. Store (`store/`)

- SQLite WAL, `busy_timeout=10s`, FK on. Tables: `schema_version, tasks, baselines, runs, trajectories, steps, blobs, alignments (cache; droppable), verdicts`. All writes in short `BEGIN IMMEDIATE` transactions with 3× jittered retry on BUSY → then clean error naming the likely concurrent process + per-job-DB guidance. Never on NFS (documented).
- **Migrations from day one:** ordered migration list; v1 ships as migration #1 so the runner is exercised from first release. DB newer than code → hard error ("upgrade offtrack"), never forward-compat reads. Pre-migration backup copy.
- **Baseline export/import:** `baselines/<suite>/<task_id>.json`, schema-versioned (`offtrack_schema: 1`), canonical serialization (stable git diffs), payloads stripped by default (`--with-blobs` embeds up to a cap). `record` auto-exports; `check` transparently imports when DB lacks the baseline (files are CI's source of truth; DB is a cache). Import idempotent by content_hash. Reader accepts schema N and N−1; newer → exit 4 with upgrade message.
- CI is **read-mostly by design**: baselines from committed JSON, each job writes its own local runs DB — cross-job write contention structurally avoided.

## 3. Task suites + runner (`suite.py`, `runner.py`)

`offtrack.yaml` (version, config{repetitions=5, timeout_s, align, verdict, metrics, mask, ignore_steps}, suites[].tasks[]): each task has `run.command` (subprocess) **or** `run.entrypoint` (`module:function` — still executed in a child process for crash containment); `input` (JSON → `OFFTRACK_TASK_INPUT`), `matrix` (e.g. models → task_key suffix), `env` with `${VAR}` interpolation (undefined → load error listing all), per-task mask/repetitions. `config_hash` (volatile fields excluded) stamps baselines for staleness detection.

- Injected env: `OFFTRACK_RUN_ID / TASK_KEY / ATTEMPT / TASK_INPUT / TRACE_DIR` (fresh dir per attempt — **directory-per-attempt is the trace-association mechanism**, no time-window guessing). `--autopatch` prepends a `sitecustomize.py` bootstrap installing SDK shims for zero-code capture.
- Multiple traces in one dir: prefer traces carrying injected `offtrack.run_id` attrs; else pick largest + warn + `ambiguous_trace` meta flag.

**Runner scenario matrix:** nonzero exit with traces → ingest partial, status=error, counted as divergent sample (`on_crash: count_divergent|fail|exclude`), rendered "crashed after step k" — never conflated with behavioral mismatch · timeout → SIGTERM, 10 s grace, SIGKILL, ingest flushed spans · exit 0 + zero traces → status=empty, ERROR-sample; all attempts empty → task verdict **ERROR** (never PASS/FAIL) + `offtrack doctor` hint · torn/partial spans → parse valid, drop torn tail, status=partial · entrypoint raise → sidecar traceback shown in report · baseline config_hash mismatch → verdict capped **INCONCLUSIVE (stale baseline — re-record)** unless `--allow-stale`.

## 4. Ingest adapters (`ingest/`)

All adapters emit one internal capture-event JSONL; a shared `TrajectoryBuilder` sorts, groups parallels, truncates, hashes, totals (adapters are translation-only; policy lives in the builder).

- **OTel GenAI** (`otel.py`): accepts OTLP/JSON + collector file-exporter JSONL (sniffed). Per-span field **coalescing across both semconv generations + OpenInference** (`gen_ai.provider.name`/`gen_ai.system`/`llm.provider`; usage token key variants; `gen_ai.operation.name`/`openinference.span.kind`/heuristics for classification). Only leaf semantic spans become steps. Scenario matrix: mixed generations per-span → coalescing just works (debug log records which generation matched); conflicts → new wins + warn once; orphaned tool spans → attach at root by time + warn; multiple trace_ids → group then selection rule; span without end_time → latency NULL, trajectory partial; unclassifiable spans → skipped with count, **`offtrack ingest --explain <dir>`** prints per-span classification decisions (the debuggability valve); duplicate span ids → dedup keep-last; non-JSON tool args → `{"__raw__": s}`.
- **OpenAI/Anthropic shims** (`shims.py`, `offtrack.capture.install()`): patch chat.completions/responses + messages (sync/async/streaming accumulated). Tool steps reconstructed by message-delta (pair `tool_call_id`/`tool_use_id` in response N with results in request N+1); parallel calls share `parallel_group`; last assistant msg without tool calls → final_answer. Flush per call to `shim-<pid>.jsonl` (crash-safe). Fire-and-forget tools → result null/status unknown; multi-conversation processes → partitioned by system-prompt hash + message-prefix chaining, ambiguity flagged.
- **LangGraph callback** (`langgraph.py`): `on_chat_model_start/end`, `on_tool_start/end` (real executed args/results), node transitions → handoff.
- **Claude Code JSONL** (`claude_code.py`): tool_use/tool_result pairing; Task-subagent sidechains collapse to one `handoff` step (child steps in meta, excluded from v1 alignment).

## 5. Alignment engine (`align/`)

Pipeline: drop `ignore_steps` → optional `collapse_repeats` (+`repeat_tolerance`) → canonicalize parallel groups (sort within group by (name, compare_hash); OTel grouping via sibling+time-overlap) → apply masks (memoized per (content_hash, mask_hash)) → **Needleman–Wunsch** vs each baseline variant.

- **Similarity:** different type → 0. tool_call: names unequal (after alias map) → 0; equal → `0.4 + 0.6·args_sim` (recursive structural: dict = 0.5·key-Jaccard + 0.5·mean value-sim; lists order-sensitive with length penalty; strings ≤2 KiB difflib ratio, longer hash-equal→1 else 0.3; numbers rel_tol; truncated stubs 0.6·shape+0.4·head). llm_call: `0.5 + 0.3·tool-intent (multiset of emitted tool names) + 0.2·model match` — model term **auto-exempted when candidate declares a different model** (the bump is the experiment). handoff: target equal → 1 else 0.2. final_answer: presence → 1.0 (content divergence reported informationally; semantic equivalence is the extension point). Prompt text never compared in v1.
- **Scoring:** `pair = 2·sim − 1`, `gap = −0.45` → prefers insert/delete over pairing unrelated steps; ties prefer diagonal for stable output.
- **First divergence:** first gap (`missing_step`/`extra_step`) or pair with sim < 0.85 (`changed_step`, field-level arg diff). Also **resync point** ("diverged at step 7, resynced at 11" vs "never resynced"). **Multi-variant baselines:** dedup N recordings by masked-trajectory hash into k variants; align vs each; best match by normalized score; divergent only if the best variant diverges; report names the closest variant.
- **Guardrails:** ≤500 steps full DP (pair sims memoized) · >500 banded DP (band 64, widened once, "approximate" annotation) · >2000 hard cap, align prefix, `truncated_alignment` flag · empty candidate → all-gap, divergence at step 0 annotated with status · over-masking (>80% of arg content) → "comparison may be vacuous" warning.
- **Extension point:** `Matcher` protocol (`similarity(a, b, ctx) -> float | None`, chain with defer); v1 registers `StructuralMatcher` only; config + entry-point discovery for future embedding/LLM matchers.

## 6. Stats + verdicts (`stats.py`) — exact, stdlib-only

- Baseline self-divergence `p̂_b` via **leave-one-out** over the n_b recordings (variance measured, not assumed). Candidate `p̂_c` = divergent/attempts. ERROR-samples never count toward either side.
- **FAIL** iff one-sided Fisher's exact p < α (0.05) **and** effect `p̂_c − p̂_b ≥ 0.30` (floor prevents trivial-but-significant fails). **PASS** iff exact Clopper–Pearson one-sided 95% upper bound on p_c ≤ max(p̂_b + pass_bound, pass_bound), `pass_bound = 0.45` (so 0/5 passes, with honest wording: "no divergence in 5/5 runs; 95% UB 45%"). Else **INCONCLUSIVE** with a prescription: smallest additional n that could resolve → "run ~4 more (`offtrack check --more 4`)". Opt-in `deterministic: true`: p̂_b=0 ∧ n_b≥3 → any divergence FAILs immediately.
- Defaults n=5/5; docs state detectable effect sizes plainly (Δ≈0.6 at n=5, Δ≈0.35 at n=10). Honesty is the INCONCLUSIVE verdict.
- **Metrics:** median_c vs median_b with relative thresholds (cost 20% fail, tokens 20% fail, latency 50% warn) gated by exact permutation test p<0.05; if n too small to ever reach 0.05 → "suggestive (+34%) — not gating at n=5", never FAIL.
- Aggregation: task = worst(behavioral, gating metrics); suite = FAIL > INCONCLUSIVE > PASS; ERROR reported separately.

## 7. Masking DSL (`mask.py`) — the false-alarm firewall

- `config.mask` + per-task merge. `builtin: [uuids, iso_timestamps, epoch_timestamps]` ON by default. Rules: `path` (in-house ~100-line JSONPath subset: `$ .key [n] .* ..key` — documented grammar, config error outside it), `field` sugar (any depth), `step` glob scoping, `kind: value_regex`, actions `drop(default)|hash|round:N|normalize_ws|lowercase`. Drop → sentinel `"__masked__"` (key presence still scores).
- **`offtrack mask suggest`**: aligns baseline recordings against each other, lists fields that vary *within* the baseline set → ready-to-paste rules; auto-runs as a hint at `record` time when self-divergence > 0. This + builtin masks is the #1 credibility defense.

## 8. Renderer (`render/`) — one verdict document, three backends

- Versioned JSON verdict doc is canonical; terminal/markdown renderers are pure functions over it.
- **Terminal (rich):** per-task verdict badge, `p̂_b → p̂_c`, closest-variant note, two-column aligned view windowed ±context (default 2) around first divergence — identical steps compressed to one dim `=` line, changed steps yellow with inline `+/−/~` key-level arg diff (masked keys `⊘`), gaps as `∅ extra step: …`, steps after divergence summarized never diffed. Cost/latency delta line printed **even on green** (observability-when-passing signal). Caveats block renders all warnings — never buried. NO_COLOR + non-TTY fallback.
- **Markdown (PR):** summary table (task | verdict | first divergence | Δcost | Δp50), `<details>` per failed task, capped ~60 KB with deterministic truncation, `<!-- offtrack-report -->` marker for comment upsert.

## 9. CLI (`cli/`)

Commands: `init` · `record [--runs 5 --promote --label]` · `check [--against latest|LABEL|path --runs 5 --report terminal|md|json|github --inconclusive-as --budget-usd --allow-stale --more N]` · `diff <A> [B] [--full --context N]` · `show <run> [--step --page --raw]` · `list {runs|baselines|tasks}` · `baseline {promote|export|import|list}` · `ingest --explain <dir>` · `mask suggest` · `pricing update` · `doctor [--repair]`. Global: `--db, --config, --json, --no-color, -q/-v`.

- `init`: creates `.offtrack/` (self-gitignoring), `offtrack.yaml` with two commented example tasks (entrypoint + command + fake-mode example), `baselines/.gitkeep` + one-line README ("commit these"); idempotent ("exists, skipped"); detects pytest → plugin hint; detects `.github/workflows/` → Action hint; exit 0 always.
- Error-message contract (helper enforced): every error names **(1) what broke, (2) what's safe/unaffected, (3) the exact next command**.

**Failure-UX matrix (test fixture list):** no API key → 1-token preflight per provider *before* running suite, exit 4, fake-mode hint, never print key fragments · no traces captured → "0 trace events — is the adapter attached?" + per-adapter hint, zero-step runs never stored as baseline, never PASS · corrupted DB → integrity_check on open, exit 4, "baselines/ unaffected", `doctor --repair` rebuilds + re-imports, old file kept, never auto-delete · baseline schema newer → exit 4 upgrade message; older → in-memory migrate + re-export suggestion · 200-step trajectory → elide identical spans (`= steps 4–61 identical`), `show` paginates 25/page, banded alignment note · zero baselines → exit 4 "setup error, not a test failure" + the two fix commands · CI without network → distinct from auth error, exit 4, "offtrack needs no network; your agent's provider does" + fork-safe fake-mode pointer — **never FAIL for infra** · ambiguous adapters → dedup by span id + pin hint.

## 10. pytest plugin (`pytest_plugin.py`)

- Entry point `pytest11`, ships in main package (import-guarded; `[pytest]` extra pins floor only).
- API: `@pytest.mark.offtrack(task=...)` + `offtrack` fixture → `with offtrack.record() as trace:` → `trace.assert_matches_baseline()` (reuses the terminal renderer plain-text), `assert_no_divergence(before_step=)`, `assert_cost_under(usd=)`, `assert_max_steps()`, `trace.first_divergence` property. INCONCLUSIVE → `pytest.skip` with reason (`--offtrack-inconclusive=fail` to flip). No baseline → record + warn + pass by default (`--offtrack-require-baseline` for CI). Marker-only mode (`record=True`) for zero-touch capture.
- Same `.offtrack/` DB as CLI (`source="pytest"`, nodeid in meta) — one store, two front doors. **xdist-safe via spill-and-merge:** workers append to `.offtrack/pending/<worker>.jsonl` (no locking); controller merges at sessionfinish; in-test assertions read baselines from JSON files (read-only, fork-safe). Non-xdist uses same path with `worker_id=main` — one code path.

## 11. GitHub Action (`action.yml`, composite)

- Inputs: suite, baselines, runs, inconclusive-as, comment (`true|false|update`, default update — upsert via marker), offtrack-version (`latest`; empty = `pip install .` for dogfood), working-directory. Secrets via `env:` only. Outputs: verdict, report-path, first-divergence JSON.
- Status mapping: PASS→success · FAIL→failure · INCONCLUSIVE→**exit 0 + `::warning::` annotation by default** (blocking merges on inconclusive trains users to ignore the tool; strictness opt-in via `inconclusive-as: fail`). Always writes `$GITHUB_STEP_SUMMARY` (works on forks where comments can't); full traces as artifact (14-day retention).
- Forked PRs: comment step `continue-on-error` + log line; docs explicitly warn **against** `pull_request_target` + fork checkout; recommended fork-safe gate = fake mode, real-model check on push-to-main. Matrix-across-models documented via `OFFTRACK_MODEL_OVERRIDE`.

## 12. Demo: `examples/refund-agent/` (built on day 1 — it is the test fixture for everything)

~150-line support agent, four local deterministic tools (`lookup_order, check_refund_policy, issue_refund, escalate`), written refund policy (auto ≤$500, else policy-check + escalate). `providers.py`: real provider (OpenAI or Anthropic key, 10-line switch) **or** `FakeLLM` driven by `fake_scripts/careful.json` / `sloppy.json` personas — a scripted LLM, not a mocked test, so the entire real pipeline executes offline.
**Marquee scene:** golden = lookup → check_refund_policy → escalate on an $842 refund; sloppy persona (stand-in for a cheaper model) skips the policy check and calls `issue_refund` directly → **first divergence at step 3, cheaper AND faster AND catastrophic** — the comment/table juxtaposing FAIL with −38% cost is the whole pitch in one row.

## 13. Repo quality + dogfooding (the portfolio showpiece)

- `ci.yml`: {3.10, 3.12, 3.14} × ubuntu + one macos leg — uv sync → ruff check+format → mypy --strict → pytest --cov (fail <85%) → build + twine check.
- `dogfood.yml` (separate badge): installs from source, runs `uses: ./` on the demo suite in fake mode — **every PR to offtrack gets an offtrack PR comment on its own demo agent**. `tests/integration/test_gate_catches_regression.py` runs the sloppy persona and asserts exit 1 + divergence at step 3 — the gate's failure path is itself under test.
- `release.yml`: tag `v*` → build → **PyPI Trusted Publishing (OIDC, no token secrets)** → GitHub Release with changelog extract → move `v0` major tag for `uses: @v0`.
- README badge "gated by offtrack" → dogfood.yml. CHANGELOG (Keep-a-Changelog) with CI check. CONTRIBUTING: uv in 3 commands + "adapters wanted" checklist (converts stars into PRs). Issue templates ask for `doctor` output + `--json` report.

## 14. README + launch

- Hero: *"git diff for AI agent runs."* / "Record golden trajectories. Re-run after any change. See the first step where your agent went off track — and gate CI on it." Quickstart = exactly 5 commands. Comparison table framed as complements (offtrack = the trajectory-diff layer) vs promptfoo/EvalView/LangSmith. Roadmap: HTML viewer, CrewAI/Pydantic-AI adapters, `offtrack bisect`, divergence clustering, semantic matchers.
- GIF (vhs-scripted, fake mode, ≤25 s): record → one-line model change → check fails at step 3 (hold) → PR comment → `pip install offtrack`.
- Launch week: Mon PyPI v0.1.0 + soft-post r/AI_Agents (catches broken quickstarts) → Tue fixes + r/LangChain (war-story framing, not self-promo) → Wed 14:00 UTC **Show HN**: "Show HN: Offtrack – git diff for AI agent runs" (honest-limitations paragraph; ask for feedback on the alignment approach; ship a fix during the thread) → Thu X/LinkedIn thread → Fri public triage + good-first-issues.
- Resume bullets (3 drafts exist in the surfaces design; finalize post-launch with real numbers).

---

## Build sequence (14 days, TDD per superpowers flow)

| Days | Deliverable (each ends green + committed) |
|------|-------------------------------------------|
| 1 | Repo scaffold (pyproject, uv, ruff/mypy/pytest, src layout, CI skeleton) + **demo agent with FakeLLM personas** + `model.py` (types, canonical JSON, hashing, truncation) |
| 2 | `store/` (schema, migrations runner, WAL/concurrency, blobs) + baseline export/import round-trip |
| 3 | `ingest/builder.py` + shims (OpenAI/Anthropic capture → JSONL → Trajectory); vertical slice: demo agent recorded end-to-end |
| 4 | `suite.py` + `runner.py` (subprocess + entrypoint, trace-dir association, crash/timeout/empty matrix) |
| 5 | `align/` (similarity, NW, first-divergence, resync, multi-variant, guardrails) — the crown jewel gets its own day + property tests |
| 6 | `mask.py` (DSL, builtins, `mask suggest`) + `stats.py` (Fisher, Clopper–Pearson, permutation tests, verdict rules) |
| 7 | `render/` (JSON doc → terminal → markdown) + `cli/` record/check/diff happy path → **GIF-able milestone** |
| 8 | Remaining CLI (init, show, list, baseline, doctor, ingest --explain, pricing) + failure-UX matrix as tests |
| 9 | OTel GenAI importer (coalescing table, classification, --explain) + committed corpus of real framework exports |
| 10 | LangGraph callback + Claude Code reader + pytest plugin (spill-and-merge, assertions) |
| 11 | action.yml + dogfood.yml + release.yml + integration test asserting the gate catches the sloppy persona |
| 12 | Hardening pass against the 10 credibility edge cases; `--jobs`, budget guard; mypy --strict + coverage ≥85% |
| 13 | README + docs/ (quickstart, ci, pytest, concepts, failure-modes) + vhs GIF + CHANGELOG |
| 14 | v0.1.0: PyPI trusted publishing, tag, GitHub Release; launch-week posts drafted; soft-launch r/AI_Agents |

## Top 10 credibility risks (each has a designed mitigation above)

1. Volatile args → false FAILs (masks ON by default + `mask suggest` + field-level diff) · 2. False PASS at tiny n (exact upper bounds + honest wording + INCONCLUSIVE prescriptions) · 3. Legit stochastic variance mislabeled (multi-variant baselines, measured p̂_b as null) · 4. OTel semconv chaos (coalescing table + `ingest --explain` + test corpus) · 5. Retry-loop count variance (NW insertions + collapse_repeats) · 6. Reordered parallel calls (group canonical sort) · 7. Trace association mix-ups (dir-per-attempt + injected ids + ignore_steps) · 8. Huge trajectories/payloads (truncation fast-paths, banded DP, caps that warn) · 9. Stale baselines (config_hash → INCONCLUSIVE + re-record) · 10. CI concurrency (read-mostly design, WAL, clean errors).

## Verification

- **Per unit:** pytest with golden fixtures (synthetic trajectories for alignment; recorded real-framework OTel exports for ingest); property tests on alignment invariants (self-alignment score max, symmetry, mask idempotence); exact-stats unit tests against hand-computed tables.
- **End-to-end:** `offtrack record --promote && offtrack check` on the demo in fake mode (careful persona → PASS exit 0); swap to sloppy persona → FAIL exit 1 with first divergence at step 3; integration test asserts both.
- **Surfaces:** CLI via Typer CliRunner; pytest plugin via pytester; Action via dogfood.yml on a real PR.
- **Dogfood:** the repo's own CI gates its own demo agent on every PR — if that badge is green, the product works.
- **Release check:** `pip install offtrack` in a clean venv → quickstart 5 commands run clean on the example.

## Out of scope for v1 (roadmap, stated in README)

LLM-judge/embedding semantic matching (Matcher protocol ready) · hosted dashboard/HTML viewer · TS/JS port · CrewAI/Pydantic-AI adapters · `offtrack bisect` · replay/mocking of recorded LLM responses.
