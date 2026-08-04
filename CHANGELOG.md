# Changelog

All notable changes to offtrack are documented here. Format: [Keep a Changelog](https://keepachangelog.com/); versioning: SemVer with the 0.x caveat (0.MINOR may break; the baseline *file* schema is versioned independently via `offtrack_schema`).

## [Unreleased]

## [0.2.0] - 2026-08-04

Zero-code capture.

### Added
- `offtrack.capture.install()` — patches the OpenAI and Anthropic Python SDKs; every LLM call and tool call recorded with zero agent-code changes (tool steps reconstructed by message delta)
- LangGraph/LangChain adapter: `OfftrackCallbackHandler` recording real executed tool args/results, model names, and token usage
- Claude Code session ingest: `offtrack ingest claude-code <session.jsonl>` (Task sidechains collapse to handoff steps)
- `offtrack ingest explain <file>` — per-span OTel classification decisions

### Fixed
- Builder crash when timestamped and untimestamped steps mixed (naive/aware datetime comparison)

## [0.1.0] - 2026-08-04

Initial release.

### Added
- Canonical trajectory model: `llm_call | tool_call | handoff | final_answer` steps with two-tier hashing and payload truncation (16 KiB inline / 512 KiB blob / structure-only stubs)
- Local-first SQLite store (WAL) with day-one migrations and committable, schema-versioned baseline JSON exports
- Capture-event JSONL ingestion with crash-safe reads and per-attempt trace directories
- OpenTelemetry GenAI importer covering both semconv attribute generations and the OpenInference flavor, with per-span classification decisions
- Needleman-Wunsch trajectory alignment: first-divergence localization, resync detection, multi-variant baselines, banded DP for long runs
- Masking DSL (builtin UUID/timestamp masks ON by default, JSONPath-subset rules, `mask suggest` from baseline self-variance)
- Exact statistical verdicts: Fisher one-sided + effect floor for FAIL, Clopper-Pearson upper bound for PASS, INCONCLUSIVE with run-count prescriptions, permutation-tested cost/token/latency gates
- CLI: `init · record · check · diff · show · list · baseline · doctor` with exit codes 0/1/3/4
- pytest plugin: `offtrack` fixture, baseline assertions, xdist-safe spill-and-merge
- GitHub Action (composite): PR comment upsert, step summary, verdict outputs, fork-safe degradation
- Offline demo: `examples/refund-agent` with scripted careful/sloppy personas — the repo dogfoods its own gate on every PR
