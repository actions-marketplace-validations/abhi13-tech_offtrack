set shell := ["bash", "-c"]

# Run everything CI runs
check: lint typecheck test

lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

typecheck:
    uv run mypy

test:
    uv run pytest -q

cov:
    uv run pytest --cov --cov-report=term-missing -q

# Run the gate against the demo locally (fake mode, $0)
dogfood:
    cd examples/refund-agent && AGENT_MODEL=fake-careful uv run offtrack check

# Reproduce the marquee regression
regress:
    cd examples/refund-agent && AGENT_MODEL=fake-sloppy uv run offtrack check || true

# Re-record demo baselines (after intentional demo changes)
rebaseline:
    cd examples/refund-agent && rm -rf .offtrack && AGENT_MODEL=fake-careful uv run offtrack record

build:
    uv build
