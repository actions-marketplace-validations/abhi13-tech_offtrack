"""offtrack.yaml loading and validation.

One config file: global config + mask rules + task suites together, with
per-task overrides. ${VAR} interpolates from the environment; every
undefined variable is reported at load time in a single error.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from offtrack.mask import parse_rules
from offtrack.model import canonical_json, sha256_hex


class SuiteConfigError(ValueError):
    pass


class AlignConfig(BaseModel):
    divergence_threshold: float = 0.85
    rel_tol: float = 0.0
    collapse_repeats: bool = False
    aliases: dict[str, str] = Field(default_factory=dict)


class VerdictConfig(BaseModel):
    alpha: float = 0.05
    min_effect: float = 0.30
    pass_bound: float = 0.46
    deterministic: bool = False


class MetricRule(BaseModel):
    threshold: float
    action: str = "fail"

    @field_validator("action")
    @classmethod
    def _check_action(cls, v: str) -> str:
        if v not in ("fail", "warn"):
            raise ValueError(f"metric action must be 'fail' or 'warn', got {v!r}")
        return v


DEFAULT_METRICS = {
    "cost": MetricRule(threshold=0.20, action="fail"),
    "tokens": MetricRule(threshold=0.20, action="fail"),
    "latency": MetricRule(threshold=0.50, action="warn"),
}


class GlobalConfig(BaseModel):
    repetitions: int = 5
    timeout_s: int = 300
    on_crash: str = "count_divergent"  # count_divergent | fail | exclude
    align: AlignConfig = Field(default_factory=AlignConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    metrics: dict[str, MetricRule] = Field(default_factory=lambda: dict(DEFAULT_METRICS))
    mask: dict[str, Any] = Field(default_factory=dict)
    ignore_steps: list[str] = Field(default_factory=list)


class RunSpec(BaseModel):
    command: list[str] | None = None
    entrypoint: str | None = None  # "module:function"

    @field_validator("entrypoint")
    @classmethod
    def _check_entrypoint(cls, v: str | None) -> str | None:
        if v is not None and ":" not in v:
            raise ValueError(f"entrypoint must be 'module:function', got {v!r}")
        return v


class Task(BaseModel):
    id: str
    run: RunSpec
    input: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    repetitions: int | None = None
    timeout_s: int | None = None
    mask: dict[str, Any] = Field(default_factory=dict)
    matrix: dict[str, list[str]] = Field(default_factory=dict)


class Suite(BaseModel):
    name: str
    tasks: list[Task]


class ResolvedTask(BaseModel):
    """A task flattened with suite name, matrix slot, and global defaults."""

    task_key: str  # "<suite>/<task_id>[#k=v]"
    suite: str
    task: Task
    matrix_env: dict[str, str] = Field(default_factory=dict)
    repetitions: int
    timeout_s: int
    mask_config: dict[str, Any]
    # Hashed from the UN-interpolated task definition: editing offtrack.yaml
    # marks baselines stale, but changing an env var (e.g. the model under
    # test) is the experiment — never staleness.
    config_hash: str = ""


class SuiteFile(BaseModel):
    version: int = 1
    config: GlobalConfig = Field(default_factory=GlobalConfig)
    suites: list[Suite] = Field(default_factory=list)


_VAR_RE = re.compile(r"\$\{([\w.]+)\}")


def _interpolate(value: Any, missing: set[str], matrix: dict[str, str]) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name.startswith("matrix."):
                name = name.removeprefix("matrix.")
            if name in matrix:
                return matrix[name]
            env = os.environ.get(name)
            if env is None:
                missing.add(m.group(1))
                return m.group(0)
            return env

        return _VAR_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, missing, matrix) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, missing, matrix) for v in value]
    return value


def load_suite_file(path: Path) -> SuiteFile:
    if not path.exists():
        raise SuiteConfigError(
            f"no config found at {path}. Nothing else is affected. "
            "Run `offtrack init` to scaffold one."
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise SuiteConfigError(f"{path} is not valid YAML: {e}") from e
    try:
        sf = SuiteFile.model_validate(raw)
    except Exception as e:
        raise SuiteConfigError(f"{path} failed validation: {e}") from e
    # Validate mask rules eagerly so config errors surface at load time.
    parse_rules(sf.config.mask)
    for suite in sf.suites:
        for task in suite.tasks:
            parse_rules({**sf.config.mask, **task.mask} if task.mask else sf.config.mask)
            if task.run.command is None and task.run.entrypoint is None:
                raise SuiteConfigError(
                    f"task {suite.name}/{task.id}: run needs 'command' or 'entrypoint'"
                )
    return sf


def resolve_tasks(sf: SuiteFile, only: list[str] | None = None) -> list[ResolvedTask]:
    """Expand suites × tasks × matrix into ResolvedTasks, interpolating env."""
    resolved: list[ResolvedTask] = []
    missing: set[str] = set()
    for suite in sf.suites:
        for task in suite.tasks:
            raw_hash = sha256_hex(
                canonical_json(
                    {
                        "run": task.run.model_dump(),
                        "input": task.input,
                        "env": task.env,  # un-interpolated: ${VARS} stay literal
                        "mask": task.mask,
                    }
                )
            )[:16]
            slots: list[dict[str, str]] = [{}]
            for key, values in task.matrix.items():
                slots = [dict(s, **{key: v}) for s in slots for v in values]
            for slot in slots:
                suffix = "#" + ",".join(f"{k}={v}" for k, v in sorted(slot.items())) if slot else ""
                interpolated = _interpolate(task.model_dump(), missing, slot)
                resolved_task = Task.model_validate(interpolated)
                mask_config = dict(sf.config.mask)
                if resolved_task.mask:
                    merged_rules = list(mask_config.get("rules", [])) + list(
                        resolved_task.mask.get("rules", [])
                    )
                    mask_config = {**mask_config, **resolved_task.mask, "rules": merged_rules}
                resolved.append(
                    ResolvedTask(
                        task_key=f"{suite.name}/{task.id}{suffix}",
                        suite=suite.name,
                        task=resolved_task,
                        matrix_env=slot,
                        repetitions=resolved_task.repetitions or sf.config.repetitions,
                        timeout_s=resolved_task.timeout_s or sf.config.timeout_s,
                        mask_config=mask_config,
                        config_hash=raw_hash,
                    )
                )
    if missing:
        raise SuiteConfigError(
            f"undefined environment variable(s) in offtrack.yaml: {sorted(missing)}. "
            "Set them or remove the references."
        )
    if only:
        wanted = set(only)
        resolved = [
            r
            for r in resolved
            if r.task_key in wanted
            or r.task.id in wanted
            or r.task_key.split("#")[0] in wanted
            or r.suite in wanted
        ]
        if not resolved:
            raise SuiteConfigError(
                f"no tasks match {sorted(wanted)}. "
                "Use `offtrack list tasks` to see available task keys."
            )
    return resolved
