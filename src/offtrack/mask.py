"""Masking DSL — the false-alarm firewall.

Masks volatile fields (ids, timestamps, cursors) so they never count against
a match. Applied at COMPARE time only: stored data stays raw, so users tune
masks against already-recorded runs. Every comparison is stamped with the
mask_hash of the rules that shaped it.

Rule forms (offtrack.yaml `config.mask` / per-task `mask`):
  builtin: [uuids, iso_timestamps, epoch_timestamps]     # ON by default
  rules:
    - path: "$.args.request_id"          # JSONPath subset, see _PathExpr
    - field: session_id                  # any key with this name, any depth
    - step: "search_*"                   # scope rule to matching step names
      path: "$.result.score"
      action: round:2                    # drop | hash | round:N | normalize_ws | lowercase
    - kind: value_regex
      pattern: "^[0-9a-f]{8}-"
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from offtrack.model import MASKED, Step, canonical_json, sha256_hex

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
EPOCH_RE = re.compile(r"^1[5-9]\d{8}(\d{3})?(\.\d+)?$")  # 2017–2033, s or ms

BUILTINS = {"uuids", "iso_timestamps", "epoch_timestamps"}
DEFAULT_BUILTINS = ["uuids", "iso_timestamps", "epoch_timestamps"]


class MaskConfigError(ValueError):
    pass


@dataclass
class MaskRule:
    path: list[str] | None = None  # parsed path tokens
    field_name: str | None = None
    step_glob: str | None = None
    action: str = "drop"
    value_pattern: re.Pattern[str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


_PATH_TOKEN = re.compile(r"\.\.?([A-Za-z_][A-Za-z0-9_]*|\*)|\[(\d+|\*)\]")


def _parse_path(path: str) -> list[str]:
    """Parse the documented JSONPath subset: $ .key ..key .* [n] [*].

    Tokens: "key" (descend), "*" (any key/index), "**key" (recursive descend
    to key), "[n]" (index), produced from `.key`, `..key`, `.*`, `[n]`, `[*]`.
    """
    if not path.startswith("$"):
        raise MaskConfigError(
            f"mask path must start with '$': {path!r}. "
            "Supported syntax: $.key, $..key, $.*, $.key[0], $.key[*]"
        )
    tokens: list[str] = []
    pos = 1
    while pos < len(path):
        m = _PATH_TOKEN.match(path, pos)
        if not m:
            raise MaskConfigError(
                f"unsupported mask path syntax at {path[pos:]!r} in {path!r}. "
                "Supported: $.key, $..key (recursive), $.* , $.key[0], $.key[*]"
            )
        if m.group(1) is not None:
            token = m.group(1)
            tokens.append(f"**{token}" if path[m.start() : m.start() + 2] == ".." else token)
        else:
            tokens.append(f"[{m.group(2)}]")
        pos = m.end()
    if not tokens:
        raise MaskConfigError(f"empty mask path: {path!r}")
    return tokens


def parse_rules(config: dict[str, Any] | None) -> list[MaskRule]:
    config = config or {}
    rules: list[MaskRule] = []
    builtins = config.get("builtin", DEFAULT_BUILTINS)
    for b in builtins:
        if b not in BUILTINS:
            raise MaskConfigError(f"unknown builtin mask {b!r}; available: {sorted(BUILTINS)}")
    for name in builtins:
        pattern = {"uuids": UUID_RE, "iso_timestamps": ISO_TS_RE, "epoch_timestamps": EPOCH_RE}[
            name
        ]
        rules.append(MaskRule(value_pattern=pattern, raw={"builtin": name}))

    for r in config.get("rules", []):
        rule = MaskRule(
            step_glob=r.get("step"),
            action=r.get("action", "drop"),
            raw=r,
        )
        if r.get("kind") == "value_regex":
            try:
                rule.value_pattern = re.compile(r["pattern"])
            except (KeyError, re.error) as e:
                raise MaskConfigError(f"invalid value_regex rule {r!r}: {e}") from e
        elif "path" in r:
            rule.path = _parse_path(r["path"])
        elif "field" in r:
            rule.field_name = str(r["field"])
        else:
            raise MaskConfigError(
                f"mask rule needs 'path', 'field', or kind: value_regex — got {r!r}"
            )
        if rule.action != "drop" and not re.fullmatch(
            r"hash|round:\d+|normalize_ws|lowercase", rule.action
        ):
            raise MaskConfigError(
                f"unknown mask action {rule.action!r}; "
                "supported: drop, hash, round:N, normalize_ws, lowercase"
            )
        rules.append(rule)
    return rules


def mask_hash(rules: list[MaskRule]) -> str:
    return sha256_hex(canonical_json([r.raw for r in rules]))[:16]


def _apply_action(value: Any, action: str) -> Any:
    if action == "drop":
        return MASKED
    if action == "hash":
        return f"__hash_{sha256_hex(canonical_json(value))[:12]}__"
    if action.startswith("round:"):
        digits = int(action.split(":")[1])
        if isinstance(value, int | float) and not isinstance(value, bool):
            return round(float(value), digits)
        return value
    if action == "normalize_ws":
        return re.sub(r"\s+", " ", str(value)).strip() if isinstance(value, str) else value
    if action == "lowercase":
        return value.lower() if isinstance(value, str) else value
    return MASKED


def _walk_path(value: Any, tokens: list[str], action: str) -> Any:
    if not tokens:
        return _apply_action(value, action)
    head, rest = tokens[0], tokens[1:]

    if head.startswith("**"):
        key = head[2:]
        if isinstance(value, dict):
            return {
                k: (
                    _walk_path(v, rest, action)
                    if k == key
                    else _walk_path(v, tokens, action)  # keep descending
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_walk_path(v, tokens, action) for v in value]
        return value

    if head == "*":
        if isinstance(value, dict):
            return {k: _walk_path(v, rest, action) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk_path(v, rest, action) for v in value]
        return value

    if head.startswith("["):
        idx_token = head[1:-1]
        if not isinstance(value, list):
            return value
        if idx_token == "*":
            return [_walk_path(v, rest, action) for v in value]
        idx = int(idx_token)
        return [_walk_path(v, rest, action) if i == idx else v for i, v in enumerate(value)]

    if isinstance(value, dict) and head in value:
        return {k: (_walk_path(v, rest, action) if k == head else v) for k, v in value.items()}
    return value


def _mask_field(value: Any, name: str, action: str) -> Any:
    if isinstance(value, dict):
        return {
            k: (_apply_action(v, action) if k == name else _mask_field(v, name, action))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_mask_field(v, name, action) for v in value]
    return value


def _mask_values(value: Any, pattern: re.Pattern[str], action: str) -> Any:
    if isinstance(value, dict):
        return {k: _mask_values(v, pattern, action) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_values(v, pattern, action) for v in value]
    if isinstance(value, str) and pattern.match(value):
        return _apply_action(value, action)
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and pattern.match(repr(value))
    ):
        return _apply_action(value, action)
    return value


def apply_masks(value: Any, rules: list[MaskRule], step_name: str = "") -> Any:
    for rule in rules:
        if rule.step_glob and not fnmatch.fnmatch(step_name, rule.step_glob):
            continue
        if rule.path is not None:
            # Paths address the step payload root: $.args…, $.result…
            value = _walk_path(value, rule.path, rule.action)
        elif rule.field_name is not None:
            value = _mask_field(value, rule.field_name, rule.action)
        elif rule.value_pattern is not None:
            value = _mask_values(value, rule.value_pattern, rule.action)
    return value


def masked_step(step: Step, rules: list[MaskRule]) -> Step:
    """Compare-form of a step: payload with masks applied, content re-hashed."""
    if not rules:
        return step
    payload = {"args": step.args, "result": step.result}
    masked = apply_masks(payload, rules, step_name=step.name)
    clone = step.model_copy(deep=True)
    clone.args = masked["args"]
    clone.result = masked["result"]
    clone.content_hash = clone.compute_content_hash()
    return clone


def masked_trajectory_steps(steps: list[Step], rules: list[MaskRule]) -> list[Step]:
    return [masked_step(s, rules) for s in steps]


def suggest_masks(recordings: list[list[Step]]) -> list[dict[str, Any]]:
    """Propose mask rules from fields that vary WITHIN the baseline set.

    Aligns recordings positionally by (type, name) and reports arg fields
    whose values differ across recordings — volatile by construction.
    """
    if len(recordings) < 2:
        return []
    suggestions: dict[tuple[str, str], dict[str, Any]] = {}
    reference = recordings[0]
    for other in recordings[1:]:
        for a, b in zip(reference, other, strict=False):
            if a.type != b.type or a.name != b.name:
                continue
            for path in _diff_paths(a.args, b.args, "$.args"):
                key = (a.name, path)
                if key not in suggestions:
                    suggestions[key] = {"step": a.name, "path": path}
    return sorted(suggestions.values(), key=lambda s: (s["step"], s["path"]))


def _diff_paths(a: Any, b: Any, prefix: str) -> list[str]:
    if type(a) is not type(b):
        return [prefix]
    if isinstance(a, dict):
        paths = []
        for k in set(a) & set(b):
            paths.extend(_diff_paths(a[k], b[k], f"{prefix}.{k}"))
        return paths
    if isinstance(a, list):
        paths = []
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            paths.extend(_diff_paths(x, y, f"{prefix}[{i}]"))
        return paths
    return [] if a == b else [prefix]
