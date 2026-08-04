"""Canonical trajectory model: every ingest adapter converts into these types,
and everything downstream (store, alignment, stats, render) consumes only them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Payload size policy (bytes of canonical JSON). See PLAN.md §1.
INLINE_MAX = 16 * 1024
BLOB_MAX = 512 * 1024
HEAD_BYTES = 2 * 1024

NONFINITE = "__nonfinite__"
MASKED = "__masked__"
TRUNCATED_KEY = "__offtrack_truncated__"


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    HANDOFF = "handoff"
    FINAL_ANSWER = "final_answer"


class StepStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TrajStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"
    EMPTY = "empty"


def _canon(value: Any, warnings: list[str] | None = None) -> Any:
    """Normalize a JSON-ish value for canonical serialization."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            if warnings is not None:
                warnings.append("non-finite float replaced with sentinel")
            return NONFINITE
        if value == 0.0:  # collapse -0.0
            return 0.0
        return value
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", str(k)): _canon(v, warnings) for k, v in value.items()}
    if isinstance(value, list):
        return [_canon(v, warnings) for v in value]
    return value


def canonical_json(value: Any, warnings: list[str] | None = None) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NFC strings, repr-stable floats."""
    return json.dumps(
        _canon(value, warnings), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def json_shape(value: Any, depth: int = 6) -> Any:
    """Key-tree of a JSON value with leaf values elided — used for comparing
    truncated payloads by structure."""
    if depth <= 0:
        return "..."
    if isinstance(value, dict):
        return {
            k: json_shape(v, depth - 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list):
        inner = [json_shape(v, depth - 1) for v in value[:8]]
        if len(value) > 8:
            inner.append(f"...+{len(value) - 8}")
        return inner
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "num"
    if value is None:
        return "null"
    return type(value).__name__


class TruncationResult(BaseModel):
    """Outcome of applying the payload-size policy to one value."""

    inline: Any  # what gets stored inline (original value or stub)
    blob: bytes | None = None  # compressed full payload for the blobs table
    blob_sha256: str | None = None
    truncated: bool = False
    dropped: bool = False  # >BLOB_MAX: full payload not retained anywhere


def truncate_payload(value: Any) -> TruncationResult:
    """Apply the inline/blob/drop policy from PLAN.md §1 to one args/result value."""
    if value is None:
        return TruncationResult(inline=None)
    canon = canonical_json(value)
    raw = canon.encode("utf-8")
    if len(raw) <= INLINE_MAX:
        return TruncationResult(inline=value)

    stub = {
        TRUNCATED_KEY: True,
        "sha256": sha256_hex(raw),
        "size": len(raw),
        "head": canon[:HEAD_BYTES],
        "shape": json_shape(value),
    }
    if len(raw) <= BLOB_MAX:
        import zlib

        return TruncationResult(
            inline=stub,
            blob=zlib.compress(raw, 6),
            blob_sha256=stub["sha256"],
            truncated=True,
        )
    return TruncationResult(inline=stub, truncated=True, dropped=True)


def is_truncated_stub(value: Any) -> bool:
    return isinstance(value, dict) and value.get(TRUNCATED_KEY) is True


# --- ULID (time-sortable id, no external dependency) -------------------------

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    ts = int(time.time() * 1000)
    chars = ["0"] * 10
    for i in range(9, -1, -1):
        chars[i] = _ULID_ALPHABET[ts & 31]
        ts >>= 5
    rand = int.from_bytes(os.urandom(10), "big")
    rchars = ["0"] * 16
    for i in range(15, -1, -1):
        rchars[i] = _ULID_ALPHABET[rand & 31]
        rand >>= 5
    return "".join(chars) + "".join(rchars)


# --- Core models -------------------------------------------------------------


class Step(BaseModel):
    idx: int
    type: StepType
    name: str
    args: Any = None
    result: Any = None
    status: StepStatus = StepStatus.OK
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: int | None = None
    parallel_group: str | None = None
    content_hash: str = ""
    args_blob: str | None = None
    result_blob: str | None = None

    def compute_content_hash(self) -> str:
        """Structural identity of the step, unmasked (PLAN.md two-tier hashing)."""
        payload = {
            "type": self.type.value,
            "name": self.name,
            "args": self.args,
            "result_shape": json_shape(self.result) if self.result is not None else None,
        }
        return sha256_hex(canonical_json(payload))

    def finalize(self) -> Step:
        if self.started_at and self.ended_at:
            ms = int((self.ended_at - self.started_at).total_seconds() * 1000)
            self.latency_ms = ms if ms >= 0 else None  # clock skew → NULL
        self.content_hash = self.compute_content_hash()
        return self


class Trajectory(BaseModel):
    trajectory_id: str = Field(default_factory=new_ulid)
    task_key: str
    kind: Literal["baseline", "candidate"]
    attempt: int = 0
    status: TrajStatus = TrajStatus.COMPLETE
    source: str = "unknown"
    steps: list[Step] = Field(default_factory=list)
    content_hash: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    wall_ms: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def finalize(self) -> Trajectory:
        for i, step in enumerate(self.steps):
            step.idx = i
            step.finalize()
        self.content_hash = sha256_hex(
            canonical_json([s.content_hash for s in self.steps] + [self.status.value])
        )
        self.tokens_in = _sum_or_none(s.tokens_in for s in self.steps)
        self.tokens_out = _sum_or_none(s.tokens_out for s in self.steps)
        self.cost_usd = _sum_or_none(s.cost_usd for s in self.steps)
        starts = [s.started_at for s in self.steps if s.started_at]
        ends = [s.ended_at for s in self.steps if s.ended_at]
        if starts and ends:
            self.started_at = min(starts)
            self.ended_at = max(ends)
            ms = int((self.ended_at - self.started_at).total_seconds() * 1000)
            self.wall_ms = ms if ms >= 0 else None
        if not self.steps and self.status == TrajStatus.COMPLETE:
            self.status = TrajStatus.EMPTY
        return self

    @property
    def totals_are_lower_bound(self) -> bool:
        """True when some steps lack token counts, so totals carry a '≥' marker."""
        return any(s.tokens_in is None and s.type == StepType.LLM_CALL for s in self.steps)


def _sum_or_none(values: Any) -> Any:
    present = [v for v in values if v is not None]
    return sum(present) if present else None
