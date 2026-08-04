"""Unit tests for the canonical trajectory model."""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timedelta, timezone

from offtrack.model import (
    BLOB_MAX,
    INLINE_MAX,
    NONFINITE,
    Step,
    StepType,
    Trajectory,
    TrajStatus,
    canonical_json,
    is_truncated_stub,
    json_shape,
    new_ulid,
    truncate_payload,
)


class TestCanonicalJson:
    def test_sorted_keys_no_whitespace(self):
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_nested_sorting(self):
        assert canonical_json({"z": {"y": 1, "x": 2}}) == '{"z":{"x":2,"y":1}}'

    def test_negative_zero_collapsed(self):
        assert canonical_json(-0.0) == canonical_json(0.0)

    def test_nan_replaced_with_sentinel_and_warning(self):
        warnings: list[str] = []
        out = canonical_json(float("nan"), warnings)
        assert out == f'"{NONFINITE}"'
        assert warnings

    def test_inf_replaced(self):
        assert canonical_json(float("inf")) == f'"{NONFINITE}"'

    def test_unicode_nfc_normalized(self):
        # e + combining acute vs precomposed é must hash identically
        assert canonical_json("é") == canonical_json("é")

    def test_deterministic_across_calls(self):
        v = {"list": [1, {"b": 2, "a": [3.5, "x"]}], "s": "é"}
        assert canonical_json(v) == canonical_json(v)


class TestTruncation:
    def test_small_payload_inline(self):
        r = truncate_payload({"k": "v"})
        assert r.inline == {"k": "v"} and not r.truncated and r.blob is None

    def test_none_payload(self):
        r = truncate_payload(None)
        assert r.inline is None and not r.truncated

    def test_medium_payload_stubbed_with_blob(self):
        big = {"data": "x" * (INLINE_MAX + 100)}
        r = truncate_payload(big)
        assert is_truncated_stub(r.inline)
        assert r.blob is not None and r.blob_sha256 == r.inline["sha256"]
        # blob round-trips to the canonical JSON
        assert json.loads(zlib.decompress(r.blob)) == big
        assert not r.dropped

    def test_huge_payload_dropped(self):
        big = {"data": "x" * (BLOB_MAX + 100)}
        r = truncate_payload(big)
        assert is_truncated_stub(r.inline) and r.blob is None and r.dropped

    def test_stub_has_head_and_shape(self):
        big = {"key": "y" * (INLINE_MAX * 2)}
        r = truncate_payload(big)
        assert r.inline["head"].startswith('{"key"')
        assert r.inline["shape"] == {"key": "str"}


class TestJsonShape:
    def test_shapes(self):
        assert json_shape({"a": 1, "b": "s", "c": [True, None]}) == {
            "a": "num",
            "b": "str",
            "c": ["bool", "null"],
        }

    def test_long_list_elided(self):
        assert json_shape(list(range(20)))[-1] == "...+12"


class TestStepHashing:
    def make(self, **kw) -> Step:
        base = dict(idx=0, type=StepType.TOOL_CALL, name="lookup", args={"id": 1})
        base.update(kw)
        return Step(**base).finalize()

    def test_same_content_same_hash(self):
        assert self.make().content_hash == self.make().content_hash

    def test_different_args_different_hash(self):
        assert self.make().content_hash != self.make(args={"id": 2}).content_hash

    def test_different_name_different_hash(self):
        assert self.make().content_hash != self.make(name="other").content_hash

    def test_result_contributes_shape_only(self):
        # same shape, different values → same hash (results are stochastic)
        a = self.make(result={"total": 100})
        b = self.make(result={"total": 999})
        assert a.content_hash == b.content_hash
        c = self.make(result={"different_key": 1})
        assert a.content_hash != c.content_hash

    def test_latency_from_timestamps(self):
        t0 = datetime(2026, 8, 3, tzinfo=timezone.utc)
        s = self.make(started_at=t0, ended_at=t0 + timedelta(milliseconds=250))
        assert s.latency_ms == 250

    def test_clock_skew_gives_null_latency(self):
        t0 = datetime(2026, 8, 3, tzinfo=timezone.utc)
        s = self.make(started_at=t0, ended_at=t0 - timedelta(seconds=1))
        assert s.latency_ms is None


class TestTrajectory:
    def make_traj(self, n=3) -> Trajectory:
        steps = [
            Step(
                idx=0,
                type=StepType.TOOL_CALL,
                name=f"tool{i}",
                args={"i": i},
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.001,
            )
            for i in range(n)
        ]
        return Trajectory(task_key="s/t", kind="baseline", steps=steps).finalize()

    def test_totals(self):
        t = self.make_traj()
        assert t.tokens_in == 30 and t.tokens_out == 15
        assert t.cost_usd is not None and abs(t.cost_usd - 0.003) < 1e-9

    def test_idx_reassigned(self):
        t = self.make_traj()
        assert [s.idx for s in t.steps] == [0, 1, 2]

    def test_content_hash_stable_and_order_sensitive(self):
        a, b = self.make_traj(), self.make_traj()
        assert a.content_hash == b.content_hash
        c = self.make_traj()
        c.steps.reverse()
        c.finalize()
        assert a.content_hash != c.content_hash

    def test_empty_trajectory_status(self):
        t = Trajectory(task_key="s/t", kind="candidate").finalize()
        assert t.status == TrajStatus.EMPTY

    def test_missing_tokens_sum_over_present(self):
        t = self.make_traj()
        t.steps[0].tokens_in = None
        t.finalize()
        assert t.tokens_in == 20


class TestUlid:
    def test_sortable_and_unique(self):
        ids = [new_ulid() for _ in range(50)]
        assert len(set(ids)) == 50
        assert all(len(i) == 26 for i in ids)
