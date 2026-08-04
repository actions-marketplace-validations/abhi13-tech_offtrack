"""Alignment engine: similarity, NW, first divergence, variants, guardrails."""

from __future__ import annotations

from offtrack.align import align, args_sim, best_variant_match, step_sim
from offtrack.align.engine import GAP_A, GAP_B, PAIR
from offtrack.model import Step, StepType, Trajectory


def tool(name: str, args=None, idx=0, **kw) -> Step:
    return Step(idx=idx, type=StepType.TOOL_CALL, name=name, args=args or {}, **kw).finalize()


def llm(model="m1", intent=None, idx=0) -> Step:
    return Step(
        idx=idx,
        type=StepType.LLM_CALL,
        name=model,
        model=model,
        args={"tool_intent": intent or []},
    ).finalize()


def final(text="done", idx=0) -> Step:
    return Step(idx=idx, type=StepType.FINAL_ANSWER, name="final", result=text).finalize()


def traj(steps, key="s/t", kind="baseline") -> Trajectory:
    return Trajectory(task_key=key, kind=kind, steps=list(steps)).finalize()


class TestArgsSim:
    def test_identical(self):
        assert args_sim({"a": 1}, {"a": 1}) == 1.0

    def test_disjoint_keys(self):
        assert args_sim({"a": 1}, {"b": 2}) == 0.0

    def test_shared_keys_different_values(self):
        s = args_sim({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert 0.5 < s < 1.0

    def test_none_handling(self):
        assert args_sim(None, None) == 1.0
        assert args_sim({"a": 1}, None) == 0.0

    def test_masked_always_matches(self):
        assert args_sim("__masked__", "anything at all") == 1.0

    def test_numbers_rel_tol(self):
        assert args_sim(100.0, 101.0) == 0.0
        assert args_sim(100.0, 101.0, rel_tol=0.05) == 1.0

    def test_bool_not_conflated_with_int(self):
        assert args_sim(True, 1) == 0.0

    def test_string_similarity(self):
        assert args_sim("hello world", "hello world") == 1.0
        assert 0.5 < args_sim("hello world", "hello worlds") < 1.0

    def test_lists_length_penalty(self):
        assert args_sim([1, 2, 3], [1, 2, 3]) == 1.0
        assert args_sim([1, 2, 3, 4], [1, 2]) < 0.6


class TestStepSim:
    def test_different_types_zero(self):
        assert step_sim(tool("x"), llm()) == 0.0

    def test_tool_name_gate(self):
        assert step_sim(tool("lookup", {"id": 1}), tool("refund", {"id": 1})) == 0.0

    def test_same_tool_same_args(self):
        assert step_sim(tool("lookup", {"id": 1}), tool("lookup", {"id": 1})) == 1.0

    def test_same_tool_different_args_partial(self):
        s = step_sim(tool("lookup", {"id": 1}), tool("lookup", {"id": 2}))
        assert 0.4 <= s < 1.0

    def test_alias_map(self):
        s = step_sim(
            tool("search_v2", {"q": "x"}),
            tool("search", {"q": "x"}),
            aliases={"search_v2": "search"},
        )
        assert s == 1.0

    def test_llm_intent_match(self):
        assert step_sim(llm("m", ["lookup"]), llm("m", ["lookup"])) == 1.0

    def test_llm_intent_mismatch(self):
        s = step_sim(llm("m", ["lookup"]), llm("m", ["refund"]))
        assert s == 0.7  # 0.5 + 0.3*0 + 0.2*1

    def test_model_exemption(self):
        a, b = llm("gpt-4o", ["x"]), llm("gpt-5", ["x"])
        assert step_sim(a, b) == 0.8  # model term lost
        assert step_sim(a, b, model_exempt=True) == 1.0

    def test_final_answer_presence_match(self):
        assert step_sim(final("yes"), final("completely different")) == 1.0


class TestAlign:
    def careful(self):
        return [
            llm("m", ["lookup_order"], 0),
            tool("lookup_order", {"order_id": "T1"}, 1),
            llm("m", ["check_refund_policy"], 2),
            tool("check_refund_policy", {"amount_usd": 842.0}, 3),
            llm("m", ["escalate"], 4),
            tool("escalate", {"order_id": "T1", "reason": "over limit"}, 5),
            llm("m", [], 6),
            final("escalated", 7),
        ]

    def sloppy(self):
        return [
            llm("m", ["lookup_order"], 0),
            tool("lookup_order", {"order_id": "T1"}, 1),
            llm("m", ["issue_refund"], 2),
            tool("issue_refund", {"order_id": "T1", "amount_usd": 842.0}, 3),
            llm("m", [], 4),
            final("refunded", 5),
        ]

    def test_self_alignment_clean(self):
        a = align(self.careful(), self.careful())
        assert not a.is_divergent
        assert all(op.kind == PAIR and op.sim == 1.0 for op in a.ops)
        assert a.norm_score == 1.0

    def test_marquee_regression_first_divergence(self):
        """The demo scenario: sloppy persona skips the policy check."""
        a = align(self.careful(), self.sloppy())
        assert a.is_divergent
        # First two steps (llm+lookup) align cleanly; divergence at op 2.
        assert a.first_divergence == 2
        first_op = a.ops[a.first_divergence]
        assert first_op.kind in ("changed_step", PAIR, GAP_A) or a.divergence_kind in (
            "changed_step",
            "missing_step",
        )

    def test_extra_retry_is_extra_step(self):
        base = [tool("search", {"q": "x"}, 0), final("ok", 1)]
        cand = [tool("search", {"q": "x"}, 0), tool("search", {"q": "x"}, 1), final("ok", 2)]
        a = align(base, cand)
        assert a.is_divergent and a.divergence_kind == "extra_step"
        assert a.ops[a.first_divergence].kind == GAP_B

    def test_missing_step(self):
        base = [tool("a", {}, 0), tool("b", {}, 1), final("ok", 2)]
        cand = [tool("a", {}, 0), final("ok", 1)]
        a = align(base, cand)
        assert a.divergence_kind == "missing_step"
        assert a.ops[a.first_divergence].kind == GAP_A

    def test_resync_detection(self):
        base = [tool("a", {}, 0), tool("b", {}, 1), tool("c", {}, 2), tool("d", {}, 3)]
        cand = [tool("a", {}, 0), tool("X", {}, 1), tool("c", {}, 2), tool("d", {}, 3)]
        a = align(base, cand)
        assert a.is_divergent and a.first_divergence is not None
        assert a.resync_op is not None and a.resync_op > a.first_divergence

    def test_empty_candidate_all_gaps(self):
        a = align(self.careful(), [])
        assert a.first_divergence == 0
        assert all(op.kind == GAP_A for op in a.ops)

    def test_both_empty_clean(self):
        a = align([], [])
        assert not a.is_divergent and a.ops == []

    def test_changed_args_below_threshold(self):
        base = [tool("refund", {"order": "T1", "amount": 100, "currency": "USD"}, 0)]
        cand = [tool("refund", {"order": "T2", "amount": 999, "currency": "EUR"}, 0)]
        a = align(base, cand)
        assert a.is_divergent and a.divergence_kind == "changed_step"

    def test_banded_alignment_flag(self):
        base = [tool(f"t{i}", {"i": i}, i) for i in range(520)]
        a = align(base, base)
        assert a.approximate and not a.is_divergent
        assert any("banded" in w for w in a.warnings)

    def test_hard_cap(self):
        base = [tool("t", {"i": i}, i) for i in range(2100)]
        a = align(base, base)
        assert a.truncated


class TestVariants:
    def test_best_variant_wins(self):
        careful = TestAlign().careful
        v1 = traj(careful())
        # variant 2: different path recorded twice
        alt_steps = [
            llm("m", ["lookup_order"], 0),
            tool("lookup_order", {"order_id": "T1"}, 1),
            llm("m", [], 2),
            final("cannot find", 3),
        ]
        v2a, v2b = traj(alt_steps), traj(alt_steps)
        cand = traj(alt_steps, kind="candidate")
        m = best_variant_match([v1, v2a, v2b], cand)
        assert not m.alignment.is_divergent
        assert m.variant_count == 2 and m.variant_seen == 2

    def test_divergent_only_if_best_diverges(self):
        base = traj(TestAlign().careful())
        cand = traj(TestAlign().sloppy(), kind="candidate")
        m = best_variant_match([base], cand)
        assert m.alignment.is_divergent
