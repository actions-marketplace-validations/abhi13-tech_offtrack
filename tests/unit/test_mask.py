"""Masking DSL: parsing, application, builtins, suggestions."""

from __future__ import annotations

import pytest

from offtrack.mask import (
    MaskConfigError,
    apply_masks,
    mask_hash,
    masked_step,
    parse_rules,
    suggest_masks,
)
from offtrack.model import MASKED, Step, StepType


def rules_for(config):
    return parse_rules(config)


NO_BUILTINS = {"builtin": []}


class TestParsing:
    def test_defaults_include_builtins(self):
        rules = parse_rules(None)
        assert len(rules) == 3

    def test_unknown_builtin_rejected(self):
        with pytest.raises(MaskConfigError, match="unknown builtin"):
            parse_rules({"builtin": ["nope"]})

    def test_bad_path_rejected_with_supported_syntax(self):
        with pytest.raises(MaskConfigError, match="Supported"):
            parse_rules({"builtin": [], "rules": [{"path": "$.args[?(@.x)]"}]})

    def test_path_must_start_with_dollar(self):
        with pytest.raises(MaskConfigError, match="must start"):
            parse_rules({"builtin": [], "rules": [{"path": "args.x"}]})

    def test_unknown_action_rejected(self):
        with pytest.raises(MaskConfigError, match="unknown mask action"):
            parse_rules({"builtin": [], "rules": [{"field": "x", "action": "explode"}]})

    def test_rule_needs_selector(self):
        with pytest.raises(MaskConfigError, match="needs"):
            parse_rules({"builtin": [], "rules": [{"action": "drop"}]})

    def test_mask_hash_stable_and_sensitive(self):
        a = parse_rules({"builtin": [], "rules": [{"field": "x"}]})
        b = parse_rules({"builtin": [], "rules": [{"field": "x"}]})
        c = parse_rules({"builtin": [], "rules": [{"field": "y"}]})
        assert mask_hash(a) == mask_hash(b) != mask_hash(c)


class TestApplication:
    def test_field_any_depth(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"field": "session_id"}]})
        out = apply_masks({"a": {"session_id": "s1", "keep": 1}, "session_id": "s2"}, rules)
        assert out == {"a": {"session_id": MASKED, "keep": 1}, "session_id": MASKED}

    def test_path_exact(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"path": "$.args.request_id"}]})
        out = apply_masks({"args": {"request_id": "r1", "keep": 2}, "result": None}, rules)
        assert out["args"] == {"request_id": MASKED, "keep": 2}

    def test_path_wildcard_and_index(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"path": "$.args.hits[*].score"}]})
        out = apply_masks({"args": {"hits": [{"score": 1, "id": "a"}, {"score": 2}]}}, rules)
        assert out["args"]["hits"] == [{"score": MASKED, "id": "a"}, {"score": MASKED}]

    def test_recursive_descend(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"path": "$..trace_id"}]})
        out = apply_masks({"a": {"b": {"trace_id": "t"}}, "trace_id": "u"}, rules)
        assert out == {"a": {"b": {"trace_id": MASKED}}, "trace_id": MASKED}

    def test_step_glob_scoping(self):
        rules = rules_for(
            {**NO_BUILTINS, "rules": [{"step": "search_*", "path": "$.args.session"}]}
        )
        payload = {"args": {"session": "x"}}
        assert apply_masks(payload, rules, step_name="search_web")["args"]["session"] == MASKED
        assert apply_masks(payload, rules, step_name="lookup")["args"]["session"] == "x"

    def test_round_action(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"path": "$.args.score", "action": "round:2"}]})
        assert apply_masks({"args": {"score": 0.98765}}, rules)["args"]["score"] == 0.99

    def test_hash_action_deterministic(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"field": "key", "action": "hash"}]})
        a = apply_masks({"key": "secret"}, rules)
        b = apply_masks({"key": "secret"}, rules)
        assert a == b and a["key"].startswith("__hash_")

    def test_normalize_ws(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"field": "text", "action": "normalize_ws"}]})
        assert apply_masks({"text": "  a \n b  "}, rules)["text"] == "a b"


class TestBuiltins:
    def test_uuid_masked(self):
        rules = parse_rules({})
        out = apply_masks({"id": "123e4567-e89b-42d3-a456-426614174000", "keep": "hello"}, rules)
        assert out["id"] == MASKED and out["keep"] == "hello"

    def test_iso_timestamp_masked(self):
        out = apply_masks({"at": "2026-08-04T12:00:00Z"}, parse_rules({}))
        assert out["at"] == MASKED

    def test_epoch_masked(self):
        out = apply_masks({"ts": 1754300000, "n": 42}, parse_rules({}))
        assert out["ts"] == MASKED and out["n"] == 42


class TestMaskedStep:
    def make(self, args):
        return Step(idx=0, type=StepType.TOOL_CALL, name="t", args=args).finalize()

    def test_masked_steps_hash_equal(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"field": "request_id"}]})
        a = masked_step(self.make({"q": "x", "request_id": "r1"}), rules)
        b = masked_step(self.make({"q": "x", "request_id": "r2"}), rules)
        assert a.content_hash == b.content_hash

    def test_original_untouched(self):
        rules = rules_for({**NO_BUILTINS, "rules": [{"field": "q"}]})
        orig = self.make({"q": "x"})
        masked_step(orig, rules)
        assert orig.args == {"q": "x"}

    def test_no_rules_returns_same_object(self):
        s = self.make({"q": "x"})
        assert masked_step(s, []) is s


class TestSuggest:
    def steps(self, request_id, q="search terms"):
        return [
            Step(
                idx=0,
                type=StepType.TOOL_CALL,
                name="search",
                args={"q": q, "request_id": request_id},
            ).finalize()
        ]

    def test_volatile_field_suggested(self):
        suggestions = suggest_masks([self.steps("r1"), self.steps("r2"), self.steps("r3")])
        assert {"step": "search", "path": "$.args.request_id"} in suggestions

    def test_stable_fields_not_suggested(self):
        suggestions = suggest_masks([self.steps("r1"), self.steps("r2")])
        assert all(s["path"] != "$.args.q" for s in suggestions)

    def test_single_recording_no_suggestions(self):
        assert suggest_masks([self.steps("r1")]) == []
