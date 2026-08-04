"""Step-pair similarity: sim(a, b) ∈ [0, 1].

Deterministic and structural only (v1) — no LLM calls, no network. The
weights follow PLAN.md §5: names gate tool comparisons, arguments refine
them; llm_calls compare by tool-intent rather than prompt text.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from offtrack.model import MASKED, Step, StepType, is_truncated_stub

STR_SIM_MAX_LEN = 2048
LONG_STR_FALLBACK = 0.3


def args_sim(a: Any, b: Any, rel_tol: float = 0.0) -> float:
    """Recursive structural similarity between two JSON values."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if a == MASKED or b == MASKED:
        return 1.0  # masked values never count against a match
    if is_truncated_stub(a) or is_truncated_stub(b):
        return _stub_sim(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return _dict_sim(a, b, rel_tol)
    if isinstance(a, list) and isinstance(b, list):
        return _list_sim(a, b, rel_tol)
    if isinstance(a, bool) or isinstance(b, bool):
        both_bool = isinstance(a, bool) and isinstance(b, bool)
        return 1.0 if both_bool and a == b else 0.0
    if isinstance(a, int | float) and isinstance(b, int | float):
        if a == b:
            return 1.0
        if rel_tol > 0:
            denom = max(abs(a), abs(b))
            if denom and abs(a - b) / denom <= rel_tol:
                return 1.0
        return 0.0
    if isinstance(a, str) and isinstance(b, str):
        return _str_sim(a, b)
    return 0.0  # type mismatch


def _dict_sim(a: dict[str, Any], b: dict[str, Any], rel_tol: float) -> float:
    keys_a, keys_b = set(a), set(b)
    if not keys_a and not keys_b:
        return 1.0
    jaccard = len(keys_a & keys_b) / len(keys_a | keys_b)
    shared = keys_a & keys_b
    if shared:
        value_sim = sum(args_sim(a[k], b[k], rel_tol) for k in shared) / len(shared)
    else:
        value_sim = 0.0
    return 0.5 * jaccard + 0.5 * value_sim


def _list_sim(a: list[Any], b: list[Any], rel_tol: float) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    elem = sum(args_sim(a[i], b[i], rel_tol) for i in range(n)) / n
    length_penalty = n / max(len(a), len(b))
    return elem * length_penalty


def _str_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) > STR_SIM_MAX_LEN or len(b) > STR_SIM_MAX_LEN:
        return LONG_STR_FALLBACK  # bounded cost; hash-equality handled above via ==
    return SequenceMatcher(None, a, b).ratio()


def _stub_sim(a: Any, b: Any) -> float:
    """Truncated payloads: sha equality wins; else shape + head."""
    if is_truncated_stub(a) and is_truncated_stub(b):
        if a.get("sha256") == b.get("sha256"):
            return 1.0
        shape = 1.0 if a.get("shape") == b.get("shape") else 0.0
        head = _str_sim(str(a.get("head", "")), str(b.get("head", "")))
        return 0.6 * shape + 0.4 * head
    return 0.3  # one truncated, one inline — weakly comparable


def _tool_intent(step: Step) -> list[str]:
    """The multiset of tool names an llm_call decided to emit."""
    if isinstance(step.args, dict):
        intent = step.args.get("tool_intent")
        if isinstance(intent, list):
            return sorted(str(t) for t in intent)
    return []


def step_sim(
    a: Step,
    b: Step,
    rel_tol: float = 0.0,
    model_exempt: bool = False,
    aliases: dict[str, str] | None = None,
) -> float:
    """Similarity between two steps of the SAME comparison universe.

    model_exempt: True when the candidate run intentionally uses a different
    model (the bump is the experiment, not a divergence).
    """
    if a.type != b.type:
        return 0.0

    if a.type == StepType.TOOL_CALL:
        name_a = (aliases or {}).get(a.name, a.name)
        name_b = (aliases or {}).get(b.name, b.name)
        if name_a != name_b:
            return 0.0
        return 0.4 + 0.6 * args_sim(a.args, b.args, rel_tol)

    if a.type == StepType.LLM_CALL:
        intent_a, intent_b = _tool_intent(a), _tool_intent(b)
        if not intent_a and not intent_b:
            intent = 1.0  # both text-only turns
        else:
            multiset_a, multiset_b = set(intent_a), set(intent_b)
            union = multiset_a | multiset_b
            intent = len(multiset_a & multiset_b) / len(union) if union else 1.0
        if model_exempt:
            model = 1.0
        else:
            model = 1.0 if (a.model or "") == (b.model or "") else 0.0
        return 0.5 + 0.3 * intent + 0.2 * model

    if a.type == StepType.HANDOFF:
        return 1.0 if a.name == b.name else 0.2

    # FINAL_ANSWER: structural presence is the match; content divergence is
    # reported informationally (semantic equivalence is the Matcher extension).
    return 1.0
