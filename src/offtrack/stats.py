"""Statistical verdicts: PASS / FAIL / INCONCLUSIVE / ERROR.

Exact methods only, stdlib only (math.comb + bisection — no scipy):
- FAIL: one-sided Fisher's exact test AND a minimum-effect floor.
- PASS: exact Clopper–Pearson one-sided upper bound on the candidate
  divergence rate, allowing for the baseline's own measured variance.
- Otherwise INCONCLUSIVE, with a prescription for how many more runs
  could resolve it. Honesty is the point: the tool never claims
  sensitivity it doesn't have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Literal

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE", "ERROR"]

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_EFFECT = 0.30
# 0.46, not 0.45: the exact 95% upper bound for 0/5 is 1-0.05^(1/5) ≈ 0.4507,
# and the default recording depth (5 runs, 0 divergent) must PASS by design.
DEFAULT_PASS_BOUND = 0.46
MAX_PRESCRIPTION = 50


def fisher_exact_one_sided(k_b: int, n_b: int, k_c: int, n_c: int) -> float:
    """P(candidate divergences ≥ k_c | margins fixed), hypergeometric tail.

    Tests whether the candidate's divergence count is surprisingly high
    given the pooled rate. Small p → candidate diverges more than baseline.
    """
    total = n_b + n_c
    successes = k_b + k_c  # total divergent
    p = 0.0
    denom = comb(total, n_c)
    upper = min(successes, n_c)
    for k in range(k_c, upper + 1):
        if successes - k > n_b:
            continue
        p += comb(successes, k) * comb(total - successes, n_c - k) / denom
    return min(p, 1.0)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return min(total, 1.0)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound on a binomial proportion (bisection)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - confidence
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # Upper bound p_u satisfies P(X ≤ k | p_u) = alpha.
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def permutation_test_median(a: list[float], b: list[float], max_exact: int = 200_000) -> float:
    """One-sided exact permutation test: is median(b) surprisingly high vs a?

    Enumerates all labelings when C(n_a+n_b, n_b) ≤ max_exact; otherwise
    falls back to a deterministic stride subsample of combinations.
    """
    from itertools import combinations
    from statistics import median

    pooled = a + b
    n_b = len(b)
    observed = median(b) - median(a)
    idx = range(len(pooled))
    total_combos = comb(len(pooled), n_b)

    def diff_for(sel: tuple[int, ...]) -> float:
        group_b = [pooled[i] for i in sel]
        group_a = [pooled[i] for i in idx if i not in set(sel)]
        return median(group_b) - median(group_a)

    if total_combos <= max_exact:
        combos = combinations(idx, n_b)
        count = ge = 0
        for sel in combos:
            count += 1
            if diff_for(sel) >= observed - 1e-12:
                ge += 1
        return ge / count if count else 1.0
    # Deterministic stride subsample (no RNG — reproducible verdicts).
    stride = total_combos // max_exact + 1
    count = ge = 0
    for i, sel in enumerate(combinations(idx, n_b)):
        if i % stride:
            continue
        count += 1
        if diff_for(sel) >= observed - 1e-12:
            ge += 1
    return ge / count if count else 1.0


@dataclass
class VerdictInputs:
    n_baseline: int  # usable baseline recordings
    k_baseline: int  # of those, divergent under leave-one-out self-alignment
    n_candidate: int  # usable candidate attempts
    k_candidate: int  # of those, divergent vs baseline
    errors: int = 0  # attempts with no signal (never counted either side)
    alpha: float = DEFAULT_ALPHA
    min_effect: float = DEFAULT_MIN_EFFECT
    pass_bound: float = DEFAULT_PASS_BOUND
    deterministic: bool = False


@dataclass
class VerdictResult:
    verdict: Verdict
    p_value: float | None
    rate_baseline: float | None
    rate_candidate: float | None
    upper_bound_candidate: float | None
    reason: str
    prescription: int | None = None  # additional runs that could resolve INCONCLUSIVE
    warnings: list[str] = field(default_factory=list)


def behavioral_verdict(v: VerdictInputs) -> VerdictResult:
    warnings: list[str] = []
    if v.n_candidate == 0:
        return VerdictResult(
            "ERROR",
            None,
            None,
            None,
            None,
            "no usable candidate attempts (all failed to produce traces)",
            warnings=warnings,
        )
    if v.n_baseline == 0:
        return VerdictResult(
            "ERROR",
            None,
            None,
            None,
            None,
            "no usable baseline recordings",
            warnings=warnings,
        )
    if v.errors:
        warnings.append(f"{v.errors} attempt(s) produced no signal and were excluded")

    p_b = v.k_baseline / v.n_baseline
    p_c = v.k_candidate / v.n_candidate

    # Deterministic fast path: agent proven stable, any divergence is real.
    if v.deterministic and v.k_baseline == 0 and v.n_baseline >= 3 and v.k_candidate > 0:
        return VerdictResult(
            "FAIL",
            None,
            p_b,
            p_c,
            None,
            f"deterministic mode: baseline stable over {v.n_baseline} recordings, "
            f"candidate diverged in {v.k_candidate}/{v.n_candidate}",
            warnings=warnings,
        )

    p = fisher_exact_one_sided(v.k_baseline, v.n_baseline, v.k_candidate, v.n_candidate)
    effect = p_c - p_b
    if p < v.alpha and effect >= v.min_effect:
        return VerdictResult(
            "FAIL",
            p,
            p_b,
            p_c,
            None,
            f"divergence rate rose {p_b:.0%} → {p_c:.0%} "
            f"(Fisher exact p={p:.3f}, effect {effect:+.0%})",
            warnings=warnings,
        )

    ub = clopper_pearson_upper(v.k_candidate, v.n_candidate)
    allowed = max(p_b + v.pass_bound, v.pass_bound)
    if ub <= allowed:
        return VerdictResult(
            "PASS",
            p,
            p_b,
            p_c,
            ub,
            f"no excess divergence in {v.n_candidate - v.k_candidate}/{v.n_candidate} runs "
            f"(95% upper bound on divergence rate: {ub:.0%})",
            warnings=warnings,
        )

    prescription = _prescribe(v)
    reason = (
        f"cannot separate signal from noise at n={v.n_candidate} "
        f"(divergence {p_b:.0%} → {p_c:.0%}, p={p:.3f}, 95% UB {ub:.0%})"
    )
    return VerdictResult(
        "INCONCLUSIVE", p, p_b, p_c, ub, reason, prescription=prescription, warnings=warnings
    )


def _prescribe(v: VerdictInputs) -> int | None:
    """Smallest additional n_candidate that could resolve, if rates persist."""
    p_b = v.k_baseline / v.n_baseline
    p_c = v.k_candidate / v.n_candidate
    for extra in range(1, MAX_PRESCRIPTION + 1):
        n = v.n_candidate + extra
        k = round(p_c * n)
        p = fisher_exact_one_sided(v.k_baseline, v.n_baseline, k, n)
        effect = (k / n) - p_b
        if p < v.alpha and effect >= v.min_effect:
            return extra
        ub = clopper_pearson_upper(k, n)
        if ub <= max(p_b + v.pass_bound, v.pass_bound):
            return extra
    return None


@dataclass
class MetricComparison:
    name: str
    baseline: list[float]
    candidate: list[float]
    threshold: float  # relative increase that matters (0.2 = +20%)
    action: Literal["fail", "warn"]


@dataclass
class MetricResult:
    name: str
    verdict: Literal["PASS", "FAIL", "WARN", "SKIP"]
    median_baseline: float | None
    median_candidate: float | None
    change: float | None  # relative
    p_value: float | None
    reason: str


def metric_verdict(m: MetricComparison, alpha: float = DEFAULT_ALPHA) -> MetricResult:
    from statistics import median

    base = [x for x in m.baseline if x is not None]
    cand = [x for x in m.candidate if x is not None]
    if not base or not cand:
        return MetricResult(m.name, "SKIP", None, None, None, None, "insufficient data")
    mb, mc = median(base), median(cand)
    if mb == 0:
        return MetricResult(m.name, "SKIP", mb, mc, None, None, "baseline median is zero")
    change = (mc - mb) / mb
    if change <= m.threshold:
        return MetricResult(
            m.name, "PASS", mb, mc, change, None, f"median {change:+.0%} within threshold"
        )
    # Median rose past threshold — is it statistically supported?
    min_possible_p = 1.0 / comb(len(base) + len(cand), len(cand))
    if min_possible_p > alpha:
        return MetricResult(
            m.name,
            "WARN",
            mb,
            mc,
            change,
            None,
            f"suggestive ({change:+.0%} median) — not gating at n={len(cand)}",
        )
    p = permutation_test_median(base, cand)
    if p < alpha:
        verdict: Literal["FAIL", "WARN"] = "FAIL" if m.action == "fail" else "WARN"
        return MetricResult(
            m.name, verdict, mb, mc, change, p, f"median {change:+.0%} (permutation p={p:.3f})"
        )
    return MetricResult(
        m.name, "PASS", mb, mc, change, p, f"median {change:+.0%} not significant (p={p:.3f})"
    )


def aggregate_verdicts(behavioral: Verdict, metrics: list[MetricResult]) -> Verdict:
    """Task verdict = worst of behavioral and gating metric outcomes."""
    if behavioral == "ERROR":
        return "ERROR"
    if behavioral == "FAIL" or any(m.verdict == "FAIL" for m in metrics):
        return "FAIL"
    if behavioral == "INCONCLUSIVE":
        return "INCONCLUSIVE"
    return "PASS"


def suite_verdict(task_verdicts: list[Verdict]) -> Verdict:
    """Suite = FAIL > ERROR > INCONCLUSIVE > PASS."""
    if not task_verdicts:
        return "ERROR"
    if "FAIL" in task_verdicts:
        return "FAIL"
    if "ERROR" in task_verdicts:
        return "ERROR"
    if "INCONCLUSIVE" in task_verdicts:
        return "INCONCLUSIVE"
    return "PASS"
