"""Comparison orchestration: baselines + candidates → verdict document.

The verdict document (a plain dict, JSON-serializable) is the single canonical
output; every renderer (terminal, markdown, JSON) is a pure function over it.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from offtrack.align import best_variant_match
from offtrack.align.engine import VariantMatch
from offtrack.mask import MaskRule, mask_hash, masked_trajectory_steps, parse_rules
from offtrack.model import Step, Trajectory, TrajStatus
from offtrack.stats import (
    MetricComparison,
    VerdictInputs,
    aggregate_verdicts,
    behavioral_verdict,
    metric_verdict,
    suite_verdict,
)
from offtrack.suite import GlobalConfig, ResolvedTask

REPORT_VERSION = 1


def _prepare(traj: Trajectory, rules: list[MaskRule], ignore: list[str]) -> list[Step]:
    steps = [s for s in traj.steps if not any(fnmatch.fnmatch(s.name, g) for g in ignore)]
    return masked_trajectory_steps(steps, rules)


def _models_of(traj: Trajectory) -> set[str]:
    return {s.model for s in traj.steps if s.model}


def _step_summary(s: Step | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {
        "idx": s.idx,
        "type": s.type.value,
        "name": s.name,
        "args": s.args,
        "status": s.status.value,
    }


def _divergence_detail(
    match: VariantMatch, base_steps: list[Step], cand_steps: list[Step]
) -> dict[str, Any] | None:
    al = match.alignment
    if al.first_divergence is None:
        return None
    op = al.ops[al.first_divergence]
    base_step = base_steps[op.a_idx] if op.a_idx is not None else None
    cand_step = cand_steps[op.b_idx] if op.b_idx is not None else None
    # Context: the aligned pairs just before the divergence.
    context = []
    for prior in al.ops[max(0, al.first_divergence - 2) : al.first_divergence]:
        if prior.a_idx is not None:
            context.append(_step_summary(base_steps[prior.a_idx]))
    return {
        "op_index": al.first_divergence,
        "kind": al.divergence_kind,
        "baseline_step": _step_summary(base_step),
        "candidate_step": _step_summary(cand_step),
        "context_before": context,
        "resynced": al.resync_op is not None,
        "variant": {
            "index": match.variant_index,
            "count": match.variant_count,
            "seen": match.variant_seen,
        },
        "sim": op.sim,
    }


def compare_task(
    rt: ResolvedTask,
    config: GlobalConfig,
    baselines: list[Trajectory],
    candidates: list[Trajectory],
    baseline_config_hash: str | None = None,
    allow_stale: bool = False,
) -> dict[str, Any]:
    """Compare candidate runs for one task against its baseline recordings."""
    warnings: list[str] = []
    rules = parse_rules(rt.mask_config)
    mhash = mask_hash(rules)
    ignore = config.ignore_steps

    usable_baselines = [b for b in baselines if b.status not in (TrajStatus.EMPTY,)]
    prepared_base = [(b, _prepare(b, rules, ignore)) for b in usable_baselines]

    # Leave-one-out baseline self-divergence: variance measured, not assumed.
    k_b = 0
    if len(prepared_base) >= 2:
        for i, (traj_i, steps_i) in enumerate(prepared_base):
            others = [
                t.model_copy(update={"steps": s})
                for j, (t, s) in enumerate(prepared_base)
                if j != i
            ]
            probe = traj_i.model_copy(update={"steps": steps_i})
            m = best_variant_match(
                others,
                probe,
                threshold=config.align.divergence_threshold,
                rel_tol=config.align.rel_tol,
                aliases=config.align.aliases,
            )
            if m.alignment.is_divergent:
                k_b += 1

    base_models = (
        set().union(*(_models_of(b) for b in usable_baselines)) if usable_baselines else set()
    )

    # Candidates: crashes count per on_crash policy; EMPTY are error-samples.
    errors = sum(1 for c in candidates if c.status == TrajStatus.EMPTY)
    usable = [c for c in candidates if c.status != TrajStatus.EMPTY]
    k_c = 0
    n_c = 0
    worst_match: VariantMatch | None = None
    worst_base_steps: list[Step] = []
    worst_cand_steps: list[Step] = []
    crash_divergent = 0

    masked_variants = [t.model_copy(update={"steps": s}) for t, s in prepared_base]

    for cand in usable:
        cand_models = _models_of(cand)
        model_exempt = bool(base_models) and bool(cand_models) and base_models != cand_models
        cand_steps = _prepare(cand, rules, ignore)
        probe = cand.model_copy(update={"steps": cand_steps})

        crashed = cand.status in (TrajStatus.ERROR, TrajStatus.TIMEOUT, TrajStatus.PARTIAL)
        if crashed and config.on_crash == "exclude":
            warnings.append(
                f"attempt {cand.attempt}: {cand.status.value} — excluded per on_crash policy"
            )
            continue

        n_c += 1
        divergent: bool
        if not masked_variants:
            divergent = True
            match = None
        else:
            match = best_variant_match(
                masked_variants,
                probe,
                threshold=config.align.divergence_threshold,
                rel_tol=config.align.rel_tol,
                model_exempt=model_exempt,
                aliases=config.align.aliases,
            )
            divergent = match.alignment.is_divergent
            for w in match.alignment.warnings:
                if w not in warnings:
                    warnings.append(w)

        if crashed:
            divergent = True
            crash_divergent += 1
            warnings.append(
                f"attempt {cand.attempt}: agent {cand.status.value} after step "
                f"{len(cand.steps)} — counted as divergent (on_crash policy)"
            )

        if divergent:
            k_c += 1
            if match is not None and (
                worst_match is None or match.alignment.norm_score < worst_match.alignment.norm_score
            ):
                worst_match = match
                best_variant = masked_variants[0].steps
                for vi, (_t, s) in enumerate(prepared_base):
                    if vi == match.variant_index:
                        best_variant = s
                        break
                worst_base_steps = best_variant
                worst_cand_steps = cand_steps

    inputs = VerdictInputs(
        n_baseline=len(prepared_base),
        k_baseline=k_b,
        n_candidate=n_c,
        k_candidate=k_c,
        errors=errors,
        alpha=config.verdict.alpha,
        min_effect=config.verdict.min_effect,
        pass_bound=config.verdict.pass_bound,
        deterministic=config.verdict.deterministic,
    )
    behavioral = behavioral_verdict(inputs)
    warnings.extend(behavioral.warnings)

    # Stale-baseline cap: config drift means the comparison may be misleading.
    stale = (
        baseline_config_hash is not None
        and baseline_config_hash != rt.config_hash
        and not allow_stale
    )
    verdict_value = behavioral.verdict
    if stale and verdict_value in ("PASS", "FAIL"):
        warnings.append(
            "baseline recorded against a different task config — verdict capped at "
            "INCONCLUSIVE. Re-record with `offtrack record`, or pass --allow-stale."
        )
        verdict_value = "INCONCLUSIVE"

    metrics: list[dict[str, Any]] = []
    metric_results = []
    for name, rule in config.metrics.items():
        attr = {"cost": "cost_usd", "tokens": "tokens_in", "latency": "wall_ms"}.get(name)
        if attr is None:
            continue
        base_vals = [getattr(b, attr) for b in usable_baselines if getattr(b, attr) is not None]
        cand_vals = [getattr(c, attr) for c in usable if getattr(c, attr) is not None]
        mr = metric_verdict(
            MetricComparison(name, base_vals, cand_vals, rule.threshold, rule.action),  # type: ignore[arg-type]
            alpha=config.verdict.alpha,
        )
        metric_results.append(mr)
        metrics.append(
            {
                "name": mr.name,
                "verdict": mr.verdict,
                "median_baseline": mr.median_baseline,
                "median_candidate": mr.median_candidate,
                "change": mr.change,
                "p_value": mr.p_value,
                "reason": mr.reason,
            }
        )

    task_verdict = aggregate_verdicts(verdict_value, metric_results)

    detail = None
    if worst_match is not None:
        detail = _divergence_detail(worst_match, worst_base_steps, worst_cand_steps)

    return {
        "task_key": rt.task_key,
        "verdict": task_verdict,
        "behavioral": {
            "verdict": verdict_value,
            "reason": behavioral.reason,
            "rate_baseline": behavioral.rate_baseline,
            "rate_candidate": behavioral.rate_candidate,
            "p_value": behavioral.p_value,
            "upper_bound": behavioral.upper_bound_candidate,
            "prescription": behavioral.prescription,
            "n_baseline": len(prepared_base),
            "n_candidate": n_c,
            "k_candidate": k_c,
        },
        "first_divergence": detail,
        "metrics": metrics,
        "warnings": warnings,
        "mask_hash": mhash,
        "stale_baseline": stale,
    }


def build_report(run_id: str, task_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "verdict": suite_verdict([t["verdict"] for t in task_reports]),
        "tasks": task_reports,
    }
