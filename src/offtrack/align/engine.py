"""Needleman–Wunsch alignment over trajectory steps, first-divergence
localization, resync detection, and multi-variant baseline matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from offtrack.align.similarity import step_sim
from offtrack.model import Step, Trajectory

OpKind = Literal["pair", "missing_step", "extra_step"]

PAIR: Final[OpKind] = "pair"
GAP_A: Final[OpKind] = "missing_step"  # baseline-only (candidate lacks it)
GAP_B: Final[OpKind] = "extra_step"  # candidate-only

DEFAULT_GAP = -0.45
DEFAULT_DIVERGENCE_THRESHOLD = 0.85
MAX_FULL_STEPS = 500
BAND_WIDTH = 64
HARD_CAP = 2000
RESYNC_RUN = 2


@dataclass
class AlignOp:
    kind: OpKind
    a_idx: int | None  # baseline step idx
    b_idx: int | None  # candidate step idx
    sim: float | None = None  # for pairs


@dataclass
class Alignment:
    ops: list[AlignOp]
    score: float
    norm_score: float
    first_divergence: int | None  # index into ops, None = clean
    divergence_kind: str | None
    resync_op: int | None  # first op index of a ≥RESYNC_RUN clean run after divergence
    approximate: bool = False
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def is_divergent(self) -> bool:
        return self.first_divergence is not None


@dataclass
class VariantMatch:
    """Best-variant result for candidate vs a multi-recording baseline."""

    alignment: Alignment
    variant_index: int  # which distinct baseline variant matched best
    variant_count: int
    variant_seen: int  # how many recordings share the best variant


def _pair_score(sim: float) -> float:
    return 2.0 * sim - 1.0


def align(
    baseline: list[Step],
    candidate: list[Step],
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    gap: float = DEFAULT_GAP,
    rel_tol: float = 0.0,
    model_exempt: bool = False,
    aliases: dict[str, str] | None = None,
) -> Alignment:
    """Global alignment of candidate against one baseline variant."""
    warnings: list[str] = []
    a, b = baseline, candidate
    truncated = False
    if len(a) > HARD_CAP or len(b) > HARD_CAP:
        warnings.append(f"trajectory over {HARD_CAP} steps — compared first {HARD_CAP} only")
        a, b = a[:HARD_CAP], b[:HARD_CAP]
        truncated = True

    band: int | None = None
    approximate = False
    if max(len(a), len(b)) > MAX_FULL_STEPS:
        band = BAND_WIDTH
        approximate = True
        warnings.append(f"approximate alignment (banded, >{MAX_FULL_STEPS} steps)")

    memo: dict[tuple[str, str], float] = {}

    def sim(i: int, j: int) -> float:
        key = (a[i].content_hash, b[j].content_hash)
        if key not in memo:
            memo[key] = step_sim(
                a[i], b[j], rel_tol=rel_tol, model_exempt=model_exempt, aliases=aliases
            )
        return memo[key]

    ops = _needleman_wunsch(a, b, sim, gap, band)
    score = sum(
        _pair_score(op.sim) if op.kind == PAIR and op.sim is not None else gap for op in ops
    )
    denom = max(len(a), len(b), 1)
    norm_score = score / denom

    first_div: int | None = None
    div_kind: str | None = None
    for i, op in enumerate(ops):
        if op.kind != PAIR:
            first_div, div_kind = i, op.kind
            break
        if op.sim is not None and op.sim < threshold:
            first_div, div_kind = i, "changed_step"
            break

    resync: int | None = None
    if first_div is not None:
        run = 0
        for i in range(first_div + 1, len(ops)):
            op = ops[i]
            if op.kind == PAIR and op.sim is not None and op.sim >= threshold:
                run += 1
                if run >= RESYNC_RUN:
                    resync = i - run + 1
                    break
            else:
                run = 0

    return Alignment(
        ops=ops,
        score=score,
        norm_score=norm_score,
        first_divergence=first_div,
        divergence_kind=div_kind,
        resync_op=resync,
        approximate=approximate,
        truncated=truncated,
        warnings=warnings,
    )


def _needleman_wunsch(
    a: list[Step],
    b: list[Step],
    sim: object,
    gap: float,
    band: int | None,
) -> list[AlignOp]:
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return []
    NEG = float("-inf")

    def in_band(i: int, j: int) -> bool:
        return band is None or abs(i - j) <= band

    # score[i][j]: best score aligning a[:i] with b[:j]
    score = [[NEG] * (m + 1) for _ in range(n + 1)]
    score[0][0] = 0.0
    for i in range(1, n + 1):
        if in_band(i, 0):
            score[i][0] = i * gap
    for j in range(1, m + 1):
        if in_band(0, j):
            score[0][j] = j * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not in_band(i, j):
                continue
            best = NEG
            if score[i - 1][j - 1] != NEG:
                best = score[i - 1][j - 1] + _pair_score(sim(i - 1, j - 1))  # type: ignore[operator]
            if score[i - 1][j] != NEG:
                best = max(best, score[i - 1][j] + gap)
            if score[i][j - 1] != NEG:
                best = max(best, score[i][j - 1] + gap)
            score[i][j] = best

    # Traceback, preferring diagonal on ties for stable output.
    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        current = score[i][j]
        if (
            i > 0
            and j > 0
            and score[i - 1][j - 1] != NEG
            and abs(current - (score[i - 1][j - 1] + _pair_score(sim(i - 1, j - 1)))) < 1e-9  # type: ignore[operator]
        ):
            ops.append(AlignOp(PAIR, i - 1, j - 1, sim(i - 1, j - 1)))  # type: ignore[operator]
            i, j = i - 1, j - 1
        elif i > 0 and score[i - 1][j] != NEG and abs(current - (score[i - 1][j] + gap)) < 1e-9:
            ops.append(AlignOp(GAP_A, i - 1, None))
            i -= 1
        elif j > 0 and score[i][j - 1] != NEG and abs(current - (score[i][j - 1] + gap)) < 1e-9:
            ops.append(AlignOp(GAP_B, None, j - 1))
            j -= 1
        else:  # banded dead-end: force whichever move is available
            if i > 0:
                ops.append(AlignOp(GAP_A, i - 1, None))
                i -= 1
            else:
                ops.append(AlignOp(GAP_B, None, j - 1))
                j -= 1
    ops.reverse()
    return ops


def dedup_variants(recordings: list[Trajectory]) -> list[tuple[Trajectory, int]]:
    """Distinct baseline variants with their multiplicities, stable order."""
    variants: list[tuple[Trajectory, int]] = []
    index: dict[str, int] = {}
    for t in recordings:
        if t.content_hash in index:
            variants[index[t.content_hash]] = (
                variants[index[t.content_hash]][0],
                variants[index[t.content_hash]][1] + 1,
            )
        else:
            index[t.content_hash] = len(variants)
            variants.append((t, 1))
    return variants


def best_variant_match(
    baseline_recordings: list[Trajectory],
    candidate: Trajectory,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    gap: float = DEFAULT_GAP,
    rel_tol: float = 0.0,
    model_exempt: bool = False,
    aliases: dict[str, str] | None = None,
) -> VariantMatch:
    """Align candidate against every distinct baseline variant; best norm_score
    wins. Divergent only if the best variant diverges."""
    variants = dedup_variants(baseline_recordings)
    if not variants:
        raise ValueError("no baseline recordings to match against")
    best: VariantMatch | None = None
    for vi, (variant, seen) in enumerate(variants):
        alignment = align(
            variant.steps,
            candidate.steps,
            threshold=threshold,
            gap=gap,
            rel_tol=rel_tol,
            model_exempt=model_exempt,
            aliases=aliases,
        )
        match = VariantMatch(alignment, vi, len(variants), seen)
        if best is None or alignment.norm_score > best.alignment.norm_score:
            best = match
    assert best is not None
    return best
