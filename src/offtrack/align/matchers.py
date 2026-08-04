"""Matcher chain: pluggable step-similarity, run before the structural default.

    class Matcher(Protocol):
        def similarity(self, a: Step, b: Step, ctx: AlignContext) -> float | None: ...

Matchers run in order; the first non-None wins, None defers down the chain.
The chain always ends with the structural matcher, so a custom matcher only
needs to handle the cases it improves on.

Final-answer comparison modes (config.align.final_answer):
- "presence" (default): any final answer matches any other — content
  divergence is reported informationally only (v1 behavior).
- "lexical": local deterministic token-cosine + sequence blend. Offline, $0,
  reproducible.
- "embedding": OpenAI embeddings (requires offtrack[openai] + OPENAI_API_KEY),
  cosine similarity, cached in SQLite by content hash — each distinct answer
  is embedded once, ever. Degrades to lexical with a warning when unavailable:
  degrade, never lie.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

from offtrack.align.similarity import step_sim
from offtrack.model import Step, StepType


@dataclass
class AlignContext:
    rel_tol: float = 0.0
    model_exempt: bool = False
    aliases: dict[str, str] | None = None
    final_answer_mode: str = "presence"  # presence | lexical | embedding
    final_answer_threshold: float = 0.7
    warnings: list[str] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    embedder: Any = None  # injected: Callable[[str], list[float]] | None


class Matcher(Protocol):
    def similarity(self, a: Step, b: Step, ctx: AlignContext) -> float | None: ...


class StructuralMatcher:
    """The v1 default: never defers."""

    def similarity(self, a: Step, b: Step, ctx: AlignContext) -> float | None:
        return step_sim(
            a,
            b,
            rel_tol=ctx.rel_tol,
            model_exempt=ctx.model_exempt,
            aliases=ctx.aliases,
        )


_WORD_RE = re.compile(r"[a-z0-9]+")


def _answer_text(step: Step) -> str:
    if isinstance(step.result, str):
        return step.result
    if step.result is None:
        return ""
    import json

    return json.dumps(step.result, sort_keys=True, default=str)


def lexical_similarity(a: str, b: str) -> float:
    """Deterministic local text similarity: token cosine blended with a
    sequence ratio, robust to reordering and paraphrase-lite rewording."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    ta, tb = Counter(_WORD_RE.findall(a.lower())), Counter(_WORD_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    shared = set(ta) & set(tb)
    dot = sum(ta[w] * tb[w] for w in shared)
    norm = math.sqrt(sum(v * v for v in ta.values())) * math.sqrt(sum(v * v for v in tb.values()))
    cosine = dot / norm if norm else 0.0
    seq = SequenceMatcher(None, a[:2048], b[:2048]).ratio()
    return 0.7 * cosine + 0.3 * seq


class LexicalAnswerMatcher:
    """Compares final_answer content locally; defers on everything else.

    Similarity at or above ctx.final_answer_threshold reads as a clean match
    (1.0); below it, the raw score flows through and flags a changed_step.
    """

    def similarity(self, a: Step, b: Step, ctx: AlignContext) -> float | None:
        if a.type != StepType.FINAL_ANSWER or b.type != StepType.FINAL_ANSWER:
            return None
        sim = lexical_similarity(_answer_text(a), _answer_text(b))
        # Below threshold: floor at 0.1 so the answers still PAIR in the
        # alignment (a changed_step reads better than missing+extra steps;
        # two gaps cost -0.9, so any pair score above that wins).
        return 1.0 if sim >= ctx.final_answer_threshold else max(sim, 0.1)


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=False))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


class EmbeddingAnswerMatcher:
    """Embedding cosine over final answers; caches by step content hash.

    ctx.embedder is injected (a callable str -> vector). When absent, this
    matcher defers — the chain's LexicalAnswerMatcher then handles the step,
    and a warning explains the degradation once per comparison run.
    """

    def similarity(self, a: Step, b: Step, ctx: AlignContext) -> float | None:
        if a.type != StepType.FINAL_ANSWER or b.type != StepType.FINAL_ANSWER:
            return None
        if ctx.embedder is None:
            msg = (
                "embedding matcher unavailable (no embedder — set OPENAI_API_KEY "
                "and install offtrack[openai]) — final answers compared lexically"
            )
            if msg not in ctx.warnings:
                ctx.warnings.append(msg)
            return None  # defer to lexical
        va = self._embed(a, ctx)
        vb = self._embed(b, ctx)
        if va is None or vb is None:
            return None
        sim = cosine(va, vb)
        return 1.0 if sim >= ctx.final_answer_threshold else max(sim, 0.1)

    def _embed(self, step: Step, ctx: AlignContext) -> list[float] | None:
        key = f"emb:{step.content_hash}:{hash(_answer_text(step)) & 0xFFFFFFFF}"
        if key in ctx.cache:
            cached: list[float] | None = ctx.cache[key]
            return cached
        vector: list[float] | None
        try:
            vector = list(ctx.embedder(_answer_text(step)))
        except Exception as e:  # network/API failure degrades, never crashes
            msg = f"embedding failed ({type(e).__name__}) — compared lexically"
            if msg not in ctx.warnings:
                ctx.warnings.append(msg)
            vector = None
        ctx.cache[key] = vector
        return vector


def default_openai_embedder() -> Any:
    """Build an OpenAI embedder if the SDK and key are available, else None."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI()

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
        return list(response.data[0].embedding)

    return embed


def build_chain(ctx: AlignContext) -> list[Matcher]:
    """Compose the matcher chain for a comparison run from its context."""
    chain: list[Matcher] = []
    if ctx.final_answer_mode == "embedding":
        chain.append(EmbeddingAnswerMatcher())
        chain.append(LexicalAnswerMatcher())  # the documented degradation path
    elif ctx.final_answer_mode == "lexical":
        chain.append(LexicalAnswerMatcher())
    chain.append(StructuralMatcher())
    return chain


def chain_similarity(chain: list[Matcher], a: Step, b: Step, ctx: AlignContext) -> float:
    for matcher in chain:
        sim = matcher.similarity(a, b, ctx)
        if sim is not None:
            return sim
    return 0.0  # unreachable: StructuralMatcher never defers
