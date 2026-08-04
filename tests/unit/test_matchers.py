"""Matcher chain: lexical/embedding final-answer comparison, defer semantics."""

from __future__ import annotations

from offtrack.align import align
from offtrack.align.matchers import (
    AlignContext,
    EmbeddingAnswerMatcher,
    LexicalAnswerMatcher,
    build_chain,
    chain_similarity,
    lexical_similarity,
)
from offtrack.model import Step, StepType


def final(text, idx=0):
    return Step(idx=idx, type=StepType.FINAL_ANSWER, name="final", result=text).finalize()


def tool(name, idx=0):
    return Step(idx=idx, type=StepType.TOOL_CALL, name=name, args={}).finalize()


class TestLexicalSimilarity:
    def test_identical(self):
        assert lexical_similarity("refund issued", "refund issued") == 1.0

    def test_reworded_same_content_high(self):
        a = "Your refund of $842 has been escalated to a human agent for approval."
        b = "I escalated the $842 refund to a human agent who must approve it."
        assert lexical_similarity(a, b) > 0.5

    def test_different_content_low(self):
        a = "Your refund has been escalated for approval."
        b = "The weather in Paris is sunny with light winds today."
        assert lexical_similarity(a, b) < 0.3

    def test_empty(self):
        assert lexical_similarity("", "anything") == 0.0


class TestChain:
    def test_lexical_defers_on_non_final(self):
        ctx = AlignContext(final_answer_mode="lexical")
        assert LexicalAnswerMatcher().similarity(tool("a"), tool("a"), ctx) is None

    def test_chain_order_and_fallthrough(self):
        ctx = AlignContext(final_answer_mode="embedding")  # no embedder
        chain = build_chain(ctx)
        assert [type(m).__name__ for m in chain] == [
            "EmbeddingAnswerMatcher",
            "LexicalAnswerMatcher",
            "StructuralMatcher",
        ]
        # Final answers fall through embedding (no embedder) to lexical.
        sim = chain_similarity(chain, final("same text"), final("same text"), ctx)
        assert sim == 1.0
        assert any("compared lexically" in w for w in ctx.warnings)

    def test_presence_mode_is_structural_only(self):
        ctx = AlignContext(final_answer_mode="presence")
        chain = build_chain(ctx)
        assert [type(m).__name__ for m in chain] == ["StructuralMatcher"]
        # v1 semantics: any answer matches any answer
        assert chain_similarity(chain, final("a"), final("b"), ctx) == 1.0


class TestEmbeddingMatcher:
    def fake_embedder(self, calls):
        def embed(text):
            calls.append(text)
            # Orthogonal vectors for different words, same vector for same text.
            return [float(hash(w) % 97) for w in sorted(text.split())][:8] or [1.0]

        return embed

    def test_same_text_matches(self):
        calls = []
        ctx = AlignContext(final_answer_mode="embedding", embedder=self.fake_embedder(calls))
        m = EmbeddingAnswerMatcher()
        assert m.similarity(final("hello world"), final("hello world"), ctx) == 1.0

    def test_cache_embeds_each_answer_once(self):
        calls = []
        ctx = AlignContext(final_answer_mode="embedding", embedder=self.fake_embedder(calls))
        m = EmbeddingAnswerMatcher()
        a, b = final("answer one"), final("answer two")
        m.similarity(a, b, ctx)
        m.similarity(a, b, ctx)
        m.similarity(b, a, ctx)
        assert len(calls) == 2  # one embed per distinct answer, ever

    def test_embedder_failure_degrades_with_warning(self):
        def broken(text):
            raise ConnectionError("api down")

        ctx = AlignContext(final_answer_mode="embedding", embedder=broken)
        chain = build_chain(ctx)
        sim = chain_similarity(chain, final("same"), final("same"), ctx)
        assert sim == 1.0  # lexical fallback still matched
        assert any("embedding failed" in w for w in ctx.warnings)


class TestEndToEnd:
    def steps(self, answer):
        return [tool("lookup", 0), final(answer, 1)]

    def test_presence_mode_ignores_answer_change(self):
        a = align(self.steps("the total is 42"), self.steps("completely different"))
        assert not a.is_divergent

    def test_lexical_mode_flags_answer_change(self):
        ctx = AlignContext(final_answer_mode="lexical")
        a = align(
            self.steps("the total is 42"),
            self.steps("I could not find any records for that order."),
            ctx=ctx,
        )
        assert a.is_divergent and a.divergence_kind == "changed_step"
        assert a.ops[a.first_divergence].b_idx == 1

    def test_lexical_mode_tolerates_rewording(self):
        ctx = AlignContext(final_answer_mode="lexical", final_answer_threshold=0.5)
        a = align(
            self.steps("Your refund of $842 was escalated to a human agent."),
            self.steps("I escalated your $842 refund to a human agent."),
            ctx=ctx,
        )
        assert not a.is_divergent
