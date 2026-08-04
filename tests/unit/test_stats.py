"""Stats layer: exact tests verified against hand-computed values."""

from __future__ import annotations

from offtrack.stats import (
    MetricComparison,
    VerdictInputs,
    aggregate_verdicts,
    behavioral_verdict,
    clopper_pearson_upper,
    fisher_exact_one_sided,
    metric_verdict,
    permutation_test_median,
    suite_verdict,
)


class TestFisherExact:
    def test_0of5_vs_5of5(self):
        # Hand-computed: p = 1/C(10,5) = 1/252
        p = fisher_exact_one_sided(0, 5, 5, 5)
        assert abs(p - 1 / 252) < 1e-9

    def test_0of5_vs_4of5(self):
        # p = [C(4,4)C(6,1) + C(4,5)*0] / C(10,5) = 6/252... plus k=4 term:
        # k≥4: k=4: C(4,4)*C(6,1)=6; total successes=4 so k=5 impossible → 6/252
        p = fisher_exact_one_sided(0, 5, 4, 5)
        assert abs(p - 6 / 252) < 1e-9

    def test_0of5_vs_3of5_not_significant(self):
        p = fisher_exact_one_sided(0, 5, 3, 5)
        assert p > 0.05  # 0.083...

    def test_same_rates_high_p(self):
        assert fisher_exact_one_sided(2, 5, 2, 5) > 0.5

    def test_zero_everything(self):
        assert fisher_exact_one_sided(0, 5, 0, 5) == 1.0


class TestClopperPearson:
    def test_0_of_5(self):
        # UB = 1 - 0.05^(1/5) ≈ 0.4507
        ub = clopper_pearson_upper(0, 5)
        assert abs(ub - (1 - 0.05 ** (1 / 5))) < 1e-6

    def test_0_of_10(self):
        ub = clopper_pearson_upper(0, 10)
        assert abs(ub - (1 - 0.05 ** (1 / 10))) < 1e-6

    def test_all_divergent(self):
        assert clopper_pearson_upper(5, 5) == 1.0

    def test_n_zero(self):
        assert clopper_pearson_upper(0, 0) == 1.0


class TestBehavioralVerdict:
    def make(self, k_b=0, n_b=5, k_c=0, n_c=5, **kw) -> VerdictInputs:
        return VerdictInputs(n_baseline=n_b, k_baseline=k_b, n_candidate=n_c, k_candidate=k_c, **kw)

    def test_clean_pass(self):
        r = behavioral_verdict(self.make())
        assert r.verdict == "PASS"
        assert "upper bound" in r.reason

    def test_full_divergence_fails(self):
        r = behavioral_verdict(self.make(k_c=5))
        assert r.verdict == "FAIL"
        assert r.p_value is not None and r.p_value < 0.05

    def test_4of5_fails(self):
        r = behavioral_verdict(self.make(k_c=4))
        assert r.verdict == "FAIL"

    def test_3of5_inconclusive_with_prescription(self):
        r = behavioral_verdict(self.make(k_c=3))
        assert r.verdict == "INCONCLUSIVE"
        assert r.prescription is not None and r.prescription > 0

    def test_noisy_baseline_absorbs_divergence(self):
        # Baseline itself diverges 2/5 → candidate 2/5 is not a regression.
        r = behavioral_verdict(self.make(k_b=2, k_c=2))
        assert r.verdict in ("PASS", "INCONCLUSIVE")
        assert r.verdict != "FAIL"

    def test_min_effect_floor_blocks_trivial_fail(self):
        # Large n, significant but tiny effect: 0/40 vs 8/40 = 20% < 30% floor.
        r = behavioral_verdict(self.make(n_b=40, k_b=0, n_c=40, k_c=8))
        assert r.verdict != "FAIL"

    def test_no_candidates_error(self):
        r = behavioral_verdict(self.make(n_c=0))
        assert r.verdict == "ERROR"

    def test_no_baseline_error(self):
        r = behavioral_verdict(self.make(n_b=0))
        assert r.verdict == "ERROR"

    def test_error_samples_warned(self):
        r = behavioral_verdict(self.make(errors=2))
        assert any("no signal" in w for w in r.warnings)

    def test_deterministic_fast_path(self):
        r = behavioral_verdict(self.make(k_c=1, deterministic=True))
        assert r.verdict == "FAIL"
        assert "deterministic" in r.reason

    def test_deterministic_needs_stable_baseline(self):
        r = behavioral_verdict(self.make(k_b=1, k_c=1, deterministic=True))
        assert r.verdict != "FAIL" or "deterministic" not in r.reason


class TestPermutation:
    def test_identical_groups_high_p(self):
        p = permutation_test_median([1.0, 1.1, 0.9], [1.0, 1.1, 0.9])
        assert p > 0.4

    def test_clear_shift_low_p(self):
        p = permutation_test_median([1.0, 1.1, 0.9, 1.05, 0.95], [2.0, 2.1, 1.9, 2.05, 1.95])
        assert p < 0.05


class TestMetricVerdict:
    def test_within_threshold_passes(self):
        m = MetricComparison("cost", [1.0] * 5, [1.1] * 5, threshold=0.2, action="fail")
        assert metric_verdict(m).verdict == "PASS"

    def test_small_n_suggestive_never_fails(self):
        m = MetricComparison("cost", [1.0, 1.0], [2.0, 2.0], threshold=0.2, action="fail")
        r = metric_verdict(m)
        assert r.verdict == "WARN"
        assert "not gating" in r.reason

    def test_supported_increase_fails(self):
        m = MetricComparison(
            "cost",
            [1.0, 1.05, 0.95, 1.02, 0.98],
            [1.5, 1.55, 1.45, 1.52, 1.48],
            threshold=0.2,
            action="fail",
        )
        r = metric_verdict(m)
        assert r.verdict == "FAIL" and r.p_value is not None

    def test_warn_action_never_fails(self):
        m = MetricComparison(
            "latency",
            [1.0, 1.05, 0.95, 1.02, 0.98],
            [2.0, 2.1, 1.9, 2.05, 1.95],
            threshold=0.5,
            action="warn",
        )
        assert metric_verdict(m).verdict == "WARN"

    def test_empty_data_skips(self):
        m = MetricComparison("cost", [], [1.0], threshold=0.2, action="fail")
        assert metric_verdict(m).verdict == "SKIP"


class TestAggregation:
    def test_task_worst_wins(self):
        from offtrack.stats import MetricResult

        fail_metric = MetricResult("cost", "FAIL", 1, 2, 1.0, 0.01, "x")
        assert aggregate_verdicts("PASS", [fail_metric]) == "FAIL"
        assert aggregate_verdicts("INCONCLUSIVE", []) == "INCONCLUSIVE"
        assert aggregate_verdicts("ERROR", [fail_metric]) == "ERROR"

    def test_suite_ordering(self):
        assert suite_verdict(["PASS", "FAIL", "INCONCLUSIVE"]) == "FAIL"
        assert suite_verdict(["PASS", "ERROR"]) == "ERROR"
        assert suite_verdict(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
        assert suite_verdict(["PASS", "PASS"]) == "PASS"
        assert suite_verdict([]) == "ERROR"
