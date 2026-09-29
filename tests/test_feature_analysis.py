"""Stage 2 feature-analysis tests on synthetic data (no data/ access)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_analysis import (  # noqa: E402
    FeatureDecision,
    MAD_SCALE,
    V1_FEATURE_DECISIONS,
    assign_sessions,
    between_group_stability,
    bowley_skewness,
    calibration_representativeness,
    compare_groups,
    drug_overlap_context,
    envelope_peak_profile,
    exact_event_bounds,
    feature_summary,
    interior_frame_stds,
    kept_features,
    linear_r2,
    mad,
    rank_r2,
    redundancy_groups,
    robust_z,
    session_gap_margin,
    smallest_step,
    spearman_long,
    subsample_statistics,
    summarize_features,
    validate_decisions,
    within_group_spearman,
)
from inhale_dataset import DETECTION_COLUMNS, FEATURE_COLUMNS, STRIDE_S, WINDOW_S  # noqa: E402

SR = 8000


class RobustStatisticTests(unittest.TestCase):
    def test_mad_and_robust_z(self):
        values = [1, 2, 3, 4, 100]
        self.assertEqual(mad(values), 1.0)
        self.assertAlmostEqual(robust_z(values, 3, 1.0)[-1], 97 / MAD_SCALE)
        self.assertTrue(np.isinf(robust_z([4.0], 3.0, 0.0)[0]))

    def test_bowley_skewness_sign(self):
        self.assertAlmostEqual(bowley_skewness(np.arange(11)), 0.0)
        self.assertGreater(bowley_skewness(np.exp(np.linspace(0, 3, 101))), 0)

    def test_smallest_step_reports_quantisation(self):
        self.assertAlmostEqual(smallest_step([0.2, 0.216, 0.248, 0.216]), 0.016)

    def test_summary_flags_zero_mad(self):
        summary = feature_summary(np.full(10, 5.0))
        self.assertEqual(summary["mad"], 0.0)
        self.assertEqual(summary["frac_at_median"], 1.0)
        self.assertTrue(np.isinf(summary["max_abs_robust_z"]))

    def test_summary_counts_extremes_and_ignores_nonfinite(self):
        values = np.r_[np.linspace(-1, 1, 41), 25.0, np.nan]
        summary = feature_summary(values)
        self.assertEqual(summary["count"], 42)
        self.assertEqual(summary["n_extreme_high"], 1)
        self.assertEqual(summary["n_extreme_low"], 0)
        self.assertGreater(summary["top1_share_sq_z"], 0.5)

    def test_log_rows_only_for_positive_features(self):
        table = pd.DataFrame({"positive": [1.0, 2.0, 3.0], "has_zero": [0.0, 1.0, 2.0]})
        summary = summarize_features(table, ["positive", "has_zero"], {"all": pd.Series(True, index=table.index)})
        self.assertEqual(sorted(summary.loc[summary["feature"] == "positive", "scale"]), ["log", "raw"])
        self.assertEqual(summary.loc[summary["feature"] == "has_zero", "scale"].tolist(), ["raw"])

    def test_subsamples_are_deterministic(self):
        values = np.arange(100.0)
        first, second = subsample_statistics(values, 20, 50, seed=1), subsample_statistics(values, 20, 50, seed=1)
        np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            subsample_statistics(values, 101, 5)


class ComparisonTests(unittest.TestCase):
    def test_identical_groups(self):
        values = np.linspace(0, 1, 30)
        result = compare_groups(values, values)
        self.assertAlmostEqual(result["cliffs_delta"], 0.0)
        self.assertAlmostEqual(result["median_shift_in_b_mad"], 0.0)
        self.assertAlmostEqual(result["mad_ratio_a_over_b"], 1.0)
        self.assertAlmostEqual(result["expected_frac_outside_range"], 2 / 31)

    def test_calibration_representativeness_detects_a_narrow_shifted_subset(self):
        rng = np.random.default_rng(0)
        table = pd.DataFrame({"x": np.r_[rng.normal(3, 0.1, 20), rng.normal(0, 1, 200)]})
        mask = pd.Series(np.r_[np.ones(20, bool), np.zeros(200, bool)])
        result = calibration_representativeness(table, ["x"], mask, draws=200).iloc[0]
        self.assertEqual(result["n_calibration"], 20)
        self.assertLess(result["subset_frac_mad_as_small"], 0.05)
        self.assertLess(result["subset_p_median_shift"], 0.05)
        self.assertGreater(result["frac_rest_outside_calibration_range"], 0.5)


class RedundancyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=300)
        self.table = pd.DataFrame({"x": x, "x_cubed": x ** 3, "noise": rng.normal(size=300)})

    def test_rank_r2_separates_dependent_and_independent_features(self):
        self.assertGreater(rank_r2(self.table, "x_cubed", ["x"]), 0.99)
        self.assertLess(rank_r2(self.table, "noise", ["x"]), 0.05)
        self.assertEqual(rank_r2(self.table, "x", []), 0.0)

    def test_linear_r2_exact_fit(self):
        self.assertAlmostEqual(linear_r2([1, 3, 5, 7], [0, 1, 2, 3]), 1.0)

    def test_spearman_long_lists_each_pair_once(self):
        long = spearman_long(self.table, ["x", "x_cubed", "noise"])
        self.assertEqual(len(long), 3)
        self.assertAlmostEqual(long.iloc[0]["spearman_rho"], 1.0)

    def test_redundancy_groups_are_connected_components(self):
        rho = pd.DataFrame(
            [[1, 0.9, 0.1, 0.0], [0.9, 1, 0.85, 0.0], [0.1, 0.85, 1, 0.0], [0.0, 0.0, 0.0, 1]],
            index=list("abcd"), columns=list("abcd"),
        )
        self.assertEqual(redundancy_groups(rho, 0.8), [["a", "b", "c"]])

    def test_within_group_spearman_removes_group_offsets(self):
        rng = np.random.default_rng(5)
        groups = np.repeat(["g1", "g2", "g3"], 100)
        offset = np.repeat([0.0, 10.0, 20.0], 100)
        table = pd.DataFrame({"group": groups, "a": offset + rng.normal(size=300), "b": offset + rng.normal(size=300)})
        self.assertGreater(table[["a", "b"]].corr(method="spearman").loc["a", "b"], 0.8)
        self.assertLess(abs(within_group_spearman(table, ["a", "b"], "group").loc["a", "b"]), 0.2)


class SessionTests(unittest.TestCase):
    def test_sessions_split_on_large_gaps_and_keep_input_order(self):
        stamps = pd.Series(pd.to_datetime([
            "2018-01-23 10:20", "2018-01-22 17:41", "2018-01-23 10:00", "2018-01-23 12:00", "2018-01-22 17:45",
        ]))
        self.assertEqual(
            assign_sessions(stamps, 25).tolist(),
            ["2018-01-23#1", "2018-01-22#1", "2018-01-23#1", "2018-01-23#2", "2018-01-22#1"],
        )
        margin = session_gap_margin(stamps, 25)
        self.assertAlmostEqual(margin["largest_gap_within_session_min"], 20.0)
        self.assertAlmostEqual(margin["smallest_gap_between_sessions_min"], 100.0)

    def test_between_group_stability(self):
        rng = np.random.default_rng(2)
        table = pd.DataFrame({
            "group": np.r_[np.repeat(["a", "b"], 50), ["tiny"] * 2],
            "shifted": np.r_[rng.normal(0, 1, 50), rng.normal(5, 1, 50), 0, 0],
            "same": rng.normal(0, 1, 102),
        })
        summary, per_group = between_group_stability(table, ["shifted", "same"], "group", min_group_size=5)
        summary = summary.set_index("feature")
        self.assertEqual(summary.loc["shifted", "groups_tested"], 2)
        self.assertGreater(summary.loc["shifted", "epsilon_squared"], 0.5)
        self.assertLess(summary.loc["same", "epsilon_squared"], 0.1)
        self.assertFalse(per_group.loc[per_group["group"] == "tiny", "tested"].any())


class EdgeEffectTests(unittest.TestCase):
    def test_exact_event_bounds_rebuild_upstream_floats(self):
        start, end = exact_event_bounds(3.792, 3.992, 12.0)
        self.assertEqual(start, 237 * STRIDE_S)
        self.assertEqual(end, 237 * STRIDE_S + WINDOW_S)
        self.assertEqual(exact_event_bounds(11.584, 12.0, 12.0)[1], 12.0)

    def test_peak_positions(self):
        t = np.arange(SR) / SR
        carrier = np.sin(2 * np.pi * 440 * t)
        burst = carrier * np.exp(-((t - 0.5) ** 2) / 0.005)
        self.assertEqual(envelope_peak_profile(burst, SR)["peak_position"], "interior")
        self.assertEqual(envelope_peak_profile(carrier * (1 - t), SR)["peak_position"], "first")
        self.assertIn(envelope_peak_profile(carrier * t, SR)["peak_position"], ("last", "partial_tail"))

    def test_plateau_has_wider_near_peak_span_than_burst(self):
        t = np.arange(SR) / SR
        carrier = np.sin(2 * np.pi * 440 * t)
        plateau = envelope_peak_profile(carrier, SR)
        burst = envelope_peak_profile(carrier * np.exp(-((t - 0.5) ** 2) / 0.005), SR)
        self.assertGreater(plateau["near_peak_span_frac"], burst["near_peak_span_frac"])
        self.assertEqual(plateau["n_partial_frames"], 3)

    def test_interior_frame_stds(self):
        noise = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)
        result = interior_frame_stds(noise, SR)
        self.assertLess(result["zcr_interior_frame_fraction"], result["spectral_centroid_interior_frame_fraction"])
        self.assertTrue(np.isfinite(result["spectral_flatness_std_interior"]))
        with self.assertRaises(ValueError):
            interior_frame_stds(noise, 16000)


class DecisionTests(unittest.TestCase):
    def test_v1_decisions_are_complete_and_valid(self):
        validate_decisions()
        self.assertTrue(set(kept_features()).isdisjoint(DETECTION_COLUMNS))
        self.assertEqual(kept_features(), [f for f in FEATURE_COLUMNS if f in kept_features()])

    def test_validation_rejects_incomplete_or_unknown_decisions(self):
        incomplete = dict(V1_FEATURE_DECISIONS)
        incomplete.pop("duration_s")
        with self.assertRaises(ValueError):
            validate_decisions(incomplete)
        wrong = dict(V1_FEATURE_DECISIONS, duration_s=FeatureDecision("MAYBE", "none", "x"))
        with self.assertRaises(ValueError):
            validate_decisions(wrong)
        no_reason = dict(V1_FEATURE_DECISIONS, duration_s=FeatureDecision("KEEP", "none", " "))
        with self.assertRaises(ValueError):
            validate_decisions(no_reason)


class AnnotationContextTests(unittest.TestCase):
    def test_drug_overlap_context_uses_usable_annotated_events_only(self):
        table = pd.DataFrame({
            "recording_file": [f"r{i}" for i in range(10)], "event_id": 0,
            "usable": [True] * 9 + [False], "x": np.arange(10.0),
        })
        audit = pd.DataFrame({
            "recording_file": [f"r{i}" for i in range(10)], "event_id": 0,
            "recording_annotated": [True] * 8 + [False, True],
            "overlap_drug": [0.1, 0.2, 0.3, 0, 0, 0, 0, 0, np.nan, 0.5],
        })
        result = drug_overlap_context(table, audit, ["x"]).iloc[0]
        self.assertEqual(result["n_drug_overlap"], 3)
        self.assertEqual(result["n_no_drug_overlap"], 5)


if __name__ == "__main__":
    unittest.main()
