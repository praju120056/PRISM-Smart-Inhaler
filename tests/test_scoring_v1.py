"""Stage 4 candidate-score tests (synthetic data; no data/ access)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_v1 import fit_baseline, primary_calibration_mask  # noqa: E402
from scoring_v1 import (  # noqa: E402
    ScoringError,
    combined_scores,
    distribution_summary,
    feature_contributions,
    iid_normal_reference,
    inflation_decomposition,
    isolated_extreme_ranks,
    leave_one_out_scores,
    pair_analysis,
    single_feature_influence,
    summarize_contributions,
)

FEATURES = ["duration_s", "mean_rms", "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_flatness_mean", "spectral_flatness_std", "spectral_rolloff_std"]


def z_frame(rows):
    return pd.DataFrame(rows, columns=[f"z_{f}" for f in FEATURES])


def events(n=30, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(rng.normal(1.0, 0.2, (n, len(FEATURES))), columns=FEATURES)
    table.insert(0, "recording_file", [f"rec{i:03d}.wav" for i in range(n)])
    table.insert(1, "event_id", 0)
    table["usable"] = True
    table["usable_order"] = np.arange(1, n + 1, dtype=float)
    return table


class ScoreFormulaTests(unittest.TestCase):
    def test_exact_formulas_and_max_feature(self):
        scores = combined_scores(z_frame([[1, -2, 3, 0, 0, 0, 0]]), FEATURES).iloc[0]
        self.assertAlmostEqual(scores["mean_abs_z"], 6 / 7)
        self.assertAlmostEqual(scores["rms_z"], np.sqrt(14 / 7))
        self.assertEqual(scores["max_abs_z"], 3.0)
        self.assertEqual(scores["max_abs_z_feature"], "spectral_centroid_mean")

    def test_negative_extreme_counts_by_magnitude_and_ties_are_listed(self):
        scores = combined_scores(z_frame([[-4, 4, 1, 0, 0, 0, 0]]), FEATURES).iloc[0]
        self.assertEqual(scores["max_abs_z"], 4.0)
        self.assertEqual(scores["max_abs_z_feature"], "duration_s;mean_rms")

    def test_all_seven_features_are_required(self):
        with self.assertRaises(ScoringError):
            combined_scores(z_frame([[0] * 7]).drop(columns="z_spectral_rolloff_std"), FEATURES)
        with self.assertRaises(ScoringError):
            combined_scores(z_frame([[0] * 7]), [])

    def test_non_finite_and_empty_inputs(self):
        with self.assertRaises(ScoringError):
            combined_scores(z_frame([[np.nan, 0, 0, 0, 0, 0, 0]]), FEATURES)
        self.assertTrue(combined_scores(z_frame([]), FEATURES).empty)
        self.assertTrue(feature_contributions(z_frame([]), FEATURES).empty)

    def test_scores_are_deterministic(self):
        z = z_frame(np.random.default_rng(1).normal(size=(50, 7)))
        pd.testing.assert_frame_equal(combined_scores(z, FEATURES), combined_scores(z, FEATURES))
        self.assertEqual(iid_normal_reference(7, 1000, 3), iid_normal_reference(7, 1000, 3))


class ContributionTests(unittest.TestCase):
    def test_contribution_definitions(self):
        contributions = feature_contributions(z_frame([[1, -2, 3, 0, 0, 0, 0]]), FEATURES).iloc[0]
        self.assertEqual(contributions["abs_z_mean_rms"], 2.0)                    # mean_abs_z: |z_j|
        self.assertAlmostEqual(contributions["abs_share_mean_rms"], 2 / 6)
        self.assertAlmostEqual(contributions["rms_share_mean_rms"], 4 / 14)       # rms_z: z_j^2 / sum z^2
        self.assertAlmostEqual(sum(contributions[f"rms_share_{f}"] for f in FEATURES), 1.0)
        self.assertTrue(contributions["is_max_spectral_centroid_mean"])           # max_abs_z: argmax
        self.assertEqual(sum(bool(contributions[f"is_max_{f}"]) for f in FEATURES), 1)

    def test_all_zero_event_has_undefined_shares(self):
        contributions = feature_contributions(z_frame([[0] * 7]), FEATURES).iloc[0]
        self.assertTrue(np.isnan(contributions["rms_share_duration_s"]))
        self.assertEqual(combined_scores(z_frame([[0] * 7]), FEATURES).iloc[0]["rms_z"], 0.0)

    def test_single_feature_influence(self):
        one = single_feature_influence(z_frame([[3, 0, 0, 0, 0, 0, 0]]), FEATURES).iloc[0]
        self.assertEqual((one["top1_share_mean_abs"], one["top1_share_rms"]), (1.0, 1.0))
        self.assertEqual((one["mean_abs_without_top"], one["rms_without_top"], one["max_abs_without_top"]), (0, 0, 0))
        flat = single_feature_influence(z_frame([[1] * 7]), FEATURES).iloc[0]
        self.assertAlmostEqual(flat["top1_share_rms"], 1 / 7)
        self.assertAlmostEqual(flat["rms_without_top"], 1.0)
        self.assertAlmostEqual(flat["max_abs_without_top"], 1.0)

    def test_isolated_extreme_events_rank_highest_under_max(self):
        rows = np.full((10, 7), 1.0)
        rows[0] = [6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]    # one isolated extreme feature
        rows[1:, :] = np.linspace(1.2, 1.6, 9)[:, None]   # broad, moderate deviations
        z = z_frame(rows)
        result = isolated_extreme_ranks(combined_scores(z, FEATURES), single_feature_influence(z, FEATURES))
        self.assertEqual(result["n_isolated"], 1)
        ranks = result["median_percentile_rank"]
        self.assertEqual(ranks["max_abs_z"], 1.0)
        self.assertLess(ranks["mean_abs_z"], ranks["rms_z"])

    def test_summarized_argmax_fractions_sum_to_one(self):
        contributions = feature_contributions(z_frame(np.random.default_rng(2).normal(size=(40, 7))), FEATURES)
        summary = summarize_contributions({"all": contributions}, FEATURES, "overall")
        self.assertAlmostEqual(summary["max_abs_z__argmax_fraction"].sum(), 1.0)
        self.assertAlmostEqual(summary["rms_z__mean_share"].sum(), 1.0)

    def test_inflation_decomposition_is_exact(self):
        rng = np.random.default_rng(4)
        target, reference = z_frame(rng.normal(0, 3, (60, 7))), z_frame(rng.normal(0, 1, (20, 7)))
        decomposition = inflation_decomposition(target, reference, FEATURES)
        difference = combined_scores(target, FEATURES)["mean_abs_z"].mean() - combined_scores(reference, FEATURES)["mean_abs_z"].mean()
        self.assertAlmostEqual(decomposition["mean_abs_z__mean_difference_contribution"].sum(), difference)
        squared = (combined_scores(target, FEATURES)["rms_z"] ** 2).mean() - (combined_scores(reference, FEATURES)["rms_z"] ** 2).mean()
        self.assertAlmostEqual(decomposition["rms_z_squared__mean_difference_contribution"].sum(), squared)

    def test_pair_analysis_detects_joint_domination(self):
        rng = np.random.default_rng(5)
        rows = rng.normal(0, 0.1, (30, 7))
        rows[:, 1] = rng.uniform(3, 4, 30)
        rows[:, 2] = -rows[:, 1] + rng.normal(0, 0.01, 30)
        z = z_frame(rows)
        result = pair_analysis(z, feature_contributions(z, FEATURES), FEATURES)
        self.assertEqual(result["fraction_pair_are_top_two"], 1.0)
        self.assertEqual(result["fraction_opposite_sign"], 1.0)
        self.assertLess(result["spearman_z"], -0.9)
        self.assertAlmostEqual(result["top_two_reference_if_exchangeable"], 1 / 21)


class LeaveOneOutTests(unittest.TestCase):
    def setUp(self):
        self.table = events(30)
        self.calibration = self.table[primary_calibration_mask(self.table)]

    def test_each_event_is_scored_against_the_other_nineteen(self):
        scores, parameters = leave_one_out_scores(self.calibration, FEATURES)
        self.assertEqual(len(scores), 20)
        self.assertTrue((scores["n_training"] == 19).all())
        self.assertEqual(list(scores["recording_file"]), list(self.calibration["recording_file"]))
        for position, index in enumerate(self.calibration.index):
            expected = fit_baseline(self.calibration.drop(index=index), FEATURES)
            held_out = self.calibration.loc[[index]]
            z = expected.z_scores(held_out).iloc[0]
            self.assertAlmostEqual(scores.iloc[position]["z_mean_rms"], z["z_mean_rms"])
            stored = parameters[(parameters["held_out_recording_file"] == held_out["recording_file"].iloc[0])
                                & (parameters["feature"] == "mean_rms")].iloc[0]
            self.assertAlmostEqual(stored["median"], expected.parameters["mean_rms"].median)

    def test_held_out_event_is_not_in_its_baseline(self):
        duplicated = pd.concat([self.calibration, self.calibration.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ScoringError, "leaked"):
            leave_one_out_scores(duplicated, FEATURES)

    def test_leave_one_out_differs_from_in_sample(self):
        loo, _ = leave_one_out_scores(self.calibration, FEATURES)
        in_sample = combined_scores(fit_baseline(self.calibration, FEATURES).z_scores(self.calibration), FEATURES)
        self.assertGreater(np.median(loo["rms_z"].to_numpy() / in_sample["rms_z"].to_numpy()), 1.0)

    def test_needs_at_least_three_events(self):
        with self.assertRaises(ScoringError):
            leave_one_out_scores(self.calibration.iloc[:2], FEATURES)

    def test_non_calibration_events_cannot_change_calibration_scores(self):
        mask = primary_calibration_mask(self.table)
        reference = combined_scores(fit_baseline(self.table[mask], FEATURES).z_scores(self.table[mask]), FEATURES)
        changed = self.table.copy()
        changed.loc[~mask, FEATURES] = 1e6
        scores = combined_scores(fit_baseline(changed[mask], FEATURES).z_scores(changed[mask]), FEATURES)
        pd.testing.assert_frame_equal(scores, reference)


class DistributionSummaryTests(unittest.TestCase):
    def test_summary_values(self):
        summary = distribution_summary([1, 2, 3, 4, 5])
        self.assertEqual((summary["n"], summary["median"], summary["p50"], summary["max"]), (5, 3.0, 3.0, 5.0))
        self.assertAlmostEqual(summary["sd"], np.std([1, 2, 3, 4, 5], ddof=1))
        self.assertTrue(np.isnan(distribution_summary([2.0])["sd"]))
        self.assertEqual(distribution_summary([]), {"n": 0})


if __name__ == "__main__":
    unittest.main()
