"""Stage 5 baseline-strategy tests (synthetic data; no data/ access)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_strategies import (  # noqa: E402
    LEVEL_FEATURES,
    MIN_SESSION_EVENTS_LOCATION,
    MIN_SESSION_EVENTS_SCALE,
    STRATEGIES,
    VARIABILITY_FEATURES,
    WARMUP_EVENTS,
    hybrid_table,
    evaluate_contributions,
    evaluate_features,
    evaluate_scores,
    first20_control,
    leave_one_out_session_medians,
    leave_one_session_out,
    parameter_variability,
    scale_heterogeneity_test,
    session_in_sample_reference,
    session_location_residuals,
    session_scale_stability,
    session_scale_z,
    warmup_name,
    warmup_residuals,
    warmup_status,
)
from baseline_v1 import BaselineError, fit_baseline  # noqa: E402
from feature_analysis import MAD_SCALE  # noqa: E402

FEATURES = ["duration_s", "mean_rms", "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_flatness_mean", "spectral_flatness_std", "spectral_rolloff_std"]


def sessions_table(sizes=(30, 25, 8, 3), shifts=(0.0, 1.0, -1.0, 2.0), spreads=None, seed=0):
    rng = np.random.default_rng(seed)
    spreads = spreads or [1.0] * len(sizes)
    rows = []
    order = 1
    for s, (n, shift, spread) in enumerate(zip(sizes, shifts, spreads)):
        for i in range(n):
            rows.append({"recording_file": f"s{s}_r{i:02d}.wav", "event_id": 0, "session": f"S{s}",
                         "usable_order": order,
                         **{f: shift + spread * rng.normal() for f in FEATURES}})
            order += 1
    return pd.DataFrame(rows)


class LeaveOneSessionOutTests(unittest.TestCase):
    def setUp(self):
        self.table = sessions_table()
        self.sessions = self.table["session"]

    def test_every_event_scored_once_by_a_baseline_that_excludes_its_session(self):
        z, parameters, references = leave_one_session_out(self.table, self.sessions, self.table, self.sessions, FEATURES)
        self.assertTrue(z.index.equals(self.table.index))
        for session in self.sessions.unique():
            expected = fit_baseline(self.table[self.sessions != session], FEATURES)
            fold = parameters[(parameters["fold"] == session) & (parameters["feature"] == "mean_rms")].iloc[0]
            self.assertAlmostEqual(fold["center"], expected.parameters["mean_rms"].median)
            self.assertAlmostEqual(fold["scale"], expected.parameters["mean_rms"].scale)
            self.assertEqual(fold["n_train"], int((self.sessions != session).sum()))
            held = self.table[self.sessions == session]
            np.testing.assert_allclose(z.loc[held.index].to_numpy(), expected.z_scores(held).to_numpy())
        self.assertEqual(set(references["fold"]), set(self.sessions))

    def test_held_out_values_never_enter_their_own_fold(self):
        _, reference, _ = leave_one_session_out(self.table, self.sessions, self.table, self.sessions, FEATURES)
        changed = self.table.copy()
        changed.loc[self.sessions == "S1", FEATURES] += 1e6   # shifted, spread kept
        z, parameters, _ = leave_one_session_out(changed, self.sessions, changed, self.sessions, FEATURES)
        fold = lambda frame: frame[frame["fold"] == "S1"].reset_index(drop=True)
        pd.testing.assert_frame_equal(fold(parameters), fold(reference))
        self.assertTrue((z.loc[self.sessions == "S1"].abs() > 1e3).all().all())

    def test_single_session_cannot_be_held_out(self):
        one = self.table[self.sessions == "S0"]
        with self.assertRaises(BaselineError):
            leave_one_session_out(one, one["session"], one, one["session"], FEATURES)


class SessionNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.table = sessions_table()

    def test_leave_one_out_session_median_excludes_the_event(self):
        medians = leave_one_out_session_medians(self.table, FEATURES)
        group = self.table[self.table["session"] == "S2"]
        first = group.index[0]
        expected = np.median(group.loc[group.index[1:], "mean_rms"])
        self.assertAlmostEqual(medians.loc[first, "mean_rms"], expected)
        small = self.table["session"] == "S3"                      # 3 events < 5
        self.assertTrue(medians.loc[small].isna().all().all())
        self.assertEqual(MIN_SESSION_EVENTS_LOCATION, 5)

    def test_location_residuals(self):
        residuals = session_location_residuals(self.table, FEATURES)
        medians = leave_one_out_session_medians(self.table, FEATURES)
        pd.testing.assert_frame_equal(residuals, self.table[FEATURES] - medians)

    def test_location_normalization_removes_session_shift(self):
        table = sessions_table(sizes=(40, 40), shifts=(0.0, 10.0))
        residuals = session_location_residuals(table, FEATURES)
        by_session = residuals.groupby(table["session"])["mean_rms"].median()
        self.assertLess(abs(by_session["S1"] - by_session["S0"]), 1.0)

    def test_warmup_uses_first_k_events_by_order(self):
        table = sessions_table(sizes=(12, 4))
        table = table.sample(frac=1.0, random_state=1)            # shuffled row order
        warm = warmup_residuals(table, FEATURES, k=5)
        s0 = table[table["session"] == "S0"].sort_values("usable_order")
        location = s0.iloc[:5]["mean_rms"].median()
        later = s0.index[5:]
        np.testing.assert_allclose(warm.loc[later, "mean_rms"], table.loc[later, "mean_rms"] - location)
        self.assertTrue(warm.loc[s0.index[:5]].isna().all().all())     # warm-up events are not scored
        self.assertTrue(warm.loc[table["session"] == "S1"].isna().all().all())  # 4 <= k: not scored
        with self.assertRaises(ValueError):
            warmup_residuals(table, FEATURES, k=0)

    def test_session_scale_z_uses_other_events_only(self):
        table = sessions_table(sizes=(25, 6))
        z = session_scale_z(table, FEATURES)
        group = table[table["session"] == "S0"]
        index = group.index[3]
        others = group.drop(index=index)["duration_s"].to_numpy()
        median = np.median(others)
        expected = (group.loc[index, "duration_s"] - median) / (MAD_SCALE * np.median(np.abs(others - median)))
        self.assertAlmostEqual(z.loc[index, "z_duration_s"], expected)
        self.assertTrue(z.loc[table["session"] == "S1"].isna().all().all())  # 6 < 20
        self.assertEqual(MIN_SESSION_EVENTS_SCALE, 20)

    def test_session_scale_z_zero_mad_raises(self):
        table = sessions_table(sizes=(25,))
        table["duration_s"] = 1.0
        with self.assertRaises(BaselineError):
            session_scale_z(table, FEATURES)

    def test_hybrid_table_normalizes_level_features_only(self):
        residuals = session_location_residuals(self.table, FEATURES)
        hybrid = hybrid_table(self.table[FEATURES], residuals)
        for feature in LEVEL_FEATURES:
            pd.testing.assert_series_equal(hybrid[feature], residuals[feature])
        for feature in VARIABILITY_FEATURES:
            pd.testing.assert_series_equal(hybrid[feature], self.table[feature])
        self.assertEqual(set(LEVEL_FEATURES) | set(VARIABILITY_FEATURES), set(FEATURES))

    def test_in_sample_session_reference_only_for_large_sessions(self):
        reference = session_in_sample_reference(sessions_table(sizes=(25, 6)), FEATURES)
        self.assertEqual(list(reference["fold"]), ["S0"])


class First20ControlTests(unittest.TestCase):
    def test_calibration_rows_use_leave_one_out_z(self):
        context = pd.DataFrame({"recording_file": ["a", "b", "c"], "event_id": 0, "calibration": [True, False, True]})
        columns = [f"z_{f}" for f in FEATURES]
        primary = pd.DataFrame({"recording_file": ["a", "b", "c"], "event_id": 0, **{c: [1.0, 2.0, 3.0] for c in columns}})
        loo = pd.DataFrame({"recording_file": ["c", "a"], "event_id": 0, **{c: [30.0, 10.0] for c in columns}})
        z = first20_control(context, primary, loo, FEATURES)
        self.assertEqual(z["z_mean_rms"].tolist(), [10.0, 2.0, 30.0])
        with self.assertRaises(BaselineError):
            first20_control(context, primary, loo.iloc[:1], FEATURES)


class ScaleStabilityTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_wider_for_small_sessions(self):
        table = sessions_table(sizes=(60, 6), shifts=(0.0, 0.0))
        first = session_scale_stability(table, FEATURES, draws=300, seed=4)
        second = session_scale_stability(table, FEATURES, draws=300, seed=4)
        pd.testing.assert_frame_equal(first, second)
        cv = first.groupby("session")["bootstrap_cv"].median()
        self.assertGreater(cv["S1"], cv["S0"])

    def test_heterogeneity_test_detects_scale_differences_only_when_present(self):
        heterogeneous = sessions_table(sizes=(30, 30, 30, 30), shifts=(0, 3, -3, 1), spreads=[0.5, 4.0, 0.5, 4.0])
        result = scale_heterogeneity_test(heterogeneous, FEATURES, permutations=200, seed=1)
        self.assertTrue((result["permutation_p"] < 0.05).all())
        homogeneous = sessions_table(sizes=(30, 30, 30, 30), shifts=(0, 3, -3, 1), seed=7)
        result = scale_heterogeneity_test(homogeneous, FEATURES, permutations=200, seed=1)
        self.assertGreater(result["permutation_p"].median(), 0.05)
        self.assertEqual(set(result["family"]), {"level", "within-event variability", "all"})
        self.assertEqual(result.iloc[-1]["feature"], "ALL_FEATURES_JOINT")


class EvaluationTests(unittest.TestCase):
    def frame(self, shift=0.0):
        rng = np.random.default_rng(3)
        sessions = np.repeat(["S0", "S1", "S2"], 20)
        values = np.abs(rng.normal(1, 0.2, 60)) + np.where(sessions == "S2", shift, 0.0)
        frame = pd.DataFrame({"session": sessions, "mean_abs_z": values, "rms_z": values, "max_abs_z": 2 * values,
                              "reference_mean_abs_z": 0.5, "reference_rms_z": 0.5, "reference_max_abs_z": 1.0})
        for f in FEATURES:
            frame[f"z_{f}"] = rng.normal(size=60)
        return frame

    def test_ratio_and_session_dependence(self):
        mask = np.ones(60, bool)
        flat = {r["score"]: r for r in evaluate_scores(self.frame(), mask)}
        self.assertAlmostEqual(flat["rms_z"]["median_over_reference"], flat["rms_z"]["median"] / 0.5)
        self.assertLess(flat["rms_z"]["session_epsilon_squared"], 0.1)
        shifted = {r["score"]: r for r in evaluate_scores(self.frame(shift=3.0), mask)}
        self.assertGreater(shifted["rms_z"]["session_epsilon_squared"], 0.5)
        self.assertGreater(shifted["rms_z"]["session_median_max_over_min"], 3)

    def test_incomplete_coverage_is_not_evaluated(self):
        frame = self.frame()
        frame.loc[0, "rms_z"] = np.nan
        self.assertEqual(evaluate_scores(frame, np.ones(60, bool)), [])

    def test_feature_and_contribution_summaries(self):
        frame = self.frame()
        features = evaluate_features(frame, np.ones(60, bool), FEATURES)
        self.assertEqual(len(features), 7)
        self.assertEqual({r["family"] for r in features}, {"level", "within-event variability"})
        contributions = evaluate_contributions(frame, np.ones(60, bool), FEATURES)
        self.assertAlmostEqual(sum(r["rms_z__mean_share"] for r in contributions), 1.0)
        self.assertAlmostEqual(sum(r["max_abs_z__argmax_fraction"] for r in contributions), 1.0)

    def test_parameter_variability(self):
        parameters = pd.DataFrame({"strategy": "C", "feature": "mean_rms", "center": [1.0, 2.0], "scale": [0.5, 1.5]})
        row = parameter_variability(parameters, {"mean_rms": (1.5, 1.0)}).iloc[0]
        self.assertEqual((row["center_range_in_pooled_scale"], row["scale_min_over_pooled"], row["scale_max_over_pooled"]),
                         (1.0, 0.5, 1.5))


class StatusTests(unittest.TestCase):
    def test_statuses_state_what_each_strategy_can_claim(self):
        self.assertIn("in-sample", STRATEGIES["B_pooled_in_sample"])
        self.assertIn("offline diagnostic", STRATEGIES["D1_offline_session_location"])
        self.assertIn("offline diagnostic", STRATEGIES["D3_offline_session_location_scale"])
        self.assertIn("deployable", STRATEGIES["C_loso_global"])
        self.assertIn("primary", warmup_status(WARMUP_EVENTS))
        self.assertIn("sensitivity", warmup_status(3))
        self.assertEqual(warmup_name(5), "D2_warmup5_session_location")
        self.assertEqual(warmup_name(5, level_only=True), "D2L_warmup5_level_location")
        self.assertIn("level features only", warmup_status(5, level_only=True))
        self.assertIn("offline diagnostic", STRATEGIES["D1L_offline_level_location"])


if __name__ == "__main__":
    unittest.main()
