"""Stage 3 V1 robust-baseline tests (synthetic data; no data/ access)."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_v1 import (  # noqa: E402
    BaselineError,
    GROUPS,
    RobustBaseline,
    calibration_design_stats,
    comparison_groups,
    fit_baseline,
    load_baseline,
    load_feature_selection,
    primary_calibration_mask,
    random_calibration_sets,
    random_session_spread_sets,
    session_decomposition,
    session_spread_set,
    summarize_z_by_group,
    summarize_z_by_session,
)
from feature_analysis import MAD_SCALE  # noqa: E402

FEATURES = ["duration_s", "mean_rms"]


def events(n=30, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "recording_file": [f"rec{i:03d}.wav" for i in range(n)],
        "event_id": 0,
        "usable": True,
        "usable_order": np.arange(1, n + 1, dtype=float),
        "duration_s": rng.normal(1.6, 0.3, n),
        "mean_rms": rng.normal(0.18, 0.03, n),
    })
    return table


def write_selection(directory, keep, transforms=None, dataset_sha=None):
    path = Path(directory) / "selection.json"
    payload = {"keep": keep, "transforms": transforms or {f: "none" for f in keep}}
    if dataset_sha:
        payload["dataset"] = {"sha256": dataset_sha}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FitTests(unittest.TestCase):
    def test_median_mad_and_scale(self):
        calibration = pd.DataFrame({"duration_s": [1, 2, 3, 4, 100.0], "mean_rms": [0.1, 0.2, 0.3, 0.4, 0.5]})
        baseline = fit_baseline(calibration, FEATURES)
        self.assertEqual(baseline.parameters["duration_s"].median, 3.0)
        self.assertEqual(baseline.parameters["duration_s"].mad, 1.0)
        self.assertAlmostEqual(baseline.parameters["duration_s"].scale, MAD_SCALE)
        self.assertAlmostEqual(baseline.parameters["mean_rms"].mad, 0.1)

    def test_scaling_convention_matches_normal_consistency_constant(self):
        self.assertAlmostEqual(MAD_SCALE, 1 / stats.norm.ppf(0.75), places=4)
        calibration = pd.DataFrame({"duration_s": [1, 2, 3, 4, 5.0], "mean_rms": [1, 2, 3, 4, 5.0]})
        baseline = fit_baseline(calibration, FEATURES)
        z = baseline.z_scores(pd.DataFrame({"duration_s": [3 + MAD_SCALE], "mean_rms": [3.0]}))
        self.assertAlmostEqual(z.iloc[0]["z_duration_s"], 1.0)
        self.assertAlmostEqual(z.iloc[0]["z_mean_rms"], 0.0)

    def test_zero_mad_fails_clearly(self):
        calibration = pd.DataFrame({"duration_s": [1.0, 1.0, 1.0, 2.0], "mean_rms": [1.0, 2.0, 3.0, 4.0]})
        with self.assertRaisesRegex(BaselineError, "MAD of duration_s"):
            fit_baseline(calibration, FEATURES)

    def test_non_finite_calibration_values_fail(self):
        calibration = pd.DataFrame({"duration_s": [1.0, np.nan, 3.0], "mean_rms": [1.0, 2.0, np.inf]})
        with self.assertRaisesRegex(BaselineError, "non-finite"):
            fit_baseline(calibration, FEATURES)

    def test_non_finite_scoring_input_fails(self):
        baseline = fit_baseline(events(), FEATURES)
        bad = events().iloc[:3].copy()
        bad.loc[bad.index[1], "mean_rms"] = np.nan
        with self.assertRaisesRegex(BaselineError, "mean_rms"):
            baseline.z_scores(bad)

    def test_empty_and_missing_columns_fail(self):
        with self.assertRaises(BaselineError):
            fit_baseline(events().iloc[0:0], FEATURES)
        with self.assertRaises(BaselineError):
            fit_baseline(events().drop(columns="mean_rms"), FEATURES)

    def test_fit_is_deterministic_and_keeps_event_keys(self):
        table = events()
        first, second = fit_baseline(table, FEATURES), fit_baseline(table, FEATURES)
        self.assertEqual(first, second)
        self.assertEqual(first.calibration_events[0], ("rec000.wav", 0))

    def test_json_round_trip(self):
        baseline = fit_baseline(events(), FEATURES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({"baseline": baseline.to_dict()}), encoding="utf-8")
            self.assertEqual(load_baseline(path), baseline)


class CalibrationTests(unittest.TestCase):
    def test_primary_calibration_is_usable_order_1_to_20(self):
        table = events(30)
        table.loc[3, "usable"] = False          # an excluded event cannot be calibration
        table.loc[3, "usable_order"] = np.nan
        table.loc[4:, "usable_order"] -= 1
        mask = primary_calibration_mask(table)
        self.assertEqual(int(mask.sum()), 20)
        self.assertFalse(mask.iloc[3])
        self.assertTrue(mask.iloc[20])           # usable_order 20 after the renumbering
        self.assertFalse(mask.iloc[21])

    def test_primary_calibration_requires_twenty_events(self):
        with self.assertRaises(BaselineError):
            primary_calibration_mask(events(15))

    def test_no_leakage_from_non_calibration_events(self):
        table = events(40)
        mask = primary_calibration_mask(table)
        reference = fit_baseline(table[mask], FEATURES)
        changed = table.copy()
        changed.loc[~mask, FEATURES] = 1e6       # non-calibration events must not matter
        self.assertEqual(fit_baseline(changed[mask], FEATURES).parameters, reference.parameters)

    def test_z_scores_preserve_index_and_feature_names(self):
        table = events(25).set_index(pd.Index(range(100, 125)))
        z = fit_baseline(table, FEATURES).z_scores(table)
        self.assertEqual(list(z.columns), ["z_duration_s", "z_mean_rms"])
        self.assertTrue(z.index.equals(table.index))


class SelectionTests(unittest.TestCase):
    def test_loads_keep_list_in_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_selection(directory, ["spectral_rolloff_std", "duration_s"])
            self.assertEqual(load_feature_selection(path), ["spectral_rolloff_std", "duration_s"])

    def test_rejects_invalid_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            for keep, transforms in (([], None), (["confidence"], None), (["not_a_feature"], None),
                                     (["duration_s", "duration_s"], None), (["mean_rms"], {"mean_rms": "log"})):
                with self.assertRaises(BaselineError):
                    load_feature_selection(write_selection(directory, keep, transforms))

    def test_rejects_selection_made_on_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.csv"
            dataset.write_text("a\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(BaselineError, "different dataset"):
                load_feature_selection(write_selection(directory, ["duration_s"], dataset_sha="0" * 64), dataset)


class GroupAndSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.table = events(60, seed=1)
        self.sessions = pd.Series(np.repeat(["s1", "s2", "s3"], 20), index=self.table.index)
        self.mask = pd.Series(np.r_[np.ones(10, bool), np.zeros(50, bool)], index=self.table.index)

    def test_comparison_groups(self):
        self.mask.iloc[:] = False
        self.mask.iloc[[0, 1, 25]] = True
        groups = comparison_groups(self.sessions, self.mask)
        self.assertEqual(groups.iloc[0], "calibration")
        self.assertEqual(groups.iloc[5], "calibration_session_other")   # s1 has calibration events
        self.assertEqual(groups.iloc[30], "calibration_session_other")  # s2 has one
        self.assertEqual(groups.iloc[45], "other_session")

    def test_group_and_session_summaries(self):
        z = fit_baseline(self.table[self.mask], FEATURES).z_scores(self.table)
        groups = comparison_groups(self.sessions, self.mask)
        summary = summarize_z_by_group(z, groups, FEATURES)
        self.assertEqual(set(summary["group"]), set(GROUPS))
        calibration = summary[(summary["group"] == "calibration") & (summary["feature"] == "duration_s")].iloc[0]
        self.assertAlmostEqual(calibration["median_z"], 0.0)
        self.assertAlmostEqual(calibration["robust_sd_z"], 1.0)
        self.assertEqual(calibration["frac_outside_calibration_z_range"], 0.0)
        by_session = summarize_z_by_session(z, self.sessions, self.mask, FEATURES)
        s1 = by_session[(by_session["session"] == "s1") & (by_session["feature"] == "mean_rms")].iloc[0]
        self.assertEqual((s1["n_calibration_events"], s1["n_compared"]), (10, 10))
        decomposition = session_decomposition(by_session)
        self.assertEqual(int(decomposition.iloc[0]["sessions"]), 3)

    def test_random_sets_are_deterministic_and_unique(self):
        first = random_calibration_sets(60, 20, 5, seed=7)
        second = random_calibration_sets(60, 20, 5, seed=7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(np.unique(a)), 20)

    def test_session_spread_round_robin(self):
        sessions = np.array(["a"] * 10 + ["b"] * 3 + ["c"] * 10)
        order = np.arange(23, dtype=float)
        chosen = session_spread_set(sessions, order, 9, ["a", "b", "c"])
        # three passes over three sessions: first three events of each session
        np.testing.assert_array_equal(chosen, [0, 1, 2, 10, 11, 12, 13, 14, 15])
        spread = random_session_spread_sets(sessions, 20, 3, seed=2)
        self.assertTrue(all(len(np.unique(s)) == 20 for s in spread))
        self.assertTrue(all(np.sum(sessions[s] == "b") == 3 for s in spread))
        with self.assertRaises(BaselineError):
            session_spread_set(sessions, order, 30, ["a", "b", "c"])

    def test_design_stats_use_only_the_calibration_rows(self):
        values = self.table[FEATURES].to_numpy()
        index = np.arange(10)
        stats_frame = calibration_design_stats(values, self.sessions.to_numpy(), [index], FEATURES, "test")
        expected = fit_baseline(self.table.iloc[:10], FEATURES).parameters["duration_s"]
        full_scale = MAD_SCALE * np.median(np.abs(values[:, 0] - np.median(values[:, 0])))
        row = stats_frame[(stats_frame["feature"] == "duration_s") & (stats_frame["group"] == "all_non_calibration")].iloc[0]
        self.assertAlmostEqual(row["scale_ratio"], expected.scale / full_scale)
        self.assertEqual(row["n_compared"], 50)
        other = stats_frame[(stats_frame["feature"] == "duration_s") & (stats_frame["group"] == "other_session")].iloc[0]
        self.assertEqual(other["n_compared"], 40)


if __name__ == "__main__":
    unittest.main()
