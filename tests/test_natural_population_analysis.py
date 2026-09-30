"""Stage 6 natural-population tests (synthetic data; no data/ access)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natural_population_analysis import (  # noqa: E402
    PERTURBATION_ORDER,
    PopulationError,
    add_white_noise,
    apply_gain,
    apply_tilt,
    assign_populations,
    cliffs_delta,
    compare_to_reference,
    duration_matched_comparison,
    comparison_groups,
    feature_attribution,
    heldout_scores,
    hodges_lehmann_shift,
    measure_segment,
    overlap_matrix,
    perturb,
    perturbation_plan,
    recording_groups,
    session_cluster_bootstrap_delta,
    session_variation,
    stratified_auc,
    stratified_permutation_p,
    summarize_perturbations,
    within_session_percentiles,
    within_session_rows,
)
from scoring_v1 import combined_scores  # noqa: E402

FEATURES = ["duration_s", "mean_rms", "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_flatness_mean", "spectral_flatness_std", "spectral_rolloff_std"]
SR = 8000


def stage1_like(n_per_session=(20, 20, 20), seed=0):
    """Usable events plus excluded ones with overlapping reasons, in several sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, n in enumerate(n_per_session):
        for i in range(n):
            rows.append({"recording_file": f"s{s}_{i:02d}.wav", "event_id": 0, "session": f"S{s}",
                         **{f: rng.normal(1.0, 0.1) for f in FEATURES},
                         "flag_short_duration": False, "flag_close_neighbor": False,
                         "flag_recording_boundary": False, "flag_nonfinite_feature": False,
                         "usable": True, "exclusion_reasons": ""})
    specials = [("short_duration", (True, False, False)), ("close_neighbor", (False, True, False)),
                ("short_duration;close_neighbor", (True, True, False)),
                ("short_duration;recording_boundary", (True, False, True))]
    for s in range(len(n_per_session)):
        for k, (reasons, (short, close, boundary)) in enumerate(specials):
            rows.append({"recording_file": f"s{s}_x{k}.wav", "event_id": 1, "session": f"S{s}",
                         **{f: rng.normal(1.0, 0.1) for f in FEATURES},
                         "flag_short_duration": short, "flag_close_neighbor": close,
                         "flag_recording_boundary": boundary, "flag_nonfinite_feature": False,
                         "usable": False, "exclusion_reasons": reasons})
    return pd.DataFrame(rows)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.table = stage1_like()
        self.populations = assign_populations(self.table)

    def test_membership_follows_stage1_flags_and_keeps_overlap(self):
        self.assertEqual((self.populations["population"] == "excluded").sum(), 12)
        self.assertEqual(int(self.populations["in_too_short"].sum()), 9)        # 3 sessions x 3 short
        self.assertEqual(int(self.populations["in_close_neighbor"].sum()), 6)
        self.assertEqual(set(self.populations["exclusion_group"]),
                         {"usable", "too_short", "close_neighbor", "too_short+close_neighbor",
                          "too_short+recording_boundary"})

    def test_overlap_matrix_counts_co_membership(self):
        overlap = overlap_matrix(self.populations)
        self.assertEqual(overlap.loc["too_short", "close_neighbor"], 3)
        self.assertEqual(overlap.loc["too_short", "recording_boundary"], 3)
        self.assertEqual(overlap.loc["too_short", "too_short"], 9)

    def test_groups_are_not_simple_sums(self):
        groups = comparison_groups(self.populations)
        self.assertEqual(groups["excluded_all"].sum(), 12)
        self.assertEqual(groups["any_too_short"].sum() + groups["any_close_neighbor"].sum(), 15)  # > 12: overlap
        self.assertEqual(groups["only_too_short+close_neighbor"].sum(), 3)

    def test_inconsistent_stage1_flags_raise(self):
        broken = self.table.copy()
        broken.loc[0, "flag_short_duration"] = True       # usable but flagged
        with self.assertRaises(PopulationError):
            assign_populations(broken)
        broken = self.table.copy()
        broken.loc[broken.index[-1], "exclusion_reasons"] = "close_neighbor"   # text disagrees with flags
        with self.assertRaises(PopulationError):
            assign_populations(broken)


class HeldOutBaselineTests(unittest.TestCase):
    def setUp(self):
        self.table = stage1_like()
        self.sessions = self.table["session"]

    def test_baselines_use_usable_events_of_other_sessions_only(self):
        z, parameters, baselines = heldout_scores(self.table, FEATURES, self.sessions)
        excluded = set(zip(self.table.loc[~self.table["usable"], "recording_file"],
                           self.table.loc[~self.table["usable"], "event_id"]))
        session_of = dict(zip(zip(self.table["recording_file"], self.table["event_id"]), self.sessions))
        for session, baseline in baselines.items():
            keys = set(baseline.calibration_events)
            self.assertFalse(keys & excluded)                                   # no excluded event in any fit
            self.assertFalse(any(session_of[k] == session for k in keys))      # held-out session never in its fit
            self.assertEqual(baseline.n_calibration, int((self.table["usable"] & (self.sessions != session)).sum()))
        self.assertTrue(z.index.equals(self.table.index))

    def test_excluded_values_cannot_move_any_baseline(self):
        _, reference, _ = heldout_scores(self.table, FEATURES, self.sessions)
        changed = self.table.copy()
        changed.loc[~changed["usable"], FEATURES] += 100.0
        z, parameters, _ = heldout_scores(changed, FEATURES, self.sessions)
        pd.testing.assert_frame_equal(parameters, reference)
        self.assertTrue((z.loc[~changed["usable"]].abs() > 100).all().all())

    def test_scores_are_reproducible(self):
        first, _, _ = heldout_scores(self.table, FEATURES, self.sessions)
        second, _, _ = heldout_scores(self.table, FEATURES, self.sessions)
        pd.testing.assert_frame_equal(combined_scores(first, FEATURES), combined_scores(second, FEATURES))


class EffectSizeTests(unittest.TestCase):
    def test_cliffs_delta_and_shift(self):
        self.assertEqual(cliffs_delta([5, 6, 7], [1, 2, 3]), 1.0)
        self.assertEqual(cliffs_delta([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(cliffs_delta([1, 2], [3, 4]), -1.0)
        self.assertEqual(hodges_lehmann_shift([5, 6], [1, 2]), 4.0)

    def test_session_controlled_statistics(self):
        rng = np.random.default_rng(1)
        sessions = np.repeat(["A", "B"], 30)
        offset = np.where(sessions == "B", 10.0, 0.0)
        group = np.tile(np.r_[np.ones(5, bool), np.zeros(25, bool)], 2)
        scores = offset + rng.normal(size=60) + np.where(group, 3.0, 0.0)
        self.assertGreater(stratified_auc(scores, sessions, group, ~group), 0.95)
        percentiles = within_session_percentiles(scores, sessions, group, ~group)
        self.assertEqual(len(percentiles), 10)
        self.assertGreater(np.median(percentiles), 0.9)
        rows = within_session_rows(scores, sessions, group, ~group)
        self.assertEqual(list(rows["session"]), ["A", "B"])
        self.assertTrue((rows["difference"] > 0).all())
        self.assertLess(stratified_permutation_p(scores, sessions, group, ~group, permutations=200, seed=2), 0.05)

    def test_no_difference_is_not_flagged_and_session_offset_is_not_mistaken(self):
        rng = np.random.default_rng(3)
        sessions = np.repeat(["A", "B"], 30)
        group = np.r_[np.zeros(30, bool), np.ones(30, bool)] & (rng.random(60) < 0.3)
        scores = np.where(sessions == "B", 10.0, 0.0) + rng.normal(size=60)
        # The pooled comparison is confounded by session B; the within-session one is not.
        self.assertGreater(cliffs_delta(scores[group], scores[~group]), 0.5)
        self.assertTrue(np.isnan(stratified_auc(scores, sessions, group, ~group)) or
                        abs(stratified_auc(scores, sessions, group, ~group) - 0.5) < 0.3)

    def test_bootstrap_interval_is_deterministic_and_contains_estimate(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(1, 1, 40), rng.normal(0, 1, 80)
        sa, sb = np.repeat(["A", "B"], 20), np.repeat(["A", "B"], 40)
        first = session_cluster_bootstrap_delta(a, sa, b, sb, draws=300, seed=5)
        self.assertEqual(first, session_cluster_bootstrap_delta(a, sa, b, sb, draws=300, seed=5))
        self.assertLessEqual(first[0], cliffs_delta(a, b))
        self.assertGreaterEqual(first[1], cliffs_delta(a, b))

    def test_compare_to_reference_rows(self):
        table = stage1_like()
        populations = assign_populations(table)
        z, _, _ = heldout_scores(table, FEATURES, table["session"])
        frame = pd.concat([table[["session"]], populations, combined_scores(z, FEATURES)], axis=1)
        effects = compare_to_reference(frame, comparison_groups(populations), draws=50, permutations=50)
        self.assertIn("excluded_all", set(effects["group"]))
        self.assertEqual(set(effects["score"]), {"mean_abs_z", "rms_z", "max_abs_z"})
        self.assertTrue(((effects["auc"] >= 0) & (effects["auc"] <= 1)).all())


class AttributionTests(unittest.TestCase):
    def test_single_feature_shift_is_attributed_to_that_feature(self):
        rng = np.random.default_rng(6)
        z = pd.DataFrame(rng.normal(size=(200, 7)), columns=[f"z_{f}" for f in FEATURES])
        group = np.r_[np.ones(40, bool), np.zeros(160, bool)]
        z.loc[group, "z_duration_s"] -= 6.0
        attribution = feature_attribution(z, {"usable": ~group, "short": group}, FEATURES).set_index(["group", "feature"])
        self.assertGreater(attribution.loc[("short", "duration_s"), "univariate_auc_abs_z"], 0.95)
        self.assertLess(abs(attribution.loc[("short", "mean_rms"), "univariate_auc_abs_z"] - 0.5), 0.15)
        self.assertLess(abs(attribution.loc[("short", "duration_s"), "rms_z_auc_without_feature"] - 0.5), 0.15)
        self.assertGreater(attribution.loc[("short", "duration_s"), "argmax_fraction"], 0.9)
        self.assertLess(attribution.loc[("short", "duration_s"), "median_z"], -4)

    def test_duration_matching_isolates_non_duration_deviation(self):
        rng = np.random.default_rng(8)
        n = 120
        frame = pd.DataFrame({"duration_s": rng.uniform(0.5, 2.5, n)})
        z = pd.DataFrame(rng.normal(size=(n, 7)), columns=[f"z_{f}" for f in FEATURES])
        group = np.zeros(n, bool)
        group[:15] = True
        frame.loc[group, "duration_s"] = rng.uniform(0.5, 0.8, 15)       # short-ish, like close neighbours
        z.loc[group, "z_duration_s"] -= 3.0                               # duration alone deviates
        per_event, summary = duration_matched_comparison(frame, z, group, ~group, FEATURES, k=5)
        self.assertEqual(len(per_event), 15)
        self.assertLess(abs(summary["cliffs_delta_vs_matched"]), 0.35)   # nothing beyond duration
        self.assertLess(summary["median_duration_gap_s"], 0.2)
        z.loc[group, "z_spectral_centroid_std"] += 4.0                    # now a second feature deviates
        _, shifted = duration_matched_comparison(frame, z, group, ~group, FEATURES, k=5)
        self.assertGreater(shifted["cliffs_delta_vs_matched"], 0.7)
        self.assertGreater(shifted["frac_paired_difference_positive"], 0.8)

    def test_session_variation_rows(self):
        frame = pd.DataFrame({"population": "usable", "session": np.repeat(["A", "B"], 10),
                              "mean_abs_z": np.r_[np.ones(10), 2 * np.ones(10)],
                              "rms_z": np.r_[np.ones(10), 2 * np.ones(10)],
                              "max_abs_z": np.r_[np.ones(10), 2 * np.ones(10)]})
        rows = session_variation(frame).set_index(["session", "score"])
        self.assertEqual(rows.loc[("B", "rms_z"), "median_ratio_to_other_usable"], 2.0)
        self.assertEqual(rows.loc[("A", "rms_z"), "cliffs_delta_vs_other_usable"], -1.0)

    def test_recording_groups(self):
        recordings = pd.DataFrame({"recording_file": ["a.wav", "b.wav", "c.wav"]})
        events = pd.DataFrame({"recording_file": ["a.wav", "b.wav"], "usable": [True, False]})
        self.assertEqual(recording_groups(recordings, events).tolist(),
                         ["usable_event", "only_excluded_events", "no_detected_event"])


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        t = np.arange(SR) / SR
        self.segment = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.normal(size=SR)).astype(np.float32)

    def test_transformations_are_deterministic_and_do_not_mutate_input(self):
        original = self.segment.copy()
        for transform, magnitude in perturbation_plan():
            first = perturb(self.segment, transform, magnitude, np.random.default_rng([1, 2]))
            second = perturb(self.segment, transform, magnitude, np.random.default_rng([1, 2]))
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(self.segment, original)
        with self.assertRaises(ValueError):
            perturb(self.segment, "reverse", 1.0, np.random.default_rng(0))

    def test_transformation_definitions(self):
        np.testing.assert_allclose(apply_gain(self.segment, 2.0), 2.0 * self.segment.astype(np.float64))
        tilted = apply_tilt(self.segment, 0.9)
        self.assertAlmostEqual(np.sqrt(np.mean(tilted ** 2)), np.sqrt(np.mean(self.segment.astype(float) ** 2)))
        noisy = add_white_noise(self.segment, 10.0, np.random.default_rng(3))
        noise_power = np.mean((noisy - self.segment) ** 2)
        self.assertAlmostEqual(10 * np.log10(np.mean(self.segment.astype(float) ** 2) / noise_power), 10.0, delta=0.3)
        np.testing.assert_array_equal(perturb(self.segment, "identity", 0.0, np.random.default_rng(0)),
                                      self.segment.astype(np.float64))

    def test_existing_measurements_respond_predictably(self):
        base = measure_segment(self.segment, SR)
        louder = measure_segment(apply_gain(self.segment, 2.0), SR)
        self.assertAlmostEqual(louder["mean_rms"] / base["mean_rms"], 2.0, places=3)
        self.assertAlmostEqual(louder["spectral_centroid_mean"], base["spectral_centroid_mean"], places=4)
        brighter = measure_segment(apply_tilt(self.segment, 0.9), SR)
        self.assertGreater(brighter["spectral_centroid_mean"], base["spectral_centroid_mean"])
        noisier = measure_segment(add_white_noise(self.segment, 10.0, np.random.default_rng(1)), SR)
        self.assertGreater(noisier["spectral_flatness_mean"], base["spectral_flatness_mean"])

    def test_perturbation_summary_monotonicity(self):
        rows = []
        for row in range(4):
            for transform, magnitude in perturbation_plan():
                gain = magnitude if transform == "gain" else 1.0
                values = {f"z_{f}": 0.0 for f in FEATURES}
                values["z_mean_rms"] = np.log2(gain) + row * 0.1
                rows.append({"row": row, "session": "S", "transform": transform, "magnitude": magnitude, **values})
        frame = pd.DataFrame(rows)
        frame = pd.concat([frame, combined_scores(frame[[f"z_{f}" for f in FEATURES]], FEATURES)], axis=1)
        changes, monotonic, _ = summarize_perturbations(frame, FEATURES)
        gain_rms = monotonic[(monotonic["transform"] == "gain") & (monotonic["feature"] == "mean_rms")].iloc[0]
        self.assertEqual(gain_rms["frac_events_monotone"], 1.0)
        distance = monotonic[(monotonic["transform"] == "gain") & (monotonic["feature"] == "DISTANCE_FROM_OWN_Z")].iloc[0]
        self.assertEqual(distance["frac_events_monotone"], 1.0)
        doubled = changes[(changes["transform"] == "gain") & np.isclose(changes["magnitude"], 2.0)].iloc[0]
        self.assertAlmostEqual(doubled["median_delta_z_mean_rms"], 1.0)
        self.assertIn(1.0, PERTURBATION_ORDER["gain"])


if __name__ == "__main__":
    unittest.main()
