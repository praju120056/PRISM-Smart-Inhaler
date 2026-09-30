"""Stage 7 representation-analysis tests (synthetic data; no data/ access)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from representation_analysis import (  # noqa: E402
    CANDIDATE_FEATURES,
    DURATION_GATE_S,
    EW_MAP,
    RELATIVE_LEVEL,
    build_representations,
    contribution_rows,
    excluded_diagnostics,
    frame_spectral_statistics,
    mcd_loso,
    monotone_either,
    monotone_outward,
    multivariate_gate,
    perturbation_response,
    recording_background_rms,
    relative_level_db,
    scoreable_mask,
    session_dependence,
)
from natural_population_analysis import heldout_scores  # noqa: E402

V1 = ["duration_s", "mean_rms", "spectral_centroid_mean", "spectral_centroid_std",
      "spectral_flatness_mean", "spectral_flatness_std", "spectral_rolloff_std"]
SR = 8000


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.representations = build_representations(V1)

    def test_representations_are_derived_from_v1(self):
        r = self.representations
        self.assertEqual(r["R0_current7"], V1)
        self.assertNotIn("duration_s", r["R1_no_duration"])
        self.assertEqual(r["R1_no_duration"], r["R2_duration_as_usability"])
        self.assertNotIn("spectral_flatness_std", r["R3_no_flatness_std"])
        self.assertIn("ew_spectral_flatness_std", r["R4_ew_flatness_std"])
        self.assertEqual(len(r["R4_ew_flatness_std"]), 7)
        self.assertNotIn("mean_rms", r["R5_no_mean_rms"])
        self.assertIn(RELATIVE_LEVEL, r["R6_relative_level"])
        self.assertEqual(r["R8_robust_candidate"], [RELATIVE_LEVEL, *EW_MAP.values()])
        for features in r.values():
            self.assertTrue(set(features) <= set(V1) | set(CANDIDATE_FEATURES))

    def test_duration_gate_only_for_r2(self):
        frame = pd.DataFrame({"duration_s": [0.3, 0.5, 1.2]})
        np.testing.assert_array_equal(scoreable_mask(frame, "R2_duration_as_usability"), [False, True, True])
        np.testing.assert_array_equal(scoreable_mask(frame, "R1_no_duration"), [True, True, True])
        self.assertEqual(DURATION_GATE_S, 0.5)


class CandidateFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        t = np.arange(SR) / SR
        envelope = np.clip(np.sin(np.pi * t), 0, None)
        self.segment = (envelope * (0.3 * rng.normal(size=SR))).astype(np.float32)

    def test_unweighted_statistics_reproduce_the_extractor(self):
        from librosa_extractor import extract_features_from_audio

        stats_ = frame_spectral_statistics(self.segment, SR)
        spectral = extract_features_from_audio(self.segment, SR)[:, 120:]
        self.assertAlmostEqual(stats_["uw_spectral_centroid_mean"], float(spectral[:, 0].mean()), places=6)
        self.assertAlmostEqual(stats_["uw_spectral_flatness_std"], float(spectral[:, 1].std()), places=6)
        self.assertLessEqual(stats_["effective_frames"], stats_["frames"])

    def test_energy_weighting_downweights_quiet_frames(self):
        rng = np.random.default_rng(1)
        loud = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)          # tonal, low flatness
        quiet = 1e-3 * rng.normal(size=SR // 4)                             # noise-like, high flatness
        segment = np.concatenate([quiet, loud, quiet]).astype(np.float32)
        stats_ = frame_spectral_statistics(segment, SR)
        self.assertLess(stats_["ew_spectral_flatness_mean"], stats_["uw_spectral_flatness_mean"])
        self.assertLess(stats_["ew_spectral_flatness_std"], stats_["uw_spectral_flatness_std"])

    def test_energy_weighting_is_gain_invariant(self):
        base = frame_spectral_statistics(self.segment, SR)
        louder = frame_spectral_statistics(self.segment * 2, SR)
        for key in ("ew_spectral_centroid_mean", "ew_spectral_flatness_std", "ew_spectral_rolloff_std"):
            self.assertAlmostEqual(base[key], louder[key], places=5)

    def test_silent_segment_raises(self):
        with self.assertRaises(ValueError):
            frame_spectral_statistics(np.zeros(SR, np.float32), SR)

    def test_relative_level_cancels_recording_gain_but_not_event_gain(self):
        rng = np.random.default_rng(2)
        recording = (0.05 * rng.normal(size=4 * SR)).astype(np.float32)
        event_rms = 0.2
        base = relative_level_db(event_rms, recording_background_rms(recording, SR))
        recording_gain = relative_level_db(2 * event_rms, recording_background_rms(2 * recording, SR))
        event_gain = relative_level_db(2 * event_rms, recording_background_rms(recording, SR))
        self.assertAlmostEqual(base, recording_gain, places=5)
        self.assertAlmostEqual(event_gain - base, 20 * np.log10(2), places=5)
        with self.assertRaises(ValueError):
            relative_level_db(0.1, 0.0)


def synthetic_table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(4):
        for i in range(25):
            rows.append({"recording_file": f"s{s}_{i}.wav", "event_id": 0, "session": f"S{s}", "usable": True,
                         **{f: rng.normal(s * 0.5, 1.0) for f in ("a", "b", "c")}})
        for i in range(4):
            rows.append({"recording_file": f"s{s}_x{i}.wav", "event_id": 1, "session": f"S{s}", "usable": False,
                         **{f: rng.normal(3.0, 1.0) for f in ("a", "b", "c")}})
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_session_dependence_detects_offsets(self):
        rng = np.random.default_rng(3)
        sessions = np.repeat(["A", "B", "C"], 30)
        flat = session_dependence(rng.normal(size=90), sessions, np.ones(90, bool))
        shifted = session_dependence(rng.normal(size=90) + np.repeat([0, 3, 6], 30), sessions, np.ones(90, bool))
        self.assertLess(flat["session_epsilon_squared"], 0.1)
        self.assertGreater(shifted["session_epsilon_squared"], 0.7)
        self.assertGreater(shifted["max_abs_session_vs_rest_delta"], 0.9)

    def test_excluded_diagnostics_respect_the_gate(self):
        rng = np.random.default_rng(4)
        n = 60
        scores = np.r_[rng.normal(1, 0.2, 50), rng.normal(3, 0.2, 10)]
        sessions = np.tile(["A", "B"], 30)
        groups = {"usable": np.r_[np.ones(50, bool), np.zeros(10, bool)],
                  "excluded_all": np.r_[np.zeros(50, bool), np.ones(10, bool)],
                  "any_too_short": np.r_[np.zeros(50, bool), np.ones(5, bool), np.zeros(5, bool)],
                  "only_close_neighbor": np.r_[np.zeros(55, bool), np.ones(5, bool)]}
        eligible = ~groups["any_too_short"]
        rows = {r["group"]: r for r in excluded_diagnostics(scores, sessions, groups, eligible, draws=100)}
        self.assertEqual(rows["excluded_all"]["n_group"], 5)            # too-short events are not scoreable
        self.assertEqual(rows["any_too_short"]["n_group"], 0)
        self.assertGreater(rows["only_close_neighbor"]["cliffs_delta"], 0.9)

    def test_contribution_rows(self):
        z = pd.DataFrame({"z_a": [3.0, 0.1], "z_b": [0.1, 2.0]})
        rows = contribution_rows(z, ["a", "b"], {"all": np.array([True, True])})
        shares = {r["feature"]: r["argmax_fraction"] for r in rows}
        self.assertEqual(shares, {"a": 0.5, "b": 0.5})


class PerturbationTests(unittest.TestCase):
    def test_monotonicity_helpers(self):
        matrix = np.array([[2.0, 1.0, 0.0, 1.0, 3.0], [2.0, 1.0, 0.0, 2.0, 1.0]])
        np.testing.assert_array_equal(monotone_outward(matrix, 2), [True, False])
        np.testing.assert_array_equal(monotone_either(np.array([[1, 2, 3], [1, 3, 2]])), [True, False])

    def test_float_jitter_is_not_counted_as_non_monotone(self):
        jitter = np.array([[0.0, 3e-6, -2e-6, 1e-6]])           # an invariant feature under float32 rounding
        np.testing.assert_array_equal(monotone_either(jitter), [True])
        real = np.array([[0.0, 0.3, -0.2, 0.1]])
        np.testing.assert_array_equal(monotone_either(real), [False])

    def test_perturbation_response_measures_dominance_and_direction(self):
        rows = []
        for row in range(10):
            session = "A" if row < 5 else "B"
            plan = [("identity", 0.0)] + [("gain", g) for g in (0.5, 1 / np.sqrt(2), np.sqrt(2), 2.0)] + \
                   [("noise", s) for s in (30.0, 20.0, 10.0)] + [("tilt", a) for a in (-0.5, 0.5, 0.9)] + \
                   [("recording_gain", g) for g in (0.5, 2.0)]
            for transform, magnitude in plan:
                level = np.log2(magnitude) if transform in ("gain", "recording_gain") else 0.0
                rows.append({"row": row, "session": session, "transform": transform, "magnitude": magnitude,
                             "z_a": 1.0 + level, "z_b": 1.0})
        frame = pd.DataFrame(rows)
        result = perturbation_response(frame, {"both": ["a", "b"], "b_only": ["b"]}, min_session_events=5)
        doubled = result[(result["representation"] == "both") & (result["transform"] == "gain")
                         & np.isclose(result["magnitude"], 2.0)].iloc[0]
        self.assertEqual(doubled["median_single_feature_dominance"], 1.0)
        self.assertEqual(doubled["most_common_dominant_feature"], "a")
        self.assertGreater(doubled["median_delta_rms_z"], 0)
        self.assertEqual(doubled["frac_sessions_median_delta_positive"], 1.0)
        self.assertEqual(doubled["family_distance_monotone_fraction"], 1.0)
        invariant = result[(result["representation"] == "b_only") & (result["transform"] == "gain")]
        self.assertTrue((invariant["median_delta_rms_z"] == 0).all())


class MultivariateTests(unittest.TestCase):
    def test_mcd_is_fitted_without_the_held_out_session_and_gate_reports(self):
        table = synthetic_table()
        sessions = table["session"]
        z, _, baselines = heldout_scores(table, ["a", "b", "c"], sessions)
        diagnostics, scores, references, models = mcd_loso(table, z, sessions, baselines, {"abc": ["a", "b", "c"]},
                                                           seed=1)
        self.assertEqual(set(diagnostics["fold"]), set(sessions))
        for _, row in diagnostics.iterrows():
            self.assertEqual(row["n_train"], int((table["usable"] & (sessions != row["fold"])).sum()))
            self.assertGreaterEqual(row["condition_number"], 1.0)
        self.assertEqual(len(scores["abc"]), len(table))
        self.assertTrue(np.isfinite(scores["abc"]).all())
        gate = multivariate_gate(diagnostics)
        self.assertIn("passed", gate["abc"])
        self.assertEqual(gate["abc"]["min_n_train_per_feature"], 25.0)   # 75 usable training events / 3 features

    def test_mcd_is_deterministic(self):
        table = synthetic_table(seed=5)
        sessions = table["session"]
        z, _, baselines = heldout_scores(table, ["a", "b", "c"], sessions)
        first = mcd_loso(table, z, sessions, baselines, {"abc": ["a", "b", "c"]}, seed=2)[1]["abc"]
        second = mcd_loso(table, z, sessions, baselines, {"abc": ["a", "b", "c"]}, seed=2)[1]["abc"]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
