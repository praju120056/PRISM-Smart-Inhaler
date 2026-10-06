"""Stage 9 assessment-layer tests (synthetic data and a stub detector; no data/ access)."""

from fractions import Fraction
from pathlib import Path
import copy
import json
import math
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from feature_extractor import create_feature_windows  # noqa: E402
from librosa_extractor import extract_features_from_audio  # noqa: E402
from post_event import DEFAULT_MODEL_PATH  # noqa: E402
from prism_assessment import (  # noqa: E402
    ALPHA,
    EVENT_KEYS,
    OUTPUT_KEYS,
    STRIDE_S,
    VARIANT_SHIFTS,
    WINDOW_S,
    AssessmentError,
    AssessmentReference,
    assess_recording,
    bootstrap_band,
    reference_cut,
    fit_reference,
    load_reference,
    output_json_schema,
    segmentation_variants,
    tail_count_limit,
    tail_probability,
    validate_assessment_output,
    window_count,
    window_index_bounds,
    write_reference,
)
from prism_inference import V2_FEATURES, FrozenBaseline  # noqa: E402

SR = 8000


def synthetic_baseline() -> FrozenBaseline:
    center, mad = (0.37, 0.13, 0.04, 0.08), (0.019, 0.02, 0.006, 0.011)
    return FrozenBaseline(baseline_id="test", features=V2_FEATURES, center=center, mad=mad,
                          scale=tuple(1.4826 * m for m in mad), mad_scale=1.4826, n_events=100, n_sessions=5, source={})


def synthetic_reference(categorical=True, n=200, seed=0) -> AssessmentReference:
    rng = np.random.default_rng(seed)
    scores = np.sort(np.abs(rng.normal(1.0, 0.4, n)))
    sessions = tuple(f"s{i % 7}" for i in range(n))
    low, high, _ = bootstrap_band(scores, sessions, draws=200, seed=1)
    return AssessmentReference(reference_id="test-ref", baseline=synthetic_baseline(),
                               calibration_scores=tuple(float(s) for s in scores), calibration_sessions=sessions,
                               alpha=ALPHA, cut=reference_cut(scores), band=(low, high), categorical=categorical)


def synthetic_table(sessions=6, per_session=20, seed=2) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = sessions * per_session
    values = rng.normal(size=(n, 4)) * [0.02, 0.03, 0.006, 0.01] + [0.37, 0.13, 0.04, 0.08]
    table = pd.DataFrame(values, columns=list(V2_FEATURES))
    table["usable"] = True
    table["recording_file"] = [f"r{i}.wav" for i in range(n)]
    table["event_id"] = 0
    return table, pd.Series([f"S{i // per_session}" for i in range(n)], index=table.index)


class StubDetector:
    """Inhale in the given window-index ranges, Noise elsewhere (mimics OnnxEventClassifier)."""

    window_frames = config.WINDOW_SIZE
    class_names = tuple(config.LABEL_NAMES)
    model_path = Path(DEFAULT_MODEL_PATH)

    def __init__(self, inhale_ranges=()):
        self.inhale = {i for a, b in inhale_ranges for i in range(a, b + 1)}

    def predict_probabilities(self, windows):
        probabilities = np.tile(np.array([0.03, 0.03, 0.04, 0.90], np.float32), (len(windows), 1))
        for i in self.inhale:
            if i < len(windows):
                probabilities[i] = [0.02, 0.03, 0.90, 0.05]
        return probabilities


def noise_recording(seconds=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.05 * rng.standard_normal(len(t)) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t))).astype(np.float32)


class ReferenceTailTests(unittest.TestCase):
    def test_tail_count_limit_is_exact(self):
        self.assertEqual(tail_count_limit(18), -1)           # (1 + 0) / 19 > 0.05: nothing can be outside
        self.assertEqual(tail_count_limit(19), 0)
        self.assertEqual(tail_count_limit(318), 14)
        self.assertEqual(tail_count_limit(399), 19)          # 0.05 * 400 = 20 exactly, no float rounding
        self.assertEqual(tail_count_limit(99, Fraction(1, 100)), 0)

    def test_cut_and_probability_agree(self):
        rng = np.random.default_rng(3)
        for n in (19, 20, 57, 318, 399):
            calibration = np.sort(rng.gamma(2.0, 0.5, n))
            cut = reference_cut(calibration)
            probes = np.concatenate([calibration, (calibration[1:] + calibration[:-1]) / 2, [0.0, 1e6]])
            for s in probes:
                self.assertEqual(tail_probability(s, calibration) <= float(ALPHA), s > cut, (n, s))

    def test_too_small_reference_has_infinite_cut(self):
        self.assertEqual(reference_cut(np.linspace(0, 1, 18)), math.inf)

    def test_tail_probability_formula(self):
        calibration = np.array([0.5, 1.0, 1.5, 2.0])
        self.assertAlmostEqual(tail_probability(1.0, calibration), (1 + 3) / 5)
        self.assertAlmostEqual(tail_probability(3.0, calibration), 1 / 5)
        self.assertAlmostEqual(tail_probability(0.0, calibration), 1.0)

    def test_bootstrap_band_is_deterministic_and_resamples_sessions(self):
        rng = np.random.default_rng(4)
        scores = rng.gamma(2.0, 0.5, 120)
        sessions = [f"s{i % 6}" for i in range(120)]
        a = bootstrap_band(scores, sessions, draws=300, seed=9)
        b = bootstrap_band(scores, sessions, draws=300, seed=9)
        self.assertEqual(a[:2], b[:2])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertLessEqual(a[0], a[1])
        self.assertFalse(np.array_equal(a[2], bootstrap_band(scores, sessions, draws=300, seed=10)[2]))


class ReferenceTests(unittest.TestCase):
    def test_fit_reference_uses_leave_one_session_out_calibration(self):
        table, sessions = synthetic_table()
        reference = fit_reference(table, sessions, "r", categorical=True, draws=100)
        self.assertEqual(len(reference.calibration_scores), len(table))
        self.assertEqual(reference.cut, reference_cut(reference.calibration_scores))
        # Recompute one event's calibration score by hand: baseline fitted without its session.
        from baseline_v1 import fit_baseline

        event = table.index[0]
        others = table[sessions != sessions[event]]
        fitted = fit_baseline(others, V2_FEATURES)
        z = fitted.z_scores(table.loc[[event]]).to_numpy()[0]
        expected = math.sqrt(float(np.mean(z ** 2)))
        self.assertTrue(np.isclose(reference.calibration_scores, expected, rtol=0, atol=1e-14).any())

    def test_excluded_sessions_never_enter_the_reference(self):
        table, sessions = synthetic_table()
        reference = fit_reference(table, sessions, "r", categorical=True, exclude_sessions=["S0"], draws=50)
        self.assertNotIn("S0", reference.calibration_sessions)
        self.assertEqual(len(reference.calibration_scores), int((sessions != "S0").sum()))
        self.assertEqual(reference.baseline.n_sessions, 5)

    def test_reference_json_round_trip_and_validation(self):
        reference = synthetic_reference()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ref.json"
            write_reference(reference, path)
            self.assertEqual(load_reference(path), reference)
            data = json.loads(path.read_text())
            data["cut"] = data["cut"] + 0.01
            with self.assertRaises(AssessmentError):
                AssessmentReference.from_dict(data)
            data = json.loads(path.read_text())
            data["contract_version"] = "other"
            with self.assertRaises(AssessmentError):
                AssessmentReference.from_dict(data)

    def test_reference_rejects_unsorted_or_invalid_calibration(self):
        good = synthetic_reference()
        with self.assertRaises(AssessmentError):
            AssessmentReference(**{**good.__dict__, "calibration_scores": tuple(reversed(good.calibration_scores))})
        with self.assertRaises(AssessmentError):
            AssessmentReference(**{**good.__dict__, "band": (2.0, 1.0)})

    def test_reliability_uses_the_band(self):
        reference = synthetic_reference()
        low, high = reference.band
        self.assertEqual(reference.reliability([high + 0.1, high + 0.2]), "STABLE")
        self.assertEqual(reference.reliability([low - 0.1, low]), "STABLE")
        self.assertEqual(reference.reliability([high + 0.1, low - 0.1]), "BORDERLINE")
        self.assertEqual(reference.reliability([(low + high) / 2]), "BORDERLINE")


class SegmentationTests(unittest.TestCase):
    def test_window_index_bounds_round_trip(self):
        for first, last in ((0, 0), (10, 90), (37, 300)):
            start, end = first * STRIDE_S, last * STRIDE_S + WINDOW_S
            self.assertEqual(window_index_bounds(start, end), (first, last))
        with self.assertRaises(AssessmentError):
            window_index_bounds(0.5001, 1.2)

    def test_window_count_matches_the_extractor(self):
        rng = np.random.default_rng(5)
        for n in (1536, 1600, 8000, 12345):
            features = extract_features_from_audio(0.01 * rng.standard_normal(n).astype(np.float32), SR)
            windows = create_feature_windows(features, window_size=config.WINDOW_SIZE, stride=config.WINDOW_STRIDE)
            self.assertEqual(window_count(n), len(windows), n)

    def test_variants_move_each_boundary_by_one_stride(self):
        n = 10 * SR
        first, last = 50, 120
        variants = segmentation_variants(first * STRIDE_S, last * STRIDE_S + WINDOW_S, n)
        expected = {((first + ds) * STRIDE_S, min((last + de) * STRIDE_S + WINDOW_S, n / SR)) for ds, de in VARIANT_SHIFTS}
        self.assertEqual(set(variants), expected)
        self.assertEqual(len(variants), 8)

    def test_variants_stay_inside_the_recording(self):
        n = 10 * SR
        at_start = segmentation_variants(0.0, 40 * STRIDE_S + WINDOW_S, n)
        self.assertTrue(all(start >= 0 for start, _ in at_start))
        self.assertEqual(len(at_start), 5)                    # ds = -1 is impossible
        last = window_count(n) - 1
        at_end = segmentation_variants(400 * STRIDE_S, last * STRIDE_S + WINDOW_S, n)
        self.assertEqual(len(at_end), 5)                      # de = +1 is impossible
        single = segmentation_variants(100 * STRIDE_S, 100 * STRIDE_S + WINDOW_S, n)
        self.assertTrue(all(end - start >= WINDOW_S - 1e-12 for start, end in single))


class AssessRecordingTests(unittest.TestCase):
    def setUp(self):
        self.audio = noise_recording()

    def test_scoreable_and_short_events(self):
        reference = synthetic_reference()
        detector = StubDetector([(100, 180), (300, 305)])     # ~1.48 s scoreable, ~0.28 s too short
        out = assess_recording(self.audio, SR, reference, detector=detector, input_domain="reference_dataset")
        self.assertEqual(tuple(out), OUTPUT_KEYS)
        self.assertEqual(out["n_events"], 2)
        first, second = out["events"]
        self.assertEqual(tuple(first), EVENT_KEYS)
        self.assertEqual(first["scoreability"], "SCOREABLE")
        self.assertIn(first["assessment"], ("WITHIN_REFERENCE_RANGE", "OUTSIDE_REFERENCE_RANGE"))
        self.assertEqual(first["assessment"] == "OUTSIDE_REFERENCE_RANGE", first["aggregate_deviation"] > reference.cut)
        self.assertAlmostEqual(first["reference_tail_probability"], reference.tail_probability(first["aggregate_deviation"]))
        self.assertIn(first["assessment_reliability"], ("STABLE", "BORDERLINE"))
        self.assertEqual(first["segmentation_variants"], 8)
        lo, hi = first["segmentation_deviation_range"]
        self.assertLessEqual(lo, first["aggregate_deviation"])
        self.assertLessEqual(first["aggregate_deviation"], hi)
        self.assertEqual(first["start_sample"], int(np.floor(first["start_time"] * SR)))
        self.assertEqual(second["scoreability"], "NOT_SCOREABLE")
        self.assertEqual(second["assessment"], "NOT_ASSESSED")
        self.assertIn("short_duration", second["not_scoreable_reasons"])
        self.assertIsNone(second["reference_tail_probability"])
        json.dumps(out, allow_nan=False)

    def test_continuous_only_reference(self):
        out = assess_recording(self.audio, SR, synthetic_reference(categorical=False),
                               detector=StubDetector([(100, 180)]))
        event = out["events"][0]
        self.assertEqual(event["assessment"], "CONTINUOUS_ONLY")
        self.assertIsNone(event["assessment_reliability"])
        self.assertIsNotNone(event["reference_tail_probability"])
        self.assertEqual(out["n_outside_reference_range"], 0)

    def test_no_inhalation_and_input_error(self):
        reference = synthetic_reference()
        silent = assess_recording(self.audio, SR, reference, detector=StubDetector())
        self.assertEqual(silent["recording_status"], "NO_INHALATION_DETECTED")
        self.assertEqual(silent["events"], [])
        error = assess_recording(np.zeros(16000, np.float32), 16000, reference, detector=StubDetector())
        self.assertEqual((error["recording_status"], error["error"]), ("INPUT_ERROR", "unsupported_sample_rate"))

    def test_validation_rejects_tampered_outputs(self):
        reference = synthetic_reference()
        out = assess_recording(self.audio, SR, reference, detector=StubDetector([(100, 180)]))
        flipped = copy.deepcopy(out)
        event = flipped["events"][0]
        event["assessment"] = ("WITHIN_REFERENCE_RANGE" if event["assessment"] == "OUTSIDE_REFERENCE_RANGE"
                               else "OUTSIDE_REFERENCE_RANGE")
        flipped["n_outside_reference_range"] = int(event["assessment"] == "OUTSIDE_REFERENCE_RANGE")
        with self.assertRaises(AssessmentError):
            validate_assessment_output(flipped)
        missing = copy.deepcopy(out)
        del missing["events"][0]["reason"]
        with self.assertRaises(AssessmentError):
            validate_assessment_output(missing)
        wrong_score = copy.deepcopy(out)
        wrong_score["events"][0]["aggregate_deviation"] += 0.5
        with self.assertRaises(AssessmentError):
            validate_assessment_output(wrong_score)

    def test_json_schema_matches_keys(self):
        schema = output_json_schema()
        self.assertEqual(schema["required"], list(OUTPUT_KEYS))
        self.assertEqual(schema["properties"]["events"]["items"]["required"], list(EVENT_KEYS))


if __name__ == "__main__":
    unittest.main()
