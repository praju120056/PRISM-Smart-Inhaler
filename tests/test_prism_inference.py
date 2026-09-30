"""Stage 8 inference-contract tests (synthetic audio and a stub detector; no data/ access)."""

from pathlib import Path
import copy
import json
import math
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from baseline_v1 import fit_baseline  # noqa: E402
from post_event import DEFAULT_MODEL_PATH, InhaleEvent, OnnxEventClassifier, analyze_inhalation  # noqa: E402
from prism_inference import (  # noqa: E402
    CONTRACT_VERSION,
    EVENT_KEYS,
    INPUT_ERRORS,
    INTERPRETATION,
    LEVEL_CHANNEL,
    MIN_SAMPLES,
    NOT_SCOREABLE_REASONS,
    OUTPUT_KEYS,
    RECORDING_STATUSES,
    V2_FEATURES,
    ContractError,
    FrozenBaseline,
    analyze_recording,
    anomaly_score,
    check_input,
    decode_pcm16le,
    event_measurements,
    feature_schema,
    frozen_from_fit,
    load_baseline,
    not_scoreable_reasons,
    output_json_schema,
    validate_output,
    write_baseline,
)

SR = 8000
STRIDE = config.WINDOW_STRIDE * config.LIBROSA_HOP_LENGTH / SR   # 0.016 s


def synthetic_baseline() -> FrozenBaseline:
    center, mad = (0.37, 0.13, 0.04, 0.08), (0.019, 0.02, 0.006, 0.011)
    return FrozenBaseline(baseline_id="test", features=V2_FEATURES, center=center, mad=mad,
                          scale=tuple(1.4826 * m for m in mad), mad_scale=1.4826, n_events=100, n_sessions=5, source={})


class StubDetector:
    """Mimics OnnxEventClassifier: Inhale in the given window-index ranges, Noise elsewhere."""

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


class FeatureOrderAndBaselineTests(unittest.TestCase):
    def test_v2_order_is_fixed(self):
        self.assertEqual(V2_FEATURES, ("spectral_centroid_mean", "spectral_flatness_mean",
                                       "spectral_centroid_std", "spectral_rolloff_std"))
        self.assertEqual([f["name"] for f in feature_schema()["anomaly_features_in_order"]], list(V2_FEATURES))
        self.assertEqual(output_json_schema()["properties"]["feature_order"]["const"], list(V2_FEATURES))

    def test_baseline_rejects_reordered_or_invalid_parameters(self):
        good = synthetic_baseline()
        with self.assertRaises(ContractError):
            FrozenBaseline(**{**good.__dict__, "features": tuple(reversed(V2_FEATURES))})
        with self.assertRaises(ContractError):
            FrozenBaseline(**{**good.__dict__, "mad": (0.0, *good.mad[1:])})
        with self.assertRaises(ContractError):
            FrozenBaseline(**{**good.__dict__, "scale": (1.0, *good.scale[1:])})
        with self.assertRaises(ContractError):
            FrozenBaseline(**{**good.__dict__, "center": (math.nan, *good.center[1:])})

    def test_baseline_json_round_trip_and_version_check(self):
        import tempfile

        baseline = synthetic_baseline()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "b.json"
            write_baseline(baseline, path)
            self.assertEqual(load_baseline(path), baseline)
            data = json.loads(path.read_text())
            data["contract_version"] = "other"
            with self.assertRaises(ContractError):
                FrozenBaseline.from_dict(data)

    def test_frozen_from_fit_reproduces_in_memory_z(self):
        rng = np.random.default_rng(1)
        table = pd.DataFrame(rng.normal(size=(60, 4)) * [0.02, 0.03, 0.006, 0.01] + [0.37, 0.13, 0.04, 0.08],
                             columns=list(V2_FEATURES))
        fitted = fit_baseline(table, V2_FEATURES)
        frozen = frozen_from_fit(fitted, "t", 3, {})
        np.testing.assert_allclose(frozen.z(table.to_numpy()), fitted.z_scores(table).to_numpy(), rtol=0, atol=1e-12)
        with self.assertRaises(ContractError):
            frozen_from_fit(fit_baseline(table, list(reversed(V2_FEATURES))), "t", 3, {})

    def test_score_is_rms_of_z(self):
        z = np.array([3.0, -4.0, 0.0, 0.0])
        self.assertAlmostEqual(anomaly_score(z), math.sqrt(25 / 4))


class InputTests(unittest.TestCase):
    def test_input_errors(self):
        ok = np.zeros(MIN_SAMPLES, np.float32)
        self.assertIsNone(check_input(ok, SR)[1])
        self.assertEqual(check_input(ok, 16000)[1], "unsupported_sample_rate")
        self.assertEqual(check_input(np.zeros(MIN_SAMPLES - 1), SR)[1], "shorter_than_one_detector_window")
        self.assertEqual(check_input(np.array([]), SR)[1], "empty_audio")
        self.assertEqual(check_input(np.zeros((2, 2, 2)), SR)[1], "invalid_shape")
        bad = ok.copy()
        bad[3] = np.nan
        self.assertEqual(check_input(bad, SR)[1], "nonfinite_audio")
        bad[3] = 1.5
        self.assertEqual(check_input(bad, SR)[1], "amplitude_out_of_range")
        self.assertTrue(set(INPUT_ERRORS) >= {"unsupported_sample_rate", "shorter_than_one_detector_window"})

    def test_pcm16_decoding_matches_soundfile_convention(self):
        samples = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
        decoded = decode_pcm16le(samples.tobytes())
        np.testing.assert_array_equal(decoded, samples.astype(np.float32) / 32768.0)
        self.assertEqual(decoded.dtype, np.float32)
        with self.assertRaises(ContractError):
            decode_pcm16le(b"\x00")

    def test_stereo_is_averaged_and_float64_equals_float32(self):
        mono = noise_recording(2.0)
        stereo = np.column_stack([mono, mono])
        np.testing.assert_array_equal(check_input(stereo, SR)[0], mono)
        detector, baseline = StubDetector([(10, 60)]), synthetic_baseline()
        a = analyze_recording(mono, SR, detector=detector, baseline=baseline)
        b = analyze_recording(mono.astype(np.float64), SR, detector=detector, baseline=baseline)
        self.assertEqual(a, b)


class FeatureExtractionTests(unittest.TestCase):
    def test_event_measurements_match_existing_extractor(self):
        from librosa_extractor import extract_features_from_audio

        audio = noise_recording(3.0, seed=2)
        event = InhaleEvent("Inhale", 0.4, 2.0, 1.6, 0.9, 0.9, 10, ())
        measured = event_measurements(audio, event)
        segment = audio[int(np.floor(0.4 * SR)):int(np.ceil(2.0 * SR))]
        spectral = extract_features_from_audio(segment, SR)[:, 120:123]
        self.assertEqual(measured["spectral_centroid_mean"], float(spectral[:, 0].mean()))
        self.assertEqual(measured["spectral_flatness_mean"], float(spectral[:, 1].mean()))
        self.assertEqual(measured["spectral_centroid_std"], float(spectral[:, 0].std()))
        self.assertEqual(measured["spectral_rolloff_std"], float(spectral[:, 2].std()))
        self.assertEqual(measured[LEVEL_CHANNEL], analyze_inhalation(audio, event, sample_rate=SR)["mean_rms"])

    def test_features_are_gain_invariant_and_finite_at_low_level(self):
        audio = noise_recording(3.0, seed=3)
        event = InhaleEvent("Inhale", 0.4, 2.0, 1.6, 0.9, 0.9, 10, ())
        loud, quiet = event_measurements(audio, event), event_measurements(audio * 1e-3, event)
        for feature in ("spectral_centroid_mean", "spectral_centroid_std", "spectral_rolloff_std"):
            self.assertAlmostEqual(loud[feature], quiet[feature], places=4)
        self.assertTrue(all(math.isfinite(v) for v in quiet.values()))
        silent = event_measurements(np.zeros_like(audio), event)     # digital silence: finite, zero level
        self.assertIsNotNone(silent)
        self.assertTrue(all(math.isfinite(silent[f]) for f in V2_FEATURES))
        self.assertEqual(silent[LEVEL_CHANNEL], 0.0)


class UsabilityAndStatusTests(unittest.TestCase):
    def setUp(self):
        self.baseline = synthetic_baseline()
        self.audio = noise_recording(10.0)

    def test_stage1_rule_reasons(self):
        # windows: D 0-50 (starts at 0), A 100-200 (1.6-3.4 s), B 215-260 (0.92 s, gap 0.04 s < 0.2), C 400-405 (0.28 s)
        output = analyze_recording(self.audio, SR, detector=StubDetector([(0, 50), (100, 200), (215, 260), (400, 405)]),
                                   baseline=self.baseline)
        reasons = [e["not_scoreable_reasons"] for e in output["events"]]
        self.assertEqual(reasons, [["recording_boundary"], ["close_neighbor"], ["close_neighbor"], ["short_duration"]])
        self.assertEqual(output["n_scored"], 0)
        self.assertTrue(all(e["anomaly_score"] is None and e["feature_z_scores"] is None for e in output["events"]))
        self.assertTrue(all(e["feature_values"] is not None for e in output["events"]))

    def test_scored_event_fields(self):
        output = analyze_recording(self.audio, SR, detector=StubDetector([(100, 200)]), baseline=self.baseline,
                                   recording_id="r1", recorded_at="2026-09-30T10:00:00Z")
        self.assertEqual(output["recording_status"], "EVENTS_DETECTED")
        event = output["events"][0]
        self.assertEqual(event["status"], "SCORE_ONLY")
        self.assertAlmostEqual(event["start_time"], 100 * STRIDE)
        self.assertAlmostEqual(event["end_time"], 200 * STRIDE + 0.2)
        self.assertEqual(list(event["feature_values"]), list(V2_FEATURES))
        z = self.baseline.z([event["feature_values"][f] for f in V2_FEATURES])
        self.assertEqual(event["anomaly_score"], anomaly_score(z))
        self.assertAlmostEqual(event["detector_confidence"], 0.9, places=6)
        self.assertEqual(output["interpretation"], INTERPRETATION)
        self.assertFalse(output["baseline_domain_validated"])

    def test_gap_of_exactly_one_window_is_not_close(self):
        # A ends at 200*0.016+0.2 = 3.4 s; B starts at window 225 = 3.6 s -> gap 0.2 s exactly (not < 0.2)
        output = analyze_recording(self.audio, SR, detector=StubDetector([(100, 200), (225, 300)]), baseline=self.baseline)
        self.assertEqual([e["status"] for e in output["events"]], ["SCORE_ONLY", "SCORE_ONLY"])

    def test_boundary_rule_uses_recording_end(self):
        events = [InhaleEvent("Inhale", 1.0, 3.0, 2.0, 0.9, 0.9, 1, ()), InhaleEvent("Inhale", 6.0, 9.995, 3.995, 0.9, 0.9, 1, ())]
        self.assertEqual(not_scoreable_reasons(events, 10.0, [True, True]), [[], ["recording_boundary"]])
        self.assertEqual(not_scoreable_reasons(events, 10.0, [False, True])[0], ["nonfinite_feature"])


class NoInhalationTests(unittest.TestCase):
    def test_no_detected_inhalation_is_a_recording_state(self):
        output = analyze_recording(noise_recording(5.0), SR, detector=StubDetector(), baseline=synthetic_baseline())
        self.assertEqual(output["recording_status"], "NO_INHALATION_DETECTED")
        self.assertEqual((output["events"], output["n_events"], output["n_scored"], output["error"]), ([], 0, 0, None))

    def test_silence_with_the_real_detector(self):
        output = analyze_recording(np.zeros(2 * SR, np.float32), SR, detector=OnnxEventClassifier(),
                                   baseline=synthetic_baseline())
        self.assertEqual(output["recording_status"], "NO_INHALATION_DETECTED")

    def test_input_error_has_no_events(self):
        output = analyze_recording(np.zeros(100, np.float32), SR, detector=StubDetector(), baseline=synthetic_baseline())
        self.assertEqual((output["recording_status"], output["error"], output["events"]),
                         ("INPUT_ERROR", "shorter_than_one_detector_window", []))


class OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.output = analyze_recording(noise_recording(10.0), SR, detector=StubDetector([(100, 200), (400, 405)]),
                                        baseline=synthetic_baseline())

    def assert_invalid(self, mutate):
        broken = copy.deepcopy(self.output)
        mutate(broken)
        with self.assertRaises(ContractError):
            validate_output(broken)

    def test_valid_output_and_key_order(self):
        validate_output(self.output)
        self.assertEqual(tuple(self.output), OUTPUT_KEYS)
        self.assertEqual(tuple(self.output["events"][0]), EVENT_KEYS)
        json.dumps(self.output, allow_nan=False)

    def test_invalid_outputs_are_rejected(self):
        self.assert_invalid(lambda o: o.update(extra=1))
        self.assert_invalid(lambda o: o.update(recording_status="ANOMALY"))
        self.assert_invalid(lambda o: o["events"][0].update(status="NORMAL"))
        self.assert_invalid(lambda o: o["events"][0].update(anomaly_score=None))
        self.assert_invalid(lambda o: o["events"][0].update(anomaly_score=o["events"][0]["anomaly_score"] + 0.1))
        self.assert_invalid(lambda o: o["events"][1].update(anomaly_score=1.0))
        self.assert_invalid(lambda o: o["events"][1].update(not_scoreable_reasons=[]))
        self.assert_invalid(lambda o: o["events"][0]["feature_values"].update(spectral_centroid_mean=float("nan")))
        self.assert_invalid(lambda o: o.update(n_scored=2))
        self.assert_invalid(lambda o: o.update(recording_status="NO_INHALATION_DETECTED"))
        self.assert_invalid(lambda o: o.update(baseline_domain_validated=True))

    def test_schema_mirrors_validator_constants(self):
        schema = output_json_schema()
        self.assertEqual(schema["required"], list(OUTPUT_KEYS))
        self.assertEqual(schema["properties"]["recording_status"]["enum"], list(RECORDING_STATUSES))
        self.assertEqual(schema["properties"]["contract_version"]["const"], CONTRACT_VERSION)
        event = schema["properties"]["events"]["items"]
        self.assertEqual(event["required"], list(EVENT_KEYS))
        self.assertEqual(event["properties"]["not_scoreable_reasons"]["items"]["enum"], list(NOT_SCOREABLE_REASONS))
        self.assertEqual(event["properties"]["status"]["enum"], ["SCORE_ONLY", "NOT_SCOREABLE"])
        text = json.dumps(schema)
        for forbidden in ('"NORMAL"', '"ANOMALY"', "threshold\":"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
