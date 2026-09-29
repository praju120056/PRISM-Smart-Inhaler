"""Focused temporal grouping tests; they do not load the ONNX model."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from post_event import (  # noqa: E402
    TemporalGroupingConfig,
    WindowPrediction,
    group_inhale_events,
)


def prediction(index: int, label: str, start: float, end: float, confidence: float = 0.9):
    probabilities = {"Drug": 0.02, "Exhale": 0.03, "Inhale": 0.04, "Noise": 0.01}
    probabilities[label] = confidence
    return WindowPrediction(index, start, end, label, confidence, probabilities)


class TemporalGroupingTests(unittest.TestCase):
    def test_overlapping_inhale_windows_form_one_event(self):
        stream = [
            prediction(0, "Noise", 0.00, 0.20),
            prediction(1, "Inhale", 0.02, 0.22),
            prediction(2, "Inhale", 0.04, 0.24),
        ]
        events = group_inhale_events(stream)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].window_indices, (1, 2))
        self.assertAlmostEqual(events[0].start, 0.02)
        self.assertAlmostEqual(events[0].end, 0.24)

    def test_configured_short_gap_is_merged(self):
        stream = [
            prediction(0, "Inhale", 0.00, 0.10),
            prediction(1, "Noise", 0.10, 0.20),
            prediction(2, "Inhale", 0.20, 0.30),
        ]
        events = group_inhale_events(stream, TemporalGroupingConfig(max_gap_s=0.11))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].window_indices, (0, 2))

    def test_long_gap_stays_separate(self):
        stream = [
            prediction(0, "Inhale", 0.00, 0.10),
            prediction(1, "Noise", 0.10, 0.20),
            prediction(2, "Inhale", 0.50, 0.60),
        ]
        events = group_inhale_events(stream, TemporalGroupingConfig(max_gap_s=0.11))
        self.assertEqual(len(events), 2)

    def test_minimum_duration_filters_a_short_candidate(self):
        stream = [prediction(0, "Inhale", 0.00, 0.05)]
        events = group_inhale_events(
            stream, TemporalGroupingConfig(min_event_duration_s=0.10)
        )
        self.assertEqual(events, [])

    def test_optional_majority_smoothing_repairs_one_label_glitch(self):
        stream = [
            prediction(0, "Inhale", 0.00, 0.10),
            prediction(1, "Noise", 0.10, 0.20),
            prediction(2, "Inhale", 0.20, 0.30),
        ]
        events = group_inhale_events(
            stream, TemporalGroupingConfig(smoothing_window=3)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].window_indices, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
