"""Stage 1 dataset-finalization tests; they use synthetic tables, not data/."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inhale_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    UsabilityRule,
    add_recording_context,
    apply_usability_rule,
    audit_events_against_annotations,
    clean_annotations,
    compare_event_tables,
    covered_duration,
    interval_iou,
    match_inhale_annotations,
    parse_recording_timestamp,
)

REC_A = "rec2018-01-22_17h41m33.475s.wav"
REC_B = "rec2018-01-23_09h00m00.000s.wav"
REC_C = "rec2018-01-21_08h00m00.000s.wav"  # earliest timestamp, deliberately listed last
DURATIONS = {REC_A: 12.0, REC_B: 12.0, REC_C: 12.0}


def events_table(rows):
    """Build an upstream-shaped event table; each row gives file, id, start, end."""
    records = []
    for recording_id, (recording, event_id, start, end) in enumerate(rows):
        record = {
            "recording_id": recording_id,
            "event_id": event_id,
            "recording_file": recording,
            "label": "Inhale",
            "start_s": start,
            "end_s": end,
            "confidence": 0.9,
            "max_confidence": 0.95,
            "window_count": int(round((end - start - 0.2) / 0.016)) + 1,
        }
        record.update({column: 0.1 for column in FEATURE_COLUMNS})
        record["duration_s"] = end - start
        records.append(record)
    return pd.DataFrame(records)


def finalize(events, rule=UsabilityRule()):
    return apply_usability_rule(add_recording_context(events, DURATIONS), rule)


class HelperTests(unittest.TestCase):
    def test_timestamp_is_parsed_from_filename(self):
        self.assertEqual(parse_recording_timestamp(REC_A), pd.Timestamp("2018-01-22 17:41:33.475"))
        with self.assertRaises(ValueError):
            parse_recording_timestamp("recording.wav")

    def test_interval_iou(self):
        self.assertAlmostEqual(interval_iou(0, 2, 0, 2), 1.0)
        self.assertEqual(interval_iou(0, 1, 2, 3), 0.0)
        self.assertAlmostEqual(interval_iou(0, 2, 1, 3), 1 / 3)

    def test_covered_duration_merges_overlaps_and_clips(self):
        self.assertAlmostEqual(covered_duration(0, 10, [(1, 3), (2, 4), (8, 12)]), 5.0)
        self.assertEqual(covered_duration(0, 1, []), 0.0)

    def test_rule_rejects_negative_parameters(self):
        with self.assertRaises(ValueError):
            UsabilityRule(min_duration_s=-0.1)


class UsabilityRuleTests(unittest.TestCase):
    def test_short_event_is_excluded_and_threshold_is_inclusive(self):
        table = finalize(events_table([(REC_A, 0, 2.0, 2.4), (REC_B, 0, 2.0, 2.5)]))
        self.assertEqual(table["usable"].tolist(), [False, True])
        self.assertEqual(table["exclusion_reasons"].tolist(), ["short_duration", ""])

    def test_close_neighbours_are_both_excluded(self):
        table = finalize(events_table([(REC_A, 0, 2.0, 3.0), (REC_A, 1, 3.008, 4.0)]))
        self.assertTrue(table["flag_close_neighbor"].all())
        self.assertFalse(table["usable"].any())

    def test_gap_equal_to_threshold_is_not_close(self):
        # 3.6 - 3.4 is 0.19999999999999973 in floating point.
        table = finalize(events_table([(REC_A, 0, 2.8, 3.4), (REC_A, 1, 3.6, 4.4)]))
        self.assertAlmostEqual(table["gap_prev_s"].iloc[1], 0.2)
        self.assertFalse(table["flag_close_neighbor"].any())

    def test_distant_events_in_one_recording_stay_usable(self):
        table = finalize(events_table([(REC_A, 0, 2.0, 3.0), (REC_A, 1, 6.0, 7.0)]))
        self.assertTrue(table["usable"].all())
        self.assertEqual(table["n_events_in_recording"].tolist(), [2, 2])

    def test_recording_boundary_events_are_excluded(self):
        table = finalize(events_table([(REC_A, 0, 0.0, 1.0), (REC_B, 0, 11.0, 12.0)]))
        self.assertTrue(table["flag_recording_boundary"].all())
        self.assertTrue(np.isnan(table["inhale_window_fraction"].iloc[1]))

    def test_nonfinite_feature_is_excluded(self):
        events = events_table([(REC_A, 0, 2.0, 3.0)])
        events.loc[0, "zcr_std"] = np.nan
        table = finalize(events)
        self.assertEqual(table["exclusion_reasons"].iloc[0], "nonfinite_feature")

    def test_usable_order_is_chronological_not_input_order(self):
        events = events_table([(REC_B, 0, 2.0, 3.0), (REC_A, 0, 5.0, 6.0), (REC_A, 1, 1.0, 1.1), (REC_C, 0, 2.0, 3.0)])
        table = finalize(events)
        self.assertEqual(table["usable_order"].tolist(), [3, 2, pd.NA, 1])
        self.assertEqual(table["chronological_order"].tolist(), [4, 3, 2, 1])

    def test_upstream_columns_and_rows_are_unchanged(self):
        events = events_table([(REC_A, 0, 2.0, 2.3), (REC_B, 0, 2.0, 3.5)])
        table = finalize(events)
        pd.testing.assert_frame_equal(table[events.columns], events)

    def test_rule_is_deterministic(self):
        events = events_table([(REC_A, 0, 2.0, 3.0), (REC_A, 1, 3.1, 3.3), (REC_B, 0, 2.0, 3.5)])
        pd.testing.assert_frame_equal(finalize(events), finalize(events))

    def test_bridged_windows_reduce_inhale_window_fraction(self):
        events = events_table([(REC_A, 0, 2.0, 3.0)])
        events.loc[0, "window_count"] = 40  # 51 windows are spanned by a 1.0 s event
        self.assertAlmostEqual(finalize(events)["inhale_window_fraction"].iloc[0], 40 / 51)

    def test_empty_table_is_supported(self):
        table = finalize(events_table([(REC_A, 0, 2.0, 3.0)]).iloc[0:0])
        self.assertTrue(table.empty)
        self.assertIn("usable", table.columns)

    def test_missing_recording_duration_raises(self):
        events = events_table([("rec2018-02-01_10h00m00.000s.wav", 0, 2.0, 3.0)])
        with self.assertRaises(ValueError):
            add_recording_context(events, DURATIONS)


class AnnotationAuditTests(unittest.TestCase):
    def setUp(self):
        self.annotations = pd.DataFrame(
            [
                (REC_A, "Inhale", 16000, 24000),  # 2.0-3.0 s
                (REC_A, "Inhale", 16000, 24000),  # exact duplicate
                (REC_A, "Noise", 9860, 9860),     # zero length
                (REC_A, "Drug", 24000, 26000),
                (REC_B, "Exhale", 8000, 16000),
            ],
            columns=["filename", "label", "start_sample", "end_sample"],
        )

    def test_clean_annotations_reports_and_drops_invalid_rows(self):
        clean, summary = clean_annotations(self.annotations)
        self.assertEqual(len(clean), 3)
        self.assertEqual(summary["exact_duplicate_rows"], 1)
        self.assertEqual(summary["non_positive_length_rows"], 1)
        self.assertEqual(summary["labels_raw"]["Inhale"]["rows"], 2)
        self.assertEqual(summary["labels_clean"]["Inhale"]["rows"], 1)

    def test_overlapping_inhale_annotations_are_reported(self):
        overlapping = pd.DataFrame(
            [(REC_A, "Inhale", 0, 5000), (REC_A, "Inhale", 4000, 9000)],
            columns=["filename", "label", "start_sample", "end_sample"],
        )
        _, summary = clean_annotations(overlapping)
        self.assertEqual(summary["recordings_with_overlapping_inhale_annotations"], [REC_A])

    def test_event_status_categories(self):
        clean, _ = clean_annotations(self.annotations)
        events = events_table([
            (REC_A, 0, 2.05, 3.05),  # matches the Inhale annotation
            (REC_A, 1, 6.0, 7.0),    # annotated recording, outside its Inhale
            (REC_B, 0, 1.0, 2.0),    # annotated recording with no Inhale label
            (REC_C, 0, 2.0, 3.0),    # recording without annotations
        ])
        audit = audit_events_against_annotations(events, clean)
        self.assertEqual(
            audit["annotation_status"].tolist(),
            ["matched_inhale", "outside_inhale_annotations", "no_inhale_annotation_in_recording", "unannotated_recording"],
        )
        self.assertAlmostEqual(audit["overlap_exhale"].iloc[2], 1.0)
        self.assertTrue(np.isnan(audit["best_inhale_iou"].iloc[3]))

    def test_inhale_annotation_matching(self):
        clean, _ = clean_annotations(self.annotations)
        matched = match_inhale_annotations(events_table([(REC_A, 0, 2.05, 3.05)]), clean)
        self.assertTrue(matched["matched"].iloc[0])
        unmatched = match_inhale_annotations(events_table([(REC_B, 0, 2.0, 3.0)]), clean)
        self.assertFalse(unmatched["matched"].iloc[0])
        self.assertEqual(unmatched["best_iou"].iloc[0], 0.0)


class CompareTablesTests(unittest.TestCase):
    def test_identical_and_changed_tables(self):
        reference = events_table([(REC_A, 0, 2.0, 3.0)])
        self.assertTrue(compare_event_tables(reference, reference.copy())["identical"])
        changed = reference.copy()
        changed.loc[0, "mean_rms"] += 1e-6
        report = compare_event_tables(reference, changed)
        self.assertFalse(report["identical"])
        self.assertGreater(report["columns"]["mean_rms"]["max_abs_diff"], 0)
        self.assertFalse(compare_event_tables(reference, reference.drop(columns="zcr_std"))["identical"])


if __name__ == "__main__":
    unittest.main()
