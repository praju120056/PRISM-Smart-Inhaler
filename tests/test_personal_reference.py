"""Stage 10 / V3 personal-reference tests: mechanical properties on synthetic sittings (no data/ access needed,
except the optional real-golden check, which is skipped when the dataset is absent)."""

from datetime import datetime, timedelta
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # reuse the Stage 9 test fixtures

import config  # noqa: E402
from personal_reference import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MEDIAN_SE_FACTOR,
    PERSONAL_KEYS,
    SENSITIVITY_CONFIG,
    SENSITIVITY_CONFIG_PATH,
    STAGE9_DIR,
    PersonalReference,
    PersonalReferenceConfig,
    PersonalReferenceError,
    PersonalReferenceStore,
    assess_recording_with_personal,
    bounded_consensus_update,
    load_config,
    validate_personal_output,
    verify_audit,
)
from post_event import DEFAULT_MODEL_PATH  # noqa: E402
from prism_assessment import assess_recording  # noqa: E402
from prism_inference import V2_FEATURES  # noqa: E402
from test_prism_assessment import SR, StubDetector, noise_recording, synthetic_reference  # noqa: E402

START = datetime(2026, 1, 5, 8, 0, 0)
PATTERN = np.array([1.0, -1.0, 0.5, -0.5])


def make(cfg=None, user="user-1", device="device-1", reference_id="ref-1"):
    return PersonalReference(cfg or PersonalReferenceConfig(), user_id=user, device_id=device,
                             population_reference_id=reference_id, created_at=START.isoformat())


def sitting(center, n=11, spread=0.5):
    """n (odd) events whose per-feature median is exactly ``center``."""
    offsets = np.linspace(-spread, spread, n)
    return np.broadcast_to(np.asarray(center, dtype=np.float64), (4,)) + offsets[:, None] * PATTERN


def feed(personal, sittings, start=START, extra_days=None, close=True):
    """One recording per sitting, one sitting per day (plus ``extra_days[k]`` before sitting k)."""
    outputs, t = [], start
    for k, z in enumerate(sittings):
        if k:
            t += timedelta(days=1 + (extra_days or {}).get(k, 0))
        outputs.append(personal.observe_recording(list(z), [True] * len(z), t.isoformat()))
        validate_personal_output(outputs[-1])
    if close:
        personal.close_sitting()
    return outputs


def closed(personal):
    return [e for e in personal.audit if e["kind"] == "sitting_closed"]


def rms(v):
    return float(np.sqrt(np.mean(np.square(np.asarray(v, dtype=np.float64)))))


def file_hashes(paths):
    files = sorted(p for root in paths for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_provisional_design(self):
        cfg = PersonalReferenceConfig()
        self.assertEqual(cfg.features, V2_FEATURES)
        self.assertEqual((cfg.consensus_window, cfg.consensus_required), (10, 9))
        self.assertEqual((cfg.eta, cfg.clip_c, cfg.trust_radius), (0.3, 2.0, 0.9))
        self.assertEqual((cfg.min_events_per_sitting, cfg.s_min, cfg.sitting_gap_minutes), (10, 10, 25.0))
        self.assertEqual((SENSITIVITY_CONFIG.consensus_window, SENSITIVITY_CONFIG.consensus_required), (10, 8))
        self.assertAlmostEqual(cfg.sitting_bound(10), 0.3 * 2.0 * MEDIAN_SE_FACTOR / math.sqrt(10))

    def test_invalid_configs_are_rejected(self):
        for bad in ({"consensus_required": 5}, {"consensus_required": 11}, {"consensus_window": 1, "consensus_required": 1},
                    {"eta": 0.0}, {"eta": 1.5}, {"clip_c": 0.0}, {"trust_radius": -1.0}, {"max_gap_days": 0.0},
                    {"event_sd": (1.0, 1.0, 1.0)}, {"event_sd": (1.0, 1.0, 1.0, 0.0)}, {"s_min": 0},
                    {"min_events_per_sitting": True}, {"features": V2_FEATURES[::-1]}, {"eta": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(PersonalReferenceError):
                PersonalReferenceConfig(**bad)

    def test_round_trip_and_files_match_the_code(self):
        cfg = PersonalReferenceConfig()
        self.assertEqual(PersonalReferenceConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)
        with self.assertRaises(PersonalReferenceError):
            PersonalReferenceConfig.from_dict({**cfg.to_dict(), "extra": 1})
        for path, expected in ((DEFAULT_CONFIG_PATH, cfg), (SENSITIVITY_CONFIG_PATH, SENSITIVITY_CONFIG)):
            if path.exists():
                self.assertEqual(load_config(path), expected)
                self.assertEqual(load_config(path).sha256, expected.sha256)


class InitializationTests(unittest.TestCase):                  # required test 1
    def test_initial_state_is_the_population_reference(self):
        personal = make()
        np.testing.assert_array_equal(personal.baseline, np.zeros(4))
        self.assertEqual((personal.status, personal.baseline_version, personal.n_qualifying), ("WARMUP", 0, 0))
        self.assertEqual(personal.trust_radius, 0.9)
        self.assertEqual([e["kind"] for e in personal.audit], ["initialized"])
        self.assertEqual(personal.state_id, make().state_id)                 # deterministic
        self.assertNotEqual(personal.state_id, make(device="device-2").state_id)
        out = personal.observe_recording([[3.0, 0.0, 0.0, 0.0], None], [True, False], START.isoformat())
        validate_personal_output(out)
        self.assertEqual(tuple(out), PERSONAL_KEYS)
        self.assertEqual([e["personal_assessment"] for e in out["events"]], ["POPULATION_ONLY", "NOT_ASSESSED"])
        self.assertIsNone(out["events"][0]["personal_deviation"])
        self.assertEqual(out["baseline_used"], dict.fromkeys(V2_FEATURES, 0.0))


class WarmupTests(unittest.TestCase):
    def test_personal_channel_reports_only_after_s_min_qualifying_sittings(self):
        personal = make()
        small = sitting(0.0, n=9)                                   # 9 events < 10: never qualifies
        outputs = feed(personal, [sitting(0.0)] * 9 + [small] * 3 + [sitting(0.0)] * 2)
        statuses = [o["personal_status"] for o in outputs]
        self.assertEqual(statuses, ["WARMUP"] * 13 + ["ESTABLISHED"])
        self.assertEqual([e["decision"] for e in closed(personal)][9:12], ["NOT_QUALIFYING"] * 3)
        self.assertEqual(outputs[-1]["events"][0]["personal_assessment"], "PERSONAL_DEVIATION")
        self.assertTrue(all(e["personal_deviation"] is None for o in outputs[:13] for e in o["events"]))

    def test_unscoreable_events_are_neither_assessed_nor_buffered(self):
        personal = make()
        out = personal.observe_recording([None] * 20, [False] * 20, START.isoformat())
        self.assertEqual({e["personal_assessment"] for e in out["events"]}, {"NOT_ASSESSED"})
        entry = personal.close_sitting()
        self.assertEqual((entry["decision"], entry["n_qualifying_events"], entry["n_events_total"]),
                         ("NOT_QUALIFYING", 0, 20))
        with self.assertRaises(PersonalReferenceError):
            personal.observe_recording([[np.nan, 0, 0, 0]], [True], (START + timedelta(days=1)).isoformat())


class GateTests(unittest.TestCase):
    cfg = PersonalReferenceConfig()

    def update(self, window, cfg=None, baseline=(0, 0, 0, 0)):
        window = np.asarray(window, dtype=np.float64)
        return bounded_consensus_update(baseline, window, window[-1], 11, cfg or self.cfg, 0.9)

    def test_nine_of_ten_in_one_direction_opens_the_gate(self):      # required test 6
        nine = [[0.4] * 4] * 9
        result = self.update([[-0.2] * 4] + nine)                     # 9 of 10 positive, current positive
        self.assertTrue(result["gate"].all())
        np.testing.assert_array_equal(result["direction"], [1, 1, 1, 1])
        eight = [[-0.2] * 4] * 2 + [[0.4] * 4] * 8
        self.assertFalse(self.update(eight)["gate"].any())             # 8 of 10 is not enough
        dissenter = [[0.4] * 4] * 9 + [[-0.2] * 4]
        self.assertFalse(self.update(dissenter)["gate"].any())         # the current sitting disagrees
        negative = self.update([[-0.4] * 4] * 10)
        self.assertTrue(negative["gate"].all())
        np.testing.assert_array_equal(negative["direction"], [-1, -1, -1, -1])
        np.testing.assert_array_less(negative["step"], 0)

    def test_gate_is_per_feature_and_directional_not_closeness(self):
        window = np.full((10, 4), 0.4)
        window[:, 1] = [0.4, -0.4] * 5                                  # feature 1 has no direction
        window[:, 3] = 5.0                                              # feature 3 far from B, consistently
        result = self.update(window)
        np.testing.assert_array_equal(result["gate"], [True, False, True, True])
        self.assertTrue(result["clipped"][3])                          # far, so clipped - but the gate is open

    def test_eight_of_ten_sensitivity_configuration(self):             # required test 7
        eight = [[-0.2] * 4] * 2 + [[0.4] * 4] * 8
        self.assertTrue(self.update(eight, SENSITIVITY_CONFIG)["gate"].all())
        self.assertFalse(self.update(eight)["gate"].any())
        seven = [[-0.2] * 4] * 3 + [[0.4] * 4] * 7
        self.assertFalse(self.update(seven, SENSITIVITY_CONFIG)["gate"].any())

    def test_one_sitting_cannot_open_the_gate(self):                   # required test 8
        for cfg in (self.cfg, SENSITIVITY_CONFIG, PersonalReferenceConfig(consensus_window=2, consensus_required=2)):
            with self.subTest(cfg=cfg.config_id):
                self.assertFalse(self.update([[50.0] * 4], cfg)["gate"].any())
                personal = make(cfg)
                feed(personal, [sitting(1e6)])
                self.assertEqual(closed(personal)[0]["decision"], "NO_UPDATE")
                np.testing.assert_array_equal(personal.baseline, np.zeros(4))
        self.assertFalse(self.update([[0.4] * 4] * 9)["gate"].any())     # 9 agreeing but window not full

    def test_zero_innovation_and_zero_offsets_are_safe(self):
        result = self.update([[0.0] * 4] * 10)
        self.assertFalse(result["gate"].any())
        np.testing.assert_array_equal(result["step"], np.zeros(4))
        self.assertTrue(np.isfinite(result["weight"]).all())


class InfluenceTests(unittest.TestCase):
    cfg = PersonalReferenceConfig()

    def test_one_event_outlier_has_bounded_magnitude_free_influence(self):   # required test 2
        wide = PersonalReferenceConfig(clip_c=1e6)            # no clipping: isolates the median's protection
        prior = [sitting(0.3)] * 9
        results = {}
        for magnitude in (None, 1e3, 1e9, 1e300):
            current = sitting(0.3)
            if magnitude is not None:
                current[0] = magnitude
            personal = make(wide)
            feed(personal, prior + [current])
            results[magnitude] = personal.baseline.copy()
        gap = np.diff(np.sort(sitting(0.3), axis=0), axis=0).max()   # largest order-statistic spacing
        for magnitude in (1e3, 1e9, 1e300):
            self.assertLessEqual(np.abs(results[magnitude] - results[None]).max(), wide.eta * gap + 1e-15)
            np.testing.assert_array_equal(results[magnitude], results[1e3])   # independent of magnitude

    def test_one_sitting_outlier_with_closed_gate_has_zero_influence(self):  # required test 3
        personal = make()
        feed(personal, [sitting(0.3 if k % 2 else -0.3) for k in range(10)] + [sitting(1e6)])
        self.assertEqual(closed(personal)[-1]["decision"], "NO_UPDATE")
        np.testing.assert_array_equal(personal.baseline, np.zeros(4))

    def test_one_sitting_outlier_with_open_gate_is_capped(self):            # required test 3
        personal = make()
        feed(personal, [sitting(1.0)] * 9)
        before = personal.baseline.copy()
        feed(personal, [sitting(1e6)], start=START + timedelta(days=9))
        moved = personal.baseline - before
        se = MEDIAN_SE_FACTOR / math.sqrt(11)
        np.testing.assert_allclose(moved, np.full(4, 0.3 * 2.0 * se), rtol=1e-12)
        self.assertAlmostEqual(rms(moved), self.cfg.sitting_bound(11), places=12)

    def test_extreme_values_cannot_arbitrarily_move_the_baseline(self):     # required test 17
        bound = self.cfg.sitting_bound(self.cfg.min_events_per_sitting)
        finals = []
        for magnitude in (1e3, 1e9, 1e300):
            personal = make()
            outputs = feed(personal, [sitting(1.0)] * 9 + [np.full((10, 4), magnitude)])
            finals.append(personal.baseline.copy())
            for entry in closed(personal):
                self.assertLessEqual(rms(np.subtract(entry["baseline_after"], entry["baseline_before"])), bound + 1e-15)
            self.assertTrue(all(math.isfinite(o["trust_region_distance"]) for o in outputs))
        np.testing.assert_array_equal(finals[0], finals[1])
        np.testing.assert_array_equal(finals[1], finals[2])
        personal = make(PersonalReferenceConfig(s_min=1))
        feed(personal, [sitting(0.0)])
        out = personal.observe_recording([[1e300] * 4], [True], (START + timedelta(days=3)).isoformat())
        self.assertTrue(math.isfinite(out["events"][0]["personal_deviation"]))
        validate_personal_output(out)


class ConvergenceTests(unittest.TestCase):
    def test_sustained_shift_converges_monotonically(self):                # required test 4
        personal = make()
        target = np.full(4, 0.6)                                       # rms 0.6 < rho
        feed(personal, [sitting(target)] * 60)
        errors = [rms(np.subtract(e["baseline_after"], target)) for e in closed(personal)]
        self.assertLess(errors[-1], 1e-3)
        self.assertTrue(all(b <= a + 1e-15 for a, b in zip(errors, errors[1:])))
        first = next(i for i, e in enumerate(closed(personal)) if e["decision"] == "UPDATED")
        self.assertEqual(first, 9)                                     # the 10th qualifying sitting

    def test_sustained_large_shift_eventually_moves_the_baseline(self):    # required test 18
        far = PersonalReferenceConfig(trust_radius=10.0)
        personal = make(far)
        feed(personal, [sitting(3.0)] * 80)
        np.testing.assert_allclose(personal.baseline, np.full(4, 3.0), atol=1e-3)
        steps = [rms(np.subtract(e["baseline_after"], e["baseline_before"])) for e in closed(personal)]
        self.assertLessEqual(max(steps), far.sitting_bound(11) + 1e-15)   # many bounded steps, never one jump
        anchored = make()
        feed(anchored, [sitting(3.0)] * 40)
        self.assertAlmostEqual(anchored.trust_region_distance, 0.9, places=12)

    def test_no_change_jitter_stays_small(self):                          # required test 5
        rng = np.random.default_rng(20261007)
        personal = make()
        feed(personal, [rng.normal(size=(12, 4)) for _ in range(300)])
        distances = [rms(e["baseline_after"]) for e in closed(personal)]
        updates = sum(e["decision"] == "UPDATED" for e in closed(personal))
        # Independent-null gate rate ~ 4 features x 2 x 0.5 x P(>= 8 of 9) = 0.078 per sitting.
        self.assertLess(max(distances), 0.25)
        self.assertLess(updates / 300, 0.15)
        self.assertEqual(personal.n_capped_updates, 0)


class TrustRegionTests(unittest.TestCase):                            # required test 9
    def test_baseline_never_leaves_the_trust_region(self):
        personal = make()
        outputs = feed(personal, [sitting(5.0)] * 60)
        for entry in closed(personal):
            self.assertLessEqual(rms(entry["baseline_after"]), 0.9 * (1 + 1e-12))
        self.assertGreater(personal.n_capped_updates, 0)
        self.assertTrue(any(e["trust_region_capped"] for e in closed(personal) if "trust_region_capped" in e))
        self.assertTrue(personal.at_trust_region_boundary and outputs[-1]["at_trust_region_boundary"])
        np.testing.assert_allclose(personal.baseline, np.full(4, 0.9), rtol=1e-12)

    def test_reenrollment_is_the_only_way_beyond_the_trust_region(self):
        personal = make()
        feed(personal, [sitting(5.0)] * 30)
        with self.assertRaises(PersonalReferenceError):
            personal.reenroll(sitting(2.0), "supervised re-enrollment", (START + timedelta(days=40)).isoformat())
        entry = personal.reenroll(sitting(2.0), "supervised re-enrollment", (START + timedelta(days=40)).isoformat(),
                                  trust_radius=2.5)
        np.testing.assert_array_equal(personal.baseline, np.full(4, 2.0))
        self.assertEqual((personal.trust_radius, personal.status, entry["kind"]), (2.5, "WARMUP", "reenrollment"))
        reset = personal.reset("device replaced", (START + timedelta(days=41)).isoformat())
        np.testing.assert_array_equal(personal.baseline, np.zeros(4))
        self.assertEqual((personal.trust_radius, personal.status, reset["kind"]), (0.9, "WARMUP", "reset"))
        PersonalReference.replay(personal.audit)

    def test_open_sitting_blocks_reset(self):
        personal = make()
        personal.observe_recording([[0, 0, 0, 0]], [True], START.isoformat())
        with self.assertRaises(PersonalReferenceError):
            personal.reset("x", START.isoformat())


class LongGapTests(unittest.TestCase):                                # required test 10
    def test_long_gap_clears_stale_consensus(self):
        control = make()
        feed(control, [sitting(1.0)] * 10)
        self.assertEqual(closed(control)[-1]["decision"], "UPDATED")
        personal = make()
        outputs = feed(personal, [sitting(1.0)] * 19, extra_days={9: 20})
        entries = closed(personal)
        self.assertTrue(entries[9]["gap_reset"])
        self.assertEqual(entries[9]["consensus_window_size"], 1)
        self.assertTrue(outputs[9]["reference_stale"])
        self.assertFalse(outputs[8]["reference_stale"])
        self.assertEqual([e["decision"] for e in entries[9:]], ["NO_UPDATE"] * 9 + ["UPDATED"])
        np.testing.assert_array_equal(personal.baseline != 0, [True] * 4)


class OrderingTests(unittest.TestCase):
    cfg = PersonalReferenceConfig(s_min=3)

    @staticmethod
    def multi_recording_stream(shift_from=4, n_sittings=16, mutate=None):
        """Sittings of 3 recordings x 4 events, 5 min apart; optional mutation (sitting, recording) -> z."""
        rng = np.random.default_rng(5)
        stream = []
        for k in range(n_sittings):
            center = 1.0 if k >= shift_from else 0.0
            for r in range(3):
                z = center + 0.3 * rng.normal(size=(4, 4))
                if mutate and (k, r) in mutate:
                    z = mutate[(k, r)](z)
                stream.append((k, (START + timedelta(days=k, minutes=5 * r)).isoformat(), z))
        return stream

    def run_stream(self, stream):
        personal = make(self.cfg)
        outputs = []
        for k, at, z in stream:
            outputs.append((k, personal.observe_recording(list(z), [True] * len(z), at)))
        personal.close_sitting()
        return personal, outputs

    def test_assess_before_adapt(self):                                 # required test 11
        stream = self.multi_recording_stream()
        personal, outputs = self.run_stream(stream)
        entries = closed(personal)
        self.assertGreater(sum(e["decision"] == "UPDATED" for e in entries), 0)
        z_by_output = [z for _, _, z in stream]
        for (k, out), z in zip(outputs, z_by_output):
            used = np.array([out["baseline_used"][f] for f in V2_FEATURES])
            expected_b = np.zeros(4) if k == 0 else np.array(entries[k - 1]["baseline_after"])
            np.testing.assert_array_equal(used, expected_b)            # the baseline from BEFORE sitting k
            expected_id = personal.audit[0]["state_id_after"] if k == 0 else entries[k - 1]["state_id_after"]
            self.assertEqual(out["state_id_used"], expected_id)
            np.testing.assert_array_equal(entries[k]["baseline_before"], used)   # sitting k adapts only afterwards
            if out["personal_status"] == "ESTABLISHED":
                for event, row in zip(out["events"], z):
                    self.assertEqual(event["personal_deviation"], rms(row - used))

    def test_current_sitting_cannot_change_its_own_assessments(self):   # required test 12 (mutation)
        target = 12
        clean_personal, clean = self.run_stream(self.multi_recording_stream())
        mutation = {(target, 1): lambda z: z + 3.0, (target, 2): lambda z: np.full_like(z, 1e9)}
        mutated_personal, mutated = self.run_stream(self.multi_recording_stream(mutate=mutation))
        for (k, a), (_, b) in zip(clean, mutated):
            if k < target:
                self.assertEqual(a, b)                                  # earlier sittings untouched
            elif k == target:
                for key in ("state_id_used", "baseline_used", "baseline_version_used", "personal_status"):
                    self.assertEqual(a[key], b[key])                    # same snapshot despite the mutation
        first_clean = next(o for k, o in clean if k == target)
        first_mutated = next(o for k, o in mutated if k == target)
        self.assertEqual(first_clean, first_mutated)                    # assessment emitted before the mutation
        entries_clean, entries_mutated = closed(clean_personal), closed(mutated_personal)
        self.assertNotEqual(entries_clean[target]["baseline_after"], entries_mutated[target]["baseline_after"])

    def test_emitted_outputs_are_not_aliased_to_the_state(self):
        personal = make(self.cfg)
        stream = self.multi_recording_stream()
        emitted, snapshots = [], []
        for _, at, z in stream:
            emitted.append(personal.observe_recording(list(z), [True] * len(z), at))
            snapshots.append(copy.deepcopy(emitted[-1]))
        personal.close_sitting()
        self.assertEqual(emitted, snapshots)
        some_closed = next(o["closed_sitting"] for o in emitted if o["closed_sitting"])
        some_closed["baseline_after"][0] = 123.0
        verify_audit(personal.audit)

    def test_one_recording_per_sitting_equals_one_recording_per_event(self):
        rng = np.random.default_rng(11)
        sittings = [rng.normal(size=(12, 4)) + (0.9 if k > 5 else 0.0) for k in range(25)]
        batched, single = make(self.cfg), make(self.cfg)
        batched_out = feed(batched, sittings)
        single_out = []
        for k, z in enumerate(sittings):
            for i, row in enumerate(z):
                at = (START + timedelta(days=k, seconds=30 * i)).isoformat()
                single_out.append(single.observe_recording([row], [True], at)["events"][0])
        single.close_sitting()
        np.testing.assert_array_equal(batched.baseline, single.baseline)
        self.assertEqual([e for o in batched_out for e in o["events"]],
                         [{**e, "event_id": i} for e, i in zip(single_out, [i for z in sittings for i in range(len(z))])])
        drop = lambda e: {k: v for k, v in e.items() if k not in ("n_recordings", "at", "ended_at", "sequence",
                                                                  "prev_entry_sha256", "entry_sha256", "state_id_before",
                                                                  "state_id_after", "days_since_previous_qualifying")}
        self.assertEqual([drop(e) for e in closed(batched)], [drop(e) for e in closed(single)])

    def test_chronological_order_and_sitting_boundaries(self):
        personal = make()
        personal.observe_recording([[0, 0, 0, 0]], [True], START.isoformat())
        same = personal.observe_recording([[0, 0, 0, 0]], [True], (START + timedelta(minutes=25)).isoformat())
        new = personal.observe_recording([[0, 0, 0, 0]], [True], (START + timedelta(minutes=50, seconds=1)).isoformat())
        self.assertEqual((same["sitting_index"], new["sitting_index"]), (0, 1))
        self.assertIsNone(same["closed_sitting"])
        self.assertEqual(new["closed_sitting"]["sitting_index"], 0)
        with self.assertRaises(PersonalReferenceError):
            personal.observe_recording([[0, 0, 0, 0]], [True], START.isoformat())


class ReproducibilityTests(unittest.TestCase):
    def stream(self):
        rng = np.random.default_rng(9)
        return [rng.normal(size=(12, 4)) + (0.8 if k > 12 else 0.0) for k in range(40)]

    def test_deterministic_replay_and_persistence(self):                 # required test 13
        a, b = make(), make()
        out_a, out_b = feed(a, self.stream()), feed(b, self.stream())
        self.assertEqual(out_a, out_b)
        text = lambda p: json.dumps(p.to_dict(), sort_keys=True, allow_nan=False)
        self.assertEqual(text(a), text(b))
        with tempfile.TemporaryDirectory() as folder:
            store = PersonalReferenceStore(folder)
            t = START
            for k, z in enumerate(self.stream()):                       # save + reload after every recording
                if k:
                    t += timedelta(days=1)
                personal = store.load_or_create("user-1", "device-1", PersonalReferenceConfig(), "ref-1",
                                                START.isoformat())
                personal.observe_recording(list(z), [True] * len(z), t.isoformat())
                store.save(personal)
            personal = store.load("user-1", "device-1", PersonalReferenceConfig())
            personal.close_sitting()
            self.assertEqual(text(personal), text(a))
        replayed = PersonalReference.replay(a.audit)
        self.assertEqual(replayed.state_id, a.state_id)
        np.testing.assert_array_equal(replayed.baseline, a.baseline)

    def test_audit_log_is_complete_and_tamper_evident(self):              # required test 14
        personal = make()
        feed(personal, self.stream())
        verify_audit(personal.audit)
        entries = closed(personal)
        self.assertEqual(len(entries), 40)
        for previous, entry in zip(entries, entries[1:]):
            self.assertEqual(entry["baseline_before"], previous["baseline_after"])
        for entry in entries:
            moved = entry["baseline_after"] != entry["baseline_before"]
            self.assertEqual(entry["decision"] == "UPDATED", moved)
            if entry["decision"] != "NOT_QUALIFYING":
                expected = np.add(entry["baseline_before"], entry["step"])
                np.testing.assert_array_equal(entry["proposed"], expected)
        self.assertEqual(personal.baseline_version, sum(e["decision"] == "UPDATED" for e in entries))
        tampered = copy.deepcopy(personal.audit)
        tampered[15]["sitting_median"][0] += 1e-9
        with self.assertRaises(PersonalReferenceError):
            verify_audit(tampered)
        rehashed = copy.deepcopy(personal.audit)                        # consistent hashes, wrong content
        rehashed[15]["baseline_after"][0] += 0.5
        from personal_reference import _sha256_json  # noqa: E402
        previous = rehashed[14]["entry_sha256"]
        for entry in rehashed[15:]:
            entry["prev_entry_sha256"] = previous
            entry["entry_sha256"] = _sha256_json({k: v for k, v in entry.items() if k != "entry_sha256"})
            previous = entry["entry_sha256"]
        verify_audit(rehashed)
        with self.assertRaises(PersonalReferenceError):
            PersonalReference.replay(rehashed)
        broken = personal.to_dict()
        broken["baseline"] = [0.0, 0.0, 0.0, 0.1]
        with self.assertRaises(PersonalReferenceError):
            PersonalReference.from_dict(json.loads(json.dumps(broken)))


class PopulationChannelTests(unittest.TestCase):
    def test_population_output_is_byte_identical_to_stage9(self):        # required test 15
        reference = synthetic_reference()
        audio, detector = noise_recording(), StubDetector([(100, 180), (300, 305)])
        personal = make(reference_id=reference.reference_id)
        at = START.isoformat()
        combined = assess_recording_with_personal(audio, SR, reference, personal, recorded_at=at, detector=detector,
                                                  input_domain="reference_dataset", recording_id="r.wav")
        alone = assess_recording(audio, SR, reference, detector=detector, input_domain="reference_dataset",
                                 recording_id="r.wav", recorded_at=at)
        self.assertEqual(tuple(combined), ("contract_version", "population", "personal"))
        self.assertEqual(json.dumps(combined["population"], allow_nan=False).encode(),
                         json.dumps(alone, allow_nan=False).encode())
        before = copy.deepcopy(alone)
        other = make(reference_id=reference.reference_id)
        other.observe_population_output(alone, at)
        self.assertEqual(alone, before)                                 # the personal channel only reads
        with self.assertRaises(PersonalReferenceError):
            make(reference_id="another-ref").observe_population_output(alone, at)

    @unittest.skipUnless((Path(config.DATA_DIR) / "rec2018-01-22_17h41m33.475s.wav").exists()
                         and Path(DEFAULT_MODEL_PATH).exists(), "dataset WAV or ONNX model not available")
    def test_real_recording_population_channel_equals_the_stage9_golden_vector(self):
        from post_event import OnnxEventClassifier
        from prism_assessment import load_reference
        from prism_inference import read_wav
        golden = STAGE9_DIR / "golden"
        case = json.loads((golden / "golden_manifest.json").read_text(encoding="utf-8"))["cases"][0]
        expected = json.loads((golden / case["expected"]).read_text(encoding="utf-8"))
        reference = load_reference()
        waveform, rate = read_wav(Path(config.DATA_DIR) / case["input"]["file"])
        personal = make(reference_id=reference.reference_id)
        combined = assess_recording_with_personal(waveform, rate, reference, personal, recorded_at=START.isoformat(),
                                                  detector=OnnxEventClassifier(), input_domain=case["input_domain"],
                                                  recording_id=case["input"]["file"])
        same_call = assess_recording(waveform, rate, reference, detector=OnnxEventClassifier(),
                                     input_domain=case["input_domain"], recording_id=case["input"]["file"],
                                     recorded_at=START.isoformat())
        self.assertEqual(json.dumps(combined["population"], allow_nan=False).encode(),
                         json.dumps(same_call, allow_nan=False).encode())
        alone = assess_recording(waveform, rate, reference, detector=OnnxEventClassifier(),
                                 input_domain=case["input_domain"], recording_id=case["input"]["file"])
        self.assertEqual(alone, expected)                                # Stage 9 golden vector, unchanged
        strip = lambda out: {k: v for k, v in out.items() if k != "input"}   # recorded_at lives in "input"
        self.assertEqual(json.dumps(strip(combined["population"])), json.dumps(strip(alone)))


class Stage9IsolationTests(unittest.TestCase):                        # required test 16
    def test_personal_reference_never_modifies_stage9_files(self):
        guarded = [STAGE9_DIR, Path(config.ROOT_DIR) / "src" / "prism_assessment.py",
                   Path(config.ROOT_DIR) / "src" / "assessment_validation.py"]
        before = file_hashes([p for p in guarded if p.exists()])
        reference = synthetic_reference()
        with tempfile.TemporaryDirectory() as folder:
            store = PersonalReferenceStore(folder)
            personal = store.load_or_create("u", "d", PersonalReferenceConfig(), reference.reference_id,
                                            START.isoformat())
            for k in range(3):
                assess_recording_with_personal(noise_recording(seed=k), SR, reference, personal,
                                               recorded_at=(START + timedelta(days=k)).isoformat(),
                                               detector=StubDetector([(100, 180)]))
                store.save(personal)
        self.assertEqual(file_hashes([p for p in guarded if p.exists()]), before)
        for root in (STAGE9_DIR, STAGE9_DIR / "golden"):
            with self.assertRaises(PersonalReferenceError):
                PersonalReferenceStore(root)

    def test_states_are_separate_per_user_and_device(self):
        with tempfile.TemporaryDirectory() as folder:
            store = PersonalReferenceStore(folder)
            phone, inhaler = make(device="phone"), make(device="inhaler")
            feed(phone, [sitting(1.0)] * 12)
            feed(inhaler, [sitting(0.0)] * 12)
            store.save(phone)
            store.save(inhaler)
            self.assertNotEqual(store.path("user-1", "phone"), store.path("user-1", "inhaler"))
            self.assertGreater(store.load("user-1", "phone").trust_region_distance, 0)
            self.assertEqual(store.load("user-1", "inhaler").trust_region_distance, 0)
            self.assertIsNone(store.load("user-2", "phone"))
            with self.assertRaises(PersonalReferenceError):
                store.load("user-1", "phone", SENSITIVITY_CONFIG)
            with self.assertRaises(PersonalReferenceError):
                store.load_or_create("user-1", "phone", PersonalReferenceConfig(), "another-ref", START.isoformat())


if __name__ == "__main__":
    unittest.main()
