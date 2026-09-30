"""Stage 8 V2 validation tests (synthetic data; no data/ access)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline_v1 import fit_baseline  # noqa: E402
from natural_population_analysis import heldout_scores  # noqa: E402
from prism_inference import V2_FEATURES  # noqa: E402
from v2_validation import (  # noqa: E402
    ABLATIONS,
    COMPARATORS,
    GATE_RULES,
    LEVEL_VARIANT,
    PREREGISTRATION,
    V2,
    build_contract,
    build_v2_representations,
    compare_runs,
    correlation_stability,
    direction_consistency,
    evaluate_gate,
    feature_scale_stability,
    gate_outcome,
    rank_r2,
)

V1 = ["duration_s", "mean_rms", "spectral_centroid_mean", "spectral_centroid_std",
      "spectral_flatness_mean", "spectral_flatness_std", "spectral_rolloff_std"]


def synthetic_table(seed=0, sessions=4, per_session=30):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        for i in range(per_session):
            rows.append({"recording_file": f"s{s}_{i}.wav", "event_id": 0, "session": f"S{s}", "usable": i % 10 != 9,
                         **{f: rng.normal(1.0 + 0.2 * s, 0.1 + 0.02 * k) for k, f in enumerate(V2)}})
    return pd.DataFrame(rows)


class RepresentationTests(unittest.TestCase):
    def test_v2_representations(self):
        reps = build_v2_representations(V1)
        self.assertEqual(reps["V2"], list(V2_FEATURES))
        self.assertEqual(set(COMPARATORS) | {"V2", *ABLATIONS, LEVEL_VARIANT}, set(reps))
        for name, removed in zip(ABLATIONS, V2):
            self.assertEqual(reps[name], [f for f in V2 if f != removed])
        self.assertEqual(reps[LEVEL_VARIANT], [*V2, "mean_rms"])
        self.assertNotIn("duration_s", reps["V2"])
        self.assertNotIn("mean_rms", reps["V2"])


class HeldOutIsolationTests(unittest.TestCase):
    def test_held_out_session_never_enters_its_fold(self):
        table = synthetic_table()
        sessions = table["session"]
        z, parameters, baselines = heldout_scores(table, V2, sessions)
        changed = table.copy()
        changed.loc[sessions == "S0", V2] += 100.0        # corrupt the held-out session only
        z2, parameters2, _ = heldout_scores(changed, V2, sessions)
        fold0 = parameters[parameters["fold"] == "S0"].reset_index(drop=True)
        pd.testing.assert_frame_equal(fold0, parameters2[parameters2["fold"] == "S0"].reset_index(drop=True))
        expected = fit_baseline(table[table["usable"] & (sessions != "S0")], V2)
        for feature in V2:
            self.assertEqual(baselines["S0"].parameters[feature].median, expected.parameters[feature].median)
        self.assertFalse(np.allclose(z.loc[sessions == "S0"], z2.loc[sessions == "S0"]))   # its own z do change


class StabilityTests(unittest.TestCase):
    def test_feature_scale_stability(self):
        table = synthetic_table(seed=1)
        _, parameters, _ = heldout_scores(table, V2, table["session"])
        pooled = fit_baseline(table[table["usable"]], V2)
        result = feature_scale_stability(parameters, pooled, V2).set_index("feature")
        self.assertEqual(list(result.index), V2)
        self.assertTrue((result["min_scale_ratio"] > 0).all() and result["all_mad_positive_finite"].all())
        self.assertTrue((result["max_center_shift_sd"] >= 0).all())

    def test_correlation_stability_detects_a_session_driving_the_correlation(self):
        rng = np.random.default_rng(2)
        rows = []
        for s in range(4):
            for i in range(40):
                a = rng.normal()
                b = (3 * a if s == 0 else 0) + rng.normal()     # only session 0 couples a and b
                rows.append({"session": f"S{s}", "usable": True, V2[0]: a, V2[1]: b, V2[2]: rng.normal(), V2[3]: rng.normal()})
        table = pd.DataFrame(rows)
        pairs, folds = correlation_stability(table, table["session"], V2)
        worst = pairs.set_index(["feature_a", "feature_b"]).loc[(V2[0], V2[1])]
        self.assertGreater(worst["max_abs_shift_vs_pooled"], 0.2)
        self.assertEqual(worst["fold_of_max_shift"], "S0")
        self.assertEqual(len(folds), 4 * 6)

    def test_rank_r2(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(300, 2))
        self.assertGreater(rank_r2(x[:, 0] + 0.01 * rng.normal(size=300), x), 0.95)
        self.assertLess(rank_r2(rng.normal(size=300), x), 0.05)


class PerturbationDirectionTests(unittest.TestCase):
    def test_direction_consistency(self):
        rows = []
        for row in range(20):
            session = "A" if row < 7 else "B" if row < 14 else "C"
            rows.append({"row": row, "session": session, "transform": "identity", "magnitude": 0.0, "q": 0.0})
            rows.append({"row": row, "session": session, "transform": "noise", "magnitude": 10.0, "q": 0.5})
            rows.append({"row": row, "session": session, "transform": "tilt", "magnitude": 0.9,
                         "q": -0.3 if session == "C" else 0.3})        # session C reverses direction
            rows.append({"row": row, "session": session, "transform": "gain", "magnitude": 2.0, "q": 1e-6 * (row % 3)})
        result = direction_consistency(pd.DataFrame(rows), {"q": lambda b: b["q"].to_numpy()}).set_index("transform")
        self.assertEqual(result.loc["noise", "frac_sessions_same_sign"], 1.0)
        self.assertTrue(result.loc["noise", "responds"])
        self.assertAlmostEqual(result.loc["tilt", "frac_sessions_same_sign"], 2 / 3)
        self.assertAlmostEqual(result.loc["tilt", "frac_events_same_sign"], 14 / 20)
        self.assertFalse(result.loc["gain", "responds"])
        self.assertLessEqual(result.loc["gain", "p95_abs_delta"], 2e-6)


def passing_evidence() -> dict:
    comparison = pd.DataFrame({"representation": ["V2", "R0_current7"], "session_epsilon_squared": [0.15, 0.21],
                               "generalization_ratio": [1.1, 1.1]})
    feature_scale = pd.DataFrame({"feature": V2, "max_center_shift_sd": 0.3, "fold_of_max_shift": "S1",
                                  "min_scale_ratio": 0.9, "max_scale_ratio": 1.1, "all_mad_positive_finite": True,
                                  "min_fold_mad": 0.01})
    correlation = pd.DataFrame({"feature_a": [V2[0]], "feature_b": [V2[1]], "max_abs_shift_vs_pooled": [0.1],
                                "fold_of_max_shift": ["S1"]})
    direction_rows = []
    for quantity in [*(f"z_{f}" for f in V2), "rms_z[V2]"]:
        for transform, magnitude in (("noise", 10.0), ("tilt", -0.5), ("tilt", 0.9), ("gain", 2.0), ("recording_gain", 2.0)):
            direction_rows.append({"quantity": quantity, "transform": transform, "magnitude": magnitude,
                                   "p95_abs_delta": 0.01, "responds": transform in ("noise", "tilt"),
                                   "frac_sessions_same_sign": 1.0})
    response = pd.DataFrame({"representation": "V2", "transform": ["noise", "tilt"], "family_distance_monotone_fraction": 0.99})
    feature_perturbation = pd.DataFrame({"feature": V2, "transform": "noise", "median_delta_z@30": 0.5, "monotone_fraction": 0.3})
    confounds = pd.DataFrame({"representation": ["V2"], "pooled_rho_background_rms": [0.1],
                              "within_session_rho_background_rms": [0.05]})
    reproduction = {"recordings": 3, "events_contract": 2, "recordings_with_count_mismatch": 0, "max_abs_diff_bounds_s": 0.0,
                    "max_abs_diff_confidence": 0.0, "usability_mismatches": 0, "reason_mismatches": 0,
                    "no_inhalation_detected": 1, "max_rel_diff_features": {"a": 0.0}, "passed": True}
    return {"comparison": comparison, "feature_scale": feature_scale, "correlation": correlation,
            "direction": pd.DataFrame(direction_rows), "response": response, "feature_perturbation": feature_perturbation,
            "confounds": confounds, "reproduction": reproduction, "roundtrip_max_abs_diff": 0.0,
            "leakage": {"excluded_events_in_any_fit": 0, "held_out_session_events_in_own_fit": 0},
            "rerun": {"mismatches": [], "passed": True, "files_compared": 5}}


class GateTests(unittest.TestCase):
    def test_rules_match_the_committed_preregistration(self):
        registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
        self.assertEqual({c["id"]: c["rule"] for c in registered["criteria"]}, GATE_RULES)
        self.assertEqual(registered["representation_under_test"]["features_in_order"], list(V2_FEATURES))

    def test_passing_evidence_passes(self):
        rows = evaluate_gate(passing_evidence())
        self.assertEqual([r["id"] for r in rows], list(GATE_RULES))
        self.assertEqual(gate_outcome(rows), "PASS")

    def test_each_failure_is_reported(self):
        def column(key, name, value):
            return lambda e: e.__setitem__(key, e[key].assign(**{name: value}))

        def item(key, name, value):
            return lambda e: e[key].__setitem__(name, value)

        cases = {
            "G1a": column("comparison", "session_epsilon_squared", [0.3, 0.21]),
            "G1b": column("comparison", "generalization_ratio", [1.2, 1.1]),
            "G2a": column("feature_scale", "max_center_shift_sd", 0.6),
            "G2b": column("feature_scale", "max_scale_ratio", 1.6),
            "G3a": column("correlation", "max_abs_shift_vs_pooled", [0.25]),
            "G4a": column("direction", "p95_abs_delta", 0.2),
            "G4b": column("response", "family_distance_monotone_fraction", 0.9),
            "G4c": column("direction", "frac_sessions_same_sign", 0.8),
            "G4d": column("feature_perturbation", "median_delta_z@30", 1.5),
            "G5a": column("confounds", "pooled_rho_background_rms", [-0.4]),
            "G5b": column("confounds", "within_session_rho_background_rms", [0.35]),
            "G6a": lambda e: e.__setitem__("rerun", {"mismatches": ["x.csv"], "passed": False, "files_compared": 5}),
            "G6b": item("reproduction", "passed", False),
            "G6c": lambda e: e.__setitem__("roundtrip_max_abs_diff", 1e-9),
            "G6d": item("leakage", "held_out_session_events_in_own_fit", 1),
        }
        for criterion, mutate in cases.items():
            evidence = passing_evidence()
            mutate(evidence)
            rows = {r["id"]: r for r in evaluate_gate(evidence)}
            self.assertFalse(rows[criterion]["passed"], criterion)
            self.assertEqual(sum(r["passed"] is False for r in rows.values()), 1, criterion)

    def test_missing_rerun_is_incomplete_not_pass(self):
        evidence = passing_evidence()
        evidence["rerun"] = None
        self.assertEqual(gate_outcome(evaluate_gate(evidence)), "INCOMPLETE")

    def test_contract_status_follows_gate_and_has_no_threshold(self):
        from prism_inference import FrozenBaseline

        baseline = FrozenBaseline("b", tuple(V2), (0.3, 0.1, 0.04, 0.08), (0.02, 0.02, 0.006, 0.01),
                                  tuple(1.4826 * m for m in (0.02, 0.02, 0.006, 0.01)), 1.4826, 318, 18, {})
        for outcome, status in (("PASS", "FROZEN_FOR_MVP_ENGINEERING"), ("FAIL", "DRAFT_NOT_FROZEN")):
            contract = build_contract(baseline, "0" * 64, "1" * 64, outcome, [{"id": "G1a", "passed": outcome == "PASS"}], [])
            self.assertEqual(contract["status"], status)
            self.assertEqual(contract["acceptance_gate"]["failing_criteria"], [] if outcome == "PASS" else ["G1a"])
            self.assertIn("NOT", contract["status_meaning"])
            self.assertIsNone(contract["scoring"]["threshold"])
            self.assertIsNone(contract["scoring"]["classification"])
            self.assertEqual(contract["feature_extraction"]["anomaly_features_in_order"][0]["name"], V2[0])
            self.assertFalse(contract["level_channel"]["in_anomaly_score"])
            self.assertIn("NO_INHALATION_DETECTED", contract["output"]["recording_statuses"])
            json.dumps(contract, allow_nan=False)


class RerunComparisonTests(unittest.TestCase):
    def test_compare_runs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for folder in (a, b):
                Path(folder, "t.csv").write_text("x\n1\n")
                Path(folder, "inference_contract_v2.json").write_text(json.dumps({"status": folder, "k": 1}))
            Path(a, "acceptance_gate_results.csv").write_text("G6a,None\n")    # records the G6a verdict itself
            Path(b, "acceptance_gate_results.csv").write_text("G6a,True\n")
            self.assertTrue(compare_runs(Path(a), Path(b))["passed"])       # gate-dependent fields are excluded
            Path(b, "t.csv").write_text("x\n2\n")
            self.assertEqual(compare_runs(Path(a), Path(b))["mismatches"], ["t.csv"])


if __name__ == "__main__":
    unittest.main()
