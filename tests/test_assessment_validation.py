"""Stage 9 analysis tests: pre-registration, leakage, statistics and gate logic (synthetic data)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assessment_validation import (  # noqa: E402
    GATE_RULES,
    K1_MAX_RATE,
    K2_MAX_RELATIVE_HALF_WIDTH,
    K3_GAIN_TOLERANCE,
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
    REPRESENTATIONS,
    StageError,
    cohens_kappa,
    compare_runs,
    k3_evaluation,
    loso_scores,
    outer_evaluation,
    overdispersion,
    representation_score,
    require_clean_output_dir,
    robust_correlation,
    run_analysis,
    session_bootstrap_rate,
)
from inhale_dataset import _sha256  # noqa: E402
from prism_inference import V2_FEATURES  # noqa: E402


def synthetic_table(sessions=6, per_session=15, seed=7):
    rng = np.random.default_rng(seed)
    n = sessions * per_session
    offsets = np.repeat(rng.normal(0, 0.5, size=(sessions, 4)), per_session, axis=0)
    values = (rng.normal(size=(n, 4)) + offsets) * [0.02, 0.03, 0.006, 0.01] + [0.37, 0.13, 0.04, 0.08]
    table = pd.DataFrame(values, columns=list(V2_FEATURES))
    table["usable"] = True
    table["recording_file"] = [f"r{i}.wav" for i in range(n)]
    table["event_id"] = 0
    return table, pd.Series([f"S{i // per_session}" for i in range(n)], index=table.index)


class PreregistrationTests(unittest.TestCase):
    def test_preregistration_is_unchanged(self):
        self.assertEqual(_sha256(PREREGISTRATION), PREREGISTRATION_SHA256)

    def test_rule_constants_restate_the_preregistration(self):
        gate = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))["category_gate"]
        self.assertIn("contains 0.05", gate["K1_marginal_validity"])
        self.assertIn(f"<= {K1_MAX_RATE:.2f}", gate["K1_marginal_validity"])
        self.assertIn(f"<= {K2_MAX_RELATIVE_HALF_WIDTH:.2f}", gate["K2_reference_stability"])
        self.assertIn(f"<= {K3_GAIN_TOLERANCE}", gate["K3_perturbation_behaviour"])
        self.assertEqual(set(GATE_RULES), {"K1", "K2", "K3", "K4"})
        reps = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))["representations"]
        self.assertEqual(set(reps), set(REPRESENTATIONS))


class ScoreTests(unittest.TestCase):
    def test_robust_correlation_properties(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3))
        x[:, 2] = x[:, 0] * 2 + 1                              # perfectly monotone pair -> r = 1
        r = robust_correlation(x)
        np.testing.assert_allclose(np.diag(r), 1.0)
        np.testing.assert_allclose(r, r.T)
        self.assertAlmostEqual(r[0, 2], 1.0)

    def test_correlation_aware_score_with_identity_equals_rms(self):
        z = np.random.default_rng(2).normal(size=(10, 4))
        np.testing.assert_allclose(representation_score(z, "rc", np.eye(4)), representation_score(z, "rms"))


class LeakageTests(unittest.TestCase):
    def test_loso_score_does_not_depend_on_its_own_session(self):
        table, sessions = synthetic_table()
        before = loso_scores(table, sessions, list(V2_FEATURES), "rms")
        changed = table.copy()
        same_session = changed.index[(sessions == "S0").to_numpy()]
        changed.loc[same_session[1:], list(V2_FEATURES)] *= 1.5   # other events of the same session
        after = loso_scores(changed, sessions, list(V2_FEATURES), "rms")
        self.assertEqual(before[same_session[0]], after[same_session[0]])

    def test_nested_calibration_excludes_the_held_out_session(self):
        table, sessions = synthetic_table()
        evaluation, models, cuts = outer_evaluation(table, sessions, list(V2_FEATURES), "rms")
        for session, block in evaluation.groupby("session"):
            self.assertTrue((block["n_calibration"] == int((sessions != session).sum())).all())
            self.assertEqual(models[session][0].n_calibration, int((sessions != session).sum()))
        self.assertEqual(len(evaluation), len(table))


class StatisticsTests(unittest.TestCase):
    def test_session_bootstrap_rate_is_deterministic(self):
        rng = np.random.default_rng(3)
        outside = rng.random(300) < 0.05
        sessions = np.array([f"s{i % 12}" for i in range(300)])
        a, b = session_bootstrap_rate(outside, sessions), session_bootstrap_rate(outside, sessions)
        self.assertEqual(a, b)
        self.assertLessEqual(a[1], a[0])
        self.assertLessEqual(a[0], a[2])

    def test_overdispersion_detects_heterogeneity(self):
        rng = np.random.default_rng(4)
        sessions = np.repeat([f"s{i}" for i in range(10)], 30)
        homogeneous = rng.random(300) < 0.1
        heterogeneous = np.concatenate([np.ones(30, bool), np.ones(30, bool), np.zeros(240, bool)])
        self.assertGreater(overdispersion(homogeneous, sessions, draws=500)["parametric_bootstrap_p"], 0.01)
        self.assertLess(overdispersion(heterogeneous, sessions, draws=500)["parametric_bootstrap_p"], 0.01)

    def test_cohens_kappa(self):
        a = np.array([True, False, False, True, False])
        self.assertEqual(cohens_kappa(a, a), 1.0)
        self.assertLess(cohens_kappa(a, ~a), 0)


class GateLogicTests(unittest.TestCase):
    @staticmethod
    def summary(identity=0.05, noise=(0.2, 0.4, 0.8), tilt=(0.1, 0.3), gain_delta=0.0):
        rows = [("identity", 0.0, identity), ("tilt", -0.5, 0.1)]
        rows += [("noise", m, v) for m, v in zip((30.0, 20.0, 10.0), noise)]
        rows += [("tilt", m, v) for m, v in zip((0.5, 0.9), tilt)]
        rows += [(t, m, identity + gain_delta) for t, m in (("gain", 0.5), ("gain", 1 / np.sqrt(2)), ("gain", np.sqrt(2)),
                                                            ("gain", 2.0), ("recording_gain", 0.5), ("recording_gain", 2.0))]
        return pd.DataFrame(rows, columns=["transform", "magnitude", "outside_fraction"])

    def test_k3_passes_monotone_gain_invariant_responses(self):
        self.assertTrue(k3_evaluation(self.summary())["passed"])

    def test_k3_fails_non_monotone_or_gain_dependent(self):
        self.assertFalse(k3_evaluation(self.summary(noise=(0.4, 0.2, 0.8)))["passed"])
        self.assertFalse(k3_evaluation(self.summary(tilt=(0.3, 0.1)))["passed"])
        self.assertFalse(k3_evaluation(self.summary(gain_delta=0.05))["passed"])


class ReproducibilityTests(unittest.TestCase):
    def test_compare_runs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for folder in (a, b):
                Path(folder, "x.csv").write_text("1,2\n")
                Path(folder, "e2e_timing.csv").write_text(folder)          # excluded: wall-clock times
            self.assertTrue(compare_runs(Path(a), Path(b))["passed"])
            Path(b, "x.csv").write_text("1,3\n")
            result = compare_runs(Path(a), Path(b))
            self.assertFalse(result["passed"])
            self.assertEqual(result["mismatches"], ["x.csv"])

    def test_stale_file_would_contaminate_k4(self):
        """The failure mode the output-directory guard prevents: K4 compares every file present."""
        with tempfile.TemporaryDirectory() as current, tempfile.TemporaryDirectory() as reference:
            for folder in (current, reference):
                Path(folder, "heldout_assessment.csv").write_text("1,2\n")
            Path(current, "stale_from_earlier_run.json").write_text("{}")
            self.assertEqual(compare_runs(Path(current), Path(reference))["mismatches"], ["stale_from_earlier_run.json"])
            with self.assertRaises(StageError):
                require_clean_output_dir(Path(current))


class OutputDirectoryTests(unittest.TestCase):
    def test_empty_or_preregistration_only_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            require_clean_output_dir(Path(folder))                       # empty
            Path(folder, PREREGISTRATION.name).write_text("{}")
            require_clean_output_dir(Path(folder))                       # only the pre-registration

    def test_any_leftover_file_or_folder_is_refused(self):
        for leftover in ("heldout_assessment.csv", "golden/", "notes.txt", ".hidden"):
            with self.subTest(leftover=leftover), tempfile.TemporaryDirectory() as folder:
                Path(folder, PREREGISTRATION.name).write_text("{}")
                if leftover.endswith("/"):
                    Path(folder, leftover).mkdir()
                else:
                    Path(folder, leftover).write_text("x")
                with self.assertRaises(StageError):
                    require_clean_output_dir(Path(folder))

    def test_error_is_deterministic_and_names_the_leftovers(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("b.csv", "a.json"):
                Path(folder, name).write_text("x")
            Path(folder, "golden").mkdir()
            messages = []
            for _ in range(2):
                with self.assertRaises(StageError) as caught:
                    require_clean_output_dir(Path(folder))
                messages.append(str(caught.exception))
            self.assertEqual(messages[0], messages[1])
            self.assertIn("is not empty: a.json, b.csv, golden/.", messages[0])
            self.assertIn("--output-dir", messages[0])

    def test_run_analysis_refuses_before_writing_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            stale = Path(folder, "heldout_assessment.csv")
            stale.write_text("stale\n")
            with self.assertRaises(StageError):
                run_analysis(data_dir=Path(folder, "no_data_needed"), output_dir=folder, make_plots=False)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["heldout_assessment.csv"])
            self.assertEqual(stale.read_text(), "stale\n")


if __name__ == "__main__":
    unittest.main()
