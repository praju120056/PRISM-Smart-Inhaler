"""Stage 9: final event-level assessment - representation check, LOSO calibration, gate and contract.

See PRISM_RESEARCH_LOG.md Entry 12.  Every decision rule restates
results/final_assessment/stage9_preregistration.json, written before any Stage 9 result; its
SHA-256 is fixed below and checked at run time.

Question: can the V2 deviation score be turned into a reference-calibrated event-level assessment
(empirical reference-tail probability and one-sided 95% upper reference limit) whose calibration
holds on recording sessions held out from the same PRISM corpus, which is stable, predictable under
controlled acoustic perturbations and reproducible - and is a categorical statement justified, or
must the output stay continuous?  This is an internal same-corpus check, not external validation.
No normal/abnormal, anomaly, technique-quality or clinical label is produced.

Protocol (leakage-free): for every held-out session h, the reference (V2 baseline + nested
leave-one-session-out calibration scores + cut + bootstrap band) is fitted on the usable events of
the other sessions only, frozen, and then applied to the events of h.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config
from baseline_v1 import DEFAULT_SELECTION_JSON, fit_baseline, load_feature_selection, recording_sessions
from feature_analysis import DEFAULT_DATASET_CSV, MIN_GROUP_SIZE, load_stage1_table, within_group_spearman
from inhale_dataset import _git_state, _json_default, _relative, _sha256
from natural_population_analysis import PERTURBATION_ORDER, assign_populations, cliffs_delta
from post_event import DEFAULT_MODEL_PATH, OnnxEventClassifier
from prism_assessment import (
    ALPHA,
    ASSESSMENT_CONTRACT_VERSION,
    BAND_PERCENTILES,
    BOOTSTRAP_DRAWS,
    CATEGORIES,
    EVENT_KEYS,
    INTERPRETATION,
    RELIABILITY,
    SEED,
    AssessmentReference,
    assess_recording,
    reference_cut,
    fit_reference,
    loso_calibration_scores,
    output_json_schema,
    rms_scores,
    tail_count_limit,
    tail_probability,
    write_reference,
)
from prism_inference import (
    FORBIDDEN_DERIVED_OUTPUTS,
    MIN_SAMPLES,
    SAMPLE_RATE,
    V2_FEATURES,
    read_wav,
)
from representation_analysis import (
    BASELINE_INK,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MCD_MAX_CONDITION_NUMBER,
    MCD_MAX_CORRELATION_SHIFT,
    MCD_MIN_EVENTS_PER_FEATURE,
    SERIES,
    SURFACE,
    _style,
    feature_perturbation_table,
    monotone_outward,
    session_dependence,
)
from v2_validation import (
    G1B_MAX_GENERALIZATION_RATIO,
    G2A_MAX_CENTER_SHIFT_SD,
    G2B_SCALE_RATIO_RANGE,
    G3A_MAX_CORRELATION_SHIFT,
    G4A_MAX_P95_GAIN_DELTA,
    G4B_MIN_MONOTONE_FRACTION,
    G4C_LEVELS,
    G4C_MIN_SESSION_AGREEMENT,
    G4D_FRAGILE_DELTA,
    G4D_FRAGILE_MONOTONE,
    G5_MAX_ABS_RHO,
    direction_consistency,
)


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "final_assessment"
PREREGISTRATION = DEFAULT_OUTPUT_DIR / "stage9_preregistration.json"
PREREGISTRATION_SHA256 = "5b5cdf6ee4498ddb69b21be6b54546d1bd1fc0ec3ddde72ad519dc90c61801e6"
V2_DIR = Path(config.RESULTS_DIR) / "v2_validation"
V2_PERTURBED = V2_DIR / "perturbed_features.csv"
V2_GOLDEN = V2_DIR / "golden"
STAGE7_CANDIDATES = Path(config.RESULTS_DIR) / "representation_analysis" / "candidate_features.csv"
REFERENCE_ID = "prism-assessment-ref-2026-10-06"

V2 = list(V2_FEATURES)
SHORT = {"spectral_centroid_mean": "centroid_mean", "spectral_flatness_mean": "flatness_mean",
         "spectral_centroid_std": "centroid_std", "spectral_rolloff_std": "rolloff_std"}
# Pre-registered candidates: name -> (features, kind); kind "rms" = sqrt(mean z^2), "rc" = robust correlation-aware.
REPRESENTATIONS = {"V2": (V2, "rms"), "V2_RC": (V2, "rc"),
                   "V2_G4C": (["spectral_centroid_mean", "spectral_rolloff_std"], "rms")}
LOFO = {f"V2_minus_{SHORT[f]}": ([g for g in V2 if g != f], "rms") for f in V2}

# Pre-registered category gate (restates stage9_preregistration.json).
K1_CONFIDENCE = 0.95
K1_MAX_RATE = 0.10
K2_MAX_RELATIVE_HALF_WIDTH = 0.20
K3_GAIN_TOLERANCE = 0.02
K3_NOISE_ORDER = (30.0, 20.0, 10.0)
K3_TILT_ORDER = (0.5, 0.9)
GAIN_CONDITIONS = (("gain", 0.5), ("gain", 1 / np.sqrt(2)), ("gain", np.sqrt(2)), ("gain", 2.0),
                   ("recording_gain", 0.5), ("recording_gain", 2.0))
REPORTED_ALPHAS = (Fraction(1, 100), ALPHA, Fraction(1, 10))
FAMILY_ORDER = {**PERTURBATION_ORDER, "recording_gain": [0.5, 1.0, 2.0]}
IDENTITY_VALUE = {"gain": 1.0, "noise": np.inf, "tilt": 0.0, "recording_gain": 1.0}
K4_EXCLUDED = ("analysis_summary.json", "category_gate_results.json", "e2e_timing.csv", "assessment_contract_v1.json",
               "assessment_reference_v1.json")
GATE_RULES = {
    "K1": "outer-LOSO exceedance at alpha 0.05: 95% session-bootstrap interval contains 0.05 and point estimate <= 0.10",
    "K2": "deployment reference: (c_hi - c_lo) / (2 c_alpha) <= 0.20",
    "K3": "OUTSIDE fraction non-decreasing over noise 30->20->10 dB and tilt 0.5->0.9; |gain - identity| <= 0.02",
    "K4": "a complete second run produces byte-identical CSV/JSON outputs",
}


class StageError(RuntimeError):
    """A Stage 9 consistency check failed."""


def require_clean_output_dir(output: Path) -> None:
    """Refuse an output directory that holds anything except the pre-registration.

    K4 compares every CSV/JSON present in the output directory with the reference run, so files left by an
    earlier run would enter K4 (Entry 13).  Running only into an empty directory makes K4 see exactly the files
    produced by the current run.
    """
    leftovers = sorted(p.name + ("/" if p.is_dir() else "") for p in output.iterdir() if p.name != PREREGISTRATION.name)
    if leftovers:
        shown = ", ".join(leftovers[:5]) + (f", ... ({len(leftovers)} entries)" if len(leftovers) > 5 else "")
        raise StageError(f"output directory {output} is not empty: {shown}. Stage 9 must run into an empty directory "
                         f"(only {PREREGISTRATION.name} may be present) because K4 compares every CSV/JSON in it; "
                         "remove the earlier outputs or choose another --output-dir")


# ── Representation scores ───────────────────────────────────────────────────

def robust_correlation(z: np.ndarray) -> np.ndarray:
    """Pearson-equivalent correlation from Spearman's rho: r = 2 sin(pi rho / 6) (normal-theory identity)."""
    k = z.shape[1]
    rho = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho[i, j] = rho[j, i] = stats.spearmanr(z[:, i], z[:, j]).statistic
    r = 2 * np.sin(np.pi * rho / 6)
    np.fill_diagonal(r, 1.0)
    return r


def representation_score(z: np.ndarray, kind: str, correlation: np.ndarray | None = None) -> np.ndarray:
    if kind == "rms":
        return rms_scores(z)
    inverse = np.linalg.inv(correlation)
    return np.sqrt(np.einsum("ij,jk,ik->i", z, inverse, z) / z.shape[1])


def fold_model(training: pd.DataFrame, features: Sequence[str], kind: str):
    fitted = fit_baseline(training, features)
    z_train = fitted.z_scores(training).to_numpy()
    correlation = robust_correlation(z_train) if kind == "rc" else None
    return fitted, correlation, z_train


def score_with(model, frame: pd.DataFrame, kind: str) -> np.ndarray:
    fitted, correlation, _ = model
    return representation_score(fitted.z_scores(frame).to_numpy(), kind, correlation)


def loso_scores(table: pd.DataFrame, sessions: pd.Series, features: Sequence[str], kind: str) -> pd.Series:
    """Each event of ``table`` (usable only) scored by a model fitted without its own session."""
    values = sessions.loc[table.index].to_numpy()
    scores = pd.Series(np.nan, index=table.index)
    for session in sorted(np.unique(values)):
        model = fold_model(table[values != session], features, kind)
        held = table[values == session]
        scores.loc[held.index] = score_with(model, held, kind)
    return scores


def outer_evaluation(table: pd.DataFrame, sessions: pd.Series, features: Sequence[str], kind: str,
                     alpha: Fraction = ALPHA):
    """Held-out score, p and OUTSIDE flag of every event of every session with usable events (nested calibration)."""
    usable = table["usable"].astype(bool).to_numpy()
    values = sessions.to_numpy()
    rows, models, cuts = [], {}, {}
    for session in sorted(np.unique(values[usable])):
        training = table[usable & (values != session)]
        model = fold_model(training, features, kind)
        calibration = np.sort(loso_scores(training, sessions, features, kind).to_numpy())
        cut = reference_cut(calibration, alpha)
        models[session], cuts[session] = model, cut
        targets = table[values == session]
        for index, score in zip(targets.index, score_with(model, targets, kind)):
            rows.append({"row": index, "session": session, "usable": bool(table.loc[index, "usable"]),
                         "score": float(score), "p": tail_probability(score, calibration),
                         "outside": bool(score > cut), "cut": cut, "n_calibration": len(calibration),
                         "in_sample_reference_median": float(np.median(representation_score(model[2], kind, model[1])))})
    return pd.DataFrame(rows).set_index("row").loc[[i for i in table.index if values[i] in cuts]], models, cuts


# ── Exceedance statistics ───────────────────────────────────────────────────

def session_bootstrap_rate(outside: np.ndarray, sessions: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                           seed: int = SEED, confidence: float = K1_CONFIDENCE) -> tuple[float, float, float]:
    labels = sorted(set(sessions.tolist()))
    k = np.array([outside[sessions == s].sum() for s in labels], dtype=float)
    n = np.array([(sessions == s).sum() for s in labels], dtype=float)
    rng = np.random.default_rng(seed)
    rates = np.empty(draws)
    for b in range(draws):
        chosen = rng.integers(0, len(labels), size=len(labels))
        rates[b] = k[chosen].sum() / n[chosen].sum()
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(rates, [tail, 100 - tail])
    return float(outside.mean()), float(low), float(high)


def overdispersion(outside: np.ndarray, sessions: np.ndarray, min_events: int = MIN_GROUP_SIZE,
                   draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> dict:
    """Chi-square heterogeneity of per-session exceedance counts; parametric-bootstrap p under a common rate."""
    labels, counts = np.unique(sessions, return_counts=True)
    tested = labels[counts >= min_events]
    k = np.array([outside[sessions == s].sum() for s in tested], dtype=float)
    n = np.array([(sessions == s).sum() for s in tested], dtype=float)
    rate = k.sum() / n.sum()

    def statistic(kk, r):
        return float(np.sum((kk - n * r) ** 2 / (n * r * (1 - r)))) if 0 < r < 1 else 0.0

    observed = statistic(k, rate)
    rng = np.random.default_rng(seed)
    simulated = np.empty(draws)
    for b in range(draws):
        kk = rng.binomial(n.astype(int), rate)
        simulated[b] = statistic(kk, kk.sum() / n.sum())
    return {"sessions_tested": int(len(tested)), "pooled_rate": rate, "chi_square": observed, "df": int(len(tested) - 1),
            "parametric_bootstrap_p": float((1 + np.sum(simulated >= observed)) / (draws + 1))}


def per_session_exceedance(outside: np.ndarray, sessions: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for session in sorted(set(sessions.tolist())):
        inside = sessions == session
        k, n = int(outside[inside].sum()), int(inside.sum())
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        rows.append({"session": session, "n": n, "outside": k, "rate": k / n, "exact_ci_low": ci.low,
                     "exact_ci_high": ci.high, "median_score": float(np.median(scores[inside]))})
    return pd.DataFrame(rows)


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    observed = np.mean(a == b)
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


# ── Representation gate (Stage 8 criteria recomputed per candidate) ─────────

def max_correlation_shift(frame: pd.DataFrame, sessions: pd.Series, features: Sequence[str]) -> float:
    usable = frame["usable"].astype(bool).to_numpy()
    pooled = frame.loc[usable, list(features)].to_numpy()
    worst = 0.0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            rho = stats.spearmanr(pooled[:, i], pooled[:, j]).statistic
            for session in sorted(sessions[usable].unique()):
                fold = frame.loc[usable & (sessions != session).to_numpy(), list(features)].to_numpy()
                worst = max(worst, abs(stats.spearmanr(fold[:, i], fold[:, j]).statistic - rho))
    return worst


def perturbed_frame(perturbed: pd.DataFrame, models_by_rep: Mapping[str, Mapping], features_by_rep: Mapping) -> pd.DataFrame:
    """Add z (V2 features, outer-fold V2 baseline) and every representation's score to the perturbed table."""
    out = perturbed.copy()
    for name, models in models_by_rep.items():
        features, kind = features_by_rep[name]
        scores = np.empty(len(out))
        for session, block in out.groupby("session"):
            positions = out.index.get_indexer(block.index)
            scores[positions] = score_with(models[session], block, kind)
        out[f"score[{name}]"] = scores
    return out


def distance_monotone(perturbed: pd.DataFrame, features: Sequence[str], kind: str, models: Mapping) -> dict:
    """Share of events whose distance from their own unperturbed point grows with intensity (noise, tilt)."""
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    result = {}
    for family in ("noise", "tilt"):
        blocks, position = [], None
        for k, magnitude in enumerate(FAMILY_ORDER[family]):
            if magnitude == IDENTITY_VALUE[family]:
                blocks.append(identity)
                position = k
            else:
                blocks.append(perturbed[(perturbed["transform"] == family)
                                        & np.isclose(perturbed["magnitude"], magnitude)].set_index("row"))
        index = identity.index
        columns = []
        for block in blocks:
            distance = np.empty(len(index))
            for session, rows in identity.groupby("session"):
                fitted, correlation, _ = models[session]
                delta = (fitted.z_scores(block.loc[rows.index]).to_numpy()
                         - fitted.z_scores(identity.loc[rows.index]).to_numpy())
                distance[index.get_indexer(rows.index)] = representation_score(delta, kind, correlation)
            columns.append(distance)
        result[family] = float(monotone_outward(np.column_stack(columns), position).mean())
    return result


def representation_gate(name: str, features: Sequence[str], kind: str, frame: pd.DataFrame, sessions: pd.Series,
                        evaluation: pd.DataFrame, models: Mapping, perturbed: pd.DataFrame, epsilon_r0: float,
                        rc_diagnostics: Mapping | None) -> list[dict]:
    rows = []

    def add(criterion, value, passed, detail=""):
        rows.append({"representation": name, "id": criterion, "value": value, "passed": bool(passed), "detail": detail})

    usable_eval = evaluation[evaluation["usable"]]
    session_values = usable_eval["session"].to_numpy()
    scores = usable_eval["score"].to_numpy()
    dependence = session_dependence(scores, session_values, np.ones(len(scores), bool))
    add("G1a", dependence["session_epsilon_squared"], dependence["session_epsilon_squared"] <= epsilon_r0,
        f"epsilon2 {dependence['session_epsilon_squared']:.4f} vs R0 {epsilon_r0:.4f}")
    ratio = float(np.median(scores) / np.median(usable_eval["in_sample_reference_median"]))
    add("G1b", ratio, ratio <= G1B_MAX_GENERALIZATION_RATIO)
    usable = frame["usable"].astype(bool).to_numpy()
    pooled = fit_baseline(frame[usable], features)
    shifts, ratios, positive = [], [], True
    for session, (fitted, _, _) in models.items():
        for f in features:
            shifts.append(abs(fitted.parameters[f].median - pooled.parameters[f].median) / pooled.parameters[f].scale)
            ratios.append(fitted.parameters[f].scale / pooled.parameters[f].scale)
            positive &= bool(np.isfinite(fitted.parameters[f].mad) and fitted.parameters[f].mad > 0)
    add("G2a", max(shifts), max(shifts) <= G2A_MAX_CENTER_SHIFT_SD)
    add("G2b", [min(ratios), max(ratios)], G2B_SCALE_RATIO_RANGE[0] <= min(ratios) and max(ratios) <= G2B_SCALE_RATIO_RANGE[1])
    add("G2c", positive, positive)
    shift = max_correlation_shift(frame, sessions, features)
    add("G3a", shift, shift <= G3A_MAX_CORRELATION_SHIFT)
    quantities = {f"z_{f}": (lambda b, f=f: b[f"z_{f}"].to_numpy(dtype=float)) for f in features}
    quantities[f"score[{name}]"] = lambda b: b[f"score[{name}]"].to_numpy(dtype=float)
    direction = direction_consistency(perturbed, quantities)
    gains = direction[(direction["quantity"] == f"score[{name}]") & direction["transform"].isin(["gain", "recording_gain"])]
    add("G4a", float(gains["p95_abs_delta"].max()), gains["p95_abs_delta"].max() <= G4A_MAX_P95_GAIN_DELTA)
    monotone = distance_monotone(perturbed, features, kind, models)
    add("G4b", monotone, min(monotone.values()) >= G4B_MIN_MONOTONE_FRACTION)
    cases, failures = [], []
    for quantity in [*(f"z_{f}" for f in features), f"score[{name}]"]:
        for transform, magnitude in G4C_LEVELS:
            row = direction[(direction["quantity"] == quantity) & (direction["transform"] == transform)
                            & np.isclose(direction["magnitude"], magnitude)].iloc[0]
            if row["responds"]:
                cases.append(f"{quantity} {transform} {magnitude:g}: {row['frac_sessions_same_sign']:.3f}")
                if not row["frac_sessions_same_sign"] >= G4C_MIN_SESSION_AGREEMENT:
                    failures.append(cases[-1])
    add("G4c", {"applicable_cases": len(cases), "failing_cases": len(failures)}, not failures,
        "FAILING: " + "; ".join(failures) if failures else "all: " + "; ".join(cases))
    noise = feature_perturbation_table(perturbed, features)
    noise = noise[noise["transform"] == "noise"].set_index("feature")
    fragile = [f for f in features if abs(noise.loc[f, "median_delta_z@30"]) >= G4D_FRAGILE_DELTA
               and noise.loc[f, "monotone_fraction"] < G4D_FRAGILE_MONOTONE]
    add("G4d", fragile, not fragile)
    data = pd.DataFrame({"score": scores, "background_rms": frame.loc[usable_eval.index, "background_rms"].to_numpy(),
                         "session": session_values})
    pooled_rho = float(stats.spearmanr(data["score"], data["background_rms"]).statistic)
    within_rho = float(within_group_spearman(data, ["score", "background_rms"], "session").loc["score", "background_rms"])
    add("G5a", pooled_rho, abs(pooled_rho) < G5_MAX_ABS_RHO)
    add("G5b", within_rho, abs(within_rho) < G5_MAX_ABS_RHO)
    if rc_diagnostics is not None:
        add("RC_admissible", rc_diagnostics, rc_diagnostics["passed"], "Stage 7 multivariate gate thresholds")
    return rows


def rc_admissibility(frame: pd.DataFrame, models: Mapping) -> dict:
    usable = frame["usable"].astype(bool).to_numpy()
    _, pooled_correlation, _ = fold_model(frame[usable], V2, "rc")
    conditions, shifts, per_feature = [], [], []
    for _, (fitted, correlation, z_train) in models.items():
        conditions.append(float(np.linalg.cond(correlation)))
        shifts.append(float(np.abs(correlation - pooled_correlation).max()))
        per_feature.append(len(z_train) / len(V2))
    passed = (min(per_feature) >= MCD_MIN_EVENTS_PER_FEATURE and max(conditions) <= MCD_MAX_CONDITION_NUMBER
              and max(shifts) <= MCD_MAX_CORRELATION_SHIFT)
    return {"passed": bool(passed), "max_condition_number": max(conditions), "max_correlation_shift": max(shifts),
            "min_training_events_per_feature": min(per_feature), "pooled_correlation": pooled_correlation.tolist()}


def exceedance_k1(evaluation: pd.DataFrame) -> dict:
    usable = evaluation[evaluation["usable"]]
    rate, low, high = session_bootstrap_rate(usable["outside"].to_numpy(), usable["session"].to_numpy())
    passed = low <= float(ALPHA) <= high and rate <= K1_MAX_RATE
    return {"rate": rate, "ci_low": low, "ci_high": high, "n": int(len(usable)), "outside": int(usable["outside"].sum()),
            "passed": bool(passed)}


# ── Perturbation behaviour of the assessment (K3) ───────────────────────────

def perturbation_flags(perturbed: pd.DataFrame, references: Mapping[str, AssessmentReference]) -> pd.DataFrame:
    rows = []
    for session, block in perturbed.groupby("session"):
        reference = references[session]
        z = reference.baseline.z(block[V2].to_numpy())
        scores = rms_scores(z)
        calibration = np.asarray(reference.calibration_scores)
        for (_, row), score in zip(block.iterrows(), scores):
            rows.append({"row": int(row["row"]), "session": session, "transform": row["transform"],
                         "magnitude": float(row["magnitude"]), "score": float(score),
                         "p": tail_probability(score, calibration), "outside": bool(score > reference.cut)})
    flags = pd.DataFrame(rows)
    summary = (flags.groupby(["transform", "magnitude"], sort=False)
               .agg(n=("outside", "size"), outside_fraction=("outside", "mean"), median_score=("score", "median"),
                    median_p=("p", "median")).reset_index())
    return flags, summary


def k3_evaluation(summary: pd.DataFrame) -> dict:
    def fraction(transform, magnitude):
        block = summary[(summary["transform"] == transform) & np.isclose(summary["magnitude"], magnitude)]
        return float(block["outside_fraction"].iloc[0])

    identity = fraction("identity", 0.0)
    noise = [fraction("noise", m) for m in K3_NOISE_ORDER]
    tilt = [fraction("tilt", m) for m in K3_TILT_ORDER]
    gains = {f"{t} x{m:.3g}": fraction(t, m) - identity for t, m in GAIN_CONDITIONS}
    passed = (all(np.diff(noise) >= 0) and all(np.diff(tilt) >= 0)
              and max(abs(v) for v in gains.values()) <= K3_GAIN_TOLERANCE)
    return {"identity": identity, "noise_30_20_10": noise, "tilt_0.5_0.9": tilt, "tilt_-0.5": fraction("tilt", -0.5),
            "gain_minus_identity": gains, "passed": bool(passed)}


# ── Stability analyses (reported) ───────────────────────────────────────────

def event_removal(table: pd.DataFrame, sessions: pd.Series, cut: float) -> pd.DataFrame:
    usable = table[table["usable"].astype(bool)]
    rows = []
    for index in usable.index:
        rest = usable.drop(index)
        cut_without = reference_cut(loso_calibration_scores(rest, sessions).to_numpy())
        rows.append({"row": int(index), "recording_file": table.loc[index, "recording_file"],
                     "event_id": int(table.loc[index, "event_id"]), "cut_without_event": cut_without,
                     "relative_change": (cut_without - cut) / cut})
    return pd.DataFrame(rows)


# ── End-to-end runs of the canonical pipeline ───────────────────────────────

def run_end_to_end(data_dir: Path, references_for: Mapping[str, AssessmentReference], default: AssessmentReference,
                   session_of: Mapping[str, str], detector: OnnxEventClassifier, mode: str):
    recording_rows, event_rows, timing, outputs = [], [], [], {}
    for wav in sorted(data_dir.glob("*.wav")):
        session = session_of[wav.name]
        reference = references_for.get(session, default)
        started = time.perf_counter()
        try:
            waveform, rate = read_wav(wav)
            output = assess_recording(waveform, rate, reference, detector=detector, input_domain="reference_dataset",
                                      recording_id=wav.name)
            failure = None
        except Exception as error:          # recorded, never silent
            output, failure = None, f"{type(error).__name__}: {error}"
        timing.append({"mode": mode, "recording_file": wav.name, "seconds": time.perf_counter() - started})
        recording_rows.append({"mode": mode, "recording_file": wav.name, "session": session,
                               "reference_id": reference.reference_id, "failure": failure,
                               "recording_status": None if output is None else output["recording_status"],
                               "error": None if output is None else output["error"],
                               "n_events": None if output is None else output["n_events"],
                               "n_scoreable": None if output is None else output["n_scoreable"],
                               "n_outside_reference_range": None if output is None else output["n_outside_reference_range"]})
        if output is None:
            continue
        outputs[wav.name] = output
        for event in output["events"]:
            row = {"mode": mode, "recording_file": wav.name, "session": session}
            for key in EVENT_KEYS:
                value = event[key]
                if key in ("feature_values", "feature_deviations"):
                    for f in V2:
                        row[f"{key}.{f}"] = None if value is None else value[f]
                elif key == "segmentation_deviation_range":
                    row["segmentation_min"], row["segmentation_max"] = (None, None) if value is None else value
                elif key == "not_scoreable_reasons":
                    row[key] = ";".join(value)
                else:
                    row[key] = value
            event_rows.append(row)
    return pd.DataFrame(recording_rows), pd.DataFrame(event_rows), pd.DataFrame(timing), outputs


def e2e_summary(recordings: pd.DataFrame, events: pd.DataFrame) -> dict:
    numeric = events.select_dtypes(include=[np.number])
    scoreable = events[events["scoreability"] == "SCOREABLE"]
    reasons = events.loc[events["scoreability"] == "NOT_SCOREABLE", "not_scoreable_reasons"].value_counts().to_dict()
    return {"recordings": int(len(recordings)), "failed_recordings": int(recordings["failure"].notna().sum()),
            "failures": recordings.loc[recordings["failure"].notna(), ["recording_file", "failure"]].to_dict("records"),
            "recording_status": recordings["recording_status"].value_counts().to_dict(),
            "events": int(len(events)), "scoreable": int(len(scoreable)),
            "not_scoreable": int((events["scoreability"] == "NOT_SCOREABLE").sum()),
            "not_scoreable_reason_combinations": reasons,
            "assessment": events["assessment"].value_counts().to_dict(),
            "reliability_of_assessed": scoreable["assessment_reliability"].value_counts(dropna=False).to_dict(),
            "non_finite_numeric_values": int((~np.isfinite(numeric.to_numpy(dtype=float)) & numeric.notna().to_numpy()).sum()),
            "aggregate_deviation_percentiles": dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                                        np.percentile(scoreable["aggregate_deviation"], [5, 25, 50, 75, 95]).tolist())),
            "tail_probability_percentiles": dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                                     np.percentile(scoreable["reference_tail_probability"], [5, 25, 50, 75, 95]).tolist())),
            "segmentation_variants": scoreable["segmentation_variants"].value_counts().sort_index().to_dict()}


# ── Golden vectors, contract, reproducibility ───────────────────────────────

def write_golden(output: Path, reference: AssessmentReference, detector, data_dir: Path) -> dict:
    manifest_v2 = json.loads((V2_GOLDEN / "golden_manifest.json").read_text(encoding="utf-8"))
    golden = output / "golden"
    (golden / "expected").mkdir(parents=True, exist_ok=True)
    cases = []
    for case in manifest_v2["cases"]:
        source = case["input"]
        if source["type"] == "dataset_wav":
            path = data_dir / source["file"]
            waveform, rate = read_wav(path)
            domain = "reference_dataset"
        elif source["type"] == "synthetic_wav":
            path = V2_GOLDEN / source["path"]
            waveform, rate = read_wav(path)
            domain = "unknown"
        elif case["case"] == "input_error_sample_rate_16k":
            waveform, rate, domain, path = np.zeros(16000, np.float32), 16000, "unknown", None
        else:
            waveform, rate, domain, path = np.zeros(MIN_SAMPLES - 1, np.float32), SAMPLE_RATE, "unknown", None
        result = assess_recording(waveform, rate, reference, detector=detector, input_domain=domain,
                                  recording_id=None if path is None else path.name)
        (golden / "expected" / f"{case['case']}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                                  encoding="utf-8")
        cases.append({"case": case["case"], "input": source, "input_domain": domain,
                      "expected": f"expected/{case['case']}.json"})
    manifest = {"contract_version": ASSESSMENT_CONTRACT_VERSION, "reference_id": reference.reference_id,
                "purpose": "Conformance vectors for re-implementations of the assessment layer (engineering, not validation).",
                "inputs": "Same inputs as results/v2_validation/golden (dataset WAVs by file name and SHA-256).",
                "tolerances": {"inference_fields": "as results/v2_validation/golden/golden_manifest.json",
                               "reference_tail_probability": "exact (it is a count ratio) when aggregate_deviation is "
                                                             "within 0.01 of the expected value and not within 0.01 of a "
                                                             "calibration score",
                               "assessment_and_reliability": "exact"},
                "cases": cases}
    (golden / "golden_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def compare_runs(current: Path, reference_run: Path) -> dict:
    files = sorted(p.relative_to(current) for p in current.rglob("*")
                   if p.is_file() and p.suffix in (".csv", ".json") and p.name not in K4_EXCLUDED
                   and p.name != PREREGISTRATION.name)
    mismatches = [str(f).replace("\\", "/") for f in files
                  if not (reference_run / f).exists() or (reference_run / f).read_bytes() != (current / f).read_bytes()]
    return {"files_compared": len(files), "mismatches": mismatches, "passed": not mismatches}


def cut_text(reference: AssessmentReference) -> str:
    n = len(reference.calibration_scores)
    m = tail_count_limit(n, reference.alpha)
    return (f"one-sided 95% upper reference limit c* = the (n - m)-th smallest c_i, m = floor(alpha (n + 1) - 1): "
            f"c* = {reference.cut!r} (rank {n - m} of {n}); p <= alpha iff a > c*; effective attainable level "
            f"(m + 1) / (n + 1) = {m + 1}/{n + 1}")


def build_contract(reference: AssessmentReference, reference_sha256: str, outcome: str, gate: Mapping,
                   representation: Mapping, limitations: Sequence[str]) -> dict:
    return {
        "contract_version": ASSESSMENT_CONTRACT_VERSION,
        "status": "CATEGORICAL_ASSESSMENT_ADOPTED" if outcome == "PASS" else "CONTINUOUS_ONLY",
        "status_meaning": ("The pre-registered Stage 9 gate (K1-K4), evaluated held-out by recording session on the same "
                           "PRISM corpus, passed: events receive WITHIN_REFERENCE_RANGE / OUTSIDE_REFERENCE_RANGE against the "
                           "one-sided 95% upper reference limit, with a STABLE / BORDERLINE label." if outcome == "PASS" else
                           "The pre-registered Stage 9 gate failed: events receive the continuous deviation and empirical "
                           "reference-tail probability only (assessment CONTINUOUS_ONLY)."),
        "representation_status": ("V2 representation retained; inherits DRAFT_NOT_FROZEN from Stage 8 (G4c: per-feature "
                                  "direction under strong spectral tilt). The assessment's calibration was checked held-out "
                                  "by recording session on the same PRISM corpus only; it is not externally validated and "
                                  "not validated for new users, devices, microphones or PRISM hardware."),
        "representation_comparison": representation,
        "pipeline": ["8 kHz mono audio", "inference contract prism-inference-v2.0 (detector, Inhale events, scoreability, "
                     "V2 features, robust z, aggregate deviation)",
                     "empirical reference-tail probability (+1-corrected empirical upper-tail probability; not a "
                     "hypothesis-test p-value)",
                     "reference-range assessment against the one-sided 95% upper reference limit (if adopted)",
                     "stability (reference-session resampling band, +/-1 detector-window boundary shifts)"],
        "inference_contract": "results/v2_validation/inference_contract_v2.json (unchanged)",
        "reference": {"file": "results/final_assessment/assessment_reference_v1.json", "sha256": reference_sha256,
                      "reference_id": reference.reference_id, "n_events": len(reference.calibration_scores),
                      "n_sessions": reference.n_sessions, "alpha": str(reference.alpha), "cut": reference.cut,
                      "band": list(reference.band), "band_percentiles": list(BAND_PERCENTILES),
                      "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": SEED,
                      "fitting_at_inference": "never; any update is a new reference_id requiring its own validation"},
        "assessment": {
            "aggregate_deviation": "sqrt(mean_j z_j^2) over the 4 V2 features (the inference contract's anomaly_score)",
            "reference_tail_probability": "empirical reference-tail probability p = (1 + #{i : c_i >= a}) / (n + 1); "
                                          "a = the event's aggregate deviation; c_i = leave-one-session-out deviations of "
                                          "the reference's usable events (n = n_events). An empirical upper-tail "
                                          "probability with the +1 correction: not a hypothesis-test p-value, and no "
                                          "finite-sample guarantee is claimed",
            "cut": cut_text(reference),
            "categories": {"WITHIN_REFERENCE_RANGE": "p > 0.05: deviation at or below the one-sided 95% upper reference limit",
                           "OUTSIDE_REFERENCE_RANGE": "p <= 0.05: deviation above the one-sided 95% upper reference limit",
                           "meaning": "acoustic deviation within or outside the empirical reference distribution of the "
                                      "PRISM reference corpus; not a normal/abnormal, anomaly, technique-quality or "
                                      "clinical classification",
                           "CONTINUOUS_ONLY": "used when the categorical statement is not adopted",
                           "NOT_ASSESSED": "event not scoreable (usability rule v1); reasons are given"},
            "reliability": {"STABLE": "the category is unchanged under resampling of the reference sessions and under "
                                      "every allowed +/-1 detector-window boundary shift (the event deviation and every "
                                      "segmentation variant lie on one side of the band); not a probability that the "
                                      "category is correct",
                            "BORDERLINE": "the category changes under at least one of those perturbations",
                            "band": "5th-95th percentile of the session-bootstrap cut",
                            "segmentation_variants": "event bounds moved by one detector stride (0.016 s) at start and/or "
                                                     "end; reference float formula"},
            "meaning": INTERPRETATION,
        },
        "category_gate": {"preregistration": _relative(PREREGISTRATION), "preregistration_sha256": PREREGISTRATION_SHA256,
                          "outcome": outcome, "criteria": gate,
                          "scope": "held-out by recording session on the same PRISM corpus: an internal same-corpus "
                                   "consistency/calibration check, not external validation"},
        "output": {"schema_file": "results/final_assessment/assessment_output.schema.json",
                   "categories": list(CATEGORIES), "reliability": list(RELIABILITY),
                   "forbidden_derived_outputs": list(FORBIDDEN_DERIVED_OUTPUTS)},
        "conformance": {"golden_manifest": "results/final_assessment/golden/golden_manifest.json",
                        "reference_implementation": "src/prism_assessment.py (python src/prism_assessment.py <wav>)"},
        "personalization": "not implemented: the data contain no user or device identifiers (sessions are inferred "
                           "recording sittings); see PRISM_RESEARCH_LOG.md Entries 12-13",
        "known_limitations": list(limitations),
    }


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_calibration(per_session: pd.DataFrame, k1: Mapping, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = per_session.sort_values("median_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    y = np.arange(len(data))
    ax.hlines(y, data["exact_ci_low"], data["exact_ci_high"], color=INK_MUTED, linewidth=1.4)
    ax.scatter(data["rate"], y, color=SERIES[0], zorder=3, s=22)
    ax.axvline(float(ALPHA), color=BASELINE_INK, linestyle="--", linewidth=1)
    ax.axvspan(k1["ci_low"], k1["ci_high"], color=SERIES[0], alpha=0.12, linewidth=0)
    ax.set_yticks(y, [f"{s} (n={n})" for s, n in zip(data["session"], data["n"])], fontsize=7, color=INK_SECONDARY)
    ax.set_xlabel("held-out share OUTSIDE_REFERENCE_RANGE (exact 95% CI)", color=INK_SECONDARY)
    ax.set_title(f"Leave-one-session-out calibration at alpha 0.05: pooled {k1['rate']:.3f} "
                 f"[{k1['ci_low']:.3f}, {k1['ci_high']:.3f}]", color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reference(reference: AssessmentReference, heldout: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    bins = np.linspace(0, max(4.0, float(heldout["score"].max()) + 0.1), 50)
    ax.hist(reference.calibration_scores, bins=bins, color=SERIES[0], alpha=0.55, label="reference (LOSO) scores")
    excluded = heldout[~heldout["usable"]]
    ax.hist(excluded["score"], bins=bins, color=SERIES[1], alpha=0.7, label="Stage 1-excluded events (diagnostic)")
    ax.axvspan(*reference.band, color=INK_MUTED, alpha=0.18, linewidth=0, label="bootstrap band (5-95%)")
    ax.axvline(reference.cut, color=BASELINE_INK, linewidth=1.2, label=f"cut {reference.cut:.3f} (alpha 0.05)")
    ax.set_xlabel("aggregate deviation (rms of V2 robust z)", color=INK_SECONDARY)
    ax.set_ylabel("events", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_perturbation(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    identity = float(summary.loc[summary["transform"] == "identity", "outside_fraction"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    panels = (("noise", [np.inf, 30.0, 20.0, 10.0], ["none", "30 dB", "20 dB", "10 dB"], "white noise (SNR)"),
              ("tilt", [-0.5, 0.0, 0.5, 0.9], ["-0.5", "0", "0.5", "0.9"], "spectral tilt a"),
              ("gain", [0.5, 1 / np.sqrt(2), 1.0, np.sqrt(2), 2.0], ["-6", "-3", "0", "+3", "+6"], "event gain (dB)"))
    for ax, (transform, magnitudes, labels, title) in zip(axes, panels):
        values = []
        for m in magnitudes:
            if m == IDENTITY_VALUE[transform]:
                values.append(identity)
            else:
                values.append(float(summary.loc[(summary["transform"] == transform)
                                                & np.isclose(summary["magnitude"], m), "outside_fraction"].iloc[0]))
        ax.plot(range(len(values)), values, marker="o", color=SERIES[0])
        ax.axhline(float(ALPHA), color=BASELINE_INK, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(values)), labels, color=INK_SECONDARY)
        ax.set_title(title, color=INK, fontsize=10)
        _style(ax)
    axes[0].set_ylabel("share OUTSIDE_REFERENCE_RANGE", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_analysis(dataset_csv: str | Path = DEFAULT_DATASET_CSV, data_dir: str | Path = config.DATA_DIR,
                 output_dir: str | Path = DEFAULT_OUTPUT_DIR, reference_run: str | Path | None = None,
                 make_plots: bool = True) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    require_clean_output_dir(output)          # before anything is read, computed or written
    data_dir = Path(data_dir)
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise StageError("stage9_preregistration.json differs from the recorded pre-registration")
    inputs = [Path(dataset_csv), Path(DEFAULT_SELECTION_JSON), V2_PERTURBED, STAGE7_CANDIDATES, PREREGISTRATION,
              Path(DEFAULT_MODEL_PATH)]
    hashes_before = {str(p): _sha256(p) for p in inputs}

    table = load_stage1_table(dataset_csv).reset_index(drop=True)
    session_of = recording_sessions(data_dir)
    sessions = table["recording_file"].map(session_of)
    populations = assign_populations(table)
    candidates = pd.read_csv(STAGE7_CANDIDATES)
    if not ((candidates["recording_file"] == table["recording_file"]) & (candidates["event_id"] == table["event_id"])).all():
        raise StageError("Stage 7 candidate features are not aligned with the Stage 1 table")
    frame = table.assign(background_rms=candidates["background_rms"])
    usable = frame["usable"].astype(bool).to_numpy()
    v1 = load_feature_selection(DEFAULT_SELECTION_JSON, dataset_csv)

    # ── Part A: representation comparison (pre-registered decision rule) ──
    r0_eval, _, _ = outer_evaluation(frame, sessions, v1, "rms")
    r0_usable = r0_eval[r0_eval["usable"]]
    epsilon_r0 = session_dependence(r0_usable["score"].to_numpy(), r0_usable["session"].to_numpy(),
                                    np.ones(len(r0_usable), bool))["session_epsilon_squared"]
    perturbed_raw = pd.read_csv(V2_PERTURBED)
    evaluations, models_by_rep, gate_rows, k1_by_rep = {}, {}, [], {}
    for name, (features, kind) in REPRESENTATIONS.items():
        evaluations[name], models_by_rep[name], _ = outer_evaluation(frame, sessions, features, kind)
    perturbed = perturbed_raw.copy()
    v2_models = models_by_rep["V2"]
    for f in V2:                                   # z of perturbed rows from each event's outer-fold V2 baseline
        perturbed[f"z_{f}"] = np.nan
    for session, block in perturbed.groupby("session"):
        z = v2_models[session][0].z_scores(block)
        for f in V2:
            perturbed.loc[block.index, f"z_{f}"] = z[f"z_{f}"]
    stored_z_diff = float(max(np.abs(perturbed[f"z_{f}"] - perturbed_raw[f"z_{f}"]).max() for f in V2))
    perturbed = perturbed_frame(perturbed, models_by_rep, REPRESENTATIONS)
    rc_check = rc_admissibility(frame, models_by_rep["V2_RC"])
    for name, (features, kind) in REPRESENTATIONS.items():
        if name == "V2_RC" and not rc_check["passed"]:
            gate_rows.append({"representation": name, "id": "RC_admissible", "value": rc_check, "passed": False,
                              "detail": "not compared: Stage 7 multivariate gate failed"})
            continue
        gate_rows += representation_gate(name, features, kind, frame, sessions, evaluations[name], models_by_rep[name],
                                         perturbed, epsilon_r0, rc_check if name == "V2_RC" else None)
        k1_by_rep[name] = exceedance_k1(evaluations[name])
    gate_table = pd.DataFrame(gate_rows)
    representation_summary = {}
    for name in REPRESENTATIONS:
        block = gate_table[gate_table["representation"] == name]
        failing = block.loc[~block["passed"], "id"].tolist()
        k1 = k1_by_rep.get(name)
        representation_summary[name] = {
            "features": REPRESENTATIONS[name][0], "kind": REPRESENTATIONS[name][1], "failing_criteria": failing,
            "passes_all_gate_criteria": not failing, "k1": k1,
            "session_epsilon_squared": next((r["value"] for r in gate_rows if r["representation"] == name and r["id"] == "G1a"),
                                            None)}
    qualifying = [n for n in ("V2_RC", "V2_G4C") if representation_summary[n]["passes_all_gate_criteria"]
                  and representation_summary[n]["k1"] and representation_summary[n]["k1"]["passed"]
                  and not representation_summary["V2"]["passes_all_gate_criteria"]]
    selected = (sorted(qualifying, key=lambda n: (representation_summary[n]["session_epsilon_squared"],
                                                  len(REPRESENTATIONS[n][0])))[0] if qualifying else "V2")
    representation_decision = {"selected": selected, "qualifying_alternatives": qualifying,
                               "rule": "an alternative replaces V2 only if it passes every Stage 8 criterion including "
                                       "G4c and K1 while V2 does not; otherwise V2 is retained",
                               "summary": representation_summary}
    if selected != "V2":
        raise StageError(f"pre-registered rule selected {selected}; the production library implements V2 only - "
                         "implement the selected representation before continuing")
    gate_table.assign(value=gate_table["value"].map(lambda v: json.dumps(v, default=_json_default))).to_csv(
        output / "representation_gate.csv", index=False)

    # ── Part B: references for V2 (production code) and LOSO calibration ──
    deployment = fit_reference(frame, sessions, REFERENCE_ID, categorical=True,
                               source={"dataset": _relative(Path(dataset_csv)), "dataset_sha256": _sha256(Path(dataset_csv)),
                                       "stage": "Stage 9 (PRISM_RESEARCH_LOG.md Entry 12)"})
    outer_refs = {s: fit_reference(frame, sessions, f"{REFERENCE_ID}/loso/{s}", categorical=True, exclude_sessions=[s])
                  for s in sorted(sessions[usable].unique())}
    v2_eval = evaluations["V2"]
    check_cut = max(abs(outer_refs[s].cut - c) for s, c in
                    {s: v2_eval.loc[v2_eval["session"] == s, "cut"].iloc[0] for s in outer_refs}.items())
    heldout_rows = []
    for index in v2_eval.index:
        session = v2_eval.loc[index, "session"]
        reference = outer_refs[session]
        score = float(rms_scores(reference.baseline.z(frame.loc[[index], V2].to_numpy()))[0])
        heldout_rows.append({"row": int(index), "recording_file": frame.loc[index, "recording_file"],
                             "event_id": int(frame.loc[index, "event_id"]), "session": session,
                             "usable": bool(frame.loc[index, "usable"]),
                             "exclusion_group": populations.loc[index, "exclusion_group"], "score": score,
                             "p": reference.tail_probability(score), "outside": reference.outside(score),
                             "cut": reference.cut, "band_low": reference.band[0], "band_high": reference.band[1],
                             **{f"z_{f}": float(z) for f, z in zip(V2, reference.baseline.z(frame.loc[index, V2].to_numpy(float)))}})
    heldout = pd.DataFrame(heldout_rows)
    check_scores = float(np.abs(heldout["score"].to_numpy() - v2_eval["score"].to_numpy()).max())
    if check_cut > 0 or check_scores > 1e-12:
        raise StageError(f"production references disagree with the generic evaluation (cut {check_cut}, score {check_scores})")
    heldout.to_csv(output / "heldout_assessment.csv", index=False)
    held_usable = heldout[heldout["usable"]]

    k1 = exceedance_k1(heldout)
    per_session = per_session_exceedance(held_usable["outside"].to_numpy(), held_usable["session"].to_numpy(),
                                         held_usable["score"].to_numpy())
    per_session.to_csv(output / "calibration_by_session.csv", index=False)
    dispersion = overdispersion(held_usable["outside"].to_numpy(), held_usable["session"].to_numpy())
    alphas = []
    for alpha in REPORTED_ALPHAS:
        flags = np.array([held_usable.loc[i, "score"] > reference_cut(outer_refs[s].calibration_scores, alpha)
                          for i, s in zip(held_usable.index, held_usable["session"])])
        rate, low, high = session_bootstrap_rate(flags, held_usable["session"].to_numpy())
        alphas.append({"alpha": float(alpha), "rate": rate, "ci_low": low, "ci_high": high})
    ks = stats.kstest(held_usable["p"].to_numpy(), "uniform")
    calibration = {"k1": k1, "overdispersion": dispersion, "by_alpha": alphas,
                   "pit_ks_statistic": float(ks.statistic), "pit_ks_p_descriptive": float(ks.pvalue),
                   "held_out_p_percentiles": dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                                      np.percentile(held_usable["p"], [5, 25, 50, 75, 95]).tolist()))}

    # ── K2: reference stability; reported: session and event removal ──
    half_width = (deployment.band[1] - deployment.band[0]) / (2 * deployment.cut)
    k2 = {"cut": deployment.cut, "band": list(deployment.band), "relative_half_width": half_width,
          "passed": bool(half_width <= K2_MAX_RELATIVE_HALF_WIDTH)}
    session_cuts = pd.DataFrame([{"held_out_session": s, "cut": r.cut, "band_low": r.band[0], "band_high": r.band[1],
                                  "n_calibration": len(r.calibration_scores),
                                  "relative_to_deployment": r.cut / deployment.cut} for s, r in outer_refs.items()])
    session_cuts.to_csv(output / "session_removal_cuts.csv", index=False)
    removal = event_removal(frame, sessions, deployment.cut)
    removal.to_csv(output / "event_removal_cuts.csv", index=False)
    max_change = float(removal["relative_change"].abs().max())
    near = int(np.sum(np.abs(np.asarray(deployment.calibration_scores) - deployment.cut) <= max_change * deployment.cut))
    stability = {"session_removal": {"min_cut": float(session_cuts["cut"].min()), "max_cut": float(session_cuts["cut"].max()),
                                     "max_relative_deviation": float((session_cuts["relative_to_deployment"] - 1).abs().max())},
                 "event_removal": {"max_relative_change": max_change,
                                   "events_changing_cut": int((removal["relative_change"] != 0).sum()),
                                   "reference_events_within_max_change_of_cut": near}}

    # ── K3: controlled perturbations ──
    flags, flag_summary = perturbation_flags(perturbed_raw, outer_refs)
    flag_summary.to_csv(output / "perturbation_assessment.csv", index=False)
    k3 = k3_evaluation(flag_summary)
    identity_flags = flags[flags["transform"] == "identity"].set_index("row")
    identity_check = bool((identity_flags.loc[held_usable["row"], "outside"].to_numpy() == held_usable["outside"].to_numpy()).all())

    # ── Reported: dominance, leave-one-feature-out agreement, natural extremes ──
    outside_usable = held_usable[held_usable["outside"]]
    z_columns = [f"z_{f}" for f in V2]

    def dominant(block):
        squared = block[z_columns].to_numpy() ** 2
        return pd.Series(np.array(V2)[squared.argmax(axis=1)]).value_counts().to_dict(), \
            float(np.median(squared.max(axis=1) / squared.sum(axis=1)))

    dom_outside, share_outside = dominant(outside_usable) if len(outside_usable) else ({}, float("nan"))
    dom_all, share_all = dominant(held_usable)
    lofo_rows = []
    for name, (features, kind) in LOFO.items():
        lofo_eval, _, _ = outer_evaluation(frame, sessions, features, kind)
        a = held_usable["outside"].to_numpy()
        b = lofo_eval.loc[held_usable["row"], "outside"].to_numpy()
        lofo_rows.append({"representation": name, "outside_rate": float(b.mean()), "agreement_with_V2": float(np.mean(a == b)),
                          "cohens_kappa": cohens_kappa(a, b), "v2_outside_also_outside": float(b[a].mean()) if a.any() else np.nan})
    lofo = pd.DataFrame(lofo_rows)
    lofo.to_csv(output / "feature_ablation_agreement.csv", index=False)
    excluded = heldout[~heldout["usable"]]
    natural = {"usable": {"n": int(len(held_usable)), "outside_rate": float(held_usable["outside"].mean()),
                          "median_score": float(held_usable["score"].median())},
               "excluded_all": {"n": int(len(excluded)), "outside_rate": float(excluded["outside"].mean()),
                                "median_score": float(excluded["score"].median()),
                                "cliffs_delta_score_vs_usable": cliffs_delta(excluded["score"], held_usable["score"])},
               "by_exclusion_group": {g: {"n": int(len(b)), "outside_rate": float(b["outside"].mean()),
                                          "median_score": float(b["score"].median())}
                                      for g, b in excluded.groupby("exclusion_group")},
               "note": "diagnostic only: excluded events are NOT_ASSESSED in production and are not anomalies"}
    reported = {"dominant_feature_outside": dom_outside, "median_dominant_share_outside": share_outside,
                "dominant_feature_all_usable": dom_all, "median_dominant_share_all_usable": share_all,
                "natural_extremes": natural, "identity_flags_match_heldout": identity_check,
                "stored_perturbed_z_max_abs_diff": stored_z_diff}

    # ── End-to-end: canonical pipeline, LOSO references (held-out evaluation) ──
    detector = OnnxEventClassifier()
    rec_loso, events_loso, timing_loso, _ = run_end_to_end(data_dir, outer_refs, deployment, session_of, detector, "loso")
    events_loso.to_csv(output / "e2e_loso_events.csv", index=False)
    rec_loso.to_csv(output / "e2e_loso_recordings.csv", index=False)
    e2e_loso = e2e_summary(rec_loso, events_loso)
    merged = events_loso.merge(heldout[["recording_file", "event_id", "score", "p", "outside", "usable"]],
                               on=["recording_file", "event_id"], how="left")
    stage1_counts = table.groupby("recording_file").size()
    e2e_counts = rec_loso.set_index("recording_file")["n_events"]
    scoreable_e2e = merged["scoreability"] == "SCOREABLE"
    consistency = {
        "events_e2e": int(len(events_loso)), "events_stage1": int(len(table)),
        "recordings_with_event_count_mismatch": int(sum(e2e_counts.get(r, 0) != stage1_counts.get(r, 0)
                                                        for r in set(e2e_counts.index) | set(stage1_counts.index))),
        "scoreability_vs_stage1_usable_mismatches": int((scoreable_e2e != merged["usable"].fillna(False)).sum()),
        "max_abs_score_diff": float(np.abs(merged.loc[scoreable_e2e, "aggregate_deviation"]
                                           - merged.loc[scoreable_e2e, "score"]).max()),
        "p_mismatches": int((merged.loc[scoreable_e2e, "reference_tail_probability"] != merged.loc[scoreable_e2e, "p"]).sum()),
        "category_mismatches": int(((merged.loc[scoreable_e2e, "assessment"] == "OUTSIDE_REFERENCE_RANGE")
                                    != merged.loc[scoreable_e2e, "outside"].astype(bool)).sum()),
    }
    assessed = events_loso[events_loso["scoreability"] == "SCOREABLE"]
    reliability = {"overall": assessed["assessment_reliability"].value_counts().to_dict(),
                   "by_assessment": {a: b["assessment_reliability"].value_counts().to_dict()
                                     for a, b in assessed.groupby("assessment")},
                   "median_segmentation_range": float((assessed["segmentation_max"] - assessed["segmentation_min"]).median()),
                   "p95_segmentation_range": float(np.percentile(assessed["segmentation_max"] - assessed["segmentation_min"], 95))}

    # ── K4 and the gate ──
    rerun = compare_runs(output, Path(reference_run)) if reference_run else None
    gate = {"K1": {"rule": GATE_RULES["K1"], **k1}, "K2": {"rule": GATE_RULES["K2"], **k2},
            "K3": {"rule": GATE_RULES["K3"], **k3},
            "K4": {"rule": GATE_RULES["K4"], **(rerun or {"passed": None, "detail": "no reference run given"})}}
    passed = [g["passed"] for g in gate.values()]
    outcome = "FAIL" if any(p is False for p in passed) else ("PASS" if all(passed) else "INCOMPLETE")
    (output / "category_gate_results.json").write_text(json.dumps({"outcome": outcome, "criteria": gate,
                                                                   "preregistration_sha256": PREREGISTRATION_SHA256},
                                                                  indent=2, default=_json_default) + "\n", encoding="utf-8")

    # ── Production reference, canonical deployment run, contract, golden vectors ──
    final = replace(deployment, categorical=outcome == "PASS")
    reference_path = output / "assessment_reference_v1.json"
    write_reference(final, reference_path)
    rec_dep, events_dep, timing_dep, _ = run_end_to_end(data_dir, {}, final, session_of, detector, "deployment")
    events_dep.to_csv(output / "e2e_deployment_events.csv", index=False)
    rec_dep.to_csv(output / "e2e_deployment_recordings.csv", index=False)
    pd.concat([timing_loso, timing_dep]).to_csv(output / "e2e_timing.csv", index=False)
    e2e_deployment = e2e_summary(rec_dep, events_dep)
    (output / "assessment_output.schema.json").write_text(json.dumps(output_json_schema(), indent=2) + "\n", encoding="utf-8")
    golden = write_golden(output, final, detector, data_dir)
    limitations = [
        "No clinical or technique ground truth: OUTSIDE_REFERENCE_RANGE describes acoustic deviation relative to the "
        "PRISM reference corpus; it is not a normal/abnormal, anomaly, technique-quality or clinical classification.",
        f"Session dependence remains: per-session held-out exceedance at alpha 0.05 ranges "
        f"{per_session['rate'].min():.2f}-{per_session['rate'].max():.2f} (overdispersion parametric p "
        f"{dispersion['parametric_bootstrap_p']:.3f}); calibration holds marginally over sessions, not per session "
        "(held-out by recording session on the same PRISM corpus).",
        "Reference = 318 events from 18 inferred recording sittings of unknown subjects/devices; global, not personal.",
        "V2 representation is DRAFT_NOT_FROZEN (G4c); PRISM hardware audio is not validated (baseline_domain_validated "
        "false) and needs a hardware reference study.",
        "Same-corpus evaluation: the same 318 events informed the usability rule, feature selection, baseline strategy, "
        "V2 design, Stage 8 gate thresholds and Stage 9 conventions; the held-out-by-session check is an internal "
        "consistency/calibration check, not external validation, and there is no independent test set.",
        "Not validated for new users, devices, microphones or PRISM hardware.",
        "The detector is in-sample for about two-thirds of the reference recordings.",
    ]
    contract = build_contract(final, _sha256(reference_path), outcome, gate, representation_decision, limitations)
    (output / "assessment_contract_v1.json").write_text(json.dumps(contract, indent=2, default=_json_default) + "\n",
                                                        encoding="utf-8")
    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_calibration(per_session, k1, output / "calibration_by_session.png")
        plot_reference(final, heldout, output / "reference_distribution.png")
        plot_perturbation(flag_summary, output / "perturbation_assessment.png")

    hashes_after = {p: _sha256(Path(p)) for p in hashes_before}
    summary = {
        "stage": "Stage 9 - final event-level assessment (reference calibration; no clinical labels)",
        "preregistration_sha256": PREREGISTRATION_SHA256, "representation_decision": representation_decision,
        "gate_outcome": outcome, "contract_status": contract["status"], "gate": gate, "calibration": calibration,
        "stability": stability, "reliability_loso": reliability, "reported": reported,
        "lofo": lofo.to_dict("records"), "e2e_loso": e2e_loso, "e2e_deployment": e2e_deployment,
        "e2e_consistency": consistency, "deployment_reference": {"cut": final.cut, "band": list(final.band),
                                                                 "n_events": len(final.calibration_scores),
                                                                 "n_sessions": final.n_sessions,
                                                                 "categorical": final.categorical},
        "golden_cases": [c["case"] for c in golden["cases"]],
        "inputs_unchanged": hashes_before == hashes_after,
        "input_sha256": {_relative(Path(p)): h for p, h in hashes_before.items()},
        "runtime_seconds": {"loso_e2e_total": float(timing_loso["seconds"].sum()),
                            "deployment_e2e_total": float(timing_dep["seconds"].sum())},
        "provenance": {"git": _git_state()},
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n",
                                                  encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 9 final assessment: calibration, gate, end-to-end runs, contract")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="must be empty (only stage9_preregistration.json may be present); the run refuses otherwise")
    parser.add_argument("--reference-run", help="a previous complete run to compare byte for byte (K4)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_analysis(output_dir=args.output_dir, reference_run=args.reference_run, make_plots=not args.no_plots)
    print(json.dumps({k: result[k] for k in ("gate_outcome", "contract_status", "representation_decision",
                                             "e2e_consistency", "inputs_unchanged")}, indent=2, default=_json_default))
