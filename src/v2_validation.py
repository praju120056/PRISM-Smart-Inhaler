"""Stage 8: V2 representation validation, acceptance gate and inference contract.

See PRISM_RESEARCH_LOG.md, Entry 10.

Question: does the 4-feature V2 representation (spectral_centroid_mean,
spectral_flatness_mean, spectral_centroid_std, spectral_rolloff_std) behave
consistently enough across sessions and controlled acoustic perturbations to
be frozen temporarily as the PRISM MVP inference representation?

The acceptance gate was pre-registered in
results/v2_validation/acceptance_gate_preregistration.json (commit 55baeba)
before any combined-V2 result was computed; the rule constants below restate
it and a test checks that they match.  Protocol: Strategy C (Stage 5), i.e.
per-feature median / 1.4826*MAD fitted per leave-one-session-out fold on the
USABLE events of the other sessions only; rms_z is the score.  No threshold is
chosen and no event is labelled NORMAL or ANOMALY.  Excluded (Stage 1) status
is a descriptive population only, never anomaly ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config
from baseline_v1 import DEFAULT_SELECTION_JSON, fit_baseline, load_feature_selection, recording_sessions
from feature_analysis import DEFAULT_DATASET_CSV, MIN_GROUP_SIZE, exact_event_bounds, load_stage1_table, within_group_spearman
from inhale_dataset import _git_state, _json_default, _relative, _sha256
from natural_population_analysis import assign_populations, cliffs_delta, comparison_groups, heldout_scores
from post_event import DEFAULT_MODEL_PATH, OnnxEventClassifier
from prism_inference import (
    CONTRACT_VERSION,
    DETECTOR_WINDOW_FRAMES,
    FORBIDDEN_DERIVED_OUTPUTS,
    GROUPING,
    INPUT_DOMAINS,
    INPUT_ERRORS,
    INTERPRETATION,
    LEVEL_CHANNEL,
    MAD_SCALE,
    MIN_SAMPLES,
    NOT_SCOREABLE_REASONS,
    RECORDING_STATUSES,
    SAMPLE_RATE,
    USABILITY,
    V2_FEATURES,
    EVENT_STATUSES,
    analyze_recording,
    feature_schema,
    frozen_from_fit,
    load_baseline,
    output_json_schema,
    read_wav,
    write_baseline,
)
from representation_analysis import (
    FAMILY_ORDER,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    GRID,
    BASELINE_INK,
    SERIES,
    SURFACE,
    _style,
    build_representations,
    contribution_rows,
    excluded_diagnostics,
    feature_perturbation_table,
    fold_references,
    perturbation_measurements,
    perturbation_response,
    session_dependence,
)
from scoring_v1 import combined_scores


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "v2_validation"
PREREGISTRATION = DEFAULT_OUTPUT_DIR / "acceptance_gate_preregistration.json"
PREREGISTRATION_COMMIT = "55baeba"
STAGE7_DIR = Path(config.RESULTS_DIR) / "representation_analysis"
STAGE7_CANDIDATES = STAGE7_DIR / "candidate_features.csv"
STAGE7_COMPARISON = STAGE7_DIR / "representation_comparison.csv"
STAGE7_PERTURBED = STAGE7_DIR / "perturbed_features.csv"
BASELINE_ID = "prism-v2-global-2026-09-30"
SEED = 20261005

V2 = list(V2_FEATURES)
SHORT = {"spectral_centroid_mean": "centroid_mean", "spectral_flatness_mean": "flatness_mean",
         "spectral_centroid_std": "centroid_std", "spectral_rolloff_std": "rolloff_std"}
COMPARATORS = ("R0_current7", "R1_no_duration", "R3_no_flatness_std", "R5_no_mean_rms", "R6_relative_level")
ABLATIONS = tuple(f"V2_minus_{SHORT[f]}" for f in V2)
LEVEL_VARIANT = "V2_plus_mean_rms"          # Part E only: what adding the level channel back would do

# ── Pre-registered acceptance gate (restates acceptance_gate_preregistration.json) ──
G1B_MAX_GENERALIZATION_RATIO = 1.15
G2A_MAX_CENTER_SHIFT_SD = 0.5
G2B_SCALE_RATIO_RANGE = (2 / 3, 3 / 2)
G3A_MAX_CORRELATION_SHIFT = 0.2
G4A_MAX_P95_GAIN_DELTA = 0.1
G4B_MIN_MONOTONE_FRACTION = 0.95
G4C_MIN_SESSION_AGREEMENT = 0.9
G4C_MIN_RESPONSE = 0.1
G4C_LEVELS = (("noise", 10.0), ("tilt", -0.5), ("tilt", 0.9))
G4D_FRAGILE_DELTA, G4D_FRAGILE_MONOTONE = 1.0, 0.5
G5_MAX_ABS_RHO = 0.3
G6B_TIME_TOLERANCE_S, G6B_CONFIDENCE_TOLERANCE, G6B_FEATURE_RELATIVE_TOLERANCE = 1e-9, 1e-6, 1e-6
G6C_TOLERANCE = 1e-12
GATE_RULES = {
    "G1a": "epsilon2(V2) <= epsilon2(R0_current7), computed in the same run",
    "G1b": "<= 1.15",
    "G2a": "<= 0.5",
    "G2b": "within [2/3, 3/2] for every fold",
    "G2c": "> 0 and finite",
    "G3a": "<= 0.2",
    "G4a": "<= 0.1 at every level",
    "G4b": ">= 0.95 for each family",
    "G4c": ">= 0.9 in every applicable case",
    "G4d": "no feature has |median delta z| >= 1 together with a monotone fraction < 0.5",
    "G5a": "|rho| < 0.3",
    "G5b": "|rho| < 0.3",
    "G6a": "all CSV outputs, the frozen baseline JSON and the contract JSON are byte-identical",
    "G6b": ("identical number of events per recording; event bounds equal to the detector's exact bounds within 1e-9 s; "
            "detector confidence within 1e-6; V2 features and mean_rms within 1e-6 relative; identical usability "
            "decisions; recordings without events reported as NO_INHALATION_DETECTED"),
    "G6c": "max |difference| <= 1e-12",
    "G6d": "0 and 0",
}
GAIN_LEVELS = (("gain", 0.5), ("gain", 1 / np.sqrt(2)), ("gain", np.sqrt(2)), ("gain", 2.0),
               ("recording_gain", 0.5), ("recording_gain", 2.0))


def build_v2_representations(v1: Sequence[str]) -> dict[str, list[str]]:
    """Stage 7 comparators, V2, its leave-one-feature-out ablations and the Part E level variant."""
    stage7 = build_representations(v1)
    representations = {name: stage7[name] for name in COMPARATORS}
    representations["V2"] = list(V2)
    for feature, name in zip(V2, ABLATIONS):
        representations[name] = [f for f in V2 if f != feature]
    representations[LEVEL_VARIANT] = [*V2, LEVEL_CHANNEL]
    return representations


def rms(z: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return combined_scores(z[[f"z_{f}" for f in features]], list(features))["rms_z"].to_numpy()


# ── Part A: combined representation ─────────────────────────────────────────

def feature_scale_stability(fold_parameters: pd.DataFrame, pooled, features: Sequence[str]) -> pd.DataFrame:
    """Fold centre shift (pooled robust SD units) and fold / pooled scale ratio per feature."""
    rows = []
    for feature in features:
        folds = fold_parameters[fold_parameters["feature"] == feature]
        center, scale = pooled.parameters[feature].median, pooled.parameters[feature].scale
        shift = (folds["center"] - center).abs() / scale
        ratio = folds["scale"] / scale
        rows.append({"feature": feature, "pooled_center": center, "pooled_mad": pooled.parameters[feature].mad,
                     "pooled_scale": scale, "n_pooled": pooled.n_calibration, "n_folds": int(len(folds)),
                     "max_center_shift_sd": float(shift.max()), "fold_of_max_shift": folds.loc[shift.idxmax(), "fold"],
                     "min_scale_ratio": float(ratio.min()), "max_scale_ratio": float(ratio.max()),
                     "min_fold_mad": float((folds["scale"] / MAD_SCALE).min()),
                     "all_mad_positive_finite": bool(np.isfinite(folds["scale"]).all() and (folds["scale"] > 0).all()
                                                     and np.isfinite(scale) and scale > 0)})
    return pd.DataFrame(rows)


def correlation_stability(frame: pd.DataFrame, sessions: pd.Series, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of V2 feature pairs: pooled, within-session and per LOSO training set."""
    usable = frame["usable"].astype(bool).to_numpy()
    pooled = pd.DataFrame(stats.spearmanr(frame.loc[usable, list(features)]).statistic, index=features, columns=features)
    within = within_group_spearman(frame[usable].assign(session=sessions[usable].to_numpy()), list(features), "session")
    fold_rows = []
    for session in sorted(sessions[usable].unique()):
        training = frame.loc[usable & (sessions != session).to_numpy(), list(features)]
        matrix = stats.spearmanr(training).statistic
        for i, a in enumerate(features):
            for j, b in enumerate(features):
                if i < j:
                    fold_rows.append({"fold": session, "feature_a": a, "feature_b": b, "n_train": int(len(training)),
                                      "rho": float(matrix[i, j]), "shift_vs_pooled": float(matrix[i, j] - pooled.loc[a, b])})
    folds = pd.DataFrame(fold_rows)
    pair_rows = []
    for (a, b), block in folds.groupby(["feature_a", "feature_b"], sort=False):
        worst = block["shift_vs_pooled"].abs().idxmax()
        pair_rows.append({"feature_a": a, "feature_b": b, "pooled_rho": float(pooled.loc[a, b]),
                          "within_session_rho": float(within.loc[a, b]), "min_fold_rho": float(block["rho"].min()),
                          "max_fold_rho": float(block["rho"].max()),
                          "max_abs_shift_vs_pooled": float(block["shift_vs_pooled"].abs().max()),
                          "fold_of_max_shift": block.loc[worst, "fold"]})
    return pd.DataFrame(pair_rows), folds


def z_distributions(z: pd.DataFrame, usable: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = z.loc[usable, f"z_{feature}"].to_numpy()
        p = np.percentile(values, [5, 25, 50, 75, 95])
        rows.append({"feature": feature, "n": int(len(values)), "p05": p[0], "p25": p[1], "median": p[2], "p75": p[3],
                     "p95": p[4], "robust_sd": float(MAD_SCALE * np.median(np.abs(values - np.median(values)))),
                     "max_abs_z": float(np.abs(values).max())})
    return pd.DataFrame(rows)


def representation_rows(frame: pd.DataFrame, z: pd.DataFrame, sessions: pd.Series, groups: Mapping[str, np.ndarray],
                        references: Mapping, representations: Mapping[str, Sequence[str]]):
    """Comparison table, per-session medians, contributions and excluded diagnostics (descriptive)."""
    usable = groups["usable"]
    everything = np.ones(len(frame), bool)
    comparison, session_rows, contributions, excluded_rows, scores = [], [], [], [], {}
    for name, features in representations.items():
        values = rms(z, features)
        scores[name] = values
        reference = np.array([references[name][s] for s in sessions])
        p = np.percentile(values[usable], [5, 25, 50, 75, 95])
        diagnostics = {row["group"]: row for row in excluded_diagnostics(values, sessions, groups, everything,
                                                                            names=("excluded_all", "only_close_neighbor"),
                                                                            seed=SEED)}
        comparison.append({"representation": name, "features": ";".join(features), "d": len(features),
                           "usable_p05": p[0], "usable_p25": p[1], "usable_median_rms_z": p[2], "usable_p75": p[3],
                           "usable_p95": p[4],
                           "generalization_ratio": float(np.median(values[usable]) / np.median(reference[usable])),
                           **session_dependence(values, sessions, usable),
                           "diagnostic_excluded_all_delta": diagnostics["excluded_all"]["cliffs_delta"],
                           "diagnostic_close_only_delta": diagnostics["only_close_neighbor"]["cliffs_delta"]})
        for row in diagnostics.values():
            excluded_rows.append({"representation": name, **row})
        for row in contribution_rows(z, features, {"usable": usable}):
            contributions.append({"representation": name, **row})
        labels, counts = np.unique(sessions[usable], return_counts=True)
        for session in labels[counts >= MIN_GROUP_SIZE]:
            inside = usable & (sessions == session).to_numpy()
            session_rows.append({"representation": name, "session": session, "n": int(inside.sum()),
                                 "median_rms_z": float(np.median(values[inside])),
                                 "cliffs_delta_vs_other_usable": cliffs_delta(values[inside], values[usable & ~inside])})
    return pd.DataFrame(comparison), pd.DataFrame(session_rows), pd.DataFrame(contributions), pd.DataFrame(excluded_rows), scores


def session_feature_medians(z: pd.DataFrame, sessions: pd.Series, usable: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    labels, counts = np.unique(sessions[usable], return_counts=True)
    for session, count in zip(labels, counts):
        inside = usable & (sessions == session).to_numpy()
        rows.append({"session": session, "n": int(count), "tested": bool(count >= MIN_GROUP_SIZE),
                     **{f"median_z_{f}": float(np.median(z.loc[inside, f"z_{f}"])) for f in features}})
    return pd.DataFrame(rows)


def confound_table(scores: Mapping[str, np.ndarray], frame: pd.DataFrame, sessions: pd.Series,
                   usable: np.ndarray) -> pd.DataFrame:
    """Spearman of each representation's held-out rms_z with recording/segmentation quantities (usable events)."""
    quantities = ("background_rms", "duration_s", "confidence", "mean_rms")
    rows = []
    for name, values in scores.items():
        data = frame.loc[usable, list(quantities)].assign(score=values[usable], session=sessions[usable].to_numpy())
        within = within_group_spearman(data, ["score", *quantities], "session")
        row = {"representation": name}
        for quantity in quantities:
            row[f"pooled_rho_{quantity}"] = float(stats.spearmanr(data["score"], data[quantity]).statistic)
            row[f"within_session_rho_{quantity}"] = float(within.loc["score", quantity])
        rows.append(row)
    return pd.DataFrame(rows)


# ── Part C: controlled perturbations ────────────────────────────────────────

def direction_consistency(perturbed: pd.DataFrame, quantities: Mapping[str, Callable[[pd.DataFrame], np.ndarray]],
                          min_session_events: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Per quantity and perturbation level: median change, magnitude, and sign agreement across events and sessions."""
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    changed = perturbed[perturbed["transform"] != "identity"]
    rows = []
    for name, measure in quantities.items():
        base = pd.Series(measure(identity), index=identity.index)
        for (transform, magnitude), block in changed.groupby(["transform", "magnitude"], sort=False):
            block = block.set_index("row")
            delta = measure(block) - base.loc[block.index].to_numpy()
            median = float(np.median(delta))
            sign = np.sign(median)
            session_values = block["session"].to_numpy()
            labels, counts = np.unique(session_values, return_counts=True)
            tested = labels[counts >= min_session_events]
            session_medians = np.array([np.median(delta[session_values == s]) for s in tested])
            rows.append({"quantity": name, "transform": transform, "magnitude": float(magnitude), "n_events": int(len(delta)),
                         "median_delta": median, "p25_delta": float(np.percentile(delta, 25)),
                         "p75_delta": float(np.percentile(delta, 75)),
                         "p95_abs_delta": float(np.percentile(np.abs(delta), 95)), "max_abs_delta": float(np.abs(delta).max()),
                         "frac_events_same_sign": float(np.mean(np.sign(delta) == sign)) if sign else float("nan"),
                         "sessions_tested": int(len(tested)),
                         "frac_sessions_same_sign": float(np.mean(np.sign(session_medians) == sign)) if sign and len(tested)
                         else float("nan"),
                         "responds": bool(abs(median) >= G4C_MIN_RESPONSE)})
    return pd.DataFrame(rows)


def perturbation_quantities(representations: Mapping[str, Sequence[str]], features: Sequence[str]) -> dict:
    quantities = {f"z_{f}": (lambda block, f=f: block[f"z_{f}"].to_numpy(dtype=float)) for f in features}
    for name, feature_list in representations.items():
        quantities[f"rms_z[{name}]"] = lambda block, fl=tuple(feature_list): rms(block, fl)
    return quantities


def stage7_perturbation_consistency(measured: pd.DataFrame, features: Sequence[str]) -> float | None:
    """Max relative difference of re-measured features vs the Stage 7 perturbed table (stored at 6 significant digits)."""
    if not STAGE7_PERTURBED.exists():
        return None
    stage7 = pd.read_csv(STAGE7_PERTURBED)
    joined = measured.merge(stage7[["row", "transform", "magnitude", *features]], on=["row", "transform"], suffixes=("", "_s7"))
    joined = joined[np.isclose(joined["magnitude"], joined["magnitude_s7"])]
    return float(max((np.abs(joined[f] - joined[f"{f}_s7"]) / np.abs(joined[f"{f}_s7"]).clip(lower=1e-12)).max()
                     for f in features))


# ── Part E: level channel ───────────────────────────────────────────────────

def normal_scores(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values)
    return stats.norm.ppf((ranks - 0.5) / len(values))


def rank_r2(target: np.ndarray, predictors: np.ndarray) -> float:
    """R^2 of the target's normal scores regressed on the predictors' normal scores (Stage 2 'rank R^2')."""
    y = normal_scores(target)
    x = np.column_stack([np.ones(len(y))] + [normal_scores(c) for c in predictors.T])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    return float(1 - residual.var() / y.var())


def level_channel_evaluation(frame: pd.DataFrame, z: pd.DataFrame, sessions: pd.Series, usable: np.ndarray,
                             feature_perturbation: pd.DataFrame, direction: pd.DataFrame,
                             comparison: pd.DataFrame, confounds: pd.DataFrame) -> dict:
    data = frame[usable].assign(session=sessions[usable].to_numpy())
    within = within_group_spearman(data, [LEVEL_CHANNEL, *V2], "session")
    level_rows = feature_perturbation[feature_perturbation["feature"] == LEVEL_CHANNEL].set_index("transform")
    level_direction = direction[direction["quantity"] == f"z_{LEVEL_CHANNEL}"]

    def gain_delta(transform, magnitude):
        row = level_direction[(level_direction["transform"] == transform) & np.isclose(level_direction["magnitude"], magnitude)]
        return float(row["median_delta"].iloc[0])

    by_rep = comparison.set_index("representation")
    confound = confounds.set_index("representation")
    variants = {}
    for name in ("V2", LEVEL_VARIANT):
        gains = direction[(direction["quantity"] == f"rms_z[{name}]") & direction["transform"].isin(["gain", "recording_gain"])]
        variants[name] = {"session_epsilon_squared": float(by_rep.loc[name, "session_epsilon_squared"]),
                          "generalization_ratio": float(by_rep.loc[name, "generalization_ratio"]),
                          "max_p95_abs_delta_rms_z_under_gain": float(gains["p95_abs_delta"].max()),
                          "within_session_rho_background_rms": float(confound.loc[name, "within_session_rho_background_rms"])}
    return {
        "feature": LEVEL_CHANNEL,
        "held_out_z_session_epsilon_squared": session_dependence(z[f"z_{LEVEL_CHANNEL}"].to_numpy(), sessions, usable)[
            "session_epsilon_squared"],
        "median_delta_z_event_gain_x2": gain_delta("gain", 2.0),
        "median_delta_z_event_gain_x0.5": gain_delta("gain", 0.5),
        "median_delta_z_recording_gain_x2": gain_delta("recording_gain", 2.0),
        "noise_monotone_fraction": float(level_rows.loc["noise", "monotone_fraction"]),
        "pooled_spearman_with_v2": {f: float(stats.spearmanr(data[LEVEL_CHANNEL], data[f]).statistic) for f in V2},
        "within_session_spearman_with_v2": {f: float(within.loc[LEVEL_CHANNEL, f]) for f in V2},
        "rank_r2_from_v2_features": rank_r2(data[LEVEL_CHANNEL].to_numpy(), data[V2].to_numpy()),
        "pooled_spearman_with_background_rms": float(stats.spearmanr(data[LEVEL_CHANNEL], data["background_rms"]).statistic),
        "score_with_and_without_level": variants,
    }


# ── G6: reproducibility of the inference contract ───────────────────────────

def reproduce_stage1(table: pd.DataFrame, data_dir: str | Path, baseline, detector, pooled_scores: pd.Series):
    """Run the reference inference implementation on every WAV and compare with the Stage 1 event table."""
    rows, recordings, outputs = [], [], {}
    for path in sorted(Path(data_dir).glob("*.wav")):
        waveform, sample_rate = read_wav(path)
        output = analyze_recording(waveform, sample_rate, detector=detector, baseline=baseline,
                                   input_domain="reference_dataset", recording_id=path.name)
        outputs[path.name] = output
        expected = table[table["recording_file"] == path.name].sort_values("event_id")
        recordings.append({"recording_file": path.name, "recording_status": output["recording_status"],
                           "n_events_contract": output["n_events"], "n_events_stage1": int(len(expected)),
                           "n_scored": output["n_scored"]})
        for event, (index, row) in zip(output["events"], expected.iterrows()):
            start, end = exact_event_bounds(row["start_s"], row["end_s"], row["recording_duration_s"])
            reasons_stage1 = set(filter(None, str(row["exclusion_reasons"]).split(";"))) if pd.notna(row["exclusion_reasons"]) else set()
            record = {"recording_file": path.name, "event_id": event["event_id"],
                      "abs_diff_start": abs(event["start_time"] - start), "abs_diff_end": abs(event["end_time"] - end),
                      "abs_diff_duration": abs(event["duration_s"] - row["duration_s"]),
                      "abs_diff_confidence": abs(event["detector_confidence"] - row["confidence"]),
                      "usable_stage1": bool(row["usable"]), "scored_contract": event["status"] == "SCORE_ONLY",
                      "reasons_match": reasons_stage1 == set(event["not_scoreable_reasons"])}
            for feature in V2:
                value = event["feature_values"][feature] if event["feature_values"] else np.nan
                record[f"rel_diff_{feature}"] = abs(value - row[feature]) / max(abs(row[feature]), 1e-12)
            record[f"rel_diff_{LEVEL_CHANNEL}"] = abs(event[LEVEL_CHANNEL] - row[LEVEL_CHANNEL]) / max(abs(row[LEVEL_CHANNEL]), 1e-12)
            record["abs_diff_score_vs_in_memory"] = (abs(event["anomaly_score"] - pooled_scores.loc[index])
                                                     if event["anomaly_score"] is not None else np.nan)
            rows.append(record)
    events, recordings = pd.DataFrame(rows), pd.DataFrame(recordings)
    rel_columns = [c for c in events.columns if c.startswith("rel_diff_")]
    summary = {
        "recordings": int(len(recordings)),
        "events_stage1": int(len(table)), "events_contract": int(recordings["n_events_contract"].sum()),
        "recordings_with_count_mismatch": int((recordings["n_events_contract"] != recordings["n_events_stage1"]).sum()),
        "no_inhalation_detected": int((recordings["recording_status"] == "NO_INHALATION_DETECTED").sum()),
        "recordings_without_stage1_events": int((recordings["n_events_stage1"] == 0).sum()),
        "no_inhalation_matches_stage1_zero_events": bool(((recordings["recording_status"] == "NO_INHALATION_DETECTED")
                                                          == (recordings["n_events_stage1"] == 0)).all()),
        "max_abs_diff_bounds_s": float(events[["abs_diff_start", "abs_diff_end"]].to_numpy().max()),
        "max_abs_diff_duration_s": float(events["abs_diff_duration"].max()),
        "max_abs_diff_confidence": float(events["abs_diff_confidence"].max()),
        "max_rel_diff_features": {c.replace("rel_diff_", ""): float(events[c].max()) for c in rel_columns},
        "usability_mismatches": int((events["usable_stage1"] != events["scored_contract"]).sum()),
        "reason_mismatches": int((~events["reasons_match"]).sum()),
        "scored_events": int(events["scored_contract"].sum()),
        "max_abs_diff_contract_score_vs_in_memory": float(events["abs_diff_score_vs_in_memory"].max()),
        "all_outputs_validated": True,     # analyze_recording validates every output or raises
    }
    summary["passed"] = bool(
        summary["recordings_with_count_mismatch"] == 0 and summary["events_contract"] == summary["events_stage1"]
        and summary["no_inhalation_matches_stage1_zero_events"]
        and summary["max_abs_diff_bounds_s"] <= G6B_TIME_TOLERANCE_S
        and summary["max_abs_diff_duration_s"] <= G6B_TIME_TOLERANCE_S
        and summary["max_abs_diff_confidence"] <= G6B_CONFIDENCE_TOLERANCE
        and max(summary["max_rel_diff_features"].values()) <= G6B_FEATURE_RELATIVE_TOLERANCE
        and summary["usability_mismatches"] == 0 and summary["reason_mismatches"] == 0)
    return events, recordings, summary, outputs


def compare_runs(current: Path, reference: Path) -> dict:
    """G6a: byte-identity of two complete runs (gate-dependent fields of the contract are excluded)."""
    compared, mismatches = [], []
    paths = sorted(p.relative_to(current) for p in current.rglob("*") if p.is_file()
                   and p.stem != "acceptance_gate_results"      # records the G6a verdict itself
                   and (p.suffix in (".csv", ".wav") or p.parent.name == "expected"
                        or p.name in ("v2_baseline.json", "inference_output.schema.json", "v2_feature_schema.json",
                                      "golden_manifest.json")))
    for relative in paths:
        compared.append(relative.as_posix())
        other = reference / relative
        if not other.exists() or hashlib.sha256((current / relative).read_bytes()).digest() != hashlib.sha256(other.read_bytes()).digest():
            mismatches.append(relative.as_posix())
    contract_a = json.loads((current / "inference_contract_v2.json").read_text(encoding="utf-8"))
    contract_b = json.loads((reference / "inference_contract_v2.json").read_text(encoding="utf-8"))
    for contract in (contract_a, contract_b):
        for key in ("status", "status_meaning", "acceptance_gate"):
            contract.pop(key, None)
    compared.append("inference_contract_v2.json (without status / acceptance_gate)")
    if contract_a != contract_b:
        mismatches.append("inference_contract_v2.json")
    return {"reference_run": str(reference), "files_compared": len(compared), "mismatches": mismatches,
            "passed": not mismatches}


# ── Golden vectors for ports (engineering conformance, not validation) ──────

def golden_cases(outputs: Mapping[str, dict]) -> dict[str, str]:
    """First recording (sorted by name) of each output situation the next stage must reproduce."""
    def first(predicate):
        return next((name for name, output in outputs.items() if predicate(output)), None)

    def reasons(output):
        return {r for e in output["events"] for r in e["not_scoreable_reasons"]}

    chosen = {
        "single_scored_event": first(lambda o: o["n_events"] == 1 and o["n_scored"] == 1),
        "no_inhalation_detected": first(lambda o: o["recording_status"] == "NO_INHALATION_DETECTED"),
        "close_neighbors_not_scoreable": first(lambda o: "close_neighbor" in reasons(o)),
        "short_event_not_scoreable": first(lambda o: "short_duration" in reasons(o) and o["n_scored"] >= 1),
        "boundary_event_not_scoreable": first(lambda o: "recording_boundary" in reasons(o)),
    }
    return {case: name for case, name in chosen.items() if name is not None}


def write_golden(output_dir: Path, outputs: Mapping[str, dict], data_dir: Path, baseline, detector) -> dict:
    golden = output_dir / "golden"
    (golden / "expected").mkdir(parents=True, exist_ok=True)
    (golden / "inputs").mkdir(parents=True, exist_ok=True)
    cases = []
    for case, recording in golden_cases(outputs).items():
        (golden / "expected" / f"{case}.json").write_text(json.dumps(outputs[recording], indent=2, allow_nan=False) + "\n",
                                                            encoding="utf-8")
        cases.append({"case": case, "input": {"type": "dataset_wav", "file": recording,
                                              "sha256": _sha256(data_dir / recording)},
                      "input_domain": "reference_dataset", "expected": f"expected/{case}.json"})
    import soundfile as sf

    rng = np.random.default_rng(SEED)
    synthetic = {"digital_silence_2s": np.zeros(2 * SAMPLE_RATE),
                 "white_noise_3s_minus20dBFS": 0.1 * rng.standard_normal(3 * SAMPLE_RATE)}
    for case, signal in synthetic.items():
        path = golden / "inputs" / f"{case}.wav"
        pcm = np.clip(np.round(signal * 32768.0), -32768, 32767).astype(np.int16)
        sf.write(str(path), pcm, SAMPLE_RATE, subtype="PCM_16")
        waveform, rate = read_wav(path)
        output = analyze_recording(waveform, rate, detector=detector, baseline=baseline, recording_id=path.name)
        (golden / "expected" / f"{case}.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        cases.append({"case": case, "input": {"type": "synthetic_wav", "path": f"inputs/{case}.wav", "sha256": _sha256(path)},
                      "input_domain": "unknown", "expected": f"expected/{case}.json"})
    inline = {"input_error_sample_rate_16k": (np.zeros(16000, np.float32), 16000),
              "input_error_too_short": (np.zeros(MIN_SAMPLES - 1, np.float32), SAMPLE_RATE)}
    for case, (signal, rate) in inline.items():
        output = analyze_recording(signal, rate, detector=detector, baseline=baseline)
        (golden / "expected" / f"{case}.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        cases.append({"case": case, "input": {"type": "inline", "description": f"{len(signal)} zero samples at {rate} Hz"},
                      "input_domain": "unknown", "expected": f"expected/{case}.json"})
    manifest = {
        "contract_version": CONTRACT_VERSION, "baseline_id": baseline.baseline_id,
        "purpose": "Conformance vectors for re-implementations of the contract (engineering, not validation).",
        "dataset_inputs": "Dataset WAVs are not in git (data/ is ignored); they are identified by file name and SHA-256.",
        "tolerances": CONFORMANCE_TOLERANCES,
        "cases": cases,
    }
    (golden / "golden_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


CONFORMANCE_TOLERANCES = {
    "recording_status_event_count_status_reasons": "exact",
    "event_window_indices_and_times": "exact (times are multiples of 0.016 s; compare within 1e-9 s)",
    "detector_confidence_abs": 1e-4,
    "feature_values_relative": 1e-4,
    "mean_rms_relative": 1e-4,
    "feature_z_scores_abs": 0.01,
    "anomaly_score_abs": 0.01,
    "rationale": "0.01 z is one tenth of the 0.1 z 'negligible change' convention of pre-registered criterion G4a.",
}


# ── Gate ────────────────────────────────────────────────────────────────────

def evaluate_gate(evidence: Mapping) -> list[dict]:
    """Apply the pre-registered rules; each row records the measured value and the verdict."""
    comparison = evidence["comparison"].set_index("representation")
    scale = evidence["feature_scale"]
    direction = evidence["direction"]
    response = evidence["response"]
    feature_perturbation = evidence["feature_perturbation"]
    confound = evidence["confounds"].set_index("representation").loc["V2"]
    rows = []

    def add(criterion, value, passed, detail=""):
        rows.append({"id": criterion, "rule": GATE_RULES[criterion], "value": value,
                     "passed": None if passed is None else bool(passed), "detail": detail})

    eps_v2, eps_r0 = comparison.loc["V2", "session_epsilon_squared"], comparison.loc["R0_current7", "session_epsilon_squared"]
    add("G1a", float(eps_v2), eps_v2 <= eps_r0, f"epsilon2 V2 {eps_v2:.4f} vs R0 {eps_r0:.4f}")
    ratio = comparison.loc["V2", "generalization_ratio"]
    add("G1b", float(ratio), ratio <= G1B_MAX_GENERALIZATION_RATIO)
    shift = scale["max_center_shift_sd"].max()
    add("G2a", float(shift), shift <= G2A_MAX_CENTER_SHIFT_SD,
        "; ".join(f"{r.feature} {r.max_center_shift_sd:.3f} ({r.fold_of_max_shift})" for r in scale.itertuples()))
    low, high = scale["min_scale_ratio"].min(), scale["max_scale_ratio"].max()
    add("G2b", [float(low), float(high)], G2B_SCALE_RATIO_RANGE[0] <= low and high <= G2B_SCALE_RATIO_RANGE[1])
    add("G2c", bool(scale["all_mad_positive_finite"].all()), scale["all_mad_positive_finite"].all(),
        f"min fold MAD {scale['min_fold_mad'].min():.3g}")
    correlation = evidence["correlation"]
    worst = correlation.loc[correlation["max_abs_shift_vs_pooled"].idxmax()]
    add("G3a", float(worst["max_abs_shift_vs_pooled"]), worst["max_abs_shift_vs_pooled"] <= G3A_MAX_CORRELATION_SHIFT,
        f"{worst['feature_a']} / {worst['feature_b']}, fold {worst['fold_of_max_shift']}")
    gains = direction[(direction["quantity"] == "rms_z[V2]") & direction["transform"].isin(["gain", "recording_gain"])]
    add("G4a", float(gains["p95_abs_delta"].max()), gains["p95_abs_delta"].max() <= G4A_MAX_P95_GAIN_DELTA,
        "; ".join(f"{r.transform} x{r.magnitude:.3g}: {r.p95_abs_delta:.2e}" for r in gains.itertuples()))
    v2_response = response[response["representation"] == "V2"]
    monotone = {family: float(v2_response.loc[v2_response["transform"] == family, "family_distance_monotone_fraction"].min())
                for family in ("noise", "tilt")}
    add("G4b", monotone, min(monotone.values()) >= G4B_MIN_MONOTONE_FRACTION)
    cases, failures = [], []
    for quantity in [*(f"z_{f}" for f in V2), "rms_z[V2]"]:
        for transform, magnitude in G4C_LEVELS:
            row = direction[(direction["quantity"] == quantity) & (direction["transform"] == transform)
                            & np.isclose(direction["magnitude"], magnitude)].iloc[0]
            if row["responds"]:
                cases.append(f"{quantity} {transform} {magnitude:g}: {row['frac_sessions_same_sign']:.3f}")
                if not row["frac_sessions_same_sign"] >= G4C_MIN_SESSION_AGREEMENT:
                    failures.append(cases[-1])
    add("G4c", {"applicable_cases": len(cases), "failing_cases": len(failures)}, not failures,
        ("FAILING: " + "; ".join(failures) + " | ") * bool(failures) + "all: " + "; ".join(cases))
    noise = feature_perturbation[(feature_perturbation["transform"] == "noise")
                                 & feature_perturbation["feature"].isin(V2)].set_index("feature")
    fragile = [f for f in V2 if abs(noise.loc[f, "median_delta_z@30"]) >= G4D_FRAGILE_DELTA
               and noise.loc[f, "monotone_fraction"] < G4D_FRAGILE_MONOTONE]
    add("G4d", fragile, not fragile, "; ".join(f"{f}: {noise.loc[f, 'median_delta_z@30']:.2f} z, monotone "
                                               f"{noise.loc[f, 'monotone_fraction']:.2f}" for f in V2))
    rho = confound["pooled_rho_background_rms"]
    add("G5a", float(rho), abs(rho) < G5_MAX_ABS_RHO)
    rho = confound["within_session_rho_background_rms"]
    add("G5b", float(rho), abs(rho) < G5_MAX_ABS_RHO)
    rerun = evidence.get("rerun")
    add("G6a", None if rerun is None else rerun["mismatches"], None if rerun is None else rerun["passed"],
        "not evaluated: no reference run given" if rerun is None else f"{rerun['files_compared']} files compared")
    reproduction = evidence["reproduction"]
    add("G6b", {k: reproduction[k] for k in ("recordings", "events_contract", "recordings_with_count_mismatch",
                                             "max_abs_diff_bounds_s", "max_abs_diff_confidence", "usability_mismatches",
                                             "reason_mismatches", "no_inhalation_detected")}
        | {"max_rel_diff_features": max(reproduction["max_rel_diff_features"].values())}, reproduction["passed"])
    add("G6c", evidence["roundtrip_max_abs_diff"], evidence["roundtrip_max_abs_diff"] <= G6C_TOLERANCE)
    leakage = evidence["leakage"]
    add("G6d", leakage, leakage["held_out_session_events_in_own_fit"] == 0 and leakage["excluded_events_in_any_fit"] == 0)
    return rows


def gate_outcome(rows: Sequence[dict]) -> str:
    if any(r["passed"] is False for r in rows):
        return "FAIL"
    if any(r["passed"] is None for r in rows):
        return "INCOMPLETE"
    return "PASS"


CONTRACT_STATUS = {"PASS": "FROZEN_FOR_MVP_ENGINEERING", "FAIL": "DRAFT_NOT_FROZEN", "INCOMPLETE": "INCOMPLETE_NOT_FROZEN"}
STATUS_MEANING = {
    "FROZEN_FOR_MVP_ENGINEERING": (
        "The representation and interface are specified well enough for engineering integration while scientific "
        "validation continues. This is NOT a validated anomaly detector; there is no threshold and no NORMAL/ANOMALY output."),
    "DRAFT_NOT_FROZEN": (
        "The pre-registered acceptance gate failed (see acceptance_gate.failing_criteria). The interface (input, detector, "
        "event grouping, scoreability, output schema and recording states) is fully specified and reproduces the reference "
        "pipeline, so it can be implemented. The anomaly representation (V2 features and baseline) is NOT frozen: "
        "anomaly_score may change, must be treated as experimental and must not be shown to users as a health signal. "
        "Accepting V2 despite the failed criterion, or replacing it, is the project owner's decision."),
    "INCOMPLETE_NOT_FROZEN": "At least one pre-registered criterion was not evaluated; nothing is frozen.",
}


# ── Inference contract ──────────────────────────────────────────────────────

def build_contract(baseline, baseline_sha256: str, detector_sha256: str, outcome: str, gate_rows: Sequence[dict],
                   limitations: Sequence[str]) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "status": CONTRACT_STATUS[outcome],
        "status_meaning": STATUS_MEANING[CONTRACT_STATUS[outcome]],
        "acceptance_gate": {"outcome": outcome, "preregistration": "results/v2_validation/acceptance_gate_preregistration.json",
                            "preregistration_commit": PREREGISTRATION_COMMIT,
                            "results": "results/v2_validation/acceptance_gate_results.json",
                            "failing_criteria": [r["id"] for r in gate_rows if r["passed"] is False]},
        "pipeline": ["audio input (8 kHz mono)", "ONNX event detector (window classification)", "Inhale event grouping",
                     "Stage 1 usability rule v1 (scoreability)", "V2 feature extraction + level channel",
                     "frozen global baseline (loaded, never fitted at inference)", "robust z per feature",
                     "anomaly_score = sqrt(mean z^2) (SCORE_ONLY)"],
        "input": {
            "sample_rate_hz": SAMPLE_RATE, "resampling": "none; any other rate is INPUT_ERROR unsupported_sample_rate",
            "waveform": "float32 in [-1, 1]; from 16-bit little-endian PCM as sample / 32768 (see prism_inference.decode_pcm16le)",
            "full_scale_mapping": "a 24-bit microphone sample must reach 16-bit PCM as int16 = s24 >> 8 (equivalently "
                                  "float = s24 / 2^23); any other digital gain changes the level channel mean_rms",
            "channels": "mono expected; 2-D arrays are averaged over channels (existing post_event convention)",
            "preprocessing": "none: no normalization, filtering, gain control, trimming or padding before the detector",
            "minimum_length_samples": MIN_SAMPLES,
            "input_errors": list(INPUT_ERRORS),
            "input_domains": {"values": list(INPUT_DOMAINS),
                              "meaning": "baseline_domain_validated is true only for 'reference_dataset'; PRISM hardware "
                                         "audio has not been validated against this baseline"},
        },
        "event_detector": {
            "model_file": "results/inhaler_cnn.onnx", "model_sha256": detector_sha256,
            "runtime": "ONNX Runtime, CPUExecutionProvider in the reference implementation",
            "frame_features": {"per_frame": "124 = 40 MFCC | 40 delta | 40 delta-delta | centroid/4000 | flatness | "
                                            "rolloff(0.85)/4000 | zero-crossing rate",
                               "stft": "n_fft 256, hop 64, hann, center=True, pad_mode constant; |STFT| for spectral features",
                               "mfcc": "librosa.feature.mfcc: 128 mel bands (Slaney), fmin 50, fmax 4000, power 2, "
                                       "power_to_db(ref 1, amin 1e-10, top_db 80), DCT-II ortho, no lifter",
                               "deltas": "librosa.feature.delta width 9, mode 'interp' (orders 1 and 2)",
                               "zcr": "librosa.feature.zero_crossing_rate frame 2048, hop 64, center=True",
                               "normalization": "none (raw features)",
                               "reference": "src/librosa_extractor.py::extract_features_from_audio"},
            "input_tensor": {"name": "features", "dtype": "float32", "shape": ["N", DETECTOR_WINDOW_FRAMES, 124],
                             "windows": f"{DETECTOR_WINDOW_FRAMES} consecutive frames (0.2 s), stride {config.WINDOW_STRIDE} frames (0.016 s); "
                                        "recordings shorter than one window produce no windows"},
            "output_tensor": {"dtype": "float32", "shape": ["N", 4], "meaning": "raw logits",
                              "class_order": list(config.LABEL_NAMES), "probabilities": "softmax over the 4 logits"},
            "window_timing": "window i: start = i * 0.016 s, end = min(start + 0.2 s, recording duration); label = argmax",
            "validation_note": "trained on about two-thirds of the reference recordings (Entry 3); agreement with annotations is mostly in-sample",
        },
        "event_grouping": {
            "config": {"target_label": GROUPING.target_label, "smoothing_window": GROUPING.smoothing_window,
                       "max_gap_s": GROUPING.max_gap_s, "min_event_duration_s": GROUPING.min_event_duration_s,
                       "min_confidence": GROUPING.min_confidence},
            "rule": "Inhale-labelled windows in time order; a window joins the current event while its start <= the latest "
                    "end of the event's windows (overlapping 0.2 s windows bridge up to 12 strides); otherwise a new event starts",
            "boundaries": "start = first window start; end = max window end; duration_s = end - start",
            "confidence": "detector_confidence = mean P(Inhale) over the event's Inhale windows; detector_max_confidence = max",
            "event_id": "0-based chronological index within the recording",
            "reference": "src/post_event.py::generate_window_predictions, group_inhale_events",
        },
        "scoreability": {
            "rule": "Stage 1 usability rule v1; only scoreable events receive z-scores and anomaly_score",
            "not_scoreable_reasons": {
                "nonfinite_feature": "any V2 feature or mean_rms is non-finite, or extraction failed",
                "short_duration": f"round(duration_s, 6) < {USABILITY.min_duration_s} s",
                "close_neighbor": f"another event in the same recording within a gap < {USABILITY.min_neighbor_gap_s} s "
                                  "(gap = next.start - this.end, rounded to 6 decimals)",
                "recording_boundary": f"start_time <= {USABILITY.boundary_tolerance_s} s or end_time >= recording duration - "
                                      f"{USABILITY.boundary_tolerance_s} s",
            },
            "rationale": "the baseline was fitted only on events passing this rule; Stages 6-7 showed scores of other "
                         "events are dominated by segmentation artifacts",
            "deviation_from_stage1": "Stage 1 required all 13 descriptive features to be finite; inference computes only "
                                     "the 5 contract features. No reference-dataset event differs (verified by G6b).",
        },
        "feature_extraction": {"schema_file": "results/v2_validation/v2_feature_schema.json", **feature_schema()},
        "baseline": {
            "file": "results/v2_validation/v2_baseline.json", "sha256": baseline_sha256, "baseline_id": baseline.baseline_id,
            "type": "global (not personalized) robust baseline: per-feature median and 1.4826 * MAD",
            "fitted_on": f"all {baseline.n_events} usable events of {baseline.n_sessions} recording sessions of the reference dataset",
            "parameters": baseline.to_dict()["parameters"],
            "loading": "load once and verify contract_version, baseline_id, feature order, finiteness, MAD > 0 and "
                       "scale = 1.4826 * MAD (prism_inference.load_baseline); refuse to score otherwise",
            "fitting_at_inference": "never; the baseline is not updated from new events (any update is a new baseline_id "
                                    "requiring its own validation)",
            "evaluation": "leave-one-session-out (Strategy C) estimates its behaviour on unseen sessions (Entry 10)",
        },
        "scoring": {"z": "z_j = (x_j - center_j) / scale_j in float64, j in V2 order",
                    "anomaly_score": "sqrt(mean_j z_j^2) over the 4 V2 features (rms_z)",
                    "direction": "higher = further from the baseline; 0 = at the baseline median in every feature",
                    "threshold": None, "classification": None},
        "level_channel": {"name": LEVEL_CHANNEL, "in_anomaly_score": False, "z_scored": False,
                          "meaning": "uncalibrated event loudness (RMS, full scale 1.0); depends on microphone "
                                     "sensitivity, gain, distance and enclosure; reported for diagnostics only; "
                                     "not an anomaly signal; no threshold"},
        "output": {
            "schema_file": "results/v2_validation/inference_output.schema.json",
            "recording_statuses": {"EVENTS_DETECTED": "at least one Inhale event detected (each event has its own status)",
                                   "NO_INHALATION_DETECTED": "the detector found no Inhale event; this is a recording-level "
                                                             "state, never an anomalous inhalation event",
                                   "INPUT_ERROR": "input violates the contract; see error"},
            "event_statuses": {"SCORE_ONLY": "scored against the frozen baseline; no NORMAL/ANOMALY decision",
                               "NOT_SCOREABLE": "not scored; see not_scoreable_reasons"},
            "event_fields": ["event_id", "start_time", "end_time", "duration_s", "detector_confidence",
                             "detector_max_confidence", "window_count", "status", "not_scoreable_reasons",
                             "anomaly_score", "feature_values", "feature_z_scores", "mean_rms"],
            "event_timestamp": "absolute event time = input.recorded_at + start_time (recorded_at is caller-supplied)",
            "interpretation": INTERPRETATION,
            "forbidden_derived_outputs": list(FORBIDDEN_DERIVED_OUTPUTS),
            "status_enums": {"recording": list(RECORDING_STATUSES), "event": list(EVENT_STATUSES),
                             "not_scoreable_reasons": list(NOT_SCOREABLE_REASONS)},
        },
        "conformance": {"golden_manifest": "results/v2_validation/golden/golden_manifest.json",
                        "tolerances": CONFORMANCE_TOLERANCES,
                        "reference_implementation": "src/prism_inference.py (python src/prism_inference.py <wav>)"},
        "known_limitations": list(limitations),
    }


# ── Figures ─────────────────────────────────────────────────────────────────

def plot_session_robustness(session_rows: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, layout="constrained",
                                gridspec_kw={"width_ratios": [1.25, 1]})
    pivot = session_rows.pivot(index="session", columns="representation", values="median_rms_z")
    order = pivot["V2"].sort_values().index
    y = np.arange(len(order))
    axis = axes[0]
    _style(axis)
    for yy, session in zip(y, order):
        axis.plot([pivot.loc[session, "R0_current7"], pivot.loc[session, "V2"]], [yy, yy], color=GRID, linewidth=2, zorder=1)
    axis.plot(pivot.loc[order, "R0_current7"], y, "o", color=SERIES[1], label="R0 current 7 features", zorder=2)
    axis.plot(pivot.loc[order, "V2"], y, "o", color=SERIES[0], label="V2 (4 features)", zorder=3)
    axis.set_yticks(y, order, fontsize=8, color=INK_SECONDARY)
    axis.set_xlabel("session median of held-out rms_z", fontsize=9, color=INK_SECONDARY)
    axis.set_title("Held-out session medians (12 sessions with >= 5 usable events)", fontsize=9, color=INK)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")

    axis = axes[1]
    _style(axis)
    values = comparison.set_index("representation")["session_epsilon_squared"]
    names = list(values.index)[::-1]
    y = np.arange(len(names))
    colours = [SERIES[0] if n == "V2" else SERIES[2] if n.startswith("V2_") else SERIES[1] for n in names]
    axis.barh(y, values.reindex(names).to_numpy(), color=colours, height=0.65)
    for yy, name in zip(y, names):
        axis.text(values[name] + 0.004, yy, f"{values[name]:.3f}", va="center", fontsize=7, color=INK_SECONDARY)
    axis.axvline(values["R0_current7"], color=INK_MUTED, linewidth=0.8, linestyle="--")
    axis.set_yticks(y, names, fontsize=8, color=INK_SECONDARY)
    axis.set_title("Session effect on held-out rms_z (epsilon^2); dashed = R0 (gate G1a)", fontsize=9, color=INK)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.suptitle("V2 session robustness under the frozen LOSO baseline (no threshold)", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_feature_stability(fold_parameters: pd.DataFrame, pooled, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE, layout="constrained", sharey=True)
    y = np.arange(len(V2))[::-1]
    for axis, (title, band, transform) in zip(axes, (
            ("Fold centre - pooled centre (pooled robust SD)", (-G2A_MAX_CENTER_SHIFT_SD, G2A_MAX_CENTER_SHIFT_SD),
             lambda folds, p: (folds["center"] - p.median) / p.scale),
            ("Fold scale / pooled scale", G2B_SCALE_RATIO_RANGE, lambda folds, p: folds["scale"] / p.scale))):
        _style(axis)
        axis.axvspan(*band, color=GRID, alpha=0.6, linewidth=0)
        for yy, feature in zip(y, V2):
            folds = fold_parameters[fold_parameters["feature"] == feature]
            values = transform(folds, pooled.parameters[feature]).to_numpy()
            jitter = np.linspace(-0.18, 0.18, len(values))
            axis.plot(values, yy + jitter, "o", color=SERIES[0], markersize=4, alpha=0.8)
        margin = 0.25 * (band[1] - band[0])
        axis.set_xlim(band[0] - margin, band[1] + margin)
        axis.set_title(f"{title}; shaded = pre-registered band", fontsize=9, color=INK)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, V2, fontsize=8, color=INK_SECONDARY)
    figure.suptitle("V2 baseline parameters across the 18 leave-one-session-out folds (gates G2a, G2b)", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_perturbation_response(direction: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    families = (("noise", "white noise SNR (dB)", [30.0, 20.0, 10.0]), ("tilt", "spectral tilt a", [-0.5, 0.5, 0.9]),
                ("gain", "event gain", [0.5, 1 / np.sqrt(2), np.sqrt(2), 2.0]))
    colours = (*SERIES, "#eda100")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, layout="constrained")
    for axis, (family, label, levels) in zip(axes, families):
        _style(axis)
        x = np.arange(len(levels))
        for colour, quantity in zip((*colours, INK), [*(f"z_{f}" for f in V2), "rms_z[V2]"]):
            block = direction[(direction["quantity"] == quantity) & (direction["transform"] == family)]
            medians = [block.loc[np.isclose(block["magnitude"], m), "median_delta"].iloc[0] for m in levels]
            style = "--" if quantity == "rms_z[V2]" else "-"
            axis.plot(x, medians, style, marker="o", color=colour, markersize=4,
                      label=quantity.replace("z_spectral_", "z ").replace("rms_z[V2]", "V2 rms_z"))
        axis.axhline(0, color=INK_MUTED, linewidth=0.8)
        axis.set_xticks(x, [f"{m:g}" if family != "gain" else f"x{m:.3g}" for m in levels], fontsize=8)
        axis.set_xlabel(label, fontsize=9, color=INK_SECONDARY)
        axis.set_title(f"Median change vs unperturbed ({family})", fontsize=9, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("median delta (robust z units)", fontsize=9, color=INK_SECONDARY)
    axes[-1].legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc="upper right")
    axes[-1].set_title("Median change vs unperturbed (gain; note the 1e-4 scale)", fontsize=9, color=INK)
    figure.suptitle("Controlled perturbations: predictable responses to artificial changes, not evidence of inhalation anomalies",
                    color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_analysis(dataset_csv: str | Path = DEFAULT_DATASET_CSV, selection_json: str | Path = DEFAULT_SELECTION_JSON,
                 data_dir: str | Path = config.DATA_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                 reference_run: str | Path | None = None, make_plots: bool = True) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    inputs = [Path(dataset_csv), Path(selection_json), STAGE7_CANDIDATES, STAGE7_COMPARISON, STAGE7_PERTURBED,
              PREREGISTRATION, Path(DEFAULT_MODEL_PATH)]
    hashes_before = {str(p): _sha256(p) for p in inputs}

    v1 = load_feature_selection(selection_json, dataset_csv)
    representations = build_v2_representations(v1)
    table = load_stage1_table(dataset_csv).reset_index(drop=True)
    sessions = table["recording_file"].map(recording_sessions(data_dir))
    groups = comparison_groups(assign_populations(table))
    usable = groups["usable"]
    candidates = pd.read_csv(STAGE7_CANDIDATES)
    if not ((candidates["recording_file"] == table["recording_file"]) & (candidates["event_id"] == table["event_id"])).all():
        raise ValueError("Stage 7 candidate features are not aligned with the Stage 1 table")
    frame = pd.concat([table, candidates[["background_rms", "relative_level_db"]]], axis=1)

    # Part A: LOSO (Strategy C); univariate, so every representation is a column subset of one fit per fold.
    union = list(dict.fromkeys(f for features in representations.values() for f in features))
    z, fold_parameters, baselines = heldout_scores(frame, union, sessions)
    excluded_keys = {k for k, u in zip(zip(table["recording_file"], table["event_id"]), table["usable"]) if not u}
    session_of = dict(zip(zip(table["recording_file"], table["event_id"]), sessions))
    leakage = {"excluded_events_in_any_fit": int(sum(len(set(b.calibration_events) & excluded_keys) for b in baselines.values())),
               "held_out_session_events_in_own_fit": int(sum(sum(session_of[k] == s for k in b.calibration_events)
                                                             for s, b in baselines.items()))}
    references = fold_references(frame, sessions, baselines, representations, union)
    comparison, session_rows, contributions, excluded_rows, scores = representation_rows(
        frame, z, sessions, groups, references, representations)
    stage7 = pd.read_csv(STAGE7_COMPARISON).set_index("representation")
    stage7_check = float(max(abs(comparison.set_index("representation").loc[name, "session_epsilon_squared"]
                                 - stage7.loc[name, "session_epsilon_squared"]) for name in COMPARATORS))

    pooled = fit_baseline(frame[usable], V2)
    frozen = frozen_from_fit(pooled, BASELINE_ID, n_sessions=int(sessions[usable].nunique()), source={
        "dataset": _relative(Path(dataset_csv)), "dataset_sha256": _sha256(Path(dataset_csv)),
        "usability_rule": USABILITY.version, "fit": "median and 1.4826*MAD over all usable events",
        "stage": "Stage 8 (PRISM_RESEARCH_LOG.md Entry 10)", "gate_preregistration_commit": PREREGISTRATION_COMMIT})
    baseline_path = output / "v2_baseline.json"
    write_baseline(frozen, baseline_path)
    reloaded = load_baseline(baseline_path)
    in_memory = pooled.z_scores(frame[usable]).to_numpy()
    roundtrip = float(np.abs(reloaded.z(frame.loc[usable, V2].to_numpy()) - in_memory).max())
    pooled_scores = pd.Series(np.sqrt(np.mean(pooled.z_scores(frame[V2]).to_numpy() ** 2, axis=1)), index=frame.index)

    fold_v2 = fold_parameters[fold_parameters["feature"].isin(V2)]
    pooled_rows = pd.DataFrame([{"fold": "POOLED_ALL_SESSIONS", "feature": f, "n_train": pooled.n_calibration,
                                 "n_scored": np.nan, "n_scored_excluded": np.nan, "center": pooled.parameters[f].median,
                                 "scale": pooled.parameters[f].scale} for f in V2])
    pd.concat([fold_v2, pooled_rows]).assign(mad=lambda d: d["scale"] / MAD_SCALE).to_csv(
        output / "v2_baseline_parameters.csv", index=False)
    feature_scale = feature_scale_stability(fold_parameters, pooled, V2)
    correlation, correlation_folds = correlation_stability(frame, sessions, V2)
    confounds = confound_table(scores, frame, sessions, usable)
    for name, data in (("representation_comparison", comparison),
                       ("feature_scale_stability", feature_scale), ("correlation_structure", correlation),
                       ("correlation_folds", correlation_folds), ("confounds", confounds),
                       ("z_distributions", z_distributions(z, usable, [*V2, LEVEL_CHANNEL])),
                       ("session_robustness", session_rows), ("feature_contributions", contributions),
                       ("excluded_diagnostics_descriptive", excluded_rows),
                       ("v2_session_feature_medians", session_feature_medians(z, sessions, usable, V2))):
        data.to_csv(output / f"{name}.csv", index=False)

    # Part C: controlled perturbations (Stage 6/7 plan and seeds), scored by each event's own fold baseline.
    measured = perturbation_measurements(table, data_dir)
    measured["session"] = measured["row"].map(sessions)
    z_parts = [baselines[s].z_scores(block[union]) for s, block in measured.groupby("session")]
    measured = pd.concat([measured, pd.concat(z_parts).loc[measured.index]], axis=1)
    stage7_perturbed_check = stage7_perturbation_consistency(measured, [*V2, LEVEL_CHANNEL])
    measured[["row", "recording_file", "event_id", "session", "transform", "magnitude", *V2, LEVEL_CHANNEL,
              *(f"z_{f}" for f in (*V2, LEVEL_CHANNEL))]].to_csv(output / "perturbed_features.csv", index=False)
    response = perturbation_response(measured, representations)
    response.to_csv(output / "perturbation_response.csv", index=False)
    feature_perturbation = feature_perturbation_table(measured, [*V2, LEVEL_CHANNEL])
    feature_perturbation.to_csv(output / "feature_perturbation.csv", index=False)
    direction = direction_consistency(measured, perturbation_quantities(representations, [*V2, LEVEL_CHANNEL]))
    direction.to_csv(output / "perturbation_direction.csv", index=False)

    # Part D: leave-one-feature-out ablations (descriptive only).
    by_rep, by_confound = comparison.set_index("representation"), confounds.set_index("representation")
    ablation_rows = []
    for name in ("V2", *ABLATIONS):
        quantity = direction[direction["quantity"] == f"rms_z[{name}]"]
        rep_response = response[response["representation"] == name]

        def level(transform, magnitude, column="median_delta", q=quantity):
            return float(q.loc[(q["transform"] == transform) & np.isclose(q["magnitude"], magnitude), column].iloc[0])

        ablation_rows.append({
            "representation": name, "features": ";".join(representations[name]),
            "session_epsilon_squared": by_rep.loc[name, "session_epsilon_squared"],
            "generalization_ratio": by_rep.loc[name, "generalization_ratio"],
            "session_median_max_over_min": by_rep.loc[name, "session_median_max_over_min"],
            "max_abs_session_vs_rest_delta": by_rep.loc[name, "max_abs_session_vs_rest_delta"],
            "pooled_rho_background_rms": by_confound.loc[name, "pooled_rho_background_rms"],
            "within_session_rho_background_rms": by_confound.loc[name, "within_session_rho_background_rms"],
            "median_delta_rms_z_noise10db": level("noise", 10.0), "median_delta_rms_z_tilt-0.5": level("tilt", -0.5),
            "median_delta_rms_z_tilt0.9": level("tilt", 0.9),
            "median_distance_noise10db": float(rep_response.loc[(rep_response["transform"] == "noise")
                                                                & np.isclose(rep_response["magnitude"], 10), "median_distance_from_own_z"].iloc[0]),
            "median_distance_tilt0.9": float(rep_response.loc[(rep_response["transform"] == "tilt")
                                                              & np.isclose(rep_response["magnitude"], 0.9), "median_distance_from_own_z"].iloc[0]),
            "noise_distance_monotone": float(rep_response.loc[rep_response["transform"] == "noise", "family_distance_monotone_fraction"].min()),
            "tilt_distance_monotone": float(rep_response.loc[rep_response["transform"] == "tilt", "family_distance_monotone_fraction"].min()),
            "max_p95_abs_delta_under_gain": float(quantity[quantity["transform"].isin(["gain", "recording_gain"])]["p95_abs_delta"].max()),
        })
    pd.DataFrame(ablation_rows).to_csv(output / "ablation_summary.csv", index=False)

    # Part E: level channel.
    level_channel = level_channel_evaluation(frame, z, sessions, usable, feature_perturbation, direction, comparison, confounds)
    (output / "level_channel.json").write_text(json.dumps(level_channel, indent=2, default=_json_default) + "\n", encoding="utf-8")

    # G6b: the inference contract reproduces the Stage 1 events, features and usability from raw audio.
    detector = OnnxEventClassifier()
    reproduction_events, reproduction_recordings, reproduction, outputs = reproduce_stage1(
        table, data_dir, reloaded, detector, pooled_scores)
    reproduction_events.to_csv(output / "reproduction_events.csv", index=False)
    reproduction_recordings.to_csv(output / "reproduction_recordings.csv", index=False)
    (output / "reproduction_summary.json").write_text(json.dumps(reproduction, indent=2, default=_json_default) + "\n",
                                                      encoding="utf-8")

    # Contract artifacts that do not depend on the gate outcome.
    (output / "inference_output.schema.json").write_text(json.dumps(output_json_schema(), indent=2) + "\n", encoding="utf-8")
    (output / "v2_feature_schema.json").write_text(json.dumps(feature_schema(), indent=2) + "\n", encoding="utf-8")
    golden = write_golden(output, outputs, data_dir, reloaded, detector)

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_session_robustness(session_rows[session_rows["representation"].isin(["R0_current7", "V2"])], comparison,
                                output / "v2_session_robustness.png")
        plot_feature_stability(fold_parameters, pooled, output / "v2_feature_stability.png")
        plot_perturbation_response(direction, output / "v2_perturbation_response.png")

    tilt = direction[(direction["quantity"] == "z_spectral_centroid_mean") & (direction["transform"] == "tilt")
                     & np.isclose(direction["magnitude"], 0.5)].iloc[0]
    limitations = [
        "No anomaly or technique ground truth: anomaly_score measures distance from the reference-dataset baseline only.",
        "The baseline is global (18 recording sessions of unknown subjects/devices), not personalized; session "
        f"dependence remains (held-out epsilon^2 {by_rep.loc['V2', 'session_epsilon_squared']:.3f}).",
        "Not validated on PRISM hardware. A different microphone or enclosure acts like a spectral tilt: a first-order "
        f"tilt a = 0.5 shifts z(spectral_centroid_mean) by a median {tilt['median_delta']:.2f}, so hardware scores need "
        "a hardware reference study before interpretation.",
        "PRISM hardware buffers 5 s; reference recordings are 6.5-12.5 s. Detector behaviour and boundary censoring on "
        "5 s clips are untested (events touching the clip edges are NOT_SCOREABLE).",
        "The detector was trained on about two-thirds of the reference recordings; event agreement is mostly in-sample.",
        "V2 was proposed after inspecting these data (Stages 2-7) and is evaluated on the same 318 events.",
    ]

    evidence = {"comparison": comparison, "feature_scale": feature_scale, "correlation": correlation,
                "direction": direction, "response": response, "feature_perturbation": feature_perturbation,
                "confounds": confounds, "reproduction": reproduction, "roundtrip_max_abs_diff": roundtrip,
                "leakage": leakage, "rerun": None}
    detector_sha = _sha256(Path(DEFAULT_MODEL_PATH))
    contract_path = output / "inference_contract_v2.json"

    def write_contract() -> tuple[list[dict], str]:
        rows = evaluate_gate(evidence)
        verdict = gate_outcome(rows)
        contract = build_contract(reloaded, _sha256(baseline_path), detector_sha, verdict, rows, limitations)
        contract_path.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return rows, verdict

    gate_rows, outcome = write_contract()
    rerun = None
    if reference_run:     # G6a: compare every deterministic output (the contract without its gate fields) with a previous run
        rerun = evidence["rerun"] = compare_runs(output, Path(reference_run))
        gate_rows, outcome = write_contract()
    pd.DataFrame(gate_rows).assign(value=lambda d: d["value"].map(lambda v: json.dumps(v, default=_json_default))).to_csv(
        output / "acceptance_gate_results.csv", index=False)
    gate_payload = {"outcome": outcome, "contract_status": CONTRACT_STATUS[outcome],
                    "preregistration_sha256": _sha256(PREREGISTRATION), "preregistration_commit": PREREGISTRATION_COMMIT,
                    "criteria": gate_rows, "no_threshold_or_labels": True}
    (output / "acceptance_gate_results.json").write_text(json.dumps(gate_payload, indent=2, default=_json_default) + "\n",
                                                        encoding="utf-8")

    hashes_after = {path: _sha256(Path(path)) for path in hashes_before}
    summary = {
        "stage": "Stage 8 - V2 representation validation + inference contract (no threshold, no labels)",
        "v2_features": V2, "representations": representations, "baseline_id": BASELINE_ID,
        "gate_outcome": outcome, "contract_status": CONTRACT_STATUS[outcome],
        "leakage": leakage, "roundtrip_max_abs_diff": roundtrip, "reproduction": reproduction, "rerun": rerun,
        "consistency": {"max_abs_diff_comparator_epsilon2_vs_stage7": stage7_check,
                        "max_relative_diff_perturbed_features_vs_stage7": stage7_perturbed_check},
        "golden_cases": [c["case"] for c in golden["cases"]],
        "inputs_unchanged": hashes_before == hashes_after,
        "input_sha256": {_relative(Path(p)): h for p, h in hashes_before.items()},
        "parameters": {"seed": SEED, "min_session_events": MIN_GROUP_SIZE},
        "provenance": {"git": _git_state()},
        "no_threshold_or_labels": True,
    }
    summary["outputs"] = sorted(str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*") if p.is_file())
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8 V2 validation, acceptance gate and inference contract (no threshold)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-run", help="a previous complete run to compare byte-for-byte (gate G6a)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_analysis(output_dir=args.output_dir, reference_run=args.reference_run, make_plots=not args.no_plots)
    print(json.dumps({k: result[k] for k in ("gate_outcome", "contract_status", "leakage", "roundtrip_max_abs_diff",
                                             "consistency", "inputs_unchanged")}, indent=2, default=str))
