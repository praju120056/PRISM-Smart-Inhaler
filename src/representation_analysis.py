"""Stage 7: representation robustness and ablation study.

See PRISM_RESEARCH_LOG.md, Entry 9.

Question: are the current scores detecting acoustically meaningful deviations,
or are they dominated by event duration, recording level, session effects or
fragile features?  Every representation is evaluated with the Stage 5/6
leave-one-session-out (LOSO) protocol: for each session, per-feature median /
1.4826*MAD is fitted on the USABLE events of the other sessions only and
frozen; rms_z is the working score.  No threshold is chosen and no event is
labelled NORMAL or ANOMALY.  Excluded status (Stage 1) is used only as a
diagnostic population, never as anomaly ground truth.

Candidate features and representations were fixed in this file before any
Stage 7 result was computed:

* energy-weighted (ew_) spectral statistics: the extractor's per-frame
  centroid / flatness / rolloff, averaged with weights proportional to frame
  energy (librosa RMS^2 on the extractor's framing).  Rationale: low-energy
  frames (event edges, pauses) are dominated by the recording noise floor,
  which plausibly causes the Stage 6 noise fragility of spectral_flatness_std
  and the level/brightness coupling.  Computed within the event only.
* relative_level_db = 20*log10(event mean_rms / median RMS envelope of the
  whole recording).  Cancels device gain and distance (both scale the whole
  recording) while keeping how loud the event is relative to its own
  recording.  Computable at inference; uses no session information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config
from baseline_strategies import _epsilon_squared, _robust_sd
from baseline_v1 import DEFAULT_SELECTION_JSON, load_feature_selection, recording_sessions
from feature_analysis import (
    DEFAULT_DATASET_CSV,
    MIN_GROUP_SIZE,
    exact_event_bounds,
    load_stage1_table,
    within_group_spearman,
)
from inhale_dataset import UsabilityRule, _git_state, _json_default, _relative, _sha256
from natural_population_analysis import (
    BOOTSTRAP_DRAWS,
    PERTURBATION_ORDER,
    SEED as STAGE6_SEED,
    assign_populations,
    cliffs_delta,
    comparison_groups,
    heldout_scores,
    measure_segment,
    perturb,
    perturbation_plan,
    session_cluster_bootstrap_delta,
    stratified_auc,
)
from scoring_v1 import combined_scores, feature_contributions


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "representation_analysis"
STAGE6_PERTURBED = Path(config.RESULTS_DIR) / "natural_population" / "controlled" / "perturbed_events.csv"
SEED = 20261004

EW_MAP = {
    "spectral_centroid_mean": "ew_spectral_centroid_mean",
    "spectral_centroid_std": "ew_spectral_centroid_std",
    "spectral_flatness_mean": "ew_spectral_flatness_mean",
    "spectral_flatness_std": "ew_spectral_flatness_std",
    "spectral_rolloff_std": "ew_spectral_rolloff_std",
}
RELATIVE_LEVEL = "relative_level_db"
CANDIDATE_FEATURES = (*EW_MAP.values(), RELATIVE_LEVEL)
DURATION_GATE_S = UsabilityRule().min_duration_s     # R2: duration acts only through the Stage 1 usability gate
RECORDING_GAINS = (0.5, 2.0)                          # whole-recording gain: models device gain / distance
FAMILY_ORDER = {**PERTURBATION_ORDER, "recording_gain": [0.5, 1.0, 2.0]}
IDENTITY_VALUE = {"gain": 1.0, "noise": np.inf, "tilt": 0.0, "recording_gain": 1.0}

# Pre-specified gate for exploratory multivariate scoring (model diagnostics, not anomaly thresholds).
MCD_MIN_EVENTS_PER_FEATURE = 10
MCD_MAX_CONDITION_NUMBER = 100.0
MCD_MAX_CORRELATION_SHIFT = 0.2

DESCRIPTIONS = {
    "R0_current7": "current V1 representation (7 features)",
    "R1_no_duration": "duration removed",
    "R2_duration_as_usability": "duration removed from the score; events < 0.5 s are not scoreable (usability gate only)",
    "R3_no_flatness_std": "spectral_flatness_std removed",
    "R4_ew_flatness_std": "spectral_flatness_std replaced by its energy-weighted version",
    "R5_no_mean_rms": "absolute level (mean_rms) removed",
    "R6_relative_level": "mean_rms replaced by relative_level_db",
    "R7_ew_spectral": "all five spectral features replaced by energy-weighted versions",
    "R8_robust_candidate": "no duration; relative_level_db; energy-weighted spectral features",
}


def build_representations(v1: Sequence[str]) -> dict[str, list[str]]:
    """The pre-specified representations, derived from the authoritative V1 list."""
    v1 = list(v1)

    def without(features, *removed):
        return [f for f in features if f not in removed]

    def replace(features, mapping):
        return [mapping.get(f, f) for f in features]

    representations = {
        "R0_current7": v1,
        "R1_no_duration": without(v1, "duration_s"),
        "R2_duration_as_usability": without(v1, "duration_s"),
        "R3_no_flatness_std": without(v1, "spectral_flatness_std"),
        "R4_ew_flatness_std": replace(v1, {"spectral_flatness_std": EW_MAP["spectral_flatness_std"]}),
        "R5_no_mean_rms": without(v1, "mean_rms"),
        "R6_relative_level": replace(v1, {"mean_rms": RELATIVE_LEVEL}),
        "R7_ew_spectral": replace(v1, EW_MAP),
        "R8_robust_candidate": without(replace(v1, {**EW_MAP, "mean_rms": RELATIVE_LEVEL}), "duration_s"),
    }
    for name, features in representations.items():
        if len(set(features)) != len(features) or not features:
            raise ValueError(f"{name}: invalid feature list {features}")
    return representations


def scoreable_mask(frame: pd.DataFrame, representation: str) -> np.ndarray:
    """R2 scores only events that pass the duration usability gate; others score everything."""
    if representation == "R2_duration_as_usability":
        return frame["duration_s"].to_numpy(dtype=float) >= DURATION_GATE_S - 1e-9
    return np.ones(len(frame), bool)


# ── Candidate features (within-event only) ──────────────────────────────────

def frame_spectral_statistics(segment: np.ndarray, sample_rate: int) -> dict:
    """Unweighted (uw_) and energy-weighted (ew_) statistics of per-frame spectral features.

    Frames are the extractor's (n_fft 256, hop 64, centred).  Energy weights are
    librosa RMS^2 on the same framing.  uw_ values reproduce the stored features.
    """
    import librosa
    from librosa_extractor import extract_features_from_audio

    audio = np.asarray(segment, dtype=np.float32)
    features = extract_features_from_audio(audio, sample_rate)
    if features is None:
        raise ValueError("feature extraction failed")
    spectral = features[:, 3 * config.LIBROSA_N_MFCC:].astype(np.float64)
    power = librosa.feature.rms(y=audio, frame_length=config.LIBROSA_N_FFT, hop_length=config.LIBROSA_HOP_LENGTH,
                                center=True)[0].astype(np.float64) ** 2
    n = min(len(power), len(spectral))
    spectral, power = spectral[:n], power[:n]
    if not np.isfinite(power).all() or power.sum() <= 0:
        raise ValueError("segment has no energy; energy weights are undefined")
    weights = power / power.sum()
    result = {"effective_frames": float(1.0 / np.sum(weights ** 2)), "frames": int(n)}
    for column, name in ((0, "spectral_centroid"), (1, "spectral_flatness"), (2, "spectral_rolloff")):
        x = spectral[:, column]
        weighted_mean = float(np.sum(weights * x))
        result[f"uw_{name}_mean"] = float(x.mean())
        result[f"uw_{name}_std"] = float(x.std())
        result[f"ew_{name}_mean"] = weighted_mean
        result[f"ew_{name}_std"] = float(np.sqrt(np.sum(weights * (x - weighted_mean) ** 2)))
    return result


def recording_background_rms(waveform: np.ndarray, sample_rate: int) -> float:
    """Median of the existing post_event RMS envelope over the whole recording."""
    from post_event import _rms_envelope

    _, envelope = _rms_envelope(np.asarray(waveform, dtype=np.float32), sample_rate)
    return float(np.median(envelope))


def relative_level_db(mean_rms: float, background_rms: float) -> float:
    if not (mean_rms > 0 and background_rms > 0):
        raise ValueError("relative level needs positive event and background RMS")
    return float(20 * np.log10(mean_rms / background_rms))


def candidate_features_for_events(table: pd.DataFrame, data_dir: str | Path = config.DATA_DIR) -> pd.DataFrame:
    """ew_ spectral statistics and relative_level_db for every detected event (usable or not)."""
    from librosa_extractor import load_audio
    from post_event import InhaleEvent, extract_event_audio

    rows = []
    for recording, group in table.groupby("recording_file", sort=False):
        waveform, sample_rate = load_audio(str(Path(data_dir) / recording))
        background = recording_background_rms(waveform, sample_rate)
        for index, event in group.iterrows():
            start, end = exact_event_bounds(event["start_s"], event["end_s"], event["recording_duration_s"])
            segment, _ = extract_event_audio(waveform, InhaleEvent("Inhale", start, end, end - start, 0, 0, 0, ()),
                                             sample_rate=sample_rate)
            statistics = frame_spectral_statistics(segment, sample_rate)
            rows.append({"row": index, "background_rms": background,
                         RELATIVE_LEVEL: relative_level_db(event["mean_rms"], background),
                         **{EW_MAP[f]: statistics[f.replace("spectral_", "ew_spectral_", 1)] for f in EW_MAP},
                         "effective_frames": statistics["effective_frames"], "frames": statistics["frames"],
                         "uw_reproduction_error": max(abs(statistics[f"uw_{f}"] - event[f]) for f in EW_MAP)})
    return pd.DataFrame(rows).set_index("row").loc[table.index].reset_index(drop=True)


# ── Evaluation ──────────────────────────────────────────────────────────────

def fold_references(frame: pd.DataFrame, sessions: pd.Series, baselines: Mapping,
                    representations: Mapping[str, Sequence[str]], union: Sequence[str]) -> dict:
    """Per representation and fold: median in-sample rms_z of that fold's training events."""
    usable = frame["usable"].astype(bool).to_numpy()
    references = {name: {} for name in representations}
    for session, baseline in baselines.items():
        training = frame[usable & (sessions != session).to_numpy()]
        z = baseline.z_scores(training)
        for name, features in representations.items():
            references[name][session] = float(combined_scores(z[[f"z_{f}" for f in features]], features)["rms_z"].median())
    return references


def session_dependence(scores, sessions, mask, min_events: int = MIN_GROUP_SIZE) -> dict:
    """Session effect on scores of ``mask`` events: epsilon^2, medians, largest session-vs-rest delta."""
    scores, sessions = np.asarray(scores, dtype=float), np.asarray(sessions)
    mask = np.asarray(mask, bool)
    labels, counts = np.unique(sessions[mask], return_counts=True)
    tested = labels[counts >= min_events]
    groups = [scores[mask & (sessions == s)] for s in tested]
    epsilon, _ = _epsilon_squared(groups)
    medians = np.array([np.median(g) for g in groups])
    deltas = [cliffs_delta(scores[mask & (sessions == s)], scores[mask & (sessions != s)]) for s in tested]
    return {"sessions_tested": int(len(tested)), "session_epsilon_squared": epsilon,
            "session_median_min": float(medians.min()), "session_median_max": float(medians.max()),
            "session_median_max_over_min": float(medians.max() / medians.min()),
            "max_abs_session_vs_rest_delta": float(np.max(np.abs(deltas)))}


def excluded_diagnostics(scores, sessions, groups: Mapping[str, np.ndarray], eligible: np.ndarray,
                         names: Sequence[str] = ("excluded_all", "any_too_short", "only_close_neighbor"),
                         draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> list[dict]:
    """Excluded-vs-usable separation (diagnostic only: excluded is not anomaly ground truth)."""
    scores, sessions = np.asarray(scores, dtype=float), np.asarray(sessions)
    reference = groups["usable"] & eligible
    rows = []
    for name in names:
        mask = groups[name] & eligible
        if mask.sum() < 3:
            rows.append({"group": name, "n_group": int(mask.sum()), "note": "fewer than 3 scoreable events"})
            continue
        delta = cliffs_delta(scores[mask], scores[reference])
        low, high = session_cluster_bootstrap_delta(scores[mask], sessions[mask], scores[reference],
                                                    sessions[reference], draws, seed)
        rows.append({"group": name, "n_group": int(mask.sum()), "n_usable": int(reference.sum()),
                     "median_group": float(np.median(scores[mask])), "median_usable": float(np.median(scores[reference])),
                     "cliffs_delta": delta, "auc": (delta + 1) / 2,
                     "delta_session_bootstrap_low": low, "delta_session_bootstrap_high": high,
                     "stratified_auc": stratified_auc(scores, sessions, mask, reference)})
    return rows


def contribution_rows(z: pd.DataFrame, features: Sequence[str], masks: Mapping[str, np.ndarray]) -> list[dict]:
    contributions = feature_contributions(z[[f"z_{f}" for f in features]], features)
    rows = []
    for population, mask in masks.items():
        if not mask.any():
            continue
        for feature in features:
            rows.append({"population": population, "feature": feature, "n": int(mask.sum()),
                         "rms_z_mean_share": float(np.nanmean(contributions[f"rms_share_{feature}"].to_numpy()[mask])),
                         "argmax_fraction": float(contributions[f"is_max_{feature}"].to_numpy()[mask].mean()),
                         "median_abs_z": float(np.median(contributions[f"abs_z_{feature}"].to_numpy()[mask]))})
    return rows


def feature_robustness(frame: pd.DataFrame, z: pd.DataFrame, sessions: pd.Series, groups: Mapping[str, np.ndarray],
                       features: Sequence[str]) -> pd.DataFrame:
    """Per feature: session dependence of held-out z, coupling with level, excluded separation."""
    usable = groups["usable"]
    usable_frame = frame[usable].assign(session=sessions[usable].to_numpy())
    within = within_group_spearman(usable_frame, list(features), "session")
    rows = []
    for feature in features:
        values = z[f"z_{feature}"].to_numpy(dtype=float)
        dependence = session_dependence(values, sessions, usable)
        labels, counts = np.unique(sessions[usable], return_counts=True)
        session_medians = [np.median(values[usable & (sessions == s).to_numpy()]) for s in labels[counts >= MIN_GROUP_SIZE]]
        absolute = np.abs(values)
        rows.append({
            "feature": feature, "candidate": feature in CANDIDATE_FEATURES,
            "session_epsilon_squared": dependence["session_epsilon_squared"],
            "robust_sd_session_median_z": _robust_sd(session_medians),
            "spearman_with_mean_rms": float(stats.spearmanr(usable_frame[feature], usable_frame["mean_rms"]).statistic),
            "within_session_spearman_with_mean_rms": float(within.loc[feature, "mean_rms"]),
            "spearman_with_duration": float(stats.spearmanr(usable_frame[feature], usable_frame["duration_s"]).statistic),
            "abs_z_auc_excluded_vs_usable": (cliffs_delta(absolute[groups["excluded_all"]], absolute[usable]) + 1) / 2,
            "abs_z_auc_close_only_vs_usable": (cliffs_delta(absolute[groups["only_close_neighbor"]], absolute[usable]) + 1) / 2,
        })
    return pd.DataFrame(rows)


# ── Controlled perturbations ────────────────────────────────────────────────

def perturbation_measurements(table: pd.DataFrame, data_dir: str | Path = config.DATA_DIR,
                              seed: int = STAGE6_SEED) -> pd.DataFrame:
    """Existing and candidate features of perturbed usable events (event boundaries fixed).

    Event-level perturbations (Stage 6 plan, same seeds) change only the event
    segment; ``recording_gain`` scales the whole recording, so the background used
    by relative_level_db scales too.
    """
    from librosa_extractor import load_audio
    from post_event import InhaleEvent, extract_event_audio

    rows = []
    usable = table[table["usable"].astype(bool)]
    for recording, group in usable.groupby("recording_file", sort=False):
        waveform, sample_rate = load_audio(str(Path(data_dir) / recording))
        background = recording_background_rms(waveform, sample_rate)
        for index, event in group.iterrows():
            start, end = exact_event_bounds(event["start_s"], event["end_s"], event["recording_duration_s"])
            bounds = InhaleEvent("Inhale", start, end, end - start, 0, 0, 0, ())
            segment, _ = extract_event_audio(waveform, bounds, sample_rate=sample_rate)
            original = segment.copy()
            cases = [(t, m, perturb(segment, t, m, np.random.default_rng([seed, int(index)])), background)
                     for t, m in perturbation_plan()]
            for gain in RECORDING_GAINS:
                gained = (np.asarray(waveform, dtype=np.float64) * gain).astype(np.float32)
                gained_segment, _ = extract_event_audio(gained, bounds, sample_rate=sample_rate)
                cases.append(("recording_gain", gain, gained_segment.astype(np.float64),
                              recording_background_rms(gained, sample_rate)))
            for transform, magnitude, signal_values, background_rms in cases:
                measured = measure_segment(signal_values, sample_rate)
                statistics = frame_spectral_statistics(signal_values.astype(np.float32), sample_rate)
                rows.append({"row": index, "recording_file": recording, "event_id": event["event_id"],
                             "transform": transform, "magnitude": magnitude, "duration_s": event["duration_s"],
                             **{f: measured[f] for f in ("mean_rms", *EW_MAP)},
                             **{EW_MAP[f]: statistics[f.replace("spectral_", "ew_spectral_", 1)] for f in EW_MAP},
                             RELATIVE_LEVEL: relative_level_db(measured["mean_rms"], background_rms)})
            if not np.array_equal(segment, original):
                raise ValueError("a perturbation modified the original segment in place")
    return pd.DataFrame(rows)


# Changes below this many robust-z units are treated as zero.  Measured float32
# rounding jitter of exactly gain-invariant features is <= 1e-5 z (Entry 9).
MONOTONE_TOLERANCE_Z = 1e-4


def monotone_outward(matrix: np.ndarray, identity_position: int, tolerance: float = MONOTONE_TOLERANCE_Z) -> np.ndarray:
    """Per row: values do not decrease moving away from the identity column on either side."""
    ok = np.ones(len(matrix), bool)
    if identity_position < matrix.shape[1] - 1:
        ok &= (np.diff(matrix[:, identity_position:], axis=1) >= -tolerance).all(axis=1)
    if identity_position > 0:
        ok &= (np.diff(matrix[:, :identity_position + 1][:, ::-1], axis=1) >= -tolerance).all(axis=1)
    return ok


def monotone_either(matrix: np.ndarray, tolerance: float = MONOTONE_TOLERANCE_Z) -> np.ndarray:
    steps = np.diff(matrix, axis=1)
    return (steps >= -tolerance).all(axis=1) | (steps <= tolerance).all(axis=1)


def _ordered_blocks(perturbed: pd.DataFrame, transform: str) -> tuple[list[pd.DataFrame], int]:
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    blocks, position = [], None
    for k, magnitude in enumerate(FAMILY_ORDER[transform]):
        if magnitude == IDENTITY_VALUE[transform]:
            blocks.append(identity)
            position = k
        else:
            blocks.append(perturbed[(perturbed["transform"] == transform)
                                    & np.isclose(perturbed["magnitude"], magnitude)].set_index("row"))
    return blocks, position


def perturbation_response(perturbed: pd.DataFrame, representations: Mapping[str, Sequence[str]],
                          min_session_events: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Per representation and perturbation: score response, dominance, monotonicity, session direction."""
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    rows = []
    for name, features in representations.items():
        columns = [f"z_{f}" for f in features]
        base_score = combined_scores(identity[columns], features)["rms_z"]
        for transform in FAMILY_ORDER:
            blocks, position = _ordered_blocks(perturbed, transform)
            index = blocks[0].index
            distance = np.column_stack([np.sqrt(((b.loc[index, columns].to_numpy() - identity.loc[index, columns].to_numpy()) ** 2)
                                                .mean(axis=1)) for b in blocks])
            monotone = float(monotone_outward(distance, position).mean())
            for k, block in enumerate(blocks):
                if k == position:
                    continue
                delta_z = block.loc[index, columns].to_numpy() - identity.loc[index, columns].to_numpy()
                squared = delta_z ** 2
                total = squared.sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dominance = np.where(total > 0, squared.max(axis=1) / total, np.nan)
                top = pd.Series(np.array(features)[squared.argmax(axis=1)])
                change = combined_scores(block.loc[index, columns], features)["rms_z"].to_numpy() - base_score.loc[index].to_numpy()
                sessions = block.loc[index, "session"].to_numpy()
                labels, counts = np.unique(sessions, return_counts=True)
                tested = labels[counts >= min_session_events]
                session_medians = np.array([np.median(change[sessions == s]) for s in tested])
                rows.append({
                    "representation": name, "transform": transform, "magnitude": FAMILY_ORDER[transform][k],
                    "n_events": int(len(index)), "median_delta_rms_z": float(np.median(change)),
                    "frac_rms_z_increase": float(np.mean(change > 0)),
                    "median_distance_from_own_z": float(np.median(distance[:, k])),
                    "median_single_feature_dominance": float(np.nanmedian(dominance)) if np.isfinite(dominance).any()
                    else float("nan"),                      # NaN: no feature of this representation moved
                    "most_common_dominant_feature": top.mode().iloc[0] if (total > 0).any() else "none",
                    "family_distance_monotone_fraction": monotone,
                    "sessions_tested": int(len(tested)),
                    "frac_sessions_median_delta_positive": float(np.mean(session_medians > 0)) if len(tested) else np.nan,
                })
    return pd.DataFrame(rows)


def feature_perturbation_table(perturbed: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Per feature and perturbation family: median delta z per magnitude and monotone fraction."""
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    rows = []
    for transform in FAMILY_ORDER:
        blocks, position = _ordered_blocks(perturbed, transform)
        index = blocks[0].index
        for feature in features:
            matrix = np.column_stack([b.loc[index, f"z_{feature}"].to_numpy() for b in blocks])
            row = {"feature": feature, "transform": transform,
                   "monotone_fraction": float(monotone_either(matrix).mean()),
                   "max_abs_delta_z": float(np.abs(matrix - matrix[:, [position]]).max())}
            for k, magnitude in enumerate(FAMILY_ORDER[transform]):
                if k != position:
                    row[f"median_delta_z@{magnitude:g}"] = float(np.median(matrix[:, k] - matrix[:, position]))
            rows.append(row)
    return pd.DataFrame(rows)


# ── Exploratory multivariate scoring (gated) ────────────────────────────────

def multivariate_gate(diagnostics: pd.DataFrame) -> dict:
    """Apply the pre-specified conditioning / stability gate per representation."""
    result = {}
    for name, block in diagnostics.groupby("representation"):
        passed = bool((block["n_train"] >= MCD_MIN_EVENTS_PER_FEATURE * block["d"]).all()
                      and (block["condition_number"] <= MCD_MAX_CONDITION_NUMBER).all()
                      and (block["max_abs_correlation_shift_vs_pooled"] <= MCD_MAX_CORRELATION_SHIFT).all())
        result[name] = {"passed": passed,
                        "max_condition_number": float(block["condition_number"].max()),
                        "max_correlation_shift": float(block["max_abs_correlation_shift_vs_pooled"].max()),
                        "min_n_train_per_feature": float((block["n_train"] / block["d"]).min())}
    return result


def mcd_loso(frame: pd.DataFrame, z: pd.DataFrame, sessions: pd.Series, baselines: Mapping,
             representations: Mapping[str, Sequence[str]], seed: int = SEED):
    """Robust (MCD) Mahalanobis distance in robust-z space, fitted per LOSO fold on usable events only.

    Returns fold diagnostics, held-out scores sqrt(d^2 / d) for every event, per-fold
    in-sample reference medians, and the fitted estimators (for perturbation scoring).
    """
    from sklearn.covariance import MinCovDet

    usable = frame["usable"].astype(bool).to_numpy()
    pooled_z = pd.concat([baselines[s].z_scores(frame[(sessions == s).to_numpy() & usable]) for s in baselines])
    diagnostics, scores, references, models = [], {}, {}, {}
    for name, features in representations.items():
        columns = [f"z_{f}" for f in features]
        pooled_corr = _correlation(MinCovDet(random_state=seed).fit(pooled_z[columns].to_numpy()).covariance_)
        values = np.full(len(frame), np.nan)
        for session, baseline in baselines.items():
            training_mask = usable & (sessions != session).to_numpy()
            training = baseline.z_scores(frame[training_mask])[columns].to_numpy()
            model = MinCovDet(random_state=seed).fit(training)
            eigen = np.linalg.eigvalsh(model.covariance_)
            held = (sessions == session).to_numpy()
            values[held] = np.sqrt(model.mahalanobis(z.loc[held, columns].to_numpy()) / len(features))
            references.setdefault(name, {})[session] = float(np.median(np.sqrt(model.mahalanobis(training) / len(features))))
            models[(name, session)] = model
            diagnostics.append({"representation": name, "fold": session, "d": len(features),
                                "n_train": int(training_mask.sum()), "support_size": int(model.support_.sum()),
                                "condition_number": float(eigen.max() / eigen.min()),
                                "min_eigenvalue": float(eigen.min()), "max_eigenvalue": float(eigen.max()),
                                "max_abs_correlation_shift_vs_pooled":
                                    float(np.max(np.abs(_correlation(model.covariance_) - pooled_corr)))})
        scores[name] = values
    return pd.DataFrame(diagnostics), scores, references, models


def _correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_INK = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_INK)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def plot_representation_overview(comparison: pd.DataFrame, perturbation: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(DESCRIPTIONS)
    y = np.arange(len(names))[::-1]
    panels = [
        ("Session effect on usable held-out rms_z (epsilon^2)", comparison.set_index("representation")["session_epsilon_squared"], None),
        ("Excluded vs usable Cliff's delta (diagnostic only)", comparison.set_index("representation")["excluded_all_delta"],
         (comparison.set_index("representation")["excluded_all_delta_low"], comparison.set_index("representation")["excluded_all_delta_high"])),
        ("Median delta rms_z: white noise 20 dB", perturbation[(perturbation["transform"] == "noise") & np.isclose(perturbation["magnitude"], 20)].set_index("representation")["median_delta_rms_z"], None),
        ("Median delta rms_z: recording gain x2", perturbation[(perturbation["transform"] == "recording_gain") & np.isclose(perturbation["magnitude"], 2)].set_index("representation")["median_delta_rms_z"], None),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 5.5), facecolor=SURFACE, layout="constrained", sharey=True)
    for axis, (title, values, interval) in zip(axes, panels):
        _style(axis)
        v = values.reindex(names).to_numpy(dtype=float)
        if interval is not None:
            axis.hlines(y, interval[0].reindex(names), interval[1].reindex(names), color=BASELINE_INK, linewidth=2)
        axis.plot(v, y, "o", color=SERIES[0], markersize=7)
        for yy, value in zip(y, v):
            if np.isfinite(value):
                axis.text(value, yy + 0.28, f"{value:.2f}", fontsize=7, color=INK_SECONDARY, ha="center")
        axis.axvline(0, color=INK_MUTED, linewidth=0.8)
        axis.set_title(title, fontsize=9, color=INK)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, names, fontsize=8, color=INK_SECONDARY)
    figure.suptitle("Representation comparison under the frozen LOSO baseline (rms_z; no threshold)", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_feature_robustness(robustness: pd.DataFrame, feature_perturbation: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = list(robustness["feature"])[::-1]
    noise = feature_perturbation[feature_perturbation["transform"] == "noise"].set_index("feature")["median_delta_z@30"]
    panels = [("Session effect on held-out z (epsilon^2)", robustness.set_index("feature")["session_epsilon_squared"]),
              ("Median delta z at 30 dB white noise", noise),
              ("Within-session Spearman with mean_rms", robustness.set_index("feature")["within_session_spearman_with_mean_rms"])]
    figure, axes = plt.subplots(1, 3, figsize=(14, 6), facecolor=SURFACE, layout="constrained", sharey=True)
    y = np.arange(len(order))
    colours = [SERIES[1] if f in CANDIDATE_FEATURES else SERIES[0] for f in order]
    for axis, (title, values) in zip(axes, panels):
        _style(axis)
        axis.barh(y, values.reindex(order).to_numpy(dtype=float), color=colours, height=0.7)
        axis.axvline(0, color=INK_MUTED, linewidth=0.8)
        axis.set_title(title, fontsize=9, color=INK)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, order, fontsize=8, color=INK_SECONDARY)
    from matplotlib.patches import Patch
    axes[-1].legend([Patch(color=SERIES[0]), Patch(color=SERIES[1])], ["current feature", "Stage 7 candidate"],
                    loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    figure.suptitle("Per-feature robustness (usable events, held-out LOSO z)", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_duration_ablation(scores: Mapping[str, np.ndarray], groups: Mapping[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    shown = ("R0_current7", "R1_no_duration")
    populations = ("usable", "only_close_neighbor", "any_too_short")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, layout="constrained", sharey=True)
    for axis, name in zip(axes, shown):
        _style(axis)
        data = [scores[name][groups[p]] for p in populations]
        parts = axis.boxplot(data, widths=0.6, patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                             whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
                             flierprops={"marker": "o", "markersize": 3, "markeredgecolor": INK_MUTED,
                                         "markerfacecolor": "none"})
        for box, colour in zip(parts["boxes"], SERIES):
            box.set_facecolor(colour)
            box.set_edgecolor(SURFACE)
        axis.set_xticks([1, 2, 3], [f"{p}\n(n={int(groups[p].sum())})" for p in populations], fontsize=8, color=INK_SECONDARY)
        axis.set_title(f"{name}: {DESCRIPTIONS[name]}", fontsize=9, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("held-out rms_z", color=INK_SECONDARY, fontsize=9)
    figure.suptitle("Duration ablation: excluded-event scores with and without duration (diagnostic only)", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_analysis(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_perturbations: bool = True,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [Path(dataset_csv), Path(selection_json), STAGE6_PERTURBED]
    hashes_before = {str(p): _sha256(p) for p in inputs if p.exists()}

    v1 = load_feature_selection(selection_json, dataset_csv)
    representations = build_representations(v1)
    table = load_stage1_table(dataset_csv).reset_index(drop=True)
    sessions = table["recording_file"].map(recording_sessions(data_dir))
    populations = assign_populations(table)
    groups = comparison_groups(populations)

    candidates = candidate_features_for_events(table, data_dir)
    frame = pd.concat([table, candidates.drop(columns=["uw_reproduction_error"])], axis=1)
    union = list(dict.fromkeys(f for features in representations.values() for f in features))
    pd.concat([table[["recording_file", "event_id", "usable", "exclusion_reasons", "duration_s", "mean_rms"]],
               sessions.rename("session"), candidates], axis=1).to_csv(output / "candidate_features.csv", index=False)

    # One LOSO fit per fold on usable events of other sessions; features are univariate,
    # so every representation's z is a column subset of this union fit.
    z, fold_parameters, baselines = heldout_scores(frame, union, sessions)
    fold_parameters.to_csv(output / "baseline_parameters.csv", index=False)
    usable = groups["usable"]
    excluded_keys = {k for k, u in zip(zip(table["recording_file"], table["event_id"]), table["usable"]) if not u}
    session_of = dict(zip(zip(table["recording_file"], table["event_id"]), sessions))
    leakage = {
        "excluded_events_in_any_fit": int(sum(len(set(b.calibration_events) & excluded_keys) for b in baselines.values())),
        "held_out_session_events_in_own_fit": int(sum(sum(session_of[k] == s for k in b.calibration_events)
                                                      for s, b in baselines.items())),
        "uw_frame_statistics_max_reproduction_error": float(candidates["uw_reproduction_error"].max()),
    }
    references = fold_references(frame, sessions, baselines, representations, union)

    comparison_rows, duration_rows, contributions, session_rows, scores_by_rep = [], [], [], [], {}
    for name, features in representations.items():
        columns = [f"z_{f}" for f in features]
        scores = combined_scores(z[columns], features)["rms_z"].to_numpy()
        scores_by_rep[name] = scores
        eligible = scoreable_mask(frame, name)
        reference = np.array([references[name][s] for s in sessions])
        dependence = session_dependence(scores, sessions, usable)
        diagnostics = excluded_diagnostics(scores, sessions, groups, eligible)
        by_group = {row["group"]: row for row in diagnostics}
        comparison_rows.append({
            "representation": name, "description": DESCRIPTIONS[name], "features": ";".join(features),
            "d": len(features), "usable_median_rms_z": float(np.median(scores[usable])),
            "usable_p05": float(np.percentile(scores[usable], 5)), "usable_p95": float(np.percentile(scores[usable], 95)),
            "generalization_ratio": float(np.median(scores[usable]) / np.median(reference[usable])),
            **dependence,
            **{f"{g}_{k}": by_group[g].get(v, np.nan) for g in ("excluded_all", "any_too_short", "only_close_neighbor")
               for k, v in (("n", "n_group"), ("delta", "cliffs_delta"), ("delta_low", "delta_session_bootstrap_low"),
                            ("delta_high", "delta_session_bootstrap_high"), ("stratified_auc", "stratified_auc"))},
        })
        for row in diagnostics:
            duration_rows.append({"representation": name, **row})
        for row in contribution_rows(z, features, {"usable": usable, "excluded": groups["excluded_all"] & eligible}):
            contributions.append({"representation": name, **row})
        labels, counts = np.unique(sessions[usable], return_counts=True)
        for session in labels[counts >= MIN_GROUP_SIZE]:
            inside = usable & (sessions == session).to_numpy()
            session_rows.append({"representation": name, "session": session, "n": int(inside.sum()),
                                 "median_rms_z": float(np.median(scores[inside])),
                                 "cliffs_delta_vs_other_usable": cliffs_delta(scores[inside], scores[usable & ~inside])})
    comparison = pd.DataFrame(comparison_rows)
    pd.DataFrame(duration_rows).to_csv(output / "excluded_diagnostics.csv", index=False)
    pd.DataFrame(contributions).to_csv(output / "feature_contributions.csv", index=False)
    pd.DataFrame(session_rows).to_csv(output / "session_robustness.csv", index=False)

    robustness = feature_robustness(frame, z, sessions, groups, union)

    perturbation = pd.DataFrame()
    feature_perturbation = pd.DataFrame()
    stage6_check = None
    if run_perturbations:
        measured = perturbation_measurements(table, data_dir)
        measured["session"] = measured["row"].map(sessions)
        z_parts = []
        for session, block in measured.groupby("session"):
            z_parts.append(baselines[session].z_scores(block[union]))   # frozen fold baseline of the event's session
        measured = pd.concat([measured, pd.concat(z_parts).loc[measured.index]], axis=1)
        measured.to_csv(output / "perturbed_features.csv", index=False, float_format="%.6g")
        perturbation = perturbation_response(measured, representations)
        perturbation.to_csv(output / "perturbation_response.csv", index=False)
        feature_perturbation = feature_perturbation_table(measured, union)
        feature_perturbation.to_csv(output / "feature_perturbation.csv", index=False)
        if STAGE6_PERTURBED.exists():
            stage6 = pd.read_csv(STAGE6_PERTURBED)
            joined = measured.merge(stage6[["row", "transform", "magnitude", *v1]], on=["row", "transform"],
                                    suffixes=("", "_s6"))
            joined = joined[np.isclose(joined["magnitude"], joined["magnitude_s6"])]
            stage6_check = float(max((np.abs(joined[f] - joined[f"{f}_s6"]) / np.abs(joined[f"{f}_s6"]).clip(lower=1e-12)).max()
                                     for f in v1 if f != "duration_s"))
        for noise_db in (30, 20, 10):
            block = perturbation[(perturbation["transform"] == "noise") & np.isclose(perturbation["magnitude"], noise_db)]
            comparison = comparison.merge(block[["representation", "median_delta_rms_z"]].rename(
                columns={"median_delta_rms_z": f"delta_rms_z_noise{noise_db}db"}), on="representation")
        for transform, magnitude, label in (("gain", 2.0, "event_gain_x2"), ("recording_gain", 2.0, "recording_gain_x2"),
                                            ("tilt", 0.9, "tilt_0.9")):
            block = perturbation[(perturbation["transform"] == transform) & np.isclose(perturbation["magnitude"], magnitude)]
            comparison = comparison.merge(block[["representation", "median_delta_rms_z", "median_single_feature_dominance"]].rename(
                columns={"median_delta_rms_z": f"delta_rms_z_{label}",
                         "median_single_feature_dominance": f"dominance_{label}"}), on="representation")
        feature_noise = feature_perturbation[feature_perturbation["transform"] == "noise"].set_index("feature")
        robustness = robustness.merge(feature_noise[["monotone_fraction", "median_delta_z@30", "median_delta_z@20",
                                                     "median_delta_z@10"]].add_prefix("noise_"),
                                      left_on="feature", right_index=True, how="left")
        gains = feature_perturbation[feature_perturbation["transform"] == "recording_gain"].set_index("feature")
        robustness = robustness.merge(gains[["median_delta_z@2", "max_abs_delta_z"]].rename(
            columns={"median_delta_z@2": "recording_gain_x2_delta_z", "max_abs_delta_z": "recording_gain_max_abs_delta_z"}),
            left_on="feature", right_index=True, how="left")
        tilts = feature_perturbation[feature_perturbation["transform"] == "tilt"].set_index("feature")
        robustness = robustness.merge(tilts[["monotone_fraction"]].rename(columns={"monotone_fraction": "tilt_monotone_fraction"}),
                                      left_on="feature", right_index=True, how="left")
    comparison.to_csv(output / "representation_comparison.csv", index=False)
    robustness.to_csv(output / "feature_robustness.csv", index=False)

    # Ablation views.
    duration_view = comparison[comparison["representation"].isin(["R0_current7", "R1_no_duration", "R2_duration_as_usability"])]
    duration_view.to_csv(output / "duration_ablation.csv", index=False)
    flatness_view = comparison[comparison["representation"].isin(["R0_current7", "R3_no_flatness_std", "R4_ew_flatness_std"])]
    flatness_view.to_csv(output / "flatness_std_ablation.csv", index=False)

    # Multivariate: diagnostics first, comparison only for representations that pass the gate.
    diagnostics, mcd_scores, mcd_references, models = mcd_loso(frame, z, sessions, baselines, representations)
    diagnostics.to_csv(output / "multivariate_diagnostics.csv", index=False)
    gate = multivariate_gate(diagnostics)
    multivariate_rows = []
    for name, features in representations.items():
        if not gate[name]["passed"]:
            continue
        values = mcd_scores[name]
        eligible = scoreable_mask(frame, name)
        reference = np.array([mcd_references[name][s] for s in sessions])
        by_group = {row["group"]: row for row in excluded_diagnostics(values, sessions, groups, eligible)}
        rms = scores_by_rep[name]
        multivariate_rows.append({
            "representation": name, "score": "robust_mahalanobis_sqrt_d2_over_d",
            "usable_median": float(np.median(values[usable])),
            "generalization_ratio": float(np.median(values[usable]) / np.median(reference[usable])),
            **session_dependence(values, sessions, usable),
            "spearman_with_rms_z_usable": float(stats.spearmanr(values[usable], rms[usable]).statistic),
            **{f"{g}_delta": by_group[g].get("cliffs_delta", np.nan) for g in ("excluded_all", "only_close_neighbor")},
            **{f"{g}_stratified_auc": by_group[g].get("stratified_auc", np.nan) for g in ("excluded_all", "only_close_neighbor")},
        })
    pd.DataFrame(multivariate_rows).to_csv(output / "multivariate_comparison.csv", index=False)

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        if run_perturbations:
            plot_representation_overview(comparison, perturbation, output / "representation_overview.png")
            plot_feature_robustness(robustness, feature_perturbation, output / "feature_robustness.png")
        plot_duration_ablation(scores_by_rep, groups, output / "duration_ablation.png")

    hashes_after = {path: _sha256(Path(path)) for path in hashes_before}
    result = {
        "stage": "Stage 7 - representation robustness and ablation (no threshold, no labels)",
        "v1_features": v1,
        "representations": {name: {"features": f, "description": DESCRIPTIONS[name]} for name, f in representations.items()},
        "candidate_features": list(CANDIDATE_FEATURES),
        "baseline": "LOSO global median / 1.4826*MAD fitted on usable events of other sessions only (Stage 5 C)",
        "population_sizes": {name: int(mask.sum()) for name, mask in groups.items()},
        "leakage_and_reproducibility": {**leakage, "max_relative_difference_vs_stage6_perturbed_features": stage6_check},
        "multivariate_gate": {"rules": {"min_events_per_feature": MCD_MIN_EVENTS_PER_FEATURE,
                                        "max_condition_number": MCD_MAX_CONDITION_NUMBER,
                                        "max_correlation_shift_vs_pooled": MCD_MAX_CORRELATION_SHIFT},
                              "results": gate},
        "inputs_unchanged": hashes_before == hashes_after,
        "input_sha256": {_relative(Path(p)): h for p, h in hashes_before.items()},
        "parameters": {"seed": SEED, "stage6_noise_seed": STAGE6_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
                       "recording_gains": list(RECORDING_GAINS), "duration_gate_s": DURATION_GATE_S},
        "provenance": {"git": _git_state()},
        "no_threshold_or_labels": True,
    }
    write_recommendation(output, result)
    result["outputs"] = sorted(str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*") if p.is_file())
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


# Written after the analysis was run (Entry 9).  The quantitative evidence is re-read
# from the saved tables every time write_recommendation runs; nothing here is adopted
# into the V1 pipeline, and no threshold is implied.
STAGE7_DECISION: dict = {
    "decision": "B_modify_representation",
    "next_bottleneck": "C_new_longitudinal_or_ground_truth_data",
    "summary": (
        "The current 7-feature representation is not defensible as-is: duration mainly encodes "
        "segmentation, spectral_flatness_std is fragile, and absolute mean_rms is the most "
        "session-dependent feature and fully device-gain dependent.  Session dependence remains "
        "substantial in every tested representation, which these data cannot resolve."
    ),
    "feature_decisions": {
        "duration_s": "remove from the anomaly score; keep as a segmentation / usability attribute (Stage 1 gate)",
        "spectral_flatness_std": "remove (noise-, level- and tilt-fragile; removal costs no measured separation)",
        "mean_rms": "remove from the global score; report separately as an uncalibrated level channel",
        "spectral_centroid_mean": "keep; predictable responses, gain-invariant, but strongly session-dependent (flagged)",
        "spectral_flatness_mean": "keep; predictable under noise, non-monotone under tilt, session-dependent (flagged)",
        "spectral_centroid_std": "keep; session-stable, weakly responsive",
        "spectral_rolloff_std": "keep; session-stable, weakly responsive",
    },
    "not_adopted": {
        "energy_weighted_spectral_features": "reduce noise sensitivity but increase session dependence (R7)",
        "ew_spectral_flatness_std": "fixes noise/level fragility, but session dependence rises and removal is simpler",
        "relative_level_db": "not an event-level measure: in usable events Spearman -0.99 with the recording "
                             "background RMS and -0.08 with the event's own mean_rms",
        "robust_multivariate_distance": "pre-specified stability gate failed for every representation",
    },
    "proposed_v2_candidate_for_preregistration": [
        "spectral_centroid_mean", "spectral_centroid_std", "spectral_flatness_mean", "spectral_rolloff_std",
    ],
    "proposed_v2_status": "NOT evaluated as a combination in Stage 7; must be pre-registered and evaluated next",
}


def write_recommendation(output: Path, summary: Mapping) -> None:
    comparison = pd.read_csv(output / "representation_comparison.csv")
    evidence = comparison.set_index("representation")
    payload = {
        "decision": STAGE7_DECISION or {"status": "not yet decided"},
        "evidence": {
            name: {k: evidence.loc[name, k] for k in evidence.columns
                   if k not in ("description", "features") and pd.notna(evidence.loc[name, k])}
            for name in evidence.index
        },
        "multivariate_gate": summary["multivariate_gate"],
        "no_threshold_or_labels": True,
    }
    with (output / "recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7 representation robustness and ablation (no threshold)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-perturbations", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_analysis(output_dir=args.output_dir, run_perturbations=not args.skip_perturbations,
                           make_plots=not args.no_plots)
    print(json.dumps({k: outcome[k] for k in ("leakage_and_reproducibility", "inputs_unchanged", "multivariate_gate")},
                     indent=2, default=str))
