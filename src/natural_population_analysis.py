"""Stage 6: natural population separation and controlled sensitivity.

See PRISM_RESEARCH_LOG.md, Entry 8.

Question: does the frozen leave-one-session-out (LOSO) global baseline (Stage 5,
strategy C) give systematically different scores to events that Stage 1
excluded (too short, close neighbour, recording boundary) than to usable
events?  Excluded is NOT treated as anomalous: this is a natural atypical /
out-of-distribution population experiment, reported with session controls
and feature attribution.

Baseline for every event in session s: per-feature median / 1.4826*MAD of the
USABLE events of all other sessions.  Excluded events never enter a fit.

Recordings without a detected inhalation have no event to score; they are
described with window-level quantities of the existing CNN detector only.

A secondary controlled experiment perturbs the waveform of usable events
(gain, additive white noise, spectral tilt; magnitudes fixed in advance),
re-runs the existing measurement code on the same event boundaries, and scores
the result with the same frozen baselines.

No threshold is chosen and no event is labelled NORMAL or ANOMALY.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats

import config
from baseline_strategies import LEVEL_FEATURES
from baseline_v1 import (
    DEFAULT_SELECTION_JSON,
    BaselineError,
    fit_baseline,
    load_feature_selection,
    recording_sessions,
)
from feature_analysis import DEFAULT_DATASET_CSV, MIN_GROUP_SIZE, exact_event_bounds, load_stage1_table
from inhale_dataset import _git_state, _json_default, _relative, _sha256
from scoring_v1 import SCORES, combined_scores, feature_contributions


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "natural_population"
STAGE1_AUDIT = Path(config.RESULTS_DIR) / "inhale_dataset" / "event_annotation_audit.csv"
STAGE5_HELDOUT = Path(config.RESULTS_DIR) / "baseline_strategies" / "heldout_scores.csv"

FLAG_COLUMNS = {
    "too_short": "flag_short_duration",
    "close_neighbor": "flag_close_neighbor",
    "recording_boundary": "flag_recording_boundary",
    "nonfinite_feature": "flag_nonfinite_feature",
}
REASON_NAMES = {"short_duration": "too_short", "close_neighbor": "close_neighbor",
                "recording_boundary": "recording_boundary", "nonfinite_feature": "nonfinite_feature"}
BOOTSTRAP_DRAWS = 2000
PERMUTATIONS = 2000
SEED = 20261003

# Controlled perturbations: fixed before any Stage 6 result was computed.
GAINS = (0.5, 1 / np.sqrt(2), np.sqrt(2), 2.0)       # -6, -3, +3, +6 dB
SNR_DB = (30.0, 20.0, 10.0)                            # white noise relative to the event's own power
TILTS = (-0.5, 0.5, 0.9)                               # y[n] = x[n] - a*x[n-1], RMS restored afterwards
PERTURBATION_ORDER = {                                 # increasing intensity, identity included
    "gain": [0.5, 1 / np.sqrt(2), 1.0, np.sqrt(2), 2.0],
    "noise": [np.inf, 30.0, 20.0, 10.0],
    "tilt": [-0.5, 0.0, 0.5, 0.9],
}


class PopulationError(ValueError):
    """Populations cannot be constructed or scored safely."""


# ── Populations (Stage 1 definitions, never recomputed) ─────────────────────

def assign_populations(table: pd.DataFrame) -> pd.DataFrame:
    """Membership columns from the Stage 1 flags; overlaps are preserved.

    ``in_<reason>`` are overlapping memberships; ``exclusion_group`` is the exact
    Stage 1 reason combination (mutually exclusive), e.g. ``too_short+close_neighbor``.
    """
    required = ["usable", "exclusion_reasons", *FLAG_COLUMNS.values()]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise PopulationError(f"Stage 1 columns missing: {missing}")
    result = pd.DataFrame(index=table.index)
    flags = table[list(FLAG_COLUMNS.values())].astype(bool)
    usable = table["usable"].astype(bool)
    if (usable & flags.any(axis=1)).any() or (~usable & ~flags.any(axis=1)).any():
        raise PopulationError("Stage 1 'usable' disagrees with its exclusion flags")
    for name, column in FLAG_COLUMNS.items():
        result[f"in_{name}"] = flags[column].to_numpy()
    reasons = table["exclusion_reasons"].fillna("").astype(str)
    groups = []
    for is_usable, text in zip(usable, reasons):
        if is_usable:
            groups.append("usable")
        else:
            groups.append("+".join(REASON_NAMES[r] for r in text.split(";") if r))
    result["population"] = np.where(usable, "usable", "excluded")
    result["exclusion_group"] = groups
    # Consistency: the reason string must agree with the flags.
    for name in FLAG_COLUMNS:
        from_text = result["exclusion_group"].str.split("+").apply(lambda parts, n=name: n in parts)
        if not (from_text == result[f"in_{name}"]).all():
            raise PopulationError(f"exclusion_reasons disagree with flag for {name}")
    return result


def overlap_matrix(populations: pd.DataFrame) -> pd.DataFrame:
    """Pairwise co-membership counts of the overlapping exclusion categories."""
    columns = [f"in_{n}" for n in FLAG_COLUMNS]
    values = populations[columns].astype(int)
    matrix = values.T @ values
    matrix.index = matrix.columns = list(FLAG_COLUMNS)
    return matrix


def comparison_groups(populations: pd.DataFrame) -> dict[str, np.ndarray]:
    """Named boolean masks: overlapping reason groups and exact combinations."""
    groups = {
        "usable": (populations["population"] == "usable").to_numpy(),
        "excluded_all": (populations["population"] == "excluded").to_numpy(),
    }
    for name in ("too_short", "close_neighbor", "recording_boundary"):
        groups[f"any_{name}"] = populations[f"in_{name}"].to_numpy(bool)
    for combination in sorted(set(populations["exclusion_group"]) - {"usable"}):
        groups[f"only_{combination}"] = (populations["exclusion_group"] == combination).to_numpy()
    return groups


# ── Held-out scoring with the frozen LOSO baseline ─────────────────────────

def heldout_scores(
    table: pd.DataFrame, features: Sequence[str], sessions: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """z for every event, from a baseline fitted on usable events of OTHER sessions only.

    Returns z (indexed like ``table``), fold parameters, and the fitted baselines per session.
    """
    usable = table["usable"].astype(bool).to_numpy()
    session_values = sessions.to_numpy()
    parts, parameters, baselines = [], [], {}
    for session in sorted(np.unique(session_values)):
        training_mask = usable & (session_values != session)
        training = table[training_mask]
        if len(training) == 0:
            raise BaselineError(f"no usable training events outside session {session}")
        baseline = fit_baseline(training, features)
        baselines[session] = baseline
        targets = table[session_values == session]
        parts.append(baseline.z_scores(targets))
        for feature in features:
            parameters.append({"fold": session, "feature": feature, "n_train": int(training_mask.sum()),
                               "n_scored": int(len(targets)),
                               "n_scored_excluded": int((~usable & (session_values == session)).sum()),
                               "center": baseline.parameters[feature].median,
                               "scale": baseline.parameters[feature].scale})
    return pd.concat(parts).loc[table.index], pd.DataFrame(parameters), baselines


# ── Effect sizes ────────────────────────────────────────────────────────────

def mann_whitney_u(a, b) -> float:
    """U statistic of ``a`` over ``b`` (ties count 1/2)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ranks = stats.rankdata(np.concatenate([a, b]))
    return float(ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2)


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b); positive when ``a`` tends to be larger."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return 2 * mann_whitney_u(a, b) / (len(a) * len(b)) - 1


def hodges_lehmann_shift(a, b) -> float:
    """Median of all pairwise differences a_i - b_j."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.median(np.subtract.outer(a, b))) if len(a) and len(b) else float("nan")


def overlap_measures(group, reference) -> dict:
    group, reference = np.asarray(group, dtype=float), np.asarray(reference, dtype=float)
    low, median, high = np.percentile(reference, [5, 50, 95])
    return {
        "frac_group_above_reference_median": float(np.mean(group > median)),
        "frac_group_within_reference_p05_p95": float(np.mean((group >= low) & (group <= high))),
        "frac_group_above_reference_p95": float(np.mean(group > high)),
    }


def session_cluster_bootstrap_delta(
    group, group_sessions, reference, reference_sessions, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED,
) -> tuple[float, float]:
    """95% interval of Cliff's delta resampling whole sessions with replacement."""
    group, reference = np.asarray(group, dtype=float), np.asarray(reference, dtype=float)
    group_sessions, reference_sessions = np.asarray(group_sessions), np.asarray(reference_sessions)
    labels = np.unique(np.concatenate([group_sessions, reference_sessions]))
    by_group = {s: group[group_sessions == s] for s in labels}
    by_reference = {s: reference[reference_sessions == s] for s in labels}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        chosen = rng.choice(labels, size=len(labels), replace=True)
        a = np.concatenate([by_group[s] for s in chosen])
        b = np.concatenate([by_reference[s] for s in chosen])
        if len(a) and len(b):
            deltas.append(cliffs_delta(a, b))
    if not deltas:
        return float("nan"), float("nan")
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def within_session_rows(scores, sessions, group_mask, reference_mask) -> pd.DataFrame:
    """Per session with both populations: medians, difference, ratio, within-session AUC."""
    scores, sessions = np.asarray(scores, dtype=float), np.asarray(sessions)
    group_mask, reference_mask = np.asarray(group_mask, bool), np.asarray(reference_mask, bool)
    rows = []
    for session in sorted(np.unique(sessions)):
        in_session = sessions == session
        a, b = scores[in_session & group_mask], scores[in_session & reference_mask]
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append({"session": session, "n_group": int(len(a)), "n_reference": int(len(b)),
                     "median_group": float(np.median(a)), "median_reference": float(np.median(b)),
                     "difference": float(np.median(a) - np.median(b)),
                     "ratio": float(np.median(a) / np.median(b)),
                     "auc": mann_whitney_u(a, b) / (len(a) * len(b))})
    return pd.DataFrame(rows, columns=["session", "n_group", "n_reference", "median_group", "median_reference",
                                       "difference", "ratio", "auc"])


def _session_blocks(scores, sessions, group_mask, reference_mask) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per session containing both populations: ranks of its involved events and their group labels."""
    scores, sessions = np.asarray(scores, dtype=float), np.asarray(sessions)
    group_mask, reference_mask = np.asarray(group_mask, bool), np.asarray(reference_mask, bool)
    involved = group_mask | reference_mask
    blocks = []
    for session in np.unique(sessions[involved]):
        index = np.flatnonzero((sessions == session) & involved)
        labels = group_mask[index]
        if labels.any() and (~labels).any():
            blocks.append((stats.rankdata(scores[index]), labels))
    return blocks


def _stratified_auc_from_blocks(blocks, label_sets) -> float:
    numerator = denominator = 0.0
    for (ranks, _), labels in zip(blocks, label_sets):
        n_group = int(labels.sum())
        numerator += ranks[labels].sum() - n_group * (n_group + 1) / 2
        denominator += n_group * (len(labels) - n_group)
    return numerator / denominator if denominator else float("nan")


def stratified_auc(scores, sessions, group_mask, reference_mask) -> float:
    """Within-session AUC pooled over sessions: sum U_s / sum n_group,s * n_reference,s."""
    blocks = _session_blocks(scores, sessions, group_mask, reference_mask)
    return _stratified_auc_from_blocks(blocks, [labels for _, labels in blocks])


def within_session_percentiles(scores, sessions, group_mask, reference_mask) -> np.ndarray:
    """For each group event: share of same-session reference events with a lower score (ties 1/2)."""
    scores, sessions = np.asarray(scores, dtype=float), np.asarray(sessions)
    reference_mask = np.asarray(reference_mask, bool)
    values = []
    for index in np.flatnonzero(np.asarray(group_mask, bool)):
        reference = scores[(sessions == sessions[index]) & reference_mask]
        if len(reference):
            values.append(float(np.mean(reference < scores[index]) + 0.5 * np.mean(reference == scores[index])))
    return np.array(values)


def stratified_permutation_p(
    scores, sessions, group_mask, reference_mask, permutations: int = PERMUTATIONS, seed: int = SEED,
) -> float:
    """Two-sided p for the stratified AUC, permuting labels within sessions only."""
    blocks = _session_blocks(scores, sessions, group_mask, reference_mask)
    if not blocks:
        return float("nan")
    observed = _stratified_auc_from_blocks(blocks, [labels for _, labels in blocks])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        value = _stratified_auc_from_blocks(blocks, [rng.permutation(labels) for _, labels in blocks])
        extreme += abs(value - 0.5) >= abs(observed - 0.5) - 1e-12
    return (1 + extreme) / (1 + permutations)


def distribution_row(values) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0}
    p05, p25, p50, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "median": float(p50), "iqr": float(p75 - p25), "p05": float(p05), "p25": float(p25),
            "p75": float(p75), "p95": float(p95), "mean": float(x.mean()), "max": float(x.max())}


def compare_to_reference(
    frame: pd.DataFrame, groups: Mapping[str, np.ndarray], reference: str = "usable",
    min_group: int = 3, draws: int = BOOTSTRAP_DRAWS, permutations: int = PERMUTATIONS, seed: int = SEED,
) -> pd.DataFrame:
    """Pooled and session-controlled effect sizes of each group against the reference group."""
    reference_mask = groups[reference]
    sessions = frame["session"].reset_index(drop=True)
    rows = []
    for name, mask in groups.items():
        if name == reference or mask.sum() < min_group:
            continue
        for score in SCORES:
            values = frame[score].reset_index(drop=True)
            a, b = values[mask], values[reference_mask]
            low, high = session_cluster_bootstrap_delta(a, sessions[mask], b, sessions[reference_mask], draws, seed)
            percentiles = within_session_percentiles(values, sessions, mask, reference_mask)
            within = within_session_rows(values, sessions, mask, reference_mask)
            rows.append({
                "group": name, "score": score, "n_group": int(mask.sum()), "n_reference": int(reference_mask.sum()),
                "median_group": float(a.median()), "median_reference": float(b.median()),
                "median_difference": float(a.median() - b.median()),
                "median_ratio": float(a.median() / b.median()),
                "hodges_lehmann_shift": hodges_lehmann_shift(a, b),
                "cliffs_delta": cliffs_delta(a, b), "auc": (cliffs_delta(a, b) + 1) / 2,
                "cliffs_delta_session_bootstrap_low": low, "cliffs_delta_session_bootstrap_high": high,
                **overlap_measures(a, b),
                "sessions_with_both": int(len(within)),
                "stratified_auc": stratified_auc(values, sessions, mask, reference_mask),
                "stratified_permutation_p": stratified_permutation_p(values, sessions, mask, reference_mask,
                                                                     permutations, seed),
                "within_session_median_difference_median": float(within["difference"].median()) if len(within) else np.nan,
                "sessions_group_median_higher": int((within["difference"] > 0).sum()) if len(within) else 0,
                "median_within_session_percentile": float(np.median(percentiles)) if len(percentiles) else np.nan,
                "frac_within_session_percentile_above_0_9": float(np.mean(percentiles > 0.9)) if len(percentiles) else np.nan,
            })
    return pd.DataFrame(rows)


# ── Feature attribution ─────────────────────────────────────────────────────

def feature_attribution(
    z: pd.DataFrame, groups: Mapping[str, np.ndarray], features: Sequence[str], reference: str = "usable",
) -> pd.DataFrame:
    """Per group and feature: size, direction, score shares, argmax share and univariate separation.

    ``univariate_auc_abs_z``: AUC of |z_j| for the group vs the reference.
    ``rms_z_auc_without_feature``: AUC of rms_z recomputed without feature j (drop-one
    attribution; never used to select features).
    """
    contributions = feature_contributions(z, features)
    reference_mask = groups[reference]
    rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        for feature in features:
            others = [f for f in features if f != feature]
            without = combined_scores(z[[f"z_{f}" for f in others]], others)["rms_z"].to_numpy()
            absolute = np.abs(z[f"z_{feature}"].to_numpy())
            rows.append({
                "group": name, "feature": feature,
                "family": "level" if feature in LEVEL_FEATURES else "within-event variability",
                "n": int(mask.sum()),
                "median_z": float(np.median(z[f"z_{feature}"].to_numpy()[mask])),
                "median_abs_z": float(np.median(absolute[mask])),
                "mean_abs_z_share": float(contributions[f"abs_share_{feature}"].to_numpy()[mask].mean()),
                "rms_z_share": float(np.nanmean(contributions[f"rms_share_{feature}"].to_numpy()[mask])),
                "argmax_fraction": float(contributions[f"is_max_{feature}"].to_numpy()[mask].mean()),
                "univariate_auc_abs_z": (cliffs_delta(absolute[mask], absolute[reference_mask]) + 1) / 2
                if name != reference else np.nan,
                "rms_z_auc_without_feature": (cliffs_delta(without[mask], without[reference_mask]) + 1) / 2
                if name != reference else np.nan,
            })
    return pd.DataFrame(rows)


MATCHES_PER_EVENT = 5   # duration matching: nearest usable events per excluded event (fixed in advance)


def duration_matched_comparison(
    frame: pd.DataFrame, z: pd.DataFrame, group_mask, reference_mask, features: Sequence[str],
    k: int = MATCHES_PER_EVENT,
) -> tuple[pd.DataFrame, dict]:
    """Is there deviation beyond duration? Compare rms_z over the non-duration features.

    Each group event is paired with its k nearest-duration reference events (any
    session); both sides use their own held-out z-scores.  ``max_duration_gap_s``
    shows how well the matching worked.
    """
    others = [f for f in features if f != "duration_s"]
    without_duration = combined_scores(z[[f"z_{f}" for f in others]], others)["rms_z"].to_numpy()
    durations = frame["duration_s"].to_numpy(dtype=float)
    reference_index = np.flatnonzero(np.asarray(reference_mask, bool))
    rows, matched = [], set()
    for index in np.flatnonzero(np.asarray(group_mask, bool)):
        gaps = np.abs(durations[reference_index] - durations[index])
        nearest = reference_index[np.argsort(gaps, kind="mergesort")[:k]]
        matched.update(nearest.tolist())
        rows.append({"row": int(index), "duration_s": durations[index],
                     "matched_duration_median_s": float(np.median(durations[nearest])),
                     "max_duration_gap_s": float(np.abs(durations[nearest] - durations[index]).max()),
                     "rms_z_without_duration": float(without_duration[index]),
                     "matched_rms_z_without_duration_median": float(np.median(without_duration[nearest])),
                     "paired_difference": float(without_duration[index] - np.median(without_duration[nearest]))})
    per_event = pd.DataFrame(rows)
    if per_event.empty:
        return per_event, {"n": 0}
    matched_values = without_duration[sorted(matched)]
    summary = {
        "n": int(len(per_event)), "k": k,
        "median_duration_gap_s": float(per_event["max_duration_gap_s"].median()),
        "median_paired_difference": float(per_event["paired_difference"].median()),
        "frac_paired_difference_positive": float(np.mean(per_event["paired_difference"] > 0)),
        "cliffs_delta_vs_matched": cliffs_delta(per_event["rms_z_without_duration"], matched_values),
        "cliffs_delta_vs_all_reference": cliffs_delta(per_event["rms_z_without_duration"],
                                                      without_duration[reference_index]),
    }
    return per_event, summary


def session_variation(frame: pd.DataFrame, min_events: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Each usable session vs all other usable events: the ordinary cross-session shift."""
    usable = frame[frame["population"] == "usable"]
    counts = usable["session"].value_counts()
    rows = []
    for session in sorted(counts[counts >= min_events].index):
        inside = usable["session"] == session
        for score in SCORES:
            a, b = usable.loc[inside, score], usable.loc[~inside, score]
            rows.append({"session": session, "score": score, "n": int(inside.sum()),
                         "median_ratio_to_other_usable": float(a.median() / b.median()),
                         "cliffs_delta_vs_other_usable": cliffs_delta(a, b)})
    return pd.DataFrame(rows)


# ── Recording-level quantities for recordings without a usable event ──────

def recording_window_statistics(recording_paths: Sequence[Path], classifier=None) -> pd.DataFrame:
    """Window-level quantities of the EXISTING CNN detector, one row per recording."""
    from librosa_extractor import load_audio
    from post_event import (OnnxEventClassifier, _rms_envelope, generate_window_predictions,
                            group_inhale_events)

    classifier = classifier or OnnxEventClassifier()
    stride_s = config.WINDOW_STRIDE * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR
    rows = []
    for path in recording_paths:
        waveform, sample_rate = load_audio(str(path))
        predictions = generate_window_predictions(waveform, sample_rate, model=classifier)
        labels = np.array([p.label for p in predictions])
        inhale_probability = np.array([p.probabilities["Inhale"] for p in predictions])
        candidates = group_inhale_events(predictions)
        _, envelope = _rms_envelope(waveform, sample_rate)
        row = {"recording_file": Path(path).name, "recording_duration_s": len(waveform) / sample_rate,
               "n_windows": int(len(predictions)),
               "n_inhale_candidates": int(len(candidates)),
               "longest_inhale_candidate_s": float(max((c.duration for c in candidates), default=0.0)),
               "inhale_window_time_s": float((labels == "Inhale").sum() * stride_s),
               "mean_p_inhale": float(inhale_probability.mean()) if len(predictions) else np.nan,
               "p95_p_inhale": float(np.percentile(inhale_probability, 95)) if len(predictions) else np.nan,
               "max_p_inhale": float(inhale_probability.max()) if len(predictions) else np.nan,
               "recording_mean_rms": float(envelope.mean()), "recording_p95_rms": float(np.percentile(envelope, 95))}
        for label in config.LABEL_NAMES:
            row[f"frac_windows_{label.lower()}"] = float(np.mean(labels == label)) if len(predictions) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


RECORDING_QUANTITIES = ("n_inhale_candidates", "inhale_window_time_s", "frac_windows_inhale", "mean_p_inhale",
                        "max_p_inhale", "frac_windows_exhale", "frac_windows_drug", "frac_windows_noise",
                        "recording_mean_rms", "recording_p95_rms")


def recording_groups(recordings: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """usable_event / only_excluded_events / no_detected_event per recording."""
    with_events = set(events["recording_file"])
    with_usable = set(events.loc[events["usable"].astype(bool), "recording_file"])
    return recordings["recording_file"].map(
        lambda r: "usable_event" if r in with_usable else ("only_excluded_events" if r in with_events else "no_detected_event"))


def summarize_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = recordings[recordings["recording_group"] == "usable_event"]
    for group, block in recordings.groupby("recording_group"):
        for quantity in RECORDING_QUANTITIES:
            row = {"recording_group": group, "quantity": quantity, **distribution_row(block[quantity])}
            if group != "usable_event":
                row["cliffs_delta_vs_usable_event_recordings"] = cliffs_delta(block[quantity], reference[quantity])
            rows.append(row)
    return pd.DataFrame(rows)


# ── Controlled waveform perturbations ───────────────────────────────────────

def apply_gain(segment: np.ndarray, gain: float) -> np.ndarray:
    return np.asarray(segment, dtype=np.float64) * gain


def add_white_noise(segment: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at ``snr_db`` relative to the segment's own mean power."""
    x = np.asarray(segment, dtype=np.float64)
    power = np.mean(x ** 2)
    noise = rng.standard_normal(len(x)) * np.sqrt(power / (10 ** (snr_db / 10)))
    return x + noise


def apply_tilt(segment: np.ndarray, a: float) -> np.ndarray:
    """First-order FIR y[n] = x[n] - a x[n-1], rescaled to the original RMS.

    a > 0 emphasises high frequencies, a < 0 low frequencies; level is preserved so
    that only spectral shape changes.
    """
    x = np.asarray(segment, dtype=np.float64)
    y = signal.lfilter([1.0, -a], [1.0], x)
    rms_x, rms_y = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(y ** 2))
    return y * (rms_x / rms_y) if rms_y > 0 else y


def perturbation_plan() -> list[tuple[str, float]]:
    plan = [("identity", 0.0)]
    plan += [("gain", g) for g in GAINS]
    plan += [("noise", s) for s in SNR_DB]
    plan += [("tilt", a) for a in TILTS]
    return plan


def perturb(segment: np.ndarray, transform: str, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    if transform == "identity":
        return np.asarray(segment, dtype=np.float64).copy()
    if transform == "gain":
        return apply_gain(segment, magnitude)
    if transform == "noise":
        return add_white_noise(segment, magnitude, rng)
    if transform == "tilt":
        return apply_tilt(segment, magnitude)
    raise ValueError(f"unknown transform {transform!r}")


def measure_segment(segment: np.ndarray, sample_rate: int) -> dict:
    """The existing post_event measurements of a whole segment (event = the segment)."""
    from post_event import InhaleEvent, analyze_inhalation

    duration = len(segment) / sample_rate
    analysis = analyze_inhalation(segment.astype(np.float32), InhaleEvent("Inhale", 0.0, duration, duration, 0, 0, 0, ()),
                                  sample_rate=sample_rate)
    spectral = analysis["spectral"]
    return {"mean_rms": analysis["mean_rms"],
            **{f"{name}_{stat}": spectral[name][stat] for name in spectral for stat in ("mean", "std")}}


def controlled_perturbations(
    table: pd.DataFrame, features: Sequence[str], sessions: pd.Series, baselines: Mapping,
    data_dir: str | Path = config.DATA_DIR, seed: int = SEED,
) -> pd.DataFrame:
    """Perturb each usable event's waveform, re-measure, score with its frozen session baseline."""
    from librosa_extractor import load_audio
    from post_event import InhaleEvent, extract_event_audio

    rows = []
    usable = table[table["usable"].astype(bool)]
    for recording, group in usable.groupby("recording_file", sort=False):
        waveform, sample_rate = load_audio(str(Path(data_dir) / recording))
        for index, event in group.iterrows():
            start, end = exact_event_bounds(event["start_s"], event["end_s"], event["recording_duration_s"])
            segment, _ = extract_event_audio(waveform, InhaleEvent("Inhale", start, end, end - start, 0, 0, 0, ()),
                                             sample_rate=sample_rate)
            original = segment.copy()
            baseline = baselines[sessions.loc[index]]
            for transform, magnitude in perturbation_plan():
                rng = np.random.default_rng([seed, int(index)])
                measured = measure_segment(perturb(segment, transform, magnitude, rng), sample_rate)
                measured["duration_s"] = event["duration_s"]          # boundaries are unchanged
                values = pd.DataFrame([{f: measured[f] for f in features}])
                z = baseline.z_scores(values).iloc[0]
                rows.append({"row": index, "recording_file": recording, "event_id": event["event_id"],
                             "session": sessions.loc[index], "transform": transform, "magnitude": magnitude,
                             **{f: measured[f] for f in features}, **z.to_dict()})
            if not np.array_equal(segment, original):
                raise PopulationError("a perturbation modified the original segment in place")
    frame = pd.DataFrame(rows)
    scores = combined_scores(frame[[f"z_{f}" for f in features]], features)
    return pd.concat([frame, scores[list(SCORES)]], axis=1)


def summarize_perturbations(perturbed: pd.DataFrame, features: Sequence[str],
                            min_session_events: int = MIN_GROUP_SIZE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature/score change vs identity, monotonicity along intensity, cross-session spread."""
    identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
    columns = [f"z_{f}" for f in features]
    change_rows, session_rows = [], []
    for (transform, magnitude), block in perturbed[perturbed["transform"] != "identity"].groupby(["transform", "magnitude"]):
        block = block.set_index("row")
        base = identity.loc[block.index]
        delta_z = block[columns] - base[columns]
        distance = np.sqrt((delta_z ** 2).mean(axis=1))
        row = {"transform": transform, "magnitude": magnitude, "n_events": int(len(block)),
               "median_distance_from_own_z": float(distance.median())}
        for f in features:
            row[f"median_delta_z_{f}"] = float(delta_z[f"z_{f}"].median())
        for score in SCORES:
            change = block[score] - base[score]
            row[f"median_delta_{score}"] = float(change.median())
            row[f"frac_increase_{score}"] = float(np.mean(change > 0))
        change_rows.append(row)
        counts = block["session"].value_counts()
        for session in sorted(counts[counts >= min_session_events].index):
            inside = (block["session"] == session).to_numpy()
            session_rows.append({"transform": transform, "magnitude": magnitude, "session": session,
                                 "n": int(inside.sum()),
                                 "median_distance_from_own_z": float(distance[inside].median()),
                                 "median_delta_rms_z": float((block["rms_z"] - base["rms_z"])[inside].median()),
                                 **{f"median_delta_z_{f}": float(delta_z[f"z_{f}"][inside].median()) for f in features}})
    monotonic_rows = []
    for transform, order in PERTURBATION_ORDER.items():
        identity_value = {"gain": 1.0, "noise": np.inf, "tilt": 0.0}[transform]
        parts = []
        for magnitude in order:
            source = identity if magnitude == identity_value else \
                perturbed[(perturbed["transform"] == transform) & np.isclose(perturbed["magnitude"], magnitude)].set_index("row")
            parts.append(source)
        rows_index = parts[0].index
        for feature in features:
            matrix = np.column_stack([p.loc[rows_index, f"z_{feature}"].to_numpy() for p in parts])
            steps = np.diff(matrix, axis=1)
            monotone = (steps >= -1e-12).all(axis=1) | (steps <= 1e-12).all(axis=1)
            monotonic_rows.append({"transform": transform, "feature": feature,
                                   "frac_events_monotone": float(monotone.mean()),
                                   "median_total_change_z": float(np.median(matrix[:, -1] - matrix[:, 0]))})
        # Distance from the unperturbed z grows with intensity on each side of identity.
        position = [m == identity_value for m in order].index(True)
        distance = np.column_stack([
            np.sqrt(((p.loc[rows_index, columns].to_numpy() - identity.loc[rows_index, columns].to_numpy()) ** 2).mean(axis=1))
            for p in parts])
        increasing = np.ones(len(rows_index), bool)
        if position < len(order) - 1:
            increasing &= (np.diff(distance[:, position:], axis=1) >= -1e-12).all(axis=1)
        if position > 0:
            increasing &= (np.diff(distance[:, :position + 1][:, ::-1], axis=1) >= -1e-12).all(axis=1)
        monotonic_rows.append({"transform": transform, "feature": "DISTANCE_FROM_OWN_Z",
                               "frac_events_monotone": float(increasing.mean()), "median_total_change_z": np.nan})
    return pd.DataFrame(change_rows), pd.DataFrame(monotonic_rows), pd.DataFrame(session_rows)


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_INK = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_INK)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def _log_axis(axis, ticks=(0.25, 0.5, 1, 2, 4, 8, 16)) -> None:
    import matplotlib.pyplot as plt

    axis.set_yscale("log")
    axis.set_yticks(list(ticks), [f"{t:g}" for t in ticks])
    axis.yaxis.set_minor_formatter(plt.NullFormatter())


def _boxes(axis, data, colours) -> None:
    parts = axis.boxplot(data, widths=0.6, patch_artist=True,
                         medianprops={"color": INK, "linewidth": 1.2},
                         whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
                         flierprops={"marker": "o", "markersize": 3, "markeredgecolor": INK_MUTED,
                                     "markerfacecolor": "none"})
    for box, colour in zip(parts["boxes"], colours):
        box.set_facecolor(colour)
        box.set_edgecolor(SURFACE)


def plot_population_scores(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.8), facecolor=SURFACE, layout="constrained")
    populations = ("usable", "excluded")
    for axis, score in zip(axes, SCORES):
        _style(axis)
        data = [frame.loc[frame["population"] == p, score].to_numpy() for p in populations]
        _boxes(axis, data, SERIES[:2])
        rng = np.random.default_rng(0)
        for position, values, colour in zip((1, 2), data, SERIES[:2]):
            axis.plot(position + rng.uniform(-0.18, 0.18, len(values)), values, "o", markersize=2.5,
                      color=INK_MUTED, alpha=0.5)
        axis.set_xticks([1, 2], [f"usable (n={len(data[0])})", f"excluded (n={len(data[1])})"], fontsize=8,
                        color=INK_SECONDARY)
        _log_axis(axis)
        axis.set_title(score, fontsize=10, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    figure.suptitle("Held-out scores, LOSO global baseline: usable vs Stage 1-excluded events (log axis)",
                    color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_reason_scores(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["usable"] + sorted(set(frame["exclusion_group"]) - {"usable"})
    figure, axis = plt.subplots(figsize=(10, 5), facecolor=SURFACE, layout="constrained")
    _style(axis)
    data = [frame.loc[frame["exclusion_group"] == g, "rms_z"].to_numpy() for g in order]
    _boxes(axis, data, SERIES[:len(order)])
    axis.set_xticks(range(1, len(order) + 1), [f"{g}\n(n={len(d)})" for g, d in zip(order, data)],
                    fontsize=8, color=INK_SECONDARY)
    _log_axis(axis)
    axis.set_ylabel("rms_z (held-out, log axis)", color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_title("Held-out rms_z by exact Stage 1 exclusion combination", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_within_session(within: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = within.sort_values("median_reference")
    figure, axis = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE, layout="constrained")
    _style(axis)
    y = np.arange(len(rows))
    axis.hlines(y, rows["median_reference"], rows["median_group"], color=BASELINE_INK, linewidth=1.5)
    axis.plot(rows["median_reference"], y, "o", color=SERIES[0], markersize=7, label="usable events, session median")
    axis.plot(rows["median_group"], y, "o", color=SERIES[1], markersize=7, label="excluded events, session median")
    axis.set_yticks(y, [f"{s} (usable {nr}, excluded {ng})" for s, nr, ng in
                        zip(rows["session"], rows["n_reference"], rows["n_group"])], fontsize=8, color=INK_SECONDARY)
    axis.set_xscale("log")
    ticks = [0.5, 1, 2, 4, 8]
    axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axis.xaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_xlabel("rms_z (held-out, log axis)", color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    axis.set_title("Within-session comparison: usable vs excluded events", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_attribution(attribution: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = ["usable"] + sorted(g for g in attribution["group"].unique() if g.startswith("only_")
                                 and attribution.loc[attribution["group"] == g, "n"].iloc[0] >= 3)
    figure, axis = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE, layout="constrained")
    _style(axis)
    y = np.arange(len(features))[::-1]
    height = 0.8 / len(groups)
    for k, group in enumerate(groups):
        rows = attribution[attribution["group"] == group].set_index("feature").loc[list(features)]
        n = int(rows["n"].iloc[0])
        axis.barh(y + 0.4 - (k + 0.5) * height, rows["median_abs_z"], height=height * 0.9, color=SERIES[k],
                  label=f"{group} (n={n})")
    axis.set_yticks(y, [f"{f} ({'level' if f in LEVEL_FEATURES else 'variability'})" for f in features],
                    fontsize=8, color=INK_SECONDARY)
    axis.set_xlabel("Median |z| under the held-out LOSO baseline", color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    axis.set_title("Which features deviate in each Stage 1 exclusion group?", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_perturbation_response(changes: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE, layout="constrained", sharey=True)
    palette = dict(zip(features, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")))
    labels = {"gain": "Gain (x)", "noise": "White-noise SNR (dB)", "tilt": "Tilt coefficient a"}
    for axis, transform in zip(axes, ("gain", "noise", "tilt")):
        _style(axis)
        rows = changes[changes["transform"] == transform].sort_values("magnitude")
        x_values = rows["magnitude"].to_numpy()
        identity = {"gain": 1.0, "noise": 40.0, "tilt": 0.0}[transform]
        xs = np.sort(np.append(x_values, identity))
        for feature in features:
            ys = [0.0 if x == identity else rows.loc[np.isclose(rows["magnitude"], x), f"median_delta_z_{feature}"].iloc[0]
                  for x in xs]
            axis.plot(xs, ys, "o-", color=palette[feature], linewidth=1.5, markersize=4, label=feature)
        axis.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        axis.set_xlabel(labels[transform] + (" (identity plotted at 40)" if transform == "noise" else ""),
                        color=INK_SECONDARY, fontsize=9)
        if transform == "noise":
            axis.invert_xaxis()
        axis.set_title(transform, fontsize=10, color=INK)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Median change in z vs unperturbed", color=INK_SECONDARY, fontsize=9)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    figure.suptitle("Controlled waveform perturbations of usable events: per-feature z response", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_analysis(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    audit_csv: str | Path = STAGE1_AUDIT,
    stage5_heldout_csv: str | Path = STAGE5_HELDOUT,
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_recordings: bool = True,
    run_controlled: bool = True,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir)
    (output / "controlled").mkdir(parents=True, exist_ok=True)
    features = load_feature_selection(selection_json, dataset_csv)
    hashes_before = {str(p): _sha256(Path(p)) for p in (dataset_csv, audit_csv, stage5_heldout_csv)}

    table = load_stage1_table(dataset_csv).reset_index(drop=True)
    sessions = table["recording_file"].map(recording_sessions(data_dir))
    if sessions.isna().any():
        raise PopulationError("some events have no session")
    populations = assign_populations(table)
    groups = comparison_groups(populations)
    overlap = overlap_matrix(populations)

    # Frozen LOSO baseline: usable events of other sessions only.
    z, fold_parameters, baselines = heldout_scores(table, features, sessions)
    scores = combined_scores(z, features)
    audit = pd.read_csv(audit_csv)[["recording_file", "event_id", "annotation_status"]]
    frame = pd.concat([table[["recording_file", "event_id", "start_s", "end_s", "duration_s", "confidence",
                              "n_events_in_recording"]], sessions.rename("session"), populations, z, scores], axis=1)
    frame = frame.merge(audit, on=["recording_file", "event_id"], how="left", validate="one_to_one")
    frame.to_csv(output / "event_scores.csv", index=False)
    fold_parameters.to_csv(output / "baseline_parameters.csv", index=False)

    # Leakage and reproducibility checks.
    stage5 = pd.read_csv(stage5_heldout_csv)
    stage5 = stage5[stage5["strategy"] == "C_loso_global"][["recording_file", "event_id", *SCORES]]
    check = frame[frame["population"] == "usable"].merge(stage5, on=["recording_file", "event_id"], suffixes=("", "_s5"))
    stage5_difference = float(max((check[s] - check[f"{s}_s5"]).abs().max() for s in SCORES))
    keys = list(zip(table["recording_file"].astype(str), table["event_id"].astype(int)))
    excluded_keys = {k for k, u in zip(keys, table["usable"].astype(bool)) if not u}
    session_of = dict(zip(keys, sessions))
    leakage = {
        "usable_rows_matched_to_stage5": int(len(check)),
        "max_abs_score_difference_vs_stage5_C": stage5_difference,
        "folds": int(fold_parameters["fold"].nunique()),
        "excluded_events_in_any_fit": int(sum(len(set(b.calibration_events) & excluded_keys)
                                              for b in baselines.values())),
        "held_out_session_events_in_own_fit": int(sum(sum(session_of[k] == s for k in b.calibration_events)
                                                      for s, b in baselines.items())),
    }

    # Parts 3-5: populations, effect sizes, session control.
    summary_rows = []
    for name, mask in groups.items():
        for score in SCORES:
            summary_rows.append({"group": name, "score": score, **distribution_row(frame.loc[mask, score])})
    usable_rows = frame["population"] == "usable"
    for status in sorted(frame.loc[usable_rows, "annotation_status"].dropna().unique()):
        mask = usable_rows & (frame["annotation_status"] == status)
        for score in SCORES:
            summary_rows.append({"group": f"usable_annotation:{status}", "score": score,
                                 **distribution_row(frame.loc[mask, score])})
    population_summary = pd.DataFrame(summary_rows)
    population_summary.to_csv(output / "population_summary.csv", index=False)

    effects = compare_to_reference(frame, groups)
    effects.to_csv(output / "effect_sizes.csv", index=False)
    within_rows = []
    for name in ("excluded_all", "any_too_short", "any_close_neighbor"):
        for score in SCORES:
            rows = within_session_rows(frame[score], frame["session"], groups[name], groups["usable"])
            within_rows.append(rows.assign(group=name, score=score))
    within = pd.concat(within_rows, ignore_index=True)
    within.to_csv(output / "session_controlled_comparison.csv", index=False)

    reason_rows = []
    for name in overlap.index:
        for other in overlap.columns:
            reason_rows.append({"table": "overlap", "group": name, "other": other, "count": int(overlap.loc[name, other])})
    for combination, count in populations["exclusion_group"].value_counts().items():
        reason_rows.append({"table": "exact_combination", "group": combination, "other": "", "count": int(count)})
    exclusion_summary = pd.concat([pd.DataFrame(reason_rows),
                                   effects[effects["group"] != "excluded_all"].assign(table="effect")],
                                  ignore_index=True)
    exclusion_summary.to_csv(output / "exclusion_reason_summary.csv", index=False)

    # Part 6: feature attribution; Part 9: ordinary session variation.
    attribution = feature_attribution(z, groups, features)
    attribution.to_csv(output / "feature_attribution.csv", index=False)
    variation = session_variation(frame)
    variation.to_csv(output / "session_variation.csv", index=False)
    matched_parts, matched_summary = [], {}
    for name in ("only_close_neighbor", "any_close_neighbor", "any_too_short", "excluded_all"):
        per_event, summary = duration_matched_comparison(frame, z, groups[name], groups["usable"], features)
        matched_parts.append(per_event.assign(group=name))
        matched_summary[name] = summary
    pd.concat(matched_parts, ignore_index=True).to_csv(output / "duration_matched_comparison.csv", index=False)

    # Part 8: annotation cross-check (descriptive only).
    annotation_table = pd.crosstab(frame["exclusion_group"], frame["annotation_status"].fillna("missing"))
    annotation_table.to_csv(output / "annotation_crosscheck.csv")

    # Part 7: recording-level description of recordings without a usable event.
    recording_summary = pd.DataFrame()
    if run_recordings:
        paths = sorted(Path(data_dir).glob("*.wav"))
        recordings = recording_window_statistics(paths)
        recordings["session"] = recordings["recording_file"].map(recording_sessions(data_dir))
        recordings["recording_group"] = recording_groups(recordings, table)
        stage1_counts = table.groupby("recording_file").size()
        recordings["stage1_events"] = recordings["recording_file"].map(stage1_counts).fillna(0).astype(int)
        # Descriptive annotation cross-check (annotations are event boundaries, not labels).
        from inhale_dataset import clean_annotations, read_annotations

        annotations, _ = clean_annotations(read_annotations(Path(data_dir) / "annotation.csv"))
        inhale_counts = annotations[annotations["label"] == "Inhale"].groupby("filename").size()
        annotated = set(annotations["filename"])
        recordings["annotation_status"] = [
            "unannotated" if r not in annotated else ("annotated_with_inhale" if inhale_counts.get(r, 0) else
                                                      "annotated_without_inhale")
            for r in recordings["recording_file"]]
        recordings.to_csv(output / "recording_window_stats.csv", index=False)
        pd.crosstab(recordings["recording_group"], recordings["annotation_status"]).to_csv(
            output / "recording_annotation_crosscheck.csv")
        recording_summary = summarize_recordings(recordings)
        recording_summary.to_csv(output / "recording_level_summary.csv", index=False)
        leakage["recording_candidate_counts_match_stage1"] = bool(
            (recordings["n_inhale_candidates"] == recordings["stage1_events"]).all())

    # Part 11: controlled perturbations of usable events (secondary).
    controlled_summary = {}
    if run_controlled:
        perturbed = controlled_perturbations(table, features, sessions, baselines, data_dir)
        perturbed.to_csv(output / "controlled" / "perturbed_events.csv", index=False, float_format="%.6g")
        changes, monotonic, by_session = summarize_perturbations(perturbed, features)
        changes.to_csv(output / "controlled" / "perturbation_summary.csv", index=False)
        monotonic.to_csv(output / "controlled" / "monotonicity.csv", index=False)
        by_session.to_csv(output / "controlled" / "session_consistency.csv", index=False)
        identity = perturbed[perturbed["transform"] == "identity"].set_index("row")
        controlled_summary = {
            "events": int(identity.shape[0]),
            "identity_max_abs_feature_difference": float(max(
                (identity[f] - table.loc[identity.index, f]).abs().max() for f in features)),
            "plan": [list(p) for p in perturbation_plan()],
        }
        if make_plots:
            import matplotlib

            matplotlib.use("Agg")
            plot_perturbation_response(changes, features, output / "controlled" / "perturbation_response.png")

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_population_scores(frame, output / "usable_vs_excluded_score_distribution.png")
        plot_reason_scores(frame, output / "score_by_exclusion_reason.png")
        plot_within_session(within[(within["group"] == "excluded_all") & (within["score"] == "rms_z")],
                            output / "within_session_population_comparison.png")
        plot_attribution(attribution, features, output / "feature_attribution.png")

    hashes_after = {path: _sha256(Path(path)) for path in hashes_before}
    result = {
        "stage": "Stage 6 - natural population separation and controlled sensitivity (no threshold, no labels)",
        "features": features,
        "baseline": "leave-one-session-out global median / 1.4826*MAD fitted on USABLE events of other sessions",
        "population_sizes": {name: int(mask.sum()) for name, mask in groups.items()},
        "exact_exclusion_combinations": populations["exclusion_group"].value_counts().to_dict(),
        "overlap": overlap.to_dict(),
        "sessions": {"with_events": int(frame["session"].nunique()),
                     "with_excluded_events": int(frame.loc[groups["excluded_all"], "session"].nunique())},
        "leakage_and_reproducibility": leakage,
        "duration_matched_without_duration_feature": matched_summary,
        "inputs_unchanged": hashes_before == hashes_after,
        "input_sha256": {_relative(Path(p)): h for p, h in hashes_before.items()},
        "controlled": controlled_summary,
        "parameters": {"bootstrap_draws": BOOTSTRAP_DRAWS, "permutations": PERMUTATIONS, "seed": SEED,
                       "gains": list(GAINS), "snr_db": list(SNR_DB), "tilts": list(TILTS)},
        "provenance": {"git": _git_state()},
        "no_threshold_or_labels": True,
        "outputs": sorted(str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*") if p.is_file()),
    }
    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 natural population separation (no threshold)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-recordings", action="store_true", help="skip the CNN window pass over all recordings")
    parser.add_argument("--skip-controlled", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_analysis(output_dir=args.output_dir, run_recordings=not args.skip_recordings,
                           run_controlled=not args.skip_controlled, make_plots=not args.no_plots)
    print(json.dumps({k: outcome[k] for k in ("population_sizes", "leakage_and_reproducibility",
                                               "inputs_unchanged", "controlled")}, indent=2, default=str))
