"""Stage 5 baseline-strategy experiment (PRISM_RESEARCH_LOG.md, Entry 7).

Question: which baseline gives stable standardized scores on sessions that
were not used to fit it?  The 7 V1 features, the robust z definition and the
three Stage 4 candidate scores are unchanged:

    z_j = (x_j - center_j) / (1.4826 * MAD_j)
    mean_abs_z = mean |z_j|,  rms_z = sqrt(mean z_j^2),  max_abs_z = max |z_j|

Strategies (only the baseline differs):

A  first-20        Stage 3/4 baseline; historical control, reproduced from the
                   Stage 4 outputs (calibration events leave-one-out).
B  pooled          median/MAD of all 318 usable events; IN-SAMPLE, descriptive only.
C  LOSO global     for each session s: fit on every usable event of the other
                   sessions, score the events of s.  Main generalization test.
D1 offline session location   x' = x - median(other events of the same
                   session), sessions with >= 5 events; global center/scale of
                   x' fitted leave-one-session-out.  OFFLINE DIAGNOSTIC: uses
                   later events of the session.
D2 warm-up session location   x' = x - median(first k events of the
                   session); only later events are scored; global scale as D1.
                   Deployable with a k-event cold start (k = 5; 3 and 10 are
                   sensitivity analyses only).
D3 offline session location + scale   z = (x - median) / (1.4826 * MAD) using
                   the session's other events, sessions with >= 20 events.
                   OFFLINE DIAGNOSTIC.
D1L / D2L          as D1 / D2, but the session location is removed from the LEVEL
                   features only (duration, mean_rms, centroid mean, flatness
                   mean); the WITHIN-EVENT VARIABILITY features keep the global
                   leave-one-session-out treatment of C.

No threshold is chosen and no event is labelled NORMAL or ANOMALY.
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
from baseline_v1 import (
    DEFAULT_OUTPUT_DIR as BASELINE_DIR,
    DEFAULT_SELECTION_JSON,
    BaselineError,
    fit_baseline,
    load_baseline,
    load_feature_selection,
    primary_calibration_mask,
)
from feature_analysis import DEFAULT_DATASET_CSV, MAD_SCALE, MIN_GROUP_SIZE, load_stage1_table
from inhale_dataset import _git_state, _json_default, _relative, _sha256
from scoring_v1 import (
    DEFAULT_OUTPUT_DIR as SCORING_DIR,
    SCORES,
    combined_scores,
    feature_contributions,
    iid_normal_reference,
)


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "baseline_strategies"
DEFAULT_BASELINE_JSON = BASELINE_DIR / "baseline_v1.json"
STAGE4_EVENTS = SCORING_DIR / "event_scores.csv"
STAGE4_LOO = SCORING_DIR / "leave_one_out_scores.csv"

LEVEL_FEATURES = ("duration_s", "mean_rms", "spectral_centroid_mean", "spectral_flatness_mean")
VARIABILITY_FEATURES = ("spectral_centroid_std", "spectral_flatness_std", "spectral_rolloff_std")

# Pre-specified before any Stage 5 result was computed (Entry 7).
MIN_SESSION_EVENTS_LOCATION = MIN_GROUP_SIZE  # 5: leave-one-out session median over >= 4 other events
MIN_SESSION_EVENTS_SCALE = 20                 # session MAD over >= 19 other events
WARMUP_EVENTS = 5                             # deployable variant: first k events set the session location
WARMUP_SENSITIVITY = (3, 10)                  # reported, never used to choose
BOOTSTRAP_DRAWS = 2000
PERMUTATIONS = 2000
SEED = 20261002

STRATEGIES = {
    "A_first20": "deployable; historical control (Stage 3/4 first-20 calibration)",
    "B_pooled_in_sample": "descriptive only: in-sample, every scored event helped fit the baseline",
    "C_loso_global": "deployable: global baseline fitted on other sessions only",
    "D1_offline_session_location": "offline diagnostic: session median uses the session's other events, including later ones",
    "D3_offline_session_location_scale": "offline diagnostic: session median and MAD use the session's other events (sessions >= 20 events)",
    "D1L_offline_level_location": "offline diagnostic: as D1 for the level features only; variability features as C",
}


def warmup_name(k: int, level_only: bool = False) -> str:
    return f"D2L_warmup{k}_level_location" if level_only else f"D2_warmup{k}_session_location"


def warmup_status(k: int, level_only: bool = False) -> str:
    role = "primary" if k == WARMUP_EVENTS else "sensitivity only"
    scope = "level features only; variability features as C" if level_only else "all features"
    return (f"deployable with a cold start ({role}): the first {k} events of a session set its location "
            f"({scope}) and are not scored")


def hybrid_table(table: pd.DataFrame, residuals: pd.DataFrame, level_features: Sequence[str] = LEVEL_FEATURES) -> pd.DataFrame:
    """Raw features with the level columns replaced by session-location residuals."""
    hybrid = table.copy()
    for feature in level_features:
        hybrid[feature] = residuals[feature]
    return hybrid


# ── Baselines ───────────────────────────────────────────────────────────────

def leave_one_session_out(
    fit_table: pd.DataFrame,
    fit_sessions: pd.Series,
    score_table: pd.DataFrame,
    score_sessions: pd.Series,
    features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on every fit row outside session s, score the score rows of s.

    Returns z for ``score_table`` (same index), the fitted parameters per fold, and
    per fold the median in-sample scores of the training rows (the fold reference).
    """
    z_parts, parameters, references = [], [], []
    for session in sorted(score_sessions.unique()):
        training = fit_table[(fit_sessions != session).to_numpy()]
        if len(training) == 0:
            raise BaselineError(f"no training events outside session {session}")
        baseline = fit_baseline(training, features)
        targets = score_table[(score_sessions == session).to_numpy()]
        z_parts.append(baseline.z_scores(targets))
        in_sample = combined_scores(baseline.z_scores(training), features)
        references.append({"fold": session, "n_train": int(len(training)), "n_scored": int(len(targets)),
                           **{f"reference_{s}": float(in_sample[s].median()) for s in SCORES}})
        for feature in features:
            parameters.append({"fold": session, "feature": feature, "n_train": int(len(training)),
                               "center": baseline.parameters[feature].median,
                               "scale": baseline.parameters[feature].scale})
    return pd.concat(z_parts).loc[score_table.index], pd.DataFrame(parameters), pd.DataFrame(references)


def leave_one_out_session_medians(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_LOCATION,
) -> pd.DataFrame:
    """For each event, the per-feature median of the OTHER events of its session.

    NaN for sessions with fewer than ``min_events`` events.
    """
    result = pd.DataFrame(np.nan, index=table.index, columns=list(features))
    for _, group in table.groupby(session_column, sort=True):
        if len(group) < min_events:
            continue
        values = group[list(features)].to_numpy(dtype=float)
        for position, index in enumerate(group.index):
            result.loc[index] = np.median(np.delete(values, position, axis=0), axis=0)
    return result


def session_location_residuals(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_LOCATION,
) -> pd.DataFrame:
    """x - median of the other events in the same session (offline; NaN if ineligible)."""
    return table[list(features)] - leave_one_out_session_medians(table, features, session_column, min_events)


def warmup_residuals(
    table: pd.DataFrame, features: Sequence[str], k: int, session_column: str = "session",
    order_column: str = "usable_order",
) -> pd.DataFrame:
    """x - median of the session's first k events, for the session's later events only.

    Warm-up events and events of sessions with <= k events are NaN (not scored).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    result = pd.DataFrame(np.nan, index=table.index, columns=list(features))
    for _, group in table.groupby(session_column, sort=True):
        ordered = group.sort_values(order_column, kind="mergesort")
        if len(ordered) <= k:
            continue
        location = ordered.iloc[:k][list(features)].median()
        later = ordered.index[k:]
        result.loc[later] = table.loc[later, list(features)] - location
    return result


def session_scale_z(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_SCALE,
) -> pd.DataFrame:
    """z = (x - median) / (1.4826 * MAD) using the other events of the same session.

    NaN for sessions with fewer than ``min_events`` events; zero MAD raises.
    """
    result = pd.DataFrame(np.nan, index=table.index, columns=[f"z_{f}" for f in features])
    for session, group in table.groupby(session_column, sort=True):
        if len(group) < min_events:
            continue
        values = group[list(features)].to_numpy(dtype=float)
        for position, index in enumerate(group.index):
            others = np.delete(values, position, axis=0)
            median = np.median(others, axis=0)
            mad = np.median(np.abs(others - median), axis=0)
            if (mad <= 0).any():
                raise BaselineError(f"zero session MAD in {session}")
            result.loc[index] = (values[position] - median) / (MAD_SCALE * mad)
    return result


def session_in_sample_reference(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_SCALE,
) -> pd.DataFrame:
    """Per session, median in-sample scores under the session's own median/MAD (D3 reference)."""
    rows = []
    for session, group in table.groupby(session_column, sort=True):
        if len(group) < min_events:
            continue
        baseline = fit_baseline(group, features)
        scores = combined_scores(baseline.z_scores(group), features)
        rows.append({"fold": session, "n_train": int(len(group)),
                     **{f"reference_{s}": float(scores[s].median()) for s in SCORES}})
    return pd.DataFrame(rows)


def first20_control(
    context: pd.DataFrame, stage4_events: pd.DataFrame, stage4_loo: pd.DataFrame, features: Sequence[str],
) -> pd.DataFrame:
    """Stage 4 z-scores, with the 20 calibration events replaced by their leave-one-out z."""
    keys = ["recording_file", "event_id"]
    columns = [f"z_{f}" for f in features]
    primary = context[keys].merge(stage4_events[keys + columns], on=keys, how="left", validate="one_to_one")
    loo = context[keys].merge(stage4_loo[keys + columns], on=keys, how="left", validate="one_to_one")
    calibration = context["calibration"].astype(bool).to_numpy()
    if loo.loc[calibration, columns].isna().any().any() or primary[columns].isna().any().any():
        raise BaselineError("Stage 4 outputs do not cover the usable events")
    z = primary[columns].copy()
    z.loc[calibration, columns] = loo.loc[calibration, columns].to_numpy()
    z.index = context.index
    return z


# ── Session-scale stability ─────────────────────────────────────────────────

def session_scale_stability(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_LOCATION, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED,
) -> pd.DataFrame:
    """Bootstrap uncertainty of each session's robust scale (1.4826 * MAD)."""
    rng = np.random.default_rng(seed)
    rows = []
    for session, group in table.groupby(session_column, sort=True):
        if len(group) < min_events:
            continue
        values = group[list(features)].to_numpy(dtype=float)
        n = len(values)
        samples = values[rng.integers(0, n, size=(draws, n))]          # (draws, n, d)
        medians = np.median(samples, axis=1, keepdims=True)
        boot = MAD_SCALE * np.median(np.abs(samples - medians), axis=1)  # (draws, d)
        point = MAD_SCALE * np.median(np.abs(values - np.median(values, axis=0)), axis=0)
        for j, feature in enumerate(features):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = boot[:, j] / point[j]
            rows.append({
                "session": session, "feature": feature, "n": int(n), "scale": float(point[j]),
                "bootstrap_cv": float(boot[:, j].std(ddof=1) / boot[:, j].mean()) if boot[:, j].mean() > 0 else np.nan,
                "bootstrap_ratio_p05": float(np.percentile(ratio, 5)),
                "bootstrap_ratio_p95": float(np.percentile(ratio, 95)),
                "bootstrap_frac_zero_mad": float(np.mean(boot[:, j] == 0)),
                "normal_theory_relative_se": 1.166 / np.sqrt(n),
            })
    return pd.DataFrame(rows)


def scale_heterogeneity_test(
    table: pd.DataFrame, features: Sequence[str], session_column: str = "session",
    min_events: int = MIN_SESSION_EVENTS_LOCATION, permutations: int = PERMUTATIONS, seed: int = SEED,
) -> pd.DataFrame:
    """Do sessions differ in spread beyond what their sizes alone would produce?

    Residuals r = x - session median remove session location; the MAD of r within a
    session equals the session's own MAD.  Statistic: SD across sessions of
    log(MAD of r).  Null: residuals are exchangeable across sessions (shuffled
    with session sizes kept, so size-dependent MAD bias is preserved).
    """
    eligible = table.groupby(session_column)[session_column].transform("size") >= min_events
    subset = table[eligible]
    values = subset[list(features)].to_numpy(dtype=float)
    medians = subset.groupby(session_column)[list(features)].transform("median").to_numpy(dtype=float)
    residuals = values - medians
    labels = subset[session_column].to_numpy()
    sessions = np.unique(labels)
    members = [np.flatnonzero(labels == s) for s in sessions]

    def statistic(values: np.ndarray) -> np.ndarray:
        scales = np.array([np.median(np.abs(values[m] - np.median(values[m], axis=0)), axis=0) for m in members])
        with np.errstate(divide="ignore"):
            return np.log(scales).std(axis=0, ddof=1)

    observed = statistic(residuals)
    observed_joint = observed.mean()          # all features jointly: mean SD of log scales
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(features))
    exceed_joint = 0
    for _ in range(permutations):
        permuted = statistic(residuals[rng.permutation(len(residuals))])
        exceed += permuted >= observed - 1e-12
        exceed_joint += permuted.mean() >= observed_joint - 1e-12
    result = pd.DataFrame({
        "feature": list(features),
        "family": ["level" if f in LEVEL_FEATURES else "within-event variability" for f in features],
        "sessions": len(sessions), "events": int(len(subset)),
        "observed_sd_log_scale": observed,
        "permutation_p": (1 + exceed) / (1 + permutations),
    })
    joint = pd.DataFrame([{"feature": "ALL_FEATURES_JOINT", "family": "all", "sessions": len(sessions),
                           "events": int(len(subset)), "observed_sd_log_scale": observed_joint,
                           "permutation_p": (1 + exceed_joint) / (1 + permutations)}])
    return pd.concat([result, joint], ignore_index=True)


# ── Evaluation ──────────────────────────────────────────────────────────────

def _epsilon_squared(groups: Sequence[np.ndarray]) -> tuple[float, float]:
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    h, p = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return float((h - k + 1) / (n - k)), float(p)


def _robust_sd(values) -> float:
    x = np.asarray(values, dtype=float)
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def evaluate_scores(
    frame: pd.DataFrame, mask: np.ndarray, min_session_events: int = MIN_GROUP_SIZE,
) -> list[dict]:
    """Held-out score distribution, generalization ratio and session dependence.

    ``frame`` holds one strategy's events with ``session``, the three scores and
    ``reference_<score>`` (median score of the rows the baseline was fitted on).
    """
    subset = frame[mask]
    rows = []
    for score in SCORES:
        values = subset[score]
        if values.isna().any() or values.empty:
            return []
        reference = float(subset[f"reference_{score}"].median())
        counts = subset["session"].value_counts()
        tested = sorted(counts[counts >= min_session_events].index)
        session_medians = subset[subset["session"].isin(tested)].groupby("session")[score].median()
        session_reference = subset[subset["session"].isin(tested)].groupby("session")[f"reference_{score}"].median()
        ratios = session_medians / session_reference
        eps2, p = _epsilon_squared([subset.loc[subset["session"] == s, score].to_numpy() for s in tested])
        rows.append({
            "score": score, "n_events": int(len(subset)), "n_sessions": int(subset["session"].nunique()),
            "median": float(values.median()), "mean": float(values.mean()),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "reference_median": reference, "median_over_reference": float(values.median()) / reference,
            "sessions_tested": len(tested),
            "session_median_min": float(session_medians.min()), "session_median_max": float(session_medians.max()),
            "session_median_max_over_min": float(session_medians.max() / session_medians.min()),
            "session_ratio_to_reference_min": float(ratios.min()), "session_ratio_to_reference_max": float(ratios.max()),
            "session_ratio_to_reference_median": float(ratios.median()),
            "robust_sd_log_session_medians": _robust_sd(np.log(session_medians)),
            "session_epsilon_squared": eps2, "session_kruskal_p": p,
        })
    return rows


def evaluate_features(
    frame: pd.DataFrame, mask: np.ndarray, features: Sequence[str], min_session_events: int = MIN_GROUP_SIZE,
) -> list[dict]:
    """Per-feature held-out z: spread, offset and session dependence."""
    subset = frame[mask]
    if subset[[f"z_{f}" for f in features]].isna().any().any() or subset.empty:
        return []
    counts = subset["session"].value_counts()
    tested = sorted(counts[counts >= min_session_events].index)
    rows = []
    for feature in features:
        z = subset[f"z_{feature}"]
        session_medians = subset[subset["session"].isin(tested)].groupby("session")[f"z_{feature}"].median()
        eps2, _ = _epsilon_squared([subset.loc[subset["session"] == s, f"z_{feature}"].to_numpy() for s in tested])
        rows.append({
            "feature": feature, "family": "level" if feature in LEVEL_FEATURES else "within-event variability",
            "n_events": int(len(subset)), "median_z": float(z.median()), "robust_sd_z": _robust_sd(z),
            "median_abs_z": float(z.abs().median()),
            "session_median_z_min": float(session_medians.min()), "session_median_z_max": float(session_medians.max()),
            "robust_sd_session_median_z": _robust_sd(session_medians),
            "session_epsilon_squared": eps2,
        })
    return rows


def evaluate_contributions(frame: pd.DataFrame, mask: np.ndarray, features: Sequence[str]) -> list[dict]:
    subset = frame[mask]
    z = subset[[f"z_{f}" for f in features]]
    if z.isna().any().any() or z.empty:
        return []
    contributions = feature_contributions(z, features)
    return [{"feature": f, "rms_z__mean_share": float(contributions[f"rms_share_{f}"].mean()),
             "max_abs_z__argmax_fraction": float(contributions[f"is_max_{f}"].mean()),
             "mean_abs_z__median_abs_z": float(contributions[f"abs_z_{f}"].median())} for f in features]


def parameter_variability(parameters: pd.DataFrame, pooled: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    """Spread of fold centers/scales, in units of the pooled (all-event) scale."""
    rows = []
    for (strategy, feature), group in parameters.groupby(["strategy", "feature"], sort=False):
        center, scale = pooled[feature]
        rows.append({
            "strategy": strategy, "feature": feature, "folds": int(len(group)),
            "center_min": float(group["center"].min()), "center_max": float(group["center"].max()),
            "center_range_in_pooled_scale": float((group["center"].max() - group["center"].min()) / scale),
            "scale_min_over_pooled": float(group["scale"].min() / scale),
            "scale_max_over_pooled": float(group["scale"].max() / scale),
            "scale_cv": float(group["scale"].std(ddof=1) / group["scale"].mean()) if len(group) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_INK = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PLOTTED = ("A_first20", "B_pooled_in_sample", "C_loso_global", "D1_offline_session_location")
PLOT_LABELS = {
    "A_first20": "A first-20 (control)",
    "B_pooled_in_sample": "B pooled (in-sample, descriptive)",
    "C_loso_global": "C leave-one-session-out global",
    "D1_offline_session_location": "D1 offline session location",
}


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_INK)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def plot_session_medians(events: pd.DataFrame, sessions: Sequence[str], reference: float, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE, layout="constrained")
    _style(axis)
    y = np.arange(len(sessions))[::-1]
    for offset, (strategy, colour) in zip((0.27, 0.09, -0.09, -0.27), zip(PLOTTED, SERIES)):
        block = events[events["strategy"] == strategy]
        medians = block.groupby("session")["rms_z"].median().reindex(sessions)
        marker = "o" if strategy != "B_pooled_in_sample" else "s"
        axis.plot(medians.to_numpy(), y + offset, marker, color=colour, markersize=7, linestyle="none",
                  markerfacecolor=colour if strategy != "B_pooled_in_sample" else "none", markeredgewidth=1.5,
                  label=PLOT_LABELS[strategy])
    axis.axvline(reference, color=INK_SECONDARY, linewidth=0.8, linestyle="--",
                 label="Median if the 7 z-scores were independent N(0,1)")
    counts = events[events["strategy"] == "C_loso_global"]["session"].value_counts()
    axis.set_yticks(y, [f"{s} (n={counts[s]})" for s in sessions], fontsize=8, color=INK_SECONDARY)
    axis.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4]
    axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axis.xaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_xlabel("Session median of rms_z (log axis)", color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8,
                labelcolor=INK_SECONDARY)
    axis.set_title("Session medians of rms_z under each baseline strategy (sessions with >= 5 events)",
                   color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_feature_session_effect(feature_rows: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    shown = ("A_first20", "C_loso_global", "D1_offline_session_location")
    colours = dict(zip(shown, (SERIES[0], SERIES[2], SERIES[3])))
    figure, axis = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE, layout="constrained")
    _style(axis)
    y = np.arange(len(features))[::-1]
    height = 0.26
    for k, strategy in enumerate(shown):
        rows = feature_rows[(feature_rows["strategy"] == strategy) & (feature_rows["eval_set"] == "sessions_ge5")]
        values = rows.set_index("feature").loc[list(features), "session_epsilon_squared"]
        positions = y + (1 - k) * height
        axis.barh(positions, values.clip(lower=0), height=height * 0.9, color=colours[strategy],
                  label=PLOT_LABELS[strategy])
        for position, value in zip(positions, values):
            axis.text(max(value, 0) + 0.005, position, f"{value:.2f}", va="center", fontsize=6.5, color=INK_SECONDARY)
    axis.set_yticks(y, [f"{f} ({'level' if f in LEVEL_FEATURES else 'variability'})" for f in features],
                    fontsize=8, color=INK_SECONDARY)
    axis.set_xlabel("Session effect on held-out z (epsilon-squared, Kruskal-Wallis; negative shown as 0)",
                    color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    axis.set_title("How much of each feature's held-out z is explained by session?", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def _scored_frame(strategy: str, context: pd.DataFrame, z: pd.DataFrame, reference: pd.DataFrame,
                  features: Sequence[str]) -> pd.DataFrame:
    """One strategy's scored events (rows without z are dropped)."""
    z = z.reindex(context.index)
    scored = z.notna().all(axis=1)
    frame = context[scored].copy()
    frame.insert(0, "strategy", strategy)
    frame = pd.concat([frame, z[scored]], axis=1)
    scores = combined_scores(z[scored], features)
    frame = pd.concat([frame, scores], axis=1)
    reference = reference.reindex(context.index)[scored]
    for score in SCORES:
        frame[f"reference_{score}"] = reference[f"reference_{score}"]
    return frame


def _fold_reference(context: pd.DataFrame, references: pd.DataFrame, column: str = "session") -> pd.DataFrame:
    mapped = context[[column]].merge(references, left_on=column, right_on="fold", how="left")
    mapped.index = context.index
    return mapped[[f"reference_{s}" for s in SCORES]]


def _constant_reference(context: pd.DataFrame, values: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame({f"reference_{s}": values[s] for s in SCORES}, index=context.index)


def run_experiment(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    baseline_json: str | Path = DEFAULT_BASELINE_JSON,
    stage4_events_csv: str | Path = STAGE4_EVENTS,
    stage4_loo_csv: str | Path = STAGE4_LOO,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = load_feature_selection(selection_json, dataset_csv)
    stage3 = load_baseline(baseline_json)
    if list(stage3.features) != features:
        raise BaselineError("Stage 3 baseline features differ from the feature selection")

    table = load_stage1_table(dataset_csv)
    usable = table[table["usable"].astype(bool)].reset_index(drop=True)
    stage4 = pd.read_csv(stage4_events_csv)
    stage4_loo = pd.read_csv(stage4_loo_csv)
    keys = ["recording_file", "event_id"]
    context = usable[keys + ["usable_order"] + features].merge(
        stage4[keys + ["session", "calibration"]], on=keys, how="left", validate="one_to_one")
    if context["session"].isna().any():
        raise BaselineError("Stage 4 session context missing for some usable events")
    if not (context["calibration"].astype(bool).to_numpy() == primary_calibration_mask(usable).to_numpy()).all():
        raise BaselineError("Stage 4 calibration membership differs from usable_order 1-20")
    context["usable_order"] = context["usable_order"].astype(int)
    context["n_session"] = context.groupby("session")["session"].transform("size")
    info = context[["recording_file", "event_id", "session", "n_session", "calibration", "usable_order"]]
    sessions = context["session"]

    frames, parameters, strategy_status = [], [], dict(STRATEGIES)

    # A: first-20 historical control (Stage 4 values, calibration events leave-one-out).
    z_a = first20_control(context, stage4, stage4_loo, features)
    loo_scores = combined_scores(stage4_loo[[f"z_{f}" for f in features]], features)
    frames.append(_scored_frame("A_first20", info, z_a,
                                _constant_reference(info, {s: float(loo_scores[s].median()) for s in SCORES}), features))
    parameters += [{"strategy": "A_first20", "fold": "first20", "feature": f, "n_train": stage3.n_calibration,
                    "center": stage3.parameters[f].median, "scale": stage3.parameters[f].scale} for f in features]
    stage4_check = frames[-1].merge(stage4[keys + list(SCORES)], on=keys, suffixes=("", "_stage4"))
    stage4_check = stage4_check[~stage4_check["calibration"].astype(bool)]
    a_matches_stage4 = float(max((stage4_check[s] - stage4_check[f"{s}_stage4"]).abs().max() for s in SCORES))

    # B: pooled, in-sample (descriptive only).
    pooled = fit_baseline(context, features)
    z_b = pooled.z_scores(context)
    b_scores = combined_scores(z_b, features)
    frames.append(_scored_frame("B_pooled_in_sample", info, z_b,
                                _constant_reference(info, {s: float(b_scores[s].median()) for s in SCORES}), features))
    parameters += [{"strategy": "B_pooled_in_sample", "fold": "all", "feature": f, "n_train": pooled.n_calibration,
                    "center": pooled.parameters[f].median, "scale": pooled.parameters[f].scale} for f in features]

    # C: leave-one-session-out global baseline.
    z_c, params_c, refs_c = leave_one_session_out(context, sessions, context, sessions, features)
    frames.append(_scored_frame("C_loso_global", info, z_c, _fold_reference(info, refs_c), features))
    parameters += params_c.assign(strategy="C_loso_global").to_dict("records")

    # D1: offline session location (other events of the session), global scale leave-one-session-out.
    residuals = session_location_residuals(context, features)
    eligible = residuals.notna().all(axis=1)
    z_d1, params_d1, refs_d1 = leave_one_session_out(
        residuals[eligible], sessions[eligible], residuals[eligible], sessions[eligible], features)
    frames.append(_scored_frame("D1_offline_session_location", info, z_d1, _fold_reference(info, refs_d1), features))
    parameters += params_d1.assign(strategy="D1_offline_session_location").to_dict("records")

    # D2: warm-up session location (deployable with a k-event cold start); scale as D1.
    for k in (WARMUP_EVENTS, *WARMUP_SENSITIVITY):
        warm = warmup_residuals(context, features, k)
        scored = warm.notna().all(axis=1)
        z_d2, _, refs_d2 = leave_one_session_out(
            residuals[eligible], sessions[eligible], warm[scored], sessions[scored], features)
        frames.append(_scored_frame(warmup_name(k), info, z_d2, _fold_reference(info, refs_d2), features))
        strategy_status[warmup_name(k)] = warmup_status(k)

    # D1L / D2L: session location removed from the level features only.
    level_hybrid = hybrid_table(context[features], residuals)
    z_d1l, params_d1l, refs_d1l = leave_one_session_out(
        level_hybrid[eligible], sessions[eligible], level_hybrid[eligible], sessions[eligible], features)
    frames.append(_scored_frame("D1L_offline_level_location", info, z_d1l, _fold_reference(info, refs_d1l), features))
    parameters += params_d1l.assign(strategy="D1L_offline_level_location").to_dict("records")
    for k in (WARMUP_EVENTS, *WARMUP_SENSITIVITY):
        warm = warmup_residuals(context, features, k)
        scored = warm.notna().all(axis=1)
        warm_hybrid = hybrid_table(context[features], warm)[scored]
        z_d2l, _, refs_d2l = leave_one_session_out(
            level_hybrid[eligible], sessions[eligible], warm_hybrid, sessions[scored], features)
        frames.append(_scored_frame(warmup_name(k, True), info, z_d2l, _fold_reference(info, refs_d2l), features))
        strategy_status[warmup_name(k, True)] = warmup_status(k, True)

    # D3: offline session location + scale (sessions with >= 20 events).
    z_d3 = session_scale_z(context, features)
    refs_d3 = session_in_sample_reference(context, features)
    frames.append(_scored_frame("D3_offline_session_location_scale", info, z_d3, _fold_reference(info, refs_d3), features))

    events = pd.concat(frames, ignore_index=True)
    events["status"] = events["strategy"].map(strategy_status)
    events.to_csv(output / "heldout_scores.csv", index=False, float_format="%.6g")

    # Evaluation sets: every strategy is compared only on events it scores completely.
    counts = context["session"].value_counts()
    eval_sets = {
        "all_usable": np.ones(len(context), bool),
        "sessions_ge5": context["session"].map(counts).to_numpy() >= MIN_SESSION_EVENTS_LOCATION,
        "sessions_ge20": context["session"].map(counts).to_numpy() >= MIN_SESSION_EVENTS_SCALE,
    }
    for k in (WARMUP_EVENTS, *WARMUP_SENSITIVITY):
        eval_sets[f"post_warmup_k{k}"] = warmup_residuals(context, features, k).notna().all(axis=1).to_numpy()

    summary_rows, feature_rows, contribution_rows = [], [], []
    for strategy, block in events.groupby("strategy", sort=False):
        block = info[keys].merge(block, on=keys, how="left")
        for name, mask in eval_sets.items():
            for row in evaluate_scores(block, mask):
                summary_rows.append({"strategy": strategy, "eval_set": name, "status": strategy_status[strategy], **row})
            for row in evaluate_features(block, mask, features):
                feature_rows.append({"strategy": strategy, "eval_set": name, **row})
            for row in evaluate_contributions(block, mask, features):
                contribution_rows.append({"strategy": strategy, "eval_set": name, **row})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    feature_summary = pd.DataFrame(feature_rows)
    feature_summary.to_csv(output / "feature_stability.csv", index=False)
    pd.DataFrame(contribution_rows).to_csv(output / "feature_contributions.csv", index=False)

    session_medians = (events.groupby(["strategy", "session"])[list(SCORES)].median()
                       .join(events.groupby(["strategy", "session"]).size().rename("n_scored"))
                       .reset_index())
    session_medians.to_csv(output / "session_medians.csv", index=False)

    parameter_frame = pd.DataFrame(parameters)
    parameter_frame.to_csv(output / "baseline_parameters.csv", index=False)
    pooled_scale = {f: (pooled.parameters[f].median, pooled.parameters[f].scale) for f in features}
    variability = parameter_variability(parameter_frame[parameter_frame["strategy"].isin(
        ["C_loso_global", "D1_offline_session_location"])], pooled_scale)
    variability.to_csv(output / "parameter_variability.csv", index=False)

    # Is session-scale normalization defensible?
    scale_stability = session_scale_stability(context, features)
    scale_stability.to_csv(output / "session_scale_stability.csv", index=False)
    heterogeneity = scale_heterogeneity_test(context, features)
    heterogeneity.to_csv(output / "session_scale_heterogeneity.csv", index=False)

    reference = iid_normal_reference(len(features))
    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        tested_sessions = sorted(counts[counts >= MIN_SESSION_EVENTS_LOCATION].index)
        plot_session_medians(events, tested_sessions, reference["rms_z"]["p50"], output / "session_medians_rms_z.png")
        plot_feature_session_effect(feature_summary, features, output / "feature_session_effect.png")

    headline = summary[summary["score"] == "rms_z"].set_index(["strategy", "eval_set"])
    result = {
        "stage": "Stage 5 - baseline strategy experiment (no threshold, no labels)",
        "features": features,
        "strategies": strategy_status,
        "prespecified_rules": {
            "min_session_events_location": MIN_SESSION_EVENTS_LOCATION,
            "min_session_events_scale": MIN_SESSION_EVENTS_SCALE,
            "warmup_events_primary": WARMUP_EVENTS, "warmup_events_sensitivity": list(WARMUP_SENSITIVITY),
            "bootstrap_draws": BOOTSTRAP_DRAWS, "permutations": PERMUTATIONS, "seed": SEED,
        },
        "eval_sets": {name: int(mask.sum()) for name, mask in eval_sets.items()},
        "sessions": {"total": int(counts.size), "ge5": int((counts >= 5).sum()), "ge20": int((counts >= 20).sum())},
        "consistency_checks": {"A_max_abs_score_difference_vs_stage4": a_matches_stage4},
        "iid_standard_normal_reference": reference,
        "rms_z_headline": {f"{s}|{e}": {k: headline.loc[(s, e), k] for k in (
            "n_events", "median", "p05", "p95", "median_over_reference", "session_median_max_over_min",
            "session_epsilon_squared")} for s, e in headline.index},
        "inputs": {
            "dataset": {"path": _relative(Path(dataset_csv)), "sha256": _sha256(Path(dataset_csv))},
            "feature_selection": {"path": _relative(Path(selection_json)), "sha256": _sha256(Path(selection_json))},
            "stage3_baseline": {"path": _relative(Path(baseline_json)), "sha256": _sha256(Path(baseline_json))},
            "stage4_event_scores": {"path": _relative(Path(stage4_events_csv)), "sha256": _sha256(Path(stage4_events_csv))},
            "stage4_leave_one_out": {"path": _relative(Path(stage4_loo_csv)), "sha256": _sha256(Path(stage4_loo_csv))},
        },
        "provenance": {"git": _git_state()},
        "no_threshold_or_labels": True,
        "outputs": sorted(p.name for p in output.iterdir()),
    }
    with (output / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 baseline-strategy experiment (no threshold)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_experiment(output_dir=args.output_dir, make_plots=not args.no_plots)
    print(json.dumps(outcome["consistency_checks"], indent=2))
    for key, values in outcome["rms_z_headline"].items():
        print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in values.items()})
