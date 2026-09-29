"""V1 candidate combined scores (Stage 4; PRISM_RESEARCH_LOG.md, Entry 6).

Per event, with z_j the robust z-scores of the d V1 features against the
Stage 3 baseline (``results/baseline_v1/baseline_v1.json``):

    mean_abs_z = mean_j |z_j|
    rms_z      = sqrt(mean_j z_j^2)
    max_abs_z  = max_j |z_j|

Contribution definitions differ by score and are not interchangeable:

* mean_abs_z: contribution_j = |z_j|
* rms_z:      contribution_j = z_j^2 / sum_k z_k^2
* max_abs_z:  the feature(s) attaining max_j |z_j|

The 20 calibration events are also scored leave-one-out (baseline refitted on
the other 19) to estimate their out-of-sample scores.  The scores are
candidates under study: no threshold is applied and no event is labelled
NORMAL or ANOMALY.  The Stage 3 baseline and calibration design are unchanged.
"""

from __future__ import annotations

import argparse
import json
from math import comb
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
from feature_analysis import DEFAULT_DATASET_CSV, load_stage1_table
from inhale_dataset import _git_state, _json_default, _relative, _sha256


DEFAULT_BASELINE_JSON = BASELINE_DIR / "baseline_v1.json"
DEFAULT_STAGE3_SCORES = BASELINE_DIR / "robust_z_scores.csv"
DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "scoring_v1"
SCORES = ("mean_abs_z", "rms_z", "max_abs_z")
GROUPS = ("calibration_in_sample", "calibration_leave_one_out", "calibration_session_other", "other_session")
PAIR = ("mean_rms", "spectral_centroid_mean")
FOCUS_FEATURES = ("mean_rms", "spectral_centroid_mean", "spectral_centroid_std", "spectral_flatness_mean")
REFERENCE_SEED = 20261001
REFERENCE_DRAWS = 100_000


class ScoringError(ValueError):
    """Scores cannot be computed safely."""


# ── Scores and contributions ────────────────────────────────────────────────

def z_matrix(z: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """The ``z_<feature>`` columns as a finite (n, d) array, in feature order."""
    if not features:
        raise ScoringError("at least one feature is required")
    columns = [f"z_{f}" for f in features]
    missing = [c for c in columns if c not in z.columns]
    if missing:
        raise ScoringError(f"missing z-score columns: {missing}")
    values = z[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ScoringError("non-finite z-scores; combined scores are undefined")
    return values


def combined_scores(z: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """mean_abs_z, rms_z, max_abs_z and the feature(s) attaining max_abs_z."""
    values = z_matrix(z, features)
    absolute = np.abs(values)
    maximum = absolute.max(axis=1) if len(values) else np.empty(0)
    top = [";".join(f for f, a in zip(features, row) if a == m) for row, m in zip(absolute, maximum)]
    return pd.DataFrame({
        "mean_abs_z": absolute.mean(axis=1) if len(values) else np.empty(0),
        "rms_z": np.sqrt(np.mean(values ** 2, axis=1)) if len(values) else np.empty(0),
        "max_abs_z": maximum,
        "max_abs_z_feature": top,
    }, index=z.index)


def feature_contributions(z: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Per-event contributions under each score's own definition.

    ``abs_z_<f>`` = |z_f| (mean_abs_z); ``rms_share_<f>`` = z_f^2 / sum z^2 (rms_z,
    NaN when every z is 0); ``is_max_<f>`` marks the feature(s) attaining max |z|.
    ``abs_share_<f>`` = |z_f| / sum |z| is also given so mean_abs_z dominance can be
    compared across events; it is derived from, not a replacement for, |z_f|.
    """
    values = z_matrix(z, features)
    absolute, squared = np.abs(values), values ** 2
    abs_total = absolute.sum(axis=1, keepdims=True)
    sq_total = squared.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_share = np.where(abs_total > 0, absolute / abs_total, np.nan)
        rms_share = np.where(sq_total > 0, squared / sq_total, np.nan)
    is_max = absolute == absolute.max(axis=1, keepdims=True) if len(values) else np.zeros_like(values, bool)
    frame = {}
    for j, feature in enumerate(features):
        frame[f"abs_z_{feature}"] = absolute[:, j]
        frame[f"abs_share_{feature}"] = abs_share[:, j]
        frame[f"rms_share_{feature}"] = rms_share[:, j]
        frame[f"is_max_{feature}"] = is_max[:, j]
    return pd.DataFrame(frame, index=z.index)


def single_feature_influence(z: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """How much of each score rests on the event's largest |z|.

    ``top1_share_mean_abs`` = max|z| / sum|z|; ``top1_share_rms`` = max z^2 / sum z^2.
    ``*_without_top`` is the score recomputed on the other d-1 features divided by the
    full score (max_abs_z: second-largest |z| / largest).  Lower = more dependent on
    one feature.  NaN when undefined (d = 1 or a zero score).
    """
    values = z_matrix(z, features)
    d = values.shape[1]
    ordered = np.sort(np.abs(values), axis=1)[:, ::-1]
    first = ordered[:, 0] if len(values) else np.empty(0)
    abs_sum, sq_sum = ordered.sum(axis=1), (ordered ** 2).sum(axis=1)
    nan = np.full(len(values), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({
            "top1_share_mean_abs": np.where(abs_sum > 0, first / abs_sum, np.nan),
            "top1_share_rms": np.where(sq_sum > 0, first ** 2 / sq_sum, np.nan),
            "mean_abs_without_top": ((abs_sum - first) / (d - 1)) / (abs_sum / d) if d > 1 else nan,
            "rms_without_top": np.sqrt((sq_sum - first ** 2) / (d - 1)) / np.sqrt(sq_sum / d) if d > 1 else nan,
            "max_abs_without_top": ordered[:, 1] / first if d > 1 else nan,
        }, index=z.index).replace([np.inf, -np.inf], np.nan)


# ── Leave-one-out calibration ───────────────────────────────────────────────

def leave_one_out_scores(calibration: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each calibration event against a baseline fitted on the other events only."""
    if len(calibration) < 3:
        raise ScoringError("leave-one-out needs at least 3 calibration events")
    rows, parameters = [], []
    for position, index in enumerate(calibration.index):
        training = calibration.drop(index=index)
        baseline = fit_baseline(training, features)
        held_out = calibration.loc[[index]]
        key = (str(held_out["recording_file"].iloc[0]), int(held_out["event_id"].iloc[0]))
        if key in baseline.calibration_events:
            raise ScoringError(f"held-out event {key} leaked into its own baseline")
        z = baseline.z_scores(held_out)
        scores = combined_scores(z, features)
        rows.append({
            "recording_file": key[0], "event_id": key[1],
            "usable_order": int(held_out["usable_order"].iloc[0]) if "usable_order" in held_out else position + 1,
            "n_training": baseline.n_calibration,
            **z.iloc[0].to_dict(), **scores.iloc[0].to_dict(),
        })
        for feature in features:
            parameters.append({"held_out_recording_file": key[0], "held_out_event_id": key[1], "feature": feature,
                               "median": baseline.parameters[feature].median,
                               "mad": baseline.parameters[feature].mad,
                               "scale": baseline.parameters[feature].scale,
                               "n_training": baseline.n_calibration})
    return pd.DataFrame(rows), pd.DataFrame(parameters)


# ── Summaries ───────────────────────────────────────────────────────────────

def distribution_summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0}
    p05, p25, p50, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
            "min": float(x.min()), "p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
            "max": float(x.max())}


def summarize_scores(frames: Mapping[str, pd.DataFrame], extra: Sequence[str] = ()) -> pd.DataFrame:
    """One row per (group, score) for each named frame of scores."""
    rows = []
    for name, frame in frames.items():
        for score in (*SCORES, *extra):
            rows.append({"group": name, "score": score, **distribution_summary(frame[score])})
    return pd.DataFrame(rows)


def summarize_contributions(
    contributions: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    scope_type: str,
) -> pd.DataFrame:
    rows = []
    for scope, frame in contributions.items():
        n = len(frame)
        for feature in features:
            argmax = frame[f"is_max_{feature}"].astype(bool) if n else pd.Series(dtype=bool)
            rows.append({
                "scope_type": scope_type, "scope": scope, "feature": feature, "n": n,
                "mean_abs_z__median_abs_z": float(frame[f"abs_z_{feature}"].median()) if n else np.nan,
                "mean_abs_z__mean_abs_z": float(frame[f"abs_z_{feature}"].mean()) if n else np.nan,
                "mean_abs_z__mean_share": float(frame[f"abs_share_{feature}"].mean()) if n else np.nan,
                "rms_z__mean_share": float(frame[f"rms_share_{feature}"].mean()) if n else np.nan,
                "rms_z__median_share": float(frame[f"rms_share_{feature}"].median()) if n else np.nan,
                "max_abs_z__argmax_count": int(argmax.sum()) if n else 0,
                "max_abs_z__argmax_fraction": float(argmax.mean()) if n else np.nan,
            })
    return pd.DataFrame(rows)


def inflation_decomposition(z_target: pd.DataFrame, z_reference: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Exact per-feature decomposition of group-mean score differences.

    mean(mean_abs_z) = (1/d) sum_j mean|z_j|, and mean(rms_z^2) = (1/d) sum_j mean z_j^2,
    so the target-minus-reference difference of each splits exactly over features.
    For max_abs_z the argmax fractions are compared.
    """
    target, reference = z_matrix(z_target, features), z_matrix(z_reference, features)
    d = len(features)
    abs_diff = (np.abs(target).mean(axis=0) - np.abs(reference).mean(axis=0)) / d
    sq_diff = ((target ** 2).mean(axis=0) - (reference ** 2).mean(axis=0)) / d
    target_max = np.abs(target) == np.abs(target).max(axis=1, keepdims=True)
    reference_max = np.abs(reference) == np.abs(reference).max(axis=1, keepdims=True)
    return pd.DataFrame({
        "feature": list(features),
        "mean_abs_z__mean_difference_contribution": abs_diff,
        "mean_abs_z__share_of_difference": abs_diff / abs_diff.sum(),
        "rms_z_squared__mean_difference_contribution": sq_diff,
        "rms_z_squared__share_of_difference": sq_diff / sq_diff.sum(),
        "max_abs_z__argmax_fraction_target": target_max.mean(axis=0),
        "max_abs_z__argmax_fraction_reference": reference_max.mean(axis=0),
    })


def pair_analysis(z: pd.DataFrame, contributions: pd.DataFrame, features: Sequence[str], pair=PAIR) -> dict:
    """Does the correlated pair jointly dominate? Compared with equal-share references."""
    a, b = pair
    d = len(features)
    absolute = np.abs(z_matrix(z, features))
    rank = (-absolute).argsort(axis=1).argsort(axis=1)  # 0 = largest |z|
    ia, ib = features.index(a), features.index(b)
    top_two = (np.maximum(rank[:, ia], rank[:, ib]) <= 1)
    joint_rms = contributions[f"rms_share_{a}"] + contributions[f"rms_share_{b}"]
    joint_abs = contributions[f"abs_share_{a}"] + contributions[f"abs_share_{b}"]
    either_max = contributions[f"is_max_{a}"] | contributions[f"is_max_{b}"]
    return {
        "n": int(len(z)),
        "spearman_z": float(stats.spearmanr(z[f"z_{a}"], z[f"z_{b}"]).statistic) if len(z) > 2 else float("nan"),
        "fraction_opposite_sign": float(np.mean(np.sign(z[f"z_{a}"]) != np.sign(z[f"z_{b}"]))),
        "median_joint_rms_share": float(joint_rms.median()),
        "median_joint_abs_share": float(joint_abs.median()),
        "equal_share_reference": 2 / d,
        "fraction_pair_are_top_two": float(top_two.mean()),
        "top_two_reference_if_exchangeable": 1 / comb(d, 2),
        "fraction_argmax_in_pair": float(either_max.mean()),
        "argmax_reference_if_exchangeable": 2 / d,
    }


def isolated_extreme_ranks(scores: pd.DataFrame, influence: pd.DataFrame, ratio: float = 2.0) -> dict:
    """How highly each score ranks events driven by one isolated feature.

    An event is 'isolated-extreme' when its largest |z| is at least ``ratio`` times
    the second largest (a descriptive definition).  Percentile ranks are among all
    events in ``scores``; a higher rank under a score means that score responds
    more to a single isolated feature.
    """
    isolated = (influence["max_abs_without_top"] <= 1 / ratio).to_numpy()
    percentile = scores[list(SCORES)].rank(pct=True)
    return {
        "definition": f"largest |z| >= {ratio:g} x second-largest |z|",
        "n_isolated": int(isolated.sum()), "n_total": int(len(scores)),
        "median_percentile_rank": {s: float(percentile.loc[isolated, s].median()) if isolated.any() else float("nan")
                                   for s in SCORES},
        "median_percentile_rank_others": {s: float(percentile.loc[~isolated, s].median()) for s in SCORES},
        "isolated_feature_counts": scores.loc[isolated, "max_abs_z_feature"].value_counts().to_dict(),
    }


def iid_normal_reference(d: int, draws: int = REFERENCE_DRAWS, seed: int = REFERENCE_SEED) -> dict:
    """Score quantiles if the d z-scores were independent standard normal (reference only)."""
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(rng.standard_normal((draws, d)), columns=[f"z_f{j}" for j in range(d)])
    scores = combined_scores(z, [f"f{j}" for j in range(d)])
    return {score: {q: float(np.percentile(scores[score], int(q[1:]))) for q in ("p05", "p25", "p50", "p75", "p95")}
            for score in SCORES}


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_INK = "#fcfcfb", "#e1e0d9", "#c3c2b7"
GROUP_COLOURS = dict(zip(GROUPS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
GROUP_LABELS = {
    "calibration_in_sample": "Calibration, in-sample (n=20)",
    "calibration_leave_one_out": "Calibration, leave-one-out (n=20)",
    "calibration_session_other": "Same sessions, not calibration (n=39)",
    "other_session": "Other sessions (n=259)",
}


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_INK)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def plot_score_distributions(frames: Mapping[str, pd.DataFrame], reference: Mapping, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor=SURFACE, layout="constrained")
    for axis, score in zip(axes, SCORES):
        _style(axis)
        data = [frames[g][score].to_numpy() for g in GROUPS]
        parts = axis.boxplot(data, widths=0.6, patch_artist=True,
                             medianprops={"color": INK, "linewidth": 1.2},
                             whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
                             flierprops={"marker": "o", "markersize": 3, "markeredgecolor": INK_MUTED,
                                         "markerfacecolor": "none"})
        for box, group in zip(parts["boxes"], GROUPS):
            box.set_facecolor(GROUP_COLOURS[group])
            box.set_edgecolor(SURFACE)
        axis.axhline(reference[score]["p50"], color=INK_SECONDARY, linewidth=0.8, linestyle="--")
        axis.set_yscale("log")
        ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
        axis.set_yticks(ticks, [f"{t:g}" for t in ticks])
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
        axis.set_xticks(range(1, len(GROUPS) + 1), ["" for _ in GROUPS])
        axis.set_title(score, fontsize=10, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    handles = [Patch(facecolor=GROUP_COLOURS[g]) for g in GROUPS]
    handles.append(Line2D([], [], color=INK_SECONDARY, linestyle="--", linewidth=0.8))
    labels = [GROUP_LABELS[g] for g in GROUPS] + ["Median if the 7 z-scores were independent N(0,1)"]
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8,
                  labelcolor=INK_SECONDARY)
    figure.suptitle("Candidate combined scores against the primary first-20 baseline (log axis)", color=INK, fontsize=12)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_feature_dominance(summary: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = GROUPS[1:]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE, layout="constrained", sharey=True)
    y = np.arange(len(features))[::-1]
    height = 0.26
    for axis, column, title in (
        (axes[0], "max_abs_z__argmax_fraction", "max_abs_z: share of events where the feature is the max"),
        (axes[1], "rms_z__mean_share", "rms_z: mean share z_j^2 / sum z^2"),
    ):
        _style(axis)
        for k, group in enumerate(groups):
            rows = summary[(summary["scope_type"] == "group") & (summary["scope"] == group)].set_index("feature")
            axis.barh(y + (1 - k) * height, rows.loc[list(features), column], height=height * 0.9,
                      color=GROUP_COLOURS[group], label=GROUP_LABELS[group])
        axis.axvline(1 / len(features), color=INK_SECONDARY, linewidth=0.8, linestyle="--")
        axis.set_title(title, fontsize=10, color=INK)
        axis.set_yticks(y, features, fontsize=8, color=INK_SECONDARY)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlim(0, None)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    figure.suptitle("Which features dominate the scores? (dashed line = equal share 1/7)", color=INK, fontsize=12)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_scoring(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    baseline_json: str | Path = DEFAULT_BASELINE_JSON,
    stage3_scores_csv: str | Path = DEFAULT_STAGE3_SCORES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Stage 3 baseline, checked for traceability.
    baseline = load_baseline(baseline_json)
    features = list(baseline.features)
    if features != load_feature_selection(selection_json, dataset_csv):
        raise ScoringError("baseline features differ from the Stage 2 feature selection")
    table = load_stage1_table(dataset_csv)
    usable = table[table["usable"].astype(bool)].reset_index(drop=True)
    calibration_mask = primary_calibration_mask(usable)
    refit = fit_baseline(usable[calibration_mask], features)
    parameter_error = max(
        max(abs(refit.parameters[f].median - baseline.parameters[f].median),
            abs(refit.parameters[f].scale - baseline.parameters[f].scale)) for f in features)
    if parameter_error > 1e-12 or refit.calibration_events != baseline.calibration_events:
        raise ScoringError("saved Stage 3 baseline does not match the primary calibration set")

    # Event context from Stage 3 (session, group, calibration membership).
    context = pd.read_csv(stage3_scores_csv)
    context = usable[["recording_file", "event_id"]].merge(context, on=["recording_file", "event_id"], how="left",
                                                            validate="one_to_one")
    if context["session"].isna().any():
        raise ScoringError("Stage 3 context is missing for some usable events")
    if not (context["calibration"].astype(bool).to_numpy() == calibration_mask.to_numpy()).all():
        raise ScoringError("Stage 3 calibration membership differs from usable_order 1-20")
    z = baseline.z_scores(usable)
    z_error = float(np.abs(z.to_numpy() - context[list(z.columns)].to_numpy()).max())

    # 1-3. Scores and contributions for every usable event (calibration events in-sample).
    scores = combined_scores(z, features)
    contributions = feature_contributions(z, features)
    influence = single_feature_influence(z, features)
    metadata = context[["recording_file", "event_id", "recording_timestamp", "date", "session",
                        "usable_order", "calibration", "group"]]
    event_scores = pd.concat([metadata, z, scores,
                              contributions[[c for c in contributions.columns if c.startswith(("abs_z_", "rms_share_"))]],
                              influence], axis=1)
    event_scores.insert(8, "baseline_version", baseline.version)
    event_scores.to_csv(output / "event_scores.csv", index=False)

    # 5. Leave-one-out calibration.
    calibration = usable[calibration_mask]
    loo, loo_parameters = leave_one_out_scores(calibration, features)
    in_sample = scores[calibration_mask.to_numpy()].reset_index(drop=True)
    for score in SCORES:
        loo[f"in_sample_{score}"] = in_sample[score].to_numpy()
    loo["in_sample_max_abs_z_feature"] = in_sample["max_abs_z_feature"].to_numpy()
    loo.to_csv(output / "leave_one_out_scores.csv", index=False)
    loo_parameters.to_csv(output / "leave_one_out_parameters.csv", index=False)
    loo_z = loo[[f"z_{f}" for f in features]]

    # 4. Group and session distributions.
    groups = context["group"].to_numpy()
    frames = {
        "calibration_in_sample": scores[calibration_mask.to_numpy()],
        "calibration_leave_one_out": loo,
        "calibration_session_other": scores[groups == "calibration_session_other"],
        "other_session": scores[groups == "other_session"],
    }
    influence_frames = {
        "calibration_in_sample": influence[calibration_mask.to_numpy()],
        "calibration_leave_one_out": single_feature_influence(loo_z, features),
        "calibration_session_other": influence[groups == "calibration_session_other"],
        "other_session": influence[groups == "other_session"],
    }
    group_summary = summarize_scores(frames)
    group_summary.to_csv(output / "group_summary.csv", index=False)
    influence_summary = pd.DataFrame([
        {"group": name, "metric": metric, **distribution_summary(frame[metric].dropna())}
        for name, frame in influence_frames.items() for metric in frame.columns
    ])
    influence_summary.to_csv(output / "single_feature_influence.csv", index=False)

    session_rows = []
    for session in sorted(context["session"].unique()):
        in_session = (context["session"] == session).to_numpy()
        compared = scores[in_session & ~calibration_mask.to_numpy()]
        for score in SCORES:
            session_rows.append({"session": session, "score": score,
                                 "n_usable": int(in_session.sum()),
                                 "n_calibration_excluded": int((in_session & calibration_mask.to_numpy()).sum()),
                                 **distribution_summary(compared[score])})
    session_summary = pd.DataFrame(session_rows)
    session_summary.to_csv(output / "session_summary.csv", index=False)

    # 3. Feature dominance: overall, by group and by session.
    loo_contributions = feature_contributions(loo_z, features)
    group_contributions = {
        "calibration_in_sample": contributions[calibration_mask.to_numpy()],
        "calibration_leave_one_out": loo_contributions,
        "calibration_session_other": contributions[groups == "calibration_session_other"],
        "other_session": contributions[groups == "other_session"],
    }
    session_contributions = {
        s: contributions[((context["session"] == s).to_numpy()) & ~calibration_mask.to_numpy()]
        for s in sorted(context["session"].unique())
    }
    contribution_summary = pd.concat([
        summarize_contributions({"all_usable_in_sample": contributions}, features, "overall"),
        summarize_contributions(group_contributions, features, "group"),
        summarize_contributions({k: v for k, v in session_contributions.items() if len(v)}, features, "session"),
    ], ignore_index=True)
    contribution_summary.to_csv(output / "feature_contributions.csv", index=False)

    # Cross-session inflation, decomposed per feature (exact for the group means).
    inflation = pd.concat([
        inflation_decomposition(z[groups == "other_session"], loo_z, features).assign(
            target="other_session", reference="calibration_leave_one_out"),
        inflation_decomposition(z[groups == "other_session"], z[groups == "calibration_session_other"], features).assign(
            target="other_session", reference="calibration_session_other"),
        inflation_decomposition(z[groups == "calibration_session_other"], loo_z, features).assign(
            target="calibration_session_other", reference="calibration_leave_one_out"),
    ], ignore_index=True)
    inflation.to_csv(output / "inflation_decomposition.csv", index=False)

    medians = {name: {score: float(frame[score].median()) for score in SCORES} for name, frame in frames.items()}
    means = {name: {score: float(frame[score].mean()) for score in SCORES} for name, frame in frames.items()}
    inflation_ratios = {
        score: {
            "other_over_loo_calibration_median": medians["other_session"][score] / medians["calibration_leave_one_out"][score],
            "other_over_same_session_median": medians["other_session"][score] / medians["calibration_session_other"][score],
            "same_session_over_loo_calibration_median":
                medians["calibration_session_other"][score] / medians["calibration_leave_one_out"][score],
            "other_over_in_sample_calibration_median": medians["other_session"][score] / medians["calibration_in_sample"][score],
        } for score in SCORES
    }
    in_vs_loo = {}
    for score in SCORES:
        a, b = loo[f"in_sample_{score}"].to_numpy(), loo[score].to_numpy()
        in_vs_loo[score] = {
            "in_sample_median": float(np.median(a)), "loo_median": float(np.median(b)),
            "in_sample_p95": float(np.percentile(a, 95)), "loo_p95": float(np.percentile(b, 95)),
            "in_sample_max": float(a.max()), "loo_max": float(b.max()),
            "median_ratio_loo_over_in_sample": float(np.median(b / a)),
            "fraction_loo_greater": float(np.mean(b > a)),
            "wilcoxon_p": float(stats.wilcoxon(b, a).pvalue),
        }
    score_rank_correlation = {
        name: frame[list(SCORES)].corr(method="spearman").round(6).to_dict()
        for name, frame in (("all_usable_in_sample", scores), ("other_session", frames["other_session"]))
    }
    pair = {
        "calibration_leave_one_out": pair_analysis(loo_z, loo_contributions, features),
        "calibration_session_other": pair_analysis(z[groups == "calibration_session_other"],
                                                   group_contributions["calibration_session_other"], features),
        "other_session": pair_analysis(z[groups == "other_session"], group_contributions["other_session"], features),
    }
    reference = iid_normal_reference(len(features))
    largest = event_scores.loc[event_scores["max_abs_z"].idxmax()]

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_score_distributions(frames, reference, output / "score_distributions.png")
        plot_feature_dominance(contribution_summary, features, output / "feature_dominance.png")

    result = {
        "stage": "Stage 4 - V1 candidate combined scores (no threshold, no labels)",
        "scores": {"mean_abs_z": "mean_j |z_j|", "rms_z": "sqrt(mean_j z_j^2)", "max_abs_z": "max_j |z_j|",
                   "features": features, "d": len(features)},
        "baseline": {"path": _relative(Path(baseline_json)), "sha256": _sha256(Path(baseline_json)),
                     "version": baseline.version, "refit_max_abs_parameter_difference": parameter_error,
                     "max_abs_z_difference_vs_stage3": z_error},
        "dataset": {"path": _relative(Path(dataset_csv)), "sha256": _sha256(Path(dataset_csv)),
                    "usable_events": int(len(usable))},
        "provenance": {"git": _git_state(), "reference_seed": REFERENCE_SEED, "reference_draws": REFERENCE_DRAWS},
        "group_sizes": {name: int(len(frame)) for name, frame in frames.items()},
        "group_medians": medians,
        "group_means": means,
        "iid_standard_normal_reference": reference,
        "inflation_ratios": inflation_ratios,
        "in_sample_vs_leave_one_out": in_vs_loo,
        "score_rank_correlation": score_rank_correlation,
        "pair_mean_rms_spectral_centroid_mean": pair,
        "isolated_extreme_events_all_usable": isolated_extreme_ranks(scores, influence),
        "largest_max_abs_z_event": {
            "recording_file": largest["recording_file"], "event_id": int(largest["event_id"]),
            "session": largest["session"], "feature": largest["max_abs_z_feature"],
            **{score: float(largest[score]) for score in SCORES},
            **{f"{score}_rank_among_318": int((event_scores[score] >= largest[score]).sum()) for score in SCORES},
        },
        "no_threshold_or_labels": True,
        "outputs": sorted(p.name for p in output.iterdir()),
    }
    with (output / "scoring_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V1 candidate combined scores (Stage 4; no threshold)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_scoring(output_dir=args.output_dir, make_plots=not args.no_plots)
    print(json.dumps({k: outcome[k] for k in ("group_medians", "baseline")}, indent=2, default=str))
