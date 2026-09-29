"""V1 robust per-feature baseline (Stage 3; PRISM_RESEARCH_LOG.md, Entry 5).

For each feature j listed as KEEP in
``results/feature_analysis/feature_selection_v1.json``:

    median_j = median of the calibration values
    MAD_j    = median(|x_i - median_j|) over the calibration values
    scale_j  = 1.4826 * MAD_j      (MAD scaled to estimate the SD for normal data)
    z_j(x)   = (x - median_j) / scale_j

The primary calibration set is the first 20 usable events (``usable_order``
1-20 in the Stage 1 table).  The baseline is frozen: updating it means fitting
a new one.  This stage only exposes per-feature robust z-scores (calibration
deviations).  It does not combine features, choose thresholds or label events.

Sensitivity analyses refit the same baseline on alternative calibration sets
to show how much the z-scores depend on which events are used.  They never
change the primary baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import config
from feature_analysis import (
    CALIBRATION_SIZE,
    DEFAULT_DATASET_CSV,
    EXTREME_Z,
    MAD_SCALE,
    MIN_GROUP_SIZE,
    assign_sessions,
    load_stage1_table,
)
from inhale_dataset import (
    DETECTION_COLUMNS,
    FEATURE_COLUMNS,
    _git_state,
    _json_default,
    _relative,
    _sha256,
    parse_recording_timestamp,
)


DEFAULT_SELECTION_JSON = Path(config.RESULTS_DIR) / "feature_analysis" / "feature_selection_v1.json"
DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "baseline_v1"
BASELINE_VERSION = "v1"
SENSITIVITY_SEED = 20260930
SENSITIVITY_DRAWS = 2000
SINGLE_SESSION_MIN_EVENTS = 2 * CALIBRATION_SIZE  # 20 to calibrate, at least 20 left to compare
GROUPS = ("calibration", "calibration_session_other", "other_session")


class BaselineError(ValueError):
    """The baseline cannot be fitted or applied safely."""


# ── Feature selection ───────────────────────────────────────────────────────

def load_feature_selection(
    path: str | Path = DEFAULT_SELECTION_JSON,
    dataset_csv: str | Path | None = None,
) -> list[str]:
    """KEEP features from the Stage 2 selection file, validated.

    When ``dataset_csv`` is given, its hash must match the dataset the
    selection was made on.
    """
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    features = list(selection.get("keep", []))
    if not features:
        raise BaselineError("feature selection lists no KEEP features")
    if len(set(features)) != len(features):
        raise BaselineError(f"duplicate features in selection: {features}")
    if set(features) & set(DETECTION_COLUMNS):
        raise BaselineError("CNN detection columns cannot be baseline features")
    unknown = [f for f in features if f not in FEATURE_COLUMNS]
    if unknown:
        raise BaselineError(f"unknown features in selection: {unknown}")
    for feature in features:
        transform = selection.get("transforms", {}).get(feature, "none")
        if transform != "none":
            raise BaselineError(f"{feature}: transform {transform!r} is not supported by the V1 baseline")
    if dataset_csv is not None:
        recorded = selection.get("dataset", {}).get("sha256")
        if recorded and recorded != _sha256(Path(dataset_csv)):
            raise BaselineError("feature selection was made on a different dataset (sha256 mismatch)")
    return features


# ── Baseline ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureBaseline:
    median: float
    mad: float
    scale: float


@dataclass(frozen=True)
class RobustBaseline:
    """Frozen per-feature median / MAD baseline; refit to update it."""

    features: tuple[str, ...]
    parameters: Mapping[str, FeatureBaseline]
    n_calibration: int
    calibration_events: tuple[tuple[str, int], ...] = ()
    mad_scale: float = MAD_SCALE
    version: str = BASELINE_VERSION

    def z_scores(self, table: pd.DataFrame) -> pd.DataFrame:
        """Per-feature robust z-scores as columns ``z_<feature>``, indexed like ``table``."""
        missing = [f for f in self.features if f not in table.columns]
        if missing:
            raise BaselineError(f"missing feature columns: {missing}")
        values = table[list(self.features)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad = [f for f, flag in zip(self.features, ~np.isfinite(values).all(axis=0)) if flag]
            raise BaselineError(f"non-finite feature values in {bad}; robust z-scores are undefined")
        medians = np.array([self.parameters[f].median for f in self.features])
        scales = np.array([self.parameters[f].scale for f in self.features])
        return pd.DataFrame((values - medians) / scales, index=table.index,
                            columns=[f"z_{f}" for f in self.features])

    def parameters_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"feature": f, **asdict(self.parameters[f]), "n_calibration": self.n_calibration,
             "mad_scale": self.mad_scale, "baseline_version": self.version}
            for f in self.features
        ])

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mad_scale": self.mad_scale,
            "features": list(self.features),
            "parameters": {f: asdict(self.parameters[f]) for f in self.features},
            "n_calibration": self.n_calibration,
            "calibration_events": [list(key) for key in self.calibration_events],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RobustBaseline":
        features = tuple(data["features"])
        return cls(
            features=features,
            parameters={f: FeatureBaseline(**data["parameters"][f]) for f in features},
            n_calibration=int(data["n_calibration"]),
            calibration_events=tuple((str(r), int(e)) for r, e in data.get("calibration_events", [])),
            mad_scale=float(data["mad_scale"]),
            version=str(data["version"]),
        )


def fit_baseline(
    calibration: pd.DataFrame,
    features: Sequence[str],
    mad_scale: float = MAD_SCALE,
    version: str = BASELINE_VERSION,
) -> RobustBaseline:
    """Fit median / MAD per feature on ``calibration`` only."""
    features = tuple(features)
    missing = [f for f in features if f not in calibration.columns]
    if missing:
        raise BaselineError(f"missing feature columns: {missing}")
    if calibration.empty:
        raise BaselineError("calibration set is empty")
    values = calibration[list(features)].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.all():
        bad = [f for f, ok in zip(features, finite.all(axis=0)) if not ok]
        raise BaselineError(f"non-finite calibration values in {bad}")
    medians = np.median(values, axis=0)
    mads = np.median(np.abs(values - medians), axis=0)
    for feature, value in zip(features, mads):
        if not np.isfinite(value) or value <= 0:
            raise BaselineError(f"MAD of {feature} is {value}; robust z-scores would be undefined")
    keys = ()
    if {"recording_file", "event_id"} <= set(calibration.columns):
        keys = tuple(zip(calibration["recording_file"].astype(str), calibration["event_id"].astype(int)))
    return RobustBaseline(
        features=features,
        parameters={f: FeatureBaseline(float(m), float(d), float(mad_scale * d))
                    for f, m, d in zip(features, medians, mads)},
        n_calibration=int(len(calibration)),
        calibration_events=keys,
        mad_scale=mad_scale,
        version=version,
    )


def primary_calibration_mask(table: pd.DataFrame, size: int = CALIBRATION_SIZE) -> pd.Series:
    """The locked V1 calibration set: usable events with usable_order 1..size."""
    mask = table["usable"].astype(bool) & table["usable_order"].fillna(0).between(1, size)
    if int(mask.sum()) != size:
        raise BaselineError(f"expected {size} calibration events, found {int(mask.sum())}")
    return mask


def load_baseline(path: str | Path) -> RobustBaseline:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RobustBaseline.from_dict(data["baseline"] if "baseline" in data else data)


# ── Context and groups ──────────────────────────────────────────────────────

def recording_sessions(data_dir: str | Path = config.DATA_DIR) -> dict[str, str]:
    """Session label per WAV, using the Stage 2 definition (gaps > 25 min)."""
    names = [path.name for path in sorted(Path(data_dir).glob("*.wav"))]
    if not names:
        raise FileNotFoundError(f"No WAV recordings found in {data_dir}")
    sessions = assign_sessions(pd.Series([parse_recording_timestamp(name) for name in names]))
    return dict(zip(names, sessions))


def comparison_groups(sessions: pd.Series, calibration_mask: pd.Series) -> pd.Series:
    """calibration / calibration_session_other / other_session."""
    in_calibration_session = sessions.isin(set(sessions[calibration_mask]))
    labels = np.select([calibration_mask.to_numpy(), in_calibration_session.to_numpy()], GROUPS[:2], GROUPS[2])
    return pd.Series(labels, index=sessions.index, name="group")


def _robust_sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return MAD_SCALE * np.median(np.abs(values - median), axis=axis)


def summarize_z_by_group(z: pd.DataFrame, groups: pd.Series, features: Sequence[str]) -> pd.DataFrame:
    """Location, spread and tail of each feature's z-scores per comparison group.

    |z| > 3.5 is the Stage 2 extreme-value convention, reported descriptively;
    it is not an anomaly threshold.
    """
    calibration = z[groups == "calibration"]
    rows = []
    for group in GROUPS:
        subset = z[groups == group]
        for feature in features:
            values = subset[f"z_{feature}"].to_numpy(dtype=float)
            if values.size == 0:
                rows.append({"group": group, "feature": feature, "n": 0})
                continue
            absolute = np.abs(values)
            low, high = calibration[f"z_{feature}"].min(), calibration[f"z_{feature}"].max()
            rows.append({
                "group": group, "feature": feature, "n": int(values.size),
                "median_z": float(np.median(values)),
                "robust_sd_z": float(_robust_sd(values)),
                "median_abs_z": float(np.median(absolute)),
                "p90_abs_z": float(np.percentile(absolute, 90)),
                "p95_abs_z": float(np.percentile(absolute, 95)),
                "max_abs_z": float(absolute.max()),
                "frac_abs_z_gt_3_5": float(np.mean(absolute > EXTREME_Z)),
                "frac_outside_calibration_z_range": float(np.mean((values < low) | (values > high))),
            })
    return pd.DataFrame(rows)


def summarize_z_by_session(
    z: pd.DataFrame, sessions: pd.Series, calibration_mask: pd.Series, features: Sequence[str],
) -> pd.DataFrame:
    """Per session: calibration events excluded, median / robust SD / median |z| per feature."""
    rows = []
    for session in sorted(sessions.unique()):
        in_session = sessions == session
        subset = z[in_session & ~calibration_mask]
        for feature in features:
            values = subset[f"z_{feature}"].to_numpy(dtype=float)
            rows.append({
                "session": session, "feature": feature,
                "n_events": int(in_session.sum()),
                "n_calibration_events": int((in_session & calibration_mask).sum()),
                "n_compared": int(values.size),
                "median_z": float(np.median(values)) if values.size else float("nan"),
                "robust_sd_z": float(_robust_sd(values)) if values.size else float("nan"),
                "median_abs_z": float(np.median(np.abs(values))) if values.size else float("nan"),
            })
    return pd.DataFrame(rows)


def session_decomposition(session_summary: pd.DataFrame, min_events: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Within-session spread vs between-session offsets of non-calibration z-scores.

    ``median_within_session_robust_sd_z`` > 1 means sessions are internally more
    variable than the calibration set; a large ``between_session_robust_sd_of_medians``
    means sessions sit at different offsets from the calibration median.
    """
    tested = session_summary[session_summary["n_compared"] >= min_events]
    rows = []
    for feature, group in tested.groupby("feature", sort=False):
        medians = group["median_z"].to_numpy()
        rows.append({
            "feature": feature,
            "sessions": int(len(group)),
            "median_within_session_robust_sd_z": float(group["robust_sd_z"].median()),
            "between_session_robust_sd_of_medians": float(_robust_sd(medians)),
            "session_median_z_min": float(medians.min()),
            "session_median_z_max": float(medians.max()),
        })
    return pd.DataFrame(rows)


# ── Calibration sensitivity (never changes the primary baseline) ───────────

def random_calibration_sets(n_events: int, size: int, draws: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_events, size=size, replace=False)) for _ in range(draws)]


def session_spread_set(sessions: np.ndarray, order: np.ndarray, size: int, session_order: Sequence[str]) -> np.ndarray:
    """Round-robin over sessions: each pass takes the next event of every session in turn."""
    queues = {s: list(np.flatnonzero(sessions == s)[np.argsort(order[sessions == s], kind="mergesort")])
              for s in session_order}
    chosen: list[int] = []
    depth = 0
    while len(chosen) < size and any(len(q) > depth for q in queues.values()):
        for session in session_order:
            if len(chosen) == size:
                break
            if len(queues[session]) > depth:
                chosen.append(queues[session][depth])
        depth += 1
    if len(chosen) < size:
        raise BaselineError(f"only {len(chosen)} events available for a session-spread calibration")
    return np.sort(np.array(chosen))


def random_session_spread_sets(sessions: np.ndarray, size: int, draws: int, seed: int) -> list[np.ndarray]:
    """Session-spread sets with random session order and random events within sessions."""
    rng = np.random.default_rng(seed)
    labels = np.unique(sessions)
    sets = []
    for _ in range(draws):
        random_order = rng.permutation(len(sessions)).astype(float)
        sets.append(session_spread_set(sessions, random_order, size, list(rng.permutation(labels))))
    return sets


def calibration_design_stats(
    values: np.ndarray,
    sessions: np.ndarray,
    calibration_sets: Sequence[np.ndarray],
    features: Sequence[str],
    design: str,
) -> pd.DataFrame:
    """Parameters of each calibration set and the z-scores they imply for the other events.

    ``scale_ratio`` is the calibration scale over the scale of all events;
    ``median_shift`` is the calibration median minus the median of all events,
    in units of the all-event scale.  z-scores of the non-calibration events are
    summarised overall and split by whether the event's session contributed to
    the calibration set.
    """
    full_median = np.median(values, axis=0)
    full_scale = MAD_SCALE * np.median(np.abs(values - full_median), axis=0)
    rows = []
    for draw, index in enumerate(calibration_sets):
        calibration = values[index]
        median = np.median(calibration, axis=0)
        scale = MAD_SCALE * np.median(np.abs(calibration - median), axis=0)
        rest = np.ones(len(values), bool)
        rest[index] = False
        calibration_sessions = np.unique(sessions[index])
        same = rest & np.isin(sessions, calibration_sessions)
        z = (values - median) / scale
        for group, mask in (("all_non_calibration", rest), ("same_session", same), ("other_session", rest & ~same)):
            if mask.any():
                group_z = z[mask]
                median_z, robust_sd, median_abs = np.median(group_z, 0), _robust_sd(group_z), np.median(np.abs(group_z), 0)
            else:
                median_z = robust_sd = median_abs = np.full(len(features), np.nan)
            for j, feature in enumerate(features):
                rows.append({
                    "design": design, "draw": draw, "group": group, "feature": feature,
                    "n_calibration_sessions": int(len(calibration_sessions)), "n_compared": int(mask.sum()),
                    "scale_ratio": scale[j] / full_scale[j],
                    "median_shift": (median[j] - full_median[j]) / full_scale[j],
                    "median_z": median_z[j], "robust_sd_z": robust_sd[j], "median_abs_z": median_abs[j],
                })
    return pd.DataFrame(rows)


def summarize_designs(stats: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    """Quantiles per design / group / feature, plus where the primary design falls among random draws."""
    rows = []
    metrics = ("scale_ratio", "median_shift", "median_z", "robust_sd_z", "median_abs_z")
    primary = primary.set_index(["group", "feature"])
    random_draws = stats[stats["design"] == "random_20"]
    for (design, group, feature), block in stats.groupby(["design", "group", "feature"], sort=False):
        row = {"design": design, "group": group, "feature": feature, "draws": int(block["draw"].nunique()),
               "median_n_calibration_sessions": float(block["n_calibration_sessions"].median()),
               "median_n_compared": float(block["n_compared"].median())}
        for metric in metrics:
            values = block[metric].dropna().to_numpy()
            for q in (5, 50, 95):
                row[f"{metric}_p{q:02d}"] = float(np.percentile(values, q)) if values.size else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Where the primary first-20 set sits among random 20-event sets.
    reference = random_draws[random_draws["group"] == "all_non_calibration"].set_index("feature")
    percentile = {}
    for feature in primary.index.get_level_values("feature").unique():
        value = primary.loc[("all_non_calibration", feature), "scale_ratio"]
        percentile[feature] = float(np.mean(reference.loc[feature, "scale_ratio"] <= value + 1e-12))
    summary["primary_scale_ratio_percentile_among_random"] = summary["feature"].map(percentile)
    return summary


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_INK = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
GROUP_LABELS = {
    "calibration": "Calibration (first 20)",
    "calibration_session_other": "Same sessions, not calibration",
    "other_session": "Other sessions",
}


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_INK)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def plot_z_by_group(z: pd.DataFrame, groups: pd.Series, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(2, 4, figsize=(15, 8), facecolor=SURFACE, layout="constrained")
    for axis, feature in zip(axes.flat, features):
        _style(axis)
        data = [z.loc[groups == g, f"z_{feature}"].to_numpy() for g in GROUPS]
        parts = axis.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                             medianprops={"color": INK, "linewidth": 1.2},
                             whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
                             flierprops={"marker": "o", "markersize": 3, "markeredgecolor": INK_MUTED,
                                         "markerfacecolor": "none"})
        for box, colour in zip(parts["boxes"], SERIES):
            box.set_facecolor(colour)
            box.set_edgecolor(SURFACE)
        axis.axhline(0, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
        axis.set_xticks([1, 2, 3], [f"n={len(d)}" for d in data], fontsize=8, color=INK_SECONDARY)
        axis.set_title(feature, fontsize=9, color=INK)
        axis.set_ylabel("robust z (first-20 baseline)", fontsize=8, color=INK_SECONDARY)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    legend_axis.legend([Patch(facecolor=c) for c in SERIES], [GROUP_LABELS[g] for g in GROUPS],
                       loc="center", frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Per-feature robust z-scores against the primary first-20 baseline", color=INK, fontsize=12)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_scale_sensitivity(summary: pd.DataFrame, singles: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    block = summary[summary["group"] == "all_non_calibration"].set_index(["design", "feature"])
    figure, axis = plt.subplots(figsize=(9, 6), facecolor=SURFACE, layout="constrained")
    _style(axis)
    y = np.arange(len(features))[::-1]
    for offset, design, colour, label in ((0.18, "random_20", SERIES[0], "Random 20 (5-95%, median)"),
                                           (-0.02, "session_spread_random", SERIES[1],
                                            "Session-spread random 20 (5-95%, median)")):
        low = [block.loc[(design, f), "scale_ratio_p05"] for f in features]
        mid = [block.loc[(design, f), "scale_ratio_p50"] for f in features]
        high = [block.loc[(design, f), "scale_ratio_p95"] for f in features]
        axis.hlines(y + offset, low, high, color=colour, linewidth=2)
        axis.plot(mid, y + offset, "o", color=colour, markersize=6, label=label)
    primary = [block.loc[("primary_first_20", f), "scale_ratio_p50"] for f in features]
    axis.plot(primary, y - 0.2, "D", color=SERIES[2], markersize=7, label="Primary first 20")
    for session, rows in singles[singles["group"] == "all_non_calibration"].groupby("design"):
        values = rows.set_index("feature").loc[list(features), "scale_ratio"]
        axis.plot(values, y - 0.2, "x", color=INK_MUTED, markersize=6, markeredgewidth=1.5)
    axis.plot([], [], "x", color=INK_MUTED, markersize=6, markeredgewidth=1.5,
              label="First 20 of one large session (3 sessions)")
    axis.axvline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    ticks = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
    axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axis.xaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_yticks(y, features, fontsize=8, color=INK_SECONDARY)
    axis.set_xlabel("Calibration scale / scale of all usable events (log axis)", color=INK_SECONDARY, fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8,
                labelcolor=INK_SECONDARY)
    axis.set_title("How narrow is the calibration spread under different 20-event calibration sets?",
                   color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def run_baseline(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    draws: int = SENSITIVITY_DRAWS,
    seed: int = SENSITIVITY_SEED,
    make_plots: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = load_feature_selection(selection_json, dataset_csv)
    table = load_stage1_table(dataset_csv)
    usable = table[table["usable"].astype(bool)].reset_index(drop=True)
    usable["date"] = usable["recording_timestamp"].dt.strftime("%Y-%m-%d")
    usable["session"] = usable["recording_file"].map(recording_sessions(data_dir))
    if usable["session"].isna().any():
        raise BaselineError("some usable events have no session")
    calibration_mask = primary_calibration_mask(usable)

    # Primary baseline: fitted on the 20 calibration events only.
    baseline = fit_baseline(usable[calibration_mask], features)
    parameters = baseline.parameters_frame()
    full = fit_baseline(usable, features)
    parameters["full_set_median"] = [full.parameters[f].median for f in features]
    parameters["full_set_scale"] = [full.parameters[f].scale for f in features]
    parameters["scale_ratio_calibration_over_full"] = parameters["scale"] / parameters["full_set_scale"]
    parameters["median_shift_in_full_scale"] = (parameters["median"] - parameters["full_set_median"]) / parameters["full_set_scale"]
    parameters.to_csv(output / "baseline_parameters.csv", index=False)

    groups = comparison_groups(usable["session"], calibration_mask)
    z = baseline.z_scores(usable)
    context = usable[["recording_file", "event_id", "recording_timestamp", "date", "session", "usable_order"]].copy()
    context["recording_timestamp"] = context["recording_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    context["usable_order"] = context["usable_order"].astype(int)
    context["calibration"] = calibration_mask.to_numpy()
    context["group"] = groups.to_numpy()
    absolute = z.abs().rename(columns=lambda c: c.replace("z_", "abs_z_", 1))
    pd.concat([context, z, absolute], axis=1).to_csv(output / "robust_z_scores.csv", index=False)

    calibration_events = usable.loc[calibration_mask, ["usable_order", "recording_file", "event_id", "session", *features]]
    calibration_events.to_csv(output / "calibration_events.csv", index=False)

    group_summary = summarize_z_by_group(z, groups, features)
    group_summary.to_csv(output / "group_z_summary.csv", index=False)
    session_summary = summarize_z_by_session(z, usable["session"], calibration_mask, features)
    session_summary.to_csv(output / "session_z_summary.csv", index=False)
    decomposition = session_decomposition(session_summary)
    decomposition.to_csv(output / "session_decomposition.csv", index=False)

    # Sensitivity analyses: alternative calibration sets, primary baseline unchanged.
    values = usable[features].to_numpy(dtype=float)
    sessions = usable["session"].to_numpy()
    order = usable["usable_order"].to_numpy(dtype=float)
    chronological_sessions = list(usable.sort_values("usable_order")["session"].drop_duplicates())
    primary_index = np.flatnonzero(calibration_mask.to_numpy())
    designs = [
        calibration_design_stats(values, sessions, [primary_index], features, "primary_first_20"),
        calibration_design_stats(values, sessions, random_calibration_sets(len(usable), CALIBRATION_SIZE, draws, seed),
                                 features, "random_20"),
        calibration_design_stats(values, sessions, random_session_spread_sets(sessions, CALIBRATION_SIZE, draws, seed + 1),
                                 features, "session_spread_random"),
        calibration_design_stats(values, sessions,
                                 [session_spread_set(sessions, order, CALIBRATION_SIZE, chronological_sessions)],
                                 features, "session_spread_chronological"),
    ]
    session_sizes = usable["session"].value_counts()
    large_sessions = sorted(session_sizes[session_sizes >= SINGLE_SESSION_MIN_EVENTS].index)
    singles = []
    for session in large_sessions:
        members = usable[usable["session"] == session].sort_values("usable_order").index[:CALIBRATION_SIZE]
        singles.append(calibration_design_stats(values, sessions, [members.to_numpy()], features,
                                                f"single_session_first_20:{session}"))
    singles = pd.concat(singles) if singles else pd.DataFrame()
    stats = pd.concat([*designs, singles])
    stats[(stats["group"] == "all_non_calibration") & stats["design"].isin(["random_20", "session_spread_random"])].to_csv(
        output / "sensitivity_draws.csv", index=False, float_format="%.6g")
    design_summary = summarize_designs(stats, designs[0])
    design_summary.to_csv(output / "calibration_sensitivity.csv", index=False)

    # Calibration summary: the primary set's parameters against random 20-event sets.
    calibration_summary = parameters[["feature", "median", "mad", "scale", "full_set_median", "full_set_scale",
                                      "scale_ratio_calibration_over_full", "median_shift_in_full_scale"]].copy()
    random_block = design_summary[(design_summary["design"] == "random_20")
                                  & (design_summary["group"] == "all_non_calibration")].set_index("feature")
    calibration_summary["random_20_scale_ratio_p05"] = calibration_summary["feature"].map(random_block["scale_ratio_p05"])
    calibration_summary["random_20_scale_ratio_p95"] = calibration_summary["feature"].map(random_block["scale_ratio_p95"])
    calibration_summary["scale_ratio_percentile_among_random_20"] = calibration_summary["feature"].map(
        random_block["primary_scale_ratio_percentile_among_random"])
    calibration_summary["calibration_min"] = [calibration_events[f].min() for f in features]
    calibration_summary["calibration_max"] = [calibration_events[f].max() for f in features]
    calibration_summary.to_csv(output / "calibration_summary.csv", index=False)

    # Consistency with Stage 2.
    stage2 = Path(config.RESULTS_DIR) / "feature_analysis"
    checks = {}
    if (stage2 / "calibration_vs_rest.csv").exists():
        reference = pd.read_csv(stage2 / "calibration_vs_rest.csv").set_index("feature")
        checks["max_abs_diff_vs_stage2_calibration_median"] = float(max(
            abs(baseline.parameters[f].median - reference.loc[f, "median_calibration"]) for f in features))
        checks["max_abs_diff_vs_stage2_calibration_scale"] = float(max(
            abs(baseline.parameters[f].scale - reference.loc[f, "mad_scaled_calibration"]) for f in features))
    if (stage2 / "session_group_medians.csv").exists():
        reference = pd.read_csv(stage2 / "session_group_medians.csv")
        reference = reference[(reference["grouping"] == "session") & (reference["feature"] == features[0])]
        checks["session_counts_match_stage2"] = bool(
            reference.set_index("group")["n"].sort_index().equals(session_sizes.sort_index().rename("n")))

    selection_sha = _sha256(Path(selection_json))
    dataset_sha = _sha256(Path(dataset_csv))
    with (output / "baseline_v1.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "baseline": baseline.to_dict(),
            "calibration_definition": f"usable == True and usable_order in 1..{CALIBRATION_SIZE} (Stage 1 table)",
            "robust_z": "z = (x - median) / (mad_scale * MAD), MAD = median(|x - median|) over calibration events",
            "feature_selection": {"path": _relative(Path(selection_json)), "sha256": selection_sha},
            "dataset": {"path": _relative(Path(dataset_csv)), "sha256": dataset_sha},
        }, handle, indent=2)

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_z_by_group(z, groups, features, output / "z_by_group.png")
        plot_scale_sensitivity(design_summary, singles, features, output / "calibration_scale_sensitivity.png")

    # Same-session vs other-session median |z| for every single-set design that has both groups.
    same_vs_other = {}
    for design, block in design_summary[design_summary["draws"] == 1].groupby("design"):
        values_by_group = block.pivot(index="feature", columns="group", values="median_abs_z_p50")
        if {"same_session", "other_session"} <= set(values_by_group.columns) and \
                values_by_group[["same_session", "other_session"]].notna().all().all():
            same_vs_other[design] = {
                "features_closer_within_session": int((values_by_group["same_session"]
                                                       < values_by_group["other_session"]).sum()),
                "median_abs_z": values_by_group[["same_session", "other_session"]].loc[features].to_dict("index"),
            }

    group_counts = groups.value_counts().to_dict()
    result = {
        "stage": "Stage 3 - V1 robust baseline",
        "dataset": {"path": _relative(Path(dataset_csv)), "sha256": dataset_sha, "usable_events": int(len(usable))},
        "feature_selection": {"path": _relative(Path(selection_json)), "sha256": selection_sha, "features": features},
        "calibration": {
            "definition": f"usable_order 1..{CALIBRATION_SIZE}",
            "sessions": dict(usable.loc[calibration_mask, "session"].value_counts()),
            "groups": group_counts,
        },
        "provenance": {"git": _git_state(), "sensitivity_seed": seed, "sensitivity_draws": draws,
                       "session_spread_random_seed": seed + 1, "mad_scale": MAD_SCALE},
        "consistency_checks": checks,
        "single_session_designs": large_sessions,
        "same_vs_other_session_median_abs_z": same_vs_other,
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    with (output / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the V1 robust baseline and write robust z-scores (Stage 3)")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_baseline(dataset_csv=args.dataset, selection_json=args.selection,
                           output_dir=args.output_dir, make_plots=not args.no_plots)
    print(json.dumps({key: outcome[key] for key in ("calibration", "consistency_checks")}, indent=2, default=str))
