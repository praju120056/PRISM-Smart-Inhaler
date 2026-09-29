"""Stage 2 feature analysis for the V1 robust baseline.

See PRISM_RESEARCH_LOG.md, Entry 4.  Input is the Stage 1 table
``results/inhale_dataset/inhale_events_v1.csv``, which is read and never
modified.  The 13 candidate features are ``inhale_dataset.FEATURE_COLUMNS``;
the CNN detection columns are never treated as candidate features.

The analysis is descriptive:

* distributions and robust-scale (median / MAD) suitability;
* Spearman redundancy and how predictable each feature is from the others;
* variation between recording dates and recording sessions;
* how the first 20 usable events (the V1 calibration set) compare with the
  rest of the usable events;
* the RMS-envelope edge effect behind ``time_to_peak_s`` / ``peak_rms``, and
  how much segment-edge frames contribute to the ``*_std`` features.

The V1 KEEP / EXCLUDE / DEFER decisions live in :data:`V1_FEATURE_DECISIONS`
and are written next to the evidence.  Annotations are only used to interpret
measurements (``annotation_context.csv``), never to select features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config
from inhale_dataset import (
    DATASET_FILENAME,
    DEFAULT_OUTPUT_DIR as STAGE1_OUTPUT_DIR,
    DETECTION_COLUMNS,
    FEATURE_COLUMNS,
    STRIDE_S,
    WINDOW_S,
    _git_state,
    _json_default,
    _relative,
    _sha256,
    parse_recording_timestamp,
)


DEFAULT_DATASET_CSV = STAGE1_OUTPUT_DIR / DATASET_FILENAME
DEFAULT_AUDIT_CSV = STAGE1_OUTPUT_DIR / "event_annotation_audit.csv"
DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "feature_analysis"

MAD_SCALE = 1.4826      # makes the MAD consistent with the SD for normal data
EXTREME_Z = 3.5         # Iglewicz & Hoaglin (1993) modified-z cut-off (external convention)
CALIBRATION_SIZE = 20   # locked V1 calibration size (Entry 3)
MIN_GROUP_SIZE = 5      # dates / sessions with fewer usable events are described, not tested
SESSION_GAP_MIN = 25.0  # recordings further apart than this start a new session (Entry 4)
SUBSAMPLE_DRAWS = 2000
SEED = 20260929
NEAR_PEAK_FRACTION = 0.95
REDUNDANCY_RHO = 0.8    # descriptive grouping of |rho| >= 0.8; never an exclusion rule by itself

# Frame geometry of the measurements being checked.
RMS_FRAME = config.LIBROSA_N_FFT             # post_event._rms_envelope frame length
RMS_HOP = config.LIBROSA_HOP_LENGTH
STFT_PAD = config.LIBROSA_N_FFT // 2         # librosa.stft(center=True) zero-pads n_fft // 2
ZCR_FRAME = 2048                             # librosa.feature.zero_crossing_rate default frame_length
SPECTRAL_NAMES = ("spectral_centroid", "spectral_flatness", "spectral_rolloff", "zcr")
PEAK_POSITIONS = ("interior", "first", "last", "partial_tail")

DECISIONS = ("KEEP", "EXCLUDE", "DEFER")


@dataclass(frozen=True)
class FeatureDecision:
    """V1 decision for one candidate feature (only KEEP features enter V1)."""

    decision: str
    transform: str
    reason: str


# Decisions are made from the evidence written by run_feature_analysis; the
# rationale is expanded in PRISM_RESEARCH_LOG.md Entry 4.
V1_FEATURE_DECISIONS: dict[str, FeatureDecision] = {
    "duration_s": FeatureDecision(
        "KEEP", "none",
        "Acoustic inhalation length as segmented by the detector (16 ms resolution = 8% of its MAD). "
        "Non-degenerate MAD, near-symmetric body (Bowley 0.08), 13/318 events beyond |z| 3.5 on both "
        "sides; little of it is predictable from the other kept features (rank R2 0.32). Strong "
        "between-date/session variation is flagged, not used for exclusion.",
    ),
    "mean_rms": FeatureDecision(
        "KEEP", "none",
        "Sustained loudness (mean of 32 ms frame RMS). Symmetric body (Bowley -0.02), 2 extremes; log "
        "compared (skew 1.25 -> -0.31, extremes 2 -> 3) and not adopted. Relative level only: depends "
        "on gain and distance. Strongly anti-correlated with spectral_centroid_mean (rho -0.84, -0.76 "
        "within sessions): kept as a distinct construct, flagged for double counting in a univariate "
        "aggregate.",
    ),
    "peak_rms": FeatureDecision(
        "DEFER", "none",
        "Loudest single 32 ms frame. Heaviest tail of all candidates (skew 2.90, max |z| 10.9; the 5 "
        "most extreme events carry 49% of the squared robust deviation) and log does not remove it "
        "(skew 1.04). The 8 extreme events are mostly interior transients (median crest factor 2.44 "
        "vs 1.50) of unidentified source; 77% of its rank variation is explained by the kept "
        "features. Investigate the transients; revisit as a crest-factor descriptor.",
    ),
    "total_energy": FeatureDecision(
        "EXCLUDE", "none",
        "Derived quantity (integral of squared amplitude = duration x mean power): log duration_s and "
        "log mean_rms explain 98.1% of log total_energy. Information lost: <2% of its log variance "
        "(within-event amplitude modulation). Keeping it would double-count duration and loudness.",
    ),
    "time_to_peak_s": FeatureDecision(
        "DEFER", "none",
        "Argmax of a flat-topped RMS envelope: frames within 5% of the peak span a median 0.38 s, "
        "close to the feature's own robust spread (0.44 s), so much of its between-event variation "
        "may be peak-location noise; it also inherits the detector's start boundary. The segment-edge "
        "effect is rare in usable events (3/318). Revisit with a smoothed-envelope or energy-centroid "
        "timing.",
    ),
    "spectral_centroid_mean": FeatureDecision(
        "KEEP", "none",
        "Representative spectral-brightness measure: smallest relative MAD (5%), symmetric (skew "
        "0.15), 1 extreme. Preferred over spectral_rolloff_mean and zcr_mean, which estimate the same "
        "construct (rho 0.81, 0.85). See mean_rms for the loudness-brightness flag.",
    ),
    "spectral_centroid_std": FeatureDecision(
        "KEEP", "none",
        "Within-event variability of brightness: weakly related to every other feature (max |rho| "
        "0.36, rank R2 from the other kept 0.23), most session-stable feature (session eps2 0.04), "
        "zero-padded edge frames contribute ~2% (interior/all ratio 0.98).",
    ),
    "spectral_flatness_mean": FeatureDecision(
        "KEEP", "none",
        "Noise-likeness (tonality), a construct distinct from brightness and loudness (strongest "
        "partner is the deferred rolloff, rho 0.73); symmetric body, 1 extreme. Strong between-date "
        "variation (eps2 0.17) and calibration shift (+1.08 rest MAD) flagged.",
    ),
    "spectral_flatness_std": FeatureDecision(
        "KEEP", "none",
        "Within-event variability of noise-likeness: moderate correlations only (max |rho| 0.50), "
        "2 extremes, not edge-driven (interior/all ratio 0.97).",
    ),
    "spectral_rolloff_mean": FeatureDecision(
        "DEFER", "none",
        "Third estimator of spectral brightness: rho 0.81 with the centroid (0.80 within sessions), "
        "79% of its rank variation explained by the kept features; asymmetric, bounded body (Bowley "
        "-0.34); largest date effect (eps2 0.21). Information lost: upper-band spectral extent beyond "
        "the centroid. Revisit in the multivariate V2.",
    ),
    "spectral_rolloff_std": FeatureDecision(
        "KEEP", "none",
        "Within-event variability of spectral extent: low redundancy (max |rho| 0.37), no extremes, "
        "session-stable (session eps2 0.04), not edge-driven (interior/all ratio 0.99).",
    ),
    "zcr_mean": FeatureDecision(
        "DEFER", "none",
        "Time-domain brightness proxy: rho 0.85 with spectral_centroid_mean (0.84 within sessions), "
        "83% of its rank variation explained by the kept features; computed on 256 ms frames. "
        "Information lost: crossing rate beyond the centroid. Revisit in V2.",
    ),
    "zcr_std": FeatureDecision(
        "EXCLUDE", "none",
        "Measurement artifact as defined: ZCR uses 256 ms frames edge-padded at both event "
        "boundaries, and the std over unpadded frames is a median 0.45x the stored value (rank "
        "agreement 0.76). A ZCR-variability feature would need a new definition.",
    ),
}


def validate_decisions(decisions: Mapping[str, FeatureDecision] = V1_FEATURE_DECISIONS) -> None:
    """Every candidate feature decided exactly once; detection columns never kept."""
    if set(decisions) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(decisions))
        extra = sorted(set(decisions) - set(FEATURE_COLUMNS))
        raise ValueError(f"Decisions must cover the 13 features exactly; missing={missing} extra={extra}")
    for feature, item in decisions.items():
        if item.decision not in DECISIONS:
            raise ValueError(f"{feature}: unknown decision {item.decision!r}")
        if item.transform not in ("none", "log"):
            raise ValueError(f"{feature}: unknown transform {item.transform!r}")
        if not item.reason.strip():
            raise ValueError(f"{feature}: a reason is required")
    if set(decisions) & set(DETECTION_COLUMNS):
        raise ValueError("CNN detection columns must not be candidate features")


def kept_features(decisions: Mapping[str, FeatureDecision] = V1_FEATURE_DECISIONS) -> list[str]:
    """KEEP features in the canonical FEATURE_COLUMNS order."""
    return [feature for feature in FEATURE_COLUMNS if decisions[feature].decision == "KEEP"]


# ── Robust statistics ───────────────────────────────────────────────────────

def mad(values) -> float:
    """Raw median absolute deviation (unscaled)."""
    x = np.asarray(values, dtype=float)
    return float(np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def robust_z(values, center: float, raw_mad: float) -> np.ndarray:
    """(x - center) / (MAD_SCALE * raw_mad); infinite where the MAD is zero."""
    x = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - center) / (MAD_SCALE * raw_mad)


def bowley_skewness(values) -> float:
    """Quartile skewness: (Q3 + Q1 - 2 Q2) / (Q3 - Q1), in [-1, 1]."""
    q1, q2, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else 0.0


def smallest_step(values) -> float:
    """Smallest gap between distinct values (measurement resolution)."""
    unique = np.unique(np.round(np.asarray(values, dtype=float), 9))
    return float(np.diff(unique).min()) if unique.size > 1 else float("nan")


def feature_summary(values) -> dict:
    """Distribution and robust-scale diagnostics for one feature."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    median, raw_mad = float(np.median(x)), mad(x)
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    z = robust_z(x, median, raw_mad)
    squared = np.sort(z ** 2)[::-1] if raw_mad > 0 else np.array([np.nan])
    p01, p05, p25, p75, p95, p99 = np.percentile(x, [1, 5, 25, 75, 95, 99])
    n_low, n_high = int((z < -EXTREME_Z).sum()), int((z > EXTREME_Z).sum())
    return {
        "count": int(x.size),
        "min": float(x.min()), "p01": p01, "p05": p05, "p25": p25, "median": median,
        "p75": p75, "p95": p95, "p99": p99, "max": float(x.max()),
        "mean": float(x.mean()), "sd": sd,
        "mad": raw_mad, "mad_scaled": MAD_SCALE * raw_mad,
        "relative_mad": raw_mad / abs(median) if median else float("inf"),
        "sd_over_scaled_mad": sd / (MAD_SCALE * raw_mad) if raw_mad > 0 else float("inf"),
        "skewness": float(stats.skew(x)) if x.size > 2 and np.ptp(x) > 0 else float("nan"),
        "bowley_skewness": bowley_skewness(x),
        "n_extreme_low": n_low,
        "n_extreme_high": n_high,
        "frac_extreme": (n_low + n_high) / x.size,
        "max_abs_robust_z": float(np.max(np.abs(z))) if raw_mad > 0 else float("inf"),
        "top1_share_sq_z": float(squared[0] / squared.sum()) if raw_mad > 0 else float("nan"),
        "top5_share_sq_z": float(squared[:5].sum() / squared.sum()) if raw_mad > 0 else float("nan"),
        "n_unique": int(np.unique(np.round(x, 9)).size),
        "frac_at_median": float(np.mean(np.isclose(x, median, rtol=0.0, atol=1e-12))),
        "smallest_step": smallest_step(x),
        "step_over_mad": smallest_step(x) / raw_mad if raw_mad > 0 else float("inf"),
    }


def summarize_features(
    table: pd.DataFrame,
    features: Sequence[str],
    populations: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Long table of feature_summary per population, raw and (if positive) log scale."""
    rows = []
    for population, mask in populations.items():
        for feature in features:
            values = table.loc[mask, feature].to_numpy(dtype=float)
            rows.append({"population": population, "feature": feature, "scale": "raw", **feature_summary(values)})
            if values.size and (values > 0).all():
                rows.append({"population": population, "feature": feature, "scale": "log",
                             **feature_summary(np.log(values))})
    return pd.DataFrame(rows)


def subsample_statistics(values, size: int, draws: int = SUBSAMPLE_DRAWS, seed: int = SEED):
    """Medians and raw MADs of ``draws`` random subsets of ``size`` (without replacement)."""
    x = np.asarray(values, dtype=float)
    if size > x.size:
        raise ValueError(f"subset size {size} exceeds population {x.size}")
    rng = np.random.default_rng(seed)
    medians, mads = np.empty(draws), np.empty(draws)
    for i in range(draws):
        sample = rng.choice(x, size=size, replace=False)
        medians[i], mads[i] = np.median(sample), mad(sample)
    return medians, mads


def compare_groups(a, b) -> dict:
    """Location / scale / shape comparison of sample ``a`` against reference ``b``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    median_a, median_b, mad_a, mad_b = np.median(a), np.median(b), mad(a), mad(b)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    ks = stats.ks_2samp(a, b)
    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "median_a": float(median_a), "median_b": float(median_b),
        "mad_scaled_a": MAD_SCALE * mad_a, "mad_scaled_b": MAD_SCALE * mad_b,
        "median_shift_in_b_mad": float((median_a - median_b) / (MAD_SCALE * mad_b)) if mad_b > 0 else float("inf"),
        "mad_ratio_a_over_b": mad_a / mad_b if mad_b > 0 else float("inf"),
        "cliffs_delta": float(2 * u.statistic / (a.size * b.size) - 1),
        "mannwhitney_p": float(u.pvalue),
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "frac_b_outside_a_range": float(np.mean((b < a.min()) | (b > a.max()))),
        # Under exchangeability a new value falls outside the range of n values with probability 2/(n+1).
        "expected_frac_outside_range": 2 / (a.size + 1),
    }


def calibration_representativeness(
    table: pd.DataFrame,
    features: Sequence[str],
    calibration_mask: pd.Series,
    draws: int = SUBSAMPLE_DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Calibration vs rest, and where the calibration median/MAD sit among random subsets.

    ``table`` is the usable population.  Random subsets of the calibration
    size are drawn from it; ``subset_p_median_shift`` is the fraction of
    random subsets whose median is at least as far from the population median,
    and ``subset_frac_mad_as_small`` the fraction whose MAD is at most the
    calibration MAD.
    """
    calibration, rest = table.loc[calibration_mask], table.loc[~calibration_mask]
    rows = []
    for index, feature in enumerate(features):
        population = table[feature].to_numpy(dtype=float)
        population_median, population_mad = np.median(population), mad(population)
        medians, mads = subsample_statistics(population, int(calibration_mask.sum()), draws, seed + index)
        calibration_values = calibration[feature].to_numpy(dtype=float)
        shift = abs(np.median(calibration_values) - population_median)
        comparison = compare_groups(calibration_values, rest[feature].to_numpy(dtype=float))
        rows.append({
            "feature": feature,
            **{_CALIBRATION_NAMES.get(key, key): value for key, value in comparison.items()},
            "subset_p_median_shift": float(np.mean(np.abs(medians - population_median) >= shift - 1e-15)),
            "subset_frac_mad_as_small": float(np.mean(mads <= mad(calibration_values) + 1e-15)),
            "subset_mad_ratio_p05": float(np.percentile(mads, 5) / population_mad),
            "subset_mad_ratio_p95": float(np.percentile(mads, 95) / population_mad),
            "subset_median_shift_p95_in_mad": float(
                np.percentile(np.abs(medians - population_median), 95) / (MAD_SCALE * population_mad)
            ),
            "subset_p_mad_zero": float(np.mean(mads == 0)),
        })
    return pd.DataFrame(rows)


_CALIBRATION_NAMES = {
    "n_a": "n_calibration", "n_b": "n_rest",
    "median_a": "median_calibration", "median_b": "median_rest",
    "mad_scaled_a": "mad_scaled_calibration", "mad_scaled_b": "mad_scaled_rest",
    "median_shift_in_b_mad": "median_shift_in_rest_mad",
    "mad_ratio_a_over_b": "mad_ratio_calibration_over_rest",
    "frac_b_outside_a_range": "frac_rest_outside_calibration_range",
}


# ── Redundancy ──────────────────────────────────────────────────────────────

def normal_scores(values: pd.Series) -> np.ndarray:
    """Rank-based normal scores (average ranks for ties)."""
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return stats.norm.ppf((ranks - 0.5) / len(ranks))


def linear_r2(y, X) -> float:
    """R^2 of an ordinary least-squares fit with intercept."""
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float).reshape(len(y), -1)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum(residual ** 2) / total) if total > 0 else float("nan")


def rank_r2(table: pd.DataFrame, target: str, predictors: Sequence[str]) -> float:
    """Share of a feature's rank variation explained linearly by other features' ranks."""
    predictors = [p for p in predictors if p != target]
    if not predictors:
        return 0.0
    scores = {column: normal_scores(table[column]) for column in (target, *predictors)}
    return linear_r2(scores[target], np.column_stack([scores[p] for p in predictors]))


def spearman_long(table: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rho = table[list(features)].corr(method="spearman")
    pearson = table[list(features)].corr(method="pearson")
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            rows.append({"feature_a": a, "feature_b": b, "spearman_rho": rho.loc[a, b],
                         "abs_spearman_rho": abs(rho.loc[a, b]), "pearson_r": pearson.loc[a, b],
                         "n": int(len(table))})
    return pd.DataFrame(rows)


def within_group_spearman(
    table: pd.DataFrame,
    features: Sequence[str],
    group_column: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Rank correlation with between-group differences removed.

    Features are converted to within-group percentile ranks, (rank - 0.5) / n,
    in groups with at least ``min_group_size`` events, and the pooled Pearson
    correlation of those ranks is returned.  A pair that is correlated overall
    but not within groups is related through group-level differences.
    """
    counts = table[group_column].value_counts()
    subset = table[table[group_column].isin(counts[counts >= min_group_size].index)]
    ranks = subset.groupby(group_column)[list(features)].rank(method="average")
    sizes = subset.groupby(group_column)[group_column].transform("size")
    return ranks.sub(0.5).div(sizes, axis=0).corr(method="pearson")


def redundancy_groups(rho: pd.DataFrame, threshold: float = REDUNDANCY_RHO) -> list[list[str]]:
    """Connected components of the graph |rho| >= threshold (groups of size >= 2)."""
    features, seen, groups = list(rho.columns), set(), []
    for start in features:
        if start in seen:
            continue
        component, stack = [], [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in features:
                if other not in seen and abs(rho.loc[node, other]) >= threshold:
                    seen.add(other)
                    stack.append(other)
        if len(component) > 1:
            groups.append(sorted(component, key=features.index))
    return groups


# ── Dates and sessions ──────────────────────────────────────────────────────

def assign_sessions(timestamps: pd.Series, gap_minutes: float = SESSION_GAP_MIN) -> pd.Series:
    """Label recordings ``YYYY-MM-DD#k``; a gap > ``gap_minutes`` starts session k+1."""
    order = pd.to_datetime(timestamps).sort_values(kind="mergesort")
    gaps = order.diff().dt.total_seconds().div(60)
    session_number = (gaps.isna() | (gaps > gap_minutes)).cumsum()
    first_date = order.groupby(session_number).transform("first").dt.strftime("%Y-%m-%d")
    within_date = session_number.groupby(first_date).rank(method="dense").astype(int)
    labels = first_date + "#" + within_date.astype(str)
    return labels.reindex(timestamps.index)


def session_gap_margin(timestamps: pd.Series, gap_minutes: float = SESSION_GAP_MIN) -> dict:
    """Largest consecutive gap kept inside a session and smallest gap that splits one."""
    gaps = pd.to_datetime(timestamps).sort_values().diff().dt.total_seconds().div(60).dropna()
    inside, between = gaps[gaps <= gap_minutes], gaps[gaps > gap_minutes]
    return {
        "gap_minutes": gap_minutes,
        "largest_gap_within_session_min": float(inside.max()) if len(inside) else None,
        "smallest_gap_between_sessions_min": float(between.min()) if len(between) else None,
    }


def between_group_stability(
    table: pd.DataFrame,
    features: Sequence[str],
    group_column: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis / epsilon-squared per feature, plus per-group medians in pooled robust-z units."""
    counts = table[group_column].value_counts()
    tested = sorted(counts[counts >= min_group_size].index)
    summary, per_group = [], []
    for feature in features:
        pooled = table[feature].to_numpy(dtype=float)
        pooled_median, pooled_mad = np.median(pooled), mad(pooled)
        groups = [table.loc[table[group_column] == g, feature].to_numpy(dtype=float) for g in tested]
        n, k = sum(len(g) for g in groups), len(groups)
        if k >= 2:
            h, p = stats.kruskal(*groups)
            epsilon = (h - k + 1) / (n - k)
        else:
            h = p = epsilon = float("nan")
        shifts = [float(robust_z(np.median(g), pooled_median, pooled_mad)) for g in groups]
        summary.append({
            "grouping": group_column, "feature": feature, "groups_tested": k, "events_tested": n,
            "kruskal_h": float(h), "p_value": float(p), "epsilon_squared": float(epsilon),
            "group_median_z_min": min(shifts) if shifts else float("nan"),
            "group_median_z_max": max(shifts) if shifts else float("nan"),
            "group_median_z_range": (max(shifts) - min(shifts)) if shifts else float("nan"),
        })
        for group in sorted(counts.index):
            values = table.loc[table[group_column] == group, feature].to_numpy(dtype=float)
            per_group.append({
                "grouping": group_column, "group": group, "feature": feature, "n": int(values.size),
                "tested": group in tested, "median": float(np.median(values)),
                "mad_scaled": MAD_SCALE * mad(values),
                "median_z_vs_pooled": float(robust_z(np.median(values), pooled_median, pooled_mad)),
                "mad_ratio_vs_pooled": mad(values) / pooled_mad if pooled_mad > 0 else float("inf"),
            })
    return pd.DataFrame(summary), pd.DataFrame(per_group)


# ── Edge effect (needs audio) ───────────────────────────────────────────────

def exact_event_bounds(start_s: float, end_s: float, recording_duration_s: float) -> tuple[float, float]:
    """Rebuild the float event bounds that post_event produced.

    The upstream CSV stores times with limited precision, so slicing audio from
    those values can move a boundary by one sample (Entry 4).  Bounds are
    ``index * stride`` and ``last_index * stride + window`` (clamped at the
    recording end), so they can be rebuilt exactly.
    """
    start = int(round(start_s / STRIDE_S)) * STRIDE_S
    if end_s >= recording_duration_s - 1e-9:
        return float(start), float(recording_duration_s)
    last = int(round((end_s - WINDOW_S) / STRIDE_S))
    return float(start), float(min(last * STRIDE_S + WINDOW_S, recording_duration_s))


def envelope_peak_profile(segment: np.ndarray, sample_rate: int) -> dict:
    """Where the RMS-envelope peak sits, using post_event's envelope definition."""
    from post_event import _rms_envelope

    times, envelope = _rms_envelope(segment, sample_rate)
    if envelope.size == 0:
        raise ValueError("empty segment")
    frame_length = min(RMS_FRAME, len(segment))
    starts = np.arange(0, len(segment), max(1, min(RMS_HOP, frame_length)))
    full = starts + frame_length <= len(segment)
    peak = int(np.argmax(envelope))
    last = envelope.size - 1
    if peak == 0:
        position = "first"
    elif peak == last:
        position = "last"
    elif not full[peak]:
        position = "partial_tail"
    else:
        position = "interior"
    full_peak = int(np.argmax(np.where(full, envelope, -np.inf))) if full.any() else peak
    near = times[envelope >= NEAR_PEAK_FRACTION * envelope[peak]]
    duration = len(segment) / sample_rate
    return {
        "n_envelope_frames": int(envelope.size),
        "n_partial_frames": int((~full).sum()),
        "peak_index": peak,
        "peak_position": position,
        "relative_peak_position": peak / last if last else 0.0,
        "peak_rms": float(envelope[peak]),
        "time_to_peak_s": float(times[peak]),
        "mean_rms": float(envelope.mean()),
        "peak_rms_full_frames": float(envelope[full_peak]),
        "time_to_peak_full_frames_s": float(times[full_peak]),
        "mean_rms_full_frames": float(envelope[full].mean()) if full.any() else float("nan"),
        "near_peak_span_s": float(near.max() - near.min()),
        "near_peak_span_frac": float((near.max() - near.min()) / duration),
    }


def interior_frame_stds(segment: np.ndarray, sample_rate: int) -> dict:
    """Frame std of each spectral feature over all frames and over unpadded frames only."""
    from librosa_extractor import extract_features_from_audio

    if sample_rate != config.LIBROSA_SR:
        raise ValueError("interior_frame_stds expects audio at the extractor sample rate")
    features = extract_features_from_audio(segment, sample_rate)
    if features is None:
        raise ValueError("feature extraction failed")
    spectral = features[:, 3 * config.LIBROSA_N_MFCC:]
    centres = np.arange(len(spectral)) * config.LIBROSA_HOP_LENGTH
    inner = {
        "stft": (centres - STFT_PAD >= 0) & (centres + STFT_PAD <= len(segment)),
        "zcr": (centres - ZCR_FRAME // 2 >= 0) & (centres + ZCR_FRAME // 2 <= len(segment)),
    }
    result = {}
    for index, name in enumerate(SPECTRAL_NAMES):
        mask = inner["zcr" if name == "zcr" else "stft"]
        result[f"{name}_std_all_frames"] = float(spectral[:, index].std())
        result[f"{name}_std_interior"] = float(spectral[mask, index].std()) if mask.sum() >= 3 else float("nan")
        result[f"{name}_interior_frame_fraction"] = float(mask.mean())
    return result


def edge_effect_per_event(table: pd.DataFrame, data_dir: str | Path = config.DATA_DIR) -> pd.DataFrame:
    """Recompute envelope / frame diagnostics for every event from the original audio."""
    from librosa_extractor import load_audio
    from post_event import InhaleEvent, extract_event_audio

    rows = []
    for recording, group in table.groupby("recording_file", sort=False):
        waveform, sample_rate = load_audio(str(Path(data_dir) / recording))
        if waveform is None:
            raise FileNotFoundError(f"Could not load {recording}")
        for index, row in group.iterrows():
            start, end = exact_event_bounds(row["start_s"], row["end_s"], row["recording_duration_s"])
            event = InhaleEvent("Inhale", start, end, end - start, 0.0, 0.0, 0, ())
            segment, _ = extract_event_audio(waveform, event, sample_rate=sample_rate)
            profile = envelope_peak_profile(segment, sample_rate)
            first_sample, last_sample = int(np.floor(start * sample_rate)), int(np.ceil(end * sample_rate))
            before = waveform[max(0, first_sample - RMS_FRAME):first_sample]
            after = waveform[last_sample:last_sample + RMS_FRAME]
            spread = interior_frame_stds(segment, sample_rate)
            rows.append({
                "row": index, "recording_file": recording, "event_id": row["event_id"],
                **profile,
                "crest_factor": profile["peak_rms"] / profile["mean_rms"],
                "rms_before_event": float(np.sqrt(np.mean(before ** 2))) if before.size else float("nan"),
                "rms_after_event": float(np.sqrt(np.mean(after ** 2))) if after.size else float("nan"),
                **spread,
                "reproduction_error": max(
                    abs(profile["mean_rms"] - row["mean_rms"]),
                    abs(profile["peak_rms"] - row["peak_rms"]),
                    abs(profile["time_to_peak_s"] - row["time_to_peak_s"]),
                    *(abs(spread[f"{name}_std_all_frames"] - row[f"{name}_std"]) for name in SPECTRAL_NAMES),
                ),
            })
    result = pd.DataFrame(rows).set_index("row").loc[table.index]
    edge = result["peak_position"] != "interior"
    boundary_is_louder = np.where(
        result["peak_position"] == "first", result["rms_before_event"] > result["peak_rms"],
        result["rms_after_event"] > result["peak_rms"],
    )
    result["louder_outside_event"] = np.where(edge, boundary_is_louder, False)
    return result.reset_index(drop=True)


def summarize_edge_effect(table: pd.DataFrame, edge: pd.DataFrame) -> pd.DataFrame:
    """Counts and medians by peak position, for all and for usable events."""
    joined = pd.concat([table.reset_index(drop=True)[["usable", "duration_s"]], edge], axis=1)
    rows = []
    for population, mask in (("all", np.ones(len(joined), bool)), ("usable", joined["usable"].to_numpy())):
        subset = joined[mask]
        for position in (*PEAK_POSITIONS, "any_edge"):
            chosen = subset[subset["peak_position"] != "interior"] if position == "any_edge" \
                else subset[subset["peak_position"] == position]
            peak_change = (chosen["peak_rms"] - chosen["peak_rms_full_frames"]).abs()
            rows.append({
                "population": population, "peak_position": position,
                "n": int(len(chosen)), "fraction": len(chosen) / len(subset) if len(subset) else float("nan"),
                "median_duration_s": chosen["duration_s"].median(),
                "median_time_to_peak_s": chosen["time_to_peak_s"].median(),
                "median_relative_peak_position": chosen["relative_peak_position"].median(),
                "median_peak_rms": chosen["peak_rms"].median(),
                "median_mean_rms": chosen["mean_rms"].median(),
                "median_crest_factor": chosen["crest_factor"].median(),
                "median_near_peak_span_frac": chosen["near_peak_span_frac"].median(),
                "n_peak_changes_without_partial_frames": int((peak_change > 1e-9).sum()),
                "max_peak_change_without_partial_frames": float(peak_change.max()) if len(chosen) else float("nan"),
                "n_louder_just_outside_event": int(chosen["louder_outside_event"].sum()),
            })
    return pd.DataFrame(rows)


def edge_vs_interior_tests(table: pd.DataFrame, edge: pd.DataFrame, population_mask) -> dict:
    """Mann-Whitney / Cliff's delta of edge-peak vs interior-peak events."""
    joined = pd.concat([table.reset_index(drop=True), edge.drop(columns=["recording_file", "event_id",
                        "peak_rms", "mean_rms", "time_to_peak_s"])], axis=1)[np.asarray(population_mask)]
    is_edge = joined["peak_position"] != "interior"
    result = {"n_edge": int(is_edge.sum()), "n_interior": int((~is_edge).sum())}
    if is_edge.sum() < 3:
        result["note"] = "fewer than 3 edge-peak events; no test"
        return result
    for column in ("duration_s", "mean_rms", "peak_rms", "time_to_peak_s", "crest_factor", "confidence"):
        comparison = compare_groups(joined.loc[is_edge, column], joined.loc[~is_edge, column])
        result[column] = {key: comparison[key] for key in ("median_a", "median_b", "cliffs_delta", "mannwhitney_p")}
    return result


def _extreme_peak_profile(usable: pd.DataFrame, edge_usable: pd.DataFrame) -> dict:
    """Where the high-side peak_rms extremes sit and how peaky they are."""
    values = usable["peak_rms"].to_numpy(dtype=float)
    extreme = robust_z(values, np.median(values), mad(values)) > EXTREME_Z
    chosen = edge_usable[extreme]
    return {
        "n": int(extreme.sum()),
        "peak_positions": chosen["peak_position"].value_counts().to_dict(),
        "crest_factor_median": float(chosen["crest_factor"].median()),
        "crest_factor_min": float(chosen["crest_factor"].min()),
        "crest_factor_max": float(chosen["crest_factor"].max()),
        "crest_factor_median_all_usable": float(edge_usable["crest_factor"].median()),
    }


def drug_overlap_context(table: pd.DataFrame, audit: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Interpretation only: features of usable events overlapping a Drug annotation vs not."""
    joined = table.merge(audit, on=["recording_file", "event_id"], validate="one_to_one")
    joined = joined[joined["usable"] & joined["recording_annotated"]]
    overlaps = joined["overlap_drug"] > 0
    rows = []
    for feature in features:
        if overlaps.sum() < 3:
            break
        comparison = compare_groups(joined.loc[overlaps, feature], joined.loc[~overlaps, feature])
        rows.append({"feature": feature, "n_drug_overlap": int(overlaps.sum()), "n_no_drug_overlap": int((~overlaps).sum()),
                     "median_drug_overlap": comparison["median_a"], "median_no_drug_overlap": comparison["median_b"],
                     "median_overlap_fraction": float(joined.loc[overlaps, "overlap_drug"].median()),
                     "cliffs_delta": comparison["cliffs_delta"], "mannwhitney_p": comparison["mannwhitney_p"]})
    return pd.DataFrame(rows)


# ── Figures (static PNG; reference palette of the dataviz guidance) ─────────

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
DIVERGING = ("#1c5cab", "#f0efec", "#e34948")
FEATURE_ORDER = (
    "duration_s", "time_to_peak_s", "mean_rms", "peak_rms", "total_energy",
    "spectral_centroid_mean", "spectral_rolloff_mean", "zcr_mean", "spectral_flatness_mean",
    "spectral_centroid_std", "spectral_rolloff_std", "spectral_flatness_std", "zcr_std",
)


def _style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE)
    axis.tick_params(colors=INK_MUTED, labelsize=8)


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("prism_diverging", DIVERGING)


def _annotate_cells(axis, values: np.ndarray, limit: float, fmt: str) -> None:
    for (i, j), value in np.ndenumerate(values):
        if np.isnan(value):
            continue
        colour = "#ffffff" if abs(value) > 0.6 * limit else INK
        axis.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color=colour)


def plot_feature_distributions(usable: pd.DataFrame, calibration_mask: pd.Series, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 4, figsize=(14, 11), facecolor=SURFACE, layout="constrained")
    for axis, feature in zip(axes.flat, FEATURE_ORDER):
        _style_axis(axis)
        values = usable[feature].to_numpy(dtype=float)
        axis.hist(values, bins=30, color=SERIES_1, edgecolor=SURFACE, linewidth=0.8,
                  label=f"Usable events (n={len(usable)})")
        low = axis.get_ylim()[1] * -0.06
        axis.plot(usable.loc[calibration_mask, feature], np.full(int(calibration_mask.sum()), low), "|",
                  color=SERIES_2, markersize=9, markeredgewidth=1.6, label="First 20 usable (V1 calibration)")
        axis.axvline(np.median(values), color=INK_SECONDARY, linewidth=1, linestyle="--", label="Median, all usable")
        axis.set_title(feature, fontsize=9, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    for axis in list(axes.flat)[len(FEATURE_ORDER):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Candidate feature distributions, usable inhale events", color=INK, fontsize=12)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_spearman(rho: pd.DataFrame, n_events: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = list(FEATURE_ORDER)
    values = rho.loc[order, order].to_numpy()
    figure, axis = plt.subplots(figsize=(9.5, 8), facecolor=SURFACE, layout="constrained")
    image = axis.imshow(values, cmap=_diverging_cmap(), vmin=-1, vmax=1)
    axis.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8, color=INK_SECONDARY)
    axis.set_yticks(range(len(order)), order, fontsize=8, color=INK_SECONDARY)
    axis.set_xticks(np.arange(len(order) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    _annotate_cells(axis, values, 1.0, ".2f")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Spearman rho", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    axis.set_title(f"Rank correlation between candidate features (usable events, n={n_events})", color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_group_shifts(
    per_group: pd.DataFrame,
    grouping: str,
    calibration_counts: Mapping[str, int],
    path: Path,
) -> None:
    """Heatmap of each tested group's median in pooled robust-z units."""
    import matplotlib.pyplot as plt

    groups = per_group[(per_group["grouping"] == grouping) & per_group["tested"]]
    columns = sorted(groups["group"].unique())
    matrix = groups.pivot(index="feature", columns="group", values="median_z_vs_pooled").loc[list(FEATURE_ORDER), columns]
    counts = groups.drop_duplicates("group").set_index("group")["n"]
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    figure, axis = plt.subplots(figsize=(max(10, 1.05 * len(columns) + 3), 7.5), facecolor=SURFACE, layout="constrained")
    image = axis.imshow(matrix.to_numpy(), cmap=_diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    labels = [
        f"{g}\nn={counts[g]}" + (f"\n{calibration_counts[g]} calib." if calibration_counts.get(g) else "")
        for g in columns
    ]
    axis.set_xticks(range(len(columns)), labels, fontsize=8, color=INK_SECONDARY)
    axis.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8, color=INK_SECONDARY)
    axis.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(matrix.index) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    _annotate_cells(axis, matrix.to_numpy(), limit, "+.2f")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label(f"{grouping.capitalize()} median, robust z vs all usable events", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    axis.set_title(f"Between-{grouping} shift of each feature ({grouping}s with >= {MIN_GROUP_SIZE} usable events)",
                   color=INK, fontsize=11)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ── Orchestration ───────────────────────────────────────────────────────────

def load_stage1_table(path: str | Path = DEFAULT_DATASET_CSV) -> pd.DataFrame:
    table = pd.read_csv(path, parse_dates=["recording_timestamp"])
    missing = [c for c in (*FEATURE_COLUMNS, "usable", "usable_order", "recording_file") if c not in table.columns]
    if missing:
        raise ValueError(f"Stage 1 table is missing columns: {missing}")
    return table


def feature_selection_table(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    usable: pd.DataFrame,
    date_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    decisions: Mapping[str, FeatureDecision] = V1_FEATURE_DECISIONS,
) -> pd.DataFrame:
    """One row per candidate feature: decision, reason, and the evidence behind it."""
    validate_decisions(decisions)
    keep = kept_features(decisions)
    raw = summary[(summary["population"] == "usable") & (summary["scale"] == "raw")].set_index("feature")
    dates = date_summary.set_index("feature")
    calibration = calibration.set_index("feature")
    rows = []
    for feature in FEATURE_COLUMNS:
        pairs = correlations[(correlations["feature_a"] == feature) | (correlations["feature_b"] == feature)]
        strongest = pairs.loc[pairs["abs_spearman_rho"].idxmax()]
        partner = strongest["feature_b"] if strongest["feature_a"] == feature else strongest["feature_a"]
        item = decisions[feature]
        rows.append({
            "feature": feature, "decision": item.decision, "v1_transform": item.transform, "reason": item.reason,
            "relative_mad": raw.loc[feature, "relative_mad"],
            "skewness": raw.loc[feature, "skewness"],
            "bowley_skewness": raw.loc[feature, "bowley_skewness"],
            "n_extreme": int(raw.loc[feature, "n_extreme_low"] + raw.loc[feature, "n_extreme_high"]),
            "max_abs_robust_z": raw.loc[feature, "max_abs_robust_z"],
            "strongest_partner": partner,
            "strongest_spearman_rho": strongest["spearman_rho"],
            "strongest_partner_within_session_rho": strongest.get("within_session_rho", float("nan")),
            "rank_r2_from_other_12": rank_r2(usable, feature, FEATURE_COLUMNS),
            "rank_r2_from_other_kept": rank_r2(usable, feature, keep),
            "date_epsilon_squared": dates.loc[feature, "epsilon_squared"],
            "calibration_median_shift_in_rest_mad": calibration.loc[feature, "median_shift_in_rest_mad"],
            "calibration_mad_ratio": calibration.loc[feature, "mad_ratio_calibration_over_rest"],
        })
    return pd.DataFrame(rows)


def run_feature_analysis(
    dataset_csv: str | Path = DEFAULT_DATASET_CSV,
    audit_csv: str | Path = DEFAULT_AUDIT_CSV,
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    make_plots: bool = True,
) -> dict:
    validate_decisions()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = load_stage1_table(dataset_csv)
    features = list(FEATURE_COLUMNS)
    usable_mask = table["usable"].astype(bool)
    usable = table[usable_mask].reset_index(drop=True)
    calibration_mask = usable["usable_order"].between(1, CALIBRATION_SIZE)
    if int(calibration_mask.sum()) != CALIBRATION_SIZE:
        raise ValueError("Stage 1 table does not contain the first 20 usable events")

    # 1-2. Distributions and robust-scale suitability.
    summary = summarize_features(table, features, {"usable": usable_mask, "all": pd.Series(True, index=table.index)})
    stability = []
    for index, feature in enumerate(features):
        values = usable[feature].to_numpy(dtype=float)
        medians, mads = subsample_statistics(values, CALIBRATION_SIZE, seed=SEED + index)
        stability.append({
            "feature": feature,
            "n20_mad_ratio_p05": float(np.percentile(mads, 5) / mad(values)),
            "n20_mad_ratio_p95": float(np.percentile(mads, 95) / mad(values)),
            "n20_median_shift_p95_in_mad": float(np.percentile(np.abs(medians - np.median(values)), 95)
                                                 / (MAD_SCALE * mad(values))),
            "n20_p_mad_zero": float(np.mean(mads == 0)),
        })
    summary = summary.merge(pd.DataFrame(stability).assign(population="usable", scale="raw"),
                            on=["population", "feature", "scale"], how="left")
    summary.to_csv(output / "feature_summary.csv", index=False)

    # Dates and sessions (needed by the redundancy and stability sections).
    wav_names = [path.name for path in sorted(Path(data_dir).glob("*.wav"))]
    wav_timestamps = pd.Series([parse_recording_timestamp(name) for name in wav_names])
    wav_sessions = assign_sessions(wav_timestamps)
    usable["date"] = usable["recording_timestamp"].dt.strftime("%Y-%m-%d")
    usable["session"] = usable["recording_file"].map(dict(zip(wav_names, wav_sessions)))

    # 3. Redundancy, overall and within sessions.
    correlations = spearman_long(usable, features)
    within_session = within_group_spearman(usable, features, "session")
    correlations["within_session_rho"] = [
        within_session.loc[a, b] for a, b in zip(correlations["feature_a"], correlations["feature_b"])
    ]
    correlations.to_csv(output / "feature_correlations.csv", index=False)
    rho = usable[features].corr(method="spearman")
    log_energy = np.log(usable[["total_energy", "duration_s", "mean_rms"]])
    energy_identity_r2 = linear_r2(log_energy["total_energy"], log_energy[["duration_s", "mean_rms"]])

    # 4. Stability between dates and sessions.
    date_summary, date_groups = between_group_stability(usable, features, "date")
    session_summary, session_groups = between_group_stability(usable, features, "session")
    pd.concat([date_summary, session_summary]).to_csv(output / "session_stability.csv", index=False)
    pd.concat([date_groups, session_groups]).to_csv(output / "session_group_medians.csv", index=False)

    # 5. Calibration set.
    calibration = calibration_representativeness(usable, features, calibration_mask)
    calibration.to_csv(output / "calibration_vs_rest.csv", index=False)

    # 6. Edge effect and frame-edge contribution (from the original audio).
    edge = edge_effect_per_event(table, data_dir)
    edge.to_csv(output / "edge_effect_events.csv", index=False)
    edge_summary = summarize_edge_effect(table, edge)
    edge_summary.to_csv(output / "edge_effect_analysis.csv", index=False)
    edge_usable = edge[usable_mask.to_numpy()]
    frame_edge = {}
    for name in SPECTRAL_NAMES:
        full, interior = edge_usable[f"{name}_std_all_frames"], edge_usable[f"{name}_std_interior"]
        valid = interior.notna()
        frame_edge[f"{name}_std"] = {
            "events_with_interior_frames": int(valid.sum()),
            "spearman_all_vs_interior": float(stats.spearmanr(full[valid], interior[valid]).statistic),
            "median_ratio_interior_over_all": float((interior[valid] / full[valid]).median()),
            "median_interior_frame_fraction": float(edge_usable[f"{name}_interior_frame_fraction"].median()),
        }

    # Interpretation only (annotations never select features).
    drug_context = drug_overlap_context(table, pd.read_csv(audit_csv), features)
    drug_context.to_csv(output / "annotation_context.csv", index=False)

    # 7. Decisions with their evidence.
    selection = feature_selection_table(summary, correlations, usable, date_summary, calibration)
    selection.to_csv(output / "feature_selection_v1.csv", index=False)
    keep = kept_features()
    dataset_sha = _sha256(Path(dataset_csv))
    selection_json = {
        "version": "v1",
        "dataset": {"path": _relative(Path(dataset_csv)), "sha256": dataset_sha, "population": "usable == True"},
        "keep": keep,
        "transforms": {feature: V1_FEATURE_DECISIONS[feature].transform for feature in keep},
        "exclude": [f for f in FEATURE_COLUMNS if V1_FEATURE_DECISIONS[f].decision == "EXCLUDE"],
        "defer": [f for f in FEATURE_COLUMNS if V1_FEATURE_DECISIONS[f].decision == "DEFER"],
        "never_features": list(DETECTION_COLUMNS),
        "decisions": {f: asdict(d) for f, d in V1_FEATURE_DECISIONS.items()},
        "rationale": "PRISM_RESEARCH_LOG.md Entry 4",
    }
    with (output / "feature_selection_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_json, handle, indent=2)

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        plot_feature_distributions(usable, calibration_mask, output / "feature_distributions.png")
        plot_spearman(rho, len(usable), output / "feature_spearman.png")
        for grouping, groups in (("date", date_groups), ("session", session_groups)):
            calibration_counts = usable.loc[calibration_mask, grouping].value_counts().to_dict()
            plot_group_shifts(groups, grouping, calibration_counts, output / f"{grouping}_shift_heatmap.png")

    session_counts = usable["session"].value_counts()
    result = {
        "stage": "Stage 2 - feature analysis and V1 feature selection",
        "dataset": {"path": _relative(Path(dataset_csv)), "sha256": dataset_sha,
                    "events": int(len(table)), "usable": int(usable_mask.sum()),
                    "calibration_events": CALIBRATION_SIZE},
        "provenance": {"git": _git_state(), "seed": SEED, "subsample_draws": SUBSAMPLE_DRAWS},
        "parameters": {"mad_scale": MAD_SCALE, "extreme_robust_z": EXTREME_Z, "min_group_size": MIN_GROUP_SIZE,
                       "near_peak_fraction": NEAR_PEAK_FRACTION, "redundancy_rho_descriptive": REDUNDANCY_RHO},
        "redundancy": {
            "groups_abs_rho_ge_0_8": redundancy_groups(rho, REDUNDANCY_RHO),
            "log_total_energy_from_log_duration_and_log_mean_rms_r2": energy_identity_r2,
        },
        "sessions": {
            **session_gap_margin(wav_timestamps),
            "sessions_total": int(wav_sessions.nunique()),
            "sessions_with_usable_events": int(session_counts.size),
            "sessions_tested": int((session_counts >= MIN_GROUP_SIZE).sum()),
            "dates_with_usable_events": int(usable["date"].nunique()),
            "dates_tested": int((usable["date"].value_counts() >= MIN_GROUP_SIZE).sum()),
            "calibration_sessions": sorted(usable.loc[calibration_mask, "session"].unique()),
        },
        "edge_effect": {
            "reproduction_max_abs_error": float(edge["reproduction_error"].max()),
            "tests_all_events": edge_vs_interior_tests(table, edge, np.ones(len(table), bool)),
            "tests_usable_events": edge_vs_interior_tests(table, edge, usable_mask.to_numpy()),
            "near_peak_span_frac_usable": edge_usable["near_peak_span_frac"].describe().to_dict(),
            "near_peak_span_s_usable_median": float(edge_usable["near_peak_span_s"].median()),
            "time_to_peak_scaled_mad_usable": MAD_SCALE * mad(usable["time_to_peak_s"]),
            "peak_rms_extreme_high_usable": _extreme_peak_profile(usable, edge_usable),
            "frame_edge_contribution_to_std_features": frame_edge,
        },
        "decisions": {"keep": keep, "exclude": selection_json["exclude"], "defer": selection_json["defer"]},
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    with (output / "feature_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 feature analysis for the V1 baseline")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outcome = run_feature_analysis(dataset_csv=args.dataset, output_dir=args.output_dir, make_plots=not args.no_plots)
    print(f"KEEP {outcome['decisions']['keep']}")
    print(f"EXCLUDE {outcome['decisions']['exclude']}  DEFER {outcome['decisions']['defer']}")
