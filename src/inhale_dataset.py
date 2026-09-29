"""Finalize the CNN-detected inhale-event table for baseline modeling.

Stage 1 of the anomaly-detection work (PRISM_RESEARCH_LOG.md, Entry 3).

The upstream table written by ``explore_inhalations.py`` is left untouched.
This module reads it and adds, without changing any upstream row or value:

* recording context (timestamp parsed from the filename, recording duration,
  gaps to neighbouring candidates in the same recording);
* an explicit, annotation-independent usability rule with per-event reasons;
* a separate audit against ``data/annotation.csv`` that is used only to check
  the rule, never to decide it per event.

"Usable" means eligible for baseline calibration and baseline-consistency
evaluation.  It is not a technique-quality label, and excluded events are kept
in the output with their exclusion reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import config


DEFAULT_EVENTS_CSV = Path(config.RESULTS_DIR) / "post_event" / "inhalation_events.csv"
DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "inhale_dataset"
DATASET_FILENAME = "inhale_events_v1.csv"

# The 13 per-event acoustic/timing measurements written by explore_inhalations.py.
FEATURE_COLUMNS = (
    "duration_s", "mean_rms", "peak_rms", "total_energy", "time_to_peak_s",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "zcr_mean", "zcr_std",
)
# Event-detector outputs: they describe the CNN's certainty, not the inhalation.
DETECTION_COLUMNS = ("confidence", "max_confidence", "window_count")
REQUIRED_COLUMNS = ("recording_file", "event_id", "start_s", "end_s", *FEATURE_COLUMNS, *DETECTION_COLUMNS)

# CNN analysis-window timing used by post_event.generate_window_predictions.
WINDOW_S = config.WINDOW_SIZE * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR    # 0.200 s
STRIDE_S = config.WINDOW_STRIDE * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR  # 0.016 s

MATCH_IOU = 0.5  # conventional temporal-IoU criterion for "annotation matched"
ANNOTATION_LABELS = tuple(config.LABEL_NAMES)
EXCLUSION_REASONS = ("nonfinite_feature", "short_duration", "close_neighbor", "recording_boundary")

# Event times are multiples of 8 ms; rounding removes float noise before
# threshold comparisons so that e.g. a 0.200 s gap is not read as 0.19999.
_TIME_DECIMALS = 6
_RECORDING_NAME = re.compile(r"^rec(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})m(\d{2}(?:\.\d+)?)s\.wav$")


@dataclass(frozen=True)
class UsabilityRule:
    """Annotation-independent eligibility rule for baseline modeling.

    ``min_duration_s``: shorter candidates are excluded as non-inhalation
    detections (see Entry 3 for the annotation evidence).
    ``min_neighbor_gap_s``: candidates separated from another candidate in the
    same recording by less than one CNN analysis window are ambiguous
    segmentation (possibly fragments of one inhalation); they are excluded,
    not merged, so no upstream measurement is altered.
    ``boundary_tolerance_s``: candidates reaching the first or last analysis
    window of the recording are censored by the recording boundary.
    """

    version: str = "v1"
    min_duration_s: float = 0.5
    min_neighbor_gap_s: float = WINDOW_S
    boundary_tolerance_s: float = STRIDE_S / 2

    def __post_init__(self) -> None:
        for name in ("min_duration_s", "min_neighbor_gap_s", "boundary_tolerance_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_recording_timestamp(filename: str) -> pd.Timestamp:
    """Parse ``recYYYY-MM-DD_HHhMMmSS.sss.wav`` into a timestamp."""
    match = _RECORDING_NAME.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognised recording filename: {filename}")
    date, hours, minutes, seconds = match.groups()
    return pd.Timestamp(f"{date} {hours}:{minutes}:{seconds}")


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Temporal intersection-over-union of two intervals."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def covered_duration(start: float, end: float, intervals: Sequence[tuple[float, float]]) -> float:
    """Length of ``[start, end]`` covered by the union of ``intervals``."""
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end)
    covered, run_start, run_end = 0.0, None, None
    for a, b in clipped:
        if run_end is None or a > run_end:
            if run_end is not None:
                covered += run_end - run_start
            run_start, run_end = a, b
        else:
            run_end = max(run_end, b)
    if run_end is not None:
        covered += run_end - run_start
    return covered


def _chronological_index(table: pd.DataFrame) -> pd.Index:
    return table.sort_values(
        ["recording_timestamp", "recording_file", "start_s", "event_id"], kind="mergesort"
    ).index


def add_recording_context(
    events: pd.DataFrame,
    recording_durations: Mapping[str, float],
) -> pd.DataFrame:
    """Add timestamp, recording duration, neighbour gaps, and chronology.

    Upstream columns and row order are preserved.
    """
    _require_columns(events, REQUIRED_COLUMNS)
    if not events.index.is_unique:
        raise ValueError("events index must be unique")
    missing = sorted(set(events["recording_file"]) - set(recording_durations))
    if missing:
        raise ValueError(f"Missing recording durations for {len(missing)} recordings, e.g. {missing[:3]}")

    table = events.copy()
    table["recording_timestamp"] = pd.to_datetime(
        table["recording_file"].map(parse_recording_timestamp)
    )
    table["recording_duration_s"] = table["recording_file"].map(recording_durations).astype(float)
    table["n_events_in_recording"] = table.groupby("recording_file")["event_id"].transform("size")

    ordered = table.loc[_chronological_index(table)]
    by_recording = ordered.groupby("recording_file", sort=False)
    table["gap_prev_s"] = (ordered["start_s"] - by_recording["end_s"].shift()).round(_TIME_DECIMALS)
    table["gap_next_s"] = (by_recording["start_s"].shift(-1) - ordered["end_s"]).round(_TIME_DECIMALS)

    # Fraction of the spanned analysis windows that the CNN labelled Inhale.
    # Values < 1 mean overlapping windows bridged non-Inhale windows inside the
    # event.  Undefined when the event end was clamped to the recording end.
    clamped = table["end_s"] >= table["recording_duration_s"] - 1e-9
    spanned = np.round((table["end_s"] - table["start_s"] - WINDOW_S) / STRIDE_S) + 1
    table["inhale_window_fraction"] = (table["window_count"] / spanned).where(~clamped)

    table["chronological_order"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table.loc[ordered.index, "chronological_order"] = np.arange(1, len(ordered) + 1)
    return table


def apply_usability_rule(table: pd.DataFrame, rule: UsabilityRule = UsabilityRule()) -> pd.DataFrame:
    """Flag each event and mark it usable when no exclusion reason applies."""
    _require_columns(
        table,
        (*REQUIRED_COLUMNS, "recording_timestamp", "recording_duration_s", "gap_prev_s", "gap_next_s"),
    )
    result = table.copy()
    features = result[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    nearest_gap = result[["gap_prev_s", "gap_next_s"]].min(axis=1).round(_TIME_DECIMALS)
    tolerance = rule.boundary_tolerance_s
    flags = {
        "nonfinite_feature": ~np.isfinite(features).all(axis=1),
        "short_duration": result["duration_s"].round(_TIME_DECIMALS) < rule.min_duration_s,
        "close_neighbor": nearest_gap < rule.min_neighbor_gap_s,  # NaN (no neighbour) -> False
        "recording_boundary": (result["start_s"] <= tolerance)
        | (result["end_s"] >= result["recording_duration_s"] - tolerance),
    }
    for reason in EXCLUSION_REASONS:
        result[f"flag_{reason}"] = np.asarray(flags[reason], dtype=bool)

    matrix = result[[f"flag_{reason}" for reason in EXCLUSION_REASONS]].to_numpy()
    result["usable"] = ~matrix.any(axis=1)
    result["exclusion_reasons"] = [
        ";".join(reason for reason, flagged in zip(EXCLUSION_REASONS, row) if flagged) for row in matrix
    ]
    usable_index = _chronological_index(result.loc[result["usable"]])
    result["usable_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[usable_index, "usable_order"] = np.arange(1, len(usable_index) + 1)
    result["usability_rule"] = rule.version
    return result


# ── Annotation audit (checks the rule; never used to decide it) ─────────────

def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read annotation.csv with the same convention as ``loader.load_annotation``."""
    annotations = pd.read_csv(
        path, header=None, names=["filename", "label", "start_sample", "end_sample"]
    )
    annotations["label"] = annotations["label"].str.strip().str.capitalize()
    return annotations


def _label_counts(annotations: pd.DataFrame) -> dict:
    return {
        label: {"rows": int(len(group)), "recordings": int(group["filename"].nunique())}
        for label, group in annotations.groupby("label")
    }


def clean_annotations(annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows and non-positive-length intervals; report both."""
    _require_columns(annotations, ("filename", "label", "start_sample", "end_sample"))
    duplicated = annotations.duplicated(keep="first")
    invalid = annotations["end_sample"] <= annotations["start_sample"]
    clean = annotations.loc[~duplicated & ~invalid].reset_index(drop=True)

    overlapping_inhale = []
    for filename, group in clean[clean["label"] == "Inhale"].groupby("filename"):
        spans = sorted(zip(group["start_sample"], group["end_sample"]))
        if any(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:])):
            overlapping_inhale.append(filename)

    summary = {
        "raw_rows": int(len(annotations)),
        "exact_duplicate_rows": int(duplicated.sum()),
        "non_positive_length_rows": int(invalid.sum()),
        "clean_rows": int(len(clean)),
        "annotated_recordings": int(annotations["filename"].nunique()),
        "labels_raw": _label_counts(annotations),
        "labels_clean": _label_counts(clean),
        "duplicate_rows": annotations.loc[duplicated].to_dict("records"),
        "non_positive_length_rows_detail": annotations.loc[invalid].to_dict("records"),
        "recordings_with_overlapping_inhale_annotations": overlapping_inhale,
    }
    return clean, summary


def _annotation_seconds(annotations: pd.DataFrame, sample_rate: int) -> dict[str, pd.DataFrame]:
    seconds = annotations.assign(
        start_s=annotations["start_sample"] / sample_rate,
        end_s=annotations["end_sample"] / sample_rate,
    )
    return {filename: group for filename, group in seconds.groupby("filename")}


def audit_events_against_annotations(
    events: pd.DataFrame,
    annotations: pd.DataFrame,
    sample_rate: int = config.LIBROSA_SR,
) -> pd.DataFrame:
    """Per-event overlap with cleaned annotations.

    ``annotation_status`` separates recordings without annotations from
    recordings that are annotated but have no Inhale label, because the latter
    are not necessarily free of inhalations (Entry 3).
    """
    _require_columns(events, ("recording_file", "event_id", "start_s", "end_s"))
    by_file = _annotation_seconds(annotations, sample_rate)
    rows = []
    for event in events.itertuples(index=False):
        group = by_file.get(event.recording_file)
        row = {"recording_file": event.recording_file, "event_id": event.event_id}
        if group is None:
            row.update(
                recording_annotated=False,
                recording_has_inhale_annotation=False,
                best_inhale_iou=np.nan,
                **{f"overlap_{label.lower()}": np.nan for label in ANNOTATION_LABELS},
                overlap_any_annotation=np.nan,
                annotation_status="unannotated_recording",
            )
            rows.append(row)
            continue

        duration = event.end_s - event.start_s
        inhale = group[group["label"] == "Inhale"]
        best_iou = max(
            (interval_iou(event.start_s, event.end_s, a, b) for a, b in zip(inhale["start_s"], inhale["end_s"])),
            default=0.0,
        )
        overlaps = {
            f"overlap_{label.lower()}": covered_duration(
                event.start_s, event.end_s,
                list(zip(group.loc[group["label"] == label, "start_s"], group.loc[group["label"] == label, "end_s"])),
            ) / duration
            for label in ANNOTATION_LABELS
        }
        if best_iou >= MATCH_IOU:
            status = "matched_inhale"
        elif best_iou > 0:
            status = "partial_inhale_overlap"
        elif len(inhale):
            status = "outside_inhale_annotations"
        else:
            status = "no_inhale_annotation_in_recording"
        row.update(
            recording_annotated=True,
            recording_has_inhale_annotation=bool(len(inhale)),
            best_inhale_iou=best_iou,
            **overlaps,
            overlap_any_annotation=covered_duration(
                event.start_s, event.end_s, list(zip(group["start_s"], group["end_s"]))
            ) / duration,
            annotation_status=status,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def match_inhale_annotations(
    events: pd.DataFrame,
    annotations: pd.DataFrame,
    sample_rate: int = config.LIBROSA_SR,
) -> pd.DataFrame:
    """For every Inhale annotation, the best-overlapping detected event."""
    inhale = annotations[annotations["label"] == "Inhale"]
    by_recording = {name: group for name, group in events.groupby("recording_file")}
    rows = []
    for annotation in inhale.itertuples(index=False):
        start, end = annotation.start_sample / sample_rate, annotation.end_sample / sample_rate
        candidates = by_recording.get(annotation.filename, events.iloc[0:0])
        ious = [interval_iou(start, end, s, e) for s, e in zip(candidates["start_s"], candidates["end_s"])]
        best = int(np.argmax(ious)) if ious else None
        rows.append({
            "recording_file": annotation.filename,
            "annotation_start_s": start,
            "annotation_end_s": end,
            "annotation_duration_s": end - start,
            "best_event_id": None if best is None else int(candidates["event_id"].iloc[best]),
            "best_iou": ious[best] if ious else 0.0,
            "matched": bool(ious) and ious[best] >= MATCH_IOU,
        })
    return pd.DataFrame(rows)


def usability_sensitivity(
    context: pd.DataFrame,
    audit: pd.DataFrame,
    min_durations: Sequence[float] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    min_gaps: Sequence[float] = (0.0, 0.1, 0.2, 0.4),
) -> pd.DataFrame:
    """How the usable set and its annotation status change with the rule."""
    status = context[["recording_file", "event_id"]].merge(
        audit[["recording_file", "event_id", "annotation_status"]],
        on=["recording_file", "event_id"], how="left", validate="one_to_one",
    )["annotation_status"].to_numpy()
    matched = status == "matched_inhale"
    unannotated = status == "unannotated_recording"
    rows = []
    for min_duration in min_durations:
        for min_gap in min_gaps:
            rule = UsabilityRule(min_duration_s=min_duration, min_neighbor_gap_s=min_gap)
            usable = apply_usability_rule(context, rule)["usable"].to_numpy()
            rows.append({
                "min_duration_s": min_duration,
                "min_neighbor_gap_s": min_gap,
                "usable_events": int(usable.sum()),
                "excluded_events": int((~usable).sum()),
                "usable_matched_inhale": int((usable & matched).sum()),
                "excluded_matched_inhale": int((~usable & matched).sum()),
                "usable_annotated_not_matched": int((usable & ~matched & ~unannotated).sum()),
                "usable_unannotated_recording": int((usable & unannotated).sum()),
            })
    return pd.DataFrame(rows)


def compare_event_tables(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Column-wise comparison used to check that the upstream CSV reproduces."""
    report = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "same_columns": list(reference.columns) == list(candidate.columns),
        "columns": {},
    }
    comparable = report["same_columns"] and reference.shape == candidate.shape
    if comparable:
        for column in reference.columns:
            a, b = reference[column], candidate[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
                both_nan = np.isnan(x) & np.isnan(y)
                diff = np.where(both_nan, 0.0, np.abs(x - y))
                diff = np.where(np.isnan(diff), np.inf, diff)
                report["columns"][column] = {"max_abs_diff": float(diff.max()) if diff.size else 0.0}
            else:
                report["columns"][column] = {"all_equal": bool((a.astype(str) == b.astype(str)).all())}
    report["identical"] = bool(comparable) and all(
        stats.get("max_abs_diff", 0.0) == 0.0 and stats.get("all_equal", True)
        for stats in report["columns"].values()
    )
    return report


# ── Orchestration ───────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=config.ROOT_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    try:
        return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(config.ROOT_DIR).resolve()).as_posix()
    except ValueError:
        return str(path)


def recording_durations(data_dir: str | Path) -> tuple[dict[str, float], dict]:
    """Duration of every WAV in ``data_dir`` from its header, plus a format summary."""
    import soundfile as sf

    infos = {path.name: sf.info(str(path)) for path in sorted(Path(data_dir).glob("*.wav"))}
    if not infos:
        raise FileNotFoundError(f"No WAV recordings found in {data_dir}")
    durations = {name: info.frames / info.samplerate for name, info in infos.items()}
    summary = {
        "wav_recordings": len(infos),
        "sample_rates": sorted({info.samplerate for info in infos.values()}),
        "channels": sorted({info.channels for info in infos.values()}),
        "duration_s": {
            "min": min(durations.values()), "median": float(np.median(list(durations.values()))),
            "max": max(durations.values()),
        },
    }
    return durations, summary


def build_inhale_dataset(
    events_csv: str | Path = DEFAULT_EVENTS_CSV,
    data_dir: str | Path = config.DATA_DIR,
    annotation_csv: str | Path = config.ANNOTATION_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rule: UsabilityRule = UsabilityRule(),
    verify_against: str | Path | None = None,
) -> dict:
    """Write the finalized event table, annotation audit, and summary JSON."""
    events_path, output_path = Path(events_csv), Path(output_dir)
    events = pd.read_csv(events_path)
    durations, wav_summary = recording_durations(data_dir)
    if wav_summary["sample_rates"] != [config.LIBROSA_SR]:
        raise ValueError(f"Expected {config.LIBROSA_SR} Hz recordings, found {wav_summary['sample_rates']}")

    table = apply_usability_rule(add_recording_context(events, durations), rule)
    raw_annotations = read_annotations(annotation_csv)
    annotations, annotation_summary = clean_annotations(raw_annotations)
    audit = audit_events_against_annotations(events, annotations)
    matches = match_inhale_annotations(events, annotations)
    sensitivity = usability_sensitivity(table, audit)

    output_path.mkdir(parents=True, exist_ok=True)
    table.assign(recording_timestamp=table["recording_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")).to_csv(
        output_path / DATASET_FILENAME, index=False
    )
    audit.to_csv(output_path / "event_annotation_audit.csv", index=False)
    matches.to_csv(output_path / "inhale_annotation_matches.csv", index=False)
    sensitivity.to_csv(output_path / "usability_sensitivity.csv", index=False)

    joined = table.merge(audit, on=["recording_file", "event_id"], validate="one_to_one")
    matched = matches[matches["matched"]]
    first_usable = joined[joined["usable_order"].fillna(0).between(1, 20)].sort_values("usable_order")
    per_recording = events.groupby("recording_file").size()
    reason_sets = table.loc[~table["usable"], "exclusion_reasons"].value_counts()
    summary = {
        "stage": "Stage 1 - inhale-event dataset finalization",
        "source": {
            "events_csv": _relative(events_path),
            "events_csv_sha256": _sha256(events_path),
            "annotation_csv": _relative(Path(annotation_csv)),
            "annotation_csv_sha256": _sha256(Path(annotation_csv)),
            **wav_summary,
        },
        "provenance": {
            "git": _git_state(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "event_table": {
            "events": int(len(events)),
            "upstream_columns": int(events.shape[1]),
            "feature_columns": list(FEATURE_COLUMNS),
            "detection_columns": list(DETECTION_COLUMNS),
            "missing_values": int(events.isna().sum().sum()),
            "nonfinite_feature_values": int((~np.isfinite(events[list(FEATURE_COLUMNS)].to_numpy(float))).sum()),
            "duplicate_event_keys": int(events.duplicated(["recording_file", "event_id"]).sum()),
            "recordings_with_events": int(per_recording.size),
            "recordings_without_events": int(wav_summary["wav_recordings"] - per_recording.size),
            "recordings_by_event_count": {str(k): int(v) for k, v in per_recording.value_counts().sort_index().items()},
            "chronological_order_matches_recording_id": bool(
                (table.sort_values("chronological_order")["recording_id"].diff().dropna() >= 0).all()
            ),
            "events_with_bridged_non_inhale_windows": int((table["inhale_window_fraction"] < 1 - 1e-9).sum()),
            "inhale_window_fraction_min": float(table["inhale_window_fraction"].min()),
            "duration_s": table["duration_s"].describe().to_dict(),
        },
        "usability_rule": asdict(rule),
        "usability": {
            "usable": int(table["usable"].sum()),
            "excluded": int((~table["usable"]).sum()),
            "flag_counts": {reason: int(table[f"flag_{reason}"].sum()) for reason in EXCLUSION_REASONS},
            "exclusion_reason_combinations": {k: int(v) for k, v in reason_sets.items()},
            "usable_recordings": int(table.loc[table["usable"], "recording_file"].nunique()),
            "usable_events_per_recording": {
                str(k): int(v)
                for k, v in table[table["usable"]].groupby("recording_file").size().value_counts().sort_index().items()
            },
        },
        "annotation_audit": {
            **annotation_summary,
            "match_iou": MATCH_IOU,
            "inhale_annotations_matched": int(matched.shape[0]),
            "inhale_annotations_total": int(matches.shape[0]),
            "matched_iou_mean": float(matched["best_iou"].mean()),
            "matched_iou_median": float(matched["best_iou"].median()),
            "unmatched_inhale_annotations": matches.loc[~matches["matched"]].to_dict("records"),
            "event_status_counts": audit["annotation_status"].value_counts().to_dict(),
            "event_status_by_usability": {
                "usable": joined.loc[joined["usable"], "annotation_status"].value_counts().to_dict(),
                "excluded": joined.loc[~joined["usable"], "annotation_status"].value_counts().to_dict(),
            },
            "short_duration_events_overlapping_inhale_annotation": int(
                (joined["flag_short_duration"] & (joined["overlap_inhale"] > 0)).sum()
            ),
            "matched_event_duration_min_s": float(
                joined.loc[joined["annotation_status"] == "matched_inhale", "duration_s"].min()
            ),
            "inhale_annotation_duration_s": matches["annotation_duration_s"].describe().to_dict(),
        },
        "first_20_usable_events": {
            "recordings": int(first_usable["recording_file"].nunique()),
            "first_recording": first_usable["recording_file"].iloc[0] if len(first_usable) else None,
            "last_recording": first_usable["recording_file"].iloc[-1] if len(first_usable) else None,
            "annotation_status": first_usable["annotation_status"].value_counts().to_dict(),
        },
        "outputs": [
            DATASET_FILENAME, "event_annotation_audit.csv", "inhale_annotation_matches.csv",
            "usability_sensitivity.csv", "dataset_summary.json",
        ],
    }

    if verify_against is not None:
        candidate_path = Path(verify_against)
        check = compare_event_tables(events, pd.read_csv(candidate_path))
        check.update(
            reference=_relative(events_path), reference_sha256=_sha256(events_path),
            candidate=str(candidate_path), candidate_sha256=_sha256(candidate_path),
        )
        with (output_path / "reproducibility_check.json").open("w", encoding="utf-8") as handle:
            json.dump(check, handle, indent=2, default=_json_default)
        summary["reproducibility_check"] = {"identical": check["identical"]}
        summary["outputs"].append("reproducibility_check.json")

    with (output_path / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize the inhale-event dataset (Stage 1)")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_CSV), help="Upstream event CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-duration", type=float, default=UsabilityRule.min_duration_s)
    parser.add_argument("--min-neighbor-gap", type=float, default=UsabilityRule.min_neighbor_gap_s)
    parser.add_argument(
        "--verify-against",
        help="A regenerated event CSV (explore_inhalations.py --output-dir <tmp>) to compare with --events",
    )
    args = parser.parse_args()

    chosen = UsabilityRule(min_duration_s=args.min_duration, min_neighbor_gap_s=args.min_neighbor_gap)
    if chosen != UsabilityRule():
        # Never label a non-default rule as the documented v1 rule.
        chosen = replace(chosen, version="custom")
    result = build_inhale_dataset(
        events_csv=args.events,
        output_dir=args.output_dir,
        rule=chosen,
        verify_against=args.verify_against,
    )
    usability = result["usability"]
    print(f"Events: {result['event_table']['events']}  usable: {usability['usable']}  "
          f"excluded: {usability['excluded']}  flags: {usability['flag_counts']}")
    if "reproducibility_check" in result:
        print(f"Upstream CSV reproduces exactly: {result['reproducibility_check']['identical']}")
