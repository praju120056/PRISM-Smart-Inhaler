"""PRISM final event-level assessment (Stage 9): reference calibration of the V2 deviation score.

See PRISM_RESEARCH_LOG.md Entry 12 and results/final_assessment/.

    8 kHz mono audio -> prism_inference (detector, events, scoreability, V2 features, robust z,
    aggregate deviation = rms of z), unchanged
    -> empirical reference-tail probability p = (1 + #{i : c_i >= a}) / (n + 1): the +1-corrected
       empirical upper-tail probability of the event's deviation a among the n reference deviations c_i,
       each reference inhalation scored by a baseline fitted without its own recording session
    -> assessment WITHIN_REFERENCE_RANGE / OUTSIDE_REFERENCE_RANGE at alpha = 0.05, only if the
       Stage 9 gate adopted the categorical statement (otherwise CONTINUOUS_ONLY)
    -> reliability: STABLE / BORDERLINE with respect to reference sampling (session-bootstrap band
       of the reference cut) and segmentation (event bounds moved by one detector stride)

Meaning: OUTSIDE_REFERENCE_RANGE (p <= 0.05, equivalently a > c*) says the event's acoustic deviation
lies above the one-sided 95% upper reference limit c* of the empirical reference distribution built
from the PRISM reference corpus; WITHIN_REFERENCE_RANGE says it does not.  p is an empirical tail
probability, not a hypothesis-test p-value, and carries no finite-sample guarantee: its calibration
was checked held-out by recording session on the same PRISM corpus (Entries 12-13).  This describes
acoustic deviation relative to that corpus only; it is not a normal/abnormal, anomaly,
technique-quality or clinical classification.  The reference is global (no user identity exists in
the data), not personal.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import config
from baseline_v1 import fit_baseline
from natural_population_analysis import heldout_scores
from post_event import DEFAULT_MODEL_PATH, InhaleEvent, OnnxEventClassifier
from prism_inference import (
    CONTRACT_VERSION as INFERENCE_CONTRACT_VERSION,
    INPUT_DOMAINS,
    SAMPLE_RATE,
    V2_FEATURES,
    ContractError,
    FrozenBaseline,
    analyze_recording,
    anomaly_score,
    check_input,
    event_measurements,
    frozen_from_fit,
    read_wav,
    validate_output,
)


ASSESSMENT_CONTRACT_VERSION = "prism-assessment-v1.0"
DEFAULT_REFERENCE_PATH = Path(config.RESULTS_DIR) / "final_assessment" / "assessment_reference_v1.json"

ALPHA = Fraction(1, 20)                 # 0.05: one-sided 95% upper reference limit of the deviation
BAND_PERCENTILES = (5.0, 95.0)          # session-bootstrap interval of the reference cut
BOOTSTRAP_DRAWS = 2000
SEED = 20261006

CATEGORIES = ("WITHIN_REFERENCE_RANGE", "OUTSIDE_REFERENCE_RANGE", "CONTINUOUS_ONLY", "NOT_ASSESSED")
RELIABILITY = ("STABLE", "BORDERLINE")
SCOREABILITY = ("SCOREABLE", "NOT_SCOREABLE")

# Detector timing, written exactly as post_event.generate_window_predictions computes it so that
# segmentation variants reproduce the detector's float bounds bit for bit.
STRIDE_S = config.WINDOW_STRIDE * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR      # 0.016 s
WINDOW_S = config.WINDOW_SIZE * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR        # 0.2 s
VARIANT_SHIFTS = tuple((ds, de) for ds in (-1, 0, 1) for de in (-1, 0, 1) if (ds, de) != (0, 0))

INTERPRETATION = (
    "aggregate_deviation a is the root-mean-square of robust z-scores of four spectral features relative to a "
    "global reference baseline. reference_tail_probability is the empirical reference-tail probability "
    "p = (1 + #{i : c_i >= a}) / (n + 1), where c_1..c_n are the reference deviations (n = n_reference_events; "
    "each reference inhalation scored by a baseline fitted without its own recording session): an empirical "
    "upper-tail probability of the reference deviation distribution with the +1 correction, not a hypothesis-test "
    "p-value. OUTSIDE_REFERENCE_RANGE means p <= 0.05, equivalently a above the one-sided 95% upper reference limit "
    "(reference.cut); WITHIN_REFERENCE_RANGE means p > 0.05. STABLE means the category is unchanged under "
    "resampling of the reference sessions and +/-1 detector-window boundary shifts; BORDERLINE means it changes "
    "under at least one of them. These statements describe acoustic deviation relative to the PRISM reference "
    "corpus only; they are not normal/abnormal, anomaly, technique-quality or clinical classifications."
)

OUTPUT_KEYS = ("contract_version", "inference_contract_version", "reference_id", "baseline_id", "detector_model_sha256",
               "recording_status", "error", "input", "baseline_domain_validated", "feature_order", "reference",
               "n_events", "n_scoreable", "n_outside_reference_range", "events", "interpretation")
REFERENCE_KEYS = ("alpha", "categorical", "cut", "band", "n_reference_events", "n_reference_sessions",
                  "feature_center", "feature_scale")
EVENT_KEYS = ("event_id", "start_time", "end_time", "duration_s", "start_sample", "end_sample",
              "detector_confidence", "detector_max_confidence", "window_count", "scoreability",
              "not_scoreable_reasons", "feature_values", "mean_rms", "feature_deviations", "aggregate_deviation",
              "reference_tail_probability", "assessment", "assessment_reliability", "segmentation_variants",
              "segmentation_deviation_range", "dominant_feature", "dominant_share", "reason")


class AssessmentError(ValueError):
    """Reference or output does not satisfy the assessment contract."""


# ── Reference statistics ────────────────────────────────────────────────────

def rms_scores(z: np.ndarray) -> np.ndarray:
    """Aggregate deviation per row: sqrt(mean z^2) (identical to prism_inference.anomaly_score)."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    return np.sqrt(np.mean(np.square(z), axis=1))


def tail_count_limit(n: int, alpha: Fraction = ALPHA) -> int:
    """Largest m such that (1 + m) / (n + 1) <= alpha, computed exactly; -1 if none exists."""
    return math.floor(Fraction(alpha) * (n + 1) - 1)


def reference_cut(calibration: Sequence[float], alpha: Fraction = ALPHA) -> float:
    """One-sided upper reference limit c*: the (n - m)-th smallest reference deviation, m = floor(alpha (n + 1) - 1).

    p(s) <= alpha  iff  s > c*.  +inf if the reference is too small for any p <= alpha.
    """
    values = np.sort(np.asarray(calibration, dtype=np.float64))
    m = tail_count_limit(len(values), alpha)
    if m < 0:
        return math.inf
    return float(values[len(values) - m - 1])


def tail_probability(score: float, calibration_sorted: np.ndarray) -> float:
    """Empirical reference-tail probability (1 + #{c >= s}) / (n + 1); ``calibration_sorted`` must be ascending.

    An empirical upper-tail probability with the +1 correction - not a hypothesis-test p-value.
    """
    n = len(calibration_sorted)
    at_least = n - int(np.searchsorted(calibration_sorted, score, side="left"))
    return (1 + at_least) / (n + 1)


def bootstrap_band(calibration: Sequence[float], sessions: Sequence[str], alpha: Fraction = ALPHA,
                   draws: int = BOOTSTRAP_DRAWS, seed: int = SEED,
                   percentiles: tuple[float, float] = BAND_PERCENTILES) -> tuple[float, float, np.ndarray]:
    """Session-bootstrap interval of the reference cut: resample whole sessions with replacement."""
    values = np.asarray(calibration, dtype=np.float64)
    labels = np.asarray(sessions)
    groups = [values[labels == s] for s in sorted(set(labels.tolist()))]
    rng = np.random.default_rng(seed)
    cuts = np.empty(draws)
    for b in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        cuts[b] = reference_cut(np.concatenate([groups[i] for i in chosen]), alpha)
    low, high = np.percentile(cuts, percentiles)
    return float(low), float(high), cuts


def loso_calibration_scores(table: pd.DataFrame, sessions: pd.Series,
                            features: Sequence[str] = V2_FEATURES) -> pd.Series:
    """Score every usable event of ``table`` with a baseline fitted without its own session (Strategy C).

    ``table`` must hold only the reference's usable events; the scores are measured exactly as a new
    session's events would be, which makes them the right calibration set for new events.
    """
    if not table["usable"].astype(bool).all():
        raise AssessmentError("calibration table must contain usable events only")
    if sessions.loc[table.index].nunique() < 2:
        raise AssessmentError("leave-one-session-out calibration needs at least two sessions")
    z, _, _ = heldout_scores(table, list(features), sessions.loc[table.index])
    return pd.Series(rms_scores(z[[f"z_{f}" for f in features]].to_numpy()), index=table.index, name="calibration_score")


@dataclass(frozen=True)
class AssessmentReference:
    """Frozen reference: V2 baseline + leave-one-session-out calibration scores + cut and its band."""

    reference_id: str
    baseline: FrozenBaseline
    calibration_scores: tuple[float, ...]
    calibration_sessions: tuple[str, ...]
    alpha: Fraction
    cut: float
    band: tuple[float, float]
    categorical: bool
    source: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = np.asarray(self.calibration_scores, dtype=np.float64)
        if len(scores) == 0 or len(scores) != len(self.calibration_sessions):
            raise AssessmentError("calibration scores and sessions must be non-empty and aligned")
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise AssessmentError("calibration scores must be finite and >= 0")
        if not np.all(np.diff(scores) >= 0):
            raise AssessmentError("calibration scores must be sorted ascending")
        if self.cut != reference_cut(scores, self.alpha):
            raise AssessmentError("cut does not equal the upper reference limit of the calibration scores")
        low, high = self.band
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise AssessmentError("band must be finite with low <= high")

    @property
    def n_sessions(self) -> int:
        return len(set(self.calibration_sessions))

    def tail_probability(self, score: float) -> float:
        return tail_probability(score, np.asarray(self.calibration_scores))

    def outside(self, score: float) -> bool:
        return score > self.cut

    def reliability(self, scores: Sequence[float]) -> str:
        """STABLE if the score and its segmentation variants all lie on one side of the band.

        The band is the 5th-95th percentile of the cut under session-bootstrap resampling of the reference, so
        STABLE means the category survives both perturbations; it is not a probability that it is correct.
        """
        values = np.asarray(scores, dtype=np.float64)
        low, high = self.band
        return "STABLE" if (values > high).all() or (values <= low).all() else "BORDERLINE"

    def to_dict(self) -> dict:
        return {"reference_id": self.reference_id, "contract_version": ASSESSMENT_CONTRACT_VERSION,
                "baseline": self.baseline.to_dict(), "alpha": str(self.alpha), "cut": self.cut,
                "band": list(self.band), "band_percentiles": list(BAND_PERCENTILES), "categorical": self.categorical,
                "calibration": [{"score": s, "session": g} for s, g in zip(self.calibration_scores,
                                                                            self.calibration_sessions)],
                "source": dict(self.source)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "AssessmentReference":
        if data.get("contract_version") != ASSESSMENT_CONTRACT_VERSION:
            raise AssessmentError(f"reference was issued for {data.get('contract_version')!r}")
        calibration = data["calibration"]
        return cls(reference_id=str(data["reference_id"]), baseline=FrozenBaseline.from_dict(data["baseline"]),
                   calibration_scores=tuple(float(c["score"]) for c in calibration),
                   calibration_sessions=tuple(str(c["session"]) for c in calibration),
                   alpha=Fraction(data["alpha"]), cut=float(data["cut"]), band=tuple(map(float, data["band"])),
                   categorical=bool(data["categorical"]), source=dict(data.get("source", {})))


def fit_reference(table: pd.DataFrame, sessions: pd.Series, reference_id: str, *, categorical: bool,
                  exclude_sessions: Sequence[str] = (), alpha: Fraction = ALPHA, draws: int = BOOTSTRAP_DRAWS,
                  seed: int = SEED, source: Mapping | None = None) -> AssessmentReference:
    """Fit baseline + calibration on the usable events of every session not in ``exclude_sessions``."""
    keep = table["usable"].astype(bool).to_numpy() & ~sessions.isin(list(exclude_sessions)).to_numpy()
    training = table[keep]
    training_sessions = sessions[keep]
    fitted = fit_baseline(training, V2_FEATURES)
    baseline = frozen_from_fit(fitted, f"{reference_id}/baseline", n_sessions=int(training_sessions.nunique()),
                               source={"fit": "median and 1.4826*MAD over the reference's usable events",
                                       "excluded_sessions": sorted(exclude_sessions)})
    calibration = loso_calibration_scores(training, training_sessions)
    order = np.argsort(calibration.to_numpy(), kind="mergesort")
    scores = calibration.to_numpy()[order]
    labels = training_sessions.to_numpy()[order]
    low, high, _ = bootstrap_band(scores, labels, alpha, draws, seed)
    return AssessmentReference(reference_id=reference_id, baseline=baseline,
                               calibration_scores=tuple(float(s) for s in scores),
                               calibration_sessions=tuple(str(g) for g in labels), alpha=Fraction(alpha),
                               cut=reference_cut(scores, alpha), band=(low, high), categorical=categorical,
                               source={"n_events": int(len(scores)), "n_sessions": int(training_sessions.nunique()),
                                       "excluded_sessions": sorted(exclude_sessions), "bootstrap_draws": draws,
                                       "seed": seed, **dict(source or {})})


def load_reference(path: str | Path = DEFAULT_REFERENCE_PATH) -> AssessmentReference:
    return AssessmentReference.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_reference(reference: AssessmentReference, path: str | Path) -> None:
    Path(path).write_text(json.dumps(reference.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


# ── Segmentation variants ───────────────────────────────────────────────────

def window_count(n_samples: int) -> int:
    """Number of detector windows for a recording of ``n_samples`` (frames = 1 + n // hop)."""
    frames = 1 + n_samples // config.LIBROSA_HOP_LENGTH
    return 0 if frames < config.WINDOW_SIZE else (frames - config.WINDOW_SIZE) // config.WINDOW_STRIDE + 1


def window_index_bounds(start_time: float, end_time: float) -> tuple[int, int]:
    """First and last Inhale window index of an event whose end is not clamped at the recording end."""
    first = int(round(start_time / STRIDE_S))
    last = int(round((end_time - WINDOW_S) / STRIDE_S))
    if first * STRIDE_S != start_time or last * STRIDE_S + WINDOW_S != end_time:
        raise AssessmentError("event bounds are not the detector's window bounds")
    return first, last


def segmentation_variants(start_time: float, end_time: float, n_samples: int) -> list[tuple[float, float]]:
    """Event bounds moved by one detector stride at the start and/or end (reference float formula)."""
    first, last = window_index_bounds(start_time, end_time)
    duration = n_samples / SAMPLE_RATE
    n_windows = window_count(n_samples)
    variants = []
    for ds, de in VARIANT_SHIFTS:
        a, b = first + ds, last + de
        if a < 0 or b >= n_windows or b < a:
            continue
        start = a * STRIDE_S
        end = min(b * STRIDE_S + WINDOW_S, duration)
        variants.append((start, end))
    return variants


def variant_scores(waveform: np.ndarray, bounds: Sequence[tuple[float, float]], baseline: FrozenBaseline) -> list[float]:
    """Aggregate deviation of each variant segment (unchanged V2 measurement); non-finite variants are skipped."""
    scores = []
    for start, end in bounds:
        event = InhaleEvent(label="Inhale", start=start, end=end, duration=end - start, confidence=0.0,
                            max_confidence=0.0, window_count=1, window_indices=())
        measured = event_measurements(waveform, event)
        if measured is None:
            continue
        values = [measured[f] for f in V2_FEATURES]
        if all(math.isfinite(v) for v in values):
            scores.append(anomaly_score(baseline.z(values)))
    return scores


# ── Assessment of one recording ─────────────────────────────────────────────

def sample_bounds(start_time: float, end_time: float, n_samples: int) -> tuple[int, int]:
    """Integer sample slice used by the measurement (post_event.extract_event_audio)."""
    return max(0, int(np.floor(start_time * SAMPLE_RATE))), min(n_samples, int(np.ceil(end_time * SAMPLE_RATE)))


def _reason(category: str, p: float | None, dominant: str | None, reasons: Sequence[str],
            alpha: Fraction = ALPHA) -> str:
    if category == "NOT_ASSESSED":
        return "not scoreable: " + ", ".join(reasons)
    limit = f"one-sided {100 * (1 - float(alpha)):g}% upper reference limit"
    text = {"WITHIN_REFERENCE_RANGE": f"deviation at or below the {limit}",
            "OUTSIDE_REFERENCE_RANGE": f"deviation above the {limit}",
            "CONTINUOUS_ONLY": "no categorical statement (continuous deviation only)"}[category]
    return f"{text} (empirical reference-tail probability {p:.3f}; largest deviation: {dominant})"


def assess_event(event: Mapping, reference: AssessmentReference, waveform: np.ndarray | None) -> dict:
    """Attach the final assessment to one prism_inference event."""
    n_samples = len(waveform) if waveform is not None else 0
    start_sample, end_sample = sample_bounds(event["start_time"], event["end_time"], n_samples)
    out = {
        "event_id": event["event_id"], "start_time": event["start_time"], "end_time": event["end_time"],
        "duration_s": event["duration_s"], "start_sample": start_sample, "end_sample": end_sample,
        "detector_confidence": event["detector_confidence"], "detector_max_confidence": event["detector_max_confidence"],
        "window_count": event["window_count"],
        "scoreability": "SCOREABLE" if event["status"] == "SCORE_ONLY" else "NOT_SCOREABLE",
        "not_scoreable_reasons": list(event["not_scoreable_reasons"]),
        "feature_values": event["feature_values"], "mean_rms": event["mean_rms"],
        "feature_deviations": event["feature_z_scores"], "aggregate_deviation": event["anomaly_score"],
        "reference_tail_probability": None, "assessment": "NOT_ASSESSED", "assessment_reliability": None,
        "segmentation_variants": 0, "segmentation_deviation_range": None,
        "dominant_feature": None, "dominant_share": None, "reason": "",
    }
    if out["scoreability"] == "NOT_SCOREABLE":
        out["reason"] = _reason("NOT_ASSESSED", None, None, out["not_scoreable_reasons"])
        return out
    score = float(event["anomaly_score"])
    z = np.array([event["feature_z_scores"][f] for f in V2_FEATURES])
    squared = z ** 2
    dominant = V2_FEATURES[int(np.argmax(squared))]
    variants = variant_scores(waveform, segmentation_variants(event["start_time"], event["end_time"], n_samples),
                              reference.baseline)
    p = reference.tail_probability(score)
    if reference.categorical:
        category = "OUTSIDE_REFERENCE_RANGE" if reference.outside(score) else "WITHIN_REFERENCE_RANGE"
        reliability = reference.reliability([score, *variants])
    else:
        category, reliability = "CONTINUOUS_ONLY", None
    out.update({"reference_tail_probability": p, "assessment": category, "assessment_reliability": reliability,
                "segmentation_variants": len(variants),
                "segmentation_deviation_range": [float(min([score, *variants])), float(max([score, *variants]))],
                "dominant_feature": dominant,
                "dominant_share": float(squared.max() / squared.sum()) if squared.sum() > 0 else 0.0})
    out["reason"] = _reason(category, p, dominant, (), reference.alpha)
    return out


def assess_recording(audio, sample_rate: int, reference: AssessmentReference, *,
                     detector: OnnxEventClassifier | None = None, input_domain: str = "unknown",
                     recording_id: str | None = None, recorded_at: str | None = None) -> dict:
    """Raw 8 kHz mono audio -> the canonical PRISM output (inference contract V2 + final assessment)."""
    inference = analyze_recording(audio, sample_rate, detector=detector, baseline=reference.baseline,
                                  input_domain=input_domain, recording_id=recording_id, recorded_at=recorded_at)
    waveform, _ = check_input(audio, sample_rate)
    events = [assess_event(e, reference, waveform) for e in inference["events"]]
    baseline = reference.baseline
    output = {
        "contract_version": ASSESSMENT_CONTRACT_VERSION,
        "inference_contract_version": inference["contract_version"],
        "reference_id": reference.reference_id,
        "baseline_id": inference["baseline_id"],
        "detector_model_sha256": inference["detector_model_sha256"],
        "recording_status": inference["recording_status"],
        "error": inference["error"],
        "input": inference["input"],
        "baseline_domain_validated": inference["baseline_domain_validated"],
        "feature_order": inference["feature_order"],
        "reference": {"alpha": float(reference.alpha), "categorical": reference.categorical, "cut": reference.cut,
                      "band": list(reference.band), "n_reference_events": len(reference.calibration_scores),
                      "n_reference_sessions": reference.n_sessions,
                      "feature_center": dict(zip(baseline.features, baseline.center)),
                      "feature_scale": dict(zip(baseline.features, baseline.scale))},
        "n_events": len(events),
        "n_scoreable": sum(e["scoreability"] == "SCOREABLE" for e in events),
        "n_outside_reference_range": sum(e["assessment"] == "OUTSIDE_REFERENCE_RANGE" for e in events),
        "events": events,
        "interpretation": INTERPRETATION,
    }
    validate_assessment_output(output, inference)
    return output


# ── Output validation (executable form of assessment_output.schema.json) ───

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssessmentError(message)


def validate_assessment_output(output: Mapping, inference: Mapping | None = None) -> None:
    """Raise AssessmentError unless ``output`` satisfies the assessment contract."""
    _require(tuple(output) == OUTPUT_KEYS, f"output keys must be exactly {OUTPUT_KEYS}")
    _require(output["contract_version"] == ASSESSMENT_CONTRACT_VERSION, "wrong contract_version")
    _require(output["inference_contract_version"] == INFERENCE_CONTRACT_VERSION, "wrong inference_contract_version")
    _require(tuple(output["reference"]) == REFERENCE_KEYS, f"reference keys must be exactly {REFERENCE_KEYS}")
    if inference is not None:
        validate_output(inference)
    events = output["events"]
    _require(output["n_events"] == len(events), "n_events mismatch")
    _require(output["n_scoreable"] == sum(e["scoreability"] == "SCOREABLE" for e in events), "n_scoreable mismatch")
    _require(output["n_outside_reference_range"] == sum(e["assessment"] == "OUTSIDE_REFERENCE_RANGE" for e in events),
             "n_outside_reference_range mismatch")
    categorical = output["reference"]["categorical"]
    for position, event in enumerate(events):
        _require(tuple(event) == EVENT_KEYS, f"event keys must be exactly {EVENT_KEYS}")
        _require(event["event_id"] == position, "event_id must be 0, 1, ... in order")
        _require(event["scoreability"] in SCOREABILITY and event["assessment"] in CATEGORIES, "unknown status")
        _require(0 <= event["start_sample"] < event["end_sample"], "sample bounds must satisfy 0 <= start < end")
        if event["scoreability"] == "NOT_SCOREABLE":
            _require(event["assessment"] == "NOT_ASSESSED" and event["not_scoreable_reasons"], "NOT_SCOREABLE events "
                     "are NOT_ASSESSED and carry reasons")
            _require(all(event[k] is None for k in ("aggregate_deviation", "feature_deviations",
                                                     "reference_tail_probability", "assessment_reliability")),
                     "NOT_SCOREABLE events carry no deviation, probability or reliability")
            continue
        _require(not event["not_scoreable_reasons"], "SCOREABLE events have no not_scoreable_reasons")
        p, score = event["reference_tail_probability"], event["aggregate_deviation"]
        _require(p is not None and 0 < p <= 1 and math.isfinite(score) and score >= 0, "invalid score or probability")
        z = np.array([event["feature_deviations"][f] for f in V2_FEATURES])
        _require(math.isclose(score, anomaly_score(z), rel_tol=1e-12, abs_tol=1e-15), "aggregate_deviation != rms(z)")
        if categorical:
            expected = "OUTSIDE_REFERENCE_RANGE" if score > output["reference"]["cut"] else "WITHIN_REFERENCE_RANGE"
            _require(event["assessment"] == expected, "assessment disagrees with the reference cut")
            _require(event["assessment"] != "OUTSIDE_REFERENCE_RANGE" or p <= output["reference"]["alpha"] + 1e-12,
                     "OUTSIDE events must have p <= alpha")
            _require(event["assessment_reliability"] in RELIABILITY, "categorical assessments need a reliability")
        else:
            _require(event["assessment"] == "CONTINUOUS_ONLY" and event["assessment_reliability"] is None,
                     "non-categorical references give CONTINUOUS_ONLY without reliability")
        low, high = event["segmentation_deviation_range"]
        _require(low <= score <= high, "segmentation range must contain the event's own deviation")
    json.dumps(output, allow_nan=False)


def output_json_schema() -> dict:
    """JSON Schema (draft 2020-12) of the assessment output; mirrors validate_assessment_output."""
    number, nullable_number = {"type": "number"}, {"type": ["number", "null"]}
    features = {"type": "object", "properties": {f: number for f in V2_FEATURES}, "required": list(V2_FEATURES),
                "additionalProperties": False}
    nullable_features = {"oneOf": [features, {"type": "null"}]}
    event = {"type": "object", "additionalProperties": False, "required": list(EVENT_KEYS), "properties": {
        "event_id": {"type": "integer", "minimum": 0}, "start_time": number, "end_time": number, "duration_s": number,
        "start_sample": {"type": "integer", "minimum": 0}, "end_sample": {"type": "integer", "minimum": 1},
        "detector_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "detector_max_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "window_count": {"type": "integer", "minimum": 1}, "scoreability": {"enum": list(SCOREABILITY)},
        "not_scoreable_reasons": {"type": "array", "items": {"type": "string"}},
        "feature_values": nullable_features, "mean_rms": nullable_number, "feature_deviations": nullable_features,
        "aggregate_deviation": nullable_number, "reference_tail_probability": nullable_number,
        "assessment": {"enum": list(CATEGORIES)}, "assessment_reliability": {"enum": [*RELIABILITY, None]},
        "segmentation_variants": {"type": "integer", "minimum": 0},
        "segmentation_deviation_range": {"oneOf": [{"type": "array", "items": number, "minItems": 2, "maxItems": 2},
                                                   {"type": "null"}]},
        "dominant_feature": {"enum": [*V2_FEATURES, None]}, "dominant_share": nullable_number,
        "reason": {"type": "string"}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://prism.local/schemas/{ASSESSMENT_CONTRACT_VERSION}/recording-assessment.json",
            "title": "PRISM recording assessment (inference contract V2 + reference-calibrated event assessment)",
            "type": "object", "additionalProperties": False, "required": list(OUTPUT_KEYS), "properties": {
                "contract_version": {"const": ASSESSMENT_CONTRACT_VERSION},
                "inference_contract_version": {"const": INFERENCE_CONTRACT_VERSION},
                "reference_id": {"type": "string"}, "baseline_id": {"type": "string"},
                "detector_model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "recording_status": {"enum": ["EVENTS_DETECTED", "NO_INHALATION_DETECTED", "INPUT_ERROR"]},
                "error": {"type": ["string", "null"]}, "input": {"type": "object"},
                "baseline_domain_validated": {"type": "boolean"}, "feature_order": {"const": list(V2_FEATURES)},
                "reference": {"type": "object", "required": list(REFERENCE_KEYS), "additionalProperties": False,
                              "properties": {"alpha": number, "categorical": {"type": "boolean"}, "cut": number,
                                             "band": {"type": "array", "items": number, "minItems": 2, "maxItems": 2},
                                             "n_reference_events": {"type": "integer"},
                                             "n_reference_sessions": {"type": "integer"},
                                             "feature_center": features, "feature_scale": features}},
                "n_events": {"type": "integer", "minimum": 0}, "n_scoreable": {"type": "integer", "minimum": 0},
                "n_outside_reference_range": {"type": "integer", "minimum": 0},
                "events": {"type": "array", "items": event}, "interpretation": {"const": INTERPRETATION}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM final assessment of one 8 kHz mono WAV recording")
    parser.add_argument("wav")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-domain", default="unknown", choices=INPUT_DOMAINS)
    args = parser.parse_args()
    waveform_, rate = read_wav(args.wav)
    result = assess_recording(waveform_, rate, load_reference(args.reference),
                              detector=OnnxEventClassifier(model_path=args.model), input_domain=args.input_domain,
                              recording_id=Path(args.wav).name)
    print(json.dumps(result, indent=2, allow_nan=False))
