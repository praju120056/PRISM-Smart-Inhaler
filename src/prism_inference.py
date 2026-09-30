"""PRISM V2 inference contract: reference implementation (Stage 8).

See PRISM_RESEARCH_LOG.md Entry 10 and results/v2_validation/inference_contract_v2.json.

    8 kHz mono audio -> existing ONNX event detector -> grouped Inhale events
    -> Stage 1 usability rule v1 (which events may be scored)
    -> V2 features + separate level channel (mean_rms)
    -> frozen global baseline (loaded, never fitted here) -> robust z
    -> anomaly_score = rms_z

The output is SCORE_ONLY: a standardized distance from a baseline fitted on the
reference dataset.  There is no threshold, no NORMAL/ANOMALY label and no
clinical or technique-quality meaning.  A recording without a detected
inhalation is a recording-level state (NO_INHALATION_DETECTED), never an event.

Every computation that produced the Stage 1 events and features is reused
unchanged from ``post_event`` (detector, grouping, slicing, measurements), so
the contract describes exactly the events the baseline was fitted on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import config
from inhale_dataset import EXCLUSION_REASONS, UsabilityRule
from post_event import (
    DEFAULT_MODEL_PATH,
    InhaleEvent,
    OnnxEventClassifier,
    TemporalGroupingConfig,
    analyze_inhalation,
    generate_window_predictions,
    group_inhale_events,
)


CONTRACT_VERSION = "prism-inference-v2.0"
DEFAULT_BASELINE_PATH = Path(config.RESULTS_DIR) / "v2_validation" / "v2_baseline.json"

# Order is part of the contract: arrays, z-scores and the baseline all follow it.
V2_FEATURES = ("spectral_centroid_mean", "spectral_flatness_mean", "spectral_centroid_std", "spectral_rolloff_std")
LEVEL_CHANNEL = "mean_rms"
MAD_SCALE = 1.4826

SAMPLE_RATE = config.LIBROSA_SR                                        # 8000 Hz, no resampling
DETECTOR_WINDOW_FRAMES = config.WINDOW_SIZE                            # 25 frames = 0.2 s
MIN_SAMPLES = (DETECTOR_WINDOW_FRAMES - 1) * config.LIBROSA_HOP_LENGTH  # 1536: one full detector window
USABILITY = UsabilityRule()                                            # Stage 1 rule v1
GROUPING = TemporalGroupingConfig()                                    # defaults that produced the Stage 1 events
TIME_DECIMALS = 6                                                      # as inhale_dataset: removes float noise in rule checks

RECORDING_STATUSES = ("EVENTS_DETECTED", "NO_INHALATION_DETECTED", "INPUT_ERROR")
EVENT_STATUSES = ("SCORE_ONLY", "NOT_SCOREABLE")
NOT_SCOREABLE_REASONS = EXCLUSION_REASONS   # nonfinite_feature, short_duration, close_neighbor, recording_boundary
INPUT_ERRORS = ("unsupported_sample_rate", "invalid_shape", "empty_audio", "nonfinite_audio",
                "amplitude_out_of_range", "shorter_than_one_detector_window", "detector_feature_extraction_failed")
INPUT_DOMAINS = ("reference_dataset", "prism_hardware", "unknown")

INTERPRETATION = (
    "SCORE_ONLY: anomaly_score is the root-mean-square of robust z-scores of four spectral features relative "
    "to a frozen global baseline fitted on the reference dataset. It is a distance, not a probability, quality "
    "rating, NORMAL/ANOMALY decision or clinical assessment. No threshold is defined."
)
FORBIDDEN_DERIVED_OUTPUTS = (
    "NORMAL / ANOMALY labels (no validated threshold exists)",
    "quality percentages or scores rescaled to 0-100",
    "Correct / Incorrect or good / poor technique labels",
    "technique error codes (e.g. too_short, too_fast, incomplete_breath)",
    "clinical or alert states derived from anomaly_score",
)

OUTPUT_KEYS = ("contract_version", "baseline_id", "detector_model_sha256", "recording_status", "error", "input",
               "baseline_domain_validated", "feature_order", "n_events", "n_scored", "events", "interpretation")
INPUT_KEYS = ("sample_rate", "n_samples", "duration_s", "input_domain", "recording_id", "recorded_at")
EVENT_KEYS = ("event_id", "start_time", "end_time", "duration_s", "detector_confidence", "detector_max_confidence",
              "window_count", "status", "not_scoreable_reasons", "anomaly_score", "feature_values",
              "feature_z_scores", "mean_rms")


class ContractError(ValueError):
    """Input, baseline or output does not satisfy the inference contract."""


# ── Frozen baseline ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrozenBaseline:
    """Per-feature median / 1.4826*MAD in V2 order.  Loaded, never fitted, at inference."""

    baseline_id: str
    features: tuple[str, ...]
    center: tuple[float, ...]
    mad: tuple[float, ...]
    scale: tuple[float, ...]
    mad_scale: float
    n_events: int
    n_sessions: int
    source: Mapping

    def __post_init__(self) -> None:
        if tuple(self.features) != V2_FEATURES:
            raise ContractError(f"baseline features {self.features} do not match the V2 order {V2_FEATURES}")
        for name in ("center", "mad", "scale"):
            values = getattr(self, name)
            if len(values) != len(V2_FEATURES) or not all(math.isfinite(v) for v in values):
                raise ContractError(f"baseline {name} must hold {len(V2_FEATURES)} finite values")
        if not all(v > 0 for v in self.mad):
            raise ContractError("every baseline MAD must be > 0")
        if self.mad_scale != MAD_SCALE:
            raise ContractError(f"mad_scale must be {MAD_SCALE}")
        for mad, scale in zip(self.mad, self.scale):
            if not math.isclose(scale, self.mad_scale * mad, rel_tol=1e-12, abs_tol=0.0):
                raise ContractError("baseline scale must equal mad_scale * MAD")

    def z(self, values: Sequence[float]) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1] != len(V2_FEATURES):
            raise ContractError(f"expected {len(V2_FEATURES)} feature values in V2 order")
        return (x - np.asarray(self.center)) / np.asarray(self.scale)

    def to_dict(self) -> dict:
        return {"baseline_id": self.baseline_id, "contract_version": CONTRACT_VERSION,
                "features": list(self.features),
                "parameters": {f: {"center": c, "mad": m, "scale": s}
                               for f, c, m, s in zip(self.features, self.center, self.mad, self.scale)},
                "mad_scale": self.mad_scale, "n_events": self.n_events, "n_sessions": self.n_sessions,
                "source": dict(self.source)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "FrozenBaseline":
        if data.get("contract_version") != CONTRACT_VERSION:
            raise ContractError(f"baseline was issued for {data.get('contract_version')!r}, not {CONTRACT_VERSION!r}")
        features = tuple(data["features"])
        parameters = data["parameters"]
        if set(parameters) != set(features):
            raise ContractError("baseline parameters and feature list differ")
        return cls(baseline_id=str(data["baseline_id"]), features=features,
                   center=tuple(float(parameters[f]["center"]) for f in features),
                   mad=tuple(float(parameters[f]["mad"]) for f in features),
                   scale=tuple(float(parameters[f]["scale"]) for f in features),
                   mad_scale=float(data["mad_scale"]), n_events=int(data["n_events"]),
                   n_sessions=int(data["n_sessions"]), source=dict(data.get("source", {})))


def frozen_from_fit(fitted, baseline_id: str, n_sessions: int, source: Mapping) -> FrozenBaseline:
    """Convert a ``baseline_v1.RobustBaseline`` fitted on V2 features into the contract form."""
    if tuple(fitted.features) != V2_FEATURES:
        raise ContractError("the fitted baseline must use exactly the V2 features in V2 order")
    parameters = [fitted.parameters[f] for f in V2_FEATURES]
    return FrozenBaseline(baseline_id=baseline_id, features=V2_FEATURES,
                          center=tuple(p.median for p in parameters), mad=tuple(p.mad for p in parameters),
                          scale=tuple(p.scale for p in parameters), mad_scale=float(fitted.mad_scale),
                          n_events=int(fitted.n_calibration), n_sessions=int(n_sessions), source=dict(source))


def load_baseline(path: str | Path = DEFAULT_BASELINE_PATH) -> FrozenBaseline:
    return FrozenBaseline.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_baseline(baseline: FrozenBaseline, path: str | Path) -> None:
    Path(path).write_text(json.dumps(baseline.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


# ── Input ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_pcm16le(raw: bytes) -> np.ndarray:
    """16-bit little-endian PCM -> float32 in [-1, 1): sample / 32768 (the soundfile/librosa convention)."""
    if len(raw) % 2:
        raise ContractError("PCM16 byte stream must have an even length")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Float32 waveform and sample rate of a WAV file (multi-channel files are averaged to mono)."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0], int(sample_rate)


def check_input(audio, sample_rate: int) -> tuple[np.ndarray | None, str | None]:
    """Validated float32 mono waveform, or an INPUT_ERROR code."""
    if sample_rate != SAMPLE_RATE:
        return None, "unsupported_sample_rate"
    values = np.asarray(audio)
    if values.ndim == 2:   # same channel convention as post_event._as_mono
        values = values.mean(axis=1 if values.shape[0] >= values.shape[1] else 0)
    if values.ndim != 1:
        return None, "invalid_shape"
    if values.size == 0:
        return None, "empty_audio"
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        return None, "nonfinite_audio"
    if float(np.max(np.abs(values))) > 1.0:
        return None, "amplitude_out_of_range"
    if values.size < MIN_SAMPLES:
        return None, "shorter_than_one_detector_window"
    return values, None


# ── Events, usability and features ──────────────────────────────────────────

def event_measurements(waveform: np.ndarray, event: InhaleEvent, sample_rate: int = SAMPLE_RATE) -> dict | None:
    """V2 features and mean_rms of one event, via the unchanged post_event measurements."""
    try:
        analysis = analyze_inhalation(waveform, event, sample_rate=sample_rate)
    except ValueError:            # extraction failed (e.g. non-finite spectral values)
        return None
    spectral = analysis["spectral"]
    return {
        "spectral_centroid_mean": spectral["spectral_centroid"]["mean"],
        "spectral_flatness_mean": spectral["spectral_flatness"]["mean"],
        "spectral_centroid_std": spectral["spectral_centroid"]["std"],
        "spectral_rolloff_std": spectral["spectral_rolloff"]["std"],
        LEVEL_CHANNEL: analysis["mean_rms"],
    }


def not_scoreable_reasons(events: Sequence[InhaleEvent], recording_duration_s: float,
                          finite: Sequence[bool], rule: UsabilityRule = USABILITY) -> list[list[str]]:
    """Stage 1 usability rule v1 applied to one recording's detected events (chronological)."""
    reasons = []
    for i, event in enumerate(events):
        gaps = []
        if i > 0:
            gaps.append(np.round(event.start - events[i - 1].end, TIME_DECIMALS))
        if i + 1 < len(events):
            gaps.append(np.round(events[i + 1].start - event.end, TIME_DECIMALS))
        flags = {
            "nonfinite_feature": not finite[i],
            "short_duration": np.round(event.duration, TIME_DECIMALS) < rule.min_duration_s,
            "close_neighbor": bool(gaps) and min(gaps) < rule.min_neighbor_gap_s,
            "recording_boundary": event.start <= rule.boundary_tolerance_s
                                  or event.end >= recording_duration_s - rule.boundary_tolerance_s,
        }
        reasons.append([reason for reason in NOT_SCOREABLE_REASONS if flags[reason]])
    return reasons


def anomaly_score(z: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(z))))


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def _event_output(event_id: int, event: InhaleEvent, measured: dict | None, reasons: list[str],
                  baseline: FrozenBaseline) -> dict:
    values = None if measured is None else [measured[f] for f in V2_FEATURES]
    features_finite = values is not None and all(_finite(v) for v in values)
    level = None if measured is None or not _finite(measured[LEVEL_CHANNEL]) else float(measured[LEVEL_CHANNEL])
    scored = not reasons
    z = baseline.z(values) if scored else None
    return {
        "event_id": event_id,
        "start_time": float(event.start),
        "end_time": float(event.end),
        "duration_s": float(event.duration),
        "detector_confidence": float(event.confidence),
        "detector_max_confidence": float(event.max_confidence),
        "window_count": int(event.window_count),
        "status": "SCORE_ONLY" if scored else "NOT_SCOREABLE",
        "not_scoreable_reasons": list(reasons),
        "anomaly_score": anomaly_score(z) if scored else None,
        "feature_values": {f: float(v) for f, v in zip(V2_FEATURES, values)} if features_finite else None,
        "feature_z_scores": {f: float(v) for f, v in zip(V2_FEATURES, z)} if scored else None,
        "mean_rms": level,
    }


def analyze_recording(audio, sample_rate: int, *, detector: OnnxEventClassifier | None = None,
                      baseline: FrozenBaseline | None = None, input_domain: str = "unknown",
                      recording_id: str | None = None, recorded_at: str | None = None) -> dict:
    """Apply the V2 inference contract to one recording and return the validated output object."""
    if input_domain not in INPUT_DOMAINS:
        raise ContractError(f"input_domain must be one of {INPUT_DOMAINS}")
    baseline = baseline or load_baseline()
    detector = detector or OnnxEventClassifier()
    waveform, error = check_input(audio, sample_rate)
    raw = np.asarray(audio)
    n_samples = len(waveform) if waveform is not None else (int(raw.shape[0]) if raw.ndim >= 1 else 0)
    output = {
        "contract_version": CONTRACT_VERSION,
        "baseline_id": baseline.baseline_id,
        "detector_model_sha256": sha256_file(str(detector.model_path)),
        "recording_status": "INPUT_ERROR",
        "error": error,
        "input": {"sample_rate": int(sample_rate), "n_samples": n_samples,
                  "duration_s": n_samples / sample_rate if sample_rate > 0 else None,
                  "input_domain": input_domain, "recording_id": recording_id, "recorded_at": recorded_at},
        "baseline_domain_validated": input_domain == "reference_dataset",
        "feature_order": list(V2_FEATURES),
        "n_events": 0,
        "n_scored": 0,
        "events": [],
        "interpretation": INTERPRETATION,
    }
    predictions = None
    if error is None:
        try:
            predictions = generate_window_predictions(waveform, sample_rate=sample_rate, model=detector)
        except ValueError:            # the detector's 124-feature frames were not finite
            output["error"] = "detector_feature_extraction_failed"
    if predictions is not None:
        events = group_inhale_events(predictions, GROUPING)
        measured = [event_measurements(waveform, event, sample_rate) for event in events]
        finite = [m is not None and all(_finite(m[f]) for f in (*V2_FEATURES, LEVEL_CHANNEL)) for m in measured]
        reasons = not_scoreable_reasons(events, len(waveform) / sample_rate, finite)
        output["events"] = [_event_output(i, e, m, r, baseline) for i, (e, m, r) in enumerate(zip(events, measured, reasons))]
        output["recording_status"] = "EVENTS_DETECTED" if events else "NO_INHALATION_DETECTED"
        output["n_events"] = len(events)
        output["n_scored"] = sum(e["status"] == "SCORE_ONLY" for e in output["events"])
    validate_output(output)
    return output


# ── Output validation (the executable form of inference_output.schema.json) ─

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_output(output: Mapping) -> None:
    """Raise ContractError unless ``output`` satisfies the V2 output contract."""
    _require(tuple(output) == OUTPUT_KEYS, f"output keys must be exactly {OUTPUT_KEYS}")
    _require(output["contract_version"] == CONTRACT_VERSION, "wrong contract_version")
    _require(output["recording_status"] in RECORDING_STATUSES, "unknown recording_status")
    _require(tuple(output["input"]) == INPUT_KEYS, f"input keys must be exactly {INPUT_KEYS}")
    _require(output["input"]["input_domain"] in INPUT_DOMAINS, "unknown input_domain")
    _require(output["baseline_domain_validated"] == (output["input"]["input_domain"] == "reference_dataset"),
             "baseline_domain_validated must be true only for the reference dataset")
    _require(list(output["feature_order"]) == list(V2_FEATURES), "feature_order must equal the V2 order")
    events = output["events"]
    _require(output["n_events"] == len(events), "n_events must equal the number of events")
    _require(output["n_scored"] == sum(e["status"] == "SCORE_ONLY" for e in events), "n_scored mismatch")
    status = output["recording_status"]
    if status == "INPUT_ERROR":
        _require(output["error"] in INPUT_ERRORS and not events, "INPUT_ERROR needs a known error and no events")
    else:
        _require(output["error"] is None, "error must be null unless recording_status is INPUT_ERROR")
        _require((status == "EVENTS_DETECTED") == bool(events),
                 "EVENTS_DETECTED needs >= 1 event; NO_INHALATION_DETECTED needs none")
    previous_start = -math.inf
    for position, event in enumerate(events):
        _require(tuple(event) == EVENT_KEYS, f"event keys must be exactly {EVENT_KEYS}")
        _require(event["event_id"] == position, "event_id must be 0, 1, ... in chronological order")
        _require(0 <= event["start_time"] < event["end_time"], "event times must satisfy 0 <= start < end")
        _require(event["start_time"] >= previous_start, "events must be chronological")
        previous_start = event["start_time"]
        _require(math.isclose(event["duration_s"], event["end_time"] - event["start_time"], abs_tol=1e-9),
                 "duration_s must equal end_time - start_time")
        for key in ("detector_confidence", "detector_max_confidence"):
            _require(0.0 <= event[key] <= 1.0, f"{key} must lie in [0, 1]")
        _require(event["window_count"] >= 1, "window_count must be >= 1")
        _require(event["status"] in EVENT_STATUSES, "unknown event status")
        reasons = event["not_scoreable_reasons"]
        _require(all(r in NOT_SCOREABLE_REASONS for r in reasons) and len(set(reasons)) == len(reasons),
                 "unknown or repeated not_scoreable_reasons")
        values, z = event["feature_values"], event["feature_z_scores"]
        if values is not None:
            _require(list(values) == list(V2_FEATURES) and all(_finite(v) for v in values.values()),
                     "feature_values must be finite and in V2 order")
        _require(event["mean_rms"] is None or (_finite(event["mean_rms"]) and event["mean_rms"] >= 0),
                 "mean_rms must be null or finite and >= 0")
        if event["status"] == "SCORE_ONLY":
            _require(not reasons, "SCORE_ONLY events have no not_scoreable_reasons")
            _require(values is not None and z is not None and list(z) == list(V2_FEATURES)
                     and all(_finite(v) for v in z.values()), "SCORE_ONLY events need finite V2 values and z")
            score = event["anomaly_score"]
            _require(_finite(score) and score >= 0, "anomaly_score must be finite and >= 0")
            _require(math.isclose(score, anomaly_score(np.array(list(z.values()))), rel_tol=1e-12, abs_tol=1e-15),
                     "anomaly_score must equal sqrt(mean z^2)")
        else:
            _require(bool(reasons), "NOT_SCOREABLE events need at least one reason")
            _require(event["anomaly_score"] is None and z is None,
                     "NOT_SCOREABLE events carry no anomaly_score and no z-scores")
            _require(values is not None or "nonfinite_feature" in reasons,
                     "feature_values may be null only when a feature is non-finite")
    json.dumps(output, allow_nan=False)   # no NaN / infinity anywhere


# ── Machine-readable schema and feature specification ───────────────────────

def output_json_schema() -> dict:
    """JSON Schema (draft 2020-12) mirroring ``validate_output`` for non-Python consumers."""
    number = {"type": "number"}
    feature_object = {"type": "object", "properties": {f: number for f in V2_FEATURES},
                      "required": list(V2_FEATURES), "additionalProperties": False}
    event = {
        "type": "object",
        "properties": {
            "event_id": {"type": "integer", "minimum": 0},
            "start_time": {"type": "number", "minimum": 0, "description": "seconds from recording start"},
            "end_time": {"type": "number", "description": "seconds from recording start"},
            "duration_s": {"type": "number", "description": "end_time - start_time"},
            "detector_confidence": {"type": "number", "minimum": 0, "maximum": 1,
                                    "description": "mean P(Inhale) over the event's Inhale windows"},
            "detector_max_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "window_count": {"type": "integer", "minimum": 1},
            "status": {"enum": list(EVENT_STATUSES)},
            "not_scoreable_reasons": {"type": "array", "items": {"enum": list(NOT_SCOREABLE_REASONS)},
                                      "uniqueItems": True},
            "anomaly_score": {"type": ["number", "null"], "minimum": 0},
            "feature_values": {"oneOf": [feature_object, {"type": "null"}]},
            "feature_z_scores": {"oneOf": [feature_object, {"type": "null"}]},
            "mean_rms": {"type": ["number", "null"], "minimum": 0,
                         "description": "level channel: uncalibrated RMS (full scale 1.0); not an anomaly signal"},
        },
        "required": list(EVENT_KEYS),
        "additionalProperties": False,
        "allOf": [
            {"if": {"properties": {"status": {"const": "SCORE_ONLY"}}},
             "then": {"properties": {"not_scoreable_reasons": {"maxItems": 0}, "anomaly_score": {"type": "number"},
                                     "feature_values": feature_object, "feature_z_scores": feature_object}}},
            {"if": {"properties": {"status": {"const": "NOT_SCOREABLE"}}},
             "then": {"properties": {"not_scoreable_reasons": {"minItems": 1}, "anomaly_score": {"type": "null"},
                                     "feature_z_scores": {"type": "null"}}}},
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://prism.local/schemas/{CONTRACT_VERSION}/recording-output.json",
        "title": "PRISM V2 recording analysis output (SCORE_ONLY; no threshold)",
        "type": "object",
        "properties": {
            "contract_version": {"const": CONTRACT_VERSION},
            "baseline_id": {"type": "string"},
            "detector_model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "recording_status": {"enum": list(RECORDING_STATUSES)},
            "error": {"oneOf": [{"enum": list(INPUT_ERRORS)}, {"type": "null"}]},
            "input": {"type": "object", "properties": {
                "sample_rate": {"type": "integer"}, "n_samples": {"type": "integer", "minimum": 0},
                "duration_s": {"type": ["number", "null"]}, "input_domain": {"enum": list(INPUT_DOMAINS)},
                "recording_id": {"type": ["string", "null"]},
                "recorded_at": {"type": ["string", "null"], "description": "caller-supplied ISO 8601 start time"}},
                "required": list(INPUT_KEYS), "additionalProperties": False},
            "baseline_domain_validated": {"type": "boolean"},
            "feature_order": {"const": list(V2_FEATURES)},
            "n_events": {"type": "integer", "minimum": 0},
            "n_scored": {"type": "integer", "minimum": 0},
            "events": {"type": "array", "items": event},
            "interpretation": {"const": INTERPRETATION},
        },
        "required": list(OUTPUT_KEYS),
        "additionalProperties": False,
        "allOf": [
            {"if": {"properties": {"recording_status": {"const": "INPUT_ERROR"}}},
             "then": {"properties": {"error": {"enum": list(INPUT_ERRORS)}, "events": {"maxItems": 0}}}},
            {"if": {"properties": {"recording_status": {"const": "NO_INHALATION_DETECTED"}}},
             "then": {"properties": {"error": {"type": "null"}, "events": {"maxItems": 0}}}},
            {"if": {"properties": {"recording_status": {"const": "EVENTS_DETECTED"}}},
             "then": {"properties": {"error": {"type": "null"}, "events": {"minItems": 1}}}},
        ],
    }


FEATURE_DEFINITIONS = {
    "spectral_centroid_mean": "mean over frames of the spectral centroid of |STFT| divided by 4000 Hz",
    "spectral_flatness_mean": "mean over frames of spectral flatness of |STFT|^2 (geometric / arithmetic mean, power floor 1e-10)",
    "spectral_centroid_std": "population standard deviation (ddof 0) over frames of the normalized spectral centroid",
    "spectral_rolloff_std": "population standard deviation (ddof 0) over frames of the 85% spectral rolloff divided by 4000 Hz",
}


def feature_schema() -> dict:
    """Machine-readable V2 feature ordering and extraction parameters."""
    return {
        "contract_version": CONTRACT_VERSION,
        "anomaly_features_in_order": [{"index": i, "name": f, "definition": FEATURE_DEFINITIONS[f], "unit": "dimensionless"}
                                      for i, f in enumerate(V2_FEATURES)],
        "level_channel": {"name": LEVEL_CHANNEL, "unit": "RMS amplitude, full scale 1.0 (uncalibrated)",
                          "definition": "mean of an RMS envelope: 256-sample frames starting every 64 samples from the "
                                        "event start, partial final frames included, no padding",
                          "in_anomaly_score": False},
        "event_metadata": ["event_id", "start_time", "end_time", "duration_s", "detector_confidence",
                           "detector_max_confidence", "window_count"],
        "segment": "waveform[floor(start_time * 8000) : ceil(end_time * 8000)] of the float32 input, no padding beyond the STFT's own",
        "frames": {"sample_rate_hz": SAMPLE_RATE, "n_fft": config.LIBROSA_N_FFT, "hop_length": config.LIBROSA_HOP_LENGTH,
                   "window": "hann (periodic, librosa default)", "center": True, "pad_mode": "constant (zeros), n_fft // 2 each side",
                   "spectrum": "magnitude |STFT| (librosa.stft); flatness uses power |STFT|^2",
                   "frequencies_hz": "k * 8000 / 256 for k = 0..128",
                   "rolloff_percent": 0.85, "flatness_amin": 1e-10,
                   "frame_count": "1 + floor(len(segment) / 64) (frames are aligned to the MFCC frame count)"},
        "aggregation": "mean and population std (ddof 0) over all frames, computed in float32, reported as float64",
        "reference_implementation": ["src/librosa_extractor.py::extract_features_from_audio (columns 120-122)",
                                     "src/post_event.py::analyze_inhalation"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the PRISM V2 inference contract to a WAV file (SCORE_ONLY)")
    parser.add_argument("wav")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-domain", default="unknown", choices=INPUT_DOMAINS)
    args = parser.parse_args()
    waveform_, rate = read_wav(args.wav)
    result = analyze_recording(waveform_, rate, detector=OnnxEventClassifier(model_path=args.model),
                               baseline=load_baseline(args.baseline), input_domain=args.input_domain,
                               recording_id=Path(args.wav).name)
    print(json.dumps(result, indent=2, allow_nan=False))
