"""PRISM V3 personal reference (Stage 10): a slowly adapting personal location baseline, kept separate from the
frozen Stage 9 population reference.  See PRISM_RESEARCH_LOG.md Entry 15 and results/personal_reference/.

Two independent channels per recording:
    population  the Stage 9 prism_assessment.assess_recording output, passed through unchanged and never adapted
    personal    the deviation of each scoreable event from the personal baseline B of one user AND one device

Coordinates: z = (x - m0) / s0, with m0, s0 the frozen Stage 9 V2 median and 1.4826*MAD.  B is a 4-vector in
these coordinates, initialised at B_0 = 0 (the population median).  s0 is never adapted; mean_rms is not used.

Adaptation happens per sitting, only after every event of the sitting has been assessed:
    sitting     consecutive recordings of one user/device at most ``sitting_gap_minutes`` apart (Stage 2 rule)
    qualifying  >= ``min_events_per_sitting`` scoreable events, summarised by the per-feature median x_k
    window      the last ``consensus_window`` qualifying sittings (cleared by a gap > ``max_gap_days``)
    gate g_kj   1 iff the window is full, >= ``consensus_required`` of its sittings lie on the same side of B_kj
                and the current sitting lies on that side too; else 0
    weight      w_kj = g_kj * min(1, c * SE_kj / |x_kj - B_kj|),  SE_kj = sqrt(pi/2) * event_sd_j / sqrt(n_k)
    update      B_k+1 = Pi_rho(B_k + eta * w_k * (x_k - B_k)),  Pi_rho = radial projection onto rms(B) <= rho

One sitting therefore moves B by at most rms(dB) <= eta * c * rms(SE_k), whatever its values, and only when the
consensus of earlier sittings already points the same way.  Only an explicit, audited re-enrollment can place B
beyond the trust region.

Meaning: personal_deviation = sqrt(mean_j (z_j - B_j)^2) measures deviation from the personal reference
(consistency with this user's own earlier qualifying sittings on this device).  It is continuous: no personal
cut-off has been calibrated, so no categorical personal statement is made.  The personal baseline is not a
healthy, normal or correct-technique baseline; nothing here is a technique-quality or clinical classification.
Status: implemented and mechanically verified on proxy/synthetic data only; not validated as personalization
(the PRISM corpus has no user or device identifiers).
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import config
from feature_analysis import SESSION_GAP_MIN
from post_event import DEFAULT_MODEL_PATH, OnnxEventClassifier
from prism_assessment import (
    ASSESSMENT_CONTRACT_VERSION,
    DEFAULT_REFERENCE_PATH,
    AssessmentReference,
    assess_recording,
    load_reference,
)
from prism_inference import INPUT_DOMAINS, V2_FEATURES, read_wav


PERSONAL_CONTRACT_VERSION = "prism-personal-reference-v3.0-draft"
STATUS = "IMPLEMENTED_MECHANICALLY_VERIFIED_NOT_VALIDATED"
DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "personal_reference"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "personal_reference_config_v3.json"
SENSITIVITY_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "personal_reference_config_v3_8of10.json"
STAGE9_DIR = DEFAULT_REFERENCE_PATH.parent          # results/final_assessment: never written by this module

MEDIAN_SE_FACTOR = math.sqrt(math.pi / 2)           # asymptotic SE of a median: sqrt(pi/2) * sd / sqrt(n)
GENESIS_SHA256 = "0" * 64
BOUNDARY_TOLERANCE = 1e-12

STATUSES = ("WARMUP", "ESTABLISHED")
EVENT_ASSESSMENTS = ("POPULATION_ONLY", "PERSONAL_DEVIATION", "NOT_ASSESSED")
DECISIONS = ("NOT_QUALIFYING", "NO_UPDATE", "UPDATED")
AUDIT_KINDS = ("initialized", "sitting_closed", "reset", "reenrollment")

INTERPRETATION = (
    "personal_deviation is the root-mean-square over the four V2 spectral features of (z - B): z are the event's "
    "robust z-scores in the frozen Stage 9 population-reference coordinates and B is the personal baseline of this "
    "user and device, initialised at the population median (B = 0) and moved only by the sitting-level, "
    "consensus-gated, bounded update after a qualifying sitting has been fully assessed. It measures deviation from "
    "the personal reference (consistency with this user's own earlier qualifying sittings on this device); larger "
    "means less consistent. It is continuous: no personal cut-off has been calibrated, so no categorical personal "
    "statement is made. During WARMUP (fewer than s_min qualifying sittings) no personal deviation is reported and "
    "only the population channel applies. The personal baseline is not a healthy, normal or correct-technique "
    "baseline; these statements are not technique-quality, normal/abnormal or clinical classifications. Evaluated "
    "on proxy/synthetic data only; not validated as personalization."
)

OUTPUT_KEYS = ("contract_version", "population", "personal")
PERSONAL_KEYS = ("contract_version", "config_id", "config_sha256", "user_id", "device_id", "population_reference_id",
                 "recording_id", "recorded_at", "sitting_index", "closed_sitting", "personal_status", "state_id_used",
                 "baseline_version_used", "baseline_used", "n_qualifying_sittings", "s_min", "consensus_window_size",
                 "trust_radius", "trust_region_distance", "at_trust_region_boundary", "reference_stale", "n_events",
                 "n_personal_assessed", "events", "interpretation")
EVENT_KEYS = ("event_id", "personal_assessment", "personal_deviation", "personal_feature_deviations")


class PersonalReferenceError(ValueError):
    """Input, configuration or state does not satisfy the V3 personal-reference contract."""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PersonalReferenceError(message)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rms(values) -> float:
    """sqrt(mean v^2), Stage 9's formula; rescaled only when squaring would overflow (extreme inputs)."""
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore"):
        direct = float(np.sqrt(np.mean(np.square(v))))
    if math.isfinite(direct):
        return direct
    scale = float(np.max(np.abs(v)))
    return scale * float(np.sqrt(np.mean(np.square(v / scale))))


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64)]


def _ints(values) -> list[int]:
    return [int(v) for v in np.asarray(values)]


def _sha256_json(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise PersonalReferenceError(f"timestamps must be ISO 8601 strings, got {value!r}") from error


def _seconds(earlier: str, later: str) -> float:
    try:
        return (_parse_time(later) - _parse_time(earlier)).total_seconds()
    except TypeError as error:
        raise PersonalReferenceError("timestamps must consistently include or omit a time zone") from error


def _days(earlier: str, later: str) -> float:
    return _seconds(earlier, later) / 86400.0


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PersonalReferenceConfig:
    """Provisional V3 parameters (implementation parameters to be evaluated, not validated values)."""

    config_id: str = "prism-personal-v3-provisional-9of10"
    features: tuple[str, ...] = V2_FEATURES
    sitting_gap_minutes: float = SESSION_GAP_MIN       # Stage 2 sitting rule (Entry 4)
    min_events_per_sitting: int = 10                   # Stage 5: k=10 warm-up was the smallest k with a gain
    consensus_window: int = 10                         # K qualifying sittings
    consensus_required: int = 9                        # k of K on the same side of B
    eta: float = 0.3
    clip_c: float = 2.0
    event_sd: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)  # event-level SD in population-scale units (s0 = 1)
    trust_radius: float = 0.9                          # rho, on rms(B); largest Stage 2 sitting offset observed
    max_gap_days: float = 14.0                         # G_max between qualifying sittings
    s_min: int = 10                                    # qualifying sittings before the personal channel reports

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "event_sd", tuple(float(v) for v in self.event_sd))
        _check(isinstance(self.config_id, str) and bool(self.config_id), "config_id must be a non-empty string")
        _check(self.features == V2_FEATURES, f"features must be the V2 representation {V2_FEATURES}")
        for name in ("min_events_per_sitting", "consensus_window", "consensus_required", "s_min"):
            value = getattr(self, name)
            _check(isinstance(value, int) and not isinstance(value, bool) and value >= 1, f"{name} must be an int >= 1")
        _check(self.consensus_window >= 2, "consensus_window must be >= 2")
        _check(2 * self.consensus_required > self.consensus_window, "consensus_required must be a strict majority")
        _check(self.consensus_required <= self.consensus_window, "consensus_required must be <= consensus_window")
        for name in ("sitting_gap_minutes", "clip_c", "trust_radius", "max_gap_days"):
            value = getattr(self, name)
            _check(_finite_number(value) and value > 0, f"{name} must be finite and > 0")
        _check(_finite_number(self.eta) and 0 < self.eta <= 1, "eta must satisfy 0 < eta <= 1")
        _check(len(self.event_sd) == len(V2_FEATURES) and all(math.isfinite(v) and v > 0 for v in self.event_sd),
               "event_sd must hold one finite value > 0 per feature")

    def to_dict(self) -> dict:
        return {"contract_version": PERSONAL_CONTRACT_VERSION, "config_id": self.config_id,
                "features": list(self.features), "sitting_gap_minutes": self.sitting_gap_minutes,
                "min_events_per_sitting": self.min_events_per_sitting, "consensus_window": self.consensus_window,
                "consensus_required": self.consensus_required, "eta": self.eta, "clip_c": self.clip_c,
                "event_sd": list(self.event_sd), "trust_radius": self.trust_radius, "max_gap_days": self.max_gap_days,
                "s_min": self.s_min}

    @classmethod
    def from_dict(cls, data: Mapping) -> "PersonalReferenceConfig":
        if data.get("contract_version") != PERSONAL_CONTRACT_VERSION:
            raise PersonalReferenceError(f"config was issued for {data.get('contract_version')!r}")
        expected = set(cls().to_dict())
        if set(data) != expected:
            raise PersonalReferenceError(f"config keys must be exactly {sorted(expected)}")
        return cls(**{k: v for k, v in data.items() if k != "contract_version"})

    @functools.cached_property
    def sha256(self) -> str:                      # the config is frozen, so its hash is computed once
        return _sha256_json(self.to_dict())

    def sitting_bound(self, n_events: int) -> float:
        """Largest rms change of B that one qualifying sitting of ``n_events`` events can cause."""
        return self.eta * self.clip_c * _rms(standard_errors(self, n_events))


SENSITIVITY_CONFIG = PersonalReferenceConfig(config_id="prism-personal-v3-provisional-8of10", consensus_required=8)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PersonalReferenceConfig:
    return PersonalReferenceConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_config(cfg: PersonalReferenceConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


# ── Pure update rule ────────────────────────────────────────────────────────

def standard_errors(cfg: PersonalReferenceConfig, n_events: int) -> np.ndarray:
    """SE_kj of a sitting median of ``n_events`` events, in population-scale units."""
    _check(isinstance(n_events, (int, np.integer)) and n_events >= 1, "n_events must be an int >= 1")
    return MEDIAN_SE_FACTOR * np.asarray(cfg.event_sd) / math.sqrt(int(n_events))


def bounded_consensus_update(baseline: Sequence[float], window_medians: Sequence[Sequence[float]],
                             sitting_median: Sequence[float], n_events: int, cfg: PersonalReferenceConfig,
                             trust_radius: float) -> dict:
    """One qualifying sitting's update; ``window_medians`` already ends with ``sitting_median``.

    Returns every intermediate quantity so the audit log can show (and replay can re-derive) the decision.
    """
    b = np.asarray(baseline, dtype=np.float64)
    x = np.asarray(sitting_median, dtype=np.float64)
    window = np.atleast_2d(np.asarray(window_medians, dtype=np.float64))
    innovation = x - b
    offsets = window - b
    positive, negative = (offsets > 0).sum(axis=0), (offsets < 0).sum(axis=0)
    full = len(window) == cfg.consensus_window
    direction = np.where(positive >= cfg.consensus_required, 1, np.where(negative >= cfg.consensus_required, -1, 0))
    direction = direction if full else np.zeros_like(direction)
    gate = (direction != 0) & (np.sign(innovation) == direction)
    se = standard_errors(cfg, n_events)
    limit = cfg.clip_c * se
    size = np.abs(innovation)
    clipped = size > limit                                    # size > limit > 0, so the division is safe
    weight = np.where(gate, np.where(clipped, limit / np.where(clipped, size, 1.0), 1.0), 0.0)
    step = cfg.eta * weight * innovation
    proposed = b + step
    norm = _rms(proposed)
    capped = norm > trust_radius
    updated = proposed * (trust_radius / norm) if capped else proposed
    return {"innovation": innovation, "n_positive": positive, "n_negative": negative, "window_full": full,
            "direction": direction, "gate": gate, "standard_error": se, "clipped": gate & clipped,
            "weight": weight, "step": step, "proposed": proposed, "trust_region_capped": bool(capped),
            "baseline_after": updated}


# ── Personal reference state ────────────────────────────────────────────────

class PersonalReference:
    """Personal baseline of one user on one device: assess first, adapt at sitting close, audit everything."""

    def __init__(self, cfg: PersonalReferenceConfig, *, user_id: str, device_id: str, population_reference_id: str,
                 created_at: str) -> None:
        for name, value in (("user_id", user_id), ("device_id", device_id),
                            ("population_reference_id", population_reference_id)):
            _check(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
        _parse_time(created_at)
        self.config = cfg
        self.user_id, self.device_id = user_id, device_id
        self.population_reference_id = population_reference_id
        self.created_at = created_at
        self.baseline = np.zeros(len(V2_FEATURES))
        self.trust_radius = cfg.trust_radius
        self.baseline_version = 0
        self.n_qualifying = 0                       # since creation, the last reset or the last re-enrollment
        self.n_capped_updates = 0
        self.window: list[dict] = []                # qualifying-sitting summaries, oldest first
        self.last_qualifying_end: str | None = None
        self.last_recording_at: str | None = None
        self.next_sitting_index = 0
        self.open_sitting: dict | None = None
        self.audit: list[dict] = []
        self._append_audit({"kind": "initialized", "at": created_at, "contract_version": PERSONAL_CONTRACT_VERSION,
                            "config": cfg.to_dict(), "config_sha256": cfg.sha256, "user_id": user_id,
                            "device_id": device_id, "population_reference_id": population_reference_id,
                            "baseline_after": _floats(self.baseline), "trust_radius": self.trust_radius,
                            "state_id_after": self.state_id})

    # ── derived state ──
    @property
    def status(self) -> str:
        return "ESTABLISHED" if self.n_qualifying >= self.config.s_min else "WARMUP"

    @property
    def trust_region_distance(self) -> float:
        return _rms(self.baseline)

    @property
    def at_trust_region_boundary(self) -> bool:
        return self.trust_region_distance >= self.trust_radius * (1 - BOUNDARY_TOLERANCE)

    @property
    def state_id(self) -> str:
        """Identifier of everything that determines assessments and future updates (not the open sitting)."""
        return _sha256_json(self._baseline_state())[:16]

    def _baseline_state(self) -> dict:
        return {"config_sha256": self.config.sha256, "population_reference_id": self.population_reference_id,
                "user_id": self.user_id, "device_id": self.device_id, "baseline": _floats(self.baseline),
                "trust_radius": self.trust_radius, "baseline_version": self.baseline_version,
                "n_qualifying": self.n_qualifying, "n_capped_updates": self.n_capped_updates,
                "window": self.window, "last_qualifying_end": self.last_qualifying_end}

    def reference_stale(self, at: str) -> bool:
        if self.last_qualifying_end is None:
            return False
        return _days(self.last_qualifying_end, at) > self.config.max_gap_days

    # ── assessment (never adapts) ──
    def observe_recording(self, z_rows: Sequence[Sequence[float] | None], scoreable: Sequence[bool], recorded_at: str,
                          recording_id: str | None = None) -> dict:
        """Assess every event of one recording against the baseline from before its sitting, then buffer them.

        ``z_rows`` are population-reference z vectors (None allowed for events that are not scoreable).  A gap of
        more than ``sitting_gap_minutes`` since the previous recording first closes the previous sitting.
        """
        _check(len(z_rows) == len(scoreable), "z_rows and scoreable must have the same length")
        rows = []
        for z, ok in zip(z_rows, scoreable):
            if not bool(ok):
                rows.append(None)
                continue
            vector = np.asarray(z, dtype=np.float64)
            _check(vector.shape == (len(V2_FEATURES),) and np.isfinite(vector).all(),
                   "scoreable events need a finite z vector in V2 feature order")
            rows.append(vector)
        if self.last_recording_at is not None:
            _check(_seconds(self.last_recording_at, recorded_at) >= 0, "recordings must arrive in chronological order")
        closed = None
        if self.open_sitting is not None and \
                _seconds(self.open_sitting["last_recording_at"], recorded_at) > 60 * self.config.sitting_gap_minutes:
            closed = self.close_sitting()
        if self.open_sitting is None:
            self.open_sitting = {"sitting_index": self.next_sitting_index, "started_at": recorded_at,
                                 "last_recording_at": recorded_at, "n_recordings": 0, "n_events_total": 0,
                                 "state_id": self.state_id, "events": []}
            self.next_sitting_index += 1
        sitting = self.open_sitting
        if sitting["state_id"] != self.state_id:    # B can only change when no sitting is open
            raise PersonalReferenceError("baseline changed while a sitting was open")

        snapshot = self.baseline.copy()             # the baseline from before the current sitting
        status = self.status
        assessed = [z for z in rows if z is not None]
        if status == "ESTABLISHED" and assessed:
            offsets = np.vstack(assessed) - snapshot
            with np.errstate(over="ignore"):
                deviations = np.sqrt(np.mean(np.square(offsets), axis=1))   # row-wise _rms
        events, position = [], 0
        for event_id, z in enumerate(rows):
            if z is None:
                events.append({"event_id": event_id, "personal_assessment": "NOT_ASSESSED",
                               "personal_deviation": None, "personal_feature_deviations": None})
            elif status == "WARMUP":
                events.append({"event_id": event_id, "personal_assessment": "POPULATION_ONLY",
                               "personal_deviation": None, "personal_feature_deviations": None})
            else:
                d, deviation = offsets[position], float(deviations[position])
                position += 1
                events.append({"event_id": event_id, "personal_assessment": "PERSONAL_DEVIATION",
                               "personal_deviation": deviation if math.isfinite(deviation) else _rms(d),
                               "personal_feature_deviations": dict(zip(V2_FEATURES, d.tolist()))})
        output = {
            "contract_version": PERSONAL_CONTRACT_VERSION, "config_id": self.config.config_id,
            "config_sha256": self.config.sha256, "user_id": self.user_id, "device_id": self.device_id,
            "population_reference_id": self.population_reference_id, "recording_id": recording_id,
            "recorded_at": recorded_at, "sitting_index": sitting["sitting_index"], "closed_sitting": closed,
            "personal_status": status, "state_id_used": sitting["state_id"],
            "baseline_version_used": self.baseline_version, "baseline_used": dict(zip(V2_FEATURES, _floats(snapshot))),
            "n_qualifying_sittings": self.n_qualifying, "s_min": self.config.s_min,
            "consensus_window_size": len(self.window), "trust_radius": self.trust_radius,
            "trust_region_distance": self.trust_region_distance,
            "at_trust_region_boundary": self.at_trust_region_boundary,
            "reference_stale": self.reference_stale(recorded_at), "n_events": len(events),
            "n_personal_assessed": sum(e["personal_assessment"] == "PERSONAL_DEVIATION" for e in events),
            "events": events, "interpretation": INTERPRETATION,
        }
        # Only now, after every event of the recording has been assessed, does the sitting buffer grow.
        sitting["events"].extend(_floats(z) for z in rows if z is not None)
        sitting["n_events_total"] += len(rows)
        sitting["n_recordings"] += 1
        sitting["last_recording_at"] = recorded_at
        self.last_recording_at = recorded_at
        return output

    def observe_population_output(self, population: Mapping, recorded_at: str, recording_id: str | None = None) -> dict:
        """Personal channel for one Stage 9 output (read only; the population output is not modified)."""
        _check(population.get("contract_version") == ASSESSMENT_CONTRACT_VERSION, "not a Stage 9 assessment output")
        _check(population.get("reference_id") == self.population_reference_id,
               "the population output was produced with a different population reference than this personal state")
        _check(list(population.get("feature_order", [])) == list(V2_FEATURES), "feature order must be V2")
        z_rows, scoreable = [], []
        for event in population["events"]:
            ok = event["scoreability"] == "SCOREABLE"
            z_rows.append([event["feature_deviations"][f] for f in V2_FEATURES] if ok else None)
            scoreable.append(ok)
        return self.observe_recording(z_rows, scoreable, recorded_at, recording_id)

    # ── adaptation (only at sitting close) ──
    def close_sitting(self) -> dict | None:
        """Close the open sitting (all its events are already assessed) and apply the sitting-level update."""
        if self.open_sitting is None:
            return None
        sitting, self.open_sitting = self.open_sitting, None
        events = np.asarray(sitting["events"], dtype=np.float64).reshape(-1, len(V2_FEATURES))
        qualifying = len(events) >= self.config.min_events_per_sitting
        record = {"sitting_index": sitting["sitting_index"], "started_at": sitting["started_at"],
                  "ended_at": sitting["last_recording_at"], "n_recordings": sitting["n_recordings"],
                  "n_events_total": sitting["n_events_total"], "n_qualifying_events": int(len(events)),
                  "sitting_median": _floats(np.median(events, axis=0)) if qualifying else None}
        return self._apply_closed_sitting(record)

    def _apply_closed_sitting(self, record: Mapping) -> dict:
        before_id, before = self.state_id, _floats(self.baseline)
        entry = {"kind": "sitting_closed", "at": record["ended_at"], **record, "state_id_before": before_id,
                 "baseline_before": before}
        self.next_sitting_index = max(self.next_sitting_index, int(record["sitting_index"]) + 1)
        if record["sitting_median"] is None:
            entry.update({"decision": "NOT_QUALIFYING", "baseline_after": before, "state_id_after": before_id})
            return self._append_audit(entry)
        gap_days = None if self.last_qualifying_end is None else _days(self.last_qualifying_end, record["started_at"])
        gap_reset = gap_days is not None and gap_days > self.config.max_gap_days
        if gap_reset:
            self.window = []                         # stale consensus evidence never opens the gate
        self.window = (self.window + [{"sitting_index": int(record["sitting_index"]), "started_at": record["started_at"],
                                       "ended_at": record["ended_at"], "n_events": int(record["n_qualifying_events"]),
                                       "median": list(record["sitting_median"])}])[-self.config.consensus_window:]
        self.n_qualifying += 1
        self.last_qualifying_end = record["ended_at"]
        update = bounded_consensus_update(self.baseline, [s["median"] for s in self.window], record["sitting_median"],
                                          int(record["n_qualifying_events"]), self.config, self.trust_radius)
        changed = not np.array_equal(update["baseline_after"], self.baseline)
        if changed:
            self.baseline = np.asarray(update["baseline_after"], dtype=np.float64)
            self.baseline_version += 1
        self.n_capped_updates += int(update["trust_region_capped"])
        entry.update({
            "decision": "UPDATED" if changed else "NO_UPDATE",
            "days_since_previous_qualifying": gap_days, "gap_reset": bool(gap_reset),
            "consensus_window_size": len(self.window), "n_positive": _ints(update["n_positive"]),
            "n_negative": _ints(update["n_negative"]), "window_full": bool(update["window_full"]),
            "direction": _ints(update["direction"]), "gate": [bool(g) for g in update["gate"]],
            "innovation": _floats(update["innovation"]), "standard_error": _floats(update["standard_error"]),
            "clipped": [bool(c) for c in update["clipped"]], "weight": _floats(update["weight"]),
            "step": _floats(update["step"]), "proposed": _floats(update["proposed"]),
            "trust_region_capped": update["trust_region_capped"], "trust_radius": self.trust_radius,
            "baseline_after": _floats(self.baseline), "trust_region_distance_after": self.trust_region_distance,
            "n_qualifying_after": self.n_qualifying, "status_after": self.status, "state_id_after": self.state_id})
        return self._append_audit(entry)

    def reset(self, reason: str, at: str) -> dict:
        """Return to the population reference (B = 0) and to WARMUP; history stays in the audit log."""
        _check(self.open_sitting is None, "close the open sitting before a reset")
        _check(isinstance(reason, str) and bool(reason), "a reset needs a reason")
        return self._apply_reset(reason, at)

    def _apply_reset(self, reason: str, at: str) -> dict:
        _parse_time(at)
        before_id, before = self.state_id, _floats(self.baseline)
        self.baseline = np.zeros(len(V2_FEATURES))
        self.trust_radius = self.config.trust_radius
        self._restart_evidence()
        return self._append_audit({"kind": "reset", "at": at, "reason": reason, "state_id_before": before_id,
                                   "baseline_before": before, "baseline_after": _floats(self.baseline),
                                   "trust_radius": self.trust_radius, "state_id_after": self.state_id})

    def reenroll(self, enrollment_z: Sequence[Sequence[float]], reason: str, at: str,
                 trust_radius: float | None = None) -> dict:
        """Explicit re-enrollment: B = per-feature median of supervised enrollment events.

        This is the only way to place B beyond the configured trust region; doing so requires an explicit
        ``trust_radius`` at least as large as rms(B).  The personal channel returns to WARMUP afterwards.
        """
        _check(self.open_sitting is None, "close the open sitting before a re-enrollment")
        _check(isinstance(reason, str) and bool(reason), "a re-enrollment needs a reason")
        z = np.asarray(enrollment_z, dtype=np.float64)
        _check(z.ndim == 2 and z.shape[1] == len(V2_FEATURES) and np.isfinite(z).all(),
               "enrollment events must be finite z vectors in V2 feature order")
        _check(len(z) >= self.config.min_events_per_sitting,
               f"re-enrollment needs at least {self.config.min_events_per_sitting} events")
        median = np.median(z, axis=0)
        radius = self.config.trust_radius if trust_radius is None else trust_radius
        _check(_finite_number(radius) and radius > 0, "trust_radius must be finite and > 0")
        if _rms(median) > radius:
            raise PersonalReferenceError(
                f"re-enrolled baseline lies outside the trust region (rms {_rms(median):.3f} > {radius}); "
                "pass an explicit trust_radius >= its rms to re-enroll beyond it")
        return self._apply_reenrollment(_floats(median), int(len(z)), float(radius), reason, at)

    def _apply_reenrollment(self, median: Sequence[float], n_events: int, radius: float, reason: str, at: str) -> dict:
        _parse_time(at)
        before_id, before = self.state_id, _floats(self.baseline)
        self.baseline = np.asarray(median, dtype=np.float64)
        self.trust_radius = radius
        self._restart_evidence()
        return self._append_audit({"kind": "reenrollment", "at": at, "reason": reason, "n_events": n_events,
                                   "enrollment_median": list(median), "state_id_before": before_id,
                                   "baseline_before": before, "baseline_after": _floats(self.baseline),
                                   "trust_radius": self.trust_radius, "state_id_after": self.state_id})

    def _restart_evidence(self) -> None:
        self.window = []
        self.n_qualifying = 0
        self.last_qualifying_end = None
        self.baseline_version += 1

    # ── audit log ──
    def _append_audit(self, entry: dict) -> dict:
        entry = {"sequence": len(self.audit), **entry,
                 "prev_entry_sha256": self.audit[-1]["entry_sha256"] if self.audit else GENESIS_SHA256}
        entry["entry_sha256"] = _sha256_json(entry)
        self.audit.append(entry)
        return copy.deepcopy(entry)             # callers can never alter the log through the returned entry

    @classmethod
    def replay(cls, audit: Sequence[Mapping]) -> "PersonalReference":
        """Rebuild the baseline state from the audit log alone, re-deriving and checking every entry exactly."""
        verify_audit(audit)
        first = audit[0]
        _check(first["kind"] == "initialized", "the audit log must start with an 'initialized' entry")
        replayed = cls(PersonalReferenceConfig.from_dict(first["config"]), user_id=first["user_id"],
                       device_id=first["device_id"], population_reference_id=first["population_reference_id"],
                       created_at=first["at"])
        if replayed.audit[0] != dict(first):
            raise PersonalReferenceError("audit entry 0 is not reproduced by replay")
        for logged in audit[1:]:
            kind = logged["kind"]
            if kind == "sitting_closed":
                replayed._apply_closed_sitting({k: logged[k] for k in ("sitting_index", "started_at", "ended_at",
                                                                       "n_recordings", "n_events_total",
                                                                       "n_qualifying_events", "sitting_median")})
            elif kind == "reset":
                replayed._apply_reset(logged["reason"], logged["at"])
            elif kind == "reenrollment":
                replayed._apply_reenrollment(logged["enrollment_median"], logged["n_events"], logged["trust_radius"],
                                             logged["reason"], logged["at"])
            else:
                raise PersonalReferenceError(f"unexpected audit entry kind {kind!r}")
            if replayed.audit[-1] != dict(logged):
                raise PersonalReferenceError(f"audit entry {logged['sequence']} is not reproduced by replay")
        return replayed

    # ── persistence ──
    def to_dict(self) -> dict:
        return {"contract_version": PERSONAL_CONTRACT_VERSION, "config": self.config.to_dict(),
                "user_id": self.user_id, "device_id": self.device_id,
                "population_reference_id": self.population_reference_id, "created_at": self.created_at,
                "state_id": self.state_id, "status": self.status, "baseline": _floats(self.baseline),
                "trust_radius": self.trust_radius, "trust_region_distance": self.trust_region_distance,
                "baseline_version": self.baseline_version, "n_qualifying": self.n_qualifying,
                "n_capped_updates": self.n_capped_updates, "window": self.window,
                "last_qualifying_end": self.last_qualifying_end, "last_recording_at": self.last_recording_at,
                "next_sitting_index": self.next_sitting_index, "open_sitting": self.open_sitting,
                "audit": self.audit}

    @classmethod
    def from_dict(cls, data: Mapping, expected_config: PersonalReferenceConfig | None = None) -> "PersonalReference":
        """Load a saved state; the audit log is verified and replayed, and must reproduce the stored baseline."""
        _check(data.get("contract_version") == PERSONAL_CONTRACT_VERSION,
               f"state was issued for {data.get('contract_version')!r}")
        state = cls.replay(data["audit"])
        if expected_config is not None and state.config != expected_config:
            raise PersonalReferenceError("saved state was created with a different configuration")
        replayed = state.to_dict()
        for key in ("user_id", "device_id", "population_reference_id", "created_at", "baseline", "trust_radius",
                    "baseline_version", "n_qualifying", "n_capped_updates", "window", "last_qualifying_end", "state_id"):
            if replayed[key] != data[key]:
                raise PersonalReferenceError(f"saved state field {key!r} disagrees with its audit log")
        state.last_recording_at = data["last_recording_at"]
        state.next_sitting_index = int(data["next_sitting_index"])
        state.open_sitting = data["open_sitting"]
        if state.open_sitting is not None:
            _check(state.open_sitting["state_id"] == state.state_id, "open sitting was started from another state")
        return state


def verify_audit(audit: Sequence[Mapping]) -> None:
    """Raise unless the audit log is a complete, untampered hash chain."""
    _check(len(audit) > 0, "audit log is empty")
    previous = GENESIS_SHA256
    for position, entry in enumerate(audit):
        _check(entry.get("sequence") == position, f"audit entry {position} is out of sequence")
        _check(entry.get("kind") in AUDIT_KINDS, f"audit entry {position} has an unknown kind")
        _check(entry.get("prev_entry_sha256") == previous, f"audit entry {position} does not chain to its predecessor")
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        _check(_sha256_json(body) == entry.get("entry_sha256"), f"audit entry {position} was modified")
        previous = entry["entry_sha256"]


# ── Storage: one file per user AND device ───────────────────────────────────

class PersonalReferenceStore:
    """Directory of personal states keyed by (user_id, device_id); refuses the Stage 9 results directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        stage9 = STAGE9_DIR.resolve()
        if self.root == stage9 or stage9 in self.root.parents:
            raise PersonalReferenceError("personal states must not be stored inside the Stage 9 results directory")

    def path(self, user_id: str, device_id: str) -> Path:
        key = hashlib.sha256(json.dumps([user_id, device_id]).encode("utf-8")).hexdigest()[:12]
        slug = "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", part)[:40] for part in (user_id, device_id))
        return self.root / f"{slug}__{key}.json"

    def load(self, user_id: str, device_id: str, cfg: PersonalReferenceConfig | None = None) -> PersonalReference | None:
        path = self.path(user_id, device_id)
        if not path.exists():
            return None
        state = PersonalReference.from_dict(json.loads(path.read_text(encoding="utf-8")), expected_config=cfg)
        _check((state.user_id, state.device_id) == (user_id, device_id), "state file belongs to another user/device")
        return state

    def load_or_create(self, user_id: str, device_id: str, cfg: PersonalReferenceConfig, population_reference_id: str,
                       created_at: str) -> PersonalReference:
        state = self.load(user_id, device_id, cfg)
        if state is None:
            return PersonalReference(cfg, user_id=user_id, device_id=device_id,
                                     population_reference_id=population_reference_id, created_at=created_at)
        _check(state.population_reference_id == population_reference_id,
               "stored personal state is anchored to a different population reference")
        return state

    def save(self, state: PersonalReference) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(state.user_id, state.device_id)
        text = json.dumps(state.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        handle, temporary = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
        return path


# ── Both channels for one recording ─────────────────────────────────────────

def assess_recording_with_personal(audio, sample_rate: int, reference: AssessmentReference, personal: PersonalReference,
                                   *, recorded_at: str, detector: OnnxEventClassifier | None = None,
                                   input_domain: str = "unknown", recording_id: str | None = None) -> dict:
    """Population channel (Stage 9, unchanged) + personal channel (V3) for one recording."""
    _check(isinstance(recorded_at, str) and bool(recorded_at), "the personal channel needs recorded_at")
    population = assess_recording(audio, sample_rate, reference, detector=detector, input_domain=input_domain,
                                  recording_id=recording_id, recorded_at=recorded_at)
    personal_output = personal.observe_population_output(population, recorded_at, recording_id)
    output = {"contract_version": PERSONAL_CONTRACT_VERSION, "population": population, "personal": personal_output}
    validate_personal_output(personal_output)
    return output


def validate_personal_output(output: Mapping) -> None:
    """Raise PersonalReferenceError unless ``output`` satisfies the personal-channel contract."""
    _check(tuple(output) == PERSONAL_KEYS, f"personal output keys must be exactly {PERSONAL_KEYS}")
    _check(output["contract_version"] == PERSONAL_CONTRACT_VERSION, "wrong personal contract_version")
    _check(output["personal_status"] in STATUSES, "unknown personal_status")
    _check(output["interpretation"] == INTERPRETATION, "interpretation must be the fixed V3 text")
    baseline = np.array([output["baseline_used"][f] for f in V2_FEATURES])
    _check(math.isclose(output["trust_region_distance"], _rms(baseline), rel_tol=1e-12, abs_tol=1e-15),
           "trust_region_distance must equal rms(baseline_used)")
    _check(output["trust_region_distance"] <= output["trust_radius"] * (1 + BOUNDARY_TOLERANCE),
           "baseline lies outside the trust region")
    _check(output["n_events"] == len(output["events"]), "n_events mismatch")
    _check(output["n_personal_assessed"] == sum(e["personal_assessment"] == "PERSONAL_DEVIATION"
                                                for e in output["events"]), "n_personal_assessed mismatch")
    for position, event in enumerate(output["events"]):
        _check(tuple(event) == EVENT_KEYS, f"personal event keys must be exactly {EVENT_KEYS}")
        _check(event["event_id"] == position, "event_id must be 0, 1, ... in order")
        _check(event["personal_assessment"] in EVENT_ASSESSMENTS, "unknown personal_assessment")
        if event["personal_assessment"] == "PERSONAL_DEVIATION":
            _check(output["personal_status"] == "ESTABLISHED", "personal deviations are reported only when ESTABLISHED")
            d = np.array([event["personal_feature_deviations"][f] for f in V2_FEATURES])
            _check(math.isclose(event["personal_deviation"], _rms(d), rel_tol=1e-12, abs_tol=1e-15),
                   "personal_deviation != rms(personal_feature_deviations)")
        else:
            _check(event["personal_deviation"] is None and event["personal_feature_deviations"] is None,
                   "only PERSONAL_DEVIATION events carry a personal deviation")
            if event["personal_assessment"] == "POPULATION_ONLY":
                _check(output["personal_status"] == "WARMUP", "POPULATION_ONLY is the WARMUP assessment")
    json.dumps(output, allow_nan=False)


def personal_contract(cfg: PersonalReferenceConfig, sensitivity: PersonalReferenceConfig | None = None) -> dict:
    """Machine-readable statement of the V3 personal-channel contract."""
    return {
        "contract_version": PERSONAL_CONTRACT_VERSION, "status": STATUS,
        "status_meaning": "implemented and mechanically verified on proxy/synthetic data; not validated as "
                          "personalization; no personal cut-off calibrated (continuous output only)",
        "population_channel": f"Stage 9 {ASSESSMENT_CONTRACT_VERSION} output, passed through unchanged and never adapted",
        "coordinates": "z = (x - m0) / s0 with the frozen Stage 9 V2 median m0 and 1.4826*MAD scale s0; s0 is never "
                       "adapted; mean_rms is not used",
        "personal_baseline": "B in R^4 per (user_id, device_id), initialised at B_0 = 0 (the population median)",
        "personal_deviation": "a_u = sqrt(mean_j (z_j - B_j)^2), B taken from before the event's sitting",
        "sitting": "consecutive recordings at most sitting_gap_minutes apart; qualifying if >= min_events_per_sitting "
                   "scoreable events; summarised by the per-feature median x_k; one sitting = one unit of evidence",
        "consensus_gate": "g_kj = 1 iff the window holds consensus_window qualifying sittings (none before a gap > "
                          "max_gap_days), >= consensus_required of their medians lie strictly on one side of B_kj, "
                          "and the current sitting lies strictly on that side; else 0",
        "update": "w_kj = g_kj * min(1, clip_c * SE_kj / |x_kj - B_kj|) (w = g when x_kj = B_kj), SE_kj = sqrt(pi/2) * "
                  "event_sd_j / sqrt(n_k); B_k+1 = Pi_rho(B_k + eta * w_k * (x_k - B_k)), Pi_rho = radial projection "
                  "onto rms(B) <= rho",
        "bounds": {"one_sitting": "rms(B_k+1 - B_k) <= eta * clip_c * rms(SE_k) <= sitting_bound(min_events_per_sitting)",
                   "one_event": "moves its sitting median by at most one order-statistic gap; cannot change any "
                                "assessment of its own sitting",
                   "trust_region": "rms(B) <= trust_radius always; only reenroll() with an explicit trust_radius can "
                                   "exceed the configured radius",
                   "sitting_bound_at_min_events": cfg.sitting_bound(cfg.min_events_per_sitting)},
        "ordering": ["snapshot B before the sitting", "assess every event of every recording against the snapshot",
                     "emit the assessment with state_id_used", "then add the events to the sitting buffer",
                     "at sitting close decide and apply the update", "log the decision in the hash-chained audit"],
        "warmup": "personal_status WARMUP while fewer than s_min qualifying sittings exist since creation, reset or "
                  "re-enrollment: events get POPULATION_ONLY and no personal deviation",
        "long_gap": "a gap > max_gap_days between qualifying sittings clears the consensus window; reference_stale "
                    "reports a current gap > max_gap_days",
        "statuses": list(STATUSES), "event_assessments": list(EVENT_ASSESSMENTS), "decisions": list(DECISIONS),
        "output_keys": list(OUTPUT_KEYS), "personal_keys": list(PERSONAL_KEYS), "event_keys": list(EVENT_KEYS),
        "interpretation": INTERPRETATION,
        "config": cfg.to_dict(), "config_sha256": cfg.sha256,
        "sensitivity_config": sensitivity.to_dict() if sensitivity else None,
        "known_limitations": [
            "no user or device identifiers exist in the PRISM corpus: personalization is not validated",
            "a sustained acoustic change is absorbed whatever its cause (behaviour, noise, microphone, device); the "
            "mechanism cannot tell these apart",
            "parameters (K, k, eta, c, event_sd, rho, G_max, s_min, n_min) are provisional implementation values",
            "sitting medians assume independent events within a sitting; real within-sitting correlation makes SE "
            "optimistic (tighter clipping, slower adaptation)",
            "real inhaler use may give far fewer than min_events_per_sitting inhalations per sitting",
            "on the PRISM corpus only 8 of the 23 inferred sittings (18 contain a scoreable event) have >= 10 scoreable "
            "events, so a replay of the corpus as one pseudo-user never leaves WARMUP",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM population (Stage 9) + personal (V3) assessment of one WAV")
    parser.add_argument("wav")
    parser.add_argument("--user", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--recorded-at", required=True, help="ISO 8601 recording time")
    parser.add_argument("--state-dir", required=True, help="personal-state directory (one file per user and device)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-domain", default="unknown", choices=INPUT_DOMAINS)
    parser.add_argument("--close-sitting", action="store_true", help="close the sitting after this recording")
    args = parser.parse_args()
    reference_ = load_reference(args.reference)
    store = PersonalReferenceStore(args.state_dir)
    personal_ = store.load_or_create(args.user, args.device, load_config(args.config), reference_.reference_id,
                                     args.recorded_at)
    waveform_, rate = read_wav(args.wav)
    result = assess_recording_with_personal(waveform_, rate, reference_, personal_, recorded_at=args.recorded_at,
                                            detector=OnnxEventClassifier(model_path=args.model),
                                            input_domain=args.input_domain, recording_id=Path(args.wav).name)
    if args.close_sitting:
        personal_.close_sitting()
    store.save(personal_)
    print(json.dumps(result, indent=2, allow_nan=False))
