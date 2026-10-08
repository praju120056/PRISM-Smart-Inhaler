"""Stage 10: PROXY/SYNTHETIC evaluation of the V3 personal reference (mechanics only).

This is NOT personalization validation.  The PRISM corpus has no user or device identifiers, no longitudinal
personal labels and no technique labels.  What is evaluated is how the implemented mechanism behaves:

    real-corpus replay   all 361 recordings in time order through V3 as ONE pseudo-user/device, using the Stage 9
                         deployment outputs (results/final_assessment/e2e_deployment_*.csv) as the population channel
    proxy users          synthetic sequences of sittings, each resampled (with replacement) from one real inferred
                         sitting (Stage 2 session with >= min_events_per_sitting scoreable events), in the frozen
                         Stage 9 population-reference coordinates
    scenarios A-J        constructed outliers, shifts, perturbations, gaps and switches; their ground truth comes
                         from the construction, never from the model

The proxy sittings are resampled from the same 318 events the Stage 9 reference was fitted on (in-sample), and
the sitting-to-sitting variation of a proxy user is the variation between real inferred sittings, whose cause
(person, device, day, environment) is unknown.  Each proxy sitting is fed as one recording; tests show this is
identical to feeding its events one by one, because B cannot change inside a sitting.

    venv/Scripts/python.exe src/personal_reference_evaluation.py --output-dir <empty dir>
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config
from inhale_dataset import _git_state, _json_default, _relative, _sha256, parse_recording_timestamp
from personal_reference import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    PERSONAL_CONTRACT_VERSION,
    SENSITIVITY_CONFIG,
    SENSITIVITY_CONFIG_PATH,
    STAGE9_DIR,
    PersonalReference,
    PersonalReferenceConfig,
    personal_contract,
    write_config,
)
from prism_assessment import DEFAULT_REFERENCE_PATH, load_reference
from prism_inference import V2_FEATURES


STAGE9_EVENTS = STAGE9_DIR / "e2e_deployment_events.csv"
STAGE9_RECORDINGS = STAGE9_DIR / "e2e_deployment_recordings.csv"
PERTURBED = Path(config.RESULTS_DIR) / "v2_validation" / "perturbed_features.csv"
ALLOWED_PRESENT = ("README.md",)

SEED = 20261007
REPLICATES = 200
N_SITTINGS = 80                     # one sitting per day
ONSET = 30                          # first changed sitting: after the 10-sitting warm-up plus 20 unchanged sittings
LATE = 10                           # sittings summarised before the onset and at the end
EXTREME = 1e3                       # injected extreme event: z = +1000 on every feature
OUTLIER_SITTING = 40
ISOLATED_OUTLIER_SITTINGS = (25, 35, 45, 55, 65)
OPEN_GATE_SITTING = ONSET + 20      # inside the +1.0 shift, where the gate is normally open
GAP_AFTER = 6                       # long-gap scenario: the gap precedes shifted sitting ONSET + GAP_AFTER
TEMPORARY_DURATIONS = (3, 5, 9)
START = datetime(2026, 1, 5, 8, 0, 0)
PERTURBATIONS = {"noise20": ("noise", 20.0), "noise10": ("noise", 10.0), "tilt0.9": ("tilt", 0.9),
                 "tilt0.5": ("tilt", 0.5)}
POOL_PREFIXES = {"A": ("2018-01", "2018-02"), "B": ("2018-05",)}   # user/device-switch proxy: Jan-Feb vs May
CONFIGS = (PersonalReferenceConfig(), SENSITIVITY_CONFIG)

SCENARIOS = {
    "N0_null": "no change: every sitting resampled from a random real sitting",
    "A1_extreme_event": "one event (z = +1000 on all features) in sitting 40 of N0",
    "A2_extreme_event_gate_open": "one extreme event in sitting ONSET+20 of D (consensus gate normally open)",
    "B1_isolated_outliers": "one extreme event in each of sittings 25, 35, 45, 55, 65 of N0",
    "B2_outlier_sitting_noise10": "sitting 40 of N0 replaced by its 10 dB-SNR white-noise version",
    "B3_extreme_sitting_gate_open": "every event of sitting ONSET+20 of D set to +1000",
    "C_shift_0.5": "persistent +0.5 on every feature from sitting ONSET",
    "D_shift_1.0": "persistent +1.0 on every feature from sitting ONSET",
    "E_shift_2.0": "persistent +2.0 on every feature from sitting ONSET",
    "F_noise_20dB": "every event from sitting ONSET replaced by its 20 dB-SNR white-noise version (Stage 6/8)",
    "G_tilt_0.9": "every event from sitting ONSET passed through y[n] = x[n] - 0.9 x[n-1] (microphone-like tilt)",
    "G2_tilt_0.5": "as G with a = 0.5",
    "H_temporary_noise_3": "20 dB noise in sittings ONSET..ONSET+2 only",
    "H_temporary_noise_5": "20 dB noise in sittings ONSET..ONSET+4 only",
    "H_temporary_noise_9": "20 dB noise in sittings ONSET..ONSET+8 only",
    "I_long_gap_shift_1.0": "D with a gap of max_gap_days + 1 days before shifted sitting ONSET+6",
    "J0_pool_A_null": "no change: sittings resampled from the Jan-Feb 2018 sittings only",
    "J_user_device_switch": "Jan-Feb sittings before ONSET, May 2018 sittings from ONSET (unknown cause)",
}
CLEAN_PAIR = {"A1_extreme_event": "N0_null", "B1_isolated_outliers": "N0_null", "B2_outlier_sitting_noise10": "N0_null",
              "H_temporary_noise_3": "N0_null", "H_temporary_noise_5": "N0_null", "H_temporary_noise_9": "N0_null",
              "A2_extreme_event_gate_open": "D_shift_1.0", "B3_extreme_sitting_gate_open": "D_shift_1.0",
              "I_long_gap_shift_1.0": "D_shift_1.0", "J_user_device_switch": "J0_pool_A_null"}


class EvaluationError(RuntimeError):
    """Inputs or output directory do not allow a reproducible Stage 10 run."""


def require_clean_output_dir(output: Path) -> None:
    """Refuse a directory holding anything but README.md, so every result file comes from this run."""
    leftovers = sorted(p.name + ("/" if p.is_dir() else "") for p in output.iterdir() if p.name not in ALLOWED_PRESENT)
    if leftovers:
        raise EvaluationError(f"output directory {output} is not empty: {', '.join(leftovers[:5])}"
                              f"{' ...' if len(leftovers) > 5 else ''}; remove them or choose another --output-dir")


def rms_rows(values) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.atleast_2d(np.asarray(values, dtype=np.float64))), axis=1))


def project(baseline: np.ndarray, radius: float) -> np.ndarray:
    norm = float(rms_rows(baseline)[0])
    return baseline * (radius / norm) if norm > radius else baseline


# ── Inputs ──────────────────────────────────────────────────────────────────

def load_proxy_data(reference, min_events: int) -> dict:
    """Stage 9 deployment z of the scoreable events, perturbed versions, and the real sittings used as sources."""
    events = pd.read_csv(STAGE9_EVENTS, float_precision="round_trip")
    scoreable = events[events["scoreability"] == "SCOREABLE"].reset_index(drop=True)
    z = scoreable[[f"feature_deviations.{f}" for f in V2_FEATURES]].to_numpy(np.float64)
    recomputed = reference.baseline.z(scoreable[[f"feature_values.{f}" for f in V2_FEATURES]].to_numpy(np.float64))
    z_check = float(np.abs(z - recomputed).max())
    if z_check > 1e-12:
        raise EvaluationError(f"Stage 9 deviations are not in the reference coordinates (max diff {z_check})")
    perturbed_raw = pd.read_csv(PERTURBED, float_precision="round_trip")
    key = (scoreable["recording_file"] + "#" + scoreable["event_id"].astype(str)).to_numpy()

    def z_of(transform: str, magnitude: float) -> np.ndarray:
        rows = perturbed_raw[(perturbed_raw["transform"] == transform)
                             & np.isclose(perturbed_raw["magnitude"], magnitude)]
        rows = rows.set_index(rows["recording_file"] + "#" + rows["event_id"].astype(str))
        missing = sorted(set(key) - set(rows.index))
        if missing:
            raise EvaluationError(f"{len(missing)} scoreable events have no {transform} {magnitude} version")
        return reference.baseline.z(rows.loc[key, list(V2_FEATURES)].to_numpy(np.float64))

    identity = z_of("identity", 0.0)
    # The perturbation effect is added to the deployment z, so the unperturbed events stay bit-identical.
    perturbed = {name: z + (z_of(t, m) - identity) for name, (t, m) in PERTURBATIONS.items()}
    sizes = scoreable.groupby("session").size()
    qualifying = sorted(s for s in sizes.index if sizes[s] >= min_events)
    sitting_events = {s: np.flatnonzero((scoreable["session"] == s).to_numpy()) for s in qualifying}
    pools = {"all": tuple(qualifying)}
    pools.update({name: tuple(s for s in qualifying if s.startswith(prefixes)) for name, prefixes in POOL_PREFIXES.items()})
    medians = {s: np.median(z[idx], axis=0) for s, idx in sitting_events.items()}
    return {"scoreable": scoreable, "z": z, "perturbed": perturbed, "sitting_events": sitting_events, "pools": pools,
            "source_medians": medians, "cut": reference.cut, "reference_id": reference.reference_id,
            "checks": {"z_vs_reference_max_abs_diff": z_check,
                       "perturbed_identity_vs_deployment_max_abs_diff": float(np.abs(identity - z).max()),
                       "n_scoreable_events": int(len(z)), "n_sittings": int(len(sizes)),
                       "source_sittings": {s: int(sizes[s]) for s in qualifying},
                       "pools": {k: list(v) for k, v in pools.items()}}}


# ── Scenarios ───────────────────────────────────────────────────────────────

def replicate_draws(replicate: int, max_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniforms shared by every scenario of one replicate (paired comparisons)."""
    rng = np.random.default_rng([SEED, replicate])
    return rng.random(N_SITTINGS), rng.random((N_SITTINGS, max_events))


def draw_sittings(data: dict, draws, pool_of: Callable[[int], str]) -> list[np.ndarray]:
    source_u, event_u = draws
    sittings = []
    for k in range(N_SITTINGS):
        pool = data["pools"][pool_of(k)]
        source = pool[min(int(source_u[k] * len(pool)), len(pool) - 1)]
        idx = data["sitting_events"][source]
        sittings.append(idx[np.minimum((event_u[k, :len(idx)] * len(idx)).astype(int), len(idx) - 1)])
    return sittings


def build_scenario(name: str, data: dict, draws, cfg: PersonalReferenceConfig) -> dict:
    """Sittings (z arrays), extra days before given sittings, and the constructed shift vector (if any)."""
    pool_of = {"J_user_device_switch": lambda k: "A" if k < ONSET else "B",
               "J0_pool_A_null": lambda k: "A"}.get(name, lambda k: "all")
    idx = draw_sittings(data, draws, pool_of)
    z = [data["z"][i].copy() for i in idx]
    extra_days, shift = {}, None
    ones = np.ones(len(V2_FEATURES))

    def perturb(kind: str, sittings) -> None:
        for k in sittings:
            z[k] = data["perturbed"][kind][idx[k]].copy()

    def effect(kind: str) -> np.ndarray:
        return np.median(data["perturbed"][kind] - data["z"], axis=0)

    if name in ("D_shift_1.0", "A2_extreme_event_gate_open", "B3_extreme_sitting_gate_open", "I_long_gap_shift_1.0",
                "C_shift_0.5", "E_shift_2.0"):
        size = {"C_shift_0.5": 0.5, "E_shift_2.0": 2.0}.get(name, 1.0)
        for k in range(ONSET, N_SITTINGS):
            z[k] += size
        shift = size * ones
    if name == "A1_extreme_event":
        z[OUTLIER_SITTING][0] = EXTREME
    elif name == "B1_isolated_outliers":
        for k in ISOLATED_OUTLIER_SITTINGS:
            z[k][0] = EXTREME
    elif name == "B2_outlier_sitting_noise10":
        perturb("noise10", [OUTLIER_SITTING])
    elif name == "A2_extreme_event_gate_open":
        z[OPEN_GATE_SITTING][0] = EXTREME
    elif name == "B3_extreme_sitting_gate_open":
        z[OPEN_GATE_SITTING][:] = EXTREME
    elif name in ("F_noise_20dB", "G_tilt_0.9", "G2_tilt_0.5"):
        kind = {"F_noise_20dB": "noise20", "G_tilt_0.9": "tilt0.9", "G2_tilt_0.5": "tilt0.5"}[name]
        perturb(kind, range(ONSET, N_SITTINGS))
        shift = effect(kind)
    elif name.startswith("H_temporary_noise_"):
        perturb("noise20", range(ONSET, ONSET + int(name.rsplit("_", 1)[1])))
        shift = effect("noise20")
    elif name == "I_long_gap_shift_1.0":
        extra_days = {ONSET + GAP_AFTER: cfg.max_gap_days + 1}
    elif name == "J_user_device_switch":
        medians = data["source_medians"]
        shift = (np.mean([medians[s] for s in data["pools"]["B"]], axis=0)
                 - np.mean([medians[s] for s in data["pools"]["A"]], axis=0))
    return {"sittings": z, "extra_days": extra_days, "shift": shift}


def run_stream(cfg: PersonalReferenceConfig, data: dict, scenario: dict) -> dict:
    """Feed one proxy user through V3 (assess first, adapt at sitting close) and keep what the metrics need."""
    personal = PersonalReference(cfg, user_id="proxy-user", device_id="proxy-device",
                                 population_reference_id=data["reference_id"], created_at=START.isoformat())
    t, used, deviations = START, [], []
    for k, z in enumerate(scenario["sittings"]):
        if k:
            t += timedelta(days=1 + scenario["extra_days"].get(k, 0))
        out = personal.observe_recording(list(z), [True] * len(z), t.isoformat())
        used.append([out["baseline_used"][f] for f in V2_FEATURES])
        deviations.append([e["personal_deviation"] for e in out["events"]])
    personal.close_sitting()
    return {"traj": np.vstack([np.asarray(used), personal.baseline]),
            "entries": [e for e in personal.audit if e["kind"] == "sitting_closed"], "deviations": deviations}


def run_metrics(cfg: PersonalReferenceConfig, data: dict, scenario: dict, result: dict,
                clean: dict | None) -> dict:
    traj, entries = result["traj"], result["entries"]
    norms = rms_rows(traj)
    updated = [e for e in entries if e["decision"] == "UPDATED"]
    qualifying = [e for e in entries if e["decision"] != "NOT_QUALIFYING"]
    step_ratio = [float(rms_rows(np.subtract(e["baseline_after"], e["baseline_before"]))[0])
                  / cfg.sitting_bound(e["n_qualifying_events"]) for e in qualifying]
    late, pre = range(N_SITTINGS - LATE, N_SITTINGS), range(ONSET - LATE, ONSET)

    def population_outside(ks) -> float:
        return float(np.mean(np.concatenate([rms_rows(scenario["sittings"][k]) > data["cut"] for k in ks])))

    def personal_median(ks) -> float:
        values = [v for k in ks for v in result["deviations"][k] if v is not None]
        return float(np.median(values)) if values else float("nan")

    m = {"b_end_rms": float(norms[-1]), "b_max_rms": float(norms.max()),
         **{f"b_end.{f}": float(traj[-1, j]) for j, f in enumerate(V2_FEATURES)},
         "n_updates": len(updated), "n_updates_before_onset": sum(e["sitting_index"] < ONSET for e in updated),
         "n_updates_from_onset": sum(e["sitting_index"] >= ONSET for e in updated),
         "first_update_sitting": updated[0]["sitting_index"] if updated else float("nan"),
         "n_capped": sum(bool(e.get("trust_region_capped")) for e in entries),
         "ends_at_boundary": bool(norms[-1] >= cfg.trust_radius * (1 - 1e-12)),
         "n_gap_resets": sum(bool(e.get("gap_reset")) for e in entries),
         "n_not_qualifying": len(entries) - len(qualifying),
         "max_step_over_bound": max(step_ratio) if step_ratio else 0.0,
         "population_outside_pre": population_outside(pre), "population_outside_late": population_outside(late),
         "personal_deviation_pre_median": personal_median(pre), "personal_deviation_late_median": personal_median(late),
         "population_deviation_late_median": float(np.median(np.concatenate(
             [rms_rows(scenario["sittings"][k]) for k in late])))}
    shift = scenario["shift"]
    if shift is not None:
        denominator = float(shift @ shift)
        absorbed = (traj - traj[ONSET]) @ shift / denominator
        target = project(traj[ONSET] + shift, cfg.trust_radius)
        reachable = float((target - traj[ONSET]) @ shift / denominator)
        m.update({"shift_rms": float(rms_rows(shift)[0]), "absorbable_fraction": reachable,
                  "absorbed_end": float(absorbed[-1]), "absorbed_max": float(absorbed[ONSET:].max()),
                  "residual_rms_end": float(rms_rows(traj[-1] - target)[0])})
        for level in (0.5, 0.9):
            hits = np.flatnonzero(absorbed[ONSET:] >= level * reachable) if reachable > 0 else np.array([])
            m[f"sittings_to_{int(100 * level)}pct_of_reachable"] = float(hits[0]) if hits.size else float("nan")
    if clean is not None:
        difference = rms_rows(traj - clean["traj"])
        m.update({"influence_max_rms": float(difference.max()), "influence_end_rms": float(difference[-1]),
                  "extra_updates": len(updated) - sum(e["decision"] == "UPDATED" for e in clean["entries"])})
    return m


def proxy_evaluation(data: dict, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_events = max(len(i) for i in data["sitting_events"].values())
    run_rows, trajectory_rows = [], []
    order = sorted(SCENARIOS, key=lambda n: n in CLEAN_PAIR)            # clean partners first
    for cfg in CONFIGS:
        for replicate in range(replicates):
            draws = replicate_draws(replicate, max_events)
            results = {}
            for name in order:
                scenario = build_scenario(name, data, draws, cfg)
                results[name] = run_stream(cfg, data, scenario)
                metrics = run_metrics(cfg, data, scenario, results[name], results.get(CLEAN_PAIR.get(name)))
                run_rows.append({"config_id": cfg.config_id, "scenario": name, "replicate": replicate, **metrics})
                traj = results[name]["traj"]
                row = {"config_id": cfg.config_id, "scenario": name, "replicate": replicate}
                norms = rms_rows(traj)
                absorbed = ((traj - traj[ONSET]) @ scenario["shift"] / float(scenario["shift"] @ scenario["shift"])
                            if scenario["shift"] is not None else np.full(len(traj), np.nan))
                clean = results.get(CLEAN_PAIR.get(name))
                influence = rms_rows(traj - clean["traj"]) if clean is not None else np.full(len(traj), np.nan)
                for k in range(len(traj)):
                    trajectory_rows.append({**row, "sitting": k, "b_rms": norms[k], "absorbed": absorbed[k],
                                            "influence_rms": influence[k]})
    runs = pd.DataFrame(run_rows)
    trajectories = (pd.DataFrame(trajectory_rows).groupby(["config_id", "scenario", "sitting"], sort=True)
                    .agg(b_rms_mean=("b_rms", "mean"), absorbed_mean=("absorbed", "mean"),
                         absorbed_p05=("absorbed", lambda v: v.quantile(0.05)),
                         absorbed_p95=("absorbed", lambda v: v.quantile(0.95)),
                         influence_rms_mean=("influence_rms", "mean")).reset_index())
    return runs, trajectories


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [c for c in runs.columns if c not in ("config_id", "scenario", "replicate")]
    for (cfg_id, name), block in runs.groupby(["config_id", "scenario"], sort=False):
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce").astype(float)
            finite = values[np.isfinite(values)]
            rows.append({"config_id": cfg_id, "scenario": name, "metric": metric, "n": int(len(values)),
                         "n_finite": int(len(finite)),
                         "mean": float(finite.mean()) if len(finite) else float("nan"),
                         "median": float(finite.median()) if len(finite) else float("nan"),
                         "p05": float(finite.quantile(0.05)) if len(finite) else float("nan"),
                         "p95": float(finite.quantile(0.95)) if len(finite) else float("nan"),
                         "max": float(finite.max()) if len(finite) else float("nan")})
    return pd.DataFrame(rows)


# ── Real-corpus replay (one pseudo-user/device) ─────────────────────────────

def real_corpus_replay(cfg: PersonalReferenceConfig, reference) -> tuple[pd.DataFrame, pd.DataFrame, PersonalReference]:
    events = pd.read_csv(STAGE9_EVENTS, float_precision="round_trip")
    recordings = pd.read_csv(STAGE9_RECORDINGS)
    recordings = recordings.assign(recorded_at=recordings["recording_file"].map(parse_recording_timestamp))
    recordings = recordings.sort_values(["recorded_at", "recording_file"], kind="mergesort").reset_index(drop=True)
    by_recording = {name: block.sort_values("event_id") for name, block in events.groupby("recording_file")}
    personal = PersonalReference(cfg, user_id="prism-corpus-pseudo-user", device_id="prism-corpus-unknown-device",
                                 population_reference_id=reference.reference_id,
                                 created_at=recordings["recorded_at"].iloc[0].isoformat())
    rows, sitting_sessions = [], {}
    for recording in recordings.itertuples(index=False):
        block = by_recording.get(recording.recording_file, events.iloc[0:0])
        scoreable = (block["scoreability"] == "SCOREABLE").tolist()
        z_rows = [[float(r[f"feature_deviations.{f}"]) for f in V2_FEATURES] if ok else None
                  for (_, r), ok in zip(block.iterrows(), scoreable)]
        out = personal.observe_recording(z_rows, scoreable, recording.recorded_at.isoformat(), recording.recording_file)
        sitting_sessions.setdefault(out["sitting_index"], set()).add(recording.session)
        for (_, event), personal_event in zip(block.iterrows(), out["events"]):
            rows.append({"config_id": cfg.config_id, "recording_file": recording.recording_file,
                         "event_id": int(event["event_id"]), "session": recording.session,
                         "sitting_index": out["sitting_index"], "scoreability": event["scoreability"],
                         "aggregate_deviation": event["aggregate_deviation"],
                         "reference_tail_probability": event["reference_tail_probability"],
                         "population_assessment": event["assessment"], "personal_status": out["personal_status"],
                         "personal_assessment": personal_event["personal_assessment"],
                         "personal_deviation": personal_event["personal_deviation"],
                         "state_id_used": out["state_id_used"]})
    personal.close_sitting()
    sittings = pd.DataFrame([{"config_id": cfg.config_id, "sitting_index": e["sitting_index"],
                              "stage2_sessions": ";".join(sorted(sitting_sessions[e["sitting_index"]])),
                              "started_at": e["started_at"], "ended_at": e["ended_at"],
                              "n_recordings": e["n_recordings"], "n_events_total": e["n_events_total"],
                              "n_qualifying_events": e["n_qualifying_events"], "decision": e["decision"],
                              "consensus_window_size": e.get("consensus_window_size"),
                              "n_qualifying_after": e.get("n_qualifying_after"), "status_after": e.get("status_after")}
                             for e in personal.audit if e["kind"] == "sitting_closed"])
    return pd.DataFrame(rows), sittings, personal


# ── Figures (from the saved CSVs) ───────────────────────────────────────────

def plot_absorption(trajectories_csv: Path, path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(trajectories_csv)
    shown = ["C_shift_0.5", "D_shift_1.0", "E_shift_2.0", "F_noise_20dB", "G_tilt_0.9", "I_long_gap_shift_1.0",
             "J_user_device_switch"]
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5195", "#555555"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for axis, cfg in zip(axes, CONFIGS):
        for name, colour in zip(shown, colours):
            block = frame[(frame["config_id"] == cfg.config_id) & (frame["scenario"] == name)
                          & (frame["sitting"] >= ONSET)]
            axis.plot(block["sitting"] - ONSET, block["absorbed_mean"], color=colour, label=name)
        axis.axhline(1.0, color="#999999", lw=0.8, ls=":")
        axis.set_title(f"{cfg.config_id} (PROXY/SYNTHETIC)", fontsize=9)
        axis.set_xlabel("qualifying sittings since the change began")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean fraction of the constructed shift absorbed by B")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_influence(summary_csv: Path, path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(summary_csv)
    shown = ["A1_extreme_event", "B1_isolated_outliers", "B2_outlier_sitting_noise10", "A2_extreme_event_gate_open",
             "B3_extreme_sitting_gate_open", "H_temporary_noise_3", "H_temporary_noise_5", "H_temporary_noise_9"]
    fig, axis = plt.subplots(figsize=(11, 4.2))
    width = 0.38
    for offset, cfg, colour in ((-width / 2, CONFIGS[0], "#2a78d6"), (width / 2, CONFIGS[1], "#eb6834")):
        block = frame[(frame["config_id"] == cfg.config_id) & (frame["metric"] == "influence_max_rms")]
        block = block.set_index("scenario").loc[shown]
        x = np.arange(len(shown)) + offset
        axis.bar(x, block["median"], width, color=colour, alpha=0.8, label=f"{cfg.config_id}: median")
        axis.errorbar(x, block["median"], yerr=[block["median"] - block["p05"], block["p95"] - block["median"]],
                      fmt="none", ecolor="#333333", capsize=3, lw=0.8)
    bound = CONFIGS[0].sitting_bound(CONFIGS[0].min_events_per_sitting)
    axis.axhline(bound, color="#c0392b", ls="--", lw=1, label=f"one-sitting bound at n = 10 ({bound:.3f})")
    axis.set_xticks(np.arange(len(shown)))
    axis.set_xticklabels(shown, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("max rms(B - B_clean) over the stream")
    axis.set_title("Outlier / temporary-change influence on the personal baseline (PROXY/SYNTHETIC; bars 5-95%)",
                   fontsize=9)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Driver ──────────────────────────────────────────────────────────────────

def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None             # NaN (e.g. never converged) is written as null


def key_results(summary: pd.DataFrame) -> dict:
    wanted = ["b_end_rms", "n_updates", "n_updates_from_onset", "absorbed_end", "absorbable_fraction",
              "sittings_to_50pct_of_reachable", "sittings_to_90pct_of_reachable", "influence_max_rms", "extra_updates",
              "n_capped", "ends_at_boundary", "max_step_over_bound", "population_outside_late",
              "personal_deviation_pre_median", "personal_deviation_late_median"]
    out = {}
    for (cfg_id, name), block in summary.groupby(["config_id", "scenario"], sort=False):
        values = block.set_index("metric")
        out.setdefault(cfg_id, {})[name] = {
            m: {k: _finite_or_none(values.loc[m, k]) for k in ("median", "p05", "p95", "mean", "n_finite")}
            for m in wanted if m in values.index}
    return out


def run_analysis(output_dir: str | Path = DEFAULT_OUTPUT_DIR, replicates: int = REPLICATES,
                 make_plots: bool = True) -> dict:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    require_clean_output_dir(output)
    inputs = [STAGE9_EVENTS, STAGE9_RECORDINGS, PERTURBED, Path(DEFAULT_REFERENCE_PATH)]
    hashes_before = {_relative(p): _sha256(p) for p in inputs}
    reference = load_reference()
    primary = CONFIGS[0]
    data = load_proxy_data(reference, primary.min_events_per_sitting)

    write_config(primary, output / DEFAULT_CONFIG_PATH.name)
    write_config(SENSITIVITY_CONFIG, output / SENSITIVITY_CONFIG_PATH.name)
    (output / "personal_contract_v3.json").write_text(
        json.dumps(personal_contract(primary, SENSITIVITY_CONFIG), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    replay_events, replay_sittings, replay_states = [], [], {}
    for cfg in CONFIGS:
        events, sittings, state = real_corpus_replay(cfg, reference)
        replay_events.append(events)
        replay_sittings.append(sittings)
        replay_states[cfg.config_id] = state
    replay_events, replay_sittings = pd.concat(replay_events), pd.concat(replay_sittings)
    replay_events.to_csv(output / "real_corpus_replay_events.csv", index=False)
    replay_sittings.to_csv(output / "real_corpus_replay_sittings.csv", index=False)
    (output / "real_corpus_replay_state.json").write_text(
        json.dumps(replay_states[primary.config_id].to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")

    runs, trajectories = proxy_evaluation(data, replicates)
    summary = summarize_runs(runs)
    runs.to_csv(output / "proxy_runs.csv", index=False)
    summary.to_csv(output / "proxy_summary.csv", index=False)
    trajectories.to_csv(output / "proxy_trajectories.csv", index=False)
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        plot_absorption(output / "proxy_trajectories.csv", output / "proxy_absorption.png")
        plot_influence(output / "proxy_summary.csv", output / "proxy_outlier_influence.png")

    hashes_after = {_relative(p): _sha256(p) for p in inputs}
    if hashes_after != hashes_before:
        raise EvaluationError("an input file changed during the run")
    primary_replay = replay_sittings[replay_sittings["config_id"] == primary.config_id]
    replay_primary_events = replay_events[replay_events["config_id"] == primary.config_id]
    summary_json = {
        "stage": "Stage 10 - V3 personal reference: PROXY/SYNTHETIC evaluation (not personalization validation)",
        "contract_version": PERSONAL_CONTRACT_VERSION,
        "question": "Does the implemented sitting-level, consensus-gated, bounded update behave as designed: limited "
                    "influence of isolated outliers and single unusual sittings, eventual movement under a sustained "
                    "shift, no movement from temporary changes, gap resets and trust-region capping?",
        "not_answered": "whether B represents a person, whether personal deviation detects real within-user change, "
                        "and whether a sustained shift reflects behaviour, noise, microphone or device",
        "parameters": {"seed": SEED, "replicates": replicates, "n_sittings": N_SITTINGS, "onset": ONSET,
                       "late_window": LATE, "extreme_z": EXTREME, "outlier_sitting": OUTLIER_SITTING,
                       "isolated_outlier_sittings": list(ISOLATED_OUTLIER_SITTINGS),
                       "open_gate_sitting": OPEN_GATE_SITTING, "gap_after": GAP_AFTER,
                       "temporary_durations": list(TEMPORARY_DURATIONS), "sitting_spacing": "1 day",
                       "perturbations": {k: list(v) for k, v in PERTURBATIONS.items()},
                       "pool_prefixes": {k: list(v) for k, v in POOL_PREFIXES.items()}},
        "configs": {cfg.config_id: {"config": cfg.to_dict(), "sha256": cfg.sha256,
                                    "one_sitting_bound_at_min_events": cfg.sitting_bound(cfg.min_events_per_sitting)}
                    for cfg in CONFIGS},
        "scenarios": SCENARIOS, "clean_pairs": CLEAN_PAIR, "data_checks": data["checks"],
        "real_corpus_replay": {
            "n_recordings": int(len(pd.read_csv(STAGE9_RECORDINGS))),
            "n_recordings_processed": int(primary_replay["n_recordings"].sum()),
            "n_recordings_with_events": int(replay_primary_events["recording_file"].nunique()),
            "n_events": int(len(replay_primary_events)),
            "n_scoreable_events": int((replay_primary_events["scoreability"] == "SCOREABLE").sum()),
            "n_sittings": int(len(primary_replay)),
            "sittings_match_stage2_sessions_one_to_one": bool(
                (~primary_replay["stage2_sessions"].str.contains(";")).all()
                and primary_replay["stage2_sessions"].is_unique),
            "n_qualifying_sittings": int((primary_replay["decision"] != "NOT_QUALIFYING").sum()),
            "qualifying_sitting_sizes": primary_replay.loc[primary_replay["decision"] != "NOT_QUALIFYING",
                                                           "n_qualifying_events"].astype(int).tolist(),
            "decisions": primary_replay["decision"].value_counts().to_dict(),
            "final_status": {cfg_id: state.status for cfg_id, state in replay_states.items()},
            "final_baseline": {cfg_id: state.baseline.tolist() for cfg_id, state in replay_states.items()},
            "personal_assessments": replay_primary_events["personal_assessment"].value_counts().to_dict(),
        },
        "key_results": key_results(summary),
        "inputs": hashes_before,
        "provenance": {"git": _git_state(), "python": platform.python_version(), "numpy": np.__version__,
                       "pandas": pd.__version__},
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary_json, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    return summary_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 10 PROXY/SYNTHETIC evaluation of the V3 personal reference")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.output_dir, args.replicates, make_plots=not args.no_plots)
    print(json.dumps(result["real_corpus_replay"], indent=2, default=_json_default))
    print(f"runtime {result['runtime_seconds']} s")
