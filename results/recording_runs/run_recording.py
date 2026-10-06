"""Full PRISM pipeline run on one recording, with every intermediate stage saved.

Not part of the research pipeline and introduces no threshold.  Purpose: give the
app team a stage-by-stage reference for one recording, so a mobile
re-implementation can be checked at every boundary, not only at the final output.

    1. input                    -> input.json
    2. DSP (librosa_extractor)  -> frame_features.csv      (n_frames x 124)
    3. detector (ONNX)          -> window_predictions.csv  (raw logits, softmax, argmax per 0.2 s window)
    4. inference contract V2    -> contract_output.json    (prism_inference.analyze_recording, unchanged)
    5. context (not app output) -> summary.json, diagnostic.png

Only contract_output.json is the app-facing result.  class_segments in summary.json
group Drug/Exhale windows with the Inhale grouping rule for description only: the
contract defines no Drug or Exhale events.

Usage: venv/Scripts/python.exe results/recording_runs/run_recording.py <wav> [--input-domain reference_dataset]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from baseline_v1 import fit_baseline, recording_sessions  # noqa: E402
from feature_extractor import create_feature_windows  # noqa: E402
from inhale_dataset import interval_iou, parse_recording_timestamp  # noqa: E402
from librosa_extractor import FEATURE_NAMES, extract_features_from_audio  # noqa: E402
from post_event import OnnxEventClassifier, generate_window_predictions, group_inhale_events, plot_diagnostic  # noqa: E402
from prism_inference import (GROUPING, INPUT_DOMAINS, V2_FEATURES, analyze_recording, anomaly_score,  # noqa: E402
                             load_baseline, read_wav)

OUT_ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ANNOTATION_COLOURS = {"Drug": "#d62728", "Exhale": "#1f77b4", "Inhale": "#2ca02c"}


def recorded_at(name: str) -> str | None:
    try:
        return parse_recording_timestamp(name).isoformat()
    except ValueError:
        return None


def annotations_for(name: str) -> pd.DataFrame:
    table = pd.read_csv(Path(config.DATA_DIR) / "annotation.csv", header=None,
                        names=["recording_file", "label", "start_sample", "end_sample"])
    table = table[table["recording_file"] == name].copy()
    table["start_s"] = table["start_sample"] / config.LIBROSA_SR
    table["end_s"] = table["end_sample"] / config.LIBROSA_SR
    return table.sort_values("start_s").reset_index(drop=True)


def window_agreement(windows: pd.DataFrame, annotations: pd.DataFrame) -> list[dict]:
    """Per annotated event: share of windows centred inside it whose argmax is the annotated class."""
    centre = (windows["start_s"] + windows["end_s"]) / 2
    rows = []
    for _, a in annotations.iterrows():
        inside = windows[(centre >= a["start_s"]) & (centre < a["end_s"])]
        rows.append({"label": a["label"], "start_s": a["start_s"], "end_s": a["end_s"], "n_windows": int(len(inside)),
                     "frac_windows_same_label": float((inside["label"] == a["label"]).mean()) if len(inside) else None,
                     "window_label_counts": inside["label"].value_counts().to_dict()})
    return rows


def stage1_comparison(name: str, output: dict) -> dict | None:
    table = pd.read_csv(RESULTS / "inhale_dataset" / "inhale_events_v1.csv")
    rows = table[table["recording_file"] == name].sort_values("event_id")
    if rows.empty and not output["events"]:
        return {"n_events_stage1": 0, "n_events_contract": 0}
    if len(rows) != len(output["events"]):
        return {"n_events_stage1": int(len(rows)), "n_events_contract": output["n_events"], "match": False}
    diffs = []
    for (_, row), event in zip(rows.iterrows(), output["events"]):
        values = event["feature_values"] or {}
        diffs.append({"event_id": event["event_id"],
                      "abs_diff_start_s": abs(row["start_s"] - event["start_time"]),
                      "abs_diff_end_s": abs(row["end_s"] - event["end_time"]),
                      "abs_diff_confidence": abs(row["confidence"] - event["detector_confidence"]),
                      "max_rel_diff_features": max((abs(row[f] - values[f]) / abs(row[f]) for f in values), default=None),
                      "usable_stage1": bool(row["usable"]), "scored_contract": event["status"] == "SCORE_ONLY"})
    return {"n_events_stage1": int(len(rows)), "n_events_contract": output["n_events"], "events": diffs}


def held_out_scores(name: str, output: dict) -> dict | None:
    """Context only: each scored event against a baseline fitted without its own session (Stage 5 strategy C)."""
    sessions = recording_sessions()
    if name not in sessions:
        return None
    table = pd.read_csv(RESULTS / "inhale_dataset" / "inhale_events_v1.csv")
    training = table[table["usable"] & (table["recording_file"].map(sessions) != sessions[name])]
    fitted = fit_baseline(training, V2_FEATURES)
    centre = np.array([fitted.parameters[f].median for f in V2_FEATURES])
    scale = np.array([fitted.parameters[f].scale for f in V2_FEATURES])
    events = []
    for event in output["events"]:
        if event["status"] != "SCORE_ONLY":
            continue
        z = (np.array([event["feature_values"][f] for f in V2_FEATURES]) - centre) / scale
        events.append({"event_id": event["event_id"], "anomaly_score_held_out": anomaly_score(z),
                       "feature_z_scores_held_out": dict(zip(V2_FEATURES, z.tolist())),
                       "anomaly_score_contract": event["anomaly_score"]})
    return {"session": sessions[name], "n_training_events": int(len(training)),
            "n_training_sessions": int(training["recording_file"].map(sessions).nunique()), "events": events}


def main(wav: Path, input_domain: str) -> None:
    out = OUT_ROOT / wav.stem
    out.mkdir(parents=True, exist_ok=True)
    detector, baseline = OnnxEventClassifier(), load_baseline()
    waveform, rate = read_wav(wav)

    # 1. input
    info = {"file": wav.name, "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "sample_rate": rate,
            "n_samples": int(len(waveform)), "duration_s": len(waveform) / rate,
            "peak_abs": float(np.abs(waveform).max()), "rms": float(np.sqrt(np.mean(np.square(waveform)))),
            "recorded_at": recorded_at(wav.name), "input_domain": input_domain}
    (out / "input.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # 4. inference contract (the app-facing output)
    output = analyze_recording(waveform, rate, detector=detector, baseline=baseline, input_domain=input_domain,
                               recording_id=wav.name, recorded_at=info["recorded_at"])
    (out / "contract_output.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if output["recording_status"] == "INPUT_ERROR":
        print(json.dumps(output, indent=2))
        return

    # 2. DSP: the 124 per-frame features the detector consumes
    features = extract_features_from_audio(waveform, rate)
    frames = pd.DataFrame(features, columns=FEATURE_NAMES)
    frames.insert(0, "time_s", np.arange(len(frames)) * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR)
    frames.insert(0, "frame", np.arange(len(frames)))
    frames.to_csv(out / "frame_features.csv", index=False, float_format="%.9g")

    # 3. detector: raw ONNX logits, softmax and argmax per window
    predictions = generate_window_predictions(waveform, sample_rate=rate, model=detector)
    windows_in = create_feature_windows(features, window_size=detector.window_frames, stride=config.WINDOW_STRIDE)
    logits = detector.session.run([detector.output_name], {detector.input_name: windows_in.astype(np.float32)})[0]
    labels = detector.class_names
    windows = pd.DataFrame([{"window": p.index, "start_s": p.start, "end_s": p.end,
                             "first_frame": p.index * config.WINDOW_STRIDE, "label": p.label,
                             **{f"logit_{c}": float(v) for c, v in zip(labels, row)},
                             **{f"p_{c}": p.probabilities[c] for c in labels}}
                            for p, row in zip(predictions, logits)])
    if not (windows[[f"logit_{c}" for c in labels]].to_numpy().argmax(axis=1)
            == windows["label"].map(labels.index).to_numpy()).all():
        raise RuntimeError("logit argmax disagrees with the post_event window labels")
    windows.to_csv(out / "window_predictions.csv", index=False, float_format="%.9g")

    # 5. context: descriptive segments of every class, annotation, Stage 1 and held-out comparisons
    segments = {label: [{"start_s": e.start, "end_s": e.end, "duration_s": e.duration, "mean_p": e.confidence,
                         "window_count": e.window_count}
                        for e in group_inhale_events(predictions, replace(GROUPING, target_label=label))]
                for label in ("Drug", "Exhale", "Inhale")}
    annotations = annotations_for(wav.name)
    annotated_inhale = annotations[annotations["label"] == "Inhale"]
    reference = pd.read_csv(RESULTS / "v2_validation" / "representation_comparison.csv").set_index("representation").loc["V2"]
    summary = {
        "input": info,
        "contract": {k: output[k] for k in ("contract_version", "baseline_id", "recording_status", "error",
                                            "baseline_domain_validated", "n_events", "n_scored")},
        "events": [{k: e[k] for k in ("event_id", "start_time", "end_time", "duration_s", "detector_confidence",
                                      "status", "not_scoreable_reasons", "anomaly_score", "feature_z_scores", "mean_rms")}
                   | {"iou_with_annotated_inhale": max((interval_iou(e["start_time"], e["end_time"], a["start_s"], a["end_s"])
                                                        for _, a in annotated_inhale.iterrows()), default=None)}
                   for e in output["events"]],
        "window_label_fractions": windows["label"].value_counts(normalize=True).round(4).to_dict(),
        "class_segments_descriptive": {
            "note": "Drug/Exhale windows grouped with the Inhale grouping rule for description only; "
                    "the contract defines no Drug or Exhale events.", **segments},
        "annotations": annotations[["label", "start_s", "end_s"]].to_dict("records"),
        "window_agreement_with_annotation": window_agreement(windows, annotations),
        "stage1_comparison": stage1_comparison(wav.name, output),
        "held_out_session_scores": held_out_scores(wav.name, output),
        "reference_context": {
            "note": "Held-out V2 anomaly_score (rms_z) of the 318 usable reference events (Stage 8). "
                    "Context only: no threshold exists.",
            "p05": float(reference["usable_p05"]), "median": float(reference["usable_median_rms_z"]),
            "p95": float(reference["usable_p95"])},
        "detector_note": "The ONNX detector was trained on about two-thirds of the reference recordings (training "
                         "fold not saved), so agreement with annotations on a reference recording may be in-sample.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    figure, axes = plot_diagnostic(waveform, predictions, group_inhale_events(predictions, GROUPING), sample_rate=rate)
    for _, a in annotations.iterrows():
        axes[0].axvspan(a["start_s"], a["end_s"], ymin=0.92, ymax=1.0, color=ANNOTATION_COLOURS.get(a["label"], "k"))
        axes[0].text(a["start_s"], 0.9, f"annotated {a['label']}", va="top", fontsize=8,
                     transform=axes[0].get_xaxis_transform())
    axes[0].set_title(f"{wav.name}: green band = contract Inhale event; top bars = dataset annotation")
    figure.savefig(out / "diagnostic.png", dpi=150)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--input-domain", default="unknown", choices=INPUT_DOMAINS)
    args = parser.parse_args()
    main(args.wav, args.input_domain)
