"""Batch exploration of CNN-detected inhalation events.

This is deliberately an exploratory data product, not a technique classifier.
It applies the existing ONNX event detector and ``post_event`` grouping to every
recording, then writes one descriptive row per Inhale event for inspection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import config
from post_event import (
    DEFAULT_MODEL_PATH,
    OnnxEventClassifier,
    TemporalGroupingConfig,
    analyze_inhalation,
    detect_events,
    extract_event_audio,
)


DEFAULT_OUTPUT_DIR = Path(config.RESULTS_DIR) / "post_event"


def _event_row(recording_id: int, event_id: int, recording: Path, event, analysis: dict) -> dict:
    """Flatten descriptive analysis into a CSV-friendly event row."""
    spectral = analysis["spectral"]
    row = {
        "recording_id": recording_id,
        "event_id": event_id,
        "recording_file": recording.name,
        "label": event.label,
        "start_s": event.start,
        "end_s": event.end,
        "duration_s": event.duration,
        "confidence": event.confidence,
        "max_confidence": event.max_confidence,
        "window_count": event.window_count,
        "sample_rate": analysis["sample_rate"],
        "mean_rms": analysis["mean_rms"],
        "peak_rms": analysis["peak_rms"],
        "total_energy": analysis["total_energy"],
        "time_to_peak_s": analysis["time_to_peak"],
    }
    for name, values in spectral.items():
        row[f"{name}_mean"] = values["mean"]
        row[f"{name}_std"] = values["std"]
    return row


def plot_inhalation_distributions(events: pd.DataFrame):
    """Plot descriptive distributions without assigning quality categories."""
    import matplotlib.pyplot as plt

    if events.empty:
        raise ValueError("Cannot plot distributions because no Inhale events were detected")

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    histograms = [
        ("duration_s", "Inhale duration", "Duration (s)"),
        ("mean_rms", "Mean RMS", "RMS amplitude"),
        ("peak_rms", "Peak RMS", "RMS amplitude"),
        ("total_energy", "Total energy", "Amplitude²·s"),
    ]
    for axis, (column, title, xlabel) in zip(axes.flat[:4], histograms):
        axis.hist(events[column], bins="auto", color="#4c78a8", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Event count")

    scatter_specs = [
        ("mean_rms", "Mean RMS", "Duration vs mean RMS"),
        ("total_energy", "Total energy (amplitude²·s)", "Duration vs total energy"),
    ]
    for axis, (column, ylabel, title) in zip(axes.flat[4:], scatter_specs):
        points = axis.scatter(
            events["duration_s"], events[column], c=events["confidence"],
            cmap="viridis", alpha=0.75, edgecolors="none",
        )
        axis.set_title(title)
        axis.set_xlabel("Duration (s)")
        axis.set_ylabel(ylabel)
        figure.colorbar(points, ax=axis, label="CNN event confidence")
    return figure, axes


def run_dataset_exploration(
    data_dir: str | Path = config.DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    grouping: TemporalGroupingConfig = TemporalGroupingConfig(),
    export_segments: bool = False,
    make_plots: bool = True,
    limit: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Analyze every WAV recording and save a CSV plus diagnostic summaries.

    ``export_segments`` is opt-in because it can create many WAV files.  Each
    exported segment is always sliced from the original waveform.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    recordings = sorted(data_path.glob("*.wav"))
    if limit is not None:
        recordings = recordings[:limit]
    if not recordings:
        raise FileNotFoundError(f"No WAV recordings found in {data_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    segments_path = output_path / "inhale_segments"
    if export_segments:
        segments_path.mkdir(exist_ok=True)

    classifier = OnnxEventClassifier(model_path=model_path)
    rows: list[dict] = []
    failures: list[dict] = []
    for recording_id, recording in enumerate(recordings):
        try:
            events = detect_events(recording, model=classifier, grouping=grouping)
            for event_id, event in enumerate(events):
                analysis = analyze_inhalation(recording, event)
                rows.append(_event_row(recording_id, event_id, recording, event, analysis))
                if export_segments:
                    import soundfile as sf

                    segment, sr = extract_event_audio(recording, event)
                    sf.write(segments_path / f"{recording.stem}_inhale_{event_id:02d}.wav", segment, sr)
        except Exception as exc:  # keep one damaged recording from discarding the study run
            failures.append({"recording_file": recording.name, "error": str(exc)})
        if (recording_id + 1) % 25 == 0 or recording_id + 1 == len(recordings):
            print(f"Processed {recording_id + 1}/{len(recordings)} recordings; events={len(rows)}")

    events_df = pd.DataFrame(rows)
    csv_path = output_path / "inhalation_events.csv"
    events_df.to_csv(csv_path, index=False)

    manifest = {
        "recordings_requested": len(recordings),
        "recordings_failed": len(failures),
        "inhale_events": len(events_df),
        "model_path": str(Path(model_path)),
        "grouping": asdict(grouping),
        "csv": csv_path.name,
        "failures": failures,
    }
    with (output_path / "inhalation_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if make_plots and not events_df.empty:
        figure, _ = plot_inhalation_distributions(events_df)
        figure.savefig(output_path / "inhalation_distributions.png", dpi=160)

    return events_df, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore all CNN-detected inhalation events")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--export-segments", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N recordings")
    args = parser.parse_args()

    dataframe, errors = run_dataset_exploration(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        export_segments=args.export_segments,
        make_plots=not args.no_plots,
        limit=args.limit,
    )
    print(f"Saved {len(dataframe)} inhale events. Failed recordings: {len(errors)}")
