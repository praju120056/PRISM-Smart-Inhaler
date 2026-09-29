"""Temporal inhalation-event detection and descriptive analysis.

This module sits *after* PRISM's existing four-class event detector.  It does
not retrain, replace, or interpret the CNN as a technique-quality classifier.
Its responsibilities are deliberately narrow:

``audio -> existing CNN -> window predictions -> grouped InhaleEvent -> measurements``

The grouping defaults are intentionally conservative: no label smoothing, no
confidence rejection, no extra gap allowance, and no duration rejection.  Set
the explicit :class:`TemporalGroupingConfig` values after inspecting diagnostic
plots for the recording population.  None of these parameters are clinical or
technique-quality thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import config
from feature_extractor import create_feature_windows
from librosa_extractor import extract_features_from_audio, load_audio


DEFAULT_MODEL_PATH = Path(config.RESULTS_DIR) / "inhaler_cnn.onnx"


@dataclass(frozen=True)
class WindowPrediction:
    """One overlapping CNN inference window, expressed in seconds."""

    index: int
    start: float
    end: float
    label: str
    confidence: float
    probabilities: Mapping[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InhaleEvent:
    """A temporally grouped inhalation candidate from window predictions."""

    label: str
    start: float
    end: float
    duration: float
    confidence: float
    max_confidence: float
    window_count: int
    window_indices: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TemporalGroupingConfig:
    """Explicit, non-clinical controls for post-classification grouping.

    ``max_gap_s`` is the *additional* time allowed between non-overlapping
    target windows. Overlapping windows naturally form one candidate, which is
    important because the deployed model uses 200 ms windows every 16 ms.
    ``min_event_duration_s=0`` leaves all candidates intact by default; use it
    only as an experimentally justified cleanup setting.
    """

    target_label: str = "Inhale"
    smoothing_window: int = 1
    max_gap_s: float = 0.0
    min_event_duration_s: float = 0.0
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.max_gap_s < 0:
            raise ValueError("max_gap_s must be >= 0")
        if self.min_event_duration_s < 0:
            raise ValueError("min_event_duration_s must be >= 0")
        if self.min_confidence is not None and not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")


class OnnxEventClassifier:
    """Thin, validated adapter for the existing CNN ONNX artifact."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        class_names: Sequence[str] = tuple(config.LABEL_NAMES),
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for CNN inference") from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.class_names = tuple(class_names)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one ONNX input and one output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.input_shape, self.output_shape = inputs[0].shape, outputs[0].shape

        if len(self.input_shape) != 3 or self.input_shape[2] != config.LIBROSA_N_FEATURES:
            raise ValueError(
                "Unsupported ONNX input shape "
                f"{self.input_shape}; expected (batch, frames, {config.LIBROSA_N_FEATURES})"
            )
        if not isinstance(self.input_shape[1], int):
            raise ValueError(
                f"ONNX model must expose a fixed frame count, got {self.input_shape}"
            )
        if len(self.output_shape) != 2 or self.output_shape[1] != len(self.class_names):
            raise ValueError(
                "ONNX output shape/class mapping mismatch: "
                f"shape={self.output_shape}, labels={self.class_names}"
            )
        self.window_frames = int(self.input_shape[1])

    def predict_probabilities(self, windows: np.ndarray) -> np.ndarray:
        """Run the model and convert its verified raw-logit output to softmax."""
        expected = (self.window_frames, config.LIBROSA_N_FEATURES)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"Expected windows shaped (n, {expected[0]}, {expected[1]}), "
                f"got {windows.shape}"
            )
        logits = self.session.run(
            [self.output_name], {self.input_name: windows.astype(np.float32, copy=False)}
        )[0]
        logits = np.asarray(logits, dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


def _as_mono(audio: np.ndarray) -> np.ndarray:
    """Normalise caller-provided audio arrays to a finite mono float32 vector."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1 if values.shape[0] >= values.shape[1] else 0)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio must be a non-empty mono vector or a 2-D channel array")
    if not np.isfinite(values).all():
        raise ValueError("audio contains NaN or infinite values")
    return values


def _load_input_audio(
    audio: str | Path | np.ndarray,
    sample_rate: int | None,
) -> tuple[np.ndarray, int]:
    if isinstance(audio, (str, Path)):
        values, sr = load_audio(str(audio))
        if values is None or sr is None:
            raise ValueError(f"Could not load audio: {audio}")
        return values, sr
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("sample_rate is required and must be positive for audio arrays")
    return _as_mono(audio), int(sample_rate)


def generate_window_predictions(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    model: OnnxEventClassifier | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    window_stride_frames: int = config.WINDOW_STRIDE,
) -> list[WindowPrediction]:
    """Return the existing CNN's chronological prediction stream.

    The original waveform is only used for loading and timing. Features are
    extracted through ``librosa_extractor.extract_features_from_audio`` with the
    exact trained 8 kHz / 124-feature DSP configuration.
    """
    if window_stride_frames < 1:
        raise ValueError("window_stride_frames must be >= 1")
    waveform, source_sr = _load_input_audio(audio, sample_rate)
    features = extract_features_from_audio(waveform, source_sr)
    if features is None:
        raise ValueError("Feature extraction failed for the supplied audio")
    classifier = model or OnnxEventClassifier(model_path=model_path)
    if features.shape[0] < classifier.window_frames:
        return []

    windows = create_feature_windows(
        features, window_size=classifier.window_frames, stride=window_stride_frames
    )
    probabilities = classifier.predict_probabilities(windows)
    label_indices = probabilities.argmax(axis=1)
    window_duration = classifier.window_frames * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR
    stride_duration = window_stride_frames * config.LIBROSA_HOP_LENGTH / config.LIBROSA_SR
    recording_duration = len(waveform) / source_sr

    predictions: list[WindowPrediction] = []
    for index, (class_index, probs) in enumerate(zip(label_indices, probabilities)):
        start = index * stride_duration
        label = classifier.class_names[int(class_index)]
        probabilities_by_label = {
            name: float(probability) for name, probability in zip(classifier.class_names, probs)
        }
        predictions.append(
            WindowPrediction(
                index=index,
                start=float(start),
                end=float(min(start + window_duration, recording_duration)),
                label=label,
                confidence=probabilities_by_label[label],
                probabilities=probabilities_by_label,
            )
        )
    return predictions


def smooth_window_labels(
    predictions: Sequence[WindowPrediction],
    window: int = 1,
) -> list[WindowPrediction]:
    """Optionally smooth labels with a centred majority vote.

    Probabilities remain untouched.  Exact ties retain the original centre
    label, avoiding a hidden preference for any event class.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1:
        return list(predictions)
    radius = window // 2
    smoothed: list[WindowPrediction] = []
    for i, prediction in enumerate(predictions):
        nearby = [item.label for item in predictions[max(0, i - radius): i + radius + 1]]
        counts = Counter(nearby)
        most_common = max(counts.values())
        labels = [label for label, count in counts.items() if count == most_common]
        label = prediction.label if prediction.label in labels else sorted(labels)[0]
        smoothed.append(replace(prediction, label=label, confidence=prediction.probabilities[label]))
    return smoothed


def group_inhale_events(
    predictions: Sequence[WindowPrediction],
    grouping: TemporalGroupingConfig = TemporalGroupingConfig(),
) -> list[InhaleEvent]:
    """Turn target-labelled windows into coherent, configurable candidates."""
    stream = smooth_window_labels(predictions, grouping.smoothing_window)
    selected = [
        item for item in stream
        if item.label == grouping.target_label
        and (grouping.min_confidence is None or item.confidence >= grouping.min_confidence)
    ]
    if not selected:
        return []

    events: list[InhaleEvent] = []
    current: list[WindowPrediction] = [selected[0]]

    def finish(items: Sequence[WindowPrediction]) -> None:
        start = min(item.start for item in items)
        end = max(item.end for item in items)
        duration = end - start
        if duration < grouping.min_event_duration_s:
            return
        confidences = [item.probabilities[grouping.target_label] for item in items]
        events.append(
            InhaleEvent(
                label=grouping.target_label,
                start=float(start),
                end=float(end),
                duration=float(duration),
                confidence=float(np.mean(confidences)),
                max_confidence=float(np.max(confidences)),
                window_count=len(items),
                window_indices=tuple(item.index for item in items),
            )
        )

    for candidate in selected[1:]:
        current_end = max(item.end for item in current)
        if candidate.start - current_end <= grouping.max_gap_s:
            current.append(candidate)
        else:
            finish(current)
            current = [candidate]
    finish(current)
    return events


def detect_events(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    model: OnnxEventClassifier | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    grouping: TemporalGroupingConfig = TemporalGroupingConfig(),
    window_stride_frames: int = config.WINDOW_STRIDE,
    return_predictions: bool = False,
) -> list[InhaleEvent] | tuple[list[InhaleEvent], list[WindowPrediction]]:
    """Detect inhalation candidates from a WAV path or an in-memory waveform.

    Set ``return_predictions=True`` when a diagnostic plot or notebook needs
    the full raw window stream in addition to the grouped events.
    """
    predictions = generate_window_predictions(
        audio, sample_rate=sample_rate, model=model, model_path=model_path,
        window_stride_frames=window_stride_frames,
    )
    events = group_inhale_events(predictions, grouping)
    return (events, predictions) if return_predictions else events


def extract_event_audio(
    audio: str | Path | np.ndarray,
    event: InhaleEvent,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Slice an event from the caller's original waveform, not MFCC features."""
    waveform, sr = _load_input_audio(audio, sample_rate)
    start = max(0, int(np.floor(event.start * sr)))
    end = min(len(waveform), int(np.ceil(event.end * sr)))
    return waveform[start:end].copy(), sr


def _rms_envelope(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple causal RMS envelope and its sample-relative timestamps."""
    if audio.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    frame_length = min(config.LIBROSA_N_FFT, len(audio))
    hop = max(1, min(config.LIBROSA_HOP_LENGTH, frame_length))
    starts = np.arange(0, len(audio), hop)
    envelope = np.array(
        [np.sqrt(np.mean(np.square(audio[start:start + frame_length]))) for start in starts],
        dtype=np.float32,
    )
    return starts.astype(np.float32) / sample_rate, envelope


def analyze_inhalation(
    audio: str | Path | np.ndarray,
    event: InhaleEvent,
    sample_rate: int | None = None,
) -> dict:
    """Return descriptive temporal, energy, and spectral measurements.

    Values are observations for later inspection—not good/bad technique labels
    and not clinical conclusions.  Energy is the time integral of squared
    waveform amplitude (amplitude-squared seconds).
    """
    segment, source_sr = extract_event_audio(audio, event, sample_rate)
    if segment.size == 0:
        raise ValueError("Event does not overlap the supplied audio")
    envelope_times, envelope = _rms_envelope(segment, source_sr)
    peak_index = int(np.argmax(envelope))
    features = extract_features_from_audio(segment, source_sr)
    if features is None:
        raise ValueError("Feature extraction failed for inhalation segment")

    mfcc = features[:, :config.LIBROSA_N_MFCC]
    spectral = features[:, 3 * config.LIBROSA_N_MFCC:]
    spectral_names = ("spectral_centroid", "spectral_flatness", "spectral_rolloff", "zcr")
    spectral_stats = {
        name: {"mean": float(values.mean()), "std": float(values.std())}
        for name, values in zip(spectral_names, spectral.T)
    }
    return {
        "label": event.label,
        "start": event.start,
        "end": event.end,
        "duration": len(segment) / source_sr,
        "sample_rate": source_sr,
        "mean_rms": float(envelope.mean()),
        "peak_rms": float(envelope[peak_index]),
        "total_energy": float(np.sum(np.square(segment)) / source_sr),
        "time_to_peak": float(envelope_times[peak_index]),
        "rms_envelope_time": envelope_times.tolist(),
        "rms_envelope": envelope.tolist(),
        "mfcc_mean": mfcc.mean(axis=0).tolist(),
        "mfcc_std": mfcc.std(axis=0).tolist(),
        "spectral": spectral_stats,
    }


def plot_diagnostic(
    audio: str | Path | np.ndarray,
    predictions: Sequence[WindowPrediction],
    events: Sequence[InhaleEvent],
    sample_rate: int | None = None,
):
    """Build a waveform, prediction-stream, and RMS diagnostic figure."""
    import matplotlib.pyplot as plt

    waveform, sr = _load_input_audio(audio, sample_rate)
    times = np.arange(len(waveform)) / sr
    envelope_times, envelope = _rms_envelope(waveform, sr)
    labels = tuple(config.LABEL_NAMES)
    colour = {"Drug": "#d62728", "Exhale": "#1f77b4", "Inhale": "#2ca02c", "Noise": "#7f7f7f"}
    label_index = {label: index for index, label in enumerate(labels)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, layout="constrained")
    axes[0].plot(times, waveform, color="#303030", linewidth=0.55)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("PRISM event-detection diagnostic")

    for prediction in predictions:
        axes[1].broken_barh(
            [(prediction.start, prediction.end - prediction.start)],
            (label_index[prediction.label] - 0.38, 0.76),
            facecolors=colour.get(prediction.label, "#000000"), alpha=0.75,
        )
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_ylabel("CNN label")
    axes[1].set_ylim(-0.6, len(labels) - 0.4)

    axes[2].plot(envelope_times, envelope, color="#9467bd", linewidth=1.0)
    axes[2].set_ylabel("RMS")
    axes[2].set_xlabel("Time (s)")

    for event in events:
        for axis in (axes[0], axes[1], axes[2]):
            axis.axvspan(event.start, event.end, color="#2ca02c", alpha=0.18)
    return fig, axes


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run post-event inhalation detection")
    parser.add_argument("audio", help="Input WAV file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="CNN ONNX path")
    parser.add_argument("--plot", help="Optional destination PNG for a diagnostic plot")
    args = parser.parse_args()

    detected, stream = detect_events(
        args.audio, model_path=args.model, return_predictions=True
    )
    print(json.dumps([event.to_dict() for event in detected], indent=2))
    if args.plot:
        figure, _ = plot_diagnostic(args.audio, stream, detected)
        figure.savefig(args.plot, dpi=160)
        print(f"Saved diagnostic plot: {args.plot}")
