"""
loader.py
---------
Loads annotation + per-recording features (MFCC, ZCR, RMS from audio)
and aligns frame-level labels from sample-level annotations.

Run standalone to inspect what gets loaded / skipped:
    python src/loader.py
"""

import os
import re
import glob
import wave
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load_annotation() -> pd.DataFrame:
    """
    Load annotation.csv (no header) and normalise label capitalisation.

    Returns
    -------
    pd.DataFrame  columns: filename, label, start_sample, end_sample
    """
    ann = pd.read_csv(
        config.ANNOTATION_CSV,
        header=None,
        names=["filename", "label", "start_sample", "end_sample"],
    )
    ann["label"] = ann["label"].str.strip().str.capitalize()
    return ann


def _load_audio(wav_path: str):
    """
    Read a WAV file and return (audio_float32, total_samples, sample_rate).
    Audio is normalised to [-1, 1].  Returns None on failure.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels    = wf.getnchannels()
            sampwidth     = wf.getsampwidth()
            total_samples = wf.getnframes()
            raw_bytes     = wf.readframes(total_samples)
            sample_rate   = wf.getframerate()

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype     = dtype_map.get(sampwidth, np.int16)
        audio     = np.frombuffer(raw_bytes, dtype=dtype)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        audio = audio.astype(np.float32) / np.iinfo(dtype).max
        return audio, total_samples, sample_rate

    except Exception as e:
        return None, None, None


def _compute_rms(audio: np.ndarray, n_frames: int, hop_length: int) -> np.ndarray:
    """
    Compute per-frame RMS energy directly from the raw waveform.
    Aligned to the same hop_length used by the precomputed MFCC frames.
    """
    rms = np.zeros(n_frames, dtype=np.float32)
    for fi in range(n_frames):
        start = fi * hop_length
        end   = min(start + hop_length, len(audio))
        chunk = audio[start:end]
        if len(chunk) > 0:
            rms[fi] = np.sqrt(np.mean(chunk ** 2))
    return rms


def load_recording(wav_path: str, ann: pd.DataFrame):
    """
    Load one recording: features (MFCC + ZCR + RMS) and aligned labels.

    Parameters
    ----------
    wav_path : str   Full path to the .wav file.
    ann      : pd.DataFrame  Full annotation dataframe (all recordings).

    Returns
    -------
    X_rec : np.ndarray  shape (n_frames, 40)  or None on failure
    y_rec : np.ndarray  shape (n_frames,)     string labels
    reason: str | None   Skip reason if failed, else None
    """
    wav_name = os.path.basename(wav_path)
    base     = re.sub(r"\.\d+s\.wav$", "", wav_name)

    # ── locate feature CSVs ──────────────────────────────────────
    # New layout: data/precomputed/{base}/{base}_mfcc.csv
    rec_dir   = os.path.join(config.PRECOMPUTED_DIR, base)
    mfcc_path = os.path.join(rec_dir, f"{base}_mfcc.csv")
    zcr_path  = os.path.join(rec_dir, f"{base}_zcr.csv")

    if not (os.path.exists(mfcc_path) and os.path.exists(zcr_path)):
        return None, None, "missing feature CSVs"

    # ── load raw audio ───────────────────────────────────────────
    audio, total_samples, _ = _load_audio(wav_path)
    if audio is None:
        return None, None, "audio read error"

    # ── load MFCC and ZCR CSVs ───────────────────────────────────
    # CSVs are stored as (n_features, n_frames) — transpose needed
    try:
        mfcc_raw = pd.read_csv(mfcc_path, header=None).values  # (19, n_frames)
        zcr_raw  = pd.read_csv(zcr_path,  header=None).values  # (1,  n_frames)
    except Exception as e:
        return None, None, f"CSV read error: {e}"

    # ── NaN / Inf guard ──────────────────────────────────────────
    if (np.isnan(mfcc_raw).any() or np.isinf(mfcc_raw).any() or
            np.isnan(zcr_raw).any()  or np.isinf(zcr_raw).any()):
        return None, None, "NaN or Inf in feature CSV"

    n_frames = mfcc_raw.shape[1]

    # ── frame-count consistency ──────────────────────────────────
    if zcr_raw.shape[1] != n_frames:
        return None, None, (
            f"frame mismatch: mfcc={n_frames}, zcr={zcr_raw.shape[1]}"
        )

    # ── infer hop length per recording ───────────────────────────
    hop_length = round(total_samples / n_frames)
    if abs(hop_length - config.EXPECTED_HOP) > config.HOP_TOLERANCE:
        return None, None, (
            f"unusual hop={hop_length} "
            f"(samples={total_samples}, frames={n_frames})"
        )

    # ── compute RMS from raw audio (correct approach) ────────────
    rms = _compute_rms(audio, n_frames, hop_length)

    # ── transpose to (n_frames, n_features) ─────────────────────
    mfcc = mfcc_raw.T              # (n_frames, 19)
    zcr  = zcr_raw.T               # (n_frames, 1)
    rms  = rms.reshape(-1, 1)      # (n_frames, 1)

    assert mfcc.shape[0] == zcr.shape[0] == rms.shape[0], \
        "Internal shape mismatch after transpose"

    # ── delta features: MFCC only (ZCR delta is too noisy) ───────
    delta_mfcc = np.diff(mfcc, axis=0, prepend=mfcc[[0]])  # (n_frames, 19)

    # ── stack: [MFCC | ZCR | RMS | delta_MFCC] = 40 features ────
    X_rec = np.hstack([mfcc, zcr, rms, delta_mfcc])        # (n_frames, 40)

    # ── assign labels via sample → frame mapping ─────────────────
    # Default = "Noise" (covers both unlabelled frames and explicit Noise rows)
    y_rec   = np.full(n_frames, "Noise", dtype=object)
    ann_rec = ann[ann["filename"] == wav_name]

    for _, row in ann_rec.iterrows():
        sf = max(0, min(int(row["start_sample"] / hop_length), n_frames - 1))
        ef = max(0, min(int(row["end_sample"]   / hop_length), n_frames))
        y_rec[sf:ef] = row["label"]

    return X_rec, y_rec, None


def load_all_recordings(ann: pd.DataFrame):
    """
    Iterate over every WAV in DATA_DIR, load features + labels.

    Returns
    -------
    all_X      : list[np.ndarray]  per-recording feature matrices
    all_y      : list[np.ndarray]  per-recording label arrays
    all_groups : list[np.ndarray]  per-recording group id (int)
    skipped    : list[tuple]       (wav_name, reason) for skipped files
    """
    wav_files  = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.wav")))
    all_X, all_y, all_groups, skipped = [], [], [], []

    for rec_idx, wav_path in enumerate(wav_files):
        wav_name = os.path.basename(wav_path)
        X_rec, y_rec, reason = load_recording(wav_path, ann)

        if reason is not None:
            skipped.append((wav_name, reason))
            continue

        all_X.append(X_rec)
        all_y.append(y_rec)
        all_groups.append(
            np.full(X_rec.shape[0], rec_idx, dtype=int)
        )

    return all_X, all_y, all_groups, skipped


# ──────────────────────────────────────────────────────────────
# Standalone debug entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("loader.py — standalone check")
    print("=" * 60)

    ann = load_annotation()
    print(f"\nAnnotation: {len(ann)} entries")
    print(ann["label"].value_counts().to_string())

    all_X, all_y, all_groups, skipped = load_all_recordings(ann)

    print(f"\nLoaded : {len(all_X)} recordings")
    print(f"Skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  [SKIP] {name}: {reason}")

    if all_X:
        total_frames = sum(x.shape[0] for x in all_X)
        print(f"\nTotal frames : {total_frames:,}")
        print(f"Feature dim  : {all_X[0].shape[1]}")
        sample_y = np.concatenate(all_y)
        unique, counts = np.unique(sample_y, return_counts=True)
        print("\nLabel distribution:")
        for lbl, cnt in zip(unique, counts):
            print(f"  {lbl:10s}: {cnt:7,}  ({100*cnt/len(sample_y):.1f}%)")
