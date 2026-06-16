"""
librosa_extractor.py
--------------------
Fresh feature extraction from raw WAV files using librosa.
Replaces the precomputed CSV approach with fully tunable parameters.

Feature vector per frame (124 features total):
    MFCC             n_mfcc = 40
    Delta MFCC       n_mfcc = 40
    Delta-Delta MFCC n_mfcc = 40
    Spectral Centroid      1   (normalised to [0, 1])
    Spectral Flatness      1
    Spectral Rolloff       1   (normalised to [0, 1])
    Zero Crossing Rate     1
    ─────────────────────────
    Total                124

Public API (mirrors loader.py interface):
    load_recording_librosa(wav_path, ann)  -> X_rec, y_rec, reason
    load_all_recordings_librosa(ann)       -> all_X, all_y, all_groups, skipped

Caching:
    Extracted feature arrays are saved as .npy files under
    data/extracted/<recording_base>/features.npy
    so re-runs skip re-extraction.  Delete the file to force refresh.

Run standalone:
    python src/librosa_extractor.py
"""

import os
import sys
import re
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Extraction parameters ──────────────────────────────────────────────────────
SR           = 8000     # target sample rate; recordings resampled if needed
N_MFCC       = 40       # MFCC coefficients (vs 19 in precomputed CSVs)
N_FFT        = 256      # FFT window  (~32 ms at 8 kHz; sharper time resolution)
HOP_LENGTH   = 64       # frame hop   (~8 ms at 8 kHz; 2x finer than precomputed)
N_MELS       = 128      # mel filterbank resolution
FMIN         = 50.0     # lowest frequency (Hz)
FMAX         = 4000.0   # Nyquist for 8 kHz recordings
DELTA_WIDTH  = 9        # HTK-style regression filter width for delta/delta-delta
ROLLOFF_PERC = 0.85     # spectral rolloff threshold

# ── Feature inventory ─────────────────────────────────────────────────────────
FEATURE_NAMES = (
    [f"mfcc_{i}"   for i in range(N_MFCC)] +
    [f"dmfcc_{i}"  for i in range(N_MFCC)] +
    [f"ddmfcc_{i}" for i in range(N_MFCC)] +
    ["spectral_centroid", "spectral_flatness", "spectral_rolloff", "zcr"]
)
N_FEATURES = len(FEATURE_NAMES)   # 124

# ── Paths (resolved via config so DATA_DIR is always the authoritative source) ──
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

# Import config to get the correct DATA_DIR (handles local vs. legacy dataset path)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
try:
    import config as _config
    _DATA_DIR  = _config.DATA_DIR
    _CACHE_DIR = os.path.join(_DATA_DIR, "extracted")
except Exception:
    # Fallback for very early / standalone import before config is available
    _DATA_DIR  = os.path.join(_ROOT_DIR, "data")
    _CACHE_DIR = os.path.join(_DATA_DIR, "extracted")


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(wav_path: str) -> np.ndarray | None:
    """
    Extract a (n_frames, N_FEATURES) float32 matrix from a WAV file.

    Returns None on any load or computation failure.

    Frame count:
        n_frames = ceil( total_samples / HOP_LENGTH )

    Spectral centroid and rolloff are normalised by SR/2 (Nyquist)
    so all features share a comparable numerical range before any
    external z-score normalisation.
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required: pip install librosa soundfile")

    try:
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as exc:
        return None

    # Shared magnitude spectrogram (avoids computing STFT twice)
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))

    # MFCC and delta coefficients
    mfcc   = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )                                            # (N_MFCC, n_frames)
    delta  = librosa.feature.delta(mfcc, width=DELTA_WIDTH)
    delta2 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=2)

    # Spectral features — each (1, n_frames)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr) / (SR / 2)
    flatness = librosa.feature.spectral_flatness(S=S)
    rolloff  = librosa.feature.spectral_rolloff(
        S=S, sr=sr, roll_percent=ROLLOFF_PERC
    ) / (SR / 2)
    zcr      = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)

    # Align to MFCC frame count (STFT may produce n+1 frames in edge cases)
    n = mfcc.shape[1]

    def _align(arr: np.ndarray) -> np.ndarray:
        return arr[:, :n]

    features = np.vstack([
        _align(mfcc),
        _align(delta),
        _align(delta2),
        _align(centroid),
        _align(flatness),
        _align(rolloff),
        _align(zcr),
    ]).T                                         # (n_frames, N_FEATURES)

    if not np.isfinite(features).all():
        return None

    return features.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(wav_path: str) -> str:
    """Return the .npy cache path for a given WAV file."""
    wav_name  = os.path.basename(wav_path)
    base      = re.sub(r"\.\d+s\.wav$", "", wav_name)
    cache_dir = os.path.join(_CACHE_DIR, base)
    return os.path.join(cache_dir, "features.npy")


def _load_cached(wav_path: str) -> np.ndarray | None:
    """Return cached features array or None if not found."""
    path = _cache_path(wav_path)
    if os.path.exists(path):
        return np.load(path)
    return None


def _save_cache(wav_path: str, features: np.ndarray) -> None:
    """Save features array to cache."""
    path = _cache_path(wav_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, features)


def _invalidate_cache(wav_path: str) -> None:
    """Delete the cached features for a recording (force re-extraction)."""
    path = _cache_path(wav_path)
    if os.path.exists(path):
        os.remove(path)


# ──────────────────────────────────────────────────────────────────────────────
# Label alignment
# ──────────────────────────────────────────────────────────────────────────────

def _align_labels(
    wav_name:  str,
    ann,                   # pd.DataFrame
    n_frames:  int,
) -> np.ndarray:
    """
    Map sample-level annotation ranges to frame indices using HOP_LENGTH.

    Default label is "Noise" (matches loader.py behaviour).
    """
    import pandas as pd
    y = np.full(n_frames, "Noise", dtype=object)
    ann_rec = ann[ann["filename"] == wav_name]
    for _, row in ann_rec.iterrows():
        sf = max(0, min(int(row["start_sample"] / HOP_LENGTH), n_frames - 1))
        ef = max(0, min(int(row["end_sample"]   / HOP_LENGTH), n_frames))
        y[sf:ef] = row["label"]
    return y


# ──────────────────────────────────────────────────────────────────────────────
# Public API  (same interface as loader.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_recording_librosa(
    wav_path:    str,
    ann,                        # pd.DataFrame from load_annotation()
    use_cache:   bool = True,
) -> tuple:
    """
    Extract features and labels for one recording.

    Parameters
    ----------
    wav_path  : full path to .wav file
    ann       : annotation DataFrame (all recordings)
    use_cache : if True, load from / save to data/extracted/ cache

    Returns
    -------
    X_rec  : np.ndarray (n_frames, N_FEATURES)  or None on failure
    y_rec  : np.ndarray (n_frames,)  string labels  or None on failure
    reason : str | None  skip reason if failed, else None
    """
    wav_name = os.path.basename(wav_path)

    # Try cache first
    features = _load_cached(wav_path) if use_cache else None

    if features is None:
        features = extract_features(wav_path)
        if features is None:
            return None, None, "librosa extraction failed"
        if use_cache:
            _save_cache(wav_path, features)

    y_rec = _align_labels(wav_name, ann, features.shape[0])
    return features, y_rec, None


def load_all_recordings_librosa(
    ann,
    use_cache: bool  = True,
    data_dir:  str   = _DATA_DIR,
) -> tuple:
    """
    Load all WAV recordings in data_dir.

    Returns
    -------
    all_X      : list[np.ndarray]
    all_y      : list[np.ndarray]
    all_groups : list[np.ndarray]
    skipped    : list[tuple(name, reason)]
    """
    wav_files  = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    all_X, all_y, all_groups, skipped = [], [], [], []

    for rec_idx, wav_path in enumerate(wav_files):
        wav_name = os.path.basename(wav_path)
        X_rec, y_rec, reason = load_recording_librosa(wav_path, ann, use_cache)

        if reason is not None:
            skipped.append((wav_name, reason))
            continue

        all_X.append(X_rec)
        all_y.append(y_rec)
        all_groups.append(np.full(X_rec.shape[0], rec_idx, dtype=int))

    return all_X, all_y, all_groups, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Standalone check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, _SRC_DIR)

    print("=" * 60)
    print("librosa_extractor.py -- standalone check")
    print("=" * 60)
    print(f"\nExtraction parameters:")
    print(f"  SR           = {SR} Hz")
    print(f"  N_MFCC       = {N_MFCC}")
    print(f"  N_FFT        = {N_FFT}  (~{1000*N_FFT/SR:.1f} ms)")
    print(f"  HOP_LENGTH   = {HOP_LENGTH}  (~{1000*HOP_LENGTH/SR:.1f} ms)")
    print(f"  N_MELS       = {N_MELS}")
    print(f"  FMIN/FMAX    = {FMIN} / {FMAX} Hz")
    print(f"  N_FEATURES   = {N_FEATURES}")

    # Single recording smoke test
    import glob as _glob
    wav_files = sorted(_glob.glob(os.path.join(_DATA_DIR, "*.wav")))
    if not wav_files:
        print("\nNo WAV files found in data/ -- skipping extraction test.")
        sys.exit(0)

    test_wav = wav_files[0]
    print(f"\nTest recording: {os.path.basename(test_wav)}")
    feats = extract_features(test_wav)
    if feats is None:
        print("  [FAIL] extract_features returned None")
        sys.exit(1)

    print(f"  Feature shape : {feats.shape}")
    print(f"  dtype         : {feats.dtype}")
    print(f"  NaN/Inf       : {not np.isfinite(feats).all()}")
    n_frames = feats.shape[0]
    duration_s = n_frames * HOP_LENGTH / SR
    print(f"  Frames        : {n_frames}  (~{duration_s:.2f} s)")

    # Cache round-trip
    print("\nCache round-trip test...")
    _save_cache(test_wav, feats)
    loaded = _load_cached(test_wav)
    assert loaded is not None and np.allclose(feats, loaded), "Cache mismatch"
    print("  OK")

    # Full dataset load
    print("\nFull dataset load (with cache)...")
    try:
        from loader import load_annotation
        ann = load_annotation()

        all_X, all_y, all_groups, skipped = load_all_recordings_librosa(
            ann, use_cache=True
        )

        total_frames = sum(x.shape[0] for x in all_X)
        print(f"  Loaded  : {len(all_X)} recordings")
        print(f"  Skipped : {len(skipped)}")
        print(f"  Frames  : {total_frames:,}")
        print(f"  Features: {all_X[0].shape[1]}")

        all_y_flat = np.concatenate(all_y)
        unique, counts = np.unique(all_y_flat, return_counts=True)
        print("\n  Label distribution (frame level):")
        for lbl, cnt in zip(unique, counts):
            print(f"    {lbl:10s}: {cnt:7,}  ({100*cnt/total_frames:.1f}%)")

    except Exception as exc:
        print(f"  [WARN] {exc}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
