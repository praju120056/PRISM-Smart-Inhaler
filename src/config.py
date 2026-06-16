"""
config.py
---------
All paths and hyperparameters for the pMDI inhaler pipeline.
Edit this file to change dataset location, model settings, etc.
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Directory layout ───────────────────────────────────────────
# src/ is one level inside the project root
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SRC_DIR)

_LOCAL_DATA_DIR   = os.path.join(ROOT_DIR, "data")
_LEGACY_DATA_DIR  = r"D:\project\dataset\pmdi_inhaler_dataset\data"

DATA_DIR          = os.environ.get(
    "PRISM_DATA_DIR",
    _LOCAL_DATA_DIR if os.path.exists(_LOCAL_DATA_DIR) else _LEGACY_DATA_DIR,
)
PRECOMPUTED_DIR   = os.path.join(DATA_DIR, "precomputed")   # per-recording CSV subdirs
RESULTS_DIR       = os.path.join(ROOT_DIR, "results")
ANNOTATION_CSV    = os.path.join(DATA_DIR, "annotation.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Frame / hop settings (precomputed CSV path — loader.py) ──
# Confirmed from dataset: 8 kHz recordings, hop ≈ 129 samples
EXPECTED_HOP  = 129
HOP_TOLERANCE = 20      # flag and skip if inferred hop differs by more than this

# ── Label definitions ──────────────────────────────────────────
# Order matters — kept fixed so LabelEncoder is deterministic
LABEL_NAMES = ["Drug", "Exhale", "Inhale", "Noise"]

# ── Librosa extraction parameters (librosa_extractor.py) ──────
# These MUST match the constants in librosa_extractor.py
LIBROSA_SR          = 8000
LIBROSA_N_MFCC      = 40
LIBROSA_N_FFT       = 256
LIBROSA_HOP_LENGTH  = 64
LIBROSA_N_FEATURES  = 124   # [MFCC(40)|dMFCC(40)|ddMFCC(40)|centroid|flatness|rolloff|zcr]

# ── Windowed dataset parameters (feature_extractor.py) ────────
WINDOW_SIZE     = 7     # frames per window (56 ms at HOP=64, 8 kHz)
WINDOW_STRIDE   = 2     # step between successive windows  [was 1 → 2× faster]
NOISE_BUFFER    = 20    # Noise frames to keep around each event (trim_to_events)
DRUG_MULTIPLIER = 3.0   # Drug target = multiplier × Inhale count
NORMALIZE       = None  # normalization inside build_windowed_dataset (None/global/per_window)
                         # NOTE: StandardScaler is applied per-fold inside train.py

# ── Feature names — precomputed CSV path (loader.py, 40 features) ──
N_MFCC = 19
FEATURE_NAMES = (
    [f"mfcc_{i}"       for i in range(N_MFCC)] +
    ["zcr", "rms"]                               +
    [f"delta_mfcc_{i}" for i in range(N_MFCC)]
)   # total: 40 features per frame

# ── Feature names — librosa windowed path (visualize.py) ──────
# One window = WINDOW_SIZE frames × LIBROSA_N_FEATURES features
_FRAME_NAMES = (
    [f"mfcc_{i}"         for i in range(LIBROSA_N_MFCC)] +
    [f"dmfcc_{i}"        for i in range(LIBROSA_N_MFCC)] +
    [f"ddmfcc_{i}"       for i in range(LIBROSA_N_MFCC)] +
    ["spectral_centroid", "spectral_flatness", "spectral_rolloff", "zcr"]
)   # 124 per frame
LIBROSA_FEATURE_NAMES = [
    f"{name}_f{fi}"
    for fi in range(WINDOW_SIZE)
    for name in _FRAME_NAMES
]   # 7 * 124 = 868 per window

# ── Cross-validation ───────────────────────────────────────────
N_SPLITS = 5            # GroupKFold — recording-level splits

# ── Random Forest hyperparameters ─────────────────────────────
RF_PARAMS = dict(
    n_estimators  = 100,       # was 200 — halves training time, ~same accuracy
    max_depth     = 30,        # caps tree depth — faster, less overfit
    max_features  = "sqrt",
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,
)

# XGBoost hyperparameters
# Requires: pip install xgboost
XGB_PARAMS = dict(
    objective        = "multi:softprob",
    eval_metric      = "mlogloss",
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.85,
    colsample_bytree = 0.85,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    tree_method      = "hist",
)

# ── SVM hyperparameters ────────────────────────────────────────
# Using LinearSVC (O(n) vs rbf's O(n²)) — much faster at 868 features
SVM_PARAMS = dict(
    C            = 0.1,
    class_weight = "balanced",
    max_iter     = 2000,
    dual         = False,      # faster when n_samples >> n_features
    random_state = 42,
)
SVM_SUBSAMPLE = 20_000  # can afford more samples with linear kernel
