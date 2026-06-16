# PRISM (Pulmonary Response and Inhaler System Monitor)

## Project Overview

PRISM is a smart inhaler monitoring system designed to evaluate both inhaler adherence and inhalation quality.

Unlike conventional smart inhalers that only record medication usage events, PRISM uses acoustic sensing and signal processing to analyze inhalation behavior and provide personalized feedback.

The system is intended to:

* Monitor inhaler usage events
* Analyze inhalation quality
* Detect deviations from a user's normal inhalation pattern
* Provide real-time feedback through a mobile application
* Maintain longitudinal inhalation and adherence analytics
* Support future clinician-facing dashboards

The architecture follows a mobile-first edge intelligence approach where all signal processing and machine learning inference are performed locally on the smartphone.

---

# Core Design Principles

## Mobile-First Intelligence

The smartphone is the primary compute platform.

All computationally intensive tasks should execute on-device:

* Signal processing
* Feature extraction
* Baseline modeling
* Machine learning inference
* Session analytics

The ESP32 should remain lightweight and focused on sensing, event detection, and communication.

---

## Privacy First

Raw inhalation audio should never leave the user's smartphone.

Cloud synchronization should only store:

* Session summaries
* Quality scores
* Adherence metrics
* Derived analytics

No cloud inference should be required.

---

## Personalized Monitoring

PRISM should not rely solely on population-level machine learning models.

Each user maintains a personalized inhalation baseline.

Future inhalations are evaluated using:

1. Global classification model
2. Personalized baseline deviation analysis

The final assessment should consider both.

---

# High Level Architecture

```text
INMP441 Microphone
        │
        ▼
ESP32 Firmware
        │
        ▼
Bluetooth Low Energy (BLE)
        │
        ▼
React Native Mobile Application
        │
        ├── DSP Pipeline
        ├── Feature Extraction
        ├── Baseline Engine
        ├── ML Inference
        └── Session Analytics
        │
        ▼
Firebase Cloud Services
        │
        ▼
Doctor Dashboard
```

---

# Hardware Architecture

## Required Components

* ESP32-WROOM
* INMP441 I2S Digital Microphone
* Li-Ion Battery
* TP4056 Charging Circuit

## Optional Components

* MPU6050 IMU
* Actuation Detection Switch
* Temperature / Humidity Sensor

---

# ESP32 Responsibilities

The ESP32 should remain lightweight.

Responsibilities:

* Audio acquisition
* Basic signal conditioning
* Inhalation start detection
* Inhalation end detection
* IMU acquisition
* Timestamp generation
* BLE transmission
* Power management

The ESP32 should NOT perform:

* MFCC extraction
* Spectrogram generation
* XGBoost inference
* ONNX inference
* Cloud communication

---

# Audio Acquisition Strategy

The microphone is positioned near the beginning of the inhaler airflow path.

Future enclosure designs may use a side acoustic chamber to reduce direct airflow impact and wind noise.

The system should record inhalation events rather than continuously streaming audio.

Expected inhalation duration:

* Typical: 1–3 seconds
* Maximum buffer target: 5–7 seconds

---

# BLE Transmission Strategy

The ESP32 should not continuously stream microphone data.

Workflow:

```text
Detect Inhalation Start
        ↓
Begin Local Buffering
        ↓
Record Inhalation Segment
        ↓
Detect Inhalation End
        ↓
Package Event
        ↓
Transmit Event via BLE
```

Typical event packet:

```json
{
  "timestamp": "...",
  "duration": "...",
  "audio_segment": "...",
  "imu_data": "...",
  "actuation_state": "..."
}
```

Only active inhalation events should be transmitted.

---

# Mobile Application Responsibilities

The mobile application acts as the intelligence layer of the system.

Responsibilities:

* BLE communication
* Data buffering
* Signal processing
* Noise reduction
* Segmentation
* Feature extraction
* Baseline maintenance
* Machine learning inference
* Feedback generation
* Local storage
* Cloud synchronization

---

# Planned DSP Features

Potential features include:

* MFCC
* Delta MFCC
* Delta-Delta MFCC
* RMS Energy
* Zero Crossing Rate
* Spectral Centroid
* Spectral Flatness
* Spectral Rolloff
* Inhalation Duration

Feature definitions may evolve during experimentation.

---

# Machine Learning Architecture

## Current Direction

Model:

* XGBoost

Deployment:

* Exported to ONNX format
* Executed using ONNX Runtime

Target Platforms:

* Android
* iOS

No cloud inference.

All predictions should execute locally on the device.

---

# Personalized Baseline Engine

Each user maintains a historical baseline.

Potential baseline metrics:

* Mean inhalation duration
* Mean RMS energy
* Mean spectral profile
* Historical feature distributions
* Recent session statistics

Evaluation should combine:

```text
Global Model Prediction
+
Baseline Deviation Analysis
=
Final Quality Assessment
```

The system should answer:

* Is this inhalation good?
* Is this inhalation normal for this user?

Baseline updates should only use acceptable-quality sessions to avoid adapting toward poor technique.

---

# Data Flow

## Recording Phase

```text
User Inhales
        ↓
ESP32 Detects Event
        ↓
Audio Buffered Locally
        ↓
Inhalation Ends
        ↓
Event Sent via BLE
```

## Analysis Phase

```text
Audio Segment
        ↓
Signal Processing
        ↓
Feature Extraction
        ↓
ONNX Inference
        ↓
Baseline Comparison
        ↓
Quality Assessment
        ↓
User Feedback
```

---

# Current Dataset Context

Current development is based on a pMDI inhaler audio dataset.

Classes:

* Drug
* Inhale
* Exhale
* Noise

Current experiments focus on frame-level event classification and feature extraction before moving to personalized inhalation quality assessment.

---

# Current Repository Scope

Implemented:

* Dataset processing
* Feature extraction
* Event classification
* Classical ML experimentation
* Cross-validation pipelines
* Feature engineering research

Not Yet Implemented:

* ESP32 firmware
* BLE communication layer
* React Native application
* ONNX deployment
* Personalized baseline engine
* Firebase backend
* Doctor dashboard

---

# Development Priority Order

The project should be developed in the following order:

1. Validate inhalation audio recordings
2. Build DSP pipeline
3. Evaluate feature separability
4. Build baseline engine
5. Train and compare ML models
6. Export ONNX model
7. Build mobile inference pipeline
8. Integrate ESP32 hardware
9. Add cloud synchronization
10. Build clinician-facing analytics

Hardware development should not block DSP, ML, or mobile application development.

---

# Repository Purpose

This repository currently serves as the research and experimentation environment for:

* Audio processing
* Feature extraction
* Event classification
* Model development

The repository should evolve into the reference implementation that informs future firmware, mobile, and cloud development for the complete PRISM system.


# pMDI Inhaler Event Classification Pipeline

## Overview

The pipeline classifies frame-level audio into four events: Inhale, Exhale, Drug, Noise.
Input is a set of ~10-second WAV recordings with sample-level annotations.
Output is a trained classifier (Random Forest or SVM) evaluated with recording-level cross-validation.

---

## Directory Layout

```
pmdi_inhaler_dataset/
    data/
        annotation.csv
        rec<timestamp>.wav          (WAV files; remain here)
        precomputed/                (moved from data/ root by reorganize_precomputed.py)
            rec<timestamp>/
                rec<timestamp>_mfcc.csv
                rec<timestamp>_zcr.csv
                rec<timestamp>_spect.csv
                rec<timestamp>_cwt.csv
                rec<timestamp>_cepst.csv
        extracted/                  (written by librosa_extractor.py on first run)
            rec<timestamp>/
                features.npy        (n_frames, 124) float32 cache
    results/
    src/
        config.py
        loader.py
        dataset.py
        feature_extractor.py
        librosa_extractor.py
        reorganize_precomputed.py
        train.py
        evaluate.py
        visualize.py
        run_pipeline.py
```

---

## Module Descriptions

### config.py

Central configuration. All paths, hyperparameters, and label definitions.

Key constants:

- `DATA_DIR`, `RESULTS_DIR`, `ANNOTATION_CSV`
- `EXPECTED_HOP = 129` samples (8 kHz, hop derived from frame count vs total samples)
- `HOP_TOLERANCE = 20` samples
- `LABEL_NAMES = ["Drug", "Exhale", "Inhale", "Noise"]`
- `N_MFCC = 19`  (precomputed CSV feature count; librosa_extractor uses its own)
- `N_SPLITS = 5` (GroupKFold)

---

### reorganize_precomputed.py

One-time utility. Moves all precomputed CSV files from `data/` into:

```
data/precomputed/<recording_base>/<file>.csv
```

WAV files and `annotation.csv` remain in `data/`. Idempotent; safe to run multiple times.

Run once before using the old CSV-based loader:

```
python src/reorganize_precomputed.py
```

---

### loader.py

Original feature loader based on precomputed CSVs. Still functional after reorganization
if pointed at `data/precomputed/<base>/`. Used by `run_pipeline.py`.

Per-recording feature vector (40 features):

```
[MFCC (19) | ZCR (1) | RMS (1) | delta_MFCC (19)]
```

RMS is computed from raw audio. Delta is computed with `np.diff` + prepend padding.
Label alignment uses `start_sample / hop_length` where `hop_length ~ 129` samples.

---

### librosa_extractor.py

Fresh feature extraction from raw WAV using librosa. Replaces the precomputed CSV
approach with tunable parameters targeting Drug detection specifically.

#### Extraction Parameters

| Parameter   | Value  | Rationale                                         |
|-------------|--------|---------------------------------------------------|
| SR          | 8000   | native recording rate                             |
| N_MFCC      | 40     | more spectral detail vs 19 in precomputed CSVs    |
| N_FFT       | 256    | ~32 ms window; sharper time resolution            |
| HOP_LENGTH  | 64     | ~8 ms hop; 2x finer than precomputed (~16 ms)     |
| N_MELS      | 128    | richer mel filterbank                             |
| FMIN        | 50 Hz  | excludes DC rumble                                |
| FMAX        | 4000   | Nyquist for 8 kHz                                 |
| DELTA_WIDTH | 9      | HTK-style regression filter                       |
| ROLLOFF     | 0.85   | spectral rolloff threshold                        |

#### Feature Vector (124 features per frame)

```
[MFCC (40) | delta_MFCC (40) | delta_delta_MFCC (40) |
 spectral_centroid (1) | spectral_flatness (1) |
 spectral_rolloff (1)  | zcr (1)]
```

Spectral centroid and rolloff are normalised by SR/2 (Nyquist) to [0, 1].

Rationale for spectral features over pure MFCC:

- **Spectral flatness**: Drug actuation (aerosol spray) is broadband noise;
  flatness near 1.0. Breath sounds are more tonal; flatness near 0.
- **Spectral centroid**: aerosol has higher centroid than breath sounds.
- **Spectral rolloff**: complements centroid for distinguishing impulsive events.

#### Delta Computation

Uses `librosa.feature.delta` which implements the HTK regression filter:

```
delta[t] = sum_{n=1}^{N} n * (f[t+n] - f[t-n])
           -------------------------------------
                   2 * sum_{n=1}^{N} n^2
```

where `N = (DELTA_WIDTH - 1) / 2 = 4`.

Delta-delta is computed by applying the same filter to delta:

```
delta2 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=2)
```

#### Label Alignment

Same logic as `loader.py` but using `HOP_LENGTH = 64`:

```
start_frame = floor(start_sample / HOP_LENGTH)
end_frame   = floor(end_sample   / HOP_LENGTH)
y[start_frame : end_frame] = label
```

Default fill is `"Noise"`.

#### Caching

Extracted features are saved as `.npy` files under `data/extracted/`:

```
data/extracted/rec<timestamp>/features.npy
```

On subsequent runs, the file is loaded directly; WAV is not re-read.
Delete the `.npy` file to force re-extraction for a recording.

#### Public API

```python
load_recording_librosa(wav_path, ann, use_cache=True)
    -> X_rec (n_frames, 124), y_rec (n_frames,), reason

load_all_recordings_librosa(ann, use_cache=True, data_dir)
    -> all_X, all_y, all_groups, skipped
```

Interface is identical to `loader.py` so either can be passed to `feature_extractor.py`.

#### Results on Real Dataset (361 recordings, first run)

```
Loaded  : 361 recordings
Skipped : 0
Frames  : 548,785
Features: 124

Label distribution (frame level):
  Drug   :   7,817  (1.4%)
  Exhale :  73,222  (13.3%)
  Inhale :  49,310  (9.0%)
  Noise  : 418,436  (76.2%)
```

Frame count is ~2x the precomputed CSV count (548k vs 262k) due to HOP_LENGTH 64 vs 129.

---

### feature_extractor.py

Converts frame-level MFCC arrays into windowed features with temporal context.
Works with output from either `loader.py` or `librosa_extractor.py`.

#### 1. compute_delta(features, width=9)

HTK regression filter (same formula as librosa, numpy-only fallback):

```
delta[t] = sum_{n=1}^{N} n * (f[t+n] - f[t-n])  /  ( 2 * sum_{n=1}^{N} n^2 )
```

#### 2. stack_features(mfcc, delta_width=9)

```
stacked[t] = [ MFCC[t] | delta[t] | delta_delta[t] ]
```

When fed output from `librosa_extractor.py`, the input already contains delta and
delta-delta, so this step is typically skipped or the extractor is used directly
with `create_windows`.

#### 3. create_windows(features, labels, window_size=7, stride=1)

Sliding window over frame matrix. No cross-recording mixing.

```
x_window = flatten( features[t : t + window_size] )
n_windows = floor( (n_frames - window_size) / stride ) + 1
y_window  = mode( labels[t : t + window_size] )   (majority vote)
```

#### 4. trim_to_events(X_rec, y_rec, noise_label, buffer_frames=20)

Removes Noise frames far from any annotated event:

```
keep[i] = True  iff  exists j : label[j] != Noise  and  |i - j| <= buffer_frames
```

Eliminates 3-4 seconds of irrelevant silence at recording boundaries.

#### 5. oversample_minority(X, y, label, target_n, jitter_std=0.01)

Random duplication with Gaussian jitter (numpy-only SMOTE approximation):

```
X_new[i] = X[ random_choice(Drug_indices) ] + N(0, jitter_std)
```

#### 6. balance_dataset(X, y, noise_cap, drug_multiplier=3.0)

Stage 1 — cap Noise:

```
cap = noise_cap  if set,  else  2 * count(Exhale)
```

Stage 2 — oversample Drug:

```
drug_target = min( drug_multiplier * count(Inhale), count(Exhale) )
```

Result after balancing on real data:

| Class  | Before  | After   |
|--------|---------|---------|
| Drug   | 1.4%    | ~21%    |
| Exhale | 13.3%   | ~21%    |
| Inhale | 9.0%    | ~14%    |
| Noise  | 76.2%   | ~43%    |

#### 7. build_windowed_dataset(all_X, all_y, ...)

Full per-recording pipeline:

```
trim_to_events -> stack_features -> create_windows
    [per recording, no cross-recording mixing]

balance_dataset -> normalize_windows
    [post-concatenation]
```

Key parameters:

| Parameter        | Default | Description                              |
|------------------|---------|------------------------------------------|
| window_size      | 7       | frames per window                        |
| stride           | 1       | hop between successive windows           |
| trim_noise       | True    | remove far-from-event Noise frames       |
| noise_buffer     | 20      | Noise frames to keep around each event   |
| balance          | True    | cap Noise + oversample Drug              |
| drug_multiplier  | 3.0     | Drug target = multiplier x Inhale count  |
| normalize        | None    | global / per_window / prefit / None      |

---

### train.py

GroupKFold (5 splits). Each fold trains Random Forest and SVM.
SVM subsampled to 8000 frames per class due to O(n^2) kernel cost.

---

### evaluate.py / visualize.py / run_pipeline.py

Unchanged. Compute metrics, save plots, orchestrate full pipeline.
`run_pipeline.py` uses `loader.py` by default; swap to `librosa_extractor.py`
by replacing the import and `load_all_recordings` call.

---

## Data Flow

```
annotation.csv + WAV files
        |
        +--[precomputed path]------------------------+
        |  loader.py                                  |
        |  data/precomputed/<base>/_mfcc.csv etc.     |
        |  feature vector: (n_frames, 40)             |
        |                                             |
        +--[librosa path]----------------------------+
           librosa_extractor.py                       |
           extracts from raw WAV, caches to           |
           data/extracted/<base>/features.npy         |
           feature vector: (n_frames, 124)            |
                                                      |
                          v                           |
              (all_X, all_y, all_groups)  <-----------+
                          |
                          v  [feature_extractor.py]
              trim_to_events()
              stack_features()  [if using loader.py output]
              create_windows()  ->  (n_windows, window_size * feature_dim)
                          |
                          v  [post-concatenation]
              balance_dataset()
              normalize_windows()
                          |
                          v
              GroupKFold cross-validation
              Random Forest  /  SVM
                          |
                          v
              results/  (metrics, plots, reports)
```

---

## Comparison: Precomputed CSVs vs Librosa Extraction

| Property            | loader.py (CSV)          | librosa_extractor.py      |
|---------------------|--------------------------|---------------------------|
| N_MFCC              | 19                       | 40                        |
| Hop length          | ~129 samples (~16 ms)    | 64 samples (~8 ms)        |
| Frame count / rec   | ~750                     | ~1500                     |
| Spectral features   | ZCR only                 | centroid, flatness, rolloff, ZCR |
| Total features      | 40                       | 124                       |
| Parameter control   | None (fixed externally)  | Full                      |
| First run           | Fast (CSV read)          | Slow (librosa, ~1 min)    |
| Subsequent runs     | Fast                     | Fast (npy cache)          |
| Drug detection      | Coarse temporal res.     | Fine temporal res.        |

---

## Class Imbalance Problem

Raw recordings are ~10 seconds. Annotated events occupy roughly 6-7 seconds.
The remaining 3-4 seconds are unlabelled background, filled as Noise.

This produces a Noise-to-Drug ratio of ~50:1 at frame level before balancing.

Solution applied in `feature_extractor.py`:
1. `trim_to_events` removes silence far from events.
2. `balance_dataset` caps Noise and oversamples Drug.

---

## Frame and Window Sizing

### Precomputed CSV path (loader.py)

```
hop_length ~ 129 samples
frame_rate = 8000 / 129 ~ 62 fps
window duration (size=7) = 7 / 62 ~ 113 ms
```

### Librosa path (librosa_extractor.py)

```
hop_length = 64 samples
frame_rate = 8000 / 64 = 125 fps
window duration (size=7) = 7 / 125 = 56 ms
```

Finer temporal resolution covers Drug actuation (~300 ms) with ~37 frames
vs ~18 frames in the precomputed path, giving the classifier more granularity.

---

## Constraints

- No deep learning frameworks.
- No cross-recording frame mixing in any window.
- LabelEncoder fitted on fixed `LABEL_NAMES` for deterministic class indices.
- GroupKFold ensures each recording appears in exactly one fold.
- `librosa_extractor.py` and `feature_extractor.py` do not modify any other src files.
