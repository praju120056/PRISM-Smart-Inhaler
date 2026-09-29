# PRISM - Pulmonary Response and Inhaler System Monitor

> Acoustic smart inhaler monitoring — real-time inhalation quality analysis using
> on-device ML inference. No cloud audio processing. No continuous streaming.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-green.svg)](https://onnxruntime.ai/)

---

## What Is PRISM?

PRISM is an end-to-end smart inhaler platform for pressurised Metered-Dose Inhalers
(pMDI). It records inhalation audio from a custom ESP32-based hardware attachment,
transmits it via Bluetooth Low Energy to a smartphone, and performs on-device ML
inference to classify every 8 ms audio frame into one of four event classes:

| Class | Description |
|---|---|
| **Drug** | Aerosol spray actuation (~300 ms broadband noise burst) |
| **Inhale** | Sustained inward airflow through the inhaler |
| **Exhale** | Patient exhaling prior to inhalation |
| **Noise** | Ambient background, silence, handling sounds |

The resulting event sequence enables per-session assessment of:
- Medication delivery (Drug class detected?)
- Inhalation duration and depth
- Actuator-to-inhalation coordination delay
- Pre-inhalation exhale technique
- Longitudinal deviation from personal baseline

---

## Quick Start (Research Pipeline)

### Prerequisites

```bash
pip install -r requirements.txt
# Key dependencies: librosa, numpy, scikit-learn, xgboost, torch, onnx, onnxruntime
```

### Data Setup

Place the dataset in `data/`:
```
data/
  annotation.csv          # sample-level event labels (no header row)
  rec2018-01-22_*.wav     # WAV recordings (8 kHz, 16-bit, mono)
```

The annotation CSV format (no header):
```
filename,label,start_sample,end_sample
rec2018-01-22_17h41m33.475s.wav,Inhale,512,9216
rec2018-01-22_17h41m33.475s.wav,Drug,4096,6144
```

### Run the Pipeline

```bash
# Full cross-validation: RF + SVM + XGBoost (5-fold GroupKFold)
python src/run_pipeline.py

# 1D CNN with automatic ONNX export
python src/run_pipeline.py --cnn

# Fast smoke test (3-fold, XGBoost only)
python src/run_pipeline.py --fast

# XGBoost only (faster)
python src/run_pipeline.py --xgb-only

# RF + XGBoost (skip slow SVM)
python src/run_pipeline.py --no-svm
```

### Inspect Detected Inhalations

The post-event layer reuses the exported CNN; it does not retrain or make
technique-quality judgments.  It groups its overlapping window predictions
into inhalation candidates and measures the matching original-waveform region.

```bash
python src/post_event.py data/your_recording.wav --plot results/your_recording_diagnostic.png
```

From a notebook or experiment script run from the project root:

```python
import sys
sys.path.append("src")

from post_event import detect_events, analyze_inhalation, plot_diagnostic

events, predictions = detect_events("data/your_recording.wav", return_predictions=True)
analysis = analyze_inhalation("data/your_recording.wav", events[0])
figure, axes = plot_diagnostic("data/your_recording.wav", predictions, events)
```

`TemporalGroupingConfig` exposes optional label smoothing, gap allowance,
minimum duration, and confidence controls.  Defaults apply no extra filtering;
they are temporal-cleanup controls, not validated clinical thresholds.

### Explore All Detected Inhalations

Create an event-level CSV and distribution plots across every recording:

```bash
python src/explore_inhalations.py
```

This writes `results/post_event/inhalation_events.csv`, a run manifest, and
histograms plus duration-versus-RMS/energy scatter plots. To persist each
original-waveform event segment as a WAV file too, add `--export-segments`.

### Finalize the Inhale-Event Dataset (anomaly detection, Stage 1)

```bash
python src/inhale_dataset.py
```

This reads `results/post_event/inhalation_events.csv` without modifying it. It
writes the following to `results/inhale_dataset/`:

- `inhale_events_v1.csv`: every event, plus usability flags, `exclusion_reasons`
  and a chronological `usable_order`
- an audit against `data/annotation.csv`, used only to check the rule
- a rule-sensitivity table
- `dataset_summary.json`

"Usable" means eligible for baseline modeling. It is not a technique-quality
label. See `PRISM_RESEARCH_LOG.md` Entry 3 for the rule and its evidence.

### Output

```
results/
  inhaler_cnn.onnx        # Trained 1D CNN - embed in mobile app
  cv_results.csv          # Per-fold per-class accuracy/precision/recall/F1
  confusion_matrix.png    # Aggregated confusion matrix (CNN or XGBoost)
  class_metrics.png       # Per-class bar chart
  xg/
    summary_report.txt    # XGBoost run summary with feature stats
    feature_importance_xgboost.png
```

---

## Results (Current Model)

**1D CNN, 3-fold GroupKFold cross-validation:**

| Metric | Value |
|---|---|
| Mean Accuracy | **89.0%** |
| Drug F1 | **0.950** |
| Inhale F1 | **0.893** |
| Exhale F1 | **0.867** |
| Noise F1 | **0.871** |

**XGBoost, 5-fold GroupKFold:** 89.0% ± 0.4%

The Drug class achieves the highest F1 (~0.95) due to its spectrally distinctive
broadband aerosol spray signature. Spectral Flatness is the single most discriminative
feature (aerosol flatness ~0.7-1.0 vs breath sounds ~0.1-0.4).

---

## Repository Structure

```
PRISM smart inhaler/
├── ARCHITECTURE.md         # Full technical reference (read this first for integration)
├── README.md               # This file
├── requirements.txt        # Python dependencies
│
├── data/                   # Dataset root (not committed to git)
│   ├── annotation.csv      # Sample-level event annotations
│   ├── *.wav               # Raw audio recordings
│   └── extracted/          # .npy feature cache (auto-generated by librosa_extractor.py)
│
├── results/                # Pipeline outputs
│   ├── inhaler_cnn.onnx    # **Primary deployment artifact**
│   ├── cv_results.csv      # Cross-validation metrics
│   └── xg/                 # XGBoost-specific results
│
└── src/                    # Python source
    ├── config.py           # ALL parameters (sample rate, window size, model config)
    ├── loader.py           # Legacy CSV feature loader
    ├── librosa_extractor.py # Primary feature extractor (124 features, .npy cache)
    ├── feature_extractor.py # Windowing, noise trimming, class balancing
    ├── model_cnn.py        # InhalerCNN architecture + ONNX export
    ├── train.py            # Random Forest / SVM / XGBoost cross-validation
    ├── train_cnn.py        # 1D CNN cross-validation + ONNX export
    ├── evaluate.py         # Metrics aggregation and analysis
    ├── visualize.py        # Confusion matrix and feature importance plots
    └── run_pipeline.py     # Main entry point (orchestrates everything)
```

---

## Architecture Overview

```
ESP32 (microphone + BLE)
    ↓  audio PCM via Bluetooth Low Energy
Smartphone (React Native + Native C++ + ONNX Runtime)
    ↓  derived analytics only (no audio leaves device)
Firebase (Firestore + Auth + Functions)
    ↓
Doctor Dashboard
```

**Full technical details:** See [ARCHITECTURE.md](ARCHITECTURE.md)

The end-to-end pipeline has three distinct phases:

### Phase 1 — Research (This Repository)
Audio → librosa features (124-dim, 8 ms frames) → Sliding window (25 frames = 200 ms) → 1D CNN / XGBoost → ONNX export

### Phase 2 — Mobile Application
BLE audio → Native C++ DSP (identical feature parameters) → ONNX Runtime inference → Session analytics → Firebase sync

### Phase 3 — Hardware
INMP441 I2S microphone → ESP32 DMA → RMS energy detector → BLE binary packet transmission

---

## ML Pipeline Detail

### Feature Extraction (librosa_extractor.py)

124 features per 8 ms frame:

| Feature | Dim | Description |
|---|---|---|
| MFCC | 40 | Mel-frequency cepstral coefficients (n_mels=128, fmin=50 Hz, fmax=4000 Hz) |
| Delta-MFCC | 40 | HTK regression filter (width=9, half-width N=4) |
| Delta²-MFCC | 40 | Second-order HTK regression |
| Spectral Centroid | 1 | Weighted mean frequency / Nyquist → [0,1] |
| Spectral Flatness | 1 | Geometric/arithmetic mean ratio → [0,1] |
| Spectral Rolloff | 1 | 85th percentile frequency / Nyquist → [0,1] |
| ZCR | 1 | Zero crossing rate per frame |
| **Total** | **124** | Per-frame float32 vector |

Signal processing parameters (must match in mobile app):
```python
SR           = 8000   # Hz
N_FFT        = 256    # samples (32 ms window)
HOP_LENGTH   = 64     # samples (8 ms hop; 125 fps)
N_MELS       = 128
FMIN         = 50.0   # Hz
FMAX         = 4000.0 # Hz (Nyquist)
N_MFCC       = 40
DELTA_WIDTH  = 9      # HTK regression filter
ROLLOFF_PERC = 0.85
```

Features are cached as `.npy` files in `data/extracted/<base>/features.npy`.
Delete the cache directory to force re-extraction.

### Windowed Dataset (feature_extractor.py)

```python
WINDOW_SIZE   = 25    # frames = 200 ms
WINDOW_STRIDE = 2     # frames = 16 ms step
NOISE_BUFFER  = 20    # frames retained around labelled events
```

Per-window label = majority vote over 25 frames.
Classical ML: window flattened to `(3100,)` vector.
CNN: window kept as `(25, 124)` tensor.

### Class Balancing

| Class | Before | After |
|---|---|---|
| Drug | 1.4% | 21.4% |
| Exhale | 13.3% | 21.4% |
| Inhale | 9.0% | 14.5% |
| Noise | 76.2% | 42.6% |

Noise capped to `2x count(Exhale)`.
Drug oversampled to `min(3.0 x count(Inhale), count(Exhale))` using Gaussian jitter.

### Cross-Validation

`GroupKFold(k=5)` with groups = recording index (0..360).
All frames from one recording are always in the same fold — no data leakage.

### 1D CNN Architecture (InhalerCNN)

```
Input (N, 25, 124)
Permute -> (N, 124, 25)     [features=channels, frames=length]
Stem:      Conv1d(124->128, k=3) + BN + ReLU
ResBlock1: [Conv1d(128->128) + BN + ReLU + Dropout(0.25)] x2 + skip
ResBlock2: identical
Neck:      Conv1d(128->256, k=1) + BN + ReLU + AdaptiveAvgPool1d(1)
Head:      Linear(256->64) + ReLU + Dropout(0.25) + Linear(64->4)
Output (N, 4)   [raw logits -> apply softmax in app]
~320,000 parameters | ONNX opset 17 | ~1.15 MB
```

### ONNX I/O Contract

```
Input  "features":  float32 (batch, 25, 124)
Output "logits":    float32 (batch, 4)   [RAW LOGITS - apply softmax first]
Class order:        Drug=0, Exhale=1, Inhale=2, Noise=3
```

---

## Configuration

All parameters are in [src/config.py](src/config.py). Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `DATA_DIR` | `data/` | Override with `PRISM_DATA_DIR` env var |
| `LIBROSA_SR` | 8000 | Must match mobile DSP |
| `LIBROSA_N_FFT` | 256 | Must match mobile DSP |
| `LIBROSA_HOP_LENGTH` | 64 | Must match mobile DSP |
| `WINDOW_SIZE` | 25 | Frozen at ONNX export |
| `WINDOW_STRIDE` | 2 | Frozen at ONNX export |
| `N_SPLITS` | 5 | GroupKFold folds |
| `DRUG_MULTIPLIER` | 3.0 | Drug oversample factor |

---

## Integration Guide (Mobile Team)

### ONNX Inference Setup

The trained model at `results/inhaler_cnn.onnx` expects:
- Input: `float32[1, 25, 124]` (one window at a time)
- Output: `float32[1, 4]` — raw logits (apply softmax to get probabilities)
- Class mapping: `{0: Drug, 1: Exhale, 2: Inhale, 3: Noise}`

The CNN path does not require a StandardScaler. BatchNorm is folded into Conv
weights at ONNX export.

For classical ML models (XGBoost, RF), a StandardScaler must be applied before
inference. Export `scaler_mean.npy` and `scaler_scale.npy` from `train.py` line 126.

### DSP Implementation Requirements

The mobile C++ DSP module must replicate `librosa_extractor.py` **exactly** using
the parameter table above. Feature distribution mismatch is the most common cause
of accuracy degradation in deployment. Validate by:

1. Running `python src/librosa_extractor.py` on a test WAV to get reference features
2. Running the mobile DSP on the same audio
3. Checking mean absolute error < 0.001 across all 124 feature dimensions

### Frame Rate and Timing

- Audio frame rate: `8000 / 64 = 125 fps`
- Window duration: `25 / 125 = 200 ms`
- Windows per second (stride=2): `125 / 2 = 62.5 windows/s`
- Latency for 2 s segment: DSP <20 ms + inference <120 ms = **<200 ms total**

---

## Integration Guide (Hardware Team)

### INMP441 Setup

```
BCLK  GPIO 26
WS    GPIO 22 (LRCLK)
SD    GPIO 21

Sample rate: 8000 Hz (request from driver; INMP441 supports up to 104 kHz)
Bit depth:   24-bit left-justified in 32-bit DMA word
Channel:     Left (mono)
```

Bit handling:
```c
int32_t raw = dma_buffer[i];   // 32-bit DMA word
int32_t s24 = raw >> 8;        // right-shift to get int24
float   f   = (float)s24 / (float)(1 << 23);  // normalise to [-1, 1]
```

### Inhalation Detector

```c
#define WINDOW_SAMPLES   80      // 10 ms at 8 kHz
#define ONSET_THRESHOLD  0.01f   // RMS threshold to start recording
#define OFFSET_THRESHOLD 0.005f  // RMS threshold to stop
#define HOLDOFF_MS       200     // silence duration before END

float rms = 0;
for (int i = 0; i < WINDOW_SAMPLES; i++)
    rms += samples[i] * samples[i];
rms = sqrtf(rms / WINDOW_SAMPLES);
```

### BLE Packet

See [ARCHITECTURE.md](ARCHITECTURE.md) Section 5 for the complete binary packet
format and fragmentation protocol.

---

## Integration Guide (Cloud Team)

Firebase project setup:
1. Enable Firestore, Authentication, Cloud Functions, Storage
2. Deploy Firestore security rules: users can only read/write their own documents
3. Cloud Functions: `aggregateDailyAdherence`, `detectTechniqueRegression`,
   `notifyMissedDose`, `generateClinicianReport`

Session documents are written by the mobile app. See [ARCHITECTURE.md](ARCHITECTURE.md)
Section 10 for the complete Firestore schema and session document format.

---

## Team Integration Notes

### Critical Interfaces

| Interface | Owner | Consumer | Contract |
|---|---|---|---|
| `inhaler_cnn.onnx` | ML | Mobile | Input (N,25,124) float32; Output (N,4) raw logits |
| Feature parameters | ML | Mobile | See DSP parity table in ARCHITECTURE.md §6 |
| BLE binary packet | Hardware | Mobile | See packet struct in ARCHITECTURE.md §5 |
| Session document | Mobile | Cloud | See JSON schema in ARCHITECTURE.md §10 |
| Firestore schema | Cloud | Dashboard | See collections in ARCHITECTURE.md §10 |

### No Cross-Boundary Audio

Raw audio does not cross any system boundary:
- ESP32 → Mobile: PCM audio (internal, within session)
- Mobile → Cloud: derived analytics only
- Cloud → Dashboard: aggregated metrics only

---

## Development Status

| Component | Status |
|---|---|
| ML research pipeline (RF, SVM, XGBoost, CNN) | ✅ Complete |
| ONNX export (inhaler_cnn.onnx, opset 17) | ✅ Complete |
| Cross-validation results (89.0% mean accuracy) | ✅ Complete |
| Inhale-event dataset + usability rule (anomaly Stage 1) | ✅ Implemented |
| V1 global baseline / anomaly scoring | 🔲 Not started |
| ESP32 firmware | 🔲 Not started |
| BLE protocol implementation | 🔲 Not started |
| React Native mobile app | 🔲 Not started |
| On-device DSP (native C++) | 🔲 Not started |
| ONNX Runtime mobile integration | 🔲 Not started |
| Personalized baseline engine | 🔲 Not started |
| Firebase backend | 🔲 Not started |
| Doctor dashboard | 🔲 Not started |

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full technical reference (read for all integration details)
- [src/config.py](src/config.py) — All parameters (edit to tune)
- [src/librosa_extractor.py](src/librosa_extractor.py) — Feature extraction implementation
- [src/model_cnn.py](src/model_cnn.py) — InhalerCNN architecture
- [results/inhaler_cnn.onnx](results/inhaler_cnn.onnx) — Trained model (deployment artifact)
- [results/cv_results.csv](results/cv_results.csv) — Cross-validation metrics
- [results/xg/summary_report.txt](results/xg/summary_report.txt) — XGBoost run summary

---

*PRISM — Capstone Project 2026*
