# PRISM - Pulmonary Response and Inhaler System Monitor
## Full System Architecture and Technical Reference

> **Document purpose:** Cross-team technical reference for hardware engineers,
> mobile developers, ML researchers, and cloud/backend engineers working on the
> end-to-end PRISM smart inhaler platform.
>
> **Last updated:** 2026-08-23
> **Status:** Research pipeline validated; hardware and cloud integrations in progress.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Design Principles](#2-core-design-principles)
3. [End-to-End System Diagram](#3-end-to-end-system-diagram)
4. [Hardware Layer](#4-hardware-layer)
5. [BLE Transmission Protocol](#5-ble-transmission-protocol)
6. [Mobile Application Intelligence Layer](#6-mobile-application-intelligence-layer)
7. [ML Pipeline - Research Environment](#7-ml-pipeline)
8. [ML Inference - On-Device Deployment](#8-ml-inference)
9. [Personalized Baseline Engine](#9-personalized-baseline-engine)
10. [Cloud and Firebase Integration](#10-cloud-and-firebase-integration)
11. [Doctor Dashboard](#11-doctor-dashboard)
12. [Repository Directory Layout](#12-repository-directory-layout)
13. [Data Schemas and Interfaces](#13-data-schemas-and-interfaces)
14. [Module Reference](#14-module-reference)
15. [Model Registry and Versioning](#15-model-registry-and-versioning)
16. [Development Priority Order](#16-development-priority-order)
17. [Implementation Status](#17-implementation-status)

---

## 1. System Overview

PRISM is a smart inhaler monitoring platform for pressurised Metered-Dose Inhalers (pMDI).
It combines acoustic sensing, on-device signal processing, personalized machine learning,
and longitudinal analytics to evaluate both **medication adherence** and **inhalation
quality** at a per-event granularity.

Unlike binary-trigger smart inhalers that only detect actuation, PRISM performs acoustic
analysis to classify every moment of an inhalation recording into one of four events:
**Drug, Inhale, Exhale, Noise** - enabling rich qualitative feedback about technique,
flow rate, coordination, and consistency over time.

### What PRISM Answers Per Inhalation Event

| Question | Mechanism |
|---|---|
| Was the inhaler actuated? | Drug-class frame detection |
| Was inhalation sustained and deep? | Inhale-class frame proportion and duration |
| Did the patient exhale before inhaling? | Exhale-class detection preceding Drug |
| Did actuation coincide with inhalation? | Drug/Inhale temporal overlap analysis |
| Is the patient technique improving? | Baseline engine longitudinal comparison |
| Is technique within acceptable clinical range? | Global model quality score |

---

## 2. Core Design Principles

### Mobile-First Edge Intelligence

The smartphone is the **primary compute platform**. All signal processing, feature
extraction, and ML inference execute on-device. The ESP32 is a **sensor and radio
peripheral only** - not a compute node.

```
ESP32                   Smartphone                  Cloud
-------                 --------------------        ----------------
Microphone ADC          DSP pipeline                Session summaries
I2S read                Feature extraction          Quality scores
Energy threshold        ONNX Runtime inference      Adherence metrics
Packet assembly         Baseline engine             Derived analytics
BLE transmit            Feedback generation
                        Local SQLite storage
```

### Privacy-by-Design

Raw inhalation audio **never leaves the user smartphone**. Only derived, anonymized
analytics are transmitted to the cloud. No cloud ML inference required.

### Personalized Monitoring (Two Assessment Layers)

1. **Global Classification Model** - population-trained CNN/XGBoost (ONNX) classifying
   each 8 ms audio frame into Drug / Inhale / Exhale / Noise.
2. **Personalized Baseline Engine** - per-user statistical model from historical
   sessions; new inhalations compared against personal baseline.

The final quality assessment combines both layers.

---

## 3. End-to-End System Diagram

```
+-------------------------------------------------------------------+
| HARDWARE LAYER                                                    |
|                                                                   |
| INMP441 I2S Microphone (8 kHz, 16-bit, mono)                     |
|   |                                                               |
|   v                                                               |
| ESP32-WROOM-32                                                    |
|   +-- I2S DMA read -> ring buffer                                 |
|   +-- RMS energy threshold detector (10 ms windows)              |
|   +-- [Optional] MPU6050 IMU - orientation detection             |
|   +-- [Optional] Actuation switch - valve trigger                |
|   +-- Binary packet assembly + BLE GATT write                    |
|             |                                                     |
|             | Bluetooth Low Energy 5.0 (MTU-fragmented binary)   |
+-------------|---------------------------------------------------  +
              |
+-------------v-----------------------------------------------------+
| MOBILE APPLICATION LAYER (React Native + Native Modules)          |
|                                                                   |
| BLE Manager -> fragment reassembly -> packet deserialization      |
|         |                                                         |
|         v                                                         |
| DSP Pipeline (Native C++)                                         |
|   Pre-emphasis: y[n] = x[n] - 0.97*x[n-1]                       |
|   STFT (N_FFT=256, HOP=64, Hann) -> S: (129, n_frames)           |
|   Mel filterbank (128 filters, 50-4000 Hz)                       |
|         |                                                         |
|         v                                                         |
| Feature Extractor (Native C++) - per-frame:                       |
|   MFCC[40]  Delta[40]  Delta2[40]  Centroid[1]                   |
|   Flatness[1]  Rolloff[1]  ZCR[1]                                |
|   -> float32[124] per frame                                      |
|         |                                                         |
|         v                                                         |
| Sliding Window: 25 frames (200 ms), stride 2 frames (16 ms)      |
|   -> float32[25, 124] per window                                 |
|         |                                                         |
|         v                                                         |
| ONNX Runtime: inhaler_cnn.onnx (1.15 MB)                         |
|   Input:  float32[1, 25, 124]                                    |
|   Output: float32[1, 4] raw logits                               |
|   Softmax -> probs -> argmax -> Drug/Exhale/Inhale/Noise         |
|   Frame-sequence reconstruction (majority vote)                  |
|         |                                                         |
|         v                                                         |
| Session Analytics + Baseline Engine + Composite Quality Label     |
|         |                                                         |
| Feedback UI + Local SQLite Storage                                |
|         |                                                         |
+---------|---------------------------------------------------------+
          | HTTPS / Firebase SDK (session summaries only)
+---------v---------------------------------------------------------+
| CLOUD LAYER (Firebase)                                            |
| Firestore + Auth + Functions + Storage                            |
+---------+---------------------------------------------------------+
          |
+---------v---------------------------------------------------------+
| DOCTOR DASHBOARD: adherence | quality trends | anomalies | export |
+-------------------------------------------------------------------+
```

---

## 4. Hardware Layer

### Components

| Component | Role |
|---|---|
| ESP32-WROOM-32 | Main MCU - sensing, event detection, BLE |
| INMP441 I2S Microphone | 24-bit digital MEMS microphone |
| Li-Ion Battery | 500-1000 mAh portable power |
| TP4056 | USB charging + battery protection |
| MPU6050 IMU (optional) | Orientation confirmation |
| Actuation Switch (optional) | Valve trigger confirmation |
| DHT22 / BME280 (optional) | Temperature/humidity metadata |

### INMP441 I2S Configuration

```
BCLK  GPIO 26    WS    GPIO 22    SD    GPIO 21

Mode:        I2S_MODE_MASTER | I2S_MODE_RX
Sample Rate: 8000 Hz
Bit Depth:   I2S_BITS_PER_SAMPLE_24BIT (sign-extended to 32-bit in DMA)
Channel:     I2S_CHANNEL_FMT_ONLY_LEFT (mono)
Buffer:      DMA ring buffer, 8 blocks x 512 samples
```

INMP441 outputs 24-bit left-justified in a 32-bit DMA word.
Firmware right-shifts by 8 bits to get int24, then divides by 2^23 for float32.

### Inhalation Detection State Machine (Firmware)

```
RMS(window=10 ms=80 samples) = sqrt(mean(x[n]^2))

IDLE      -> RMS > ONSET_THRESHOLD (0.01)       -> RECORDING
RECORDING -> accumulate into ring buffer
RECORDING -> RMS < OFFSET_THRESHOLD for >200ms  -> END
END       -> package binary packet, BLE transmit -> IDLE
```

ESP32 does NOT perform MFCC, spectrogram, ML inference, or cloud comms.

### Buffer: 5 s x 8000 x 2 bytes = 80 KB (fits in 520 KB SRAM)
### Power: deep sleep after 30 s idle; active current target < 80 mA

---

## 5. BLE Transmission Protocol

### GATT Profile

```
Service UUID:       <PRISM-primary-service-UUID>
Characteristic:     <PRISM-event-characteristic-UUID>
Properties:         NOTIFY + WRITE
MTU:                512 bytes requested (500 bytes usable after ATT overhead)
```

### Binary Event Packet (little-endian)

```c
struct PrismEventPacket {
    uint8_t  magic[4];         // "PRIS" = 0x50 0x52 0x49 0x53
    uint8_t  version;          // 0x01
    uint8_t  event_type;       // 0x01=inhalation 0x02=exhale-only 0x03=noise
    uint32_t timestamp_unix;   // UTC seconds
    uint16_t timestamp_ms;     // millisecond offset
    uint16_t duration_ms;      // event duration
    uint8_t  actuation_state;  // 0x00=none 0x01=switch 0x02=uncertain
    uint8_t  imu_present;      // 0x01 if IMU bytes follow
    int16_t  imu_ax, imu_ay, imu_az;  // 16384 LSB/g
    int16_t  imu_gx, imu_gy, imu_gz;  // 131 LSB/deg/s
    uint16_t audio_len;        // byte count of audio_data
    uint8_t  audio_data[];     // raw PCM int16 LE, 8 kHz mono
};
```

### Fragmentation (audio > 500 bytes)

```
byte 0: 0xFF (fragment marker)
byte 1: total fragment count
byte 2: current fragment index (0-based)
byte 3: 0x00 (reserved)
bytes 4+: payload chunk
Reassembly on mobile -> complete packet to DSP pipeline
```

---

## 6. Mobile Application Intelligence Layer

### DSP Parameter Parity (frozen at ONNX export - cannot change without retraining)

| Parameter | Value | Notes |
|---|---|---|
| Sample Rate | 8000 Hz | Resample BLE audio if needed |
| N_FFT | 256 | FFT window = 32 ms at 8 kHz |
| HOP_LENGTH | 64 | Frame hop = 8 ms; frame rate = 125 fps |
| N_MELS | 128 | Mel filterbank resolution |
| FMIN | 50 Hz | Exclude DC / sub-bass |
| FMAX | 4000 Hz | Nyquist for 8 kHz |
| N_MFCC | 40 | DCT coefficients kept |
| DELTA_WIDTH | 9 | HTK regression half-width N = 4 |
| ROLLOFF_PERC | 0.85 | 85th percentile rolloff |
| WINDOW_SIZE | 25 frames | 200 ms sliding window |
| WINDOW_STRIDE | 2 frames | 16 ms step |

### Component Responsibilities

| Component | Technology |
|---|---|
| BLE GATT client | react-native-ble-plx |
| Packet reassembly | Native C++ |
| DSP pipeline | Native C++ mirroring librosa_extractor.py |
| Feature extraction | Native C++ |
| ONNX Runtime | Android: onnxruntime-android / iOS: onnxruntime-objc |
| Baseline engine | Native C++ or JS |
| Session analytics | JavaScript / React Native |
| Local storage | SQLite (react-native-sqlite-storage) |
| Cloud sync | Firebase JS SDK |

---

## 7. ML Pipeline

This repository is the research and training environment.
It produces ONNX artifacts embedded in the mobile app.

### Dataset

- 361 recordings x ~12 s each; 8 kHz 16-bit WAV mono; ~72 min total labelled audio
- Annotation: `data/annotation.csv` (no header; columns: filename, label, start_sample, end_sample)
- Labels: Drug, Exhale, Inhale, Noise

### Raw Label Distribution (Frame Level, Before Balancing)

| Class | Frames | % | Description |
|---|---|---|---|
| Drug | 7,817 | 1.4% | Aerosol spray actuation (~300 ms burst) |
| Exhale | 73,222 | 13.3% | Patient exhaling before inhalation |
| Inhale | 49,310 | 9.0% | Sustained inward airflow |
| Noise | 418,436 | 76.2% | Ambient, silence, handling |
| **Total** | **548,785** | | 361 recordings x ~1500 frames each |

### Pipeline Commands

```bash
python src/run_pipeline.py              # RF + SVM + XGBoost, 5-fold
python src/run_pipeline.py --xgb-only  # XGBoost only, 5-fold
python src/run_pipeline.py --cnn       # 1D CNN, exports ONNX
python src/run_pipeline.py --fast      # 3-fold XGBoost smoke test
python src/run_pipeline.py --no-svm    # RF + XGBoost
```

---

### Step 0: Annotation Loading (loader.load_annotation)

```python
ann = load_annotation()
# pd.DataFrame: [filename, label, start_sample, end_sample]
# No header in CSV; label case normalised ("inhale" -> "Inhale")
```

---

### Step 1: Feature Extraction (librosa_extractor)

One-time per recording; cached to `data/extracted/<base>/features.npy`.

#### 1a. Audio Loading

```python
audio, sr = librosa.load(wav_path, sr=8000, mono=True)
# float32 [-1,1]; auto-resampled if source SR != 8000
# ~12 s -> ~96,000 samples
```

#### 1b. STFT

```python
S = np.abs(librosa.stft(audio, n_fft=256, hop_length=64))
# Window:   Hann, 256 samples (32 ms)
# Hop:      64 samples (8 ms); 75% overlap
# Output:   S shape (129, n_frames)
#   129 = N_FFT/2 + 1 frequency bins
#   n_frames = ceil(len(audio)/64) ~ 1500 for 12 s recording
#   freq resolution: 8000/256 = 31.25 Hz/bin
#   time resolution: 64/8000  = 8 ms/frame  (125 fps)
# S is shared for MFCC and all spectral features (no double STFT)
```

#### 1c. MFCC

```python
mfcc = librosa.feature.mfcc(
    y=audio, sr=sr, n_mfcc=40, n_fft=256, hop_length=64,
    n_mels=128, fmin=50.0, fmax=4000.0
)  # shape: (40, n_frames)
#
# Steps:
#   mel_spec = mel_filterbank(128 filters, 50-4000 Hz) @ |S|  -> (128, n_frames)
#   log_mel  = log(mel_spec + epsilon)                          (log compression)
#   mfcc     = DCT(log_mel)[:40]                               (decorrelation)
#
# MFCC[0] ~ log RMS energy
# MFCC[1..] ~ spectral shape coefficients
```

#### 1d. Delta Coefficients (HTK Regression Filter)

```python
delta  = librosa.feature.delta(mfcc, width=9)           # (40, n_frames)
delta2 = librosa.feature.delta(mfcc, width=9, order=2)  # (40, n_frames)
#
# HTK regression filter, half-width N = (9-1)/2 = 4:
#
#              sum_{n=1}^{4} n * (f[t+n] - f[t-n])
# delta[t] = -----------------------------------------
#                  2 * sum_{n=1}^{4} n^2
#
#           = (1*(f[t+1]-f[t-1]) + 2*(f[t+2]-f[t-2])
#              + 3*(f[t+3]-f[t-3]) + 4*(f[t+4]-f[t-4])) / 60
#
# delta2 = delta(delta)
# Edge frames: mirror-padded
#
# WHY DELTAS MATTER:
#   Static MFCC  = instantaneous spectral shape
#   Delta        = rate of spectral change (onset/offset transients)
#   Delta2       = spectral acceleration
#   Drug actuation: sharp onset + peak + decay -> large delta/delta2 signal
#   This is why Drug F1 reaches ~0.95 while Inhale/Exhale are ~0.87
```

#### 1e. Spectral Features

```python
centroid = librosa.feature.spectral_centroid(S=S, sr=sr) / (8000/2)
# Weighted mean frequency / Nyquist -> [0, 1]
# centroid_raw = sum(freq_bins * S) / sum(S)  [Hz]
# Drug aerosol: ~0.5-0.8 (2000-3200 Hz, broadband)
# Breath sound: ~0.1-0.3 (400-1200 Hz, turbulent)

flatness = librosa.feature.spectral_flatness(S=S)
# geometric_mean(S) / arithmetic_mean(S)
# = exp(mean(log(S+eps))) / (mean(S)+eps)
# 1.0 = white noise; 0.0 = pure tone
# Drug aerosol: ~0.7-1.0  (most discriminative single feature)
# Breath: ~0.1-0.4

rolloff = librosa.feature.spectral_rolloff(
    S=S, sr=sr, roll_percent=0.85
) / (8000/2)
# Smallest freq f* where sum_{f<=f*}S >= 0.85*sum_f S, normalised by 4000
# Drug: rolloff ~0.9-1.0 (broadband)
# Breath: ~0.3-0.6

zcr = librosa.feature.zero_crossing_rate(audio, hop_length=64)
# 0.5 * sum|sign(x[n])-sign(x[n-1])| / frame_len
# High ZCR: rapid oscillations (noise, high freq)
# Low ZCR:  smooth sustained signal (breath)
```

#### 1f. Feature Assembly

```python
features = np.vstack([
    mfcc,      # (40, n_frames)  spectral shape
    delta,     # (40, n_frames)  rate of change
    delta2,    # (40, n_frames)  acceleration
    centroid,  # (1,  n_frames)  [0,1]
    flatness,  # (1,  n_frames)  [0,1]
    rolloff,   # (1,  n_frames)  [0,1]
    zcr,       # (1,  n_frames)
]).T  # -> (n_frames, 124) float32

np.save("data/extracted/<base>/features.npy", features)
```

**Feature column index:**

| Columns | Feature | Dim |
|---|---|---|
| 0-39 | MFCC 0-39 | 40 |
| 40-79 | Delta-MFCC 0-39 | 40 |
| 80-119 | Delta2-MFCC 0-39 | 40 |
| 120 | Spectral centroid (norm) | 1 |
| 121 | Spectral flatness | 1 |
| 122 | Spectral rolloff (norm, roll=0.85) | 1 |
| 123 | Zero crossing rate | 1 |
| **Total** | | **124** |

#### 1g. Label Alignment (Sample -> Frame)

```python
y = np.full(n_frames, "Noise", dtype=object)  # default
for row in ann[ann["filename"] == wav_name].itertuples():
    sf = max(0, min(int(row.start_sample / 64), n_frames - 1))
    ef = max(0, min(int(row.end_sample   / 64), n_frames))
    y[sf:ef] = row.label
# HOP_LENGTH = 64; later rows win for overlapping annotations
```

---

### Step 2: Windowed Dataset Construction (feature_extractor.py)

#### 2a. Noise Trimming

```python
buffer_frames = 20  # ~160 ms at 125 fps
keep = np.zeros(n_frames, bool)
for j in np.where(y != "Noise")[0]:
    keep[max(0, j-20) : j+21] = True
X_rec, y_rec = X_rec[keep], y_rec[keep]
# Removes 3-4 s of silence per recording
# Noise:Drug ratio: ~50:1 -> ~12:1
# Dataset size reduction: ~30%
```

#### 2b. Sliding Window

```python
window_size = 25    # 200 ms = 25 * 64/8000
stride      = 2     # 16 ms  =  2 * 64/8000

for t in range(0, n_frames - window_size + 1, stride):
    x_window = features[t : t + window_size]  # (25, 124)
    x_flat   = x_window.flatten()             # (3100,) for XGBoost/RF/SVM
    # CNN: (25, 124) after reshape in train_cnn.py
    
    # Majority vote label
    uniq, cnts = np.unique(y[t:t+window_size], return_counts=True)
    y_window = uniq[cnts.argmax()]
# Drug (~300 ms) spans ~18 windows at 2-frame stride
```

#### 2c. Dataset Balancing (global, after concatenating all recordings)

```
Stage 1 - Cap Noise:
  noise_cap = 2 * count(Exhale)
  Subsample Noise windows randomly.

Stage 2 - Oversample Drug (SMOTE approximation):
  drug_target = min(3.0 * count(Inhale), count(Exhale))
  for i in range(drug_target - current_drug_count):
      src   = X[choice(Drug_indices)]
      X_new = src + N(0, sigma=0.01)   # Gaussian jitter
      y_new = "Drug"
```

**Post-balance (validated, 361 recordings, ~170,000 total windows):**

| Class | Before | After |
|---|---|---|
| Drug | 1.4% | 21.4% |
| Exhale | 13.3% | 21.4% |
| Inhale | 9.0% | 14.5% |
| Noise | 76.2% | 42.6% |

Group index preservation: recording group appended as extra column through
`balance_dataset`, then stripped out. GroupKFold integrity maintained.

---

### Step 3: Label Encoding

```python
le = LabelEncoder()
le.fit(["Drug", "Exhale", "Inhale", "Noise"])
# Drug=0  Exhale=1  Inhale=2  Noise=3
# MUST match ONNX output class order at inference time
y_int = le.transform(y_win_all)
```

---

### Step 4: Cross-Validation (GroupKFold, k=5)

```
groups:  recording index per window (0..360)
         all windows from recording i -> groups[j] = i
         no recording in both train and test in any fold

GroupKFold k=5:
  test fold:  ~72 recordings (~20% of 361)
  train fold: ~289 recordings
```

**Per-fold StandardScaler (classical ML: RF, SVM, XGBoost):**

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train ONLY
X_test_sc  = scaler.transform(X_test)        # no leakage
```

**CNN path:** No StandardScaler; BatchNorm1d handles normalisation internally.

> **Deployment (classical ML):** Export `scaler_mean.npy` (float32[124]) and
> `scaler_scale.npy` (float32[124]) from the final fold. Mobile app applies:
> `features_norm = (features - mean) / scale` before ONNX Runtime inference.
> See `train.py` line 126 for the scaler fit location.

---

### Steps 5-9: Evaluation and Output

| Step | Module | Output |
|---|---|---|
| 5 | `evaluate.aggregate_metrics` | Per-class P/R/F1 averaged over folds |
| 6 | `visualize.plot_confusion_matrices` | `results/confusion_matrix.png` |
| 7 | `visualize.plot_feature_importance` | `results/xg/feature_importance_xgboost.png` |
| 8 | `evaluate.print_misclassification_analysis` | Drug recall, Inhale/Exhale confusion rates |
| 9 | `evaluate.save_cv_csv` | `results/cv_results.csv`, `results/xg/summary_report.txt` |

---

## 8. ML Inference

### 8.1 Model Inventory

| Model | Format | File | Size | Infer. time | Status |
|---|---|---|---|---|---|
| 1D CNN (InhalerCNN) | ONNX opset 17 | `results/inhaler_cnn.onnx` | 1.15 MB | <2 ms/win | **Primary** |
| XGBoost | ONNX (onnxmltools) | TBD | TBD | ~0.1 ms/win | Alternative |
| Random Forest | sklearn | - | - | ~1 ms/win | Research only |

### 8.2 InhalerCNN Architecture

Input: `(batch, 25, 124)` permuted to `(batch, 124, 25)` (features=channels, frames=length)

```
Layer                        Output          Parameters
=====                        ======          ==========
Input                        (N, 25, 124)    --
Permute(0,2,1)               (N, 124, 25)    --

STEM
  Conv1d(124->128, k=3, p=1) (N, 128, 25)    47,872
  BatchNorm1d(128)           (N, 128, 25)    256
  ReLU

RESIDUAL BLOCK 1
  Conv1d(128->128, k=3, p=1) (N, 128, 25)    49,152
  BatchNorm1d(128)           (N, 128, 25)    256
  ReLU + Dropout(0.25)
  Conv1d(128->128, k=3, p=1) (N, 128, 25)    49,152
  BatchNorm1d(128)           (N, 128, 25)    256
  + identity skip + ReLU     (N, 128, 25)    --

RESIDUAL BLOCK 2 (identical to Block 1)      ~98,816

NECK
  Conv1d(128->256, k=1)      (N, 256, 25)    32,768
  BatchNorm1d(256)           (N, 256, 25)    512
  ReLU
  AdaptiveAvgPool1d(1)       (N, 256, 1)     --

HEAD
  Flatten                    (N, 256)        --
  Linear(256->64)            (N, 64)         16,448
  ReLU + Dropout(0.25)
  Linear(64->4)              (N, 4)          260
                                             -------
Total trainable parameters:                  ~320,000
```

**Design rationale:**
- `Conv1d` features-as-channels: each filter learns cross-feature relationships
  (e.g., MFCC[0] vs flatness) over time. XGBoost with flattened vectors cannot do this.
- Residual skip connections: prevent gradient vanishing in 7-layer depth.
- `AdaptiveAvgPool1d(1)`: collapses temporal axis -> model is window-size-agnostic.
- Bottleneck head `256->64->4`: compact with non-linear decision boundary.
- `do_constant_folding=True` at ONNX export: BatchNorm folded into Conv weights,
  reducing model size and inference latency.

### 8.3 ONNX Export Contract

```python
# From model_cnn.py::export_onnx():
torch.onnx.export(
    model, dummy_input, path,
    input_names         = ["features"],
    output_names        = ["logits"],
    dynamic_axes        = {"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version       = 17,
    do_constant_folding = True,
)

# Frozen I/O contract:
#   Input  "features":  float32 (batch, 25, 124)  [dynamic batch]
#   Output "logits":    float32 (batch, 4)          [RAW LOGITS not probs]
#   Class order:        Drug=0, Exhale=1, Inhale=2, Noise=3
#   -> apply softmax in app layer before argmax
```

### 8.4 Android Integration (Kotlin)

```kotlin
val env     = OrtEnvironment.getEnvironment()
val session = OrtSession(env, modelBytes, OrtSession.SessionOptions())
val tensor  = OnnxTensor.createTensor(
    env, FloatBuffer.wrap(windowFeatures), longArrayOf(1L, 25L, 124L)
)
val results = session.run(mapOf("features" to tensor))
val logits  = (results[0].value as Array<FloatArray>)[0]  // float[4]
val probs   = softmax(logits)  // [p_Drug, p_Exhale, p_Inhale, p_Noise]
val pred    = probs.indices.maxByOrNull { probs[it] }!!   // 0/1/2/3
```

### 8.5 iOS Integration (Swift)

```swift
let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: nil)
let tensor  = try ORTValue(
    tensorData: NSMutableData(data: Data(bytes: windowFeatures, count: 25*124*4)),
    elementType: .float, shape: [1, 25, 124]
)
let out = try session.run(withInputs: ["features": tensor],
                          outputNames: ["logits"], runOptions: nil)
// out["logits"]: float32[1,4] -> softmax -> class probs
```

### 8.6 Per-Event Inference Flow (Complete Runtime Sequence)

```
BLE audio received (raw PCM int16, 8 kHz, mono)
     |
     v [1] Pre-processing
  x = audio_int16 / 32768.0              (normalise to float32)
  y[n] = x[n] - 0.97 * x[n-1]          (pre-emphasis)
     |
     v [2] STFT
  S = |STFT(y, n_fft=256, hop=64, Hann)|   -> (129, n_frames)
  Frame rate: 125 fps
     |
     v [3] Feature extraction (per frame)
  MFCC[40]:    mel_filterbank(128, 50-4000 Hz) -> log -> DCT[:40]
  Delta[40]:   sum_{n=1}^{4} n*(f[t+n]-f[t-n]) / 60
  Delta2[40]:  delta(delta)
  Centroid[1]: sum(f*S)/sum(S) / 4000     [normalised to 0-1]
  Flatness[1]: geomean(S) / mean(S)
  Rolloff[1]:  85th_pct_freq(S) / 4000    [normalised to 0-1]
  ZCR[1]:      0.5*sum|sign(x[n])-sign(x[n-1])| / frame_len
  -> features[t]: float32[124]
     |
     v [4] Normalisation (classical ML only; skip for CNN)
  features_norm = (features - scaler_mean) / scaler_scale
     |
     v [5] Sliding window
  for t in range(0, n_frames-25+1, step=2):
      window = features[t:t+25]   -> float32[25,124]
     |
     v [6] ONNX Runtime inference (per window)
  input:  float32[1, 25, 124]
  logits: float32[1, 4]           (raw logits)
  probs:  softmax(logits)         -> [p_Drug, p_Exhale, p_Inhale, p_Noise]
  pred:   argmax(probs)           -> {0,1,2,3}
     |
     v [7] Frame sequence reconstruction
  Map window t to center frame c = t + 12
  Multiple windows per frame: majority vote over overlapping predictions
  Output: y_pred[0..n_frames-1] in {Drug, Exhale, Inhale, Noise}
     |
     v [8] Session analytics
  drug_dur     = count(Drug)   * 64/8000   [seconds]
  inhale_dur   = count(Inhale) * 64/8000

  drug_onset   = first frame where y_pred == Drug
  inhale_onset = first frame where y_pred == Inhale
  coord_delay  = |drug_onset - inhale_onset| * 64/8000   [seconds]

  pre_exhale_detected = any(y_pred[:drug_onset] == Exhale)
  insufficient_inhale = inhale_dur < 1.0 s
  late_actuation      = coord_delay > 0.5 s
  missed_dose         = drug_frames == 0
     |
     v [9] Baseline comparison + [10] Composite quality label
  GOOD | POOR | GOOD_BUT_INCONSISTENT | ABNORMAL | MISSED_DOSE
```

### 8.7 CNN Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 40 | Early stopping fires at 15-30 typically |
| Batch size | 512 | Fits 4 GB VRAM |
| Optimizer | Adam | Adaptive LR |
| Learning rate | 1e-3 -> 1e-5 | CosineAnnealingLR (T_max=40) |
| Weight decay | 1e-4 | L2 regularisation |
| Loss | CrossEntropyLoss | Class-weighted: w_c = n_classes / count_c |
| Early stopping | patience=7 | Restore best checkpoint |
| Mixed precision | AMP (CUDA) | ~2x GPU speedup; fp32 fallback on CPU/MPS |
| Dropout | 0.25 | ResBlocks + HEAD |

### 8.8 Validated Cross-Validation Results

**CNN 3-fold GroupKFold** (source: `results/cv_results.csv`):

| Fold | Accuracy | Drug F1 | Exhale F1 | Inhale F1 | Noise F1 |
|---|---|---|---|---|---|
| 1 | 0.8954 | 0.9507 | 0.8742 | 0.9080 | 0.8696 |
| 2 | 0.8847 | 0.9460 | 0.8604 | 0.8865 | 0.8696 |
| 3 | 0.8906 | 0.9517 | 0.8655 | 0.8838 | 0.8731 |
| **Mean** | **0.8902** | **0.9495** | **0.8667** | **0.8928** | **0.8708** |

**XGBoost 5-fold** (source: `results/xg/summary_report.txt`): 0.8902 +/- 0.0044

**Key observations:**
- Drug F1 ~0.95: spectrally unique broadband aerosol; spectral flatness primary discriminant
- Exhale/Inhale F1 ~0.87: hardest boundary; delta-MFCC captures directional flow transients
- Noise F1 ~0.87: reliable due to low energy and low flatness

### 8.9 Feature Importance (XGBoost Gain)

1. **Spectral Flatness** - Drug spray is white noise (~1.0); breath is tonal (~0.1-0.4)
2. **MFCC[0]** (log energy) - Drug/Inhale high vs Noise low
3. **Delta-MFCC[0..5]** - onset/offset transients critical for Drug localisation
4. **Spectral Centroid** - aerosol ~2000-3200 Hz vs breath ~400-1200 Hz
5. **Spectral Rolloff** - Drug extends to Nyquist; breath rolls off at ~1500-2000 Hz

---

## 9. Personalized Baseline Engine

The global model assesses quality vs. the population.
The baseline engine asks: **is this inhalation normal for THIS patient?**

### Baseline State (Firestore: users/{uid}/baseline)

```
session_count     : int         sessions contributing to baseline
mu_duration       : float       mean inhale duration (s)
sigma_duration    : float       std of duration
mu_energy         : float       mean MFCC[0] over Inhale frames
mu_centroid       : float       mean spectral centroid over Inhale frames
mu_mfcc           : float[40]   mean MFCC vector over Inhale frames
sigma_mfcc        : float[40]   per-coefficient std
cov_mfcc_inv      : float[40,40] inverse covariance for Mahalanobis distance
drug_recall_hist  : float[]     per-session Drug detection rate
last_updated      : timestamp
```

### Update Policy (quality-gated)

```
Qualifies if ALL true:
  global_quality_score >= 0.7
  drug_detected == True
  inhale_duration >= 0.8 s
  NOT user-flagged as poor

Update (EMA, alpha=0.1):
  mu_new = 0.1 * session_value + 0.9 * mu_old
```

### Deviation Scoring

```python
d = sqrt((session_mfcc - mu_mfcc).T @ cov_mfcc_inv @ (session_mfcc - mu_mfcc))
z_dur    = (dur    - mu_duration) / sigma_duration
z_energy = (energy - mu_energy)   / sigma_energy

deviation = 0.5*d + 0.3*abs(z_dur) + 0.2*abs(z_energy)

if deviation < 1.5:    "within_baseline"
elif deviation < 3.0:  "mild_deviation"
else:                  "significant_deviation"
```

### Composite Quality Logic

```
Global      x  Baseline              -> Composite
GOOD           within_baseline       -> GOOD
GOOD           mild_deviation        -> GOOD_BUT_INCONSISTENT
GOOD           significant_deviation -> ABNORMAL (possible disease change)
POOR           any                   -> POOR
(drug missing)                       -> MISSED_DOSE
```

---

## 10. Cloud and Firebase Integration

### Session Document (no raw audio transmitted)

```json
{
  "session_id":   "uuid-v4",
  "user_id":      "firebase-auth-uid",
  "device_id":    "esp32-mac",
  "timestamp":    "2026-08-23T14:30:00Z",
  "inhaler_type": "pMDI",
  "event_classification": {
    "drug_detected": true, "drug_duration_ms": 280,
    "inhale_duration_ms": 2340, "coordination_delay_ms": 120,
    "pre_exhale_detected": true
  },
  "quality_assessment": {
    "global_score": 0.87, "composite_label": "GOOD",
    "deviation_score": 0.23, "deviation_flag": "within_baseline"
  },
  "imu_orientation": {"inhaler_vertical": true, "tilt_degrees": 8.2},
  "model_version": "cnn_v1.0", "app_version": "1.0.0"
}
```

### Firestore Schema

```
users/{uid}/profile           demographics, inhaler type, prescriber
users/{uid}/baseline          personalized baseline state
users/{uid}/sessions/{id}     one per inhalation event (above doc)
users/{uid}/adherence/{date}  doses_scheduled/taken/missed, quality_summary
clinicians/{cid}/patients/{uid}   read-only; requires consent
```

### Cloud Functions

| Function | Trigger | Purpose |
|---|---|---|
| `aggregateDailyAdherence` | Firestore write | Daily adherence rollup |
| `detectTechniqueRegression` | Daily cron | Flag persistent poor technique |
| `notifyMissedDose` | Scheduled | Push on missed dose window |
| `generateClinicianReport` | HTTP callable | PDF/CSV export |

---

## 11. Doctor Dashboard

| View | Source | Description |
|---|---|---|
| Adherence Calendar | `adherence` collection | Doses taken vs prescribed per day |
| Quality Trend | `sessions` collection | Rolling quality scores 30/90 days |
| Technique Heatmap | `sessions` classification | Event distribution over time |
| Anomaly Timeline | `sessions` deviation_flag | ABNORMAL sessions |
| Export Panel | Cloud Functions | PDF / CSV download |

---

## 12. Repository Directory Layout

```
PRISM smart inhaler/
|-- ARCHITECTURE.md          <- This document
|-- README.md                <- Project overview and quickstart
|-- requirements.txt
|
|-- data/                    <- Dataset root (git-ignored)
|   |-- annotation.csv       <- Sample-level labels (361 recordings)
|   |-- rec<ts>.wav          <- 8 kHz 16-bit WAV mono
|   |-- rec<ts>_mfcc.csv     <- Precomputed MFCCs (legacy)
|   |-- precomputed/rec<base>/_mfcc.csv, _zcr.csv
|   +-- extracted/rec<base>/features.npy   <- float32(n_frames,124); delete to refresh
|
|-- results/
|   |-- inhaler_cnn.onnx     <- Trained CNN; primary deployment artifact
|   |-- confusion_matrix.png
|   |-- class_metrics.png
|   |-- drug_stats.png
|   |-- noise_confusion.png
|   |-- cv_results.csv       <- Per-fold per-class metrics
|   +-- xg/
|       |-- summary_report.txt
|       |-- class_metrics.png
|       |-- drug_stats.png
|       |-- noise_confusion.png
|       +-- feature_importance_xgboost.png
|
+-- src/
    |-- config.py            <- ALL paths and hyperparameters
    |-- loader.py            <- Legacy CSV loader (40 features)
    |-- librosa_extractor.py <- Primary extractor (124 features, .npy cache)
    |-- feature_extractor.py <- Windowing, trimming, balancing
    |-- dataset.py           <- Dataset assembly utilities
    |-- train.py             <- GroupKFold CV: RF + SVM + XGBoost
    |-- train_cnn.py         <- GroupKFold CV: 1D CNN (PyTorch + AMP + ONNX)
    |-- model_cnn.py         <- InhalerCNN + ONNX export
    |-- evaluate.py          <- Metrics, misclassification analysis
    |-- visualize.py         <- Confusion matrix, feature importance plots
    +-- run_pipeline.py      <- Main orchestrator
```

---

## 13. Data Schemas and Interfaces

### annotation.csv (no header row)

```
rec2018-01-22_17h41m33.475s.wav,Inhale,512,9216
rec2018-01-22_17h41m33.475s.wav,Drug,4096,6144
```

Loaded with `pd.read_csv(header=None)`.
Unannotated intervals default to "Noise" programmatically.

### Feature Matrix

```
X_rec: float32 (n_frames, 124)
y_rec: object  (n_frames,)    Drug | Exhale | Inhale | Noise

Columns:
  [0..39]   MFCC 0-39
  [40..79]  Delta-MFCC 0-39
  [80..119] Delta2-MFCC 0-39
  [120]     Spectral centroid (norm, / Nyquist)
  [121]     Spectral flatness
  [122]     Spectral rolloff (norm, roll_percent=0.85)
  [123]     Zero crossing rate
```

### Windowed Dataset

```
X_win:  float32 (n_windows, 3100)     [classical ML; 25x124 flattened]
        float32 (n_windows, 25, 124)  [CNN; reshape in train_cnn.py]
y_win:  object  (n_windows,)          Drug | Exhale | Inhale | Noise
groups: int     (n_windows,)          recording index 0..360
```

### ONNX Model I/O Contract

```
File:   results/inhaler_cnn.onnx
Opset:  17

Input  "features":  float32 (batch, 25, 124)   [dynamic batch]
Output "logits":    float32 (batch, 4)           [RAW LOGITS - apply softmax]

Class order (must match LabelEncoder):
  index 0 = Drug
  index 1 = Exhale
  index 2 = Inhale
  index 3 = Noise

Classical ML path: apply StandardScaler BEFORE ONNX inference.
  scaler_mean  = float32[124]  exported from training fold
  scaler_scale = float32[124]  exported from training fold
CNN path: no scaler needed (BatchNorm handles it).
```

---

## 14. Module Reference

### config.py

| Constant | Value | Description |
|---|---|---|
| `DATA_DIR` | `data/` | Dataset root; env: PRISM_DATA_DIR |
| `LIBROSA_SR` | 8000 | Target sample rate |
| `LIBROSA_N_FFT` | 256 | STFT window (32 ms) |
| `LIBROSA_HOP_LENGTH` | 64 | Frame hop (8 ms; 125 fps) |
| `LIBROSA_N_MFCC` | 40 | MFCC coefficients |
| `LIBROSA_N_FEATURES` | 124 | Features per frame |
| `WINDOW_SIZE` | 25 | Frames per window (200 ms) |
| `WINDOW_STRIDE` | 2 | Step (16 ms) |
| `NOISE_BUFFER` | 20 | Noise frames retained around events |
| `DRUG_MULTIPLIER` | 3.0 | Drug oversample factor |
| `LABEL_NAMES` | Drug/Exhale/Inhale/Noise | LabelEncoder order |
| `N_SPLITS` | 5 | GroupKFold folds |

### loader.py (Legacy)
CSV-based loader; 40-dim `[MFCC(19)|ZCR|RMS|dMFCC(19)]`. Maintained for compatibility.

### librosa_extractor.py (Primary)
124-dim extraction from WAV; `.npy` cache.
API: `load_recording_librosa` / `load_all_recordings_librosa`.

### feature_extractor.py
`trim_to_events` / `create_windows` / `balance_dataset` / `compute_delta`.

### model_cnn.py
`InhalerCNN` + `build_model()`. `model.export_onnx(path)` -> ONNX opset 17.

### train.py
`run_cross_validation(X, y, groups, le, ...)` - GroupKFold RF/SVM/XGBoost.

### train_cnn.py
`run_cnn_cv(X, y, groups, le, ...)` - GroupKFold CNN.
Reshapes `(N,3100)` -> `(N,25,124)`. AMP + early stopping. Exports best fold to ONNX.

### run_pipeline.py
Main orchestrator, Steps 0-9. CLI: `--fast`, `--xgb-only`, `--no-svm`, `--cnn`.

---

## 15. Model Registry and Versioning

### Current Artifacts

| Artifact | Location | Notes |
|---|---|---|
| `inhaler_cnn.onnx` | `results/inhaler_cnn.onnx` | 3-fold CNN; 0.8902 mean acc |
| `cv_results.csv` | `results/cv_results.csv` | Per-fold per-class metrics |
| XGBoost summary | `results/xg/summary_report.txt` | 5-fold; 0.8902 mean acc |

### Naming Convention
`inhaler_<arch>_v<major>.<minor>.onnx`
e.g. `inhaler_cnn_v1.0.onnx`, `inhaler_xgb_v1.0.onnx`

### Mobile Deployment Checklist

- [ ] ONNX opset = 17 verified
- [ ] Input/output shapes: (N,25,124) -> (N,4)
- [ ] scaler_mean.npy + scaler_scale.npy exported (classical ML path)
- [ ] ONNX Runtime version pinned in mobile build
- [ ] Inference time < 2 ms / window on target device
- [ ] Accuracy verified with onnxruntime on held-out set
- [ ] Model SHA-256 hash documented

---

## 16. Development Priority Order

```
Phase 1 - ML Research (IN PROGRESS)
  [x] Dataset loading + annotation parsing
  [x] librosa_extractor.py: 124 features, .npy cache
  [x] Windowed dataset: trim + sliding window + balance
  [x] GroupKFold CV: RF, SVM, XGBoost
  [x] InhalerCNN + ONNX export (opset 17)
  [x] Evaluation: confusion matrix, feature importance
  [ ] Hyperparameter sweep (optuna)
  [ ] Larger window experiments (WINDOW_SIZE 25->50)
  [ ] Scaler export (.npy) alongside ONNX for classical ML

Phase 2 - Mobile Application
  [ ] React Native scaffolding
  [ ] Native C++ DSP: STFT, MFCC, delta, spectral features
  [ ] ONNX Runtime (Android + iOS)
  [ ] StandardScaler normalisation from exported arrays
  [ ] Frame sequence reconstruction (majority vote)
  [ ] Session analytics engine
  [ ] Feedback UI + SQLite local storage

Phase 3 - ESP32 Firmware
  [ ] I2S driver: INMP441, 8 kHz, 16-bit, DMA
  [ ] RMS inhalation detector state machine
  [ ] BLE GATT profile + MTU fragmentation
  [ ] Binary packet assembly
  [ ] MPU6050 IMU integration
  [ ] Power management (deep sleep + wake)

Phase 4 - Cloud and Backend
  [ ] Firebase: Auth, Firestore, Functions, Storage
  [ ] Session document sync from mobile
  [ ] Daily adherence aggregation function
  [ ] Missed dose notification
  [ ] Clinician dashboard (web app)

Phase 5 - Personalized Baseline Engine
  [ ] Baseline Firestore schema
  [ ] Mahalanobis deviation scoring
  [ ] EMA update with quality gate
  [ ] Per-user trend analytics
```

---

## 17. Implementation Status

| Component | Status | Notes |
|---|---|---|
| Annotation loading | Complete | loader.load_annotation() |
| Librosa extraction | Complete | 124 features, .npy cache |
| Windowing + balancing | Complete | trim + windows + balance |
| Random Forest CV | Complete | GroupKFold, per-fold scaler |
| SVM CV | Complete | LinearSVC |
| XGBoost CV | Complete | 5-fold; 0.8902 mean acc |
| 1D CNN (InhalerCNN) | Complete | PyTorch, AMP, early stop |
| ONNX export | Complete | results/inhaler_cnn.onnx, opset 17 |
| ESP32 firmware | Not started | Phase 3 |
| BLE protocol | Not started | Phase 3 |
| React Native app | Not started | Phase 2 |
| Native C++ DSP | Not started | Phase 2 |
| ONNX Runtime mobile | Not started | Phase 2 |
| Personalized baseline | Not started | Phase 5 |
| Firebase backend | Not started | Phase 4 |
| Doctor dashboard | Not started | Phase 4 |

---

## Appendix A: Acoustic Properties of the Four Event Classes

| Event | Duration | Key Discriminating Features |
|---|---|---|
| Drug | ~300 ms | Flatness ~0.7-1.0; centroid ~0.5-0.8; rolloff near Nyquist; high ZCR |
| Inhale | 1-3 s | Centroid ~0.1-0.3; flatness ~0.1-0.4; negative MFCC deltas at onset |
| Exhale | 0.5-2 s | Similar to Inhale; slightly higher centroid; positive MFCC deltas |
| Noise | Variable | Low MFCC[0] (energy); moderate flatness; low centroid |

Primary challenge: **Inhale vs. Exhale** - both turbulent breath sounds.
Delta-MFCC captures directional flow transient differences that static MFCC cannot.

---

## Appendix B: Latency Budget (Mid-Range Android)

| Stage | Duration |
|---|---|
| BLE receive + fragment reassembly | <10 ms |
| DSP (STFT + 124 features) for 2 s segment | <20 ms |
| ONNX inference per window | <2 ms |
| Total inference for 2 s (~60 windows) | <120 ms |
| Session analytics | <5 ms |
| **Total: BLE receive -> feedback display** | **<200 ms** |

---

## Appendix C: FAQ

**Q: Why 8 kHz not 44.1 kHz?**
All inhaler sounds are below 4 kHz. 8 kHz halves BLE bandwidth, reduces DSP
time, and shrinks the feature cache 5x vs 44.1 kHz.

**Q: Why N_FFT=256 (32 ms window)?**
Drug actuation is ~300 ms. 32 ms gives ~9 frames per actuation, enough to capture
onset/offset transients. Larger windows blur these transients.

**Q: Why GroupKFold?**
Adjacent frames in a recording are highly correlated. Standard KFold would allow
frames from the same recording in both train and test, causing data leakage.

**Q: Why ONNX not TFLite or CoreML?**
ONNX Runtime is cross-platform (Android + iOS) from one model file. TFLite needs
TensorFlow; CoreML is iOS-only. ONNX is optimal for PyTorch dual-platform deployment.

**Q: Can the model be updated OTA?**
Yes. Download ONNX from Firebase Storage, verify SHA-256, check accuracy threshold,
hot-swap. Export scaler arrays alongside model and update together.

**Q: Why not run inference on ESP32?**
ESP32-WROOM has 520 KB SRAM. InhalerCNN needs ~1.3 MB for weights alone (float32).
Even int8-quantised, latency on an FPU-less MCU is unacceptable. Smartphones run
this model in <2 ms on NPU hardware accelerators.
