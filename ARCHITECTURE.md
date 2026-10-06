# PRISM - Pulmonary Response and Inhaler System Monitor
## Full System Architecture and Technical Reference

> **Document purpose:** Cross-team technical reference for hardware engineers,
> mobile developers, ML researchers, and cloud/backend engineers working on the
> end-to-end PRISM smart inhaler platform.
>
> **Last updated:** 2026-10-06
> **Status:** event detector implemented; post-event and anomaly research
> stages 1–8 implemented; inference contract V2 issued as `DRAFT_NOT_FROZEN`;
> hardware, mobile app, cloud and dashboard not started.

### Status vocabulary

These states are used throughout this document (AGENTS.md §18).

| State | Meaning |
|---|---|
| **IMPLEMENTED** | Code exists in this repository and is tested |
| **VALIDATED** | Implemented and supported by a recorded experiment (see `PRISM_RESEARCH_LOG.md`) |
| **EXPERIMENTAL** | Implemented, but the evidence does not support relying on it yet |
| **PLANNED** | Design intent only; no implementation |
| **NOT VALIDATED** | A design value or claim with no supporting evidence |
| **NOT SUPPORTED** | Not supported by evidence and must not be built. The available data have no ground truth for it |

### Sources of truth

| Topic | Authoritative source |
|---|---|
| Inference (detector → events → scoreability → features → score → output) | `results/v2_validation/inference_contract_v2.json`, `src/prism_inference.py` |
| Per-frame DSP | `src/librosa_extractor.py::extract_features_from_audio` |
| Research history, evidence and decisions | `PRISM_RESEARCH_LOG.md` (Entries 1–10) |
| Running the repository | `README.md` |

If this document disagrees with the contract or the code, the contract and the code are correct, and this document must be fixed.

### Revision 2026-10-06 (summary of corrections)
- **DSP (§6, §7, §8.6).**
  - **Pre-emphasis:** removed; the extractor has none.
  - **MFCC:** described as computed (power mel spectrogram, Slaney mel scale and normalisation, 80 dB floor relative to the whole input buffer).
  - **Deltas:** Savitzky–Golay with interpolated edges; delta-delta is a direct second-order filter, not delta of delta.
  - **Flatness:** computed on the power spectrum.
  - **ZCR:** uses 2048-sample frames.
  - **Frame count:** 1 + ⌊n/64⌋.
- **Inference flow (§8.6).** The frame-majority-vote reconstruction, session analytics and composite quality labels were replaced by the V2 inference contract, which is the implemented and verified flow.
- **Withdrawn design (§9, §10, §11).** The personalized baseline engine design was withdrawn (Mahalanobis on MFCCs, EMA, 1.5/3.0 deviation bands, GOOD/POOR composite labels). So were the quality fields of the session document and the quality/anomaly dashboard views. None has an evidential basis, and the contract forbids these outputs (Research Entries 2, 3 and 10).
- **New content.**
  - the anomaly research stages and their status (§9, §17);
  - integration constraints for hardware and mobile (§4, §5, §6);
  - the contract output schema (§13);
  - conformance artefacts (§15).
- **Unsupported claims corrected.** Acoustic value ranges and feature-importance rankings that the repository's evidence does not support were corrected or marked (§7, §8.9, Appendix A).

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
9. [Baseline and Anomaly Scoring](#9-baseline-and-anomaly-scoring)
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

PRISM is a research project on acoustic monitoring of pressurised Metered-Dose
Inhaler (pMDI) use. It covers two things: acoustic inhalation-event detection, and
consistency analysis of each inhalation against a baseline, which is planned to
become personalized.

The event detector is a 1D CNN (ONNX). It classifies overlapping 200 ms analysis
windows, one every 16 ms, into four acoustic event classes: **Drug, Exhale,
Inhale, Noise**. Inhale windows are grouped into inhalation events. Each event is
checked for scoreability, measured, and compared with a frozen baseline.

**Scientific constraint.** The dataset contains acoustic event annotations only.
It has no validated clinical inhalation-technique labels (Research Entry 2), so:
- PRISM makes no clinical "good/bad technique" judgement.
- An Inhale event is an acoustic event. It does not indicate correct technique.
- NORMAL would mean baseline-consistent and ANOMALY a deviation from the learned
  baseline, but no NORMAL/ANOMALY label is produced yet, because no threshold has
  been validated (§9).

### What PRISM Answers Per Recording (current state)

| Question | Mechanism | Status |
|---|---|---|
| Where is there an inhalation in the recording? | CNN windows → Inhale event grouping | IMPLEMENTED. Events agree with the annotations (259 of 260 annotated inhalations matched at IoU ≥ 0.5, mean IoU 0.876), but this is mostly in-sample for the detector (Entry 3) |
| Is the event measurable? | Usability rule v1, which decides scoreability | IMPLEMENTED (Entry 3); reproduced by the contract (Entry 10, G6b) |
| How far is the inhalation's spectrum from the reference baseline? | V2 features → frozen global baseline → `anomaly_score` | EXPERIMENTAL. `SCORE_ONLY`, no threshold; contract `DRAFT_NOT_FROZEN` (Entry 10) |
| Is the inhalation consistent with *this user's* baseline? | Personalized baseline (V3) | PLANNED. Needs longitudinal user/device identifiers |
| Was the inhaler actuated? Was there an exhale first? How were actuation and inhalation coordinated? | The detector emits Drug and Exhale windows | NOT VALIDATED. No event grouping, timing analysis or evaluation exists for Drug/Exhale, and the contract outputs none |
| Is technique clinically correct, or within a clinical range? | — | NOT SUPPORTED. There are no technique-quality labels |

---

## 2. Core Design Principles

### Mobile-First Edge Intelligence

The smartphone is the **primary compute platform**. All signal processing, feature
extraction, and ML inference execute on-device. The ESP32 is a **sensor and radio
peripheral only** - not a compute node.

```
ESP32                   Smartphone                  Cloud
-------                 --------------------        ----------------
Microphone ADC          DSP pipeline                Recording summaries
I2S read                ONNX Runtime inference      Adherence metrics
Energy threshold        Inference contract V2       Derived event data
Packet assembly         Feedback generation
BLE transmit            Local SQLite storage
```

### Privacy-by-Design

Raw inhalation audio **never leaves the user's smartphone**. Only derived
analytics are transmitted to the cloud, and no cloud ML inference is required.

### Assessment Layers

| # | Layer | Status |
|---|---|---|
| 1 | **Event detector:** a global CNN (ONNX) that classifies 200 ms windows as Drug / Exhale / Inhale / Noise | IMPLEMENTED; window-level cross-validation (§8.8) |
| 2 | **Post-event layer:** Inhale events, scoreability, event features | IMPLEMENTED (Entries 1, 3, 10) |
| 3 | **Global baseline and anomaly score:** frozen V2 baseline; `anomaly_score` as a distance | EXPERIMENTAL (Entry 10) |
| 4 | **Personalized baseline:** per-user statistics and adaptation (V3/V4) | PLANNED |
| 5 | **Threshold:** NORMAL / ANOMALY | Not started; no validated threshold methodology |

The layers do not combine into a "quality assessment". No quality score exists.

---

## 3. End-to-End System Diagram

```
+-------------------------------------------------------------------+
| HARDWARE LAYER                                    (PLANNED)       |
|                                                                   |
| INMP441 I2S Microphone (24-bit, 8 kHz, mono)                     |
|   |                                                               |
|   v                                                               |
| ESP32-WROOM-32                                                    |
|   +-- I2S DMA read -> ring buffer                                 |
|   +-- RMS energy threshold detector (10 ms windows)              |
|   +-- [Optional] MPU6050 IMU - orientation metadata              |
|   +-- [Optional] Actuation switch - valve trigger                |
|   +-- int16 PCM (s24 >> 8) packet assembly + BLE GATT            |
|             |                                                     |
|             | Bluetooth Low Energy 5.0 (MTU-fragmented binary)   |
+-------------|-----------------------------------------------------+
              |
+-------------v-----------------------------------------------------+
| MOBILE APPLICATION LAYER (PLANNED; reference implementation:      |
|                           src/prism_inference.py)                 |
|                                                                   |
| BLE Manager -> fragment reassembly -> packet deserialization      |
|         |                                                         |
|         v                                                         |
| Input check: 8 kHz exactly, float = int16 / 32768, no resampling  |
| or preprocessing (otherwise INPUT_ERROR)                          |
|         |                                                         |
|         v                                                         |
| DSP (Native C++) mirroring librosa_extractor.py exactly (§6):     |
|   STFT 256/64 Hann, centred -> MFCC[40] Delta[40] Delta2[40]      |
|   Centroid[1] Flatness[1] Rolloff[1] ZCR[1] -> float32[124]/frame |
|         |                                                         |
|         v                                                         |
| Sliding window: 25 frames (200 ms), stride 2 frames (16 ms)      |
|         |                                                         |
|         v                                                         |
| ONNX Runtime: inhaler_cnn.onnx                                    |
|   float32[N,25,124] -> logits[N,4] -> softmax -> argmax          |
|         |                                                         |
|         v                                                         |
| Inference contract V2 (§8.6):                                     |
|   Inhale window grouping -> events                                |
|   Scoreability (usability rule v1)                                |
|   V2 features on each event segment + mean_rms level channel      |
|   Frozen global baseline -> z -> anomaly_score (SCORE_ONLY)       |
|   -> contract output JSON (§13)                                   |
|         |                                                         |
| Feedback UI (display rules §9.4) + Local SQLite Storage           |
+---------|---------------------------------------------------------+
          | HTTPS / Firebase SDK (derived data only)
+---------v---------------------------------------------------------+
| CLOUD LAYER (Firebase)                            (PLANNED)       |
| Firestore + Auth + Functions + Storage                            |
+---------+---------------------------------------------------------+
          |
+---------v---------------------------------------------------------+
| DOCTOR DASHBOARD (PLANNED): adherence | inhalation event history  |
+-------------------------------------------------------------------+
```

---

## 4. Hardware Layer

**Status: PLANNED.** Only a specification exists; no firmware has been written.
No PRISM-hardware audio has been recorded or evaluated.

### Components

| Component | Role |
|---|---|
| ESP32-WROOM-32 | Main MCU - sensing, event detection, BLE |
| INMP441 I2S Microphone | 24-bit digital MEMS microphone |
| Li-Ion Battery | 500-1000 mAh portable power |
| TP4056 | USB charging + battery protection |
| MPU6050 IMU (optional) | Orientation metadata |
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

The INMP441 outputs 24-bit left-justified samples in a 32-bit DMA word.
- **24-bit value:** firmware right-shifts by 8 bits, so `s24 = raw32 >> 8`.
- **For transmission:** the BLE packet carries 16-bit PCM, `int16 = s24 >> 8`. This is the full-scale mapping the contract requires (float = int16 / 32768 ≈ s24 / 2²³).
- **No extra digital gain.** Any other digital gain changes the `mean_rms` level channel. The V2 anomaly score is gain-invariant (Entry 10, G4a).

### Inhalation Detection State Machine (Firmware)

```
RMS(window=10 ms=80 samples) = sqrt(mean(x[n]^2))

IDLE      -> RMS > ONSET_THRESHOLD (0.01)       -> RECORDING
RECORDING -> accumulate into ring buffer
RECORDING -> RMS < OFFSET_THRESHOLD for >200ms  -> END
END       -> package binary packet, BLE transmit -> IDLE
```

The thresholds 0.01 / 0.005 and the 200 ms hold-off are design values (NOT VALIDATED).
The ESP32 does NOT perform MFCC, spectrogram, ML inference, or cloud comms.

### Buffer: 5 s x 8000 x 2 bytes = 80 KB (fits in 520 KB SRAM)
### Power: deep sleep after 30 s idle; active current target < 80 mA

### Integration Constraints From the Inference Contract

These constraints follow from `inference_contract_v2.json`. None has been tested on hardware.

| Constraint | Consequence |
|---|---|
| Sample rate must be exactly 8000 Hz | Any other rate returns `INPUT_ERROR` / `unsupported_sample_rate`; the phone does not resample |
| Events whose start ≤ 0.008 s, or whose end ≥ clip duration − 0.008 s, are `NOT_SCOREABLE` (`recording_boundary`) | A clip that starts exactly at the RMS onset, with no pre-trigger audio, will probably produce an inhalation that touches the clip start and is not scored. Clips need pre-trigger and post-offset audio from the ring buffer |
| Reference recordings last 6.5–12.5 s; the hardware buffer holds 5 s | Detector behaviour and boundary censoring on 5 s clips are untested. Usable reference inhalations last 0.50–3.29 s (median 1.63 s) |
| The baseline was fitted on reference-dataset audio | `input_domain = "prism_hardware"` gives `baseline_domain_validated = false`. A microphone or enclosure change acts like a spectral tilt (a tilt a = 0.5 shifts z(spectral_centroid_mean) by a median 2.49), so hardware scores are not interpretable until a hardware reference study exists |

---

## 5. BLE Transmission Protocol

**Status: PLANNED.** This is a specification only.

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
    uint8_t  audio_data[];     // raw PCM int16 LE, 8 kHz mono (int16 = s24 >> 8)
};
```

`timestamp_unix + timestamp_ms` of the clip start become the contract input's
`recorded_at` (ISO 8601). Absolute event time = `recorded_at + start_time`.

### Fragmentation (audio > 500 bytes)

```
byte 0: 0xFF (fragment marker)
byte 1: total fragment count
byte 2: current fragment index (0-based)
byte 3: 0x00 (reserved)
bytes 4+: payload chunk
Reassembly on mobile -> complete packet to DSP pipeline
```

### Open Issues (to be resolved by the hardware/mobile owners)
1. **`audio_len` is too small.** As a `uint16` it holds at most 65,535 bytes, which is 4.1 s of 8 kHz int16 audio, but the 5 s buffer is 80,000 bytes. Either the field or the clip length must change.
2. **`event_type` cannot be determined on the device.** The firmware only runs an RMS threshold, so it cannot tell inhalation, exhale-only and noise apart. Classification happens on the phone (§8.6).

---

## 6. Mobile Application Intelligence Layer

**Status: PLANNED.** No app code is in this repository. The reference implementation of
everything after BLE reassembly is `src/prism_inference.py`.

### DSP Parameter Parity (frozen at ONNX export - cannot change without retraining)

Verified against `src/librosa_extractor.py` and the installed librosa 0.11.0. The
same values are in `inference_contract_v2.json` → `event_detector.frame_features`.

| Stage | Parameter | Value |
|---|---|---|
| Input | Sample rate | 8000 Hz exactly; **no resampling** (otherwise `INPUT_ERROR`) |
| | Scaling | float32 = int16 / 32768, mono |
| | Preprocessing | **None**: no pre-emphasis, normalisation, filtering, gain control, trimming or padding |
| STFT | N_FFT / HOP | 256 samples (32 ms) / 64 samples (8 ms, 125 frames/s) |
| | Window | Hann, periodic (librosa default) |
| | Centring | `center=True`, zero (constant) padding of 128 samples on each side |
| | Frame count | 1 + ⌊n_samples / 64⌋ (all features truncated to the MFCC frame count) |
| MFCC [0..39] | Spectrum | Mel **power** spectrogram (\|STFT\|²), computed by `librosa.feature.mfcc` from the waveform with the same STFT settings |
| | Mel bank | 128 bands, 50–4000 Hz, Slaney mel scale (`htk=False`), Slaney area normalisation |
| | Log | `power_to_db`: 10·log10(max(S, 1e-10)), ref 1.0, then **floored at (maximum over the whole input buffer − 80 dB)** |
| | DCT | Type II, orthonormal, first 40 coefficients, no liftering |
| Delta [40..79] | Filter | Savitzky–Golay (`scipy.signal.savgol_filter`), width 9, polyorder 1, deriv 1, `mode="interp"`. Interior frames = Σₙ n·(f[t+n] − f[t−n]) / 60; the first and last 4 frames differ from that formula |
| Delta² [80..119] | Filter | Savitzky–Golay width 9, polyorder 2, **deriv 2 applied to the MFCCs** (not delta of delta), `mode="interp"` |
| Centroid [120] | Definition | Σ f_k·\|S_k\| / Σ \|S_k\| on the magnitude spectrum, f_k = k·8000/256, divided by 4000 |
| Flatness [121] | Definition | On the **power** spectrum P = max(\|S\|², 1e-10): exp(mean(log P)) / mean(P) |
| Rolloff [122] | Definition | Smallest bin-centre frequency at which the cumulative magnitude reaches 0.85 × total, divided by 4000 |
| ZCR [123] | Framing | **2048-sample frames**, hop 64, centred with edge-value padding (1024 samples per side) |
| | Definition | Mean over the frame of sign-change indicators; \|x\| ≤ 1e-10 counts as 0, and 0 counts as positive |
| Output | Type | float32[n_frames, 124]. If any value is non-finite, extraction fails and the recording returns `INPUT_ERROR` / `detector_feature_extraction_failed` |
| Windows | WINDOW_SIZE / STRIDE | 25 frames (200 ms) / 2 frames (16 ms). Window i covers frames [2i, 2i+25) and time [0.016·i, min(0.016·i + 0.2, duration)] |

**Parity hazards.**
- **Whole-buffer dependence of MFCCs.** The 80 dB floor uses the loudest value in the entire buffer, so the MFCCs of every frame depend on the whole clip. The app must extract features over the full clip in one pass, as the reference does. Streaming or chunked extraction will differ wherever the floor is active.
- **Full 124-feature extraction on the event segment.** V2 event features are computed by running the same extractor again on each event's waveform segment (§8.6 step [6]). They are not averages of the full-clip frame features.

**Validation checkpoints.** Compare against the following reference outputs:
1. `results/recording_runs/<recording>/frame_features.csv`, per-frame DSP. The README target is MAE < 0.001.
2. `window_predictions.csv`, the ONNX logits and probabilities per window.
3. `contract_output.json`, the final contract output.
4. The golden vectors in `results/v2_validation/golden/`, with the tolerances in §15.

Reproduce with `venv/Scripts/python.exe results/recording_runs/run_recording.py <wav> --input-domain reference_dataset`.

### Component Responsibilities

| Component | Technology |
|---|---|
| BLE GATT client | react-native-ble-plx |
| Packet reassembly | Native C++ |
| DSP pipeline (124 features) | Native C++ mirroring `librosa_extractor.py` (table above) |
| ONNX Runtime | Android: onnxruntime-android / iOS: onnxruntime-objc |
| Inference contract V2: grouping, scoreability, V2 features, frozen baseline, output JSON | Native C++ or JS, mirroring `src/prism_inference.py` |
| Frozen V2 baseline | `v2_baseline.json`, shipped with the app; loaded and validated, never fitted on the device |
| Result display | JavaScript / React Native; follows the display rules in §9.4 |
| Local storage | SQLite (react-native-sqlite-storage) |
| Cloud sync | Firebase JS SDK |

---

## 7. ML Pipeline

This repository is the research and training environment. It produces the ONNX
event detector and the inference-contract artefacts embedded in the mobile app.

### Dataset

- **Recordings:** 361 WAVs, 8 kHz, 16-bit mono, 6.46–12.51 s long (median 12.0 s), about 72 min in total (Entry 3).
- **Recording dates and sessions:** 10 dates in 2018. Recordings are grouped into 18 "sessions", i.e. recording sittings separated by gaps of more than 25 min (Entry 4).
- **Annotation file:** `data/annotation.csv` (no header; columns filename, label, start_sample, end_sample). It covers 301 of the 361 recordings, in 1,162 rows; after cleaning, 1,159 rows (Entry 3).
- **Labels:** Drug, Exhale, Inhale, Noise.
- **No subject, protocol or technique metadata exist** (Entry 2).

### Raw Label Distribution (Frame Level, Before Balancing)

| Class | Frames | % | Description |
|---|---|---|---|
| Drug | 7,817 | 1.4% | Aerosol actuation (annotated duration median 0.47 s) |
| Exhale | 73,222 | 13.3% | Exhalation (annotated median 1.47 s) |
| Inhale | 49,310 | 9.0% | Inhalation airflow (annotated median 1.46 s) |
| Noise | 418,436 | 76.2% | Ambient, silence, handling |
| **Total** | **548,785** | | 361 recordings x ~1500 frames each |

Annotated durations come from the cleaned `annotation.csv` (Appendix A).

### Pipeline Commands

```bash
python src/run_pipeline.py              # RF + SVM + XGBoost, 5-fold
python src/run_pipeline.py --xgb-only  # XGBoost only, 5-fold
python src/run_pipeline.py --cnn       # 1D CNN, exports ONNX
python src/run_pipeline.py --fast      # 3-fold XGBoost smoke test
python src/run_pipeline.py --no-svm    # RF + XGBoost
```

The anomaly-research commands (Stages 1–8) are listed in `README.md`.

---

### Step 0: Annotation Loading (loader.load_annotation)

```python
ann = load_annotation()
# pd.DataFrame: [filename, label, start_sample, end_sample]
# No header in CSV; label case normalised ("inhale" -> "Inhale")
```

---

### Step 1: Feature Extraction (librosa_extractor)

One-time per recording; cached to `data/extracted/<base>/features.npy`, which is created on demand.

#### 1a. Audio Loading

```python
audio, sr = librosa.load(wav_path, sr=None, mono=True)   # load_audio(): source rate kept
features  = extract_features_from_audio(audio, sr)       # resamples to 8 kHz if sr != 8000
# All dataset WAVs are already 8 kHz. The inference contract never resamples (§6).
```

#### 1b. STFT

```python
S = np.abs(librosa.stft(audio, n_fft=256, hop_length=64))
# Window:   Hann, 256 samples (32 ms); center=True, zero padding 128 each side
# Hop:      64 samples (8 ms); 75% overlap
# Output:   S shape (129, n_frames)
#   129 = N_FFT/2 + 1 frequency bins
#   n_frames = 1 + floor(len(audio)/64)   e.g. 96,000 samples -> 1,501 frames
#   freq resolution: 8000/256 = 31.25 Hz/bin
#   time resolution: 64/8000  = 8 ms/frame  (125 fps)
# S is shared by centroid, flatness and rolloff. MFCC computes its own power
# mel spectrogram from the waveform with the same STFT parameters.
```

#### 1c. MFCC

```python
mfcc = librosa.feature.mfcc(
    y=audio, sr=sr, n_mfcc=40, n_fft=256, hop_length=64,
    n_mels=128, fmin=50.0, fmax=4000.0
)  # shape: (40, n_frames)
#
# Steps (librosa 0.11.0 defaults):
#   P        = |STFT(audio)|^2                                   (power, 129 bins)
#   mel_spec = slaney_mel_filterbank(128, 50-4000 Hz) @ P        -> (128, n_frames)
#   log_mel  = 10*log10(max(mel_spec, 1e-10))
#   log_mel  = max(log_mel, log_mel.max() - 80)                  (top_db over the WHOLE input)
#   mfcc     = DCT-II(log_mel, norm="ortho")[:40]
```

#### 1d. Delta Coefficients (Savitzky–Golay)

```python
delta  = librosa.feature.delta(mfcc, width=9)           # (40, n_frames)
delta2 = librosa.feature.delta(mfcc, width=9, order=2)  # (40, n_frames)
#
# librosa.feature.delta = scipy.signal.savgol_filter(data, 9, polyorder=order,
#                                                    deriv=order, mode="interp")
# Interior frames of delta equal the regression filter
#   delta[t] = sum_{n=1}^{4} n * (f[t+n] - f[t-n]) / 60
# The first/last 4 frames use a polynomial fit ("interp"), not mirror padding.
# delta2 is the second-derivative filter applied to the MFCCs, NOT delta(delta).
```

#### 1e. Spectral Features

```python
centroid = librosa.feature.spectral_centroid(S=S, sr=sr) / (8000/2)
# sum(f_k * S_k) / sum(S_k) on magnitude, normalised by Nyquist -> [0, 1]

flatness = librosa.feature.spectral_flatness(S=S)
# power = 2.0 by default: P = max(S**2, 1e-10)
# exp(mean(log P)) / mean(P);  1.0 = white noise, 0.0 = pure tone

rolloff = librosa.feature.spectral_rolloff(
    S=S, sr=sr, roll_percent=0.85
) / (8000/2)
# Smallest bin frequency f* where cumsum(S) >= 0.85 * sum(S), normalised by 4000

zcr = librosa.feature.zero_crossing_rate(audio, hop_length=64)
# frame_length = 2048 (librosa default), center=True with edge-value padding
# mean of sign-change indicators over each 2048-sample frame
```

Value ranges measured on the 318 usable inhalation events are in Appendix A. No
per-class measurements exist for Drug, Exhale or Noise in this repository.

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
]).T  # -> (n_frames, 124) float32   (each block truncated to the MFCC frame count)

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
# Removes long silent stretches at the start/end of each recording
# Recordings with no non-Noise frame are kept unchanged
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

**Post-balance (361 recordings, 170,323 windows; `results/xg/summary_report.txt`):**

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

### Step 4: Cross-Validation (GroupKFold)

```
groups:  recording index per window (0..360)
         all windows from recording i -> groups[j] = i
         no recording in both train and test in any fold

Classical ML (RF, SVM, XGBoost): GroupKFold k=5 (N_SPLITS)
1D CNN (results/cv_results.csv):  GroupKFold k=3
```

**Deployed CNN = best CNN fold.** `train_cnn.run_cnn_cv` exports the fold model with the highest test accuracy: fold 1 of 3, accuracy 0.8954.
- **Training data:** that model was trained on about two-thirds of the 361 recordings.
- **Fold membership:** which recordings were in its training fold was not saved.
- **Consequence:** event/annotation agreement on the reference dataset is mostly in-sample (Entry 3).

**Per-fold StandardScaler (classical ML: RF, SVM, XGBoost):**

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train ONLY
X_test_sc  = scaler.transform(X_test)        # no leakage
```

**CNN path:** No StandardScaler; BatchNorm1d handles normalisation internally.

> **Classical ML deployment (not used; no classical model is deployed):** export
> `scaler_mean.npy` and `scaler_scale.npy` (float32[124]) from the final fold and
> apply `(features - mean) / scale` before inference. See `train.py` for the
> scaler fit.

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

| Model | Format | File | Size | Status |
|---|---|---|---|---|
| 1D CNN (InhalerCNN) | ONNX opset 17 | `results/inhaler_cnn.onnx` | 1,181,826 bytes | **Primary**; IMPLEMENTED. SHA-256 `2e4e72d3ca040b15718ad6b09ae4ac3069c38f5c1a389327bf6b38c4e2903725` |
| XGBoost | ONNX (onnxmltools) | TBD | TBD | Not exported |
| Random Forest | sklearn | - | - | Research only |

Inference time on a phone has not been measured. "<2 ms per window" is a target (Appendix B).

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
- **Features as channels:** each `Conv1d` filter learns cross-feature relationships over time.
- **Residual skip connections:** stabilise training of the 7-layer depth.
- **`AdaptiveAvgPool1d(1)`:** collapses the temporal axis.
- **Bottleneck head:** `256->64->4`.
- **`do_constant_folding=True` at ONNX export.**

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

Illustrative only; no app code exists yet.

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

Illustrative only.

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

### 8.6 Per-Recording Inference Flow (Inference Contract V2)

**Status: IMPLEMENTED** in `src/prism_inference.py`. It reproduces the Stage 1 events,
features and scoreability exactly on all 361 reference recordings (Entry 10, G6b).
- **Contract:** `prism-inference-v2.0`, status `DRAFT_NOT_FROZEN`. The interface is fixed; the V2 representation and baseline may still change (§9).
- **Authoritative text:** `results/v2_validation/inference_contract_v2.json`.

```
Recording received (int16 PCM, 8 kHz, mono) + caller metadata (recorded_at, input_domain)
     |
     v [1] Input check  (INPUT_ERROR on failure; no events)
  x = int16 / 32768  (float32); 2-D input is averaged over channels
  errors: unsupported_sample_rate (!= 8000) | invalid_shape | empty_audio |
          nonfinite_audio | amplitude_out_of_range (|x| > 1) |
          shorter_than_one_detector_window (< 1536 samples)
     |
     v [2] DSP: 124 features per frame over the whole clip (§6 table)
  error: detector_feature_extraction_failed (non-finite features)
     |
     v [3] Detector
  windows: features[2i : 2i+25]  -> float32[N, 25, 124]
  logits = inhaler_cnn.onnx(windows); probs = softmax(logits); label = argmax
  window i spans [0.016*i, min(0.016*i + 0.2, duration)]
     |
     v [4] Inhale event grouping  (TemporalGroupingConfig defaults; no smoothing,
                                   no minimum duration, no confidence filter)
  Inhale windows in time order; a window joins the current event while its
  start <= the latest end of the event's windows (overlapping 0.2 s windows
  bridge up to 12 strides); otherwise a new event starts.
  start_time = first window start; end_time = max window end
  detector_confidence = mean P(Inhale) over the event's windows; max also reported
  No events -> recording_status NO_INHALATION_DETECTED (a recording state,
               never an anomalous event)
     |
     v [5] Scoreability (usability rule v1; Entry 3)
  NOT_SCOREABLE reasons (any number):
    short_duration      round(duration_s, 6) < 0.5 s
    close_neighbor      gap to another event in the clip < 0.2 s
    recording_boundary  start_time <= 0.008 s or end_time >= duration - 0.008 s
    nonfinite_feature   any V2 feature or mean_rms non-finite / extraction failed
     |
     v [6] Event features (on the event segment, not the clip frames)
  segment = x[floor(start_time*8000) : ceil(end_time*8000)]
  run the same 124-feature extractor on the segment; over its frames take:
    spectral_centroid_mean, spectral_flatness_mean,       (mean, float32)
    spectral_centroid_std,  spectral_rolloff_std          (population std, ddof 0)
  mean_rms = mean of RMS over 256-sample frames every 64 samples from the
             segment start (partial final frames included); level channel,
             NOT in the score
     |
     v [7] Score (scoreable events only)
  z_j = (x_j - center_j) / scale_j   (frozen v2_baseline.json; scale = 1.4826*MAD)
  anomaly_score = sqrt(mean(z_j^2)) over the 4 V2 features
  No threshold. No NORMAL/ANOMALY.
     |
     v [8] Output JSON (schema §13; validated by prism_inference.validate_output)
  recording_status: EVENTS_DETECTED | NO_INHALATION_DETECTED | INPUT_ERROR
  per event: times, confidence, status SCORE_ONLY | NOT_SCOREABLE, reasons,
             anomaly_score, feature_values, feature_z_scores, mean_rms
```

**Worked example.** `results/recording_runs/rec2018-01-22_17h41m49.809s/` shows the result for one reference recording:
- one event at 0.688–2.232 s, `SCORE_ONLY`;
- `anomaly_score` 0.555, which is in-sample because this event helped fit the deployment baseline; held out by session it is 0.571;
- every intermediate stage saved for parity testing.

**Not part of the contract.** The following are withdrawn or not supported:
- **Frame-sequence reconstruction by majority vote** (centre frame t+12). It gives different event boundaries from step [4]. Do not use it.
- **Drug / Exhale events.** The detector emits Drug and Exhale windows, but the contract defines no Drug or Exhale events. Their window labels agree with annotations on reference recordings (mostly in-sample), but event grouping, onset timing and evaluation for these classes do not exist.
- **Session analytics:**
  - `drug_dur`, `coord_delay`, `pre_exhale_detected` — NOT VALIDATED;
  - `insufficient_inhale` (< 1.0 s), `late_actuation` (> 0.5 s), `missed_dose` — NOT SUPPORTED. These thresholds have no evidential basis, and technique error codes are forbidden by the contract.
- **Composite labels** GOOD / POOR / GOOD_BUT_INCONSISTENT / ABNORMAL / MISSED_DOSE — NOT SUPPORTED. See §9.5.

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

### 8.8 Cross-Validation Results (window level)

**CNN 3-fold GroupKFold** (source: `results/cv_results.csv`):

| Fold | Accuracy | Drug F1 | Exhale F1 | Inhale F1 | Noise F1 |
|---|---|---|---|---|---|
| 1 | 0.8954 | 0.9507 | 0.8742 | 0.9080 | 0.8696 |
| 2 | 0.8847 | 0.9460 | 0.8604 | 0.8865 | 0.8696 |
| 3 | 0.8906 | 0.9517 | 0.8655 | 0.8838 | 0.8731 |
| **Mean** | **0.8902** | **0.9495** | **0.8667** | **0.8928** | **0.8708** |

**XGBoost 5-fold** (source: `results/xg/summary_report.txt`): 0.8902 +/- 0.0044

These are window-level metrics on held-out folds. The deployed model is fold 1
(§7 Step 4).

**Event level (Entry 3; mostly in-sample for the deployed model):**
- 364 Inhale events detected.
- 259 of 260 unique annotated inhalations matched at IoU ≥ 0.5 (mean IoU 0.876).
- 0 events shorter than 0.5 s overlap an annotated inhalation.

### 8.9 Feature Importance (XGBoost)

The saved artefact `results/xg/feature_importance_xgboost.png` labels all of its
top-15 features as MFCC-derived (mfcc_0, mfcc_2, mfcc_7, mfcc_17 at several
window positions). A previous revision of this document ranked spectral flatness
first; the saved artefacts do not support that ranking.

`summary_report.txt` also describes 7-frame / 868-feature windows, which
contradicts its own 3,100-feature input. The XGBoost importance artefacts should be
regenerated before any feature-importance claim is made.

---

## 9. Baseline and Anomaly Scoring

PRISM evaluates an inhalation against a baseline (AGENTS.md §1, §10). A global
V2 baseline exists and is EXPERIMENTAL. The baseline is not yet personal: the
dataset has no user identifiers (Entry 2). Threshold selection has not started.

### 9.1 Research Stages and Status

| Stage | Content | Status | Evidence |
|---|---|---|---|
| 1 | Inhale-event dataset; usability rule v1 (318 usable / 46 excluded of 364) | IMPLEMENTED | Entry 3, `results/inhale_dataset/` |
| 2 | Feature analysis; V1 feature set (7 features) | IMPLEMENTED | Entry 4, `results/feature_analysis/` |
| 3 | V1 robust baseline (median / 1.4826·MAD on the first 20 usable events) | IMPLEMENTED; superseded | Entry 5 |
| 4 | Candidate combined scores (mean\|z\|, rms_z, max\|z\|) | IMPLEMENTED | Entry 6 |
| 5 | Baseline strategies across sessions. First-20 calibration inflates other-session scores about 2×; a leave-one-session-out global baseline (strategy C) generalizes in scale (ratio about 1.1) | IMPLEMENTED; first-20 path stopped | Entry 7 |
| 6 | Natural population separation and controlled perturbations. Excluded events score higher, mainly because of duration and segmentation | IMPLEMENTED | Entry 8 |
| 7 | Representation ablation. Duration, `mean_rms` and `spectral_flatness_std` removed from the score; 4-feature V2 proposed | IMPLEMENTED | Entry 9 |
| 8 | V2 validation against a pre-registered gate; inference contract | IMPLEMENTED. Gate **failed** on G4c only, so `DRAFT_NOT_FROZEN` | Entry 10 |
| — | Threshold methodology, NORMAL / ANOMALY | Not started | — |
| — | V3 personalized baseline; V4 adaptive personalization | PLANNED | Entries 3, 7, 8 |

### 9.2 Current Baseline (V2, global, frozen)

**File:** `results/v2_validation/v2_baseline.json`
- **Identity:** `baseline_id` `prism-v2-global-2026-09-30`, SHA-256 `0eaea4b8…32d3`.
- **What it is:** a per-feature median and 1.4826·MAD over all 318 usable events in 18 sessions of the reference dataset.
- **How it is used:** loaded and validated, and never fitted or updated at inference. Any change is a new `baseline_id` and needs its own validation.

| Feature (order is part of the contract) | Centre | MAD | Scale |
|---|---|---|---|
| spectral_centroid_mean | 0.37053 | 0.018796 | 0.027867 |
| spectral_flatness_mean | 0.13224 | 0.020037 | 0.029707 |
| spectral_centroid_std | 0.039401 | 0.0060552 | 0.0089774 |
| spectral_rolloff_std | 0.080710 | 0.011403 | 0.016906 |

**Score:** `anomaly_score = sqrt(mean(z²))`, where `z_j = (x_j − centre_j) / scale_j`. Higher
means further from the baseline median. It is a distance, not a probability.

**What is known (Entry 10; held-out by session, 318 usable events):**
- **Score distribution on unseen sessions:** 5th percentile / median / 95th percentile = 0.42 / 1.00 / 1.75.
- **Session dependence remains:** ε² 0.130, and session medians span 0.64–1.31 (2.05×).
- **Gain:** invariant (95th percentile |Δ| ≤ 4.9×10⁻⁴).
- **Noise and tilt:** the score rises monotonically with added noise and spectral tilt (≥ 99.7% of events). These are responses to artificial perturbations, not evidence of detecting meaningful inhalation deviations.
- **Failed criterion G4c:** at strong spectral tilt (a = 0.9), `spectral_flatness_mean` and `spectral_centroid_std` change direction inconsistently across sessions. A microphone or enclosure change is a spectral-shape change, so hardware transfer is expected to be problematic for these two features.
- **Level channel:** `mean_rms` is reported separately and is excluded from the score. Including it doubled session dependence and made the score gain-dependent.

**Decision pending (project owner, Entry 10):** either (a) accept V2 as a documented
exception for MVP engineering, with `anomaly_score` experimental and not shown to
users, or (b) keep the interface and replace the representation after a new
pre-registered evaluation.

### 9.3 What `anomaly_score` Is Not

It is not:
- a NORMAL/ANOMALY decision, because no threshold exists;
- a quality percentage;
- a technique rating or error code;
- a clinical or alert state.

The contract lists these as `forbidden_derived_outputs`.

### 9.4 Display Rules for the App

| Output | May be shown to users? |
|---|---|
| `recording_status`, event times and durations | Yes |
| `status` and `not_scoreable_reasons` (e.g. "inhalation detected, too short to analyse") | Yes; describe them as measurement conditions, not technique errors |
| `NO_INHALATION_DETECTED` | Yes, as "no inhalation detected". Never as an anomalous inhalation |
| `anomaly_score`, `feature_z_scores` | No. They are internal and experimental while the contract is `DRAFT_NOT_FROZEN`, and they must not be presented as a health signal |
| `mean_rms` | No; it is uncalibrated loudness, for diagnostics only |
| `detector_confidence` | Diagnostics only |

### 9.5 Withdrawn Design (previous revision of this section)

The following design was never implemented. Research Entry 3 recorded it as
legacy and not adopted:

- **A per-user Firestore baseline**, maintained with EMA updates (α = 0.1) and gated on `global_quality_score ≥ 0.7`, `drug_detected` and `inhale_duration ≥ 0.8 s`.
- **A Mahalanobis distance on mean MFCC vectors.**
- **A weighted deviation:** `0.5·d + 0.3·|z_dur| + 0.2·|z_energy|`, banded at 1.5 / 3.0 into within_baseline / mild / significant.
- **Composite labels:** GOOD / POOR / GOOD_BUT_INCONSISTENT / ABNORMAL / MISSED_DOSE.

Why it was withdrawn:
- **No global quality score exists.**
- **No evidential basis:** the weights, bands and gates are unsupported.
- **Rejected inputs:** duration and absolute level were removed from the score for documented reasons (Entries 8–9).
- **Not personal:** the data have no user identity, so a per-user baseline cannot be evaluated.
- **Forbidden labels:** clinical/technique labels are forbidden by the scientific constraint (§1).

### 9.6 Planned Personalization (V3 / V4)

Before personalization can be built, the following are needed (Entries 7, 8, 10):

1. **Longitudinal recordings** with user, device and session identifiers. Ideally they include documented, deliberate acoustic variations, recorded as protocol conditions, not clinical labels.
2. **Reference recordings on PRISM hardware**, to quantify the domain shift and re-baseline for the device.
3. **A pre-registered evaluation**, which must define all of the following (AGENTS.md §10):
   - calibration samples;
   - feature set;
   - baseline statistics;
   - adaptation/update policy, conservative so that anomalies do not redefine the baseline;
   - threshold methodology;
   - handling of insufficient calibration data.

Stage 5 evidence relevant to the design:
- Deployable session-location normalization with 5 warm-up events did not improve on the global baseline.
- 10 warm-up events did improve it, but that result is sensitivity analysis only.

---

## 10. Cloud and Firebase Integration

**Status: PLANNED (not started).**

### Session Document (no raw audio transmitted)

There is one document per recording. It carries the contract output and adds no
derived quality fields.

```json
{
  "session_id":   "uuid-v4",
  "user_id":      "firebase-auth-uid",
  "device_id":    "esp32-mac",
  "recorded_at":  "2026-08-23T14:30:00Z",
  "inhaler_type": "pMDI",
  "app_version":  "1.0.0",
  "contract_version":      "prism-inference-v2.0",
  "baseline_id":           "prism-v2-global-2026-09-30",
  "detector_model_sha256": "2e4e72d3...3725",
  "input_domain":          "prism_hardware",
  "recording_status":      "EVENTS_DETECTED",
  "events": [
    {"event_id": 0, "start_time": 0.688, "end_time": 2.232, "duration_s": 1.544,
     "detector_confidence": 0.934, "status": "SCORE_ONLY", "not_scoreable_reasons": [],
     "anomaly_score": 0.555, "mean_rms": 0.177}
  ],
  "imu_orientation": {"inhaler_vertical": true, "tilt_degrees": 8.2}
}
```

- **`anomaly_score`** is stored for research analysis only. It must not drive alerts, notifications or clinician-facing quality displays (§9.4).
- **Removed from the previous revision:**
  - `event_classification`: `drug_detected`, `drug_duration_ms`, `inhale_duration_ms`, `coordination_delay_ms`, `pre_exhale_detected`;
  - `quality_assessment`: `global_score`, `composite_label`, `deviation_score`, `deviation_flag`.

  These are not validated or not supported (§8.6, §9.5).
- **IMU orientation** is optional metadata and has not been evaluated.

### Firestore Schema

```
users/{uid}/profile           demographics, inhaler type, prescriber
users/{uid}/sessions/{id}     one per recording (above doc)
users/{uid}/adherence/{date}  recordings scheduled / received
users/{uid}/baseline          RESERVED for a future personalized baseline (V3, PLANNED);
                              the current V2 baseline is a global file shipped with the app
clinicians/{cid}/patients/{uid}   read-only; requires consent
```

A received recording is not validated evidence that a dose was taken or inhaled.

### Cloud Functions

| Function | Trigger | Purpose | Status |
|---|---|---|---|
| `aggregateDailyAdherence` | Firestore write | Daily rollup of recordings received vs scheduled | PLANNED |
| `notifyMissedDose` | Scheduled | Push when no recording arrives in a scheduled window (schedule-based, not acoustic) | PLANNED |
| `generateClinicianReport` | HTTP callable | PDF/CSV export of recordings and events; no quality or anomaly labels | PLANNED |
| `detectTechniqueRegression` | — | Withdrawn: there is no validated technique measure or threshold | NOT SUPPORTED |

---

## 11. Doctor Dashboard

**Status: PLANNED (not started).**

| View | Source | Description |
|---|---|---|
| Adherence Calendar | `adherence` collection | Recordings received vs scheduled per day |
| Inhalation Event History | `sessions` collection | Recording status, detected events, durations, not-scoreable reasons |
| Export Panel | Cloud Functions | PDF / CSV download |

Removed from the previous revision:
- **Quality Trend:** no quality score exists.
- **Technique Heatmap:** Drug and Exhale events are not defined.
- **Anomaly Timeline of ABNORMAL sessions:** no threshold or label exists.

A research-only view of `anomaly_score` over time could be added for the
research team. It must not be shown as a clinical signal.

---

## 12. Repository Directory Layout

```
PRISM smart inhaler/
|-- AGENTS.md                <- Rules for agents (research and engineering discipline)
|-- ARCHITECTURE.md          <- This document
|-- CLAUDE.md                <- Claude Code instructions
|-- PRISM_RESEARCH_LOG.md    <- Append-only research history (Entries 1-10)
|-- README.md                <- Project overview and quickstart
|-- requirements.txt
|
|-- data/                    <- Dataset root (git-ignored)
|   |-- annotation.csv       <- Sample-level labels (301 of 361 recordings)
|   |-- rec<ts>.wav          <- 8 kHz 16-bit WAV mono (361)
|   |-- rec<ts>_{mfcc,cepst,cwt,spect,zcr}.csv  <- Legacy precomputed features
|   +-- extracted/rec<base>/features.npy       <- float32(n_frames,124); created on demand
|
|-- results/
|   |-- inhaler_cnn.onnx     <- Trained CNN; primary deployment artifact
|   |-- cv_results.csv       <- Per-fold per-class metrics (CNN, 3 folds)
|   |-- confusion_matrix.png, class_metrics.png, drug_stats.png, noise_confusion.png
|   |-- xg/                  <- XGBoost summary and figures
|   |-- post_event/          <- Detected Inhale events (364) + distributions
|   |-- inhale_dataset/      <- Stage 1: inhale_events_v1.csv, usability audit
|   |-- feature_analysis/    <- Stage 2: feature_selection_v1.json
|   |-- baseline_v1/         <- Stage 3
|   |-- scoring_v1/          <- Stage 4
|   |-- baseline_strategies/ <- Stage 5
|   |-- natural_population/  <- Stage 6
|   |-- representation_analysis/  <- Stage 7
|   |-- v2_validation/       <- Stage 8: contract, baseline, schemas, golden vectors, gate
|   |-- recording_runs/      <- Per-recording stage checkpoints for app parity
|   +-- phone_test/          <- Ad-hoc phone recording test (outside the contract domain)
|
|-- src/
|   |-- config.py            <- ALL paths and hyperparameters
|   |-- loader.py            <- Legacy CSV loader (40 features)
|   |-- librosa_extractor.py <- Primary extractor (124 features, .npy cache)
|   |-- feature_extractor.py <- Windowing, trimming, balancing
|   |-- dataset.py           <- Dataset assembly utilities
|   |-- train.py             <- GroupKFold CV: RF + SVM + XGBoost
|   |-- train_cnn.py         <- GroupKFold CV: 1D CNN (PyTorch + AMP + ONNX)
|   |-- model_cnn.py         <- InhalerCNN + ONNX export
|   |-- evaluate.py          <- Metrics, misclassification analysis
|   |-- visualize.py         <- Confusion matrix, feature importance plots
|   |-- run_pipeline.py      <- Detector training orchestrator
|   |-- post_event.py        <- ONNX wrapper, window predictions, event grouping, event measurements
|   |-- explore_inhalations.py        <- Dataset-wide event table
|   |-- inhale_dataset.py             <- Stage 1
|   |-- feature_analysis.py           <- Stage 2
|   |-- baseline_v1.py                <- Stage 3
|   |-- scoring_v1.py                 <- Stage 4
|   |-- baseline_strategies.py        <- Stage 5
|   |-- natural_population_analysis.py <- Stage 6
|   |-- representation_analysis.py    <- Stage 7
|   |-- v2_validation.py              <- Stage 8 (gate, contract artefacts)
|   +-- prism_inference.py            <- Inference contract V2 reference implementation
|
+-- tests/                   <- unittest suites (python -m unittest discover tests)
```

---

## 13. Data Schemas and Interfaces

### annotation.csv (no header row)

```
rec2018-01-22_17h41m33.475s.wav,Inhale,512,9216
rec2018-01-22_17h41m33.475s.wav,Drug,4096,6144
```

Loaded with `pd.read_csv(header=None)`.
Unannotated intervals default to "Noise" for detector training. Annotations are
not exhaustive, so "no annotation" is not a negative label (Entry 3).

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
SHA-256: 2e4e72d3ca040b15718ad6b09ae4ac3069c38f5c1a389327bf6b38c4e2903725

Input  "features":  float32 (batch, 25, 124)   [dynamic batch; raw, unscaled features]
Output "logits":    float32 (batch, 4)           [RAW LOGITS - apply softmax]

Class order (must match LabelEncoder):
  index 0 = Drug
  index 1 = Exhale
  index 2 = Inhale
  index 3 = Noise

CNN path: no scaler (BatchNorm folded into the exported graph).
Classical ML path (not deployed): StandardScaler before inference.
```

### Inference Contract Output (`prism-inference-v2.0`)

The full JSON Schema is `results/v2_validation/inference_output.schema.json`, and the executable
validator is `prism_inference.validate_output`. Key order is part of the contract.

| Field | Type | Meaning |
|---|---|---|
| `contract_version` | string | `prism-inference-v2.0` |
| `baseline_id` | string | `prism-v2-global-2026-09-30` |
| `detector_model_sha256` | hex string | SHA-256 of the ONNX model used |
| `recording_status` | enum | `EVENTS_DETECTED` \| `NO_INHALATION_DETECTED` \| `INPUT_ERROR` |
| `error` | enum \| null | Input error code (§8.6 [1]–[2]); null unless `INPUT_ERROR` |
| `input` | object | `sample_rate`, `n_samples`, `duration_s`, `input_domain` (`reference_dataset` \| `prism_hardware` \| `unknown`), `recording_id`, `recorded_at` (caller-supplied ISO 8601) |
| `baseline_domain_validated` | bool | true only for `reference_dataset` |
| `feature_order` | array | the 4 V2 features, in order |
| `n_events`, `n_scored` | int | event count; number of `SCORE_ONLY` events |
| `events[]` | array | chronological; fields below |
| `interpretation` | string | fixed SCORE_ONLY disclaimer |

| Event field | Type | Meaning |
|---|---|---|
| `event_id` | int | 0, 1, … in time order |
| `start_time`, `end_time`, `duration_s` | float (s) | relative to the recording start; multiples of 0.016 s (`end_time` capped at the duration) |
| `detector_confidence`, `detector_max_confidence` | float [0,1] | mean / max P(Inhale) over the event's windows |
| `window_count` | int | Inhale windows in the event |
| `status` | enum | `SCORE_ONLY` \| `NOT_SCOREABLE` |
| `not_scoreable_reasons` | array | `nonfinite_feature`, `short_duration`, `close_neighbor`, `recording_boundary`; empty if scored |
| `anomaly_score` | float ≥ 0 \| null | rms of the z-scores; null if not scoreable |
| `feature_values` | object \| null | the 4 V2 features; null only if non-finite |
| `feature_z_scores` | object \| null | z per V2 feature; null if not scoreable |
| `mean_rms` | float ≥ 0 \| null | level channel (uncalibrated), not in the score |

### Other Interfaces

| File | Content |
|---|---|
| `results/v2_validation/v2_baseline.json` | Baseline: `baseline_id`, `contract_version`, `features`, `parameters{feature: {center, mad, scale}}`, `mad_scale` 1.4826, `n_events` 318, `n_sessions` 18 |
| `results/v2_validation/v2_feature_schema.json` | V2 feature order, definitions, segment and frame parameters |
| `results/inhale_dataset/inhale_events_v1.csv` | Stage 1 table: 364 events with 13 descriptive features, flags, `usable`, `exclusion_reasons` |

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
| `N_SPLITS` | 5 | GroupKFold folds (classical ML) |

### Detector training

| Module | Content |
|---|---|
| `loader.py` (legacy) | CSV-based loader; 40-dim `[MFCC(19)\|ZCR\|RMS\|dMFCC(19)]`. Maintained for compatibility |
| `librosa_extractor.py` | `extract_features_from_audio(audio, sr)` (shared DSP), `extract_features(wav)`, `load_audio(wav)`, `load_recording_librosa`, `load_all_recordings_librosa`; `.npy` cache |
| `feature_extractor.py` | `trim_to_events`, `create_windows`, `create_feature_windows`, `balance_dataset`, `oversample_minority`, `normalize_windows`, `build_windowed_dataset`, `compute_delta` |
| `model_cnn.py` | `InhalerCNN` + `build_model()`. `model.export_onnx(path)` produces ONNX opset 17 |
| `train.py` | `run_cross_validation(X, y, groups, le, ...)`: GroupKFold RF/SVM/XGBoost |
| `train_cnn.py` | `run_cnn_cv(X, y, groups, le, ...)`: GroupKFold CNN; reshapes `(N,3100)` to `(N,25,124)`; AMP + early stopping; exports the best fold to ONNX |
| `run_pipeline.py` | Steps 0-9. CLI: `--fast`, `--xgb-only`, `--no-svm`, `--cnn` |

### Post-event and anomaly research

| Module | Content | Log |
|---|---|---|
| `post_event.py` | `OnnxEventClassifier`, `generate_window_predictions`, `TemporalGroupingConfig`, `group_inhale_events`, `extract_event_audio`, `analyze_inhalation`, `plot_diagnostic` | Entry 1 |
| `explore_inhalations.py` | Event table for all recordings: `results/post_event/inhalation_events.csv` | Entry 1 |
| `inhale_dataset.py` | `UsabilityRule` (usability rule v1), annotation audit, `inhale_events_v1.csv` | Entry 3 |
| `feature_analysis.py` | Distributions, redundancy, session stability, `feature_selection_v1.json` | Entry 4 |
| `baseline_v1.py` | `RobustBaseline`, `fit_baseline` (median / 1.4826·MAD), `recording_sessions` | Entry 5 |
| `scoring_v1.py` | Combined scores `mean_abs_z`, `rms_z`, `max_abs_z`; contributions | Entry 6 |
| `baseline_strategies.py` | Baseline strategies A–D across sessions (LOSO) | Entry 7 |
| `natural_population_analysis.py` | Excluded-vs-usable comparison; controlled perturbations | Entry 8 |
| `representation_analysis.py` | Representation ablations R0–R8; `recommendation.json` | Entry 9 |
| `v2_validation.py` | Pre-registered gate, frozen V2 baseline, contract artefacts, golden vectors | Entry 10 |
| `prism_inference.py` | Inference contract V2: `analyze_recording`, `load_baseline`, `validate_output`, `output_json_schema`; CLI `python src/prism_inference.py <wav>` | Entry 10 |

---

## 15. Model Registry and Versioning

### Current Artifacts

| Artifact | Location | Identity / Notes |
|---|---|---|
| Event detector | `results/inhaler_cnn.onnx` | SHA-256 `2e4e72d3…3725`; best of 3 CNN folds (fold 1, accuracy 0.8954); training-fold membership not saved |
| Inference contract | `results/v2_validation/inference_contract_v2.json` | `prism-inference-v2.0`, `DRAFT_NOT_FROZEN` (gate G4c failed) |
| V2 baseline | `results/v2_validation/v2_baseline.json` | `prism-v2-global-2026-09-30`, SHA-256 `0eaea4b8…32d3` |
| Output schema | `results/v2_validation/inference_output.schema.json` | JSON Schema draft 2020-12 |
| Golden vectors | `results/v2_validation/golden/` | 9 cases: 5 dataset WAVs identified by SHA-256, 2 synthetic WAVs, 2 input errors |
| Stage checkpoints | `results/recording_runs/<recording>/` | Frame features, window logits, contract output for one recording |
| `cv_results.csv` | `results/cv_results.csv` | Per-fold per-class metrics |
| XGBoost summary | `results/xg/summary_report.txt` | 5-fold; 0.8902 mean acc |

### Naming Convention
`inhaler_<arch>_v<major>.<minor>.onnx`, e.g. `inhaler_cnn_v1.0.onnx`.
- **Detector:** a new detector changes every contract output, so it requires a new contract version and new golden vectors.
- **Baseline:** a new baseline requires a new `baseline_id` and its own validation.

### Mobile Deployment Checklist

- [ ] ONNX model SHA-256 = `2e4e72d3ca040b15718ad6b09ae4ac3069c38f5c1a389327bf6b38c4e2903725`
- [ ] Input/output shapes (N,25,124) -> (N,4) raw logits; class order Drug, Exhale, Inhale, Noise
- [ ] ONNX Runtime version pinned in the mobile build
- [ ] DSP parity: frame features match `results/recording_runs/*/frame_features.csv` (MAE < 0.001)
- [ ] Detector parity: window labels and probabilities match `window_predictions.csv`
- [ ] Golden vectors pass with the contract tolerances:
  - status, event count, reasons and window times: exact (times within 1e-9 s);
  - `detector_confidence`: ±1e-4;
  - `feature_values` and `mean_rms`: ±1e-4 relative;
  - `feature_z_scores` and `anomaly_score`: ±0.01
- [ ] `v2_baseline.json` loaded and validated, never fitted on the device
- [ ] Output validates against `inference_output.schema.json`; `contract_version`, `baseline_id` and the model SHA-256 recorded in every output
- [ ] `input_domain = "prism_hardware"` for device audio (so `baseline_domain_validated = false`)
- [ ] None of the forbidden derived outputs (§9.3) is produced or displayed
- [ ] Inference time per window measured on the target device (target < 2 ms; not yet measured)

---

## 16. Development Priority Order

```
Phase 1 - ML Research
  [x] Dataset loading + annotation parsing
  [x] librosa_extractor.py: 124 features, .npy cache
  [x] Windowed dataset: trim + sliding window + balance
  [x] GroupKFold CV: RF, SVM, XGBoost
  [x] InhalerCNN + ONNX export (opset 17)
  [x] Evaluation: confusion matrix
  [ ] Regenerate XGBoost feature-importance artefacts (§8.9)
  [x] Post-event layer: Inhale event grouping and measurement (Entry 1)
  [x] Stage 1 inhale-event dataset + usability rule v1 (Entry 3)
  [x] Stages 2-7 feature analysis, baselines, scores, ablations (Entries 4-9)
  [x] Stage 8 V2 validation + inference contract (Entry 10; gate failed, DRAFT_NOT_FROZEN)
  [ ] Owner decision: accept V2 as an MVP exception, or replace it
  [ ] PRISM-hardware reference recordings (domain shift, re-baselining)
  [ ] Longitudinal data with user/device identifiers
  [ ] Threshold methodology (pre-registered), NORMAL / ANOMALY
  [ ] Personalized baseline V3 (§9.6)

Phase 2 - Mobile Application
  [ ] React Native scaffolding
  [ ] Native C++ DSP mirroring librosa_extractor.py (§6 parity table)
  [ ] ONNX Runtime (Android + iOS)
  [ ] Inference contract V2: grouping, scoreability, V2 features, frozen baseline, output JSON
  [ ] Conformance: stage checkpoints + golden vectors (§15)
  [ ] Result display following §9.4 + SQLite local storage

Phase 3 - ESP32 Firmware
  [ ] I2S driver: INMP441, 8 kHz, 24-bit -> int16 (s24 >> 8), DMA
  [ ] RMS inhalation detector state machine with pre-/post-trigger audio (§4)
  [ ] BLE GATT profile + MTU fragmentation (resolve audio_len width, §5)
  [ ] Binary packet assembly
  [ ] MPU6050 IMU integration
  [ ] Power management (deep sleep + wake)

Phase 4 - Cloud and Backend
  [ ] Firebase: Auth, Firestore, Functions, Storage
  [ ] Session document sync from mobile (§10)
  [ ] Daily adherence aggregation function
  [ ] Missed-recording notification (schedule-based)
  [ ] Clinician dashboard (web app; §11)

Phase 5 - Personalized Baseline (after Phase 1 data items)
  [ ] Pre-registered V3 design and evaluation (§9.6)
  [ ] Per-user baseline storage (users/{uid}/baseline)
  [ ] Conservative adaptation policy (V4)
```

---

## 17. Implementation Status

| Component | Status | Notes |
|---|---|---|
| Annotation loading | IMPLEMENTED | `loader.load_annotation()` |
| Librosa extraction | IMPLEMENTED | 124 features, .npy cache |
| Windowing + balancing | IMPLEMENTED | trim + windows + balance |
| Random Forest / SVM / XGBoost CV | IMPLEMENTED | GroupKFold k=5; XGBoost 0.8902 mean acc |
| 1D CNN (InhalerCNN) + ONNX export | VALIDATED (window level) | 3-fold CV mean acc 0.8902; deployed fold in-sample for about 2/3 of the recordings |
| Post-event layer (Inhale events) | VALIDATED against annotations (mostly in-sample) | Entries 1, 3 |
| Usability rule v1 / scoreability | IMPLEMENTED | Entry 3 |
| V1 baseline and scores (Stages 3–4) | IMPLEMENTED; superseded | Entries 5–6 |
| Baseline strategy, population, representation studies (Stages 5–7) | IMPLEMENTED | Entries 7–9 |
| V2 global baseline + anomaly_score | EXPERIMENTAL | Gate failed (G4c); `DRAFT_NOT_FROZEN` |
| Inference contract V2 reference implementation | IMPLEMENTED | Exact reproduction of the Stage 1 events (G6b); 9 golden vectors |
| Threshold, NORMAL / ANOMALY | Not started | — |
| Personalized baseline (V3/V4) | PLANNED | Needs longitudinal user/device data |
| Drug/Exhale event analytics (coordination, actuation) | NOT VALIDATED | Detector windows only; no event grouping or evaluation |
| Technique-quality / clinical labels | NOT SUPPORTED | No ground truth (Entry 2) |
| ESP32 firmware | PLANNED | Phase 3 |
| BLE protocol | PLANNED | Phase 3; open issues in §5 |
| React Native app | PLANNED | Phase 2 |
| Native C++ DSP | PLANNED | Phase 2 |
| ONNX Runtime mobile | PLANNED | Phase 2 |
| Firebase backend | PLANNED | Phase 4 |
| Doctor dashboard | PLANNED | Phase 4 |

---

## Appendix A: Acoustic Properties of the Event Classes

**Annotated durations** come from the cleaned `data/annotation.csv` (1,159 rows):

| Event | n | Median duration | 5–95% |
|---|---|---|---|
| Drug | 127 | 0.47 s | 0.28–0.85 s |
| Inhale | 260 | 1.46 s | 1.01–2.19 s |
| Exhale | 404 | 1.47 s | 0.91–2.02 s |
| Noise | 368 | 0.68 s | 0.30–1.37 s |

**Measured features of the 318 usable detected inhalation events** (`inhale_events_v1.csv`):

| Feature | Median | 5–95% |
|---|---|---|
| duration_s | 1.63 | 0.87–2.32 |
| spectral_centroid_mean (/4000 Hz) | 0.371 | 0.333–0.418 |
| spectral_flatness_mean | 0.132 | 0.086–0.172 |
| spectral_rolloff_mean (/4000 Hz) | 0.647 | 0.537–0.697 |
| zcr_mean | 0.358 | 0.298–0.427 |
| mean_rms (uncalibrated) | 0.184 | 0.131–0.237 |

No spectral measurements of Drug, Exhale or Noise exist in this repository. The
ranges in the previous revision (e.g. Drug flatness 0.7–1.0, breath centroid
0.1–0.3) were unreferenced, and the measured inhalation centroid (0.33–0.42)
contradicts the "breath centroid 0.1–0.3" range.

---

## Appendix B: Latency Budget (Mid-Range Android)

These are targets, not measurements.

| Stage | Target |
|---|---|
| BLE receive + fragment reassembly | <10 ms |
| DSP (STFT + 124 features) for 2 s segment | <20 ms |
| ONNX inference per window | <2 ms |
| Total inference for 2 s (~60 windows) | <120 ms |
| Event grouping, features, scoring | <5 ms |
| **Total: BLE receive -> result display** | **<200 ms** |

---

## Appendix C: FAQ

**Q: Why 8 kHz, not 44.1 kHz?**
The reference dataset is 8 kHz, so the detector and the baseline are defined at
8 kHz. Lower rates also reduce BLE bandwidth and DSP time. The contract accepts
8 kHz only.

**Q: Why N_FFT=256 (32 ms window)?**
It is the trained detector's setting. It gives 31.25 Hz bins and 8 ms hops, and
it is frozen at ONNX export.

**Q: Why GroupKFold?**
Adjacent frames in a recording are highly correlated. Standard KFold would allow
frames from the same recording in both train and test, causing data leakage.

**Q: Why ONNX not TFLite or CoreML?**
ONNX Runtime is cross-platform (Android + iOS) from one model file. TFLite needs
TensorFlow; CoreML is iOS-only. ONNX suits PyTorch dual-platform deployment.

**Q: Can the model be updated OTA?**
Technically yes: download from Firebase Storage and verify the SHA-256. But a new
detector or baseline changes every contract output, so it requires:
- a new contract version or `baseline_id`;
- new golden vectors;
- re-validation before release.

**Q: Why not run inference on ESP32?**
The ESP32-WROOM has 520 KB SRAM, and InhalerCNN's float32 weights take about
1.2 MB. Smartphone inference time has not been measured yet (Appendix B).
