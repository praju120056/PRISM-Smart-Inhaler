# PRISM Research Log

> **Document purpose:** Chronological, append-only research notebook preserving the complete scientific and engineering evolution of the PRISM (Pulmonary Response & Inhaler System Monitor) project. This log provides the historical provenance and technical baseline required to reconstruct Methods, Experiments, Results, and Discussion sections for academic papers, theses, and technical reports.
>
> **Immutability Rule:** Existing entries must NEVER be edited, rewritten, reordered, or retroactively modified. When new evidence contradicts an earlier conclusion, the earlier entry remains intact, and a new entry is appended documenting the change in evidence and reasoning.

---

# Research Entry 1 — 2026-09-18: Project Baseline, Post-Event Layer, and Inhalation Characterization

## 1. Project Objective and System Concept
PRISM (Pulmonary Response & Inhaler System Monitor) is a smart-inhaler research project focused on monitoring pressurized Metered-Dose Inhaler (pMDI) usage using acoustic sensing and machine learning. Conventional smart inhalers rely on mechanical or binary switches that record only that a canister actuation took place; they cannot evaluate how the medication was inhaled. PRISM's objective is to capture the acoustic emissions of inhaler usage to detect, segment, and characterize the complete sequence of physical events: exhalation prior to actuation, drug canister discharge, sustained inhalation airflow, and background ambient acoustic noise.

## 2. Hardware and Sensing Architecture
The intended PRISM hardware topology is an edge-sensor attachment mounted to a standard pMDI body:
- **Microphone:** INMP441 omnidirectional digital MEMS microphone with standard I2S digital output. Configured for 8 kHz sampling rate, 24-bit audio resolution (sign-extended to 32 bits via DMA, converted to normalized float32), mono channel.
- **Microcontroller:** ESP32-WROOM-32. Performs I2S DMA continuous sampling into an 8-block $\times$ 512-sample ring buffer.
- **On-Device Detection State Machine:** The microcontroller runs an energy threshold detector using 10 ms (80 samples) root-mean-square (RMS) windows. Onset threshold ($RMS > 0.01$) triggers recording into SRAM (80 KB buffer for up to 5 seconds of audio); offset threshold ($RMS < 0.005$ sustained for $> 200\text{ ms}$) terminates recording.
- **Ancillary Hardware:** Optional MPU6050 6-axis IMU for device tilt/orientation detection; optional mechanical switch on the canister well for physical actuation timing verification.
- *Status in available project record:* The physical hardware firmware and BLE GATT transmission protocol are architected and specified in detail (`ARCHITECTURE.md`), while the machine learning research codebase operates on recorded audio corpora.

## 3. Acoustic Sensing Approach
Acoustic sensing exploits the distinct turbulent airflow and discharge sound spectra generated inside and around the inhaler mouthpiece:
- **Drug Actuation:** High-velocity aerosol propellant jet discharge creating a sharp, broadband acoustic burst ($\sim 150 - 300\text{ ms}$).
- **Inhalation:** Lower-frequency, continuous turbulent airflow drawn through the mouthpiece chamber by negative thoracic pressure.
- **Exhalation:** Expiratory airflow deflected across or directed into the mouthpiece aperture prior to actuation.
- **Noise:** Handling clicks, speech, breathing pauses, room reverberation, and sensor baseline.

## 4. Mobile, Edge, and Cloud Architecture
The system employs a "mobile-first edge intelligence" model designed for patient data privacy:
- **Sensor Node (ESP32):** Acts strictly as a streaming peripheral; does not perform DSP, feature extraction, or machine learning inference. Transmits chunked binary PCM packets over Bluetooth Low Energy 5.0 (BLE GATT, MTU 512 bytes).
- **Mobile Edge Node (Smartphone / React Native + Native C++ DSP):** Primary compute unit. Reassembles BLE packets, executes identical librosa-compliant DSP/FFT pipelines in native code, evaluates local ONNX models via ONNX Runtime Mobile, computes session metrics, and presents instantaneous patient feedback. Raw audio never leaves the smartphone.
- **Cloud Layer (Firebase Firestore & Functions):** Receives strictly anonymized derived analytics, longitudinal technique aggregates, and adherence timestamps.
- **Clinical Dashboard:** Web interface for physicians to review longitudinal adherence trends, coordination timing distributions, and technique consistency.

## 5. Dataset and Recordings
- **Corpus:** 361 raw single-session pMDI acoustic recordings (`data/*.wav`), sampled at 8000 Hz, 16-bit mono PCM. Each recording is approximately 10 to 12 seconds in duration ($\approx 72\text{ minutes}$ total audio).
- **Ground Truth Temporal Annotations:** `data/annotation.csv` contains 1,164 temporal event annotations across 301 recordings (60 recordings in the folder are unannotated). File format: `filename,label,start_sample,end_sample`.
- **Precomputed Spectrogram/Feature Artifacts:** Precomputed frame CSVs exist for legacy feature pathways; modern research pipeline extracts standard feature matrices directly from WAV audio.

## 6. Existing Event-Detection Pipeline
The initial PRISM ML milestone was strictly window-level **event classification**:
- Rather than classifying the entire 12-second file with a single label, the pipeline slices the audio into a dense stream of short sliding windows to identify what acoustic event is occurring at each moment in time.
- Label determination for training: Majority vote of annotated frame labels across each window.

## 7. Feature Extraction Pipeline
The feature extraction architecture is implemented in `librosa_extractor.py` and `feature_extractor.py`:
- **Audio Sample Rate:** 8000 Hz.
- **Short-Time Fourier Transform (STFT):** Hann window of length $N_{\text{FFT}} = 256$ samples (32 ms), hop length $H = 64$ samples (8 ms frame step; 125 frames/second). Shared magnitude spectrogram $S$ with 129 frequency bins (31.25 Hz/bin resolution).
- **Per-Frame Feature Vector (124 features):**
  - 40 Mel-Frequency Cepstral Coefficients (MFCCs) computed over 128 Mel filterbanks spanning 50 Hz to 4000 Hz.
  - 40 First-Order Delta MFCCs (regression window $N = 9$).
  - 40 Second-Order Delta-Delta MFCCs.
  - 1 Spectral Centroid.
  - 1 Spectral Flatness.
  - 1 Spectral Rolloff (85th percentile threshold).
  - 1 Zero Crossing Rate (ZCR).
- **Window Assembly:**
  - Window size: 25 frames ($25 \times 8\text{ ms} = 200\text{ ms}$; historical references note 7 frames / 56 ms in early drafts, frozen at 25 frames for CNN input).
  - Window stride: 2 frames (16 ms step between successive inference windows).
  - Resulting tensor shape per window: `float32[25, 124]`.

## 8. Existing Event Classifiers & Model Contract
Two model families were developed in the training pipeline:
1. **XGBoost Classifier:** Tree-based ensemble trained on flattened window vectors (3,100 features/window), evaluated via 5-fold GroupKFold at the recording level, achieving mean accuracy of $0.8902 \pm 0.0044$.
2. **1D Convolutional Neural Network (CNN):** PyTorch architecture exported to ONNX format at `results/inhaler_cnn.onnx` (file size: 1,181,826 bytes).
- **ONNX Input/Output Contract:**
  - Input: `float32[batch_size, 25, 124]` (unnormalized raw features; no external scaler required).
  - Output: `float32[batch_size, 4]` raw unnormalized logits.
  - Class Index Order: `0: Drug`, `1: Exhale`, `2: Inhale`, `3: Noise` (deterministic mapping matching `config.LABEL_NAMES`).

## 9. Existing Classifier Evaluation
Cross-validation results for the CNN (evaluated on window-level test sets via GroupKFold cross-validation across recordings, total 170,323 windows):
- Mean Window Classification Accuracy: $\approx 89.0\%$
- **Class-Specific F1 Scores:**
  - Drug F1: $0.950$ (Precision: $0.944$, Recall: $0.956$)
  - Inhale F1: $0.893$ (Precision: $0.860$, Recall: $0.929$)
  - Exhale F1: $0.867$ (Precision: $0.863$, Recall: $0.873$)
  - Noise F1: $0.871$ (Precision: $0.888$, Recall: $0.855$)
- **CNN Confusion Matrix:**
  ```text
                Predicted
                Drug    Exhale  Inhale  Noise
  True Drug     34,877       0      89  1,565
  True Exhale       18  31,847      74  4,592
  True Inhale       46      71  22,934  1,630
  True Noise     1,979   5,043   3,587 61,971
  ```

## 10. Architectural Decision: Preserving the Classifier
- **Decision:** Do NOT retrain, modify, or replace `inhaler_cnn.onnx`.
- **Reasoning:** The window-level event detector is functional and validated for what it was trained to do: discriminating acoustic event types in short 200 ms slices. It is not an inhalation technique evaluator. Retraining it or building a second classifier prematurely would conflate event segmentation with technique assessment. The immediate required layer is the bridge between discrete window predictions and macroscopic inhalation events.

## 11. Motivation for the Post-Event Layer
The CNN outputs a dense sequence of discrete window predictions:
`Noise, Noise, Inhale, Inhale, Inhale, Inhale, Noise, Inhale, Drug, ...`
Clinical and physical questions (inhalation duration, inhalation flow rate, timing between actuation and peak inhalation, breath-hold) cannot be answered by isolated window labels. A dedicated post-event layer is required to:
1. Group consecutive, high-probability window predictions into coherent temporal events with defined start, end, and duration bounds.
2. Slice and extract the corresponding raw audio waveform from the original recording.
3. Extract acoustic and temporal properties from that raw waveform.

## 12. Implementation of `post_event.py`
A modular post-event processing library was implemented in `src/post_event.py` without modifying the underlying classifier or feature extraction standards:
- `OnnxEventClassifier`: Wraps ONNX Runtime session execution, enforcing the exact `float32[N, 25, 124]` tensor contract and mapping raw logits through softmax to class probabilities.
- `WindowPrediction`: Immutable dataclass capturing window index, start time (s), end time (s), assigned label, confidence score, and complete 4-class probability dictionary.
- `InhaleEvent`: Dataclass containing candidate event bounds (`start`, `end`, `duration`), average confidence, maximum confidence, window count, and constituent window indices.
- `TemporalGroupingConfig`: Explicit configuration dataclass exposing `target_label` (default "Inhale"), `smoothing_window` (majority vote over odd window length, default 1), `max_gap_s` (maximum gap bridged between adjacent windows, default 0.0 s), `min_event_duration_s` (minimum duration filter, default 0.0 s), and `min_confidence` (default None). Defaults apply zero arbitrary filtering.
- `extract_event_audio(recording, event)`: Extracts the exact raw audio slice from the original waveform using the event's floating-point temporal boundaries.
- `analyze_inhalation(audio, event)`: Computes physical, temporal, energy, and spectral measurements on the extracted raw inhalation slice.
- `plot_diagnostic(audio, predictions, events)`: Generates a multi-panel visual alignment showing raw audio waveform, sliding RMS envelope, CNN window probability tracks, and detected event boundary spans.

## 13. Initial Single-Recording Validation
The post-event pipeline was first tested end-to-end on `data/rec2018-01-22_17h41m33.475s.wav`:
- Result: Identified a single coherent Inhale candidate spanning $2.496\text{ s} \to 4.040\text{ s}$ (duration: $1.544\text{ s}$, mean confidence: $0.966$).
- Diagnostic plot (`results/post_event_diagnostic.png`) confirmed that the detected Inhale interval matched a sharp, sustained rise and fall in raw signal amplitude and RMS envelope, followed immediately at $4.1\text{ s}$ by a Drug actuation burst.
- Unit tests (`tests/test_post_event.py`) covering temporal grouping edge cases (contiguous windows, gap bridging, smoothing, threshold filtering) all passed (5/5).

## 14. Dataset-Wide Batch Exploration
To prevent drawing premature conclusions from single-file inspection, a batch exploration module was developed in `src/explore_inhalations.py` to process every recording in `data/`:
- **Execution:** Full run completed across all 361 WAV recordings with 0 runtime errors.
- **Output Artifacts:**
  - Detailed CSV table: `results/post_event/inhalation_events.csv` (364 event rows, 23 metrics per row).
  - Run metadata manifest: `results/post_event/inhalation_run_manifest.json`.
  - Distribution figure: `results/post_event/inhalation_distributions.png`.

## 15. Dataset-Wide Distribution Results ($N = 364$ detected events)
Descriptive statistics across all detected Inhale candidates:
- **Duration (s):** Mean $1.494 \pm 0.569$; Median $1.592$; Min $0.200$; Max $3.288$; IQR: $1.268\text{ s} - 1.816\text{ s}$.
- **Mean RMS:** Mean $0.181 \pm 0.040$; Median $0.184$; Min $0.029$; Max $0.444$; IQR: $0.160 - 0.204$.
- **Peak RMS:** Mean $0.279 \pm 0.070$; Median $0.275$; Min $0.038$; Max $0.702$; IQR: $0.247 - 0.301$.
- **Total Acoustic Energy:** Mean $0.062 \pm 0.031$; Median $0.060$; Min $0.0003$; Max $0.162$.
- **Time-to-Peak RMS (s):** Mean $0.749 \pm 0.464$; Median $0.680$; Min $0.000$; Max $2.664$.
- **Aggregated Model Confidence:** Mean $0.881 \pm 0.132$; Median $0.942$; Min $0.418$; Max $0.989$.
- **Spectral Centroid (mean normalized):** Mean $0.370 \pm 0.029$; Min $0.258$; Max $0.472$.
- **Zero Crossing Rate (mean):** Mean $0.350 \pm 0.050$; Min $0.153$; Max $0.470$.

## 16. Correlation Analysis & Feature Orthogonality
Pairwise Pearson correlation coefficients revealed key structural relationships:
- `duration` vs `mean_rms`: $r = 0.077$ (essentially zero correlation).
- `duration` vs `peak_rms`: $r = -0.032$.
- `duration` vs `total_energy`: $r = 0.709$.
- `mean_rms` vs `total_energy`: $r = 0.636$.
- `duration` vs `time_to_peak`: $r = 0.611$.
- `duration` vs `confidence`: $r = 0.773$.
- `mean_rms` vs `spectral_centroid`: $r = -0.682$.
- `peak_rms` vs `spectral_centroid`: $r = -0.545$.

## 17. Structural Interpretations & Hypotheses
1. **Orthogonality of Duration and Intensity:** The near-zero correlation ($r = 0.077$) between duration and mean RMS demonstrates that how long an inhalation lasts is decoupled from how loudly/forcefully the user inhales in this dataset. They represent independent acoustic degrees of freedom.
2. **Energy Integration:** Total acoustic energy integrates duration ($r = 0.709$) and mean intensity ($r = 0.636$), functioning as an acoustic proxy for cumulative acoustic inhalation intensity (though not proven to equal physical lung volume).
3. **Spectral Shift Observation:** Higher RMS is strongly associated with lower spectral centroid ($r = -0.682$). This confirms that higher-intensity events concentrate energy at lower acoustic frequencies, generating a hypothesis regarding turbulence spectra for future experimental testing.
4. **Pipeline Origin of Duration–Confidence Correlation:** The strong correlation between event duration and confidence ($r = 0.773$) is largely an artifact of the grouping pipeline (longer events accumulate a larger run of consecutive, high-probability window votes) rather than an intrinsic biological property of inhalation.

## 18. Discovery of Short Events & Multi-Event Recordings
Analysis of event multiplicity across the 361 recordings:
- 287 recordings (79.5%): Exactly 1 Inhale event detected.
- 31 recordings (8.6%): 2 Inhale events detected.
- 4 recordings (1.1%): 3 to 5 Inhale events detected.
- 39 recordings (10.8%): 0 Inhale events detected.

A clear division was observed in event durations:
- **Longer Inhalation Candidates ($N = 332$, $91.2\%$):** Duration $\ge 0.50\text{ s}$, median duration $1.60\text{ s}$, mean CNN confidence $0.907$.
- **Short Transient Events ($N = 32$, $8.8\%$):** Duration $< 0.50\text{ s}$, median duration $0.25\text{ s}$, mean CNN confidence $0.608$. In recordings with 2 detected events, the secondary event almost always occurs $\sim 0.2 - 0.4\text{ s}$ after the primary inhalation.

## 19. Validation Against Ground Truth Annotations (`data/annotation.csv`)
To investigate whether the 32 short events and 39 zero-event files were algorithm failures or acoustic realities, detected events were cross-referenced against sample-level ground truth bounds in `data/annotation.csv` (301 annotated files):
- **Validation of the 32 Short Events ($< 0.50\text{ s}$):**
  - Of the 32 events, 19 occurred in files with ground-truth annotations (13 were in unannotated recordings).
  - Among all 19 annotated instances, **exactly 0 overlapped with a ground-truth `Inhale` label**.
  - Ground-truth overlap: 10 overlapped with baseline unannotated background, 6 overlapped with `Noise`, 2 overlapped with `Exhale`, and 1 overlapped with `Drug`.
  - *Conclusion:* Short events ($< 0.5\text{ s}$) are confirmed to be spurious acoustic blips and boundary spillover. A minimum duration threshold (e.g. `min_event_duration_s = 0.4` or `0.5`) will eliminate them without removing any true annotated inhalations.
- **Validation of the 39 Zero-Event Recordings:**
  - 17 of the zero-event recordings were present in `data/annotation.csv` (22 were unannotated files).
  - In those 17 recordings, **exactly 0 had an `Inhale` annotation** (they contained exclusively `Noise`, `Drug`, or `Exhale`).
  - *Conclusion:* The detector did not fail to detect an inhalation in those annotated files; no inhalation was performed in those recordings.
- **Validation of Longer Candidates ($\ge 0.50\text{ s}$):**
  - Over 250 detected candidates directly align with ground-truth `Inhale` annotations.

## 20. Current Scientific Boundaries
- **What has been proven:** The post-event pipeline (`CNN window predictions → temporal grouping → original waveform extraction`) successfully isolates temporally coherent inhalation segments and extracts structured, non-degenerate temporal, energy, and spectral measurements across the recording corpus.
- **What has NOT been proven:** It has NOT been demonstrated that these acoustic characteristics encode inhalation technique quality (e.g. good vs. poor vs. erroneous technique). The existing dataset annotations (`annotation.csv`) contain only acoustic event categories (`Inhale`, `Drug`, `Exhale`, `Noise`), not clinical technique ratings or flow-volume ground truth.
- **Prohibited Claims:** No clinical thresholds (e.g., "duration $< X$ is bad technique") should be fabricated from feature distributions alone.

## 21. Current Research Question and Next Steps
- **Current Question:** Do the extracted temporal, energy, and coordination features statistically discriminate between validated inhalation technique conditions?
- **Immediate Next Actions:**
  1. Complete formal event segmentation validation (evaluating temporal overlap precision/recall against `data/annotation.csv`).
  2. Determine whether technique-condition labels (e.g. correct technique, late actuation, weak inhalation, uncoordinated breath) exist in the capstone study protocol or metadata.
  3. Analyze coordination timing between detected `Drug` actuation bursts and `Inhale` candidate intervals.
  4. Test feature discriminability against technique ground-truth classes before determining whether technique assessment requires rules, statistical scoring, or supervised classification.

---

# Research Entry 2 — 2026-09-18: Ground-Truth Dataset Audit for Inhalation Technique

## Question / Motivation
Before implementing technique assessment or building evaluative classifiers, we must establish what ground-truth labels and experimental metadata actually exist in the project repository. Specifically:
1. Do labels exist describing inhalation technique, inhalation condition, actuation timing, flow rate, or clinical correctness?
2. What do the 361 WAV recordings represent, and what metadata is associated with each recording?
3. Can the existing ground-truth annotations be joined to the 364 detected Inhale events?

## Hypothesis / Reasoning
We hypothesized that the repository might contain accompanying metadata (e.g. subject IDs, protocol conditions like "fast", "slow", "correct", or spirometry references) either embedded in filenames, in companion spreadsheets, or in documentation. A strict empirical audit was conducted across all files, git history, and data directories without inferring labels from acoustic distributions or inventing categories.

## Work Performed
1. **Repository & Directory Scan:** Systematically searched the workspace (`data/`, `results/`, `src/`, root directory, and git commit history) for metadata, condition files, spreadsheets, or protocol descriptions.
2. **Annotation Schema Audit:** Parsed `data/annotation.csv` to catalog all unique labels, column schemas, and sample-level boundaries.
3. **Filename & Timestamp Analysis:** Inspected all 361 `.wav` filenames in `data/` using regex and temporal grouping to determine naming schemas, dates, and embedded codes.
4. **Ground-Truth Join & IoU Analysis:** Cross-referenced the 364 detected Inhale events (`results/post_event/inhalation_events.csv`) against the ground-truth annotations (`data/annotation.csv`) to assess coverage, match rate, and temporal Intersection over Union (IoU).

## Results

### 1. Available Ground-Truth Labels in `data/annotation.csv`
- Format: Exactly 4 columns without header: `[filename, label, start_sample, end_sample]`.
- Total annotation entries: 1,162 rows across 301 unique recordings.
- Label inventory:
  - `Exhale`: 404 annotations across 248 unique recordings
  - `Noise`: 370 annotations across 125 unique recordings
  - `Inhale`: 261 annotations across 255 unique recordings
  - `Drug`: 127 annotations across 119 unique recordings
- **Critical finding:** No other columns, labels, or metadata exist in `annotation.csv`. There are zero technique quality ratings (e.g. good/poor), zero protocol condition labels (e.g. shallow/deep/fast/slow), zero subject/demographic IDs, and zero physical flow or volume values.

### 2. Recording Filenames and Coverage
- **Filename Schema:** Every file follows the strict timestamp template: `recYYYY-MM-DD_HHhMMmSS.sss.wav`.
- **Temporal Distribution:** Recordings were collected across 10 distinct dates in 2018:
  - 2018-01-22 (9 files), 2018-01-23 (108 files), 2018-02-01 (16 files), 2018-02-02 (36 files), 2018-02-05 (33 files), 2018-02-06 (17 files), 2018-02-08 (13 files), 2018-02-09 (6 files), 2018-05-02 (10 files), 2018-05-03 (113 files). Total: 361 files.
- **Filename Metadata:** Filenames contain no subject identifiers, condition tags, or inhaler device types.

### 3. Missing / Unannotated Recordings
- Total WAV recordings: 361.
- Total annotated recordings: 301.
- **Unannotated recordings:** 60 WAV files have no entry in `annotation.csv`. In these 60 unannotated recordings, the post-event detector found 52 inhale candidates.

### 4. Join to Detected Inhale Events & Temporal Segmentation Match
- Total detected Inhale events: 364 (312 from annotated recordings, 52 from unannotated recordings).
- **Ground-Truth Inhale Recovery:**
  - Total ground-truth Inhale annotations: 261 (spanning 255 unique files).
  - Ground-truth Inhale annotations detected by the pipeline: **257 out of 261 (98.5%)**.
  - All 257 matched events have **temporal IoU $\ge 0.50$**, with a **mean IoU of $0.879$**.
  - 100% of the 255 annotated inhale recordings had their inhalation correctly detected.
- **Unmatched Detected Events in Annotated Files ($N = 55$):**
  - 19 are the short transient blips ($< 0.50\text{ s}$) that overlap with `Noise`, unannotated baseline silence, `Exhale`, or `Drug` (0 overlap with true `Inhale`).
  - 36 are secondary or split detected candidate segments.

### 5. Ambiguity in the Dataset
- **Variable Maneuver Completeness:**
  - Only 100 of the 301 annotated recordings contain the full clinical sequence of `Drug` + `Inhale` + `Exhale`.
  - 182 annotated recordings contain NO `Drug` actuation label (they are pure breath recordings).
  - 46 annotated recordings contain NO `Inhale` label (containing only `Exhale`, `Drug`, or `Noise`).
- **Absence of Experimental Protocol Documentation:** No protocol document exists explaining whether different days represented different subjects, different inhaler types, or different prescribed technique errors.

## Interpretation
1. **Acoustic Ground Truth vs. Technique Ground Truth:** The repository contains rigorous ground truth for *acoustic event boundaries* (`Inhale`, `Drug`, `Exhale`, `Noise`), but zero ground truth for *inhalation technique quality*.
2. **Technique Classification is Currently Impossible Without External Ground Truth:** Because there are no labels indicating whether an inhalation was clinically correct, too fast, too weak, or uncoordinated, training a supervised "technique classifier" at this stage would require inventing arbitrary classes, which violates scientific discipline.
3. **Temporal Segmentation is Highly Validated:** The post-event pipeline demonstrates excellent temporal precision (mean IoU 0.879, 98.5% recall) against human-annotated acoustic boundaries.

## Decision
1. **Do NOT build a technique classifier or assign technique-quality scores.**
2. **Postpone supervised technique modeling until external protocol/technique labels are provided.**
3. **Focus immediate validation on legitimate, grounded analyses:**
   - Benchmark temporal segmentation precision/recall and boundary error against `annotation.csv`.
   - Analyze actuation-to-inhalation coordination latency on the 100-119 recordings where both `Drug` and `Inhale` events co-occur.

## Limitations / Uncertainties
- Subject identity is completely unobservable from the dataset. It is impossible to know whether the 361 recordings represent 1 subject repeating maneuvers, 10 subjects (corresponding to the 10 dates), or 361 distinct subjects.
- Without protocol records, variation in duration and RMS cannot be separated into subject physiology vs. deliberate experimental instructions.

## Next Step
1. Benchmark temporal segmentation metrics (IoU, precision, recall) across the 301 annotated files under varying `min_event_duration_s` settings to establish the optimal cleanup threshold.
2. Measure and report empirical actuation-inhalation coordination timing (actuation onset relative to inhale start and peak flow) for all dual-event recordings.
