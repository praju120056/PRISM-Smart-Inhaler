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

---

# Research Entry 3 — 2026-09-29: V1 Research Direction, Repository Audit, and Inhale-Event Dataset Finalization (Stages 0–1)

## Research Direction Recorded (set by the project owner on 2026-09-29)
- **V1 is a global, not personalized, baseline.** Recordings are not grouped by subject (Entry 2), so V1 models the whole available inhale-event dataset. Roadmap: V1 global robust univariate baseline (median / MAD robust z-scores aggregated into an anomaly score) → V2 multivariate baseline → V3 user-specific baseline (needs user/longitudinal grouping) → V4 adaptive personalization.
- **Terminology:** outputs are `NORMAL` / `ANOMALY`. "An anomaly is an inhalation event whose acoustic characteristics deviate substantially from the established baseline." NORMAL means baseline-consistent. ANOMALY is not a clinical judgement of inhalation technique.
- **Locked V1 choices:** calibrate on the first 20 *usable* inhale events. Set the threshold at an empirical percentile of calibration anomaly scores, with the percentile documented and justified. Use a conservative adaptation policy that stops anomalies from immediately contaminating the baseline. The output exposes at least status, anomaly score and deviating features.
- **Legacy design not adopted:** `ARCHITECTURE.md` §9 (Mahalanobis on MFCCs, 0.5/0.3/0.2 weights, 1.5/3.0 thresholds, GOOD/POOR composite labels) is legacy documentation. It is not the V1 method, and its thresholds are not used.
- Work proceeds in stages with one commit per completed stage on branch `anomaly-detection`.

## Question
Is the existing inhale-event table a correct, reproducible input for baseline modeling? Which detected events should be eligible ("usable") for baseline calibration and baseline-consistency evaluation?

## Stage 0 — Repository Audit
- The previously uncommitted post-event work was committed unchanged as checkpoint `5be5001` on new branch `anomaly-detection`. `main` was not modified and nothing was pushed. Compiled `.pyc` files are tracked in git but were left unstaged. Untracking them later is recommended.
- `data/` is gitignored. Dataset identity is recorded by SHA-256 in `results/inhale_dataset/dataset_summary.json`: `data/annotation.csv` = `cdcd97c5…f88f1`, `results/post_event/inhalation_events.csv` = `b4f7bb3e…4722`.
- **The event classifier was trained on part of this corpus.** `train_cnn.run_cnn_cv` exports the best cross-validation fold model, chosen by fold test accuracy. Per `results/cv_results.csv` (3 folds) that is fold 1 (accuracy 0.8954). The deployed ONNX model was therefore trained on roughly two-thirds of the 361 recordings it is now applied to, and which recordings were in its training fold was not saved. Event/annotation agreement below is mostly in-sample. It overstates how well the detector would segment unseen audio.
- No baseline or anomaly-detection code or results existed. `README.md` and `ARCHITECTURE.md` list the baseline engine as not started.

## Work Performed
1. **Reproducibility:** regenerated the event table with the current code (`python src/explore_inhalations.py --output-dir <temp dir> --no-plots`, default `TemporalGroupingConfig`) and compared it column by column with the committed CSV.
2. **Integrity checks** on the event table: shape, missing/non-finite values, keys, chronology, window structure.
3. **Annotation audit** of `data/annotation.csv`: counts, duplicates, invalid intervals, overlapping Inhale labels.
4. **Event ↔ annotation audit** using temporal IoU ≥ 0.5 as "matched".
5. **Usability rule v1 and a rule-sensitivity grid.**

Implementation: `src/inhale_dataset.py` (new). Tests: `tests/test_inhale_dataset.py` (21 tests on synthetic tables). Command: `python src/inhale_dataset.py --verify-against <regenerated CSV>`. Outputs in `results/inhale_dataset/`: `inhale_events_v1.csv`, `event_annotation_audit.csv`, `inhale_annotation_matches.csv`, `usability_sensitivity.csv`, `dataset_summary.json`, `reproducibility_check.json`. The upstream `results/post_event/inhalation_events.csv` was not modified.

## Results

### 1. Reproducibility
The regenerated table is identical to the committed one: 364 × 23, maximum absolute difference 0 in every numeric column, all text columns equal, 0 failed recordings (`reproducibility_check.json`).

### 2. Event Table Integrity (`results/post_event/inhalation_events.csv`)
- 364 events, 23 columns. 322 recordings have ≥1 event and 39 have none. Events per recording: 1 → 287 recordings, 2 → 31, 3 → 2, 4 → 1, 5 → 1.
- 0 missing values, 0 non-finite feature values, 0 duplicate `(recording_file, event_id)` keys. Every row has label `Inhale` and sample rate 8000 Hz. All 361 WAVs are 8 kHz mono, 6.46–12.51 s long (median 12.0 s).
- The 13 acoustic/timing features are `duration_s`, `mean_rms`, `peak_rms`, `total_energy`, `time_to_peak_s`, and the mean and std of `spectral_centroid`, `spectral_flatness`, `spectral_rolloff` and `zcr`. `confidence`, `max_confidence` and `window_count` are CNN detection outputs, not inhalation measurements.
- `recording_id` order equals chronological filename-timestamp order.
- **Within-event bridging:** with `max_gap_s = 0`, overlapping 200 ms windows still join Inhale windows up to 12 strides (192 ms) apart. 47 events contain non-Inhale windows bridged this way, and the minimum Inhale-window fraction is 0.41 (`inhale_window_fraction` column).
- **Time-to-peak rounding:** in 3 events `time_to_peak_s` exceeds `duration_s` by ≤ 2×10⁻⁸ s. This comes from sample rounding when the segment is sliced and has no practical effect. Some events have their RMS peak at the first or last envelope frame. This measurement issue is left for Stage 2 to quantify.

### 3. Annotation File Audit (`data/annotation.csv`)
- 1,162 data rows (the file has 1,163 lines, the last empty) across 301 recordings.
- Invalid rows: 2 exact duplicate rows (one Inhale, one Noise) and 1 zero-length Noise interval, leaving 1,159 clean rows.
- Clean label counts (rows / recordings): Inhale 260 / 255, Exhale 404 / 248, Noise 368 / 125, Drug 127 / 119.
- Inhale annotations overlap each other in 2 recordings:
  - `rec2018-02-05_10h56m11.262s`: 2 overlapping annotations.
  - `rec2018-02-06_11h39m13.131s`: 3 nested annotations of 0.362, 0.559 and 0.789 s.

  Outside these two recordings, the shortest annotated inhalation is 0.745 s.

### 4. Event ↔ Annotation Agreement (IoU ≥ 0.5)
- **Annotation side:** 259 of 260 unique Inhale annotations are matched (99.6%), with mean IoU 0.876 and median 0.887. The unmatched one is the nested 0.362 s annotation (best IoU 0.34).
- **Event side**, the 312 events in annotated recordings:
  - 257 matched.
  - 0 partially overlap an Inhale annotation: every event that overlaps one has IoU ≥ 0.5.
  - 9 lie outside the recording's Inhale annotations.
  - 46 lie in annotated recordings that have no Inhale annotation.
- **Unannotated recordings:** the 60 unannotated recordings contain 52 events.
- **Shortest match:** the shortest matched event is 0.872 s.
- **Short events:** 0 of the events shorter than 0.5 s overlap an Inhale annotation.

### 5. Corrections to Earlier Entries (Entries 1–2 preserved unchanged)
| Earlier statement | Entry | Verified now | Note |
|---|---|---|---|
| 1,164 annotations | 1 | 1,162 rows | Entry 2's 1,162 is correct |
| 261 Inhale annotations | 2 | 261 rows, 260 unique | one exact duplicate row |
| 257/261 (98.5%) of annotated inhalations detected, all IoU ≥ 0.5 | 2 | 259/260 unique (99.6%); 260/261 if the duplicate is kept | 257 is the number of *events* matched, so Entry 2 most likely counted events |
| Mean IoU 0.879 | 2 | 0.876 over matched unique annotations | |
| 36 unmatched long events are "secondary or split detected candidate segments" | 2 | 34 of the 36 are in annotated recordings with no Inhale annotation, 31 of them under no annotation of any label. No event partially overlaps an annotated inhalation | no event is a fragment of an annotated inhalation |
| Secondary event "almost always" 0.2–0.4 s after the primary | 1 | 5 of 31 two-event recordings (9 < 0.2 s, 17 > 0.4 s, median 0.60 s) | from `gap_prev_s` in `inhale_events_v1.csv` |
| Zero-event annotated recordings: "no inhalation was performed" | 1 | Not established | This assumes exhaustive annotation. Annotated recordings are not exhaustively annotated: high-confidence long events with no annotation under them exist, e.g. `rec2018-01-23_10h43m40.126s` (1.83 s, confidence 0.965) |

### 6. Usability Rule v1
An event is **usable** (eligible for baseline modeling) only if all of these hold:
1. All 13 features are finite.
2. `duration_s ≥ 0.5 s`.
3. No other candidate in the same recording lies within a gap shorter than 0.2 s, i.e. one CNN analysis window.
4. The event does not reach the recording's first or last analysis window (tolerance 0.008 s, half a stride).

The rule uses only detector output and WAV headers, so it can be applied to unannotated data. Annotations are used only to audit it.

Rationale for each criterion:
- **(2) Minimum duration.** The 0.5 s value was proposed in Entry 1 before this audit. It is below the shortest valid annotated inhalation (0.745 s), so the rule cannot exclude a duration annotators ever labelled as an inhalation. It removes candidates that the annotations show are not inhalations.
- **(3) Close neighbours.** Below one analysis window the detector cannot resolve whether two candidates are one interrupted inhalation or two inhalations. Fragment features (duration, energy, time-to-peak) would not describe a complete inhalation. Both candidates are excluded rather than merged, so no upstream value changes.
- **(4) Recording boundary.** Such events are censored by the recording. Censoring will matter in deployment, where the hardware buffer is limited to 5 s.

Outcome:
- **Totals:** 318 usable, 46 excluded.
- **Flags:** short 32, close neighbour 25, boundary 1, non-finite 0.
- **Exclusion combinations:** short only 20; close neighbour only 14; short + close neighbour 11; short + boundary 1.
- **Recordings:** 309 contribute usable events (301 with one, 7 with two, 1 with three).
- **Annotation status of usable events:** 257 matched, 30 in recordings with no Inhale annotation, 2 outside Inhale annotations, 29 in unannotated recordings.
- **Annotation status of excluded events:** 0 matched, 16, 7 and 23 respectively.

Sensitivity (`usability_sensitivity.csv`, 7 duration × 4 gap settings):
- No duration threshold from 0.0 to 0.8 s excludes an annotation-matched event.
- At the chosen gap of 0.2 s no matched event is excluded. A 0.4 s gap would exclude 2.
- Raising the duration threshold from 0.5 to 0.8 s would remove 11 more events (318 → 307). Of these, 8 are in annotated recordings without a match and 3 are in unannotated recordings. That higher value was not chosen, because it would be tuned to the annotated minimum instead of set below it.

The first 20 usable events (V1 calibration candidates, by `usable_order`) come from 20 recordings:
- 9 on 2018-01-22 (17:41–17:45) and 11 on 2018-01-23 (10:42–10:45).
- 19 are annotation-matched. One (`rec2018-01-23_10h43m40.126s`, 1.83 s, confidence 0.965) is in a recording with no Inhale annotation.

## Interpretation
1. The upstream event table is reproducible from code and structurally clean. It is a sound input for modeling.
2. Candidates shorter than 0.5 s are non-inhalation detections, not short inhalations.
3. **Splits:** there is no evidence that the detector fragments annotated inhalations. Segmentation is still ambiguous for closely spaced candidates, all of which are in recordings without Inhale annotations, so they are excluded conservatively. Merging them through `TemporalGroupingConfig.max_gap_s` would change upstream measurements. It would be a separate upstream experiment and is not part of V1.
4. **Unconfirmed usable events:** 32 usable events are not confirmed by an annotation (30 + 2). Some are probably unannotated inhalations and some may be detector false positives. They stay usable because the rule must be annotation-independent. The audit file allows sensitivity analyses restricted to annotation-matched events.

## Decision
1. The Stage 1 modeling dataset is `results/inhale_dataset/inhale_events_v1.csv`: all 364 upstream rows unchanged, plus context, flags, `usable`, `exclusion_reasons`, `chronological_order` and `usable_order`. "First N usable events" is defined by `usable_order`.
2. Usability rule v1 is as above. Its parameters live in `inhale_dataset.UsabilityRule`, and the CLI labels any non-default setting `custom`.
3. Annotation-audit columns are for evaluation and failure analysis only. They must not become anomaly features or drive calibration selection.
4. The upstream event table is not regenerated, and close candidates are not merged.

## Limitations
- **Detector trained on this corpus:** the CNN detector is in-sample for about two-thirds of the recordings (Stage 0).
- **Calibration sessions:** all 20 calibration candidates come from the first two recording sessions. Whether dates correspond to subjects or sessions is unknown (Entry 2), so a global V1 baseline calibrated on them may encode session-specific characteristics. Stage 2 should measure between-date differences before this is interpreted.
- **Annotation completeness:** annotations are not exhaustive, so "no annotation" is not a negative label.
- **Short events:** the usable set cannot contain events shorter than 0.5 s. Whether short detections are rejected or scored at inference time is a later-stage decision.
- **Rule design timing:** apart from the 0.5 s value proposed in Entry 1, the rule was designed after inspecting this dataset. It was not pre-registered.

## Next Step — Stage 2: Feature Analysis
For the 13 features, on usable events, with all events reported for context:
- Profile distributions, scale, skew, outliers and MAD = 0 risk.
- Measure redundancy with rank correlations.
- Measure stability across recording dates and between the first 20 usable events and the rest.
- Quantify measurement issues, including RMS peaks at envelope edges for `time_to_peak_s`/`peak_rms`.
- Decide whether the detection columns stay out of the feature set (default: yes, because they describe the CNN).
- Choose and document the initial V1 feature subset without silently dropping any feature.

---

# Research Entry 4 — 2026-09-30: Feature Analysis and V1 Feature Selection (Stage 2)

## Question
Which of the 13 acoustic/timing features of an inhale event are appropriate for the V1 global robust (median/MAD) baseline? Each feature is classified KEEP, EXCLUDE or DEFER with evidence. Anomaly scoring, thresholds and personalization are out of scope for this stage.

## Data
- **Dataset:** `results/inhale_dataset/inhale_events_v1.csv` (Stage 1, commit `4c25e6d`, SHA-256 `0dbae978…7cf1`), not modified.
- **Population:** the 318 usable events (usability rule v1); all 364 events are reported for context only.
- **Calibration set:** the 20 events with `usable_order` 1–20, i.e. the locked V1 calibration set.
- **Detection columns:** the CNN detection columns (`confidence`, `max_confidence`, `window_count`) are never candidate features.
- **Annotations:** used only to interpret measurements (`annotation_context.csv`), never to select features.

## Feature Definitions (as computed upstream by `post_event.analyze_inhalation` on the event segment sliced from the original 8 kHz waveform)
- **`duration_s`:** event end − start. Start is the first Inhale window's start. End is the last Inhale window's start + 0.2 s, clamped at the recording end. Resolution is 16 ms.
- **`mean_rms` / `peak_rms`:** mean / maximum of an RMS envelope computed over 256-sample (32 ms) frames starting every 64 samples (8 ms). The envelope includes 3–4 partial frames at the tail of the segment.
- **`total_energy`:** Σx² / sample rate (amplitude²·s).
- **`time_to_peak_s`:** start time, relative to event start, of the envelope frame with the maximum RMS.
- **`spectral_{centroid,flatness,rolloff}_{mean,std}`:** mean / std over frames of librosa features on the segment alone: STFT n_fft 256, hop 64, Hann, `center=True` with zero padding. Centroid and 85% rolloff are divided by 4 kHz.
- **`zcr_{mean,std}`:** librosa zero-crossing rate with its default 2048-sample (256 ms) frames, hop 64, `center=True` with edge padding.

## Methods
Implementation: `src/feature_analysis.py`, command `python src/feature_analysis.py`. Tests: `tests/test_feature_analysis.py` (23 tests). Outputs are in `results/feature_analysis/`. The run is deterministic, with seed 20260929 and 2,000 subsample draws.

1. **Distributions:**
   - Summary statistics: percentiles, mean/SD, and MAD.
   - Robust z = (x − median) / (1.4826·MAD).
   - Extremes defined as |z| > 3.5, the Iglewicz & Hoaglin (1993) modified-z convention (external literature).
   - Shape: skewness and Bowley quartile skewness.
   - Tail dominance: the share of Σz² carried by the top 1 and top 5 events.
   - Measurement resolution relative to the MAD.
   - A log-scale comparison for every strictly positive feature.
2. **Numerical stability at n = 20:** medians and MADs of 2,000 random 20-event subsets of the usable events.
3. **Redundancy:**
   - Spearman ρ (primary measure) and Pearson r.
   - Within-session Spearman: pooled correlation of within-session percentile ranks, which removes between-session offsets.
   - Rank R²: normal scores of one feature regressed on the others, both all 12 others and the kept set.
   - A log-scale identity check for `total_energy`.
   - Groups with |ρ| ≥ 0.8 are reported descriptively only; they are never an exclusion rule on their own.
4. **Stability:**
   - Kruskal–Wallis test and ε² = (H − k + 1)/(n − k) across dates and across sessions, restricted to groups with ≥ 5 usable events (8 dates, 12 sessions).
   - Group medians expressed in pooled robust-z units.
   - Sessions are split wherever consecutive recordings are more than 25 min apart. Across all 361 WAVs, the largest gap inside a session is 21.05 min and the smallest gap between sessions is 26.74 min, so any threshold in [21.05, 26.74) gives the same 23 sessions. 18 sessions contain usable events.
5. **Calibration set:**
   - First 20 usable events vs the remaining 298, compared with Mann–Whitney, Cliff's δ, KS, the median shift in rest-MAD units, the MAD ratio, and the fraction of the rest outside the calibration range. Under exchangeability that fraction is expected to be 2/21 = 0.095.
   - The calibration median and MAD are also placed within the distribution of random 20-event subsets.
6. **Edge effect:**
   - Every event (364) is recomputed from the audio.
   - Measurements: the position of the RMS-envelope peak (first frame, last frame, partial tail frame or interior), the peak without partial frames, the RMS just before and after the event, and near-peak ambiguity (the time span of frames within 5% of the peak).
   - The same pass also measures how much frames touched by padding contribute to the `*_std` features (std over unpadded frames vs all frames).

## Results

### Reproduction check and an upstream precision note
- **Precision loss:** the upstream CSV stores event times with limited precision (e.g. `3.792` for the detector's `3.7920000000000003`). Slicing audio from the CSV value can move a boundary by one sample, which adds a partial envelope frame. This changes `mean_rms` by up to 0.0087 for a 0.2 s event.
- **Fix used here:** `exact_event_bounds` rebuilds the detector's exact float bounds from the window indices. All stored `mean_rms`, `peak_rms`, `time_to_peak_s` and spectral std values then reproduce within 4.4×10⁻¹⁶.
- **Scope:** this is a reproducibility note for anyone re-slicing audio from CSV times. It is not a Stage 1 bug, and no Stage 1 logic was changed.

### 1–2. Distributions and robust-scale suitability (usable events, raw scale)
| Feature | Median | 1.4826·MAD | MAD/median | Skew | Bowley | Extremes low/high | max abs z | Top-5 share of Σz² |
|---|---|---|---|---|---|---|---|---|
| `duration_s` | 1.632 | 0.297 | 0.123 | 0.36 | 0.08 | 6/7 | 5.6 | 0.19 |
| `mean_rms` | 0.1838 | 0.0291 | 0.107 | 1.25 | -0.02 | 1/1 | 8.9 | 0.27 |
| `peak_rms` | 0.2751 | 0.0393 | 0.096 | 2.90 | 0.00 | 0/8 | 10.9 | 0.49 |
| `total_energy` | 0.06424 | 0.0253 | 0.266 | 0.66 | 0.13 | 0/1 | 3.9 | 0.14 |
| `time_to_peak_s` | 0.744 | 0.439 | 0.398 | 1.03 | 0.14 | 0/3 | 4.4 | 0.23 |
| `spectral_centroid_mean` | 0.3705 | 0.0279 | 0.051 | 0.15 | -0.03 | 1/0 | 4.0 | 0.15 |
| `spectral_centroid_std` | 0.0394 | 0.00898 | 0.154 | 0.48 | 0.13 | 0/3 | 3.9 | 0.13 |
| `spectral_flatness_mean` | 0.1322 | 0.0297 | 0.152 | -0.44 | 0.04 | 1/0 | 3.6 | 0.16 |
| `spectral_flatness_std` | 0.05567 | 0.0133 | 0.161 | 0.49 | -0.03 | 0/2 | 3.9 | 0.16 |
| `spectral_rolloff_mean` | 0.6465 | 0.0472 | 0.049 | -0.78 | -0.34 | 3/0 | 3.9 | 0.15 |
| `spectral_rolloff_std` | 0.08071 | 0.0169 | 0.141 | -0.25 | -0.01 | 0/0 | 3.1 | 0.12 |
| `zcr_mean` | 0.3578 | 0.0418 | 0.079 | -0.32 | 0.10 | 1/0 | 4.9 | 0.16 |
| `zcr_std` | 0.05903 | 0.0177 | 0.202 | 0.48 | 0.06 | 0/0 | 2.6 | 0.12 |

- **MAD health:** no feature has a zero or near-zero MAD (relative MAD 0.05–0.40). Ties at the median are negligible, the largest being 0.6% for `time_to_peak_s`. Quantisation is at most 8% of the MAD, for `duration_s` with its 16 ms step.
- **Numerical stability at n = 20:**
  - MAD: random 20-event subsets give MADs of 0.54–0.60× the population MAD at the 5th percentile and 1.30–1.57× at the 95th, depending on the feature.
  - Median: the 95th-percentile median shift is 0.49–0.58 robust SD.
  - Zero MAD: P(MAD = 0) is 0 for every feature.
  - Interpretation: median/MAD at n = 20 is numerically stable but imprecise, with a MAD uncertainty of roughly ±50%.
- **Tail-dominated feature:** `peak_rms`. Its five most extreme events carry 49% of Σz², and the maximum |z| is 10.9. Its 8 high extremes:
  - Position: 7 have an interior envelope peak and 1 peaks at the first frame.
  - Crest factor (peak/mean): median 2.44 (range 1.10–3.57), against 1.50 across usable events.
  - Interpretation: these are short loud transients inside the events, of unidentified source.
  - Drug bursts are not the explanation. Only 11 of 289 annotated usable events overlap a Drug annotation, over a median 4.1% of the event, and their `peak_rms` does not differ (δ −0.13, p 0.48).
- **Log transform:** compared for all positive features and adopted for none.
  - `mean_rms`: skew 1.25 → −0.31, maximum |z| 8.9 → 5.7, top-1 share of Σz² 0.18 → 0.07. The extreme count rises from 2 to 3 and the body is symmetric on both scales (Bowley −0.02 / −0.07).
  - `peak_rms`: skew 2.90 → 1.04, extremes 8 → 9.
  - For duration, flatness, rolloff, ZCR mean and the std features, log creates left skew and more extremes (e.g. `duration_s` 13 → 17, `spectral_flatness_mean` 1 → 5, `spectral_rolloff_std` 0 → 6).

### 3. Redundancy (Spearman, usable events; within-session ρ in brackets)
- **Strongest pairs:**
  - Brightness estimators: centroid–ZCR mean 0.85 (0.84), centroid–rolloff 0.81 (0.80), rolloff–ZCR mean 0.76 (0.77).
  - Loudness with brightness: `mean_rms`–centroid −0.84 (−0.76), `peak_rms`–centroid −0.78 (−0.68).
  - Amplitude measures: `mean_rms`–`peak_rms` 0.79 (0.73), `mean_rms`–`total_energy` 0.73 (0.70).
  - Flatness–rolloff 0.73 (0.67).
- **Descriptive |ρ| ≥ 0.8 group:** {mean_rms, spectral_centroid_mean, spectral_rolloff_mean, zcr_mean}, linked through the centroid.
- **`total_energy` is essentially derived.** Log `duration_s` and log `mean_rms` explain 98.1% of the variance of log `total_energy`.
- **Rank R² from the other 12 features:** 0.95 total_energy, 0.93 mean_rms and centroid mean, 0.88 duration and ZCR mean, 0.83 rolloff mean, 0.82 peak_rms, 0.78 flatness mean, 0.64 ZCR std, 0.62 flatness std, 0.61 rolloff std, 0.38 centroid std, 0.26 time-to-peak.
- **Rank R² from the final kept set:** rolloff mean 0.79, ZCR mean 0.83, peak_rms 0.77, total_energy 0.93, time-to-peak 0.20, ZCR std 0.48.
- **Within-session correlations** stay close to the overall ones. The loudness–brightness coupling is therefore not only a between-session effect.

### 4. Date and session stability (ε², date / session)
- **Session effects are strong for level features:**
  - duration 0.17 / 0.39
  - mean_rms 0.04 / 0.55
  - total_energy 0.06 / 0.56
  - centroid mean 0.06 / 0.51
  - rolloff mean 0.21 / 0.40
  - ZCR mean 0.14 / 0.40
  - peak_rms 0.004 / 0.36
  - flatness mean 0.17 / 0.32
- **Within-event variability features are the most stable:** centroid std 0.02 / 0.04, rolloff std 0.03 / 0.04, flatness std 0.02 / 0.17, time-to-peak 0.10 / 0.15, ZCR std 0.10 / 0.22.
- **Dates hide session differences.** In `mean_rms`, dates barely differ (ε² 0.04) while sessions differ strongly (0.55). Example: on 2018-05-03, session #2 sits at `mean_rms` +1.14 and centroid −0.91 robust z, while session #3 sits at −1.44 and +1.34. So loudness and brightness shift together between sessions on the same day.
- **Duration outlier session:** session 2018-02-06#3 has median duration −2.62 robust z.
- **Interpretation limit:** what differs between sessions is unknown (subject, device, placement or protocol), so these are not subject differences. Figures: `date_shift_heatmap.png`, `session_shift_heatmap.png`.

### 5. Calibration Set (first 20 usable events; 9 from session 2018-01-22#1, 11 from 2018-01-23#1)
KEEP features: calibration median shift (rest-MAD units), MAD ratio (calibration/rest), fraction of random 20-subsets with a MAD at least as small, subset p of the median shift, and fraction of the rest outside the calibration range (0.095 expected):

| Feature | Shift | MAD ratio | Subsets with MAD ≤ calib. | Subset p (shift) | Rest outside range |
|---|---|---|---|---|---|
| `duration_s` | +0.23 | 0.73 | 0.244 | 0.39 | 0.215 |
| `mean_rms` | -0.26 | 0.42 | 0.009 | 0.33 | 0.490 |
| `spectral_centroid_mean` | +0.49 | 0.42 | 0.006 | 0.084 | 0.413 |
| `spectral_centroid_std` | -0.02 | 0.35 | 0.005 | 0.92 | 0.426 |
| `spectral_flatness_mean` | +1.08 | 0.39 | 0.003 | <0.001 | 0.470 |
| `spectral_flatness_std` | +0.24 | 0.50 | 0.030 | 0.34 | 0.326 |
| `spectral_rolloff_std` | +0.03 | 0.82 | 0.310 | 0.88 | 0.198 |

- **Location:** the calibration set is representative in location for 6 of the 7 kept features. The exception is `spectral_flatness_mean`, shifted by +1.08 rest-MAD (Cliff's δ 0.61). The deferred rolloff mean (+0.94), ZCR mean (+1.25) and ZCR std (+1.17) are also shifted.
- **Spread:** the calibration set is **not representative in spread**. For 5 of the 7 kept features its MAD falls at or below the 3rd percentile of random 20-event subsets. 20–49% of the remaining usable events fall outside the calibration range, against the 9.5% expected.
- **Why:** the two calibration sessions are internally homogeneous. 19 of the 20 events are annotation-matched (Entry 3).
- **Hypothesis for Stage 6–7, not a result:** a median/MAD baseline fitted to these 20 events will have MADs about 2–3× too small relative to the whole corpus. Robust z of events from other sessions will therefore be inflated, and the held-out anomaly rate will partly reflect calibration homogeneity and session differences rather than inhalation deviations.

### 6. RMS-envelope Edge Effect
- **All 364 events:** the peak is at an edge in 14 events (3.8%): first frame 4, last frame 5, partial tail frame 5.
  - These events are short (median 0.42 vs 1.59 s; δ −0.84, p < 10⁻⁷) and have low CNN confidence (0.62 vs 0.95; δ −0.76).
  - Their `time_to_peak_s` is lower (0.32 vs 0.69 s; δ −0.43), and their `peak_rms` does not differ (δ 0.00, p 0.99).
  - 11 of the 14 were already excluded by the Stage 1 usability rule.
  - Dropping partial frames changes the peak for 10 of them. The largest change is 0.223, for last-frame peaks, all of which are in excluded events.
  - 4 of the 14 have louder audio just outside the event boundary, meaning the event abuts a louder neighbouring sound.
- **Usable events:** 3 of 318 (0.9%): first frame 1, partial tail 2, last frame 0.
  - Dropping partial frames changes `peak_rms` for 2 of them, by at most 0.004.
  - 1 has louder audio just outside the boundary.
  - Edge peaks force `time_to_peak_s` to about 0 or to the end of the event (relative position ≥ 0.98).
- **`time_to_peak_s` is ambiguous for another reason.** Among usable events, frames within 5% of the peak span a median 0.38 s (median 24% of the event; IQR 5–43%). That is close to the feature's robust spread of 0.44 s.
- **Padding in the `*_std` features:**
  - Spectral stds are barely affected. The unpadded-frame std over the stored value has median ratio 0.98 / 0.97 / 0.99 for centroid, flatness and rolloff, with rank agreement 0.977 / 0.978 / 0.992.
  - `zcr_std` is strongly affected: ratio 0.45, rank agreement 0.76, and a median 84% of its 256 ms frames are unpadded.
- **Decision on the edge effect:** after Stage 1 filtering it affects too few usable events (0.9%) to drive a V1 decision. It is recorded as a V1 limitation. The envelope's partial tail frames are an upstream design point to revisit if the envelope features are redefined.

## Feature Decisions (V1)
Principles:
- Features are excluded or deferred only for measurement validity, for being derived from or duplicating an already-represented construct, or for being numerically unsuitable for a median/MAD scale.
- Date/session instability and calibration representativeness are **flagged, not used for exclusion**. Using them would let the held-out distribution drive selection, and session effects may be user or device effects that personalization (V3) should absorb.
- The machine-readable decision is `results/feature_analysis/feature_selection_v1.json`, with evidence in `feature_selection_v1.csv`.

| Feature | Decision | Reason |
|---|---|---|
| `duration_s` | KEEP | Interpretable detector-segmented length. Healthy MAD, symmetric body (Bowley 0.08), 13 extremes on both sides, little predicted by the other kept features (R² 0.32). Session variation flagged. |
| `mean_rms` | KEEP | Sustained loudness. Symmetric body; log compared and not adopted. Relative level only (depends on gain and distance). Coupled to centroid (ρ −0.84, −0.76 within sessions): kept as a distinct construct and flagged. |
| `peak_rms` | DEFER | Loudest 32 ms frame. Heaviest tail (skew 2.90, max abs z 10.9, top-5 share 0.49), not fixed by log, driven by interior transients of unknown source (crest 2.44 vs 1.50). R² 0.77 from the kept features. Investigate the transients; revisit as crest factor. |
| `total_energy` | EXCLUDE | Derived: log duration + log mean_rms explain 98.1% of log energy. Loses < 2% (within-event amplitude modulation), and keeping it would double-count duration and loudness. |
| `time_to_peak_s` | DEFER | Argmax of a flat-topped envelope. Near-peak ambiguity (median 0.38 s) is close to its own spread (0.44 s), and it inherits the detector's start boundary. The edge effect is rare (3/318). Revisit with a smoothed-envelope or energy-centroid timing. |
| `spectral_centroid_mean` | KEEP | Representative brightness measure: relative MAD 5%, skew 0.15, 1 extreme. Preferred over rolloff and ZCR mean. |
| `spectral_centroid_std` | KEEP | Within-event brightness variability. Nearly independent (max abs ρ 0.36, R² 0.23), most session-stable (ε² 0.04), padding contributes about 2%. |
| `spectral_flatness_mean` | KEEP | Noise-likeness, a distinct construct (max ρ 0.73 with the deferred rolloff). Well-behaved. Date/session variation and calibration shift (+1.08) flagged. |
| `spectral_flatness_std` | KEEP | Within-event variability of noise-likeness. Moderate correlations (max abs ρ 0.50), 2 extremes, not padding-driven. |
| `spectral_rolloff_mean` | DEFER | Third brightness estimator (ρ 0.81 with centroid, R² 0.79 from the kept set), asymmetric bounded body (Bowley −0.34), largest date effect (0.21). Loses the upper-band extent beyond the centroid. Revisit in V2. |
| `spectral_rolloff_std` | KEEP | Within-event variability of spectral extent. Low redundancy (max abs ρ 0.37), no extremes, session-stable (0.04), not padding-driven. |
| `zcr_mean` | DEFER | Time-domain brightness proxy (ρ 0.85 with centroid, 0.84 within sessions; R² 0.83), 256 ms frames. Loses the crossing rate beyond the centroid. Revisit in V2. |
| `zcr_std` | EXCLUDE | Measurement artifact as defined: dominated by edge-padded 256 ms frames (unpadded std 0.45× the stored value, ρ 0.76). Would need a new definition. |

V1 feature set (7), on the raw scale:
- `duration_s`
- `mean_rms`
- `spectral_centroid_mean`
- `spectral_centroid_std`
- `spectral_flatness_mean`
- `spectral_flatness_std`
- `spectral_rolloff_std`

EXCLUDE (2): `total_energy`, `zcr_std`. DEFER (4): `peak_rms`, `time_to_peak_s`, `spectral_rolloff_mean`, `zcr_mean`.

## Interpretation
- The kept set covers five constructs: duration, sustained loudness, brightness, noise-likeness, and within-event spectral variability (three measures).
- **Loudness–brightness pair:** `mean_rms` and `spectral_centroid_mean` share a strong joint component. A univariate aggregate will count a combined "loud and dark" or "quiet and bright" deviation twice. V2 (multivariate) is the principled fix. V1 must at least report per-feature robust z so the double counting stays visible.
- **Session-sensitive features:** the features most sensitive to session are the level and brightness features. The most stable are the within-event variability features.

## Limitations
- **No held-out data was reserved for selection.** Stage 2 used all 318 usable events (unlabeled), so later "held-out" evaluation on these same events is not fully independent of feature selection. Decisions deliberately did not depend on held-out consistency, which limits but does not remove this dependence.
- **Sessions are inferred.** They are defined from recording-time gaps, and what they represent (subject, device, placement, protocol) is unknown.
- **Unknown sources remain.** The mechanism behind the loudness–brightness coupling is unknown; one hypothesis is the share of a broadband noise floor in quieter events. The source of the `peak_rms` transients is also unknown.
- **Unmeasured padding effect:** `zcr_mean` also uses edge-padded 256 ms frames, but its padding contribution was not measured.
- **Log comparison limits:** the log comparison covers skew and tails only. Its effect on a particular aggregate is a Stage 4 question.
- **CNN in-sample:** the CNN detector is in-sample for about two-thirds of the recordings (Entry 3).

## Unresolved
1. The loudness–brightness coupling: its mechanism, and how V1 aggregation should treat it.
2. Source of the interior transients behind the extreme `peak_rms` values.
3. A better-defined timing feature: smoothed-envelope peak or energy centroid.
4. Whether the upstream envelope should drop partial tail frames. This matters for a future revision; it has negligible effect on usable events.
5. The calibration set's narrow spread, and how it interacts with the locked first-20 design.

## Next Step — Stage 3 recommendation (robust baseline implementation only)
1. **Baseline:**
   - Implement a per-feature median / 1.4826·MAD baseline on the 7 KEEP features, raw scale.
   - Read the feature list and transforms from `feature_selection_v1.json`.
   - Fit it to the locked calibration set (usable_order 1–20).
   - The baseline object stores the calibration event IDs, n, medians, MADs, feature-set version and dataset hash.
   - Define explicit behaviour for MAD = 0 (none observed) and for missing or non-finite inputs.
2. **Output:** per-feature robust z for every usable event. No aggregation, threshold or NORMAL/ANOMALY yet; those are Stages 4–5.
3. **Adaptation:** frozen baseline with a documented update hook. Adaptation policy comes later.
4. **Descriptive checks, reported in the Stage 3 log entry:** distributions of per-feature robust z for (a) the calibration events, (b) the other events of the calibration sessions, and (c) other sessions. These make the effect of calibration homogeneity visible before any threshold is chosen.
5. **Sensitivity baselines to prepare, not decide:**
   - baselines fitted to random 20-event subsets;
   - baselines fitted to the first 20 events of each session with ≥ 20 usable events (session-wise, not personalized).

   Their purpose is to separate calibration-homogeneity effects from deviation detection in later stages. Adopting any of them would change the locked design and needs the project owner's approval.
