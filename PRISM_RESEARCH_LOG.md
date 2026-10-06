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

---

# Research Entry 5 — 2026-09-30: V1 Robust Baseline and Calibration Dependence (Stage 3)

## Question
How does the per-feature median/MAD baseline behave when fitted to the locked first-20 calibration set? How much do its robust z-scores for the other usable events depend on which 20 events are used for calibration?

Out of scope: anomaly scores, thresholds, NORMAL/ANOMALY labels and multivariate modelling. Large z-scores below are calibration deviations, not anomalies. Without anomaly ground truth nothing here is a false positive.

## Definitions
- **Data:** the 318 usable events in `results/inhale_dataset/inhale_events_v1.csv` (SHA-256 `0dbae978…7cf1`).
- **Features:** the KEEP list of `results/feature_analysis/feature_selection_v1.json`, read at run time rather than hard-coded:
  - `duration_s`
  - `mean_rms`
  - `spectral_centroid_mean`
  - `spectral_centroid_std`
  - `spectral_flatness_mean`
  - `spectral_flatness_std`
  - `spectral_rolloff_std`

  All seven use transform `none`. The loader rejects detection columns, unknown or duplicate features, non-`none` transforms, and a selection whose recorded dataset hash differs from the dataset in use.
- **Primary calibration set:** usable events with `usable_order` 1–20:
  - 9 events from session 2018-01-22#1 (all of that session's usable events)
  - 11 events from session 2018-01-23#1 (50 usable events)

  Sessions follow Entry 4: recordings more than 25 min apart start a new session.
- **Robust statistics**, per feature j, over the calibration events only:
  - median_j = median(x)
  - MAD_j = median(|x − median_j|)
  - scale_j = 1.4826 · MAD_j. The factor 1.4826 ≈ 1/Φ⁻¹(0.75) makes the MAD a consistent estimate of the SD for normal data.
  - z_j = (x − median_j) / scale_j
- **Failure behaviour:** fitting fails with an explicit error if any calibration value is non-finite or any MAD is zero or non-finite. Scoring fails if an input feature value is non-finite.
- **Baseline object:** the baseline is frozen and has no adaptation. An update means fitting a new baseline, which keeps the update mechanism replaceable for a later stage.
- **Implementation:** `src/baseline_v1.py`, command `python src/baseline_v1.py`. Tests: `tests/test_baseline_v1.py` (20 tests; full suite 69 passing). Outputs are in `results/baseline_v1/`.

## Consistency Checks
- The primary medians and scales equal the Stage 2 calibration statistics within 1×10⁻¹⁶ (`calibration_vs_rest.csv`).
- Per-session usable-event counts equal Stage 2's.

## Results

### 1. Primary baseline parameters (`baseline_parameters.csv`, `calibration_summary.csv`, `baseline_v1.json`)
| Feature | Median | MAD | Scale (1.4826·MAD) | All-usable median | All-usable scale | Scale ratio | Median shift (all-usable SD) | Random-20 percentile of scale |
|---|---|---|---|---|---|---|---|---|
| `duration_s` | 1.696 | 0.152 | 0.2254 | 1.632 | 0.2965 | 0.76 | +0.22 | 24.0% |
| `mean_rms` | 0.1772 | 0.008594 | 0.01274 | 0.1838 | 0.02909 | 0.44 | -0.23 | 0.9% |
| `spectral_centroid_mean` | 0.38355 | 0.00808 | 0.01198 | 0.37053 | 0.02787 | 0.43 | +0.47 | 0.4% |
| `spectral_centroid_std` | 0.039185 | 0.002298 | 0.003407 | 0.039401 | 0.008977 | 0.38 | -0.02 | 0.2% |
| `spectral_flatness_mean` | 0.16155 | 0.007428 | 0.01101 | 0.13224 | 0.02971 | 0.37 | +0.99 | 0.3% |
| `spectral_flatness_std` | 0.058829 | 0.004633 | 0.006868 | 0.055673 | 0.01327 | 0.52 | +0.24 | 3.2% |
| `spectral_rolloff_std` | 0.081281 | 0.0095 | 0.01408 | 0.08071 | 0.01691 | 0.83 | +0.03 | 30.8% |

The random-20 percentile is the share of 2,000 random 20-event calibration sets with a scale no larger than the primary one. For 5 of 7 features the primary calibration scale is narrower than 96.8–99.8% of random 20-event sets.

### 2. Robust z by comparison group (`robust_z_scores.csv`, `group_z_summary.csv`, `z_by_group.png`)
Groups:
- **Calibration:** 20 events. By construction their median z is 0 and robust SD of z is 1. These z-scores are in-sample.
- **Same sessions, not calibration:** 39 events, all from 2018-01-23#1.
- **Other sessions:** 259 events.

Each cell gives median z / robust SD of z / median |z| / % with |z| > 3.5. The 3.5 cut-off is the Stage 2 descriptive convention, not a threshold. Under normality the median |z| would be 0.67.

| Feature | Same sessions (n=39) | Other sessions (n=259) |
|---|---|---|
| `duration_s` | +0.75 / 1.79 / 0.82 / 26% | -0.46 / 1.37 / 0.96 / 8% |
| `mean_rms` | -0.42 / 1.29 / 0.82 / 3% | +0.74 / 2.55 / 2.01 / 22% |
| `spectral_centroid_mean` | +0.35 / 1.13 / 0.82 / 0% | -1.61 / 2.25 / 2.12 / 19% |
| `spectral_centroid_std` | -0.44 / 3.01 / 1.84 / 18% | +0.21 / 2.93 / 2.01 / 27% |
| `spectral_flatness_mean` | -1.07 / 2.37 / 1.29 / 21% | -2.97 / 2.41 / 2.97 / 40% |
| `spectral_flatness_std` | -0.46 / 0.93 / 0.65 / 5% | -0.51 / 2.26 / 1.68 / 11% |
| `spectral_rolloff_std` | +0.01 / 1.32 / 0.90 / 0% | -0.04 / 1.21 / 0.81 / 0% |

- **Outside the calibration range:** 5–38% of same-session events and 20–54% of other-session events fall outside the calibration events' own z range for a feature. The largest single deviation is `mean_rms` z = 20.9.
- **Session decomposition** (`session_decomposition.csv`, 11 sessions with ≥ 5 non-calibration events):
  - **Within-session robust SD of z** (median across sessions): duration 1.26, mean_rms 1.40, centroid mean 1.37, centroid std 2.87, flatness mean 1.89, flatness std 1.54, rolloff std 1.16.
  - **Between-session robust SD of session-median z:** 0.74, 1.21, 0.96, 0.89, 1.00, 0.84, 0.35 respectively.
  - **Range of session-median z:** duration −3.73 to +0.75, mean_rms −2.77 to +3.11, centroid mean −3.19 to +2.03, and flatness mean −6.62 to −1.07. Every session's median flatness is below the calibration median.

### 3. Sensitivity A: random 20-event calibration sets (`calibration_sensitivity.csv`, `sensitivity_draws.csv`)
**Method:** 2,000 random 20-event sets drawn without replacement from the 318 usable events (seed 20260930). Each is fitted like the primary baseline and applied to its own 298 non-calibration events.

**Results:**
- **Scale ratio (calibration scale / all-usable scale), 5th–95th percentile:**
  - duration 0.56–1.56
  - mean_rms 0.57–1.50
  - centroid mean 0.59–1.35
  - centroid std 0.56–1.57
  - flatness mean 0.55–1.36
  - flatness std 0.56–1.45
  - rolloff std 0.60–1.40

  Medians are 0.92–0.99. This extends Stage 2's 0.54–1.57 range to the kept features.
- **Median shift:** −0.47 to +0.46 all-usable SD (5th–95th percentile).
- **Implied z-scores for the non-calibration events:** robust SD of z has a median of 1.01–1.09 (5th–95th percentile 0.62–1.86), and median |z| is 0.73–0.75.
- **Primary set by comparison:** robust SD of z is 1.23–2.83 and median |z| 0.82–2.86. So a typical 20-event calibration gives z-scores on roughly the intended scale. The first-20 set does not, and a single random set still has a ±50% scale uncertainty.

### 4. Sensitivity B: session-aware calibration sets
Subject identity is unknown. Sessions are recording sittings, not people.

- **B1, session-spread (random):** 2,000 sets (seed 20260931). Sessions are visited in random order, and each pass takes one random unused event from every session until 20 are chosen. All 18 sessions with usable events are represented in every set; sessions with fewer events are relatively over-represented. Results by feature:

  | Feature | Scale ratio, median (5th–95th pct.) | Robust SD of non-calibration z |
  |---|---|---|
  | duration | 1.12 (0.64–1.68) | 0.90 |
  | mean_rms | 0.73 (0.44–1.04) | 1.40 |
  | centroid mean | 0.76 (0.49–1.04) | 1.34 |
  | centroid std | 1.17 (0.70–1.67) | 0.85 |
  | flatness mean | 0.82 (0.50–1.15) | 1.23 |
  | flatness std | 0.79 (0.50–1.13) | 1.31 |
  | rolloff std | 1.04 (0.70–1.47) | 0.95 |

  Median |z| is 0.61–0.99. Spreading calibration across sessions removes most of the inflation seen with the primary set. The level features still sit below 1, most likely because the all-usable scale includes large sessions at opposite offsets (e.g. `mean_rms` in 2018-05-03#2 vs #3, Entry 4). This explanation has not been tested.
- **B1', session-spread (chronological, one set):** the same round-robin using each session's earliest events. Scale ratios are 0.63–1.07 and the robust SD of non-calibration z is 0.93–1.59.
- **B2, single-session first 20:** for each session with ≥ 40 usable events (2018-01-23#1 n=50, 2018-05-03#2 n=57, 2018-05-03#3 n=52), fit on its first 20 usable events. Compare the rest of that session (30, 37 and 32 events) with all other sessions.
  - **Scale ratios:** 0.24–1.92. 20 of the 21 feature/session values are below 1; the exception is duration in 2018-01-23#1 at 1.92. The first 20 consecutive events of one sitting are generally narrower than the corpus. This is not specific to the primary set.
  - **Same session vs other sessions:** the remainder of the calibration session has a smaller median |z| than other sessions for 5/7, 6/7 and 4/7 features respectively. The primary baseline gives 6/7. Counted over all four single-set designs:
    - the level features (duration, mean_rms, centroid mean, flatness mean) are closer within the session in all four;
    - flatness std is closer in three (a tie in 2018-05-03#3);
    - centroid std is closer only in the primary design;
    - rolloff std is closer only in 2018-05-03#2.
  - **Within-session drift:** in 2018-05-03#2, the 37 events after the first 20 have median z −1.72 (`mean_rms`), +1.65 (centroid mean) and +1.10 (flatness mean) relative to their own session's first 20.
  - **Weakness:** only three sessions qualify, so B2 is descriptive.

## Interpretation
1. **The calibration set is unusually narrow.** For 5 of 7 features its scale lies in the bottom 0.2–3.2% of random 20-event sets. Consequently, other-session events have z-scores spread 2.3–2.9× wider than intended for `mean_rms`, centroid mean, centroid std, flatness mean and flatness std. The inflation is smallest for rolloff std (1.21) and duration (1.37).
2. **Both mechanisms operate, with different weight per feature.**
   - **Narrow spread:** even within a session, spread is 1.16–2.87× the calibration spread. This dominates `spectral_centroid_std`: within-session 2.87 against between-session 0.89, and other-session median z only +0.21.
   - **Session offsets:** these dominate `spectral_flatness_mean` (all sessions below the calibration median; other-session median z −2.97) and contribute strongly to `mean_rms` and `spectral_centroid_mean`, whose session medians spread 1.21 and 0.96 robust SD.
   - **Resistant features:** `spectral_rolloff_std` and `duration_s` are least affected.
3. **Within-session vs other sessions:** later events in the calibration session are closer to calibration than other sessions for the level features, and mostly for flatness std. They are not reliably closer for centroid std and rolloff std. Those are the most session-stable features (Entry 4), and their inflation comes from the narrow calibration spread rather than from sessions.
4. **General property of single sittings:** narrowness is expected from any calibration made of consecutive events in one or two sittings (B2), and recordings drift within a sitting. This matters for the future personalised design (V3). A user who calibrates in one sitting would face the same problem.
5. **Adequacy of the first-20 calibration.** It is adequate as a reference for consistency *within its own sessions*. It is **not adequate as a corpus-wide V1 baseline in its current form**: deviations of events from other sessions are dominated by calibration narrowness and session offsets, not by inhalation-level differences between events. The primary design is still the locked V1 design. Changing it (larger N, session-spread calibration or any other rule) is a decision for the project owner.

## Limitations
- **Calibration z-scores are in-sample:** each calibration event helped define the median and MAD it is scored against, so its own z-scores look better than a new event's would.
- **Sessions are inferred** from recording gaps. Session effects cannot be attributed to subject, device, placement or protocol.
- **B1 is diagnostic only.** Its sets draw on all sessions, including ones recorded after the primary calibration, so they are not a deployable calibration protocol.
- **B2 covers only three sessions.**
- **No held-out data was reserved.** All 318 usable events were also used in Stage 2 feature selection (Entry 4).
- **Pooling:** the robust SD of z pools events whose sessions differ in size; large sessions dominate pooled statistics.

## Next Step — Stage 4 recommendation (anomaly scoring, no threshold)
1. **Candidate aggregates:**
   - Define a small set of candidates over the 7 per-feature z-scores: mean |z|, root-mean-square z, and max |z|.
   - Compute them for all usable events against the primary baseline, reporting per group and per session, with per-feature contribution shares.
   - Under the primary baseline the features' z scales are not comparable (other-session robust SD 1.21–2.93). The `mean_rms` / `spectral_centroid_mean` coupling (Entry 4) also double-counts.
   - Choose the aggregate on documented properties, not on how many events it flags.
2. **Leave-one-out calibration scores:** compute them for the 20 calibration events, refitting on the other 19. In-sample calibration scores are optimistically small, and Stage 5 derives the threshold percentile from calibration scores.
3. **Owner decision before Stage 5:** a percentile threshold taken from this narrow calibration set will mark a large share of other-session events as deviating, for reasons unrelated to the individual inhalation. Options:
   - keep first-20 and report all Stage 6–7 results per session;
   - approve a revised calibration design, e.g. more events or session-spread;
   - run both, with first-20 as primary and one pre-specified alternative as sensitivity.

---

# Research Entry 6 — 2026-09-30: V1 Candidate Combined Scores (Stage 4)

**No anomaly threshold or NORMAL/ANOMALY classification was introduced in Stage 4.** Large scores are standardized calibration deviations, not anomalies.

## Question
How do three candidate ways of combining the seven Stage 3 robust z-scores differ?
- Which features drive each score?
- How much do the scores increase outside the calibration sessions?
- How optimistic are the in-sample calibration scores compared with leave-one-out scores?

The design is frozen as in Entries 3–5: 318 usable events, the 7 features from `feature_selection_v1.json`, primary calibration = first 20 usable events, per-feature median + 1.4826·MAD. There is no multivariate model, personalization, adaptation or calibration change.

## Score and Contribution Definitions
For an event with robust z-scores z_1..z_7 (d = 7), the scores are:
- mean_abs_z = mean_j |z_j|
- rms_z = sqrt(mean_j z_j²)
- max_abs_z = max_j |z_j|

There is no weighting and no re-normalization.

Each score has its own contribution definition, and the three are not interchangeable:
- **mean_abs_z:** |z_j|. The share |z_j| / Σ|z| is also reported so events can be compared; it is derived from |z_j| and does not replace it.
- **rms_z:** z_j² / Σ z_k².
- **max_abs_z:** the feature(s) attaining the maximum. Every tied feature is listed; no ties occurred.

## Implementation
`src/scoring_v1.py`, command `python src/scoring_v1.py`. Tests: `tests/test_scoring_v1.py` (18 tests; full suite 87 passing).

**Traceability:** the saved Stage 3 baseline (`results/baseline_v1/baseline_v1.json`, SHA-256 `d6fb5457…b2c5`) is used unchanged. The run checks that:
- the baseline's features equal the feature selection;
- refitting the first 20 events reproduces its parameters exactly (difference 0);
- its z-scores reproduce the Stage 3 file within 3.6×10⁻¹⁵.

**Leave-one-out (LOO):** for each of the 20 calibration events, median/MAD is refitted on the other 19 events (`n_training` = 19 for every event, with a check that the held-out event is absent). The held-out event is then scored. LOO parameters are saved in `leave_one_out_parameters.csv`, and the primary baseline is not altered.

**Decisions:**
1. **Two calibration views:** calibration events are reported both in-sample (primary baseline, which they helped define) and LOO.
2. **Sessions exclude calibration events:** session distributions exclude calibration events, so session 2018-01-22#1, whose 9 usable events are all calibration events, has no session row values.
3. **Normal reference:** a reference for d independent N(0,1) z-scores is simulated (100,000 draws, seed 20261001) to indicate scale only. The real z-scores are neither independent nor normal.
4. **"Isolated-extreme" events** are defined descriptively as events whose largest |z| is at least 2× the second largest.
5. **Exact decomposition of group-mean increases:**
   - mean(mean_abs_z) = (1/d) Σ_j mean|z_j|
   - mean(rms_z²) = (1/d) Σ_j mean z_j²

   The increase in either group mean therefore splits exactly over features. Medians do not decompose.

Outputs in `results/scoring_v1/`:
- `event_scores.csv`: all Stage 3 event metadata, 7 z-scores, 3 scores, the max feature, |z_j|, RMS shares, and single-feature influence.
- `leave_one_out_scores.csv`
- `leave_one_out_parameters.csv`
- `group_summary.csv`
- `session_summary.csv`
- `feature_contributions.csv`
- `inflation_decomposition.csv`
- `single_feature_influence.csv`
- `scoring_summary.json`
- `score_distributions.png`
- `feature_dominance.png`

## Results

### 1. Score distributions (`group_summary.csv`)
| Group | n | mean_abs_z median [p05, p95] (max) | rms_z median [p05, p95] (max) | max_abs_z median [p05, p95] (max) |
|---|---|---|---|---|
| Calibration, in-sample | 20 | 0.86 [0.41, 1.68] (1.77) | 1.07 [0.51, 1.93] (2.11) | 2.05 [0.80, 3.07] (3.11) |
| Calibration, leave-one-out | 20 | 0.96 [0.48, 1.91] (1.95) | 1.18 [0.59, 2.20] (2.30) | 2.31 [0.91, 3.28] (3.71) |
| Same sessions, not calibration | 39 | 1.20 [0.66, 3.18] (3.70) | 1.42 [0.85, 3.87] (4.76) | 2.61 [1.51, 7.54] (9.40) |
| Other sessions | 259 | 1.98 [1.04, 3.39] (9.54) | 2.44 [1.28, 4.16] (10.99) | 4.28 [2.18, 7.87] (20.92) |
| Independent N(0,1) reference | — | 0.78 [0.45, 1.20] | 0.95 [0.56, 1.42] | 1.67 [0.94, 2.68] |

**Per-session medians** (non-calibration events; mean_abs_z / rms_z / max_abs_z):
- 2018-01-23#1: 1.20 / 1.42 / 2.61 (n=39)
- 2018-01-23#2: 1.55 / 1.85 / 3.44 (n=27)
- 2018-01-23#3: 1.46 / 1.90 / 3.35 (n=29)
- 2018-02-05#1: 2.15 / 2.72 / 4.34 (n=27)
- 2018-02-06#3: 2.32 / 2.71 / 4.51 (n=13)
- 2018-05-02#1: 2.95 / 3.74 / 6.87 (n=11)
- 2018-05-03#2: 2.43 / 2.89 / 5.28 (n=57)
- 2018-05-03#3: 1.92 / 2.43 / 4.33 (n=52)

Sessions with fewer than 5 events are in `session_summary.csv`. Later sittings on the calibration day (2018-01-23#2, #3) sit closer to calibration than sittings on later dates.

### 2. How the three scores differ
- **Rank agreement** (Spearman, all 318 events): mean_abs–rms 0.977, rms–max 0.948, mean_abs–max 0.881. Within other sessions the values are 0.968, 0.926 and 0.832.
- **Relationship:** mean_abs_z and rms_z are almost interchangeable for ranking. max_abs_z departs most, and it sits on a larger scale (2.2–2.4× mean_abs_z, depending on the group).
- **Sensitivity to one feature:**
  - **Share held by the largest feature:** median 0.31 of mean_abs_z (as |z| share), 0.46 of rms_z (as z² share), and 1 of max_abs_z by definition. The p95 values are 0.48 and 0.78.
  - **Marginal effect of increasing the largest |z| by one unit:** 1/d = 0.14 for mean_abs_z, at most 1/√d ≈ 0.38 for rms_z, and 1 for max_abs_z.
- **Isolated extremes:** 37 of 318 events are isolated-extreme; the isolated feature is flatness mean in 19, centroid std in 15, flatness std in 2 and duration in 1.
  - Median percentile rank of these events: 0.68 under max_abs_z, 0.46 under rms_z, 0.42 under mean_abs_z.
  - For the other 281 events: 0.47, 0.51 and 0.51.
  - max_abs_z is therefore the score most driven by a single isolated feature. rms_z reduces that influence, and mean_abs_z reduces it most.
- **The single largest deviation** (`rec2018-05-03_11h22m05.830s.wav` event 1, session 2018-05-03#2) comes from `mean_rms` (z = 20.9). It ranks first under all three scores: mean_abs_z 9.54, rms_z 10.99, max_abs_z 20.92.

### 3. Feature dominance (`feature_contributions.csv`, `feature_dominance.png`)
| Feature | argmax share: LOO calib / same sess. / other | mean RMS share: LOO calib / same sess. / other | median abs z: LOO calib / same sess. / other |
|---|---|---|---|
| `duration_s` | 0.15 / 0.21 / 0.06 | 0.14 / 0.18 / 0.08 | 0.74 / 0.82 / 0.96 |
| `mean_rms` | 0.10 / 0.05 / 0.10 | 0.11 / 0.10 / 0.14 | 0.68 / 0.82 / 2.01 |
| `spectral_centroid_mean` | 0.15 / 0.05 / 0.05 | 0.15 / 0.08 / 0.14 | 0.80 / 0.83 / 2.12 |
| `spectral_centroid_std` | 0.15 / 0.36 / 0.26 | 0.14 / 0.24 / 0.20 | 0.68 / 1.84 / 2.01 |
| `spectral_flatness_mean` | 0.25 / 0.18 / 0.43 | 0.20 / 0.21 / 0.28 | 0.80 / 1.29 / 2.98 |
| `spectral_flatness_std` | 0.05 / 0.05 / 0.08 | 0.12 / 0.09 / 0.11 | 0.69 / 0.65 / 1.68 |
| `spectral_rolloff_std` | 0.15 / 0.10 / 0.02 | 0.13 / 0.10 / 0.05 | 0.84 / 0.90 / 0.81 |

- **Calibration events (LOO):** shares are close to equal (1/7 ≈ 0.14 each).
- **Other sessions:** `spectral_flatness_mean` dominates every score (largest |z| in 43% of events, mean RMS share 0.28), followed by `spectral_centroid_std`.
- **Rest of the calibration sessions:** `spectral_centroid_std` dominates (largest in 36% of events), followed by flatness mean and duration.
- **`spectral_rolloff_std`** contributes least outside calibration.
- **Overall (all 318 events):** the max feature is flatness mean in 122 events, centroid std 83, mean_rms 30, duration 27, flatness std 25, centroid mean 19, rolloff std 12.

### 4. Cross-session increase (`scoring_summary.json`, `inflation_decomposition.csv`)
**Median ratios:**

| Comparison | mean_abs_z | rms_z | max_abs_z |
|---|---|---|---|
| Other sessions / LOO calibration | 2.07 | 2.07 | 1.85 |
| Other sessions / same sessions | 1.66 | 1.72 | 1.64 |
| Same sessions / LOO calibration | 1.25 | 1.20 | 1.13 |
| Other sessions / in-sample calibration | 2.30 | 2.28 | 2.09 |

**Other sessions − LOO calibration.** Exact per-feature shares of the increase in group-mean mean_abs_z (and in mean rms_z²):
- flatness mean 28% (31%)
- centroid std 22% (23%)
- mean_rms 20% (19%)
- centroid mean 16% (14%)
- flatness std 12% (9%)
- duration 4% (4%)
- rolloff std 0% (0%)

The four features flagged in Stage 3 carry 84% (mean_abs_z) and 88% (rms_z²) of the increase. For max_abs_z, the share of events whose maximum is flatness mean rises from 0.25 to 0.43, and centroid std from 0.15 to 0.26.

**Other sessions − same sessions** (mean_abs_z / rms_z² shares):
- centroid mean 30% / 28%
- mean_rms 29% / 34%
- flatness mean 23% / 25%
- flatness std 20% / 15%
- centroid std 13% / 20%
- duration −15% / −22%

Same-session events deviate more in duration.

**Same sessions − LOO calibration:** flatness mean 34%, centroid std 33% and duration 28% (mean_abs_z). Within the calibration session, later events move mainly in these features.

### 5. In-sample vs leave-one-out calibration
- **All 20 events:** the LOO score exceeds the in-sample score for all 20 calibration events on all three scores (Wilcoxon signed-rank p = 1.9×10⁻⁶).
- **Medians:** mean_abs_z 0.86 → 0.96 (paired median ratio 1.11); rms_z 1.07 → 1.18 (1.08); max_abs_z 2.05 → 2.31 (1.06).
- **p95:** 1.68 → 1.91, 1.93 → 2.20, 3.07 → 3.28.
- **Maxima:** 1.77 → 1.95, 2.11 → 2.30, 3.11 → 3.71.

### 6. The correlated pair `mean_rms` / `spectral_centroid_mean`
- **Correlation of z:** Spearman −0.84 in other sessions, −0.78 in LOO calibration and −0.68 in the same sessions. The two z-scores have opposite signs in 82% of other-session events.
- **Per-event domination:** the pair does not dominate individual events.
  - Median joint share in other sessions: 0.26 of RMS and 0.30 of |z|, against an equal-share reference of 2/7 = 0.29.
  - The pair are the two largest |z| in 7.7% of other-session events, against 4.8% if features were exchangeable.
  - The pair contains the maximum feature in 15% of other-session events, against 29% for exchangeable features.
- **Cross-session difference:** here the double counting is visible. The pair accounts for 59% (mean_abs_z) and 62% (rms_z²) of the other-sessions-minus-same-sessions increase. It also accounts for 35% and 33% of the other-sessions-minus-LOO increase.

## Answers to the Stage 4 Questions
1. **How different are the three scores?** mean_abs_z and rms_z rank events almost identically (ρ 0.97–0.98). max_abs_z differs more (ρ 0.83–0.95 with the others) and sits on a scale 2.2–2.4× larger.
2. **Most sensitive to isolated extremes:** max_abs_z. Isolated-extreme events reach median percentile 0.68 under it, against 0.46 (rms_z) and 0.42 (mean_abs_z).
3. **Dominant features:**
   - **Other sessions:** `spectral_flatness_mean`, then `spectral_centroid_std`, under all three scores.
   - **Rest of the calibration session:** `spectral_centroid_std`.
   - **Everywhere:** `spectral_rolloff_std` contributes least.
4. **Increase for other sessions:** relative to LOO calibration, the median rises about 2.1× for mean_abs_z and rms_z and 1.85× for max_abs_z. The rest of the calibration session rises only 1.13–1.25×.
5. **In-sample vs LOO:** in-sample calibration scores are lower for every calibration event, by a median 6–11%.
6. **Does LOO prove the baseline narrow?** Only partly. LOO confirms that in-sample calibration scores are optimistic, but only modestly. It does not remove the narrowness: other-session scores remain about 2× the LOO calibration scores. Even LOO calibration scores exceed the independent-normal reference medians by about 1.2–1.4×. In-sample optimism is not the main reason for the cross-session gap.
7. **Same features behind the increase?** Yes. The four features flagged in Stage 3 (flatness mean, centroid std, mean_rms, centroid mean) account for 84–88% of the other-sessions-over-calibration increase, consistent with Entry 5.
8. **Do RMS or mean_abs_z reduce single-feature influence compared with max_abs_z?** Yes. The largest feature's median share is 0.31 (mean_abs_z) and 0.46 (rms_z), against 1 for max_abs_z. Isolated-extreme events are ranked lower by both.
9. **Double counting from the correlated pair?** Not within single events: the pair's joint share is about equal to two features' fair share. It is visible across sessions, where the pair jointly carries about 60% of the difference between other sessions and the calibration session.

## Implications of the Narrow First-20 Calibration
- **Scores reflect session membership:** all three candidate scores inherit the Stage 3 problem, so their level is driven largely by session membership. The median other-session score is about 2× the LOO calibration score, and 84–88% of that increase comes from four features whose calibration spread is unusually narrow or whose sessions sit at offsets.
- **Choosing a score won't fix it:** the three scores differ mainly in how they treat single extreme features, not in how they respond to this calibration problem.
- **Thresholds would be dominated by session effects:** any threshold taken from the 20 calibration scores would be driven mainly by session differences. This holds even using the LOO scores, which are only 6–11% higher.

## Limitations
- **Few calibration scores:** 20 calibration scores support only coarse percentiles (5% steps).
- **Sessions are inferred** and not attributable to subject, device or setup (Entries 4–5).
- **Normal reference is descriptive:** the independent-normal reference only indicates scale.
- **Descriptive definitions:** the isolated-extreme 2× rule and the exchangeable-feature references are descriptive choices, not tests.
- **Exact decomposition covers group means only:** it does not apply to medians or to max_abs_z.
- **No held-out data:** all 318 events were used in Stage 2 feature selection (Entry 4).
- **No ground truth:** no anomaly ground truth exists, so nothing here measures detection performance.

## Next Step — Stage 5 recommendation (future experiment; no threshold chosen here)
1. **Primary score:**
   - Pre-register one primary candidate score before any threshold work, chosen on properties, not on the distribution it produces.
   - `rms_z` is a reasonable middle ground: it ranks events almost like `mean_abs_z` but still responds to a single strongly deviating feature. `max_abs_z` is the most exposed to the narrowly calibrated features.
   - The choice belongs to the project owner.
2. **Percentile-threshold experiment:**
   - Derive candidate percentiles only from calibration LOO scores (n = 20), stating the percentile resolution this allows.
   - Report the resulting calibration-deviation rates per group and per session as descriptions.
   - Show threshold sensitivity across several percentiles, without calling exceedances anomalies or error rates.
3. **Calibration design decision:** the Stage 5 experiment should run only after the owner decides whether the first-20 calibration stays primary (Entry 5). Otherwise the threshold will mostly encode session differences.

---

# Research Entry 7 — 2026-09-30: Baseline Strategy Experiment Across Sessions (Stage 5)

**No anomaly threshold or NORMAL/ANOMALY classification was introduced in Stage 5.** Scores are standardized deviations from a baseline. NORMAL/ANOMALY terminology (baseline-consistent / substantial deviation from the established baseline) is unchanged, and no clinical or technique-quality claim is made.

## Question
Which baseline produces stable standardized scores on sessions that were **not** used to fit it? The question is not which baseline produces the smallest scores.

The 7 V1 features, the robust z definition and the three Stage 4 candidate scores are unchanged:
- z_j = (x_j − center_j) / (1.4826·MAD_j)
- mean_abs_z = mean|z_j|
- rms_z = sqrt(mean z_j²)
- max_abs_z = max|z_j|

Stages 1–4 were not modified. This stage reads their outputs.

## Data and Splits
- **Events and sessions:** 318 usable events in 18 sessions. Sessions are sittings separated by > 25 min (Entry 4); they are not subjects.
- **Session sizes:** 57, 52, 50, 29, 27, 27, 13, 11, 9, 9, 9, 8, 4, 4, 4, 2, 2, 1.
- **Evaluation sets:** every strategy is compared only on events it scores in full.

  | Set | Events | Sessions | Contents |
  |---|---|---|---|
  | `all_usable` | 318 | 18 | every usable event |
  | `sessions_ge5` | 301 | 12 | sessions with ≥ 5 events |
  | `sessions_ge20` | 242 | 6 | sessions with ≥ 20 events |
  | `post_warmup_k3` | 268 | 15 | events after a session's first 3 |
  | `post_warmup_k5` | 241 | 12 | events after a session's first 5 |
  | `post_warmup_k10` | 186 | 8 | events after a session's first 10 |
- **Pre-specified rules** (constants in `src/baseline_strategies.py`, fixed before any Stage 5 result was computed):
  - **Session-location normalization** requires ≥ 5 events. This is the Stage 2 minimum group size; the leave-one-out session median then uses ≥ 4 other events.
  - **Session-scale normalization** requires ≥ 20 events, so the session MAD comes from ≥ 19 other events. Normal-theory relative SE of the MAD is ≈ 1.166/√n, about 27% at n = 19.
  - **Deployable warm-up:** k = 5 events. k = 3 and 10 are reported as sensitivity only and are not used to choose.
  - **Resampling:** bootstrap 2,000 draws, permutations 2,000, seed 20261002.

## Strategies
| Id | Baseline | Status |
|---|---|---|
| A | First-20 calibration (Stage 3/4). Reproduced from Stage 4 outputs, with calibration events scored leave-one-out; scores match Stage 4 within 1.8×10⁻¹⁵. | Deployable; historical control |
| B | median/MAD of all 318 events, scored on the same events | **Descriptive only (in-sample)** |
| C | Leave-one-session-out (LOSO): for each session s, fit median/MAD on every usable event of the other sessions and score the events of s. Every event is scored once, by a baseline that never saw its session. | Deployable global baseline; main generalization test |
| D1 | x′ = x − median(other events of the same session), all 7 features. Center/scale of x′ fitted LOSO on the other sessions (≥ 5 events). | **Offline diagnostic**: uses later events of the session |
| D2 (k) | x′ = x − median(first k events of the session). Only later events are scored; center/scale as D1 (from other, complete sessions). | Deployable with a k-event cold start per session |
| D3 | z = (x − median) / (1.4826·MAD), both from the other events of the same session (sessions ≥ 20). | **Offline diagnostic** |
| D1L / D2L (k) | As D1 / D2, but session location removed from the LEVEL features only (duration, mean_rms, centroid mean, flatness mean). The WITHIN-EVENT VARIABILITY features (centroid std, flatness std, rolloff std) keep C-style global LOSO treatment. | Offline / deployable, as D1 / D2 |

**Reference and generalization ratio:** for each strategy, "reference" is the median score of the events its baseline was fitted on (in-sample). For C/D this is per fold; for A it is the leave-one-out calibration scores (Entry 6); for D3 it is the session's own in-sample scores. The **generalization ratio** is the median score of unseen events divided by the reference median. It is 1 when unseen sessions look like the fitting data.

**Session dependence:** measured by the Kruskal–Wallis ε² of scores (or per-feature z) across sessions with ≥ 5 scored events, together with the range of session medians.

**Order of work (disclosure):** the level-only variants (D1L, D2L) and the joint all-feature scale-heterogeneity statistic were added after the all-feature D1/D2 results were seen. They follow from the task's LEVEL / WITHIN-EVENT VARIABILITY split and Stage 2's finding that centroid std and rolloff std are session-stable. They add no tunable parameter, and all variants are reported.

Implementation: `src/baseline_strategies.py`, command `python src/baseline_strategies.py`. Tests: `tests/test_baseline_strategies.py` (19 tests; full suite 106 passing). Outputs in `results/baseline_strategies/`:
- `heldout_scores.csv`
- `strategy_summary.csv`
- `feature_stability.csv`
- `feature_contributions.csv`
- `session_medians.csv`
- `baseline_parameters.csv`
- `parameter_variability.csv`
- `session_scale_stability.csv`
- `session_scale_heterogeneity.csv`
- `experiment_summary.json`
- `session_medians_rms_z.png`
- `feature_session_effect.png`

## Results

### 1. Held-out score distributions, sessions with ≥ 5 events (301 events, 12 sessions)
Median [5th–95th percentile], generalization ratio, range of session medians and session ε². Independent-N(0,1) medians for reference: 0.78 / 0.95 / 1.67.

| Strategy | mean_abs_z | rms_z | max_abs_z | Ratio (rms_z) | rms_z session medians | ε² (rms_z) |
|---|---|---|---|---|---|---|
| A first-20 | 1.87 [0.75, 3.37] | 2.28 [0.89, 4.13] | 4.14 [1.64, 7.63] | 1.94 | 1.30–3.74 | 0.28 |
| B pooled (in-sample) | 0.81 [0.38, 1.40] | 1.01 [0.47, 1.77] | 1.72 [0.81, 3.64] | 1.02 | 0.68–1.44 | 0.14 |
| C LOSO global | 0.90 [0.41, 1.53] | 1.09 [0.51, 1.95] | 1.88 [0.89, 3.72] | 1.11 | 0.68–1.48 | 0.21 |
| D1 offline location | 0.80 [0.34, 1.72] | 0.98 [0.45, 2.12] | 1.76 [0.78, 4.23] | 1.01 | 0.76–1.55 | 0.10 |
| D1L offline level location | 0.82 [0.37, 1.75] | 0.99 [0.44, 2.11] | 1.77 [0.81, 4.15] | 1.02 | 0.68–1.43 | 0.10 |

C on all 318 events (including the 6 small sessions): rms_z median 1.09 [0.51, 1.92], generalization ratio 1.10.

**Per-feature held-out z under C** (robust SD of z / robust SD of session-median z / session ε²):

| Feature | Family | Robust SD of z | Robust SD of session medians | Session ε² |
|---|---|---|---|---|
| duration | level | 1.15 | 0.65 | 0.45 |
| mean_rms | level | 1.19 | 0.37 | 0.64 |
| centroid mean | level | 1.17 | 0.51 | 0.62 |
| flatness mean | level | 1.12 | 0.56 | 0.39 |
| centroid std | within-event variability | 0.93 | 0.29 | 0.05 |
| flatness std | within-event variability | 1.09 | 0.58 | 0.23 |
| rolloff std | within-event variability | 1.01 | 0.35 | 0.05 |

Under A the robust SDs of z were 1.23–2.73. Under D1, the session ε² of every feature is about 0 (−0.034 to −0.027), and the robust SD of session-median z is 0.01–0.07.

### 2. Deployable warm-up vs offline location, events after the first k of each session
rms_z median, generalization ratio and session ε²:

| Set (events) | C LOSO | D1 offline | D1L offline | D2 warm-up | D2L warm-up |
|---|---|---|---|---|---|
| post-k5 (241), primary k = 5 | 1.10, 1.12, 0.14 | 0.96, 0.99, 0.08 | 1.00, 1.03, 0.06 | 1.16, 1.19, 0.11 | 1.12, 1.16, 0.14 |
| post-k3 (268), k = 3 (sensitivity) | 1.10, 1.12, 0.17 | — | — | 1.64, 1.68, 0.47 | 1.51, 1.56, 0.49 |
| post-k10 (186), k = 10 (sensitivity) | 1.14, 1.15, 0.11 | 0.94, 0.97, 0.06 | 0.99, 1.03, 0.03 | 1.07, 1.10, 0.05 | 1.06, 1.10, 0.02 |

**Per-feature effect of the 5-event warm-up** (post-k5 set, session ε²; D2 → D2L → C):
- **Level features** go from C's 0.40–0.70 to 0.33–0.53. Flatness mean rises from 0.40 under C to 0.53.
- **Within-event variability features:**
  - D2 raises rolloff std from 0.06 to 0.16 and centroid std from 0.05 to 0.08.
  - D2L leaves them at C's values.
- **Offline D1** brings every feature to about 0.

### 3. Session location + scale, offline, 6 sessions with ≥ 20 events (242 events)
rms_z session ε² and range of session medians:

| Strategy | ε² | Session medians range (max/min) |
|---|---|---|
| A | 0.25 | 2.16× |
| C | 0.16 | 1.81× |
| D1 | 0.07 | 1.68× |
| D1L | 0.05 | 1.72× |
| D3 | −0.01 | 1.19× |

### 4. Is session-scale normalization defensible? (`session_scale_stability.csv`, `session_scale_heterogeneity.csv`)
- **Bootstrap CV of a session's robust scale** (median over features):
  - 0.53–0.60 for sessions of 8–9 events
  - 0.44–0.45 for 11–13 events
  - 0.27–0.29 for 27–29 events
  - 0.16–0.20 for 50–57 events
  - These match normal theory (0.39–0.41, 0.32–0.35, 0.22, 0.15–0.17) and are somewhat worse for small sessions.
  - Up to 5% of bootstrap draws gave a zero MAD in 9-event sessions.
- **Heterogeneity test.** Residuals are taken from the session median, the statistic is the SD across 12 sessions of log session-MAD, and the permutation keeps session sizes. Per-feature p-values:

  | Feature | p |
  |---|---|
  | duration | 0.19 |
  | mean_rms | 0.055 |
  | centroid mean | 0.45 |
  | centroid std | 0.28 |
  | flatness mean | 0.18 |
  | flatness std | 0.11 |
  | rolloff std | 0.079 |
  | **All features jointly** | **0.013** |

  Sessions do differ in spread overall, but no single feature's session scale is distinguishable from sampling noise at these sizes.

### 5. Baseline parameter variability across LOSO folds (`parameter_variability.csv`)
- **C (18 folds):**
  - Feature centers move by at most 0.07–0.39 pooled SD across folds (duration 0.22, mean_rms 0.35, centroid mean 0.39, flatness mean 0.28, flatness std 0.26, centroid std 0.12, rolloff std 0.07).
  - Scales stay within 0.84–1.20× the pooled scale, with CV 2–7%.
  - The global baseline is stable to which session is left out.
- **D1 (12 folds):** residual scales are 0.57–0.70× pooled (mean_rms), 0.66–0.72× (centroid mean), 0.72–0.79× (flatness mean) and 0.72–0.88× (duration). Removing session location removes about 12–43% of these features' spread. The variability features stay at 0.83–1.15×.

### 6. Feature contributions under C (`feature_contributions.csv`, sessions ≥ 5)
- **Mean RMS shares** are 0.12–0.20 per feature, roughly balanced, against A's single-feature dominance (flatness mean 0.27).
- **max_abs_z argmax:** duration 0.25, mean_rms 0.19, the others 0.09–0.14.

## Interpretation
1. **The first-20 control's ~2× inflation is a calibration-design artefact.** A baseline fitted on many sessions (C) scores unseen sessions at about the level of its fitting data: generalization ratio 1.10–1.15 against 1.9–2.0 for A. Per-feature held-out z spread is also close to 1 (0.93–1.19).
2. **A global baseline does not remove session dependence.**
   - Under C, session membership still explains ε² = 0.21 of the held-out rms_z, and session medians span 0.68–1.48 (2.2×).
   - The dependence sits in the **level** features (duration, mean_rms, centroid mean, flatness mean: ε² 0.39–0.64) and in `spectral_flatness_std` (0.23). `spectral_flatness_std` is labelled a variability feature but behaves like a level feature here, a finding recorded rather than acted on.
   - `spectral_centroid_std` and `spectral_rolloff_std` are session-stable (0.05).
3. **Location offsets are the main session effect.**
   - Removing session location offline (D1/D1L) eliminates per-feature session effects and halves the combined-score session dependence (0.21 → 0.10), with a generalization ratio of about 1.0.
   - The remaining dependence comes from sessions differing in spread. Only per-session scaling (D3) removes it, which is offline and partly true by construction.
4. **The deployable form of location normalization does not help at the pre-specified k = 5.** D2/D2L on the same events show no material improvement over C.
   - **k = 3:** clearly worse (ε² 0.47–0.49).
   - **k = 10 (sensitivity):** D2L reaches ε² 0.02 against C's 0.11 on the same events.
   - **Approximate explanation (normal theory):** the error of a k-event median is ≈ 1.25·σ_w/√k. Here σ_w is the within-session spread, about 0.57–0.88 of the global scale for level features (D1 residual scales). The error is therefore ≈ 0.32–0.49 global SD at k = 5, 0.23–0.35 at k = 10 and 0.41–0.64 at k = 3. That is comparable to the between-session offsets it is meant to remove (robust SD of session-median z under C, level features 0.37–0.65).
   - **Within-session drift:** it also limits the gain. Early events are not representative of later ones (Entry 5).
5. **Session-scale normalization is not defensible as a general method at these session sizes.**
   - Joint spread differences exist (p = 0.013), but no single feature's is significant.
   - Scale estimates are unreliable for the smaller sessions (bootstrap CV 0.44–0.60 for ≤ 13 events).
   - Only 6 sessions have ≥ 20 events.

## Answers to the Stage 5 Questions
1. **Does a pooled global robust baseline generalize?** The pooled baseline (B) is in-sample and cannot show generalization. Its leave-one-session-out counterpart (C) does generalize in **scale**, but not in session-independence. Unseen sessions score about 1.1× the fitting data, against about 1.9× for first-20. Session membership still explains about 21% of score variation.
2. **Does LOSO global normalization reduce session-dependent inflation?** Yes, materially.
   - The about 2× calibration inflation disappears (ratio 1.94 → 1.11).
   - Session dependence falls from ε² 0.28 to 0.21.
   - Per-feature z spread returns to about 1.
   - Level-feature session offsets remain.
3. **Does simple session-location normalization reduce it further?**
   - **Offline:** yes. Every feature's session effect goes to about 0, and combined ε² falls to 0.10.
   - **Deployable, with the pre-specified 5-event warm-up:** no material improvement over C.
   - **With 10 warm-up events** (sensitivity only, not confirmatory): it does improve.
4. **Is session-scale normalization defensible?** Not at these session sizes. See Interpretation 5.
5. **Candidate MVP baseline:** a **global robust baseline fitted on multiple sessions** (strategy C: per-feature median / 1.4826·MAD from a multi-session reference set, frozen), replacing the first-20 single-sitting calibration.
   - **Evidence:** it is the only deployable strategy whose held-out scores are close to its fitting distribution, with parameters stable across folds.
   - **Limitation:** it retains level-feature session dependence.
   - **Decision owner:** replacing the locked first-20 primary design requires the project owner's approval.
   - **Location normalization:** remains the most promising extension, but its deployable form needs a longer same-context reference period. That is not established here.
6. **What next, before threshold selection?** See Decision.

## Limitations
- **Few, inferred sessions:** 18 sessions, 6 with fewer than 5 events. Sessions are recording sittings; whether session effects reflect subject, device, placement or protocol is unknown. That is decisive for how normalization would be done in deployment, where a user may record one or two inhalations per sitting.
- **LOSO ignores time order:** training sessions can come after the held-out session.
- **Offline strategies** (D1, D1L, D3) use later events of the same session and are not deployable.
- **Warm-up sensitivity is not confirmatory:** the k = 10 result is sensitivity only, from this same dataset.
- **Post-hoc variants:** the level-only variants and the joint heterogeneity test were added after seeing the all-feature D1/D2 results (disclosed above).
- **ε² sensitivity:** ε² and session-median ranges rely on sessions with 8–57 events, so small sessions weigh as much as large ones.
- **No held-out data for selection:** all 318 events were already used in Stage 2 feature selection.
- **No ground truth:** nothing here measures detection of real deviations.

## Decision and Next Step
1. **Stop the first-20 path.** Do not proceed to threshold selection on the first-20 baseline. Recommend that the owner adopt the multi-session global baseline (C) as the candidate MVP baseline; this is the owner's decision.
2. **Next experiment (Stage 6), before any threshold: controlled deviations under the LOSO global baseline.**
   - Construct documented, known-magnitude deviations of held-out events, such as feature-level shifts in units of the global scale applied to one feature family at a time.
   - Measure how the three scores respond, relative to the session-dependence noise floor measured here (unperturbed held-out session medians spanning 2.2×).
   - This tells whether a threshold could separate deviations of a given size from session effects. The ground truth comes from the construction, not from clinical labels.
3. **In parallel, request longitudinal data** with user/device/session identifiers. That is the only way to test location normalization in a deployable, personalized form (V3), with warm-up sizes set a priori.
4. **No new features.** No new features are added at this point. `spectral_flatness_std`'s session sensitivity is recorded for the next feature review.

---

# Research Entry 8 — 2026-09-30: Natural Population Separation and Controlled Sensitivity (Stage 6)

**No anomaly threshold or NORMAL/ANOMALY classification was introduced in Stage 6.** Excluded events are NOT treated as anomalous, and no clinical or technique-quality claim is made. This is a natural atypical / out-of-distribution population experiment.

## Question
Does the frozen leave-one-session-out (LOSO) global baseline (Stage 5, strategy C) assign systematically different scores to naturally occurring atypical events than to usable events? Specifically:
- Is any difference robust to held-out-session evaluation and within-session comparison?
- Which features drive it?
- Does the full waveform → measurement → score pipeline respond predictably to controlled acoustic changes?

## Populations
All definitions are taken from the Stage 1 flags (`inhale_events_v1.csv`) and never recomputed; the code checks that the flags agree with Stage 1's `usable` and `exclusion_reasons`.

**Events** (364 detected):

| Group | Events |
|---|---|
| Usable | 318 |
| Excluded | 46 |
| Any too-short (< 0.5 s) | 32 |
| Any close-neighbour (another candidate within 0.2 s) | 25 |
| Any recording-boundary | 1 |
| Non-finite features | 0 |

Categories overlap, so counts are not additive. Exact combinations:

| Combination | Events |
|---|---|
| too_short only | 20 |
| close_neighbor only | 14 |
| too_short + close_neighbor | 11 |
| too_short + recording_boundary | 1 |

**Recordings** (361):

| Group | Recordings |
|---|---|
| With ≥ 1 usable event | 309 |
| Detected events but none usable | 13 |
| No detected inhalation event | 39 |

**Sessions:** events occur in 18 sessions. The 46 excluded events are in 11 sessions, every one of which also has usable events.

**Annotations** are used only as a descriptive cross-check. They are event-boundary labels, not labels of anomaly or technique.

## Held-out Protocol and Checks
- **Baseline:** for each of the 18 sessions, fit a per-feature median / 1.4826·MAD on the **usable events of all other sessions only**, as in Stage 5 C.
- **Scoring:** that baseline is frozen and scores every event of the held-out session, usable and excluded alike. Scores: mean_abs_z, rms_z and max_abs_z over the 7 V1 features (Entry 6). Nothing is tuned on the excluded population, and features and score definitions are unchanged.
- **Checks** (`analysis_summary.json`):
  - excluded events in any fit: 0
  - held-out-session events in their own fit: 0
  - usable scores equal the Stage 5 C scores within 1.5×10⁻⁵, the 6-significant-digit precision of the Stage 5 CSV
  - the existing CNN, re-run on all 361 recordings, reproduces Stage 1's event count for every recording
  - input file hashes are unchanged

Implementation: `src/natural_population_analysis.py`, command `python src/natural_population_analysis.py`. Tests: `tests/test_natural_population_analysis.py` (20 tests; full suite 126 passing). Outputs in `results/natural_population/` (and `controlled/`).

## Results

### 1. Usable vs excluded (held-out; `population_summary.csv`, `effect_sizes.csv`)
Median [IQR] (p05–p95):

| Score | Usable (318) | Excluded (46) |
|---|---|---|
| mean_abs_z | 0.88 [0.64–1.15] (0.41–1.50) | 1.59 [1.33–2.09] (0.90–2.87) |
| rms_z | 1.09 [0.77–1.40] (0.51–1.92) | 2.12 [1.81–2.47] (1.16–3.17) |
| max_abs_z | 1.85 [1.41–2.53] (0.88–3.71) | 4.50 [3.74–4.82] (2.20–5.88) |

rms_z comparison:
- Median ratio 1.95; Hodges–Lehmann shift 1.04; Cliff's δ 0.84, with a 95% interval of 0.69–0.93 from resampling whole sessions. AUC 0.92.
- Overlap: 93% of excluded events lie above the usable median, 65% above the usable 95th percentile, and 35% inside the usable 5th–95th range.

Cliff's δ for the other scores: mean_abs_z 0.78 [0.61, 0.91]; max_abs_z 0.85 [0.73, 0.94].

### 2. Session-controlled comparison (`session_controlled_comparison.csv`)
Across the 11 sessions with both populations:
- **Stratified within-session AUC:** 0.95 (rms_z), 0.93 (mean_abs_z), 0.95 (max_abs_z).
- **Significance:** permutation within sessions gives p = 0.0005, the minimum possible with 2,000 permutations.
- **Direction:** the excluded median is higher in 10 of 11 sessions for rms_z (11 of 11 for max_abs_z), with within-session ratios of 1.31–3.88.
- **Exception:** 2018-05-02#1, where the only 2 excluded events are close-neighbour-only (ratio 0.93).
- **Position within own session:** the median within-session percentile of excluded events among their session's usable events is 1.0, and 78% lie above that session's 90th percentile.

### 3. By exclusion reason (rms_z)

| Group | n | Median | Cliff's δ [session CI] | Stratified AUC (sessions) | Share above own-session p90 |
|---|---|---|---|---|---|
| Any too-short | 32 | 2.21 | 0.95 [0.94, 0.98] | 0.98 (10) | 94% |
| Too-short only | 20 | 2.22 | 0.95 [0.93, 0.98] | 0.99 (9) | 95% |
| Too-short + close-neighbour | 11 | 2.19 | 0.95 [0.92, 0.99] | 0.97 (4) | 91% |
| Any close-neighbour | 25 | 1.84 | 0.74 [0.46, 0.89] | 0.88 (6) | 64% |
| Close-neighbour only | 14 | 1.63 | 0.57 [0.29, 0.82] | 0.77 (6); p = 0.0035; 5 of 6 sessions higher | 43% |
| Recording-boundary | 1 | 2.46 | — (descriptive only) | — | — |

### 4. Feature attribution (`feature_attribution.csv`, `duration_matched_comparison.csv`)
- **All excluded events:**
  - Duration dominates: median z −4.44, largest deviation in 85% of events, 54% of rms_z, univariate |z| AUC 0.95.
  - rms_z recomputed **without duration** still separates, but less: AUC 0.92 → 0.71.
  - Other features' univariate AUCs: centroid std 0.70, rolloff std 0.68, mean_rms 0.66, flatness std 0.65, flatness mean 0.55, centroid mean 0.49. All deviate downward: quieter, less bright, less within-event variability.
- **Too-short:** duration is the largest deviation in 91% of events (59% of rms_z). Without duration the AUC is 0.74, carried mainly by lower `mean_rms` (median z −0.99, AUC 0.69) and the `*_std` features.
- **Close-neighbour only:** these events are shorter than typical usable events (duration median z −2.56; largest deviation in 71%; 42% of rms_z). Without duration the AUC is 0.65, carried by lower centroid std (AUC 0.75), flatness std (0.68) and rolloff std (0.65).
- **Duration-matched check.** The method was fixed before this check was run: each excluded event is paired with its k = 5 nearest-duration usable events, and both are compared on rms_z over the 6 non-duration features.
  - **Close-neighbour only (14):** median duration gap 0.03 s; median paired difference −0.03; only 43% of events exceed their matches; Cliff's δ vs matched 0.23 (vs all usable 0.29).
  - **Any close-neighbour (25):** gap 0.06 s; paired −0.12; 32% positive; δ 0.33.
  - **Too-short:** matching is **not achievable**, since no usable event is shorter than 0.5 s by definition (median gap 0.29 s). Their non-duration deviation is smaller than that of the shortest usable events (δ −0.24 vs matched; 0.47 vs all usable).
  - **Reading:** the residual, non-duration deviation of excluded events is largely shared by usable events of similar duration. It is consistent with a segment-length effect, not a distinct acoustic character.

### 5. Separation vs ordinary session variation (`session_variation.csv`)
- **Ordinary session shifts:** the 12 usable sessions with ≥ 5 events, each compared with all other usable events, give rms_z Cliff's δ between −0.69 and +0.50 (median ratios 0.63–1.38).
- **Excluded groups against that range:**
  - all excluded (δ 0.84) and too-short (0.95) exceed the largest session shift (|δ| 0.69)
  - close-neighbour only (0.57) lies within the range of ordinary session shifts
- The within-session comparisons in Result 2 remove session composition.

### 6. Annotation cross-check (descriptive; `annotation_crosscheck.csv`, `population_summary.csv`)
- **Excluded events:** none of the 46 matches an annotated inhalation. 23 are in unannotated recordings, 16 in annotated recordings without an Inhale label, and 7 outside Inhale annotations.
- **Usable events' rms_z medians by annotation status:** 1.07 matched (257), 1.33 in recordings without an Inhale annotation (30), 1.15 in unannotated recordings (29), 1.40 outside annotations (2).
- Annotation absence is not treated as a label.

### 7. Recordings without a usable event (`recording_level_summary.csv`, `recording_window_stats.csv`)
They have no event to score, so no anomaly score is manufactured. Instead they are described with window-level quantities of the existing CNN and the existing RMS envelope.

- **No detected event (39) vs recordings with usable events (309):**
  - mean P(Inhale) median 0.0003 vs 0.12 (δ −1.0)
  - max P(Inhale) 0.004 vs 0.997 (δ −1.0)
  - share of Noise windows 1.00 vs 0.62 (δ +0.96)
  - share of Exhale windows 0 vs 0.22 (δ −0.91)
  - recording mean RMS 0.0032 vs 0.039 (δ −0.85)
  - 95th-percentile envelope RMS 0.0094 vs 0.23 (δ −0.91)
  - 28 of the 39 have every window classified Noise, and 29 have recording mean RMS below 0.01. Only 6 reach the usable recordings' 5th-percentile loudness (0.023). They span 14 sessions.
- **Detected events but none usable (13):**
  - loudness is like usable recordings (mean RMS δ −0.06; p95 RMS δ +0.15)
  - fewer Inhale windows (0.047 vs 0.12; δ −0.64) and almost no Exhale windows (0.004 vs 0.22; δ −0.92)
  - more, shorter inhale candidates (median 2; δ +0.69)
  - lower max P(Inhale) (0.86 vs 0.997)
- **Annotation cross-check** (with Inhale annotation / annotated without Inhale / unannotated; `recording_annotation_crosscheck.csv`):
  - no-event recordings: 0 / 17 / 22
  - only-excluded recordings: 0 / 3 / 10
  - usable-event recordings: 255 / 26 / 28

  No recording without a detected event carries an Inhale annotation, so there is no annotated evidence of missed inhalations. The annotated no-event recordings are louder (median mean RMS 0.0091) than the unannotated ones (0.0012).

### 8. Controlled waveform perturbations (secondary; `controlled/`)
**Method:**
- All 318 usable events. Event boundaries are fixed (duration unchanged), and each event's own frozen LOSO baseline is used.
- The existing measurement code is re-run on the perturbed segment: `post_event.analyze_inhalation`, the unchanged 124-feature extractor.
- Magnitudes were fixed before any Stage 6 result:
  - gain ×0.5, ×0.71, ×1.41, ×2 (−6 / −3 / +3 / +6 dB)
  - white Gaussian noise at 30, 20 and 10 dB SNR relative to the event's own power (per-event deterministic seed)
  - RMS-preserving first-order tilt y[n] = x[n] − a·x[n−1] with a = −0.5, +0.5, +0.9
- The unperturbed re-measurement reproduces stored features within 1×10⁻¹⁶. No clipping is modelled.

**Results** (median Δz vs the unperturbed event; monotonicity = share of events whose z changes monotonically with intensity):

| Transform | Main response | Other responses | Distance from own z monotone | Median Δrms_z | Events with higher rms_z |
|---|---|---|---|---|---|
| Gain | `mean_rms` only: −3.07, −1.80, +2.55, +6.15; monotone in 100% | < 0.001 | 100% | +0.55, +0.22, +0.40, +1.59 | 96%, 82%, 83%, 97% |
| Noise (30 / 20 / 10 dB) | flatness mean +1.88, +3.31, +6.18 and centroid mean +0.55, +0.92, +1.73 (both 100% monotone); mean_rms +0.02, +0.10, +0.59 | flatness std +4.80, +5.68, +4.97 (not monotone, 17%); rolloff std −0.18, −0.32, −1.15 (29%) | 97% | +1.27, +1.84, +2.28 | 95–99% |
| Tilt (a = −0.5 / +0.5 / +0.9) | centroid mean −1.63, +2.49, +3.90 (100% monotone); mean_rms ≈ 0 | flatness mean −1.77, +1.23, +0.17 (not monotone, 1%) | 100% | +0.38, +0.59, +0.89 | — |

**Cross-session consistency** (12 sessions with ≥ 5 events; `session_consistency.csv`):
- The direction of every main response is the same in all sessions.
- Magnitudes scale with each session's level:
  - `mean_rms` Δz at ×2 gain: 5.50–8.42
  - flatness mean at 10 dB SNR: 5.46–7.09
  - centroid mean at a = 0.9: 3.51–4.26

**Notable:** `spectral_flatness_std` shifts by about 5 z already at 30 dB SNR and responds non-monotonically. It is very sensitive to low-level broadband noise, a possible reason for its session sensitivity (Entry 7; hypothesis, not tested).

## Interpretation
1. **The frozen baseline separates the naturally excluded events from usable events.** The separation holds under held-out-session evaluation and within sessions (10 of 11 sessions; stratified AUC 0.95), and it exceeds ordinary session-to-session variation.
2. **The separation is predominantly a duration effect.** For too-short events that is the Stage 1 exclusion criterion itself. For close-neighbour events it reflects fragmented, shorter segments.
3. **The remaining multi-feature deviation is small and duration-linked.** It is lower `mean_rms` and lower within-event spectral variability. Once close-neighbour events are duration-matched, it largely disappears (δ 0.23; 43% of paired differences positive). For too-short events it cannot be separated from segment length with these data.
4. **Segmentation, not acoustic character.** The evidence therefore points to segmentation/recording artifacts detected through duration and segment-length-dependent measurements, not to acoustically distinct inhalations.
5. **No-event recordings are acoustically distinct in the most basic sense:** most are near-silent and entirely Noise-classified. The event-level architecture cannot validly represent them, since there is no inhalation event to compare against the baseline.
6. **The pipeline itself responds predictably** to controlled level, noise and spectral-shape changes, with the expected feature moving monotonically and consistently across sessions. Two features are fragile: flatness std under noise, and flatness mean under tilt, both non-monotone.

## Answers to the Stage 6 Questions
1. **Do excluded events score higher?** Yes, systematically (rms_z δ 0.84; median 2.12 vs 1.09).
2. **Does it survive LOSO evaluation?** Yes. Every score is held-out, and no excluded or same-session event enters any fit.
3. **Does it survive within-session comparison?** Yes: stratified AUC 0.95, higher in 10 of 11 sessions. The exception is the session whose excluded events are close-neighbour only.
4. **Strongest reasons:** too-short (δ 0.95) and too-short + close-neighbour (0.95). Close-neighbour only is weaker (0.57) and within the range of ordinary session shifts.
5. **One feature or many?** Predominantly one: duration (54–59% of rms_z; largest deviation in 85–91% of events). Secondary deviations in `mean_rms` and the `*_std` features are consistent with segment length and weaken strongly under duration matching.
6. **Acoustically distinct, or artifacts?** The evidence points to segmentation/recording artifacts (short or fragmented detections), not to acoustically distinct inhalation events.
7. **No-event recordings:** they are acoustically distinct, mostly near-silent recordings with no Inhale windows. The event-level anomaly architecture cannot validly score them. A recording-level check (e.g. "no inhalation detected") would be a separate, non-anomaly output.
8. **Does the controlled experiment support the sensitivity story?** Yes, for level, noise and spectral tilt. Targeted features move monotonically with magnitude, the direction is consistent across sessions, and scores rise for most events at larger magnitudes. Caveats:
   - A ±3 dB level change moves median rms_z by only 0.2–0.4, less than ordinary session variation.
   - `spectral_flatness_std` responds non-monotonically and strongly to low-level noise.
9. **Is there enough evidence for threshold analysis?** Not yet for a validated decision rule:
   - the only strong natural separation is a duration/segmentation effect;
   - session variation remains of similar size to modest acoustic changes (Entry 7);
   - there is no independent ground truth for inhalation-level deviations.

## Limitations
- **Small groups:** too-short 32, close-neighbour only 14, boundary 1. Close-neighbour events occur in only 6 sessions.
- **Too-short events cannot be duration-matched,** because the usability rule is itself a duration cut.
- **Excluded events are not validated anomalies,** and annotation absence is not a label.
- **The controlled perturbations are synthetic, event-level only,** and keep detector boundaries fixed. They do not test how perturbations would change detection itself.
- **Sessions are inferred sittings,** not users (Entries 4–7).
- **No held-out data:** all 318 usable events were used in Stage 2 feature selection.

## Decision and Recommendation
**Recommendation: C — obtain better longitudinal / ground-truth data first**, with two targeted representation checks (B) that don't block it.
- **Why not A (thresholds now):** the only strong natural separation is duration/segmentation. Modest acoustic changes are comparable to ordinary session variation. Without user grouping and independent deviation labels, a threshold would mostly encode session membership and segment length.
- **What data would resolve it:** longitudinal recordings with user/device/session identifiers, and ideally a protocol with documented, deliberate acoustic variations (e.g. instructed faster or softer inhalations, recorded as protocol conditions, not clinical labels). These would allow personal-baseline evaluation (V3) and an independent test of deviation detection.
- **B items to address alongside:**
  1. `spectral_flatness_std`'s noise sensitivity and non-monotone response.
  2. Duration dependence of the within-event variability features for short segments.
- **Fallback if no new data can be obtained:** an explicitly exploratory threshold-methodology analysis under the LOSO baseline, using the Stage 6 controlled perturbations as the only constructed ground truth, with per-session exceedance reported.

---

# Research Entry 9 — 2026-09-30: Representation Robustness and Ablation (Stage 7)

**No anomaly threshold or NORMAL/ANOMALY classification was introduced in Stage 7.** Excluded status (Stage 1) is a diagnostic population only, not anomaly ground truth, and no clinical or technique-quality claim is made.

## Question
Are the current scores detecting acoustically meaningful deviations, or are they dominated by:
- event duration
- recording level
- session effects
- fragile features?

## Protocol
**Baseline:** as in Stages 5–6 (strategy C). For each of the 18 sessions, a per-feature median / 1.4826·MAD is fitted on the **usable events of all other sessions only** and frozen. It then scores every event of the held-out session. rms_z is the working score.
- **Univariate fit:** the baseline is univariate, so one union fit per fold gives every representation's z as a column subset.
- **Leakage checks:** 0 excluded events in any fit, and 0 held-out-session events in their own fit.
- **Input integrity:** inputs are unchanged by hash, and Stages 1–6 code, tests and results were not modified.

**Pre-specified candidates and representations** (constants in `src/representation_analysis.py`, fixed before any Stage 7 result):
- **Energy-weighted (ew_) spectral statistics.** The extractor's per-frame centroid, flatness and rolloff, averaged with weights proportional to frame energy (librosa RMS² on the same framing), within the event only.
  - Rationale: low-energy frames are dominated by the noise floor, which could explain the Stage 6 noise fragility and the level–brightness coupling.
  - Check: the unweighted versions of the same frames reproduce the stored features within 6.3×10⁻⁸.
- **relative_level_db** = 20·log10(event mean_rms / median RMS envelope of the whole recording). It cancels device gain and distance, and is computable at inference without session information.

The representations tested:

| Id | Definition |
|---|---|
| R0 | current 7 features |
| R1 | no duration |
| R2 | duration used only as the Stage 1 usability gate (< 0.5 s not scoreable; score = R1) |
| R3 | no spectral_flatness_std |
| R4 | spectral_flatness_std replaced by its ew version |
| R5 | no mean_rms |
| R6 | mean_rms replaced by relative_level_db |
| R7 | all 5 spectral features replaced by ew versions |
| R8 | relative_level_db + 5 ew spectral features (no duration) |

**Controlled perturbations:**
- The Stage 6 plan (event-level gain, white noise, RMS-preserving tilt) with the same seeds. It reproduces the Stage 6 re-measured V1 features within 5×10⁻⁶ relative, the 6-digit precision of the Stage 6 CSV.
- Plus **whole-recording gain ×0.5 and ×2**, modelling device gain or distance.
- Monotonicity treats changes below 10⁻⁴ z as zero. The measured float32 jitter of exactly gain-invariant features is ≤ 1×10⁻⁵ z.
- Stage 6's gain-monotonicity fractions for the spectral features (0.61–0.95) were computed with zero tolerance. For exactly invariant features they reflected this jitter, not a response (Entry 8 is not edited).

**Multivariate gate** (fixed in advance; model diagnostics, not anomaly thresholds). Robust MCD Mahalanobis distance in robust-z space, fitted per LOSO fold on usable events, compared only if every fold has:
- ≥ 10 training events per feature
- condition number ≤ 100
- max |correlation − pooled correlation| ≤ 0.2

**Reproducibility:** a complete second run into a separate directory reproduced all 14 CSV outputs byte for byte (seed 20261004 for bootstraps and MCD; Stage 6 noise seed 20261003).
- **Reference environment:** the project `venv` (Python 3.13.5, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, pandas 3.0.3, librosa 0.11.0). Run the analysis with `venv/Scripts/python.exe`.
- **Cross-environment check:** a run under a different environment (Anaconda: numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, pandas 2.2.3) reproduced 13 of 14 CSVs byte for byte. `multivariate_diagnostics.csv` differed (see Results §4).

Implementation: `src/representation_analysis.py`, command `python src/representation_analysis.py`. Tests: `tests/test_representation_analysis.py` (15 tests; full suite 141 passing). Outputs in `results/representation_analysis/`.

## Results

### 1. Representation comparison (`representation_comparison.csv`; usable held-out rms_z; δ = Cliff's delta vs usable, 95% session-bootstrap interval)

| Rep. | d | Usable median | Gen. ratio | Session ε² | Max session-vs-rest abs δ | All excluded δ | Too-short δ | Close-only δ | Close-only strat. AUC | Δrms_z noise 30/20/10 dB | Δrms_z rec. gain ×2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 7 | 1.087 | 1.10 | 0.211 | 0.690 | 0.836 [0.71, 0.93] | 0.952 | 0.570 [0.30, 0.80] | 0.77 | 1.27 / 1.84 / 2.28 | 1.59 |
| R1 | 6 | 1.019 | 1.11 | 0.283 | 0.611 | 0.416 [0.12, 0.70] | 0.470 | 0.292 [−0.04, 0.61] | 0.72 | 1.43 / 2.05 / 2.55 | 1.82 |
| R2 | 6 | 1.019 | 1.11 | 0.283 | 0.611 | 0.292 (n = 14 scoreable) | not scoreable | 0.292 [−0.04, 0.61] | 0.72 | as R1 | 1.82 |
| R3 | 6 | 1.073 | 1.13 | 0.199 | 0.637 | 0.849 [0.73, 0.94] | 0.961 | 0.594 [0.33, 0.83] | 0.77 | 0.35 / 0.81 / 1.82 | 1.78 |
| R4 | 7 | 1.077 | 1.11 | 0.206 | 0.669 | 0.817 [0.68, 0.92] | 0.939 | 0.538 [0.22, 0.80] | 0.80 | 0.31 / 0.79 / 2.16 | 1.59 |
| R5 | 6 | 1.055 | 1.09 | 0.136 | 0.625 | 0.861 [0.76, 0.94] | 0.961 | 0.633 [0.40, 0.84] | 0.80 | 1.47 / 2.05 / 2.51 | 0.00 |
| R6 | 7 | 1.044 | 1.08 | 0.123 | 0.577 | 0.870 [0.77, 0.95] | 0.960 | 0.665 [0.39, 0.90] | 0.85 | 1.31 / 1.87 / 2.28 | 0.00 |
| R7 | 7 | 1.029 | 1.11 | 0.325 | 0.819 | 0.737 [0.55, 0.88] | 0.890 | 0.386 [0.005, 0.75] | 0.86 | 0.02 / 0.22 / 1.46 | 1.61 |
| R8 | 6 | 0.972 | 1.14 | 0.301 | 0.766 | 0.345 [0.005, 0.63] | 0.495 | 0.002 [−0.53, 0.55] | 0.68 | 0.02 / 0.28 / 1.66 | 0.00 |

Generalization ratios (held-out median / fold in-sample median) are 1.075–1.136 in every representation.

### 2. Per-feature robustness (`feature_robustness.csv`, `feature_perturbation.csv`)
Columns: session ε² of held-out z; monotone fraction under noise; median Δz at 30 dB; max |Δz| under whole-recording gain; monotone fraction under tilt; within-session Spearman with mean_rms.

| Feature | Session ε² | Noise monotone | Δz at 30 dB | Max abs Δz, rec. gain | Tilt monotone | Within-session ρ with mean_rms |
|---|---|---|---|---|---|---|
| duration_s | 0.454 | 1.00 | 0 | 0 | 1.00 | −0.14 |
| mean_rms | 0.637 | 0.997 | 0.02 | 17.2 | 0.70 | 1 |
| spectral_centroid_mean | 0.620 | 1.00 | 0.55 | 0 | 1.00 | −0.76 |
| spectral_centroid_std | 0.053 | 0.20 | 0.41 | 0 | 0.43 | −0.08 |
| spectral_flatness_mean | 0.393 | 1.00 | 1.88 | 0.06 | 0.01 | −0.46 |
| spectral_flatness_std | 0.233 | 0.17 | 4.80 | 0.67 | 0.03 | −0.35 |
| spectral_rolloff_std | 0.053 | 0.29 | −0.18 | 0 | 0.49 | −0.14 |
| ew_spectral_flatness_std | 0.290 | 0.87 | 0.21 | 0.004 | 0.03 | −0.29 |
| relative_level_db | 0.241 | 1.00 | 0.00 | 0 | 0.76 | +0.04 |
| ew_spectral_centroid_mean | 0.591 | 1.00 | 0.03 | 0 | 1.00 | −0.84 |
| ew_spectral_centroid_std | 0.104 | 0.62 | 0.00 | 0 | 0.37 | −0.11 |
| ew_spectral_flatness_mean | 0.389 | 1.00 | 0.26 | 0.001 | 0.02 | −0.51 |
| ew_spectral_rolloff_std | 0.461 | 0.59 | −0.05 | 0 | 0.59 | +0.27 |

In this table, 0 means < 10⁻⁵ z (float32 jitter of an exactly invariant feature).

Two further measurements:
- **Level dependence of the flatness features.** Whole-recording gain changes `spectral_flatness_std` by up to 0.67 z and `spectral_flatness_mean` by up to 0.06 z. Both should be exactly gain-invariant. The likely cause is librosa's spectral-flatness power floor, active in near-silent frames (inference, not isolated experimentally).
- **relative_level_db:** a 6 dB event-level gain moves it by only 0.49 z, so its between-event robust spread is about 12 dB. In usable events it has Spearman ρ −0.99 with the recording's background RMS and −0.08 with the event's own mean_rms. The event level varies by 2.6 dB² in dB, against 136.6 dB² for the background (computed from `candidate_features.csv`). It is effectively a background-level measure.

### 3. Perturbation response of representations (`perturbation_response.csv`)
- **Distance from the unperturbed z grows with intensity** in 97–100% of events for noise (0.969 with flatness_std in the representation, 0.997 without, 1.0 for R7/R8), 99.4–100% for gain, and 100% for tilt and whole-recording gain.
- **Session direction:** median Δrms_z is positive in 92–100% of the 12 sessions with ≥ 5 events at every perturbation for R0.
- **Single-feature dominance in R0:**
  - The gain response is entirely `mean_rms` (median dominance 1.0).
  - The 30 dB and 20 dB noise responses are dominated by `spectral_flatness_std` (median dominance 0.783 and 0.685).
  - The tilt response is mostly `spectral_centroid_mean` (0.79 at a = 0.9).

### 4. Multivariate gate (`multivariate_diagnostics.csv`)
The gate **failed for every representation**:
- **Conditioning was acceptable:** max condition number 13.8–98.7, and ≥ 37 training events per feature.
- **Correlation structure was unstable:** max correlation shift vs pooled was 0.21–0.50 (median 0.07–0.12). The worst fold is always the one leaving out 2018-05-03#2 (57 events).

The correlation structure itself depends on which sessions are included, so no multivariate comparison was run. `multivariate_comparison.csv` is empty by design.

Two further points:
- **R1/R2 fail by a small margin:** 0.209 against the 0.2 limit.
- **The MCD solutions are environment-dependent.** With identical robust-z inputs (`baseline_parameters.csv` byte-identical), the Anaconda environment gave:
  - MCD support sizes that differed by up to 18 events per fold
  - max correlation shifts of 0.220–0.510 (R8: 0.221 vs 0.306 here)
  - an R4 condition number of 101.2

  The gate failed for every representation in both environments. Separately, perturbing the inputs by 1 ulp left all 162 fits unchanged. The MCD fit is therefore numerically deterministic, but its solution is not unique enough to be independent of the implementation, which is further evidence against multivariate scoring on these data.

## Interpretation
1. **Duration carries about half of the Stage 6 separation.** Removing it lowers the excluded-vs-usable δ from 0.836 to 0.416 (50% retained), too-short from 0.952 to 0.470, and close-neighbour-only from 0.570 to 0.292.
   - **Within sessions** 72% of the above-chance separation remains (stratified AUC 0.946 → 0.823).
   - **The residual is small against ordinary variation:** without duration, excluded vs usable (0.416) is smaller than the largest ordinary session-vs-rest shift in the same representation (0.611). For the only duration-eligible excluded group (close-neighbour only, R2) the interval includes 0.
   - **Link to segment length:** Stage 6's duration-matched analysis showed the residual is shared by short usable events.
2. **`spectral_flatness_std` is fragile in three independent ways:**
   - noise: 4.8 z at 30 dB SNR, monotone in only 17% of events
   - level: up to 0.67 z under a pure gain change
   - tilt: monotone in 3% of events

   It dominates the representation's noise response. Removing it cuts the 30 dB response of rms_z by 73% (1.27 → 0.35) and changes neither session dependence (0.211 → 0.199) nor the diagnostic separation (0.836 → 0.849). Its energy-weighted replacement fixes the noise and level fragility (0.21 z; 87% monotone; 0.004 z under gain), but it is still non-monotone under tilt and more session-dependent (0.290 vs 0.233).
3. **The most session-affected features are the level features:** mean_rms (ε² 0.64), centroid mean (0.62), duration (0.45) and flatness mean (0.39). The within-event variability features are least affected (centroid std, rolloff std: 0.05).
4. **Removing absolute level reduces combined session dependence by 36%** (0.211 → 0.136, R5) and makes the score invariant to device gain. Replacing it with recording-relative level (R6) gives 0.123. But relative level is not an event-level measure: it barely correlates with the event's own level (ρ −0.08) and almost entirely tracks each recording's background (ρ −0.99). R6's small gain over R5 therefore reflects background level, not the inhalation.
5. **Energy weighting does not make the representation more session-robust.**
   - It reduces noise sensitivity as intended (R7: 30 dB response 0.02).
   - Session dependence *increases*: R7 0.325; ew rolloff std 0.46 vs 0.05; ew centroid std 0.10 vs 0.05.
   - The level–brightness coupling *strengthens*: −0.84 vs −0.76 within sessions.

   Hypothesis, not tested: the loudness–brightness coupling and the session differences are properties of the energy-bearing sound (geometry, device or subject), not of the noise floor. Part of the unweighted variability features' session stability may come from low-energy, noise-floor frames.
6. **Controlled perturbations produce predictable, session-consistent responses** once the fragile feature is removed. Session dependence remains substantial in every representation (ε² ≥ 0.12), and multivariate modelling is not yet supported, because the correlation structure is session-dependent.

## Answers to the Stage 7 Questions
1. **How much of the Stage 6 separation is caused by duration?** About half of the pooled separation (δ 0.836 → 0.416), and about a quarter within sessions (stratified AUC above chance 0.446 → 0.323).
2. **Does meaningful separation remain without duration?** Some remains (δ 0.42; within-session AUC 0.82), but it is smaller than ordinary session-to-session shifts. For the duration-eligible group it is not distinguishable from zero. That the remainder reflects acoustically meaningful deviation is **not established by the current data**.
3. **Is spectral_flatness_std robust enough to retain?** No.
4. **Which features are most affected by session / recording level?** mean_rms, spectral_centroid_mean, duration_s and spectral_flatness_mean.
5. **Is a more session-robust representation possible without leakage?** Partially. Removing absolute level cuts session dependence by about 36% without leakage. No tested representation removes session dependence, and energy weighting makes it worse.
6. **Do controlled perturbations produce predictable responses?** Yes for gain (mean_rms), noise (flatness mean, centroid mean) and tilt (centroid mean). spectral_flatness_std is the exception, and some variability features respond weakly or non-monotonically.
7. **Is the current 7-feature representation scientifically defensible?** Not as-is: it mixes segmentation (duration), device-gain-dependent level (mean_rms) and a fragile feature (flatness_std) into one anomaly score.
8. **A, B or C?** **B — modify the representation.** C (new longitudinal / ground-truth data) remains the next bottleneck.

## Decision (`recommendation.json`)
- **Remove from the anomaly score:**
  - duration_s, which stays as a segmentation / usability attribute
  - spectral_flatness_std
  - mean_rms, reported separately as an uncalibrated level channel
- **Keep:** spectral_centroid_mean and spectral_flatness_mean (both flagged as session-dependent), spectral_centroid_std and spectral_rolloff_std.
- **Not adopted:** the energy-weighted features, relative_level_db and multivariate scoring (reasons above).
- **V2 candidate, not evaluated:** the resulting 4-feature set is proposed as a V2 candidate. That combination was **not evaluated** in Stage 7 and must be pre-registered and evaluated in the next stage, preferably on new data. It is not adopted into the pipeline here.

## Limitations
- **No ground truth:** excluded events are not anomalies, and there is no anomaly ground truth. "Separation" is diagnostic only.
- **Post-hoc design:** the candidate features were designed after Stage 6 and evaluated on the same 318 usable events used in Stage 2 feature selection.
- **Synthetic perturbations** keep event boundaries fixed and do not model detector responses.
- **Inferred sessions:** sessions are recording sittings, and whether session effects are subject, device or placement effects is unknown.
- **Unconfirmed mechanisms:** the amin-floor explanation for flatness level-dependence, and the energy-bearing-sound explanation for session effects, are hypotheses.

## Next Step
1. Pre-register the V2 candidate (4 features, with duration and level as separate non-scored channels) and its evaluation protocol before any further analysis.
2. Obtain longitudinal data with user/device/session identifiers and, ideally, documented deliberate acoustic variations. Without them, it is **not established by the current data** whether any representation detects meaningful inhalation-level deviations beyond session differences.

---

# Research Entry 10 — 2026-09-30: V2 Representation Validation and Inference Contract (Stage 8)

**No anomaly threshold or NORMAL/ANOMALY classification was introduced in Stage 8.** The inference output is `SCORE_ONLY`: a distance from a frozen baseline. Excluded (Stage 1) status is a descriptive population only, and no clinical or technique-quality claim is made.

## Question
Does the 4-feature V2 representation proposed in Stage 7 behave consistently enough across sessions and controlled acoustic perturbations to be frozen **temporarily** as the PRISM MVP inference representation, while scientific validation continues? The features are:
- `spectral_centroid_mean`
- `spectral_flatness_mean`
- `spectral_centroid_std`
- `spectral_rolloff_std`

If so, what exactly is the inference contract?

## Pre-registration (before any combined-V2 result)
- **Gate committed first.** The acceptance gate was written to `results/v2_validation/acceptance_gate_preregistration.json` and committed as **`55baeba`** before any result for the combined V2 representation, its ablations or its perturbation response was computed.
- **Decision rule:** V2 passes only if every criterion passes. On a pass the contract is issued as `FROZEN_FOR_MVP_ENGINEERING`; on a fail as `DRAFT_NOT_FROZEN`. Neither V2 nor the gate may be changed afterwards in this stage.
- **Code check:** the rule texts in `src/v2_validation.py` (`GATE_RULES`) are tested against the committed file.
- **Disclosure: some values were already known.** Per-feature values of the four V2 features were known from Stages 2, 5 and 7. Criteria built on them are labelled confirmatory re-checks, not independent tests:
  - G2a–G2c: fold centre/scale stability
  - G4a: gain invariance
  - G4d: no fragile feature

  The genuine tests are G1a/G1b (combined score), G3a (correlation stability), G4b/G4c (combined perturbation response, per-session direction), G5a/G5b (background confound) and G6 (reproducibility).
- **Not definable from Stages 1–7, so not gated:**
  - an absolute acceptable level of session dependence
  - any criterion that the score detects meaningful inhalation anomalies
  - transfer to PRISM hardware
- **Explicitly not criteria:** excluded-vs-usable separation or AUC.

| Id | Dimension | Rule | Basis |
|---|---|---|---|
| G1a | cross-session | session ε²(V2) ≤ ε²(R0), same run | comparative (non-inferiority, margin 0) |
| G1b | cross-session | generalization ratio ≤ 1.15 | inherited (Stage 5) |
| G2a | feature scale | max LOSO centre shift ≤ 0.5 pooled robust SD | anchored (Stage 2 random-20 imprecision) |
| G2b | feature scale | fold/pooled scale within [2/3, 3/2] | anchored (Stage 2) |
| G2c | feature scale | every MAD > 0 and finite | sanity |
| G3a | correlation | max LOSO Spearman shift of any V2 pair ≤ 0.2 | inherited (Stage 7 MCD gate), rank-based |
| G4a | perturbation / confound | 95th percentile of \|Δ rms_z\| ≤ 0.1 at every event and whole-recording gain level | convention |
| G4b | perturbation | z-distance monotone in ≥ 95% of events (noise, tilt) | convention |
| G4c | perturbation | ≥ 90% of sessions agree in sign wherever \|median Δ\| ≥ 0.1 (10 dB noise, tilt −0.5 and 0.9; per feature and for rms_z) | convention |
| G4d | perturbation | no feature with \|median Δz\| ≥ 1 at 30 dB and noise-monotone < 0.5 | Stage 7 fragility diagnosis |
| G5a / G5b | recording confound | \|Spearman(rms_z, recording background RMS)\| < 0.3, pooled / within session | convention (Cohen) |
| G6a | reproducibility | byte-identical outputs across two complete runs | engineering |
| G6b | reproducibility | the reference inference implementation reproduces the Stage 1 events, features and usability from raw audio | engineering |
| G6c | reproducibility | exported baseline JSON reproduces in-memory z within 1e-12 | engineering |
| G6d | leakage | 0 held-out-session events and 0 excluded events in any fold fit | protocol |

## Protocol
- **Baseline:** Strategy C (Stage 5). For each of the 18 sessions, per-feature median / 1.4826·MAD is fitted on the usable events of the other sessions only, frozen, and applied to the held-out session. rms_z = sqrt(mean z²).
- **Session statistics:** 12 sessions with ≥ 5 usable events (318 usable events in total).
- **Comparators:** R0_current7, R1_no_duration, R3_no_flatness_std, R5_no_mean_rms and R6_relative_level (Stage 7 definitions), recomputed in the same run. Their session ε² reproduces Stage 7 within 8.3×10⁻¹⁷.
- **Descriptive additions** (not gated): the leave-one-feature-out ablations `V2_minus_*` and `V2_plus_mean_rms` (Part E).
- **Perturbations:** the Stage 6/7 plan with the same seeds, via the unchanged Stage 7 `perturbation_measurements`:
  - event gain ×0.5, ×0.707, ×1.414, ×2
  - white noise at 30/20/10 dB SNR
  - RMS-preserving tilt a = −0.5, 0.5, 0.9
  - whole-recording gain ×0.5 and ×2

  Each perturbed event is scored by its own frozen fold baseline.
- **Frozen deployment baseline:** the same median/MAD fitted on **all 318 usable events of all 18 sessions**. LOSO estimates how this procedure behaves on unseen sessions.

Implementation:
- `src/v2_validation.py` (analysis, gate and contract), command `venv/Scripts/python.exe src/v2_validation.py [--reference-run <dir>]`.
- `src/prism_inference.py` (reference implementation of the contract; `python src/prism_inference.py <wav>`).
- Tests: `tests/test_v2_validation.py` and `tests/test_prism_inference.py` (32 tests; full suite 173 passing).
- Outputs in `results/v2_validation/`.

## Results

### Part A. V2 as a combined representation (`representation_comparison.csv`, `feature_scale_stability.csv`, `z_distributions.csv`, `correlation_structure.csv`, `feature_contributions.csv`)
All values are held-out rms_z on the 318 usable events. The excluded-vs-usable δ is descriptive only (not a criterion; excluded ≠ anomalous).

| Representation | d | Median [5–95%] | Generalization ratio | Session ε² | Session medians (max/min) | Max session-vs-rest \|δ\| | Excluded δ (descriptive) |
|---|---|---|---|---|---|---|---|
| R0_current7 | 7 | 1.087 [0.510, 1.923] | 1.102 | 0.211 | 0.685–1.483 (2.17×) | 0.690 | 0.836 |
| R1_no_duration | 6 | 1.019 [0.467, 1.906] | 1.106 | 0.283 | 0.590–1.395 (2.37×) | 0.611 | 0.416 |
| R3_no_flatness_std | 6 | 1.073 [0.477, 1.955] | 1.130 | 0.199 | 0.692–1.543 (2.23×) | 0.637 | 0.849 |
| R5_no_mean_rms | 6 | 1.055 [0.514, 1.883] | 1.089 | 0.136 | 0.721–1.582 (2.19×) | 0.625 | 0.861 |
| R6_relative_level | 7 | 1.044 [0.545, 1.794] | 1.075 | 0.123 | 0.720–1.465 (2.03×) | 0.577 | 0.870 |
| **V2** | 4 | 0.997 [0.423, 1.745] | 1.120 | 0.130 | 0.641–1.314 (2.05×) | 0.433 | 0.406 |

**Frozen deployment baseline (`v2_baseline.json`, id `prism-v2-global-2026-09-30`).** Median / 1.4826·MAD over all 318 usable events in 18 sessions:

| Feature | Centre | MAD | Scale |
|---|---|---|---|
| centroid_mean | 0.37053 | 0.018796 | 0.027867 |
| flatness_mean | 0.13224 | 0.020037 | 0.029707 |
| centroid_std | 0.039401 | 0.0060552 | 0.0089774 |
| rolloff_std | 0.080710 | 0.011403 | 0.016906 |

**LOSO fold stability.**
- **Max centre shift (pooled robust SD):** centroid_mean 0.221 (fold 2018-05-03#3), flatness_mean 0.205 (fold 2018-05-03#2), centroid_std 0.071, rolloff_std 0.050.
- **Fold/pooled scale ratios:** 0.877–1.087.
- **MADs:** every MAD > 0 (minimum fold MAD 0.0058).

**Held-out z (robust SD / max |z|).**
- **Spread and extremes:** centroid_mean 1.134 / 4.60, flatness_mean 1.069 / 4.13, centroid_std 0.964 / 3.95, rolloff_std 1.013 / 3.04. All four features are near unit spread on unseen sessions.
- **Feature contributions (rms_z share / argmax fraction):** centroid_mean 0.275 / 0.29, flatness_mean 0.231 / 0.21, centroid_std 0.260 / 0.27, rolloff_std 0.235 / 0.23. The contributions are balanced; no feature dominates.

**Correlation structure (Spearman, pooled / within-session / max LOSO shift).**

| Pair | Pooled | Within session | Max LOSO shift |
|---|---|---|---|
| centroid_mean–flatness_mean | 0.580 | 0.533 | 0.113 (fold 2018-05-03#2) |
| centroid_std–rolloff_std | 0.355 | 0.362 | 0.086 |
| flatness_mean–rolloff_std | −0.286 | −0.164 | 0.026 |

The other three pairs have |ρ| ≤ 0.15. No pair is redundant.

**Held-out session behaviour (`session_robustness.csv`, `v2_session_feature_medians.csv`, `v2_session_robustness.png`).**
- **Largest change vs R0:** the duration-outlier session 2018-02-06#3 (session median 1.483 under R0) is at 0.756 under V2. This is the largest single change, consistent with removing duration.
- **Remaining spread:** V2 session medians still span 2.05× (0.641–1.314). The highest is 2018-02-08#1.

**Recording and segmentation confounds (`confounds.csv`; V2 rms_z, usable events; pooled / within-session Spearman).**

| Quantity | Pooled ρ | Within-session ρ |
|---|---|---|
| recording background RMS | 0.001 | 0.118 |
| duration | 0.056 | −0.080 |
| detector confidence | 0.180 | −0.102 |
| mean_rms | −0.061 | −0.100 |

### Part C. Controlled perturbations (`perturbation_direction.csv`, `feature_perturbation.csv`, `perturbation_response.csv`, `v2_perturbation_response.png`)
- **Consistency with Stage 7:** the re-measured V2 features agree with the Stage 7 perturbed table within 4.9×10⁻⁶ relative, the 6-digit precision of that table.

**Per V2 feature** (median Δz at the listed levels; monotone fraction; fraction of the 12 sessions whose median Δz has the pooled sign):

| Feature | Event / recording gain | Noise 30 / 20 / 10 dB | Tilt −0.5 / 0.5 / 0.9 |
|---|---|---|---|
| centroid_mean | invariant (max \|Δz\| < 10⁻⁵) | +0.55 / +0.92 / +1.73; monotone 1.00; sessions 1.00 | −1.63 / +2.49 / +3.90; monotone 1.00; sessions 1.00 |
| flatness_mean | max \|Δz\| 0.058 (power floor) | +1.88 / +3.31 / +6.18; monotone 1.00; sessions 1.00 | −1.77 / +1.23 / **+0.17**; monotone 0.01; sessions 1.00 / 1.00 / **0.75** |
| centroid_std | invariant | +0.41 / +0.61 / +0.30; monotone 0.20; sessions 0.92 | −0.51 / +0.30 / **+0.20**; monotone 0.43; sessions 1.00 / 0.92 / **0.58** |
| rolloff_std | max \|Δz\| 0.005 | −0.18 / −0.32 / −1.15; monotone 0.29; sessions 0.58 / 0.58 / 0.92 | −0.07 / −0.83 / −1.22; monotone 0.49; sessions — / 1.00 / 1.00 |

**Direction of change at the strongest tilt (a = 0.9).**
- **flatness_mean:** only 58% of events move in the median's direction, although individual events move substantially (95th percentile |Δz| 1.61).
- **centroid_std:** also 58% of events agree, with 95th percentile |Δz| 1.63.

**Aggregate V2 rms_z.**
- **Gain:** invariant. The 95th percentile |Δ| is ≤ 4.9×10⁻⁴ at every event and whole-recording gain level, against 0.47–2.94 for R0, where mean_rms dominates.
- **Noise:** median Δ +0.51 / +1.12 / +2.41, dominated by flatness_mean (median single-feature dominance 0.66–0.85).
- **Tilt:** median Δ +0.57 / +0.94 / +1.45; at a = 0.9, centroid_mean dominates (0.83).
- **Monotonicity:** the z-space distance grows monotonically with intensity in 99.7% of events (noise) and 100% (tilt).
- **Session consistency:** the sign of the median Δ agrees in 92–100% of sessions at every noise and tilt level.

These are **predictable responses to artificial acoustic perturbations**. They are **not evidence** that the score detects meaningful inhalation anomalies, which is not established.

### Part D. Leave-one-feature-out ablations (descriptive; `ablation_summary.csv`)

| Representation | Session ε² | Generalization ratio | Session medians max/min | Noise 10 dB Δ | Tilt 0.9 Δ | Distance monotone (noise / tilt) |
|---|---|---|---|---|---|---|
| V2 | 0.130 | 1.120 | 2.05 | 2.41 | 1.45 | 0.997 / 1.000 |
| − centroid_mean | **0.024** | 1.063 | 2.31 | 2.81 | **0.25** | 0.997 / **0.572** |
| − flatness_mean | 0.170 | 1.089 | 3.16 | **0.60** | 1.70 | **0.796** / 1.000 |
| − centroid_std | 0.157 | 1.092 | 1.90 | 2.97 | 1.82 | 1.000 / 1.000 |
| − rolloff_std | 0.248 | 1.076 | 2.53 | 2.92 | 1.59 | 0.997 / 1.000 |

### Part E. mean_rms as a separate level channel (`level_channel.json`)
- **Session and gain dependence.** Held-out z(mean_rms) has session ε² 0.637. It changes by a median +6.15 z under ×2 gain and −3.07 z under ×0.5, identically for event gain and whole-recording gain.
- **Adding it back to the score (V2_plus_mean_rms)** doubles session dependence (0.130 → 0.261). It also makes the score gain-dependent: 95th percentile |Δ| 3.61 under gain, against 0.0005 for V2.
- **Overlap with V2.** 76% of mean_rms's rank variance is explained by the V2 features (rank R² 0.76), mainly through centroid_mean (Spearman −0.84 pooled, −0.76 within session). It correlates 0.19 with the recording background RMS.

### G6. Reproducibility of the inference contract (`reproduction_summary.json`, `reproduction_events.csv`)
The reference implementation `src/prism_inference.py` was applied to all 361 raw WAVs and compared with the Stage 1 event table:
- **Events:** 364 = 364 events, with no count mismatch in any recording.
- **Bounds and confidence:** event bounds are identical (maximum difference 0 s); detector confidence is within 1.1×10⁻¹⁶.
- **Features:** V2 features and mean_rms are within 7.9×10⁻¹⁵ relative.
- **Usability:** 0 usability or reason mismatches; 318 events scored.
- **Scores:** contract scores equal the in-memory scores within 5.8×10⁻¹⁵.
- **No-event recordings:** the 39 recordings without a Stage 1 event are exactly the 39 reported as `NO_INHALATION_DETECTED`.
- **Output validation:** every output passed the contract validator.
- **Baseline round trip:** baseline JSON reproduces the in-memory z exactly (maximum difference 0).
- **Leakage:** 0 / 0.
- **Byte identity:** see Deviations.

## Acceptance Gate (`acceptance_gate_results.json`)
**Outcome: FAIL. The only failing criterion is G4c.**

| Id | Measured | Verdict |
|---|---|---|
| G1a | ε² V2 0.130 vs R0 0.211 | pass |
| G1b | 1.120 | pass |
| G2a | 0.221 | pass (confirmatory) |
| G2b | 0.877–1.087 | pass (confirmatory) |
| G2c | all MAD > 0 | pass (confirmatory) |
| G3a | 0.113 | pass |
| G4a | 4.9×10⁻⁴ | pass (confirmatory) |
| G4b | noise 0.997, tilt 1.000 | pass |
| **G4c** | **2 of 14 applicable cases below 0.9: z(flatness_mean) at tilt 0.9 = 0.75 (9/12 sessions); z(centroid_std) at tilt 0.9 = 0.58 (7/12)** | **FAIL** |
| G4d | no fragile feature | pass (confirmatory) |
| G5a / G5b | ρ 0.001 / 0.118 | pass |
| G6a | 34 files identical to the reference run | pass |
| G6b | exact reproduction (above) | pass |
| G6c | 0 | pass |
| G6d | 0 / 0 | pass |

The contract is therefore issued as **`DRAFT_NOT_FROZEN`** (`inference_contract_v2.json`), as pre-registered. Neither V2 nor the gate was changed after the results were seen.

## Deviations From the Pre-registration
1. **G6a comparison scope.** The in-run G6a comparison excludes the contract's `status`, `status_meaning` and `acceptance_gate` fields and `acceptance_gate_results.*`, because those record the G6a verdict itself and cannot match a reference run in which G6a was not yet evaluated.
   - **Literal wording also met:** to satisfy the pre-registered wording ("the contract JSON … byte-identical"), two further complete runs, each evaluated against the same reference run, were compared in full. All 38 non-figure outputs were byte-identical, including the complete contract and gate results.
   - **Verdict unaffected:** no criterion's verdict depends on this.
2. **Contract design additions.** These affect contract design only, not the gate:
   - an input-error code `detector_feature_extraction_failed`, for the existing detector's failure path when frame features are non-finite (never observed);
   - scoreability requires only the 5 contract features to be finite, where Stage 1 required 13. G6b verified that no reference event changes.

## Interpretation
1. **What V2 does well.**
   - **Stability:** it is stable as a baseline (fold centres ≤ 0.22 SD, scales within 0.88–1.09×) and as a correlation structure (≤ 0.11 shift).
   - **Invariance and confounds:** it is invariant to device gain and does not track recording background noise.
   - **Scale:** held-out z has unit-like spread.
   - **Session dependence:** it is lower than R0 (0.130 vs 0.211) and similar to R5/R6.
   - **Reproducibility:** end to end from raw audio.
2. **What fails.** Two of the four features respond to a strong spectral tilt (a = 0.9, a large high-frequency emphasis) in directions that differ between events and sessions: flatness_mean (non-monotone over tilt: −1.77, +1.23, +0.17) and centroid_std. A plausible explanation (hypothesis, not tested) is that the direction depends on each recording's original spectral balance, which differs between sessions (centroid_mean session ε² 0.62).
   - **Practical relevance:** a change of microphone or enclosure is a spectral-shape change, so for these two features the shift caused by new hardware is expected to differ in direction between recordings and could not be removed by a single offset.
   - **Aggregate score:** the V2 score itself responded consistently (100% of sessions at a = 0.9). The failure is at feature level, which is what G4c was pre-registered to test.
3. **Remaining session dependence sits in spectral_centroid_mean** (post hoc, from the ablations).
   - Removing it lowers session ε² from 0.130 to 0.024, but also removes the only fully predictable tilt response (tilt-distance monotone 0.57).
   - centroid_mean carries most of the level information mean_rms used to carry (ρ −0.84), so removing mean_rms did not fully remove level/session information from V2.
   - Removing rolloff_std raises session ε² to 0.248, so the within-event variability features dilute the session effect.
   - These observations were made on the same data. They are hypotheses for a future pre-registration, not grounds to change V2 now.
4. **mean_rms belongs outside the score.** In the score it doubles session dependence and makes the score device-gain dependent. As a separate, uncalibrated level channel it keeps loudness visible for diagnostics without contaminating the score.

## Answers to the Stage 8 Questions
1. **Does V2 pass the pre-registered gate?** No.
2. **Which criterion fails?** Only G4c, on per-session direction consistency: z(spectral_flatness_mean) at tilt a = 0.9 (0.75 of sessions) and z(spectral_centroid_std) at tilt a = 0.9 (0.58). All other criteria pass.
3. **Is any individual feature problematic?**
   - **spectral_flatness_mean:** direction-inconsistent at strong tilt, non-monotone over tilt, and the most noise-sensitive V2 feature (+1.88 z at 30 dB; predictable in direction).
   - **spectral_centroid_std:** non-monotone under noise (20%) and direction-inconsistent at strong tilt.
   - **spectral_rolloff_std:** session-inconsistent under mild noise (0.58 at 30/20 dB; not gated, because G4c tests 10 dB).
   - **spectral_centroid_mean:** fully predictable, but it carries most of the remaining session dependence and level coupling.
4. **Does V2 respond predictably to controlled perturbations?** The aggregate score does: gain-invariant, monotone in ≥ 99.7% of events, direction consistent in ≥ 92% of sessions. At feature level, not fully (G4c). These are responses to artificial perturbations, not evidence of anomaly detection.
5. **How much session dependence remains?** Held-out ε² 0.130, session medians spanning 2.05× (0.64–1.31), maximum session-vs-rest |δ| 0.43, generalization ratio 1.12. Whether this reflects subjects, devices or placements is not established by the current data.
6. **Is mean_rms better treated as a separate level channel?** Yes. It is technically sensible to keep it outside the score: uncalibrated, not z-scored, not an anomaly signal.
7. **Is V2 stable enough to freeze temporarily for MVP engineering?** Not under the pre-registered rule. The contract is `DRAFT_NOT_FROZEN`. Freezing V2 as an exception, or replacing it, is the project owner's decision.
   - **What engineering can proceed with:** the interface (audio input, detector, events, scoreability, output schema, recording states, level channel) is fully specified and verified.
   - **Why the V2 decision does not block it:** the four V2 features are event-level mean/std aggregates of the per-frame spectral values the detector pipeline already computes, so implementing them costs little either way.
8. **Which questions remain unresolved?**
   - whether any representation detects meaningful inhalation deviations (no ground truth)
   - what sessions represent (subjects, devices, placements)
   - behaviour on PRISM hardware (microphone, enclosure, 5 s buffer) and whether a hardware reference set is needed (expected)
   - personalization (V3)
   - whether flatness_mean / centroid_std should be replaced, and whether centroid_mean's session dependence is acceptable
   - threshold methodology (deliberately not started)

## Inference Contract (summary; full text in `inference_contract_v2.json`)
**Input**
- 8 kHz mono float32 in [−1, 1], from 16-bit PCM as sample/32768 (24-bit sources: int16 = s24 >> 8).
- No resampling, normalization, filtering or trimming.
- At least 1,536 samples (one detector window).
- Input errors: `unsupported_sample_rate`, `invalid_shape`, `empty_audio`, `nonfinite_audio`, `amplitude_out_of_range`, `shorter_than_one_detector_window`, `detector_feature_extraction_failed`.

**Detector:** `results/inhaler_cnn.onnx`.
- Input: float32 [N, 25, 124] raw librosa features (STFT 256 / 64, Hann, centred), windows of 25 frames with stride 2.
- Output: logits [N, 4] in class order Drug, Exhale, Inhale, Noise; softmax, then argmax.
- Window i spans [0.016·i, min(0.016·i + 0.2, duration)].

**Grouping:** default `TemporalGroupingConfig`. Inhale windows join an event while start ≤ the event's latest end. Event start = first window start; end = max window end; confidence = mean P(Inhale).

**Scoreability:** Stage 1 rule v1. Events with duration < 0.5 s, a neighbour gap < 0.2 s, a start/end within 0.008 s of the recording edges, or a non-finite feature are `NOT_SCOREABLE`, with reasons.

**Features:** in this exact order:
1. spectral_centroid_mean
2. spectral_flatness_mean
3. spectral_centroid_std
4. spectral_rolloff_std

They are computed on waveform[floor(start·8000) : ceil(end·8000)] with the same STFT, as mean / population std over frames in float32. mean_rms (256-sample frames, hop 64) is the separate level channel.

**Baseline:** `v2_baseline.json`, loaded and validated, never fitted at inference.

**Score:** z_j = (x_j − centre_j) / scale_j; anomaly_score = sqrt(mean z²).

**Output:** `inference_output.schema.json`.
- **Recording status:** `EVENTS_DETECTED` / `NO_INHALATION_DETECTED` / `INPUT_ERROR`.
- **Per event:** event_id, start_time, end_time, duration_s, detector_confidence, detector_max_confidence, window_count, status (`SCORE_ONLY` / `NOT_SCOREABLE`), not_scoreable_reasons, anomaly_score, feature_values, feature_z_scores, mean_rms.
- **Explicitly forbidden derived outputs:** NORMAL/ANOMALY, quality percentages, Correct/Incorrect and technique error codes. The existing React Native app simulates `quality`, `Correct`/`Incorrect` and technique errors; none of these is supported by PRISM evidence.

**Conformance:** 9 golden cases (`golden/golden_manifest.json`):
- 5 dataset recordings, identified by SHA-256
- 2 synthetic WAVs
- 2 input errors

Tolerances: z and score ±0.01, features ±10⁻⁴ relative; event structure exact.

## Limitations
- **No ground truth:** there is no anomaly or technique ground truth, so the score is a distance from a dataset baseline only.
- **Same data:** V2 and the gate thresholds were set after Stages 2–7 on the same 318 events, and several criteria were confirmatory.
- **Conventional thresholds:** some gate thresholds (0.1 z, 95%, 90%, |ρ| 0.3) are conventions, not derived values.
- **Inferred sessions:** sessions are recording sittings. The baseline is global, not personal.
- **Synthetic perturbations:** event boundaries are fixed, and first-order tilt only approximates a hardware frequency-response change.
- **Detector in-sample:** the detector is in-sample for about two-thirds of the recordings.
- **Hardware not validated:** PRISM hardware (INMP441, enclosure, 5 s buffer) is entirely unvalidated.

## Decision and Next Step
1. **Status:** per the pre-registered rule, V2 is **not frozen**; the contract is `DRAFT_NOT_FROZEN`, and the failing criterion is G4c. No threshold, no NORMAL/ANOMALY output.
2. **Owner decision:**
   - **(a)** Accept V2 as a documented exception for MVP engineering only, with anomaly_score treated as experimental and not shown to users; or
   - **(b)** Keep the interface and replace the representation after a new pre-registered evaluation.

   Any revised feature set should be tested on data not used to design it.
3. **Engineering can start on the interface now** (hardware → audio → detector → events → scoreability → features → output schema, including `NO_INHALATION_DETECTED`), verified against the golden vectors.
4. **Next scientific step:** reference recordings on PRISM hardware (same subjects, and ideally the dataset protocol, plus deliberate, documented variations). They are needed to quantify the domain shift the tilt results predict and to re-baseline for the device. Longitudinal user/device identifiers are still required for personalization (V3).

---

# Research Entry 11 — 2026-10-06: Per-Recording Reference Run and Documentation Audit (no new experiment)

**No new experiment, threshold or NORMAL/ANOMALY output.** This entry records an engineering reference run and a documentation audit, both done for app-team synchronisation.

## Question
1. What exactly does the full pipeline (inference contract V2) output for one recording, stage by stage?
2. Does `ARCHITECTURE.md`, last updated on 2026-08-23 and never revised after Stage 1, describe the implemented system?

## Work Performed
1. **Reference run on one recording.**
   - `results/recording_runs/run_recording.py` (new) saves every stage a mobile re-implementation must reproduce: per-frame 124-dim features, raw ONNX logits and probabilities per window, the contract output, a summary and a diagnostic plot.
   - It was run on `data/rec2018-01-22_17h41m49.809s.wav` (SHA-256 `72eebb44…56b5`, `input_domain = reference_dataset`).
   - It calls the unchanged `prism_inference.analyze_recording`.
2. **Documentation audit.** Every DSP, inference, baseline and interface statement in `ARCHITECTURE.md` was checked against:
   - `src/` and the contract;
   - librosa 0.11.0 source and defaults, inspected in the project venv;
   - Entries 1–10.
3. **Documentation fixes.**
   - `ARCHITECTURE.md` rewritten to the current state.
   - `README.md` corrected where it contradicted the same evidence.

## Results
**1. Reference run** (`results/recording_runs/rec2018-01-22_17h41m49.809s/`)
- **Output:** `EVENTS_DETECTED` with one event at 0.688–2.232 s, `SCORE_ONLY`, `anomaly_score` 0.555 and `mean_rms` 0.177.
- **Agreement with earlier results:** identical to the Stage 1 row and to the Stage 8 reproduction (start, end and confidence differences 0; feature differences ≤ 1.2×10⁻¹⁵ relative).
- **Size of the intermediate stages:** 1,501 frames, 739 windows.
- **In-sample score:** this event was in the deployment-baseline fit, so the contract score is in-sample. Refitted without its own session (309 events, 17 sessions) it scores 0.571. The held-out z-scores match the Stage 5 `C_loso_global` row to 6 digits.
- **Context (not a threshold):** held-out V2 scores on unseen sessions are 0.42 / 1.00 / 1.75 (5th percentile / median / 95th percentile).

**2. Discrepancies between `ARCHITECTURE.md` (2026-08-23) and the implementation or evidence**

| Previous statement | Actual (verified) |
|---|---|
| Pre-emphasis y[n] = x[n] − 0.97·x[n−1] before the STFT | No pre-emphasis anywhere in the extractor |
| MFCC = DCT(log(mel @ \|S\|)); "S shared for MFCC and all spectral features" | `librosa.feature.mfcc` computes its own **power** mel spectrogram from the waveform: Slaney scale and normalisation, then `power_to_db` with `top_db = 80` relative to the **maximum of the whole input buffer**, then DCT-II ortho. S is shared only by centroid, flatness and rolloff |
| Deltas: HTK regression, mirror-padded edges; delta2 = delta(delta) | Savitzky–Golay, `mode="interp"`. The interior equals the regression filter, but the edges differ. Delta2 is a direct second-derivative filter (polyorder 2), not delta of delta |
| Flatness = geomean(S) / mean(S) | Computed on the power spectrum: max(S², 1e-10) |
| ZCR over the 256-sample frame | Uses librosa's default `frame_length = 2048`, centred with edge padding |
| n_frames = ceil(n / 64) | 1 + ⌊n / 64⌋ (1,501 for 96,000 samples) |
| "Resample BLE audio if needed" | The contract never resamples: a rate other than 8 kHz is `INPUT_ERROR` |
| Frame-sequence reconstruction by majority vote, then session analytics (`coord_delay`, `insufficient_inhale` < 1.0 s, `late_actuation` > 0.5 s, `missed_dose`), then composite GOOD/POOR labels | Not implemented. The implemented flow is the V2 contract (window grouping, scoreability, features on the event segment, frozen baseline, `SCORE_ONLY`). The listed analytics and labels have no evidential basis or are forbidden by the contract |
| §9 personalized engine (Mahalanobis, EMA, 1.5/3.0 bands) | Already recorded as legacy and not adopted (Entry 3); withdrawn from the document |
| XGBoost importance: spectral flatness first | The saved `feature_importance_xgboost.png` labels all of its top 15 as MFCC-derived. `summary_report.txt` describes 7-frame / 868-feature windows, contradicting its own 3,100-feature input. The artefacts need regeneration before any importance claim |
| Breath centroid ~0.1–0.3; Drug flatness 0.7–1.0 | The usable inhalation events have centroid_mean 0.33–0.42 (5–95%). No Drug/Exhale/Noise spectral measurements exist |
| BLE `audio_len` `uint16` with a 5 s buffer | 5 s × 8 kHz × 2 bytes = 80,000 bytes > 65,535. Recorded as an open issue (ARCHITECTURE §5) |

**3. Integration consequences recorded in `ARCHITECTURE.md` §4–§6.** These follow from the contract rules and are untested on hardware:
- **Pre-trigger and post-offset audio:** clips need audio before the onset and after the offset, or inhalations touching the clip edges become `NOT_SCOREABLE` (`recording_boundary`).
- **Whole-clip MFCCs:** because the 80 dB floor depends on the whole buffer, MFCCs must be computed over the whole clip in one pass.

**4. Tests:** `python -m unittest discover tests` gives 173 tests, all passing. No `src/` code was changed.

## Decision
- **`ARCHITECTURE.md` updated to the current state** (revision 2026-10-06). It now uses the AGENTS.md status vocabulary and names the contract and code as authoritative.
- **`README.md` corrected** where it contradicted the same evidence.
- **Historical entries unchanged.** Entry 1 §7 ("shared magnitude spectrogram", "regression window") is superseded by the table above for the MFCC and delta details.

## Limitations
- **One recording, in-sample.** The reference run covers one reference-dataset recording, in-sample for the deployment baseline and possibly for the detector.
- **librosa-version dependence.** The librosa internals were checked for the installed version 0.11.0. A different librosa version could change defaults; the contract's `frame_features` text and the golden vectors are the version-independent reference.

## Next Step
Unchanged from Entry 10:
- the owner's decision on V2;
- PRISM-hardware reference recordings;
- longitudinal user/device data.

Additional engineering item: regenerate the XGBoost importance artefacts, or remove them from documentation, before citing them.

---

# Research Entry 12 — 2026-10-07: Final Event-Level Assessment — Reference Calibration of the V2 Deviation (Stage 9)

**No clinical, technique-quality or NORMAL/ANOMALY label is introduced.** The new categorical output says only where an inhalation's acoustic spectral profile lies relative to the reference distribution.

## Question
Given the pipeline as it stands and no clinical ground truth, what should the final event-level output be, and what is needed to make it reliable enough to end the pipeline?

## Phase 1: Audit of the pipeline as implemented
| Stage | Implementation | Input → output | Tests | Validation (log) | Status before Stage 9 |
|---|---|---|---|---|---|
| Input check | `prism_inference.check_input` | float32 8 kHz mono → waveform or `INPUT_ERROR` | `test_prism_inference` | contract, G6b (Entry 10) | implemented |
| Frame features | `librosa_extractor.extract_features_from_audio` | waveform → float32[1 + n//64, 124] | `test_post_event` (indirect) | DSP described in ARCHITECTURE §6 (Entry 11) | implemented |
| Windows + detector | `post_event.generate_window_predictions`, `inhaler_cnn.onnx` | frames → per-window softmax over Drug/Exhale/Inhale/Noise (0.2 s windows, 0.016 s stride) | `test_post_event` | window CV 0.89 accuracy; deployed fold in-sample for about 2/3 of the recordings (Entries 1, 3) | validated at window level |
| Inhale grouping | `post_event.group_inhale_events` | windows → events (start, end, confidence) | `test_post_event` | 259/260 annotated inhalations matched, IoU 0.876 (Entry 3; mostly in-sample) | validated against annotations |
| Scoreability | `prism_inference.not_scoreable_reasons` (usability rule v1) | events → `SCORE_ONLY` / `NOT_SCOREABLE` + reasons | `test_inhale_dataset`, `test_prism_inference` | Entries 3, 8 | implemented |
| Event features | `prism_inference.event_measurements` → `post_event.analyze_inhalation` | waveform slice → 4 V2 features + `mean_rms` | `test_prism_inference` | reproduces Stage 1 within 7.9×10⁻¹⁵ (Entry 10, G6b) | implemented |
| Baseline + deviation | `FrozenBaseline` (median / 1.4826·MAD, 318 events, 18 sessions), `anomaly_score` = rms z | features → z, aggregate deviation | `test_prism_inference`, `test_v2_validation` | Stages 5–8; gate failed only on G4c | `DRAFT_NOT_FROZEN` |
| **Final assessment** | **none** | — | — | — | **missing: the pipeline ended at an uncalibrated distance** |

There are no broken links inside the Python pipeline: `prism_inference.analyze_recording` runs WAV → `SCORE_ONLY` end to end and reproduces every Stage 1 event (Entry 10). Three gaps remained:
1. **The final output had no statistical meaning.** `anomaly_score` is a distance with no reference scale, and a fixed cut such as "z > 2" would be arbitrary.
2. **Segment bounds depend on floating-point rounding.** Integer sample bounds are taken from float times (`ceil(end·8000)`), so a re-implementation can differ by one sample. The app audit found this in 129 of 364 events. The canonical output now reports the exact integer sample slice.
3. **Deployment scores on reference recordings are in-sample.** Every reference event helped fit the baseline, so its deployment score is optimistic (6–11%, Entry 6).

## Phase 2: Why there was no defensible final assessment
- **No ground truth:** there are no technique or deviation labels, so a threshold cannot be tuned to detection performance (Entries 2, 8).
- **Session dependence:** held-out ε² is 0.130 (Entry 10). Any fixed cut on the raw distance flags sessions at different rates.
- **The distance is uncalibrated:** its held-out distribution (median 1.00, p95 1.75) has no reference scale attached, and in-sample scores are optimistic (Entry 6).
- **Natural extremes are segmentation artefacts,** not inhalation deviations (Entries 8–9), so they cannot validate a cut.
- **G4c:** per-feature direction under strong tilt is inconsistent across sessions. This is a hardware-transfer risk, not a reference-domain defect.
- **Alternatives:** the Stage 7 multivariate gate blocked covariance-aware distances for 6–7 features. No alternative to V2 had been compared since.

**What the data do support:** a calibration of the distance against reference events measured exactly as a new session's events are, i.e. leave-one-session-out (LOSO) reference scores. This turns the distance into a conformal p-value, whose meaning is testable without labels. Under exchangeability, P(p ≤ α) ≤ α for reference-like events.

## Pre-registration
- **File:** `results/final_assessment/stage9_preregistration.json`, written before any Stage 9 result: SHA-256 `5b5cdf6e…01e6`, 2026-10-06T18:24:44Z.
- **Hash check:** `src/assessment_validation.py` refuses to run if the file changes, and a test checks the rule constants against it.
- **Disclosure:** it was not committed separately before the analysis (no commit was authorized at that point). The recorded hash and timestamp are the evidence of order.
- **Known before writing it:** the Entry 10 numbers listed in the file.

## Method
**Score (unchanged):** a = sqrt(mean z²) over the 4 V2 features.

**Calibration:**
- **C:** the LOSO scores of the reference's usable events.
- **Tail probability:** p(a) = (1 + #{c ∈ C : c ≥ a}) / (|C| + 1).
- **Cut:** c_α = the (n − m)-th smallest c, with m = ⌊α(n + 1) − 1⌋ computed exactly. Then p ≤ α iff a > c_α.

**Categories (α = 0.05, the central-95% reference-interval convention):**
- `WITHIN_REFERENCE_RANGE` if p > 0.05;
- `OUTSIDE_REFERENCE_RANGE` if p ≤ 0.05;
- `NOT_ASSESSED` for non-scoreable events, with reasons.

**Reliability:**
- **Band:** [c_lo, c_hi] = 5th–95th percentiles of c_α under a session bootstrap (resample whole sessions; 2,000 draws, seed 20261006).
- **Segmentation variants:** the event bounds moved by one detector stride at the start and/or end (8 variants, reference float formula).
- **Label:** `STABLE` if the event and every variant lie on one side of the band, otherwise `BORDERLINE`.

**Evaluation (leakage-free):** for each of the 18 sessions with usable events, the reference was refitted on the other sessions (baseline plus nested LOSO calibration, cut and band), frozen, and applied to that session. Reference events from sessions without usable events are scored by the deployment reference, which never saw them.

**Category gate (pre-registered):**
- **K1:** held-out exceedance at α 0.05 — the 95% session-bootstrap interval contains 0.05 and the point estimate is ≤ 0.10.
- **K2:** (c_hi − c_lo) / (2 c_α) ≤ 0.20.
- **K3:** the OUTSIDE fraction is non-decreasing over noise 30 → 20 → 10 dB and tilt 0.5 → 0.9, and the gain effect is ≤ 0.02.
- **K4:** byte-identical rerun.

**Representations (pre-registered):**
- **V2** (incumbent).
- **V2_RC:** sqrt(z′R⁻¹z/4), with R the per-fold robust correlation (2 sin(πρ/6) from Spearman ρ). Admissible only under the Stage 7 multivariate gate thresholds.
- **V2_G4C:** centroid_mean + rolloff_std, which drops the two G4c-failing features.
- **Rule:** an alternative replaces V2 only if it passes every Stage 8 criterion including G4c, and K1, while V2 does not.

**Implementation:**
- `src/prism_assessment.py`: the production assessment layer and the canonical CLI `python src/prism_assessment.py <wav>`.
- `src/assessment_validation.py`: this stage; command `venv/Scripts/python.exe src/assessment_validation.py [--reference-run <dir>]`.
- **Tests:** `tests/test_prism_assessment.py`, `tests/test_assessment_validation.py` (31 tests; full suite 204 passing).
- **Outputs:** `results/final_assessment/`.

## Results

### A. Representation comparison (`representation_gate.csv`; Stage 8 criteria recomputed)
| Representation | Session ε² | Gen. ratio | Failing criteria | K1 held-out exceedance [95% session CI] |
|---|---|---|---|---|
| **V2** | 0.130 | 1.120 | **G4c** (flatness_mean 0.75, centroid_std 0.58 of sessions at tilt 0.9) | 5.35% [0.78, 8.97] |
| V2_RC | 0.089 | 1.093 | **G4c** (same two features, and the aggregate at tilt −0.5: 0.833) | 5.03% [1.80, 8.14] |
| V2_G4C | 0.192 | 1.104 | **G4b** (noise distance monotone in only 82% of events) | 5.03% [0.95, 8.42] |

- **V2_RC was admissible:** max condition number 5.5, max correlation shift 0.114, ≥ 65 training events per feature. It lowers session dependence (0.130 → 0.089) but makes the aggregate's tilt response less consistent across sessions.
- **V2_G4C passes G4c** but loses noise monotonicity and is more session-dependent.
- **No alternative qualifies, so V2 is retained** by the pre-registered rule.
- **V2 reproduction:** its recomputed criteria reproduce Stage 8.

### B. Calibration on unseen sessions (`heldout_assessment.csv`, `calibration_by_session.csv`)
- **K1:** 17 of 318 held-out usable events are OUTSIDE: **5.35% [0.78%, 8.97%]**.
- **Other α levels:** α 0.01 gives 1.26% [0.00, 2.20]; α 0.10 gives 11.32% [3.79, 17.94].
- **Distribution of held-out p:** KS distance from uniform 0.020; p05/median/p95 = 0.050 / 0.509 / 0.953.
- **Session concentration:** 14 of the 17 OUTSIDE events come from 2018-05-03#2 (7/57) and 2018-05-03#3 (7/52). Of the rest, 2 are in 2018-01-23#3 (2/29) and 1 in 2018-02-06#3 (1/13); all other sessions have 0. Heterogeneity across the 12 sessions with ≥ 5 events: χ² 19.8, df 11, parametric-bootstrap p 0.058.
- **Interpretation:** calibration holds **marginally** over sessions, not per session.
- **OUTSIDE events are well-formed inhalations, not segmentation artefacts:**
  - median duration 1.66 s against 1.62 s for WITHIN events;
  - detector confidence 0.98 against 0.95;
  - 13 of 17 are annotation-matched;
  - Spearman of score with duration 0.06, with confidence 0.18.

### C. Reference stability
- **K2:** the deployment cut is 1.830 with band [1.620, 1.972], a relative half-width of 0.096.
- **Session removal:** the 18 fold cuts range 1.727–1.968 (0.944–1.076× the deployment cut).
- **Event removal:** a leave-one-event-out refit changes the cut by at most 3.8%. 5 reference events lie within that distance of the cut.

### D. Controlled perturbations (`perturbation_assessment.csv`; K3)
OUTSIDE fraction among the 318 held-out events:

| Condition | OUTSIDE fraction |
|---|---|
| Identity | 5.35% |
| Noise 30 / 20 / 10 dB SNR | 28.6% / 66.0% / 99.4% |
| Tilt +0.5 / +0.9 | 46.5% / 80.8% |
| Tilt −0.5 | 24.8% |
| Event gain ×0.5–×2 and whole-recording gain ×0.5 / ×2 | unchanged (Δ = 0 at every level) |

- **Predictable response:** the assessment responds monotonically to known spectral changes and is invariant to device gain.
- **Not a detection result:** these are controlled acoustic perturbations, not technique errors. They show what magnitude of spectral change the assessment resolves beyond reference variability.

### E. Reproducibility (K4)
- **K4:** run 2 against run 1 gave 9 evaluation files byte-identical.
- **Full directory:** runs 2 and 3 produced byte-identical files across all 28 outputs (golden vectors, contract, reference, end-to-end tables, figures). `analysis_summary.json` is identical except for runtime and git provenance.
- **Inputs:** unchanged by hash.

### F. Reliability (held-out end-to-end run)
- **Overall:** 288 STABLE, 30 BORDERLINE (9.4%).
- **OUTSIDE events:** 9 STABLE and 8 BORDERLINE.
- **WITHIN events:** 279 STABLE and 22 BORDERLINE.
- **Segmentation sensitivity:** a one-stride boundary change moves the score by a median range of 0.075 (p95 0.35).
- **Interpretation:** about half of the OUTSIDE statements are borderline, which is why reliability is part of the output.

### G. Dominance and leave-one-feature-out agreement
- **Largest deviation in OUTSIDE events:** centroid_mean 7, centroid_std 6, rolloff_std 3, flatness_mean 1. The median dominant share is 0.59, against 0.58 for all events.
- **Leave-one-feature-out agreement:** every feature contributes. Agreement is 0.956–0.984, with Cohen's κ 0.56 (− centroid_std), 0.63 (− centroid_mean), 0.74 (− rolloff_std) and 0.84 (− flatness_mean).

### H. Natural extreme events (diagnostic; these are `NOT_ASSESSED` in production)
- **Excluded events forced through the held-out assessment:** 24% OUTSIDE (11/46), against 5.3% of usable events (score δ 0.41).
- **By exclusion group:** close-neighbour only 14%, too-short 20%, too-short + close-neighbour 36%, too-short + boundary 1/1.
- **Consistent with Entries 8–9:** segmentation artefacts carry atypical spectra, and the scoreability rule removes them before assessment.

### I. End-to-end runs of the canonical pipeline (`e2e_*`)
**LOSO references (held-out):**
- **Recordings:** 361 processed, **0 failures**; 322 `EVENTS_DETECTED`, 39 `NO_INHALATION_DETECTED`, 0 `INPUT_ERROR`.
- **Events:** 364; 318 scoreable; 46 not scoreable (short 20, close neighbour 14, short + close 11, short + boundary 1).
- **Assessments:** 301 WITHIN, 17 OUTSIDE, 46 NOT_ASSESSED. 0 non-finite values. Every scoreable event had all 8 segmentation variants.
- **Agreement with the table-based evaluation:** 0 event-count mismatches, 0 scoreability mismatches, scores within 5.3×10⁻¹⁵, 0 p mismatches, 0 category mismatches.
- **Runtime:** mean 0.26 s per 12 s recording (max 1.2 s) on the development PC, including the 8 variants.

**Deployment reference (in-sample for these reference recordings):** 307 WITHIN, 11 OUTSIDE, 46 NOT_ASSESSED; 295 STABLE, 23 BORDERLINE. The deployment baseline is identical to `v2_baseline.json`, and the calibration distribution equals the Stage 8 held-out V2 distribution (p05/median/p95 0.4226 / 0.9973 / 1.7451).

## Category gate (`category_gate_results.json`)
| Id | Measured | Verdict |
|---|---|---|
| K1 | 5.35% [0.78%, 8.97%] | pass |
| K2 | 0.096 | pass |
| K3 | noise 0.286 → 0.660 → 0.994; tilt 0.465 → 0.808; gain Δ 0 | pass |
| K4 | 9/9 files identical (and 28/28 in the full comparison) | pass |

**Outcome: PASS. The contract is issued as `CATEGORICAL_ASSESSMENT_ADOPTED`** (`assessment_contract_v1.json`, `prism-assessment-v1.0`).

## Personalization
- **Metadata audit:** the 361 WAVs carry only `fmt`/`data` chunks (no LIST/INFO metadata). Filenames are timestamps only, and `annotation.csv` has 4 columns. No user, device, placement or protocol identifier exists, and sessions are inferred sittings.
- **Nothing personal was implemented.** The reference is global and is labelled so.
- **Within-session reference:** Stage 5 already tested it. A deployable 5-event warm-up did not improve on the global baseline; 10 events did (sensitivity only).
- **Session dependence of the final assessment:** per-session OUTSIDE rates range 0–13.5%.
- **Metadata required for future personalization:**
  - per recording: a pseudonymous user id; device id (hardware serial, microphone/firmware revision, gain configuration); inhaler/canister type; recording session id and timestamp; placement/environment notes;
  - for deliberate protocol variations: the condition label as a protocol condition, not a clinical label;
  - for evaluation: a reference recording set on PRISM hardware.
- **Data needed to evaluate a personal reference:** at least 10 events per user per context (Stage 5 sensitivity) and several sessions per user, evaluated with held-out-session splits within a user.

## Final pipeline contract (`results/final_assessment/assessment_contract_v1.json`, schema `assessment_output.schema.json`)
- **Input:** 8 kHz mono audio, as in `prism-inference-v2.0`, unchanged.
- **Per recording:**
  - inference fields: status, error, input, domain flag;
  - reference statistics: α, categorical flag, cut, band, n events/sessions, feature centres and scales.
- **Per event:**
  - **segmentation:** bounds (times plus the exact integer sample slice), duration;
  - **detector:** confidence, max confidence, window count;
  - **scoreability:** status and reasons;
  - **measurements and deviations:** feature values, `mean_rms`, per-feature z, aggregate deviation;
  - **assessment:** reference tail probability, assessment, reliability;
  - **diagnostics:** segmentation variant count and deviation range, dominant feature and share, reason.
- **Conformance:** golden vectors in `results/final_assessment/golden/`.
- **Reference implementation:** `src/prism_assessment.py`.

## Claim matrix
| # | Claim | Status | Evidence | Limitation |
|---|---|---|---|---|
| 1 | CNN event detection | Validated at window level | 3-fold CV accuracy 0.89, Inhale F1 0.89 (Entry 1) | deployed fold in-sample for about 2/3 of the recordings |
| 2 | Inhalation event formation | Validated against annotations | 259/260 annotated inhalations matched, IoU 0.876 (Entry 3) | mostly in-sample; annotations are acoustic, not technique |
| 3 | Event characterization | Implemented, reproducible | V2 features reproduce Stage 1 within 7.9×10⁻¹⁵ from raw audio (Entry 10); 0 non-finite values in 364 events | float-derived segment bounds (one-sample sensitivity) |
| 4 | Robust reference construction | Validated across sessions | fold centres ≤ 0.22 SD, scales 0.88–1.09× (Entry 10); cut stable 0.94–1.08× under session removal and ≤ 3.8% under event removal | 18 inferred sittings, unknown subjects/devices |
| 5 | Acoustic deviation scoring | Implemented; behaves predictably | gain-invariant; monotone under noise/tilt; generalization ratio 1.12 | V2 `DRAFT_NOT_FROZEN` (G4c); session ε² 0.130 |
| 6 | Final event-level assessment | **Implemented and validated on the reference domain** | held-out end-to-end run over all 361 recordings, 0 failures; exact agreement with the evaluation | reference domain only; 9.4% of assessments BORDERLINE (8/17 of OUTSIDE) |
| 7 | Statistical / reference-consistency assessment (p, WITHIN / OUTSIDE at α 0.05) | **Supported marginally** | K1–K4 pass; exceedance 5.35% [0.78, 8.97]; 1.26% at α 0.01, 11.3% at α 0.10 | not per session (0–13.5%; 14/17 in two sessions); same-data evaluation |
| 8 | Personalization | Not supported by the data | no identifiers exist (metadata audit) | needs longitudinal user/device data |
| 9 | Normal / anomalous classification | Not supported | no anomaly ground truth | OUTSIDE means atypical for the reference, not anomalous |
| 10 | Technique-quality classification | Not supported | no technique labels (Entry 2) | — |
| 11 | Clinical interpretation | Not supported | — | no clinical data; PRISM hardware not validated |

## Interpretation
1. **The strongest defensible final assessment** is a reference-calibrated statement: the event's V2 acoustic profile is within, or outside, the region that contains 95% of reference inhalations from unseen sessions. It comes with its p-value and a reliability label.
   - **Its false-flag rate is controlled marginally:** 5.35% on held-out sessions.
   - **It is stable,** gain-invariant, monotone in controlled spectral changes, and reproducible.
2. **Its main weakness is session concentration.** Most held-out OUTSIDE statements fall in two sittings whose spectra differ from the rest of the reference. Within the reference domain this is correct behaviour, because those inhalations *are* atypical for the reference. But it means an OUTSIDE statement often reflects the recording context (subject, device or placement — unknown) as much as the individual inhalation.
3. **Why categories are worth emitting:** the continuous p and deviation are always reported. The categories add a fixed, documented interpretation, not new information.

## Limitations
- **No independent data:** the same 318 events were used in Stages 2–8. Results are calibration and behaviour checks, not detection performance.
- **Session-level exchangeability is assumed** for marginal validity. Per-session validity does not hold (heterogeneity p 0.058).
- **Arbitrary conventions:** α = 0.05 and the K2 limit of 0.20 are conventions, not derived values.
- **Pre-registration order:** the pre-registration was hash-recorded, not committed, before the analysis.
- **Hardware:** PRISM hardware audio is unvalidated (`baseline_domain_validated = false`). G4c predicts direction-inconsistent feature shifts under a hardware frequency-response change.
- **Detector:** in-sample for about two-thirds of the recordings.

## Decision and Next Step
1. **Adopted:** the final event-level output is `prism-assessment-v1.0`: inference contract V2 plus reference tail probability plus WITHIN / OUTSIDE at α 0.05 plus reliability. NOT_ASSESSED events carry reasons.
2. **Retained:** V2 remains the representation (no alternative qualified). Its `DRAFT_NOT_FROZEN` status, and the owner's decision on it (Entry 10), are unchanged.
3. **Next scientific steps (unchanged):**
   - PRISM-hardware reference recordings, which need a new reference and a new `reference_id`;
   - longitudinal user/device data for personalization;
   - ideally, documented deliberate acoustic variations as an independent test of the OUTSIDE statement.

---

# Research Entry 13 — 2026-10-07: Stage 9 Audit — Terminology Correction, Leakage Audit and Final Claim (no new experiment)

**No computation changed.**
- **Unchanged:** scoring, thresholds, the feature set, baseline construction, the assessment rules and every number in the Stage 9 outputs.
- **Corrected:** terminology in code strings, outputs and documentation.
- **Entry 12 is not edited.** Where its wording is corrected, it is quoted here.
- **The pre-registration is unchanged:** `results/final_assessment/stage9_preregistration.json`, SHA-256 `5b5cdf6e…01e6`. Its historical wording is quoted below.

## Corrections to Entry 12 (quoted; Entry 12 itself is unchanged)
- **The tail probability was mislabelled.** Entry 12 and the pre-registration describe p as a "conformal p-value", and Entry 12 adds: "Under exchangeability, P(p ≤ α) ≤ α for reference-like events". That does not describe this implementation:
  - **Scores are not exchangeable.** Each reference deviation cᵢ is computed with a different baseline (fitted without event i's own session), while a new event is scored with the baseline fitted on all reference sessions. The guarantee requires one fixed scoring rule and exchangeable scores.
  - **Events are clustered by recording session.**
  - **So there is no finite-sample guarantee.** Calibration is only checked empirically (Entry 12 §B).
- **The reference-range wording was imprecise.** Entry 12's "central-95% reference-interval convention" and the output text "central 95% reference region" describe a two-sided interval. The decision is a one-sided upper limit on a non-negative scalar deviation: the **one-sided 95% upper reference limit**.
- **The validation claim was too broad.** Entry 12 calls the assessment "validated on the reference domain". The correct scope is **held-out by recording session on the same PRISM corpus**: an internal consistency/calibration check, not external validation, and not generalization to new subjects or devices.
- **K4 deviated from the pre-registered scope.**
  - **The in-run check was narrower.** It compared the 9 evaluation CSVs produced before the gate decision, against the pre-registered "byte-identical CSV/JSON outputs". The reference, contract, golden vectors and deployment run are written after the gate decision and depend on it.
  - **Literal identity was established separately,** by comparing two further complete runs file by file (Entry 12 §E; repeated below).
- **The gate's severity was overstated.** K1 compares held-out leave-one-session-out deviations with reference deviations of the same kind from the same corpus, so near-nominal exceedance is largely expected by construction. K3 was largely implied by the Stage 8 results (gain invariance, monotone noise and tilt response). K1–K4 are therefore consistency checks, not severe tests.

## A. Empirical reference-tail probability (exact definition)
p = (1 + #{i : cᵢ ≥ a}) / (n + 1), with n = 318.
- **a:** the event's V2 aggregate acoustic deviation, sqrt(¼ Σⱼ zⱼ²), with zⱼ = (xⱼ − medianⱼ) / (1.4826·MADⱼ) under the frozen reference baseline. The deployment baseline is fitted on all 318 reference events; in evaluation, on the reference events of the other sessions.
- **cᵢ:** the reference deviations. Each of the 318 usable reference events is scored by a baseline fitted without its own recording session.

**What it is:** an empirical upper-tail probability of the reference deviation distribution, with the +1 correction.
**What it is not:** a conventional hypothesis-test p-value, an exact conformal p-value, or an exact conformal prediction. No finite-sample guarantee such as P(p ≤ α) ≤ α is claimed.
**Smallest attainable value:** 1/319.

## B. Reference threshold
- **c\*** = the 304th smallest reference deviation = **1.8297** (m = ⌊0.05 · 319 − 1⌋ = 14 reference deviations lie above it; 304/318 = 95.6% lie at or below it).
- **Decision:** `OUTSIDE_REFERENCE_RANGE` iff **a > c\***, equivalently **p ≤ 0.05**. `WITHIN_REFERENCE_RANGE` otherwise.
- **Effective attainable level:** **15/319 ≈ 0.047**, the largest attainable p at or below 0.05.
- **Geometry:** in the four-dimensional feature space, the WITHIN region is a sphere (robust-z units) around the baseline centre. The decision itself is the one-sided upper limit on the scalar deviation.
- **Category meaning:** the event's acoustic deviation is within, or outside, the empirical reference distribution established from the PRISM reference corpus.

## C. Leakage audit
**Session and event level: no leakage found.** It was checked by rebuilding all 18 held-out-session references with the production code (`prism_assessment.fit_reference`):
- **Own session excluded:** no held-out session appears in its own reference's calibration.
- **Sizes consistent:** baseline and calibration sizes equal the number of usable events outside the held-out session, in all 18 folds.
- **Mutation test:** multiplying the held-out session's V2 features by 1.5 (plus 0.01), and forcing all its events usable, left its reference bit-identical in all 18 folds. The held-out event and its session do not contribute.
- **Frozen before scoring:** each reference (frozen dataclasses) is fully built before any held-out event is scored.
- **No fitting during assessment:** `assess_recording` only applies the stored baseline, calibration scores, cut and band.

**Pipeline level: the same 318 events were reused across earlier design decisions.**

| Decision | Made on this corpus |
|---|---|
| Event detector | trained on about two-thirds of these recordings (fold membership not saved) |
| Usability rule, session gap | designed from these events and timestamps (Entries 1, 3, 4) |
| Feature selection | all 318 events (Entry 4) |
| Baseline strategy | chosen from leave-one-session-out results on these events (Entry 7) |
| V2 design | after the Stage 6 results on these events (Entry 9) |
| Stage 8 gate thresholds | partly set knowing per-feature values (Entry 10) |
| Stage 9 conventions | α 0.05 is a convention; the K2/K3 thresholds and the band and boundary-shift design were set knowing the Stage 8 distribution |
| Stage 9 alternatives | compared on the same events; V2 retained |

**Consequence:** the held-out calibration check is an internal same-corpus consistency/calibration assessment, not independent external validation.

## D. Same-day sensitivity (leave-one-date-out)
- **Result:** holding out whole recording days (10 dates; baseline and nested calibration refitted without the held-out day) gives **17/318 = 5.35%**, with a 95% date-bootstrap interval of about **[0%, 10.6%]**.
- **Concentration:** **15 of the 17** OUTSIDE events occur on **2018-05-03**. The other two are 1 on 2018-01-23 and 1 on 2018-02-06.
- **Interpretation:** this reinforces the possibility of session- or day-level acoustic context effects (subject, device, placement — unknown).

## E. STABLE / BORDERLINE (exact definition)
- **Reference-session resampling:**
  - draws: **2,000** bootstrap resamples of the 18 reference sessions (whole sessions, with replacement), **seed 20261006** (numpy `default_rng`);
  - per draw: the cut is recomputed from the resampled reference deviations with that draw's n;
  - **band:** the **5th–95th percentile** of the resampled cuts, [1.620, 1.972]; relative half-width **9.6%** (6.3–13.0% across the 18 held-out-session references);
  - **storage:** the band is stored in the reference file, so assessment involves no randomness. Refitting reproduced the stored band bit for bit (numpy 2.4.6). numpy does not guarantee its generator stream across versions, so a refit under another version could differ.
- **Boundary perturbation:** the first and/or last detector window moves by **±1 detector step = 16 ms = 128 samples**.
  - **Variants:** 8 combinations, each re-measured with the unchanged V2 code and scored with the same frozen baseline.
  - **Validity:** a variant is used when it stays inside the recording and spans at least one window. **All 318 events had all 8 variants.**
- **STABLE:** the category is unchanged across the reference-session resampling and the boundary-shift perturbations. Formally, the event's deviation and all its variant deviations lie on one side of the band: all > band high, or all ≤ band low.
- **BORDERLINE:** the category changes under at least one of them.
- **What STABLE is not:** a probability that the category is correct.
- **Counts:**
  - **OUTSIDE:** **8 of 17 are BORDERLINE**. In 5 the score lies inside the band; in 3 a boundary variant crosses the band edge. 6 of the 8 are in 2018-05-03#2.
  - **WITHIN:** 22 of 301 are BORDERLINE (14 + 8).

## F. Final results (held-out by recording session on the same PRISM corpus)
| Level | OUTSIDE | Rate [95% session-bootstrap interval] |
|---|---|---|
| 1% | 4/318 | 1.26% [0.00, 2.20] |
| 5% | 17/318 | 5.35% [0.78, 8.97] |
| 10% | 36/318 | 11.3% [3.79, 17.9] |

- **At 5%:** **14 of 17** OUTSIDE events are from two recording sessions (2018-05-03#2: 7/57; 2018-05-03#3: 7/52).
- **Under whole-day holdout:** **15 of 17** occur on 2018-05-03.
- **Calibration is marginal,** not per session (per-session rates 0–13.5%; heterogeneity p 0.058).

## G. Final scientific limitation and statement
- **V2 is not frozen.** It remains **DRAFT_NOT_FROZEN** because the pre-registered **G4c** criterion failed (Entry 10).
- **Same corpus:** **all design choices were made on this same corpus**, and **no external dataset** was used.
- **No identifiers:** **no user or device identifiers** are available; sessions are inferred recording sittings.
- **No technique labels:** **no clinical technique-quality labels** exist.
- **Not validated for new users, devices, microphones or PRISM hardware.**
- **Not a classifier:** it is not a normal/abnormal, anomaly, technique-quality or clinical classifier.

**Final statement.**

> On the PRISM reference corpus (318 scoreable inhalation events from 361 recordings in 18 inferred recording sessions), the pipeline reports for each scoreable event:
> - its V2 aggregate acoustic deviation;
> - its empirical reference-tail probability, p = (1 + #{i : cᵢ ≥ a}) / 319 over the 318 leave-one-session-out reference deviations;
> - a reference-range assessment against the one-sided 95% upper reference limit (c\* = 1.8297; effective level 15/319).
>
> Held out by recording session on the same corpus, 17 of 318 events (5.35%; 95% session-bootstrap interval 0.8–9.0%) were OUTSIDE_REFERENCE_RANGE. The rates were 1.26% and 11.3% at the 1% and 10% levels.
>
> This calibration is marginal, not per session. 14 of the 17 come from two sessions recorded on 2018-05-03, and 15 of 17 fall on that day under whole-day holdout, so an OUTSIDE assessment often reflects session- or day-level acoustic differences of unknown origin. 8 of the 17 are BORDERLINE.
>
> The V2 representation is DRAFT_NOT_FROZEN (G4c), and every design choice used this corpus. There is no external dataset, no user or device identifiers, and no clinical labels.
>
> The assessment describes acoustic deviation relative to this reference corpus only. It is not a normal/abnormal, anomaly, technique-quality or clinical classification, and it is not validated for new users, devices, microphones or PRISM hardware.

## H. Changes made (terminology and documentation only)
- **`src/prism_assessment.py`:**
  - module docstring, the α comment and the fixed `INTERPRETATION` string (the output's `interpretation` and its schema constant);
  - the `tail_probability` and `reliability` docstrings;
  - the per-event `reason` text, which now says "deviation at or above/below the one-sided 95% upper reference limit (empirical reference-tail probability …)" and takes the percentage from the reference's α;
  - `conformal_cut` renamed to `reference_cut`, with the same code.
- **`src/assessment_validation.py`:**
  - the module docstring and the call sites of `reference_cut`;
  - contract text: status meaning, representation status, pipeline description, tail-probability and cut definitions (which now state c\*, its rank and the effective level 15/319), category and STABLE/BORDERLINE meanings, and a new `category_gate.scope` field;
  - the known limitations (same-corpus reuse; not validated for new users, devices, microphones or PRISM hardware).
- **`tests/test_prism_assessment.py`:** follows the rename; the test logic is unchanged.
- **`README.md`, `ARCHITECTURE.md`:**
  - the corrected terminology;
  - validation claims qualified as "held-out by recording session on the same PRISM corpus";
  - the "(anomaly detection, Stage N)" labels of Stages 1–8 neutralised to "(Stage N)";
  - a reproduction note on clean output directories.
- **Not changed:** scoring, thresholds, the feature set, baseline construction, the assessment rules, the pre-registration, and Entry 12.

## I. Regeneration and verification
- **Runs:** run A (scratch, reference run), run B (`results/final_assessment`, K4 against A), run C (scratch, K4 against A).
- **Gate:** both B and C **PASS** (K1–K4; K4 9/9 files identical).
- **B vs C:** **28 of 28 output files are byte-identical,** including the reference, contract, schema, golden vectors, end-to-end tables and figures. `analysis_summary.json` is identical except for runtime and git provenance.
- **No number changed** between the pre-correction outputs and the regenerated ones, compared value by value with exact float parsing.
  - **The only numeric differences** are wall-clock timings.
  - **Byte-identical files:** `assessment_reference_v1.json` (SHA-256 `32ada36c…cca8`), `category_gate_results.json` and every evaluation CSV and figure.
  - **Text-only differences:** the `interpretation` constant, the per-event `reason` text, and the contract's text fields.
- **Recomputation from the saved end-to-end output:**
  - p recomputed independently: 0 mismatches;
  - p ≤ 0.05 against score > c\*: 0 mismatches;
  - category against p: 0 mismatches.

  With pandas' default CSV float parser, 100 p values differ by at most 1.1×10⁻¹⁶ (one unit in the last place, a parser artefact; 0 count mismatches). Exact parsing gives 0 differences.
- **Reference refit:** refitting the deployment reference reproduces the stored band, cut and calibration bit for bit.
- **Tests:** `venv/Scripts/python.exe -m unittest discover tests` gives 204 tests, all passing.

**Issue found during regeneration: K4 depends on stale files.**
- **What happened:** the first run B was executed into the already-populated `results/final_assessment`. The in-run K4 comparison lists every CSV/JSON present in the output directory, so it compared 22 files, including 13 left from the previous version. Their wording differed, so K4 reported FAIL and the run wrote a non-categorical reference.
- **Recovery:** the directory was emptied (except the pre-registration) and run B repeated, which gave the PASS reported above. The stale-file run's outputs were discarded.
- **What the original Stage 9 run did:** it wrote into a fresh directory, so its 9-file K4 was correct.
- **Status:** the code is unchanged in this entry (terminology-only scope). The README now requires empty output directories.
- **Recommended code fix:** compare only the files written by the current run, or refuse a non-empty output directory.

---

# Research Entry 14 — 2026-10-07: Stage 9 Reproducibility Fix — K4 Output-Directory Guard (no methodological change)

**This changes infrastructure only.** The following are unchanged:
- scoring, baseline, V2 features and thresholds;
- assessment categories and STABLE/BORDERLINE logic;
- reference construction;
- the pre-registration (SHA-256 `5b5cdf6e…01e6`);
- every numerical result.

## Problem (Entry 13 §I)
- **What K4 does:** the in-run K4 check (`compare_runs`) compares every CSV/JSON present in the output directory with the reference run.
- **The flaw:** a run into a populated directory therefore included files left by an earlier run. They could make K4 fail spuriously, which made the run write a non-categorical reference. In principle, a stale file that happened to match could also make the comparison cover files the current run did not produce.

## Fix
- **Guard:** `assessment_validation.require_clean_output_dir(output)` is called at the start of `run_analysis`, before anything is read, computed or written.
  - **Rule:** it raises `StageError` unless the output directory is empty, with `stage9_preregistration.json` the only file allowed.
  - **Error message:** it names the leftover entries in sorted order and says to remove them or choose another `--output-dir`.
- **Why this design:** refusing was chosen over "compare only the files produced by this run", because it is simpler and cannot silently miss or include a stale file.
- **Documentation:** the CLI help and the README reproduction note were updated.

## Tests
Five new tests in `tests/test_assessment_validation.py`:

| Test | What it checks |
|---|---|
| `test_stale_file_would_contaminate_k4` | documents the failure mode (a stale file enters K4) and that the guard refuses that directory |
| `test_empty_or_preregistration_only_directory_is_accepted` | an empty directory, and one holding only the pre-registration, are accepted |
| `test_any_leftover_file_or_folder_is_refused` | a leftover CSV, folder, text file or hidden file is refused |
| `test_error_is_deterministic_and_names_the_leftovers` | the message is identical across calls and lists `a.json, b.csv, golden/` in sorted order |
| `test_run_analysis_refuses_before_writing_anything` | the guard is wired into `run_analysis`, and the directory is unchanged afterwards |

**Mutation check:** with the guard removed in memory, the wiring scenario fails with `FileNotFoundError` instead of `StageError`, so the test depends on the guard.
**Full suite:** 209 tests (204 + 5), all passing.

## Demonstration
Running Stage 9 into the populated `results/final_assessment` now stops immediately:
- **Error raised:** "output directory … is not empty: analysis_summary.json, … (21 entries) …".
- **Directory afterwards:** byte-identical; nothing was written.

## Regeneration from clean directories and verification
- **Procedure:** `results/final_assessment` was emptied (except the pre-registration). Then three runs:
  - run A into an empty scratch directory (reference run; gate INCOMPLETE by design);
  - run B into `results/final_assessment`, with K4 against A;
  - run C into an empty scratch directory, with K4 against A.
- **Gate:** runs B and C **PASS** (K1–K4); K4 found 9/9 files identical.
- **B vs C:** **all 28 output files are byte-identical.** `analysis_summary.json` differs only in `runtime_seconds`.
- **Unchanged from the pre-fix outputs:** all 28 output files are byte-identical to the Entry 13 outputs, apart from the wall-clock fields in `e2e_timing.csv` and `analysis_summary.json`. **0 numerical changes, 0 text changes.**
- **Recomputation from the saved end-to-end output (exact float parsing):**
  - 0 p mismatches;
  - 0 mismatches between p ≤ 0.05 and score > c\*;
  - 0 category mismatches;
  - still 17 OUTSIDE, 8 of them BORDERLINE.
- **Reference:** the stored band [1.620, 1.972], cut 1.8297 and calibration reproduce bit for bit. The reference SHA-256 is unchanged (`32ada36c…cca8`).

## Status
The output-directory contamination issue is resolved: a Stage 9 run can only write into a directory that contains nothing but the pre-registration, so K4 compares exactly the files produced by the current run.
