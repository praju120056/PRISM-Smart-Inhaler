# Full pipeline run: `rec2018-01-22_17h41m49.809s.wav` (2026-10-06)

This is an engineering reference run for app-team synchronisation. It is not a research result. It adds no threshold and no new evidence: the same recording was already reproduced exactly by Stage 8 (gate G6b, `results/v2_validation/reproduction_events.csv`). Every stage a mobile re-implementation has to reproduce is saved, so the app can be checked at each boundary, not only at the final output.

Reproduce (about 30 s):

```
venv/Scripts/python.exe results/recording_runs/run_recording.py data/rec2018-01-22_17h41m49.809s.wav --input-domain reference_dataset
```

The script works on any 8 kHz WAV and writes to `results/recording_runs/<wav stem>/`.

## Input
| | |
|---|---|
| File | `data/rec2018-01-22_17h41m49.809s.wav`, SHA-256 `72eebb44…56b5` |
| Format | 8 kHz, mono, 96,000 samples (12.0 s), peak 0.855 |
| `recorded_at` | `2018-01-22T17:41:49.809` (parsed from the file name; on the app this is supplied by the caller) |
| Domain | `reference_dataset`, so `baseline_domain_validated = true` |
| Session | `2018-01-22#1` (Stage 2 session definition) |

## The app-facing output (`contract_output.json`)
Produced by `prism_inference.analyze_recording`, unchanged. The plain CLI (`src/prism_inference.py`) gives the same output except `recorded_at`, which the CLI leaves `null`.

- `contract_version`: `prism-inference-v2.0`
- `baseline_id`: `prism-v2-global-2026-09-30`
- `recording_status`: **`EVENTS_DETECTED`**
- events: `n_events = 1`, `n_scored = 1`

| Field | Event 0 |
|---|---|
| start_time / end_time | 0.688 s / 2.232 s |
| duration_s | 1.544 |
| detector_confidence (mean P(Inhale)) / max | 0.934 / 0.987 |
| window_count | 85 |
| status | **`SCORE_ONLY`** (no `not_scoreable_reasons`) |
| anomaly_score | **0.555** |
| feature_values (centroid_mean, flatness_mean, centroid_std, rolloff_std) | 0.37868, 0.15680, 0.039247, 0.069183 |
| feature_z_scores (same order) | +0.293, +0.827, −0.017, −0.682 |
| mean_rms (level channel, not in the score) | 0.1775 |

**Context for 0.555 (not a threshold).** On unseen sessions, the 318 usable reference events score 0.42 / 1.00 / 1.75 (5th percentile / median / 95th percentile, held-out V2 rms_z, Stage 8).

- **This event was in the baseline fit.** The deployment baseline was fitted on all 318 usable events, including this one, so its contract score is in-sample.
- **Held-out score:** refitted without its own session (309 events from 17 sessions), the event scores **0.571**.
- **Cross-check:** the held-out z-scores agree with the Stage 5 `C_loso_global` row for this event to 6 digits.

## What else the pipeline sees (descriptive only, not part of the contract)
| | Detector (window argmax, grouped with the Inhale rule) | Dataset annotation | Annotated windows with the same label |
|---|---|---|---|
| Inhale | 0.688–2.232 s (the contract event) | 0.630–2.165 s | 85 / 96 (the other 11 are Noise: 9 at the start, 2 at the end) |
| Drug | 2.176–3.144 s | 2.286–3.046 s | 48 / 48 |
| Exhale | 0.000–0.296 s and 3.840–5.944 s | 3.927–5.841 s | 119 / 119 |

- **Window labels overall:** Noise 64.7%, Exhale 17.2%, Inhale 11.5%, Drug 6.6% of 739 windows.
- **Event match:** the contract event overlaps the annotated Inhale with IoU 0.92.
- **Segments can overlap.** Windows are 0.2 s long with a 16 ms step, so segments of different classes can overlap; here Inhale ends at 2.232 s and Drug starts at 2.176 s.
- **Agreement may be in-sample.** The detector was trained on about two-thirds of the reference recordings, and which ones was not saved, so these agreement numbers may be in-sample and say nothing about accuracy on new audio.
- **No Drug or Exhale events in the contract.** It defines no Drug or Exhale events, no coordination delay and no technique judgement. The table above describes the detector's raw stream only.

See `diagnostic.png`: waveform with annotation bars, the CNN label stream, the RMS envelope and the contract event (green band).

## Stage checkpoints for app parity
| Stage | File | What the app compares |
|---|---|---|
| 1. Input | `input.json` | sample rate 8000, 96,000 samples, float = int16 / 32768 |
| 2. DSP | `frame_features.csv` | 1,501 frames × 124 features (`mfcc_0..39, dmfcc_0..39, ddmfcc_0..39, spectral_centroid, spectral_flatness, spectral_rolloff, zcr`); README target MAE < 0.001 |
| 3. Detector | `window_predictions.csv` | 739 windows (25 frames, stride 2): raw ONNX logits, softmax and argmax label; window i spans [0.016·i, 0.016·i + 0.2] s, capped at the recording end |
| 4. Contract | `contract_output.json` | golden tolerances: structure exact, confidence ±1e-4, features ±1e-4 relative, z and score ±0.01 (`results/v2_validation/golden/golden_manifest.json`) |

The nine golden cases cover the final output for the other recording states: `NO_INHALATION_DETECTED`, the `short_duration`, `close_neighbor` and `recording_boundary` reasons, two synthetic inputs and two input errors. This recording adds the intermediate checkpoints for one scored event.

## Spec mismatches the app team should know about
`results/v2_validation/inference_contract_v2.json` and `src/` are authoritative. Before its 2026-10-06 revision, `ARCHITECTURE.md` (§3, §6, §8.6, §9, §10, §11) described a design that was not implemented or not supported by evidence. The points below were checked against the code and librosa 0.11.0, and `ARCHITECTURE.md` has since been corrected to match (see its revision summary and `PRISM_RESEARCH_LOG.md` Entry 11).

**DSP: what the trained detector actually consumes (`librosa_extractor.extract_features_from_audio`)**
- **No pre-emphasis.** §8.6 step [1] shows `y[n] = x[n] − 0.97·x[n−1]`, but the extractor does not apply it.
- **MFCCs use an 80 dB floor across the whole buffer.** `librosa.feature.mfcc` applies `power_to_db` with `top_db = 80` to the slaney-normalised 128-band mel power spectrogram. The floor is the loudest value in the whole input buffer minus 80 dB, so the app must extract features over the same full buffer and not in independent streaming chunks.
- **Deltas use librosa's Savitzky–Golay filter.** It runs with width 9 and `mode = "interp"`. The interior matches §8.6's Σn·Δ/60, but the edge frames differ.
- **Flatness uses the power spectrum.** It is computed on |STFT|² (`power = 2.0`), not on the magnitude as §8.6 shows.
- **ZCR uses its own framing.** Frames are 2048 samples long with hop 64, centred with edge-value padding, not the 256-sample STFT frame.

**Events.**
- **Contract rule:** an event starts at the first Inhale window's start and ends at the latest Inhale window's end. Overlapping Inhale windows merge into one event.
- **§8.6 step [7] is a different rule.** Its frame majority vote is a different segmentation and will not reproduce the contract's event boundaries.

**Forbidden derived outputs.** None of the following has a validated basis:
- the §8.6 [8]–[10] analytics: `insufficient_inhale` (< 1.0 s), `late_actuation` (> 0.5 s), `coord_delay` and `missed_dose`;
- the composite labels `GOOD`, `POOR`, `GOOD_BUT_INCONSISTENT`, `ABNORMAL` and `MISSED_DOSE`;
- the §9 deviation bands 1.5 and 3.0, the Mahalanobis/EMA baseline and the quality-gated update;
- the §10 `quality_assessment` block (`global_score`, `composite_label`, `deviation_flag`).

The contract forbids NORMAL/ANOMALY, quality percentages, Correct/Incorrect and technique error codes.

**Baseline.**
- **What is implemented:** a frozen global median / 1.4826·MAD on 4 spectral features (`v2_baseline.json`). It is loaded, never fitted, on the device.
- **What is not:** no per-user baseline exists yet (V3 is planned).

## What the app can show from this output
- **Safe to show:** `recording_status`; the event times and duration; `status` and `not_scoreable_reasons` (e.g. "inhalation detected, too short to analyse").
- **Internal or experimental only:** `anomaly_score` and `feature_z_scores`. The contract is `DRAFT_NOT_FROZEN` (gate criterion G4c failed), so the score may change and must not be shown to users as a health signal.
- **Diagnostics only:** `mean_rms` is uncalibrated loudness.
- **Not a problem with the inhalation:** `NO_INHALATION_DETECTED` is a recording state, never an anomalous inhalation.

## Limits of this run
- **One recording from the reference dataset.** It is in-domain by construction; PRISM hardware audio (INMP441, enclosure, 5 s buffer) is unvalidated, and the contract marks it `baseline_domain_validated = false`.
- **Clip length differs.** This clip is 12 s; the hardware buffer is 5 s, where events near the clip edges become `NOT_SCOREABLE` (`recording_boundary`).
- **No technique ground truth.** The score is a distance from a dataset baseline, not a technique-quality measure.

## Files (`rec2018-01-22_17h41m49.809s/`)
| File | Content |
|---|---|
| `contract_output.json` | Contract output; the app-facing result |
| `input.json` | Input identity and basic level statistics |
| `frame_features.csv` | Stage 2 DSP checkpoint (124 features per 8 ms frame) |
| `window_predictions.csv` | Stage 3 detector checkpoint (logits, probabilities, label per window) |
| `summary.json` | Machine-readable summary, including the descriptive comparisons above |
| `diagnostic.png` | Waveform, annotation, CNN label stream, RMS envelope, contract event |
