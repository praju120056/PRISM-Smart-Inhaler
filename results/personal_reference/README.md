# PRISM V3 Personal Reference (Stage 10) — contract `prism-personal-reference-v3.0-draft`

**Status: IMPLEMENTED, mechanically verified on PROXY/SYNTHETIC data only. NOT VALIDATED as personalization.**

The PRISM corpus has no user or device identifiers, no longitudinal personal labels and no technique labels.
Nothing in this directory shows that the personal baseline represents a person, or that the personal
deviation detects a real change in a person's inhalations. See `PRISM_RESEARCH_LOG.md` Entry 15.

## What V3 is

V3 adds a **separate, independent Stage 10 channel** next to the Stage 9 assessment. It does not replace or change Stage 9, which remains unchanged.

| Channel | Source | Adapts? | Output |
|---|---|---|---|
| Population | Stage 9 `prism_assessment.assess_recording` (`prism-assessment-v1.0`), passed through unchanged | Never | `WITHIN` / `OUTSIDE_REFERENCE_RANGE`, tail probability, `STABLE` / `BORDERLINE` |
| Personal | `src/personal_reference.py` (this contract) | Slowly, per sitting, after assessment | continuous `personal_deviation`, or `POPULATION_ONLY` during warm-up |

One call gives both channels:

```python
from personal_reference import PersonalReferenceStore, assess_recording_with_personal, load_config
from prism_assessment import load_reference

reference = load_reference()                        # frozen Stage 9 population reference
store = PersonalReferenceStore("<state dir>")       # one JSON state per (user_id, device_id)
personal = store.load_or_create(user_id, device_id, load_config(), reference.reference_id, recorded_at)
output = assess_recording_with_personal(audio, 8000, reference, personal, recorded_at=recorded_at, detector=detector)
store.save(personal)
# output = {"contract_version", "population": <Stage 9 output, byte-identical>, "personal": {...}}
```

or, from the command line:

```bash
venv/Scripts/python.exe src/personal_reference.py data/<recording>.wav --user <id> --device <id> \
    --recorded-at 2026-01-05T08:00:00 --state-dir <state dir> --input-domain reference_dataset
```

## Coordinates and personal baseline

- Every scoreable event is represented in the **frozen Stage 9 population-reference coordinates**:
  z = (x − m0) / s0, with m0 and s0 the Stage 9 V2 median and 1.4826·MAD (`assessment_reference_v1.json`).
- Features: `spectral_centroid_mean`, `spectral_flatness_mean`, `spectral_centroid_std`, `spectral_rolloff_std`.
  `mean_rms` is not used and not adapted. The scale s0 is never adapted.
- The personal baseline B ∈ R⁴ is kept **per user AND per device**, and starts at **B₀ = 0** (the population median).
- **Personal deviation** of an event: a_u = sqrt(mean_j (z_j − B_j)²), the same aggregate as Stage 9, with B taken
  from **before the event's sitting**.

## Sittings

- A sitting is a run of recordings of one user/device at most `sitting_gap_minutes` (25 min, the Stage 2 rule) apart.
- A sitting **qualifies** if it has at least `min_events_per_sitting` (10) scoreable events. Non-scoreable events are
  never assessed or buffered.
- A qualifying sitting is summarised by its per-feature median x_k and counts as **one** unit of evidence; its
  events are not independent votes.

## Assess before adapt (ordering)

1. When a sitting opens, the state (B, consensus window, counters) is snapshotted; its ID is `state_id_used`.
2. Every event of every recording is assessed against that snapshot and the output is emitted.
3. Only then are the recording's events added to the sitting buffer.
4. When the sitting closes (next recording more than 25 min later, or `close_sitting()`), the update is decided
   and applied, and logged in the audit.

B cannot change while a sitting is open, so the current sitting never influences the baseline used to assess it
(tested by a mutation test).

## Update rule (per qualifying sitting k, per feature j)

```
window     = the last K = consensus_window qualifying sittings (including k), cleared by a gap > max_gap_days
direction  = +1 if >= k_req of the window medians are > B_kj, -1 if >= k_req are < B_kj, else 0
gate g_kj  = 1 iff the window holds exactly K sittings, direction != 0, and sign(x_kj - B_kj) == direction
SE_kj      = sqrt(pi/2) * event_sd_j / sqrt(n_k)                  (asymptotic SE of a median, population units)
w_kj       = g_kj * min(1, c * SE_kj / |x_kj - B_kj|)             (w = g when x_kj = B_kj; then the step is 0)
B_k+1      = Pi_rho(B_k + eta * w_k * (x_k - B_k))
Pi_rho(b)  = b * rho / rms(b) if rms(b) > rho else b              (radial projection onto the trust region)
```

| Parameter | Primary (`personal_reference_config_v3.json`) | Sensitivity (`..._8of10.json`) |
|---|---|---|
| K (`consensus_window`) / k_req (`consensus_required`) | 10 / 9 | 10 / 8 |
| eta | 0.3 | 0.3 |
| c (`clip_c`) | 2.0 | 2.0 |
| event_sd | (1, 1, 1, 1) population-scale units | same |
| rho (`trust_radius`, on rms(B)) | 0.9 | 0.9 |
| G_max (`max_gap_days`) | 14 days | 14 days |
| S_min (`s_min`) | 10 qualifying sittings | 10 |
| n_min (`min_events_per_sitting`) | 10 events | 10 |
| sitting gap | 25 min | 25 min |

These are **provisional implementation parameters**, not validated values.

## Guarantees (mechanical, tested)

- **One sitting:** rms(B_k+1 − B_k) ≤ eta · c · rms(SE_k) ≤ 0.3 · 2 · sqrt(pi/2) / sqrt(10) = **0.238** for any
  sitting with ≥ 10 events, whatever its values (Euclidean bound; the projection is non-expansive). A single
  sitting can also never open the gate: the window must be full and k_req ≥ 2.
- **One event:** moves its sitting median by at most one order-statistic gap, independent of its magnitude, and
  cannot change any assessment of its own sitting.
- **Trust region:** rms(B) ≤ rho at all times. Every capped update is logged (`trust_region_capped`), and the
  output reports `at_trust_region_boundary`. Only `reenroll()` with an explicit `trust_radius` can place B beyond
  the configured radius.
- **Long gap:** a gap > G_max between qualifying sittings clears the consensus window, so stale evidence cannot
  open the gate; `reference_stale` reports a current gap > G_max.
- **Convergence:** under a sustained, consistent shift the gate keeps opening, so B moves by bounded steps until
  it reaches the shift or the trust-region boundary. The gate is directional, not "close to the baseline", so a
  persistent shift is never permanently ignored.

## Warm-up

While fewer than S_min qualifying sittings exist (since creation, reset or re-enrollment), `personal_status` is
`WARMUP`: events get `personal_assessment = POPULATION_ONLY` and `personal_deviation = null`. The population
channel is always available.

## Reset and re-enrollment

- `reset(reason, at)`: B = 0, configured trust radius, consensus window cleared, back to `WARMUP`.
- `reenroll(enrollment_z, reason, at, trust_radius=None)`: B = per-feature median of ≥ 10 supervised enrollment
  events; beyond the configured radius only with an explicit `trust_radius`. Back to `WARMUP`.

Both require no open sitting and are written to the audit log.

## State and audit

Each state file holds the configuration, B, trust radius, baseline version, counters, consensus window (sitting
summaries), last qualifying sitting, open sitting (its buffered z vectors), `state_id` and the **audit log**.

The audit log is a SHA-256 hash chain with one entry per initialisation, closed sitting (qualifying or not),
reset and re-enrollment. Each qualifying-sitting entry records the sitting median, n, timestamps, window counts,
direction, gate, SE, clipping, weight, step, proposed B, trust-region capping, B before/after and state IDs.
`PersonalReference.replay(audit)` rebuilds the state from the log alone and checks every entry bit for bit.
Loading a state verifies the chain and the replay.

## Output (`personal`)

Keys: `contract_version, config_id, config_sha256, user_id, device_id, population_reference_id, recording_id,
recorded_at, sitting_index, closed_sitting, personal_status, state_id_used, baseline_version_used, baseline_used,
n_qualifying_sittings, s_min, consensus_window_size, trust_radius, trust_region_distance,
at_trust_region_boundary, reference_stale, n_events, n_personal_assessed, events, interpretation`.

Each event has `event_id, personal_assessment (POPULATION_ONLY | PERSONAL_DEVIATION | NOT_ASSESSED),
personal_deviation, personal_feature_deviations`.

## Meaning

`personal_deviation` measures deviation from the personal reference: consistency with this user's own earlier
qualifying sittings on this device. It is **continuous**: no personal cut-off has been calibrated, so no
categorical personal statement is made.

The personal baseline is **not** a healthy, normal or correct-technique baseline. V3 does not identify correct or
incorrect technique. A sustained acoustic change is absorbed by B whatever its cause (behaviour, noise,
microphone, device, environment); the mechanism cannot tell these apart. The frozen population channel is kept
next to it so that such absorption never hides a deviation from the population reference.

## PROXY/SYNTHETIC evaluation (`src/personal_reference_evaluation.py`)

**This is not personalization validation.** It shows how the mechanism behaves on constructed data.

```bash
venv/Scripts/python.exe src/personal_reference_evaluation.py --output-dir <empty dir>   # README.md may be present
```

**Proxy users.**
- **Length:** 80 sittings, one per day.
- **Sittings:** each is a bootstrap resample, of its own size, of one randomly chosen real inferred sitting with ≥ 10 scoreable events. There are 8 such Stage 2 sessions, of 11–57 events.
- **Coordinates:** the Stage 9 deployment z-values. They are in-sample: the reference was fitted on the same 318 events.
- **Replicates:** 200 per scenario and configuration, seed 20261007, paired across scenarios.
- **Change onset:** sitting 30.
- **What the variation means:** the sitting-to-sitting variation of a proxy user is the variation between real inferred sittings, whose cause (person, device, day or environment) is unknown.

**Scenarios.** Their ground truth comes from the construction.

| | Scenario |
|---|---|
| N0 | no change |
| A1 / A2 | one extreme event (z = +1000) in a no-change stream, or in the +1.0 shift while the gate is open |
| B1 / B2 / B3 | five isolated extreme events; one sitting replaced by its 10 dB-noise version; one sitting of extreme events during the +1.0 shift |
| C / D / E | persistent +0.5 / +1.0 / +2.0 on every feature |
| F | sustained 20 dB white noise (Stage 6/8 perturbation) |
| G / G2 | sustained spectral tilt a = 0.9 / 0.5 (microphone-like) |
| H | 20 dB noise for 3, 5 or 9 sittings only |
| I | D with a 16-day interval (> G_max = 14 days) before shifted sitting 6 |
| J | Jan–Feb 2018 sittings, then May 2018 sittings (a distribution switch of unknown cause) |

**Results** (median [5th–95th percentile] over 200 replicates; primary 9-of-10 / sensitivity 8-of-10; `proxy_summary.csv`).

| Property | 9 of 10 | 8 of 10 |
|---|---|---|
| Largest single-update step / theoretical bound (all 7,200 streams) | 1.0000000000000004 (≤ 1 up to rounding) | same |
| rms(B) never above rho = 0.9 | max 0.9000000000000002 | same |
| No change (N0): updates per sitting after onset | 0.062 | 0.227 |
| No change: final rms(B) | 0.080 [0.001, 0.193] | 0.124 [0.053, 0.248] |
| A1 one extreme event: runs with any influence / largest influence | 3.5% / 0.104 | 12.5% / 0.171 |
| A2 one extreme event, gate open: any influence / largest | 11.5% / 0.106 | 21% / 0.111 |
| B1 five isolated extreme events: any influence / 95th pct / largest | 19.5% / 0.050 / 0.104 | 59% / 0.091 / 0.115 |
| B2 one 10 dB-noise sitting: any influence / 95th pct / largest | 34% / 0.104 / 0.154 | 74% / 0.133 / 0.246 |
| B3 one extreme sitting, gate open: median / 95th pct / largest; at end, 95th pct | 0.031 / 0.257 / 0.414; 0.103 | 0.061 / 0.168 / 0.334; 0.062 |
| C +0.5: share absorbed after 50 sittings (reachable 1.0) | 0.73 [0.53, 1.00] | 0.93 [0.71, 1.19] |
| C: sittings to 50% / 90% (runs reaching it) | 23.5 (192/200) / 32 (25/200) | 12 (200) / 28 (146) |
| D +1.0: share absorbed (reachable 0.90 because of the trust region) | 0.835 [0.67, 0.92] | 0.884 [0.79, 0.99] |
| D: sittings to 50% / 90% of reachable | 13 / 34 (129/200) | 11 / 21 (198/200) |
| E +2.0: absorbed (reachable 0.45); sittings to 50% / 90%; at boundary | 0.447; 10 / 13; 100% | 0.450; 9 / 11; 100% |
| F 20 dB noise (shift rms 1.72): absorbed (reachable 0.525); 50% / 90%; at boundary | 0.517; 13 / 18; 100% | 0.511; 11 / 16; 100% |
| G tilt 0.9 (rms 2.01): absorbed (reachable 0.448); 50% / 90%; at boundary | 0.441; 12 / 17; 100% | 0.452; 11 / 16; 100% |
| G2 tilt 0.5 (rms 1.41): absorbed (reachable 0.639); 50% / 90%; at boundary | 0.633; 12 / 18; 99.5% | 0.650; 10 / 15; 99% |
| H temporary noise, 3 / 5 / 9 sittings: largest influence, median | 0.050 / 0.072 / 0.242 | 0.115 / 0.172 / 0.404 |
| H 3 / 5 / 9: influence at end, median [95th pct] | 0.013 [0.093] / 0.041 [0.11] / 0.098 [0.18] | 0.028 [0.094] / 0.041 [0.10] / 0.057 [0.14] |
| I long gap: gap resets; sittings to 50% (D without gap) | 200/200; 21 (13) | 200/200; 19 (11) |
| J switch (shift rms 0.47): absorbed; sittings to 50% (runs reaching) | 0.43 [0.03, 1.13]; 28 (97/200) | 1.02 [0.49, 1.31]; 19 (190/200) |

**Both channels during sustained shifts** (median over the last 10 sittings; before the onset: population OUTSIDE 3.7%, personal deviation 0.90):

| Scenario | Population OUTSIDE share | Personal deviation (9 of 10) |
|---|---|---|
| D +1.0 | 17.7% | 0.94 |
| F 20 dB noise | 62.6% | 1.41 |
| G tilt 0.9 | 79.7% | 1.52 |

The personal channel absorbs what lies inside the trust region. The frozen population channel keeps reporting the deviation from the population reference.

**Real-corpus replay** (all 361 recordings in time order as one pseudo-user/device):
- 23 sittings, matching the Stage 2 sessions one to one.
- Only **8** sittings qualify (50, 27, 29, 27, 13, 11, 57 and 52 events).
- The personal channel therefore **never leaves WARMUP**: B stays 0, 318 events are `POPULATION_ONLY` and 46 are `NOT_ASSESSED`.
- The real corpus cannot exercise adaptation at all.

**Reproducibility.**
- Two independent runs gave byte-identical outputs: every CSV, JSON and PNG. `analysis_summary.json` differs only in `runtime_seconds`.
- The inputs (the Stage 9 deployment outputs, the Stage 9 reference and the Stage 8 perturbed features) were hashed before and after the run.

## Known failure cases and limitations

1. **Sustained condition changes are absorbed.**
   - Noise and spectral tilt drive B to the trust-region boundary within about 12–18 sittings in 99–100% of runs. The mechanism cannot tell behaviour from noise, microphone or device.
   - The trust region and the frozen population channel limit this; they do not prevent it.
2. **Distribution switches of unknown cause are partly absorbed (J).** A user or device switch would look the same.
3. **Temporary changes are not fully blocked.**
   - 3 or 5 noisy sittings still move B in 63% / 82.5% of runs (9 of 10), though only a little (end of stream, 95th percentile ≤ 0.11).
   - A 9-sitting episode reaches the consensus threshold and moves B by design, after which B partly returns.
4. **The single-sitting bound covers one update, not the total downstream influence.**
   - A sitting stays one (magnitude-free) vote in the window for the next 9 gate decisions.
   - In B3 the trajectory divergence reached 0.414 (95th percentile 0.257) against the 0.238 per-update bound.
5. **No-change adaptation.** Real sitting-to-sitting variation sometimes passes the gate: 6.2% of sittings (9 of 10) and 22.7% (8 of 10). B wanders up to rms 0.30 / 0.41.
6. **Small shifts are slow under 9 of 10.** A +0.5 shift reached 90% within 50 sittings in only 25 of 200 runs.
7. **Practicality.** ≥ 10 scoreable inhalations per sitting comes from the corpus's recording sessions. Real inhaler use gives one or two inhalations per dose, so the sitting definition must change before any deployment. This has not been evaluated.
8. **Same corpus.** The proxy users reuse the 318 events the Stage 9 reference was fitted on, and the sittings are inferred, not user-labelled.
