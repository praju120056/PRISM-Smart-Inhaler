# PRISM — Agent Instructions

## 0. Mission

PRISM (Pulmonary Response & Inhaler System Monitor) is a research project, not merely a software project.

The end goal is to produce a technically sound, reproducible, academically defensible proof-of-concept for:

1. acoustic inhalation-event detection;
2. extraction and characterization of detected inhalation events;
3. personalized baseline modeling of inhalation behavior;
4. detection of deviations from that personal baseline;
5. rigorous evaluation of the resulting anomaly-detection pipeline;
6. a reproducible body of evidence that can later support a research paper / thesis.

Optimize work toward this end goal.

Do not optimize for adding features, increasing model complexity, or making the project look more impressive unless those changes are supported by an actual research need.

---

# 1. CURRENT RESEARCH DIRECTION

The current research pipeline is:

    Audio
      ↓
    Event Detection
      ↓
    Inhale Event Extraction
      ↓
    Feature Extraction
      ↓
    Personalized Baseline
      ↓
    Deviation / Anomaly Score
      ↓
    NORMAL / ANOMALY

The terms have the following operational meanings:

- NORMAL = consistent with the learned personal baseline.
- ANOMALY = sufficiently deviates from the learned personal baseline.

These labels do NOT mean:

- NORMAL = medically correct breathing
- ANOMALY = medically incorrect breathing
- ANOMALY = clinically poor inhaler technique

Do not introduce clinical interpretation unless appropriate ground-truth data exists.

---

# 2. PRIMARY CURRENT OBJECTIVE

The existing event-classification system is substantially established.

The major unfinished research component is:

    Inhale event
      → feature vector
      → personal baseline
      → anomaly score
      → threshold
      → evaluation

Prioritize this component before adding unrelated functionality.

The immediate research phases are:

### Phase A — Dataset → Inhale Events
Verify the existing event-detection/post-event pipeline and produce the inhalation-event dataset used by the anomaly experiments.

### Phase B — Feature Representation
Determine which event-level features are appropriate for personalized baseline modeling.

### Phase C — Baseline Model
Implement a simple, interpretable personalized baseline.

Start with statistically defensible methods before considering complex ML.

### Phase D — Anomaly Scoring
Define and implement a reproducible deviation/anomaly score.

### Phase E — Threshold Selection
Determine thresholds using an explicit validation procedure.

Do not choose thresholds merely because they produce attractive results.

### Phase F — Evaluation
Evaluate:

- training/calibration-set baseline consistency;
- held-out baseline-consistent samples where possible;
- controlled deviations;
- anomaly-score separation;
- threshold sensitivity;
- computational cost.

### Phase G — Integration
Integrate the validated anomaly component with the existing event pipeline.

### Phase H — Documentation
Update architecture, README, research log, and reproducibility information only after the implementation/evidence exists.

---

# 3. EXISTING ESTABLISHED COMPONENTS

Treat the following as existing components unless the user explicitly asks to modify them.

## Event Classifier

The project contains an existing 1D CNN event classifier exported as:

    results/inhaler_cnn.onnx

ONNX contract:

    Input:  float32[batch, 25, 124]
    Output: float32[batch, 4]

Class order:

    0 = Drug
    1 = Exhale
    2 = Inhale
    3 = Noise

Do not silently change this contract.

Do not retrain or replace the event classifier merely to experiment with the anomaly-detection problem.

If a modification to the event classifier is scientifically necessary, isolate it as a separate experiment and preserve the previous implementation/results.

---

# 4. DATASET GROUND-TRUTH RULE

The available annotations describe acoustic events.

They do NOT provide validated clinical inhalation-technique labels.

Therefore:

- Do not relabel existing samples as "good" or "bad" breathing without independent ground truth.
- Do not treat "Inhale" as synonymous with "good inhalation."
- Do not treat "non-Inhale" as synonymous with "bad inhalation."
- Do not report "technique accuracy" unless a valid technique ground truth exists.
- Do not fabricate labels for the purpose of obtaining accuracy/F1.

The current anomaly-detection framing is personalized deviation detection, not clinical classification.

---

# 5. RESEARCH INTEGRITY

Every research claim must be traceable to one of:

1. an observed experiment;
2. a documented dataset property;
3. an explicit mathematical/modeling assumption;
4. external literature, clearly identified as such.

Never fabricate:

- results;
- metrics;
- sample counts;
- labels;
- experiments;
- clinical conclusions;
- model performance.

If an experiment has not been run, say that it has not been run.

If a result is preliminary, label it preliminary.

If an interpretation is speculative, label it as a hypothesis.

Never silently convert an assumption into a fact.

---

# 6. EXPERIMENTAL DISCIPLINE

For every meaningful experiment:

1. Define the research question.
2. State the hypothesis or expected behavior.
3. Identify the data used.
4. Identify the exact preprocessing.
5. Identify model/algorithm parameters.
6. Run the experiment.
7. Save reproducible outputs.
8. Record numerical results.
9. Interpret the results.
10. Record limitations.
11. Record the resulting decision.
12. Append a research-log entry.

Prefer experiments that answer a specific question over experiments that merely generate more metrics.

---

# 7. DATA LEAKAGE / EVALUATION RULES

Avoid leakage between calibration/training and evaluation.

Whenever possible:

- split at the recording/session/user level rather than randomly splitting highly correlated windows;
- never let a test sample influence the baseline used to evaluate that same sample;
- do not tune thresholds on the final evaluation set;
- preserve a clear distinction between calibration, validation, and final evaluation data.

If the available dataset structure prevents a clean split, document the limitation explicitly rather than hiding it.

---

# 8. ANOMALY-DETECTION EVALUATION

The following distinction is mandatory.

## Training / Calibration Anomaly Rate

It is valid to calculate:

    anomalous calibration samples / total calibration samples

This measures how internally self-consistent the learned baseline is.

Call it something such as:

- Training Anomaly Rate
- Calibration Anomaly Rate
- Baseline Self-Consistency Rate

Do NOT call it:

- accuracy;
- false-positive rate;
- clinical error rate.

Because the calibration samples do not have independent anomaly ground truth.

## Held-Out Baseline Samples

When independent baseline-consistent samples are available, evaluate how often the learned baseline flags them as anomalous.

This is a stronger generalization check.

## Controlled Deviations

Where possible, create a separately defined set of controlled deviations.

The deviation-generation procedure must be documented.

Do not present controlled deviations as clinical "bad technique" unless clinically validated.

Metrics may include:

- deviation detection rate;
- precision;
- recall;
- F1;
- ROC-AUC / PR-AUC where appropriate;
- anomaly-score separation.

The ground truth for controlled deviations must come from the experimental construction, not from the model itself.

---

# 9. MODELING PRINCIPLES

Prefer the simplest model that answers the research question.

Start with interpretable statistical baselines such as:

- mean/std;
- median/MAD;
- standardized feature deviations;
- distance-based scores.

Do not introduce neural networks, autoencoders, isolation forests, one-class SVMs, or other complex anomaly models unless a simpler baseline has been implemented and evaluated first.

If a more complex method is tested, compare it against the simpler baseline.

Complexity must earn its place through measurable improvement.

---

# 10. PERSONALIZATION

PRISM evaluates a user relative to their own historical behavior.

Do not assume a single population-wide baseline is equivalent to personalization.

Baseline construction must explicitly define:

- calibration samples;
- feature set;
- baseline statistics;
- adaptation/update policy;
- anomaly threshold;
- handling of insufficient calibration data;
- handling of anomalous observations during adaptation.

An anomalous observation should not automatically redefine the baseline.

Adaptation must be conservative and explicitly justified.

---

# 11. RESEARCH LOG — MANDATORY STATE

`PRISM_RESEARCH_LOG.md` is the authoritative chronological research history.

It is append-only.

NEVER:

- edit old entries;
- rewrite old conclusions;
- delete old entries;
- reorder entries;
- silently correct historical values.

If new evidence contradicts an old result, preserve the old entry and append a new entry explaining the change.

Every meaningful research experiment or result must eventually appear in the log.

The log must contain enough information to reconstruct the eventual paper's:

- Methods;
- Experiments;
- Results;
- Discussion;
- Limitations.

---

# 12. FILE RESPONSIBILITIES

Do not duplicate information unnecessarily.

### AGENTS.md
Rules for how an agent should work.

### PRISM_RESEARCH_LOG.md
Historical record of what happened.

### ARCHITECTURE.md
Current technical architecture and interfaces.

### README.md
How a developer understands and runs the repository.

### src/
Research implementation.

### tests/
Automated correctness checks.

### results/
Generated experimental evidence.

Raw dataset files must not be modified to manufacture experimental outcomes.

---

# 13. BEFORE MODIFYING CODE

Before changing an existing component:

1. Read the relevant implementation.
2. Read its tests.
3. Determine its current interface.
4. Determine whether it is part of an established experiment.
5. Reuse existing functionality where possible.

Never speculate about implementation details that have not been inspected.

Never replace working code with a new abstraction merely because it looks cleaner.

---

# 14. MINIMAL-CHANGE PRINCIPLE

PRISM is a research codebase.

Avoid:

- unnecessary refactoring;
- premature abstraction;
- unnecessary configuration systems;
- speculative features;
- duplicate pipelines;
- unrelated cleanup;
- large architectural rewrites.

Implement the smallest change that enables the next research question.

Temporary scripts may be used for experiments, but clean them up if they are no longer useful.

---

# 15. TESTING

Every meaningful new component should have tests.

At minimum, test:

- normal operation;
- edge cases;
- invalid/empty inputs where relevant;
- numerical stability;
- deterministic behavior where expected.

Never modify or delete tests simply to make an implementation pass.

Tests verify the implementation; they do not define the scientific truth.

---

# 16. REPRODUCIBILITY

Every experiment that produces a reported result should record enough information to reproduce it.

Record where relevant:

- dataset/version;
- recording/session split;
- random seed;
- feature configuration;
- model version;
- threshold;
- calibration size;
- hyperparameters;
- software environment;
- output location.

Prefer deterministic experiments where practical.

Do not overwrite important experimental outputs without preserving provenance.

---

# 17. RESULTS MANAGEMENT

Do not report numbers directly from terminal output and then discard the underlying result.

Important results should be saved under `results/`.

Prefer machine-readable outputs such as:

- CSV;
- JSON;
- NPZ.

Figures should be generated from saved result data rather than manually edited.

If an experiment changes a reported number, preserve the previous result and record the new experiment separately.

---

# 18. DOCUMENTATION INTEGRITY

Do not update README/architecture/documentation to claim that something is implemented merely because it is planned.

Use explicit states:

- IMPLEMENTED
- VALIDATED
- EXPERIMENTAL
- IN DEVELOPMENT
- PLANNED
- NOT VALIDATED

Documentation must reflect the actual repository state.

---

# 19. SCIENTIFIC LANGUAGE

Use precise terminology.

Prefer:

- "baseline-consistent"
- "baseline deviation"
- "anomalous relative to the learned baseline"
- "potential technique inconsistency"
- "acoustic event"
- "inhalation event"
- "controlled deviation"

Avoid unsupported language such as:

- "clinically correct"
- "clinically incorrect"
- "good breathing"
- "bad breathing"
- "poor inhalation"
- "effective drug delivery"
- "clinical compliance"

unless supported by appropriate evidence.

---

# 20. WHEN THE USER ASKS FOR "THE NEXT STEP"

Do not immediately invent a new model.

First inspect:

- current git status/diff;
- relevant source files;
- current tests;
- `PRISM_RESEARCH_LOG.md`;
- relevant results;
- current implementation status.

Then identify the smallest unresolved research question.

Continue from the existing state rather than restarting the project.

---

# 21. LONG-RUN / CONTEXT-COMPACTION BEHAVIOR

PRISM will be developed across multiple Claude Code sessions.

Treat the repository as persistent memory.

Before a context window becomes constrained:

1. finish or checkpoint the current experiment;
2. save important results;
3. update the research log;
4. record the current decision;
5. record the exact next step.

Never leave major work only in conversational context.

When starting a new session, reconstruct state from the repository rather than assuming prior conversational memory is available.

---

# 22. DEFAULT WORK STYLE

Be proactive, but research-driven.

When the user gives a concrete implementation task:

- inspect;
- plan briefly;
- implement;
- test;
- evaluate;
- record.

When the user gives an ambiguous research direction:

- inspect the evidence;
- identify the actual unresolved question;
- propose the smallest experiment;
- do not prematurely implement a large system.

Use parallel tool calls for independent repository inspection when useful, but keep dependent operations sequential.

Use subagents only when the work is genuinely parallel or benefits from isolated context.

---

# 23. DEFINITION OF DONE

A research task is NOT complete merely because the code runs.

A meaningful research task is complete when:

- the implementation exists;
- relevant tests pass;
- the experiment has actually been executed;
- results are saved;
- the result is interpreted;
- limitations are known;
- the research log is updated;
- documentation is updated if the current system behavior changed.

If one of these is intentionally omitted, state why.

---

# 24. CURRENT PRIORITY

Until explicitly changed by the user, prioritize:

1. Verify existing inhale-event extraction.
2. Establish the event-level feature representation.
3. Implement the personalized baseline.
4. Implement anomaly scoring.
5. Design threshold selection.
6. Evaluate calibration/training anomaly rate.
7. Evaluate held-out baseline consistency.
8. Evaluate controlled deviations.
9. Analyze failure cases.
10. Integrate the pipeline.
11. Produce reproducible figures/tables.
12. Update research documentation.
13. Prepare the evidence required for a research paper.

Do not spend substantial effort on UI, cloud infrastructure, federated learning, clinical deployment, or additional model architectures while the core personalized anomaly-detection experiment remains unresolved.