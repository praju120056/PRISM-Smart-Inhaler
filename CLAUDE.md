# PRISM — Claude Code Instructions

PRISM is a research project for acoustic inhalation-event detection and
personalized inhalation consistency analysis.

## Before doing substantial work

Read:

1. `AGENTS.md` — project-wide research and engineering rules
2. `PRISM_RESEARCH_LOG.md` — current research state and historical decisions
3. `ARCHITECTURE.md` — current system architecture
4. `README.md` — repository usage and implementation status

Do not assume the previous Claude session's context is available.
The repository is the source of truth.

## Current research objective

The existing event-detection pipeline is substantially established.

The current research focus is:

Audio
→ event detection
→ inhale event
→ feature extraction
→ personalized baseline
→ anomaly score
→ NORMAL / ANOMALY
→ rigorous evaluation

The objective is to produce a reproducible, scientifically defensible
personalized anomaly-detection component and the evidence needed for a
future paper/thesis.

## Important scientific constraint

The dataset contains acoustic event annotations, not validated clinical
inhalation-technique quality labels.

Therefore:

- Do not claim clinical "good/bad" breathing classification.
- Do not treat Inhale as equivalent to good technique.
- Do not report technique-quality accuracy without appropriate ground truth.
- NORMAL means baseline-consistent.
- ANOMALY means deviation from the learned personal baseline.

## Current priority

Prioritize, in order:

1. Verify inhale-event extraction.
2. Establish event-level features.
3. Implement the simplest defensible personalized baseline.
4. Implement anomaly scoring.
5. Establish threshold methodology.
6. Evaluate calibration/training anomaly rate.
7. Evaluate held-out baseline consistency.
8. Evaluate controlled deviations.
9. Analyze failure cases.
10. Integrate and document the validated pipeline.

Do not spend substantial effort on unrelated UI, cloud, or model
complexity while the core research question remains unresolved.

## Research workflow

For meaningful research work:

1. Inspect the current implementation and research log.
2. State the research question before implementing.
3. Make the smallest change needed.
4. Run tests.
5. Run the actual experiment.
6. Save results under `results/`.
7. Record findings and decisions in `PRISM_RESEARCH_LOG.md`.

Never fabricate or infer experimental results that were not actually run.

## User Requests

Do not ask for information that is not necessary to complete the task.

If the user requests a summary, explanation, status report, or other
non-code output, answer using the repository's existing evidence without
requiring names, recipients, or other optional personalization.

Do not interrupt a task with clarification questions when a reasonable
default is obvious.

Only ask for clarification when the missing information materially affects
the correctness of the requested work.

## Continuity

At the end of substantial work, leave the repository in a state where
another Claude session can continue without conversational context.

Update:

- code
- tests
- results
- research log

as appropriate.

## Research Status Requests

When asked for the current status of a research component:

1. Read the research log.
2. Inspect the relevant implementation and results.
3. Distinguish:
   - implemented
   - experimentally validated
   - preliminary
   - planned
   - unsupported
4. Summarize what is actually known.
5. Do not begin new implementation unless explicitly requested.

Do not rewrite historical research-log entries. Append new evidence.