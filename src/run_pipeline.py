"""
run_pipeline.py
---------------
Main entry point. Orchestrates the full pipeline per ARCHITECTURE.md:

    Step 0 -- Load annotation CSV
    Step 1 -- Extract features with librosa (124 features/frame, cached .npy)
    Step 2 -- Build windowed dataset
               trim_to_events → (stack_features) → create_windows
               → balance_dataset (Noise cap + Drug oversample)
    Step 3 -- Encode string labels with LabelEncoder (fixed LABEL_NAMES order)
    Step 4 -- Cross-validation (GroupKFold: RF + SVM)
    Step 5 -- Aggregate and print metrics
    Step 6 -- Save confusion matrix plot
    Step 7 -- Save feature importance plot
    Step 8 -- Misclassification analysis
    Step 9 -- Save cv_results.csv + summary_report.txt

Extraction path  : data/       *.wav
                   data/extracted/<base>/features.npy  (cache; auto-created)
Feature vector   : 124 per frame  [MFCC(40)|Δ(40)|ΔΔ(40)|centroid|flatness|rolloff|zcr]
Window vector    : 7 × 124 = 868 features  (sliding window, majority-vote label)

Usage:
    python src/run_pipeline.py
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ensure src/ is on the path so sibling imports work when
#    this script is run from the project root OR from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config
from loader             import load_annotation                    # annotation only
from librosa_extractor  import load_all_recordings_librosa        # 124-feature extraction
from feature_extractor  import build_windowed_dataset             # trim+window+balance
from train              import run_cross_validation
from evaluate           import (
    aggregate_metrics,
    print_misclassification_analysis,
    save_cv_csv,
    save_summary_report,
)
from visualize          import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_class_metrics,
    plot_drug_stats,
    plot_noise_confusion,
)


def main():
    # ── Step 0: annotation ────────────────────────────────────────
    print("=" * 60)
    print("STEP 0 -- Load annotation")
    print("=" * 60)
    ann = load_annotation()
    print(f"  Entries : {len(ann)}")
    print(ann["label"].value_counts().to_string())

    # ── Step 1: extract features (librosa, with .npy cache) ───────
    print("\n" + "=" * 60)
    print("STEP 1 -- Extract features (librosa_extractor, 124 features)")
    print("         [Cached under data/extracted/<rec>/features.npy]")
    print("=" * 60)
    all_X, all_y, all_groups, skipped = load_all_recordings_librosa(
        ann,
        use_cache = True,
        data_dir  = config.DATA_DIR,
    )

    print(f"\n  Loaded  : {len(all_X)} recordings")
    print(f"  Skipped : {len(skipped)}")
    for name, reason in skipped:
        print(f"    [SKIP] {name}: {reason}")

    if not all_X:
        raise RuntimeError(
            "No recordings loaded -- check DATA_DIR in config.py "
            "and that librosa is installed."
        )

    total_frames = sum(x.shape[0] for x in all_X)
    print(f"\n  Total frames : {total_frames:,}")
    print(f"  Feature dim  : {all_X[0].shape[1]}  (per frame)")

    # ── Step 2: build windowed dataset ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 -- Build windowed dataset")
    print(f"         window_size={config.WINDOW_SIZE}, stride={config.WINDOW_STRIDE}")
    print(f"         trim_noise=True (buffer={config.NOISE_BUFFER} frames)")
    print(f"         balance=True (drug_multiplier={config.DRUG_MULTIPLIER})")
    print("=" * 60)

    # NOTE: librosa_extractor already computes delta and delta-delta MFCC.
    # build_windowed_dataset would add another stack_features() pass on top,
    # which doubles deltas unnecessarily.  We skip the extra stacking by
    # passing the full 124-feature arrays directly to create_windows.
    # We do this by monkey-patching the stack step: pass all_X directly,
    # let trim_to_events and create_windows run, but disable stack_features
    # by using the same width=1 delta trick — actually easiest is to call
    # trim + create_windows manually per recording to avoid double-delta.
    from feature_extractor import trim_to_events, create_windows, balance_dataset

    Xs, ys, gs = [], [], []
    for rec_idx, (X_rec, y_rec) in enumerate(zip(all_X, all_y)):
        # trim excess Noise frames far from annotated events
        X_rec, y_rec = trim_to_events(
            X_rec, y_rec,
            noise_label   = "Noise",
            buffer_frames = config.NOISE_BUFFER,
        )
        if X_rec.shape[0] < config.WINDOW_SIZE:
            continue
        # sliding window + majority-vote labels (no extra delta stacking)
        X_win, y_win = create_windows(
            X_rec, y_rec,
            window_size = config.WINDOW_SIZE,
            stride      = config.WINDOW_STRIDE,
        )
        Xs.append(X_win)
        ys.append(y_win)
        gs.append(np.full(X_win.shape[0], rec_idx, dtype=int))

    if not Xs:
        raise RuntimeError("No windows generated — check recordings have enough frames.")

    X_win_all = np.vstack(Xs)
    y_win_all = np.concatenate(ys)
    groups    = np.concatenate(gs)

    # balance: cap Noise + oversample Drug (carry groups as extra column)
    G_col = groups.reshape(-1, 1).astype(np.float64)
    X_aug = np.hstack([X_win_all.astype(np.float64), G_col])
    rng   = np.random.default_rng(42)
    X_aug, y_win_all = balance_dataset(
        X_aug, y_win_all,
        drug_multiplier = config.DRUG_MULTIPLIER,
        noise_label     = "Noise",
        drug_label      = "Drug",
        rng             = rng,
    )
    groups    = X_aug[:, -1].astype(int)
    X_win_all = X_aug[:, :-1].astype(np.float32)

    n_windows   = X_win_all.shape[0]
    window_fdim = X_win_all.shape[1]   # WINDOW_SIZE * 124 = 868
    print(f"\n  Windows  : {n_windows:,}")
    print(f"  Features : {window_fdim}  per window  "
          f"({config.WINDOW_SIZE} frames × {config.LIBROSA_N_FEATURES})")

    unique_y, cnt_y = np.unique(y_win_all, return_counts=True)
    print("\n  Label distribution (after balancing):")
    for lbl, cnt in zip(unique_y, cnt_y):
        print(f"    {lbl:10s}: {cnt:7,}  ({100 * cnt / n_windows:.1f}%)")

    # ── Step 3: encode labels ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 -- Encode labels")
    print("=" * 60)
    le = LabelEncoder()
    le.fit(config.LABEL_NAMES)      # fixed order — deterministic across runs
    y_int = le.transform(y_win_all)
    print(f"  Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Step 4: cross-validation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 -- Cross-validation (GroupKFold, recording-level)")
    print("          Models: Random Forest, SVM, XGBoost")
    print("=" * 60)
    cv = run_cross_validation(X_win_all, y_int, groups, le)

    # ── Step 5: aggregate metrics ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 -- Results summary")
    print("=" * 60)
    rf_summary  = aggregate_metrics(cv["rf_results"],  "Random Forest", le)
    svm_summary = aggregate_metrics(cv["svm_results"], "SVM",           le)
    xgb_summary = aggregate_metrics(cv["xgb_results"], "XGBoost",       le)

    # ── Step 6: visualizations ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 -- Visualizations")
    print("=" * 60)
    plot_confusion_matrices(
        cv["rf_cm_total"], cv["svm_cm_total"], cv["xgb_cm_total"], le
    )
    plot_class_metrics(rf_summary, svm_summary, xgb_summary, le)
    plot_drug_stats(rf_summary, svm_summary, xgb_summary)
    plot_noise_confusion(
        cv["rf_cm_total"], cv["svm_cm_total"], cv["xgb_cm_total"], le
    )

    # ── Step 7: feature importance ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 -- Feature importance")
    print("=" * 60)
    # Override FEATURE_NAMES in config temporarily so visualize.py uses
    # the windowed librosa names (868 features) rather than the legacy 40.
    _orig_fn = config.FEATURE_NAMES
    config.FEATURE_NAMES = config.LIBROSA_FEATURE_NAMES
    plot_feature_importance(cv["rf_fi_folds"])
    config.FEATURE_NAMES = _orig_fn

    # ── Step 8: misclassification analysis ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 -- Misclassification analysis")
    print("=" * 60)
    print_misclassification_analysis(cv["rf_cm_total"], le)

    # ── Step 9: save outputs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9 -- Save outputs")
    print("=" * 60)
    save_cv_csv(cv["rf_results"], cv["svm_results"], cv["xgb_results"], le)
    report = save_summary_report(
        rf_summary, svm_summary, xgb_summary,
        X_win_all.shape, y_win_all,
        loaded  = len(all_X),
        skipped = skipped,
        le      = le,
    )

    print("\n" + "=" * 60)
    print("DONE -- results saved to:", config.RESULTS_DIR)
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
