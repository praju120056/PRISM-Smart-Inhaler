"""
run_pipeline.py
---------------
Main entry point. Orchestrates the full pipeline per ARCHITECTURE.md:

    Step 0 -- Load annotation CSV
    Step 1 -- Extract features with librosa (124 features/frame, cached .npy)
    Step 2 -- Build windowed dataset
               trim_to_events -> create_windows
               -> balance_dataset (Noise cap + Drug oversample)
    Step 3 -- Encode string labels with LabelEncoder (fixed LABEL_NAMES order)
    Step 4 -- Cross-validation (GroupKFold: RF + SVM + XGBoost)
    Step 5 -- Aggregate and print metrics
    Step 6 -- Save confusion matrix plot
    Step 7 -- Save feature importance plot
    Step 8 -- Misclassification analysis
    Step 9 -- Save cv_results.csv + summary_report.txt

Extraction path  : data/       *.wav
                   data/extracted/<base>/features.npy  (cache; auto-created)
Feature vector   : 124 per frame  [MFCC(40)|d(40)|dd(40)|centroid|flatness|rolloff|zcr]
Window vector    : 7 x 124 = 868 features  (sliding window, majority-vote label)

Usage:
    python src/run_pipeline.py              # full 5-fold, RF + SVM + XGBoost
    python src/run_pipeline.py --fast       # 3-fold, XGBoost only  (fastest)
    python src/run_pipeline.py --xgb-only   # 5-fold, XGBoost only  (skip RF + SVM)
    python src/run_pipeline.py --no-svm     # 5-fold, RF + XGBoost  (skip slow SVM)

NOTE -- training vs. inference
    This script trains models for research/comparison on your PC.
    The mobile device only ever runs *inference* via ONNX Runtime on a
    pre-exported model file.  XGBoost inference on one 868-feature window
    takes ~0.1 ms on a mid-range phone -- not even measurable by the user.
"""

import sys
import os
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config
from loader             import load_annotation
from librosa_extractor  import load_all_recordings_librosa
from feature_extractor  import trim_to_events, create_windows, balance_dataset
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

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument("--fast",     action="store_true",
                     help="3-fold CV, XGBoost only -- fastest iteration")
_parser.add_argument("--xgb-only", action="store_true",
                     help="5-fold CV, XGBoost only (skip RF and SVM)")
_parser.add_argument("--no-svm",   action="store_true",
                     help="5-fold CV, RF + XGBoost (skip SVM)")
_parser.add_argument("--cnn",      action="store_true",
                     help="5-fold CV, 1D CNN instead of XGBoost (requires torch)")
_args, _ = _parser.parse_known_args()

FAST_MODE = _args.fast
USE_CNN   = _args.cnn
XGB_ONLY  = (_args.xgb_only or FAST_MODE) and not USE_CNN
NO_SVM    = _args.no_svm   or XGB_ONLY or USE_CNN
N_FOLDS   = 3 if FAST_MODE else config.N_SPLITS

RUN_RF  = not XGB_ONLY and not USE_CNN
RUN_SVM = not NO_SVM
RUN_XGB = not USE_CNN   # XGBoost off when CNN is selected


def main():
    mode_str = (
        "FAST (3-fold, XGBoost only)" if FAST_MODE else
        "XGBoost only (5-fold)"        if XGB_ONLY  else
        "RF + XGBoost (5-fold)"        if NO_SVM    else
        "Full (RF + SVM + XGBoost, 5-fold)"
    )
    print("=" * 60)
    print(f"run_pipeline.py  --  {mode_str}")
    print("=" * 60)

    # ── Step 0: annotation ────────────────────────────────────────
    print("\n" + "=" * 60)
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

    # NOTE: librosa_extractor already contains delta and delta-delta MFCC.
    # We call trim + create_windows directly to avoid a redundant stacking pass.
    Xs, ys, gs = [], [], []
    for rec_idx, (X_rec, y_rec) in enumerate(zip(all_X, all_y)):
        X_rec, y_rec = trim_to_events(
            X_rec, y_rec,
            noise_label   = "Noise",
            buffer_frames = config.NOISE_BUFFER,
        )
        if X_rec.shape[0] < config.WINDOW_SIZE:
            continue
        X_win, y_win = create_windows(
            X_rec, y_rec,
            window_size = config.WINDOW_SIZE,
            stride      = config.WINDOW_STRIDE,
        )
        Xs.append(X_win)
        ys.append(y_win)
        gs.append(np.full(X_win.shape[0], rec_idx, dtype=int))

    if not Xs:
        raise RuntimeError("No windows generated -- check recordings have enough frames.")

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
    window_fdim = X_win_all.shape[1]
    print(f"\n  Windows  : {n_windows:,}")
    print(f"  Features : {window_fdim}  per window  "
          f"({config.WINDOW_SIZE} frames x {config.LIBROSA_N_FEATURES})")

    unique_y, cnt_y = np.unique(y_win_all, return_counts=True)
    print("\n  Label distribution (after balancing):")
    for lbl, cnt in zip(unique_y, cnt_y):
        print(f"    {lbl:10s}: {cnt:7,}  ({100 * cnt / n_windows:.1f}%)")

    # ── Step 3: encode labels ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 -- Encode labels")
    print("=" * 60)
    le = LabelEncoder()
    le.fit(config.LABEL_NAMES)
    y_int = le.transform(y_win_all)
    print(f"  Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Step 4: cross-validation ──────────────────────────────────
    models_str = "1D CNN (CUDA)" if USE_CNN else " + ".join(
        m for m, on in [("RF", RUN_RF), ("SVM", RUN_SVM), ("XGBoost", RUN_XGB)] if on
    )
    print("\n" + "=" * 60)
    print(f"STEP 4 -- Cross-validation (GroupKFold {N_FOLDS}-fold)")
    print(f"          Models: {models_str}")
    print("=" * 60)

    if USE_CNN:
        from train_cnn import run_cnn_cv
        cv = run_cnn_cv(
            X_win_all, y_int, groups, le,
            n_splits = N_FOLDS,
        )
    else:
        cv = run_cross_validation(
            X_win_all, y_int, groups, le,
            run_rf=RUN_RF, run_svm=RUN_SVM, run_xgb=RUN_XGB,
            n_splits=N_FOLDS,
        )

    # ── Step 5: aggregate metrics ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 -- Results summary")
    print("=" * 60)
    model_label = "CNN" if USE_CNN else "XGBoost"
    rf_summary  = aggregate_metrics(cv["rf_results"],  "Random Forest", le) if RUN_RF  else None
    svm_summary = aggregate_metrics(cv["svm_results"], "SVM",           le) if RUN_SVM else None
    xgb_summary = aggregate_metrics(cv["xgb_results"], model_label,     le)

    # ── Step 6: visualizations ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 -- Visualizations")
    print("=" * 60)
    plot_confusion_matrices(
        cv["rf_cm_total"], cv["svm_cm_total"], cv["xgb_cm_total"], le,
        run_rf=RUN_RF, run_svm=RUN_SVM,
    )
    plot_class_metrics(rf_summary, svm_summary, xgb_summary, le,
                       run_rf=RUN_RF, run_svm=RUN_SVM)
    plot_drug_stats(rf_summary, svm_summary, xgb_summary,
                    run_rf=RUN_RF, run_svm=RUN_SVM)
    plot_noise_confusion(
        cv["rf_cm_total"], cv["svm_cm_total"], cv["xgb_cm_total"], le,
        run_rf=RUN_RF, run_svm=RUN_SVM,
    )

    # ── Step 7: feature importance (RF or XGBoost) ────────────────
    print("\n" + "=" * 60)
    print("STEP 7 -- Feature importance")
    print("=" * 60)
    _orig_fn = config.FEATURE_NAMES
    config.FEATURE_NAMES = config.LIBROSA_FEATURE_NAMES
    if RUN_RF and cv["rf_fi_folds"]:
        plot_feature_importance(cv["rf_fi_folds"], title_prefix="RF")
    if cv["xgb_fi_folds"]:
        plot_feature_importance(cv["xgb_fi_folds"], title_prefix="XGBoost")
    config.FEATURE_NAMES = _orig_fn

    # ── Step 8: misclassification analysis ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 -- Misclassification analysis")
    print("=" * 60)
    cm_for_analysis = cv["xgb_cm_total"]   # use XGB (deployment model)
    print_misclassification_analysis(cm_for_analysis, le)

    # ── Step 9: save outputs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9 -- Save outputs")
    print("=" * 60)
    save_cv_csv(cv["rf_results"], cv["svm_results"], cv["xgb_results"], le)
    report = save_summary_report(
        rf_summary  or {"mean_acc": float("nan"), "std_acc": float("nan")},
        svm_summary or {"mean_acc": float("nan"), "std_acc": float("nan")},
        xgb_summary,
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
