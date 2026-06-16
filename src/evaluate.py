"""
evaluate.py
-----------
Aggregates CV fold results into per-class metrics, prints the
misclassification analysis, and saves cv_results.csv + summary_report.txt.

Run standalone (after CV results exist in memory) — useful for
re-running analysis without re-training:
    python src/evaluate.py
"""

import os
import numpy as np
import pandas as pd

import config


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def aggregate_metrics(results: list, model_name: str, le) -> dict:
    """
    Average per-class precision / recall / F1 across CV folds and print.

    Returns
    -------
    dict   keys: 'mean_acc', 'std_acc', per-class metrics
    """
    all_acc   = [r["accuracy"] for r in results]
    mean_acc  = float(np.mean(all_acc))
    std_acc   = float(np.std(all_acc))

    per_class = {
        c: {"precision": [], "recall": [], "f1": []}
        for c in le.classes_
    }
    for r in results:
        for cls in le.classes_:
            if cls in r["report"]:
                per_class[cls]["precision"].append(r["report"][cls]["precision"])
                per_class[cls]["recall"].append(r["report"][cls]["recall"])
                per_class[cls]["f1"].append(r["report"][cls]["f1-score"])

    print(f"\n{'=' * 40}")
    print(f"  {model_name}  -- mean acc: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"{'=' * 40}")
    print(f"  {'Class':12s}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")

    summary = {"mean_acc": mean_acc, "std_acc": std_acc}
    for cls in le.classes_:
        p = float(np.mean(per_class[cls]["precision"]))
        r = float(np.mean(per_class[cls]["recall"]))
        f = float(np.mean(per_class[cls]["f1"]))
        print(f"  {cls:12s}  {p:10.4f}  {r:10.4f}  {f:10.4f}")
        summary[cls] = {"precision": p, "recall": r, "f1": f}

    return summary


def print_misclassification_analysis(rf_cm: np.ndarray, le):
    """
    Print the raw confusion matrix and highlight key boundary confusions.
    """
    print("\n  Confusion matrix (RF, summed over all folds):")
    print(f"  {'':12s}", end="")
    for c in le.classes_:
        print(f"  {c:>8s}", end="")
    print()
    for i, true_cls in enumerate(le.classes_):
        print(f"  {true_cls:12s}", end="")
        for j in range(len(le.classes_)):
            print(f"  {rf_cm[i, j]:8d}", end="")
        print(f"  | total={rf_cm[i].sum()}")

    inh_idx = le.transform(["Inhale"])[0]
    exh_idx = le.transform(["Exhale"])[0]
    drg_idx = le.transform(["Drug"])[0]

    inh_total   = rf_cm[inh_idx].sum()
    exh_total   = rf_cm[exh_idx].sum()
    inh_as_exh  = rf_cm[inh_idx, exh_idx]
    exh_as_inh  = rf_cm[exh_idx, inh_idx]
    drug_recall = 100 * rf_cm[drg_idx, drg_idx] / max(rf_cm[drg_idx].sum(), 1)

    print("\n  Key boundary confusion (Inhale <-> Exhale):")
    print(
        f"    Inhale->Exhale : {inh_as_exh}/{inh_total}"
        f" = {100 * inh_as_exh / max(inh_total, 1):.1f}%"
    )
    print(
        f"    Exhale->Inhale : {exh_as_inh}/{exh_total}"
        f" = {100 * exh_as_inh / max(exh_total, 1):.1f}%"
    )
    print(f"    Drug recall    : {drug_recall:.1f}%")


def save_cv_csv(rf_results: list, svm_results: list, xgb_results: list, le):
    """Save per-fold metrics for all models to cv_results.csv."""
    rows = []
    for r in rf_results + svm_results + xgb_results:
        row = {
            "fold":     r["fold"],
            "model":    r["model"],
            "accuracy": r["accuracy"],
        }
        for cls in le.classes_:
            if cls in r["report"]:
                row[f"{cls}_precision"] = r["report"][cls]["precision"]
                row[f"{cls}_recall"]    = r["report"][cls]["recall"]
                row[f"{cls}_f1"]        = r["report"][cls]["f1-score"]
        rows.append(row)

    path = os.path.join(config.RESULTS_DIR, "cv_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_summary_report(
    rf_summary: dict,
    svm_summary: dict,
    xgb_summary: dict,
    X_shape,
    y_str: np.ndarray,
    loaded: int,
    skipped: list,
    le,
):
    """Write a human-readable summary_report.txt to results/."""
    unique, counts = np.unique(y_str, return_counts=True)
    total          = len(y_str)
    n_splits       = config.N_SPLITS

    lines = [
        "pMDI Inhaler Event Classification -- Results Summary",
        "=" * 55,
        f"Recordings loaded       : {loaded}",
        f"Total frames            : {X_shape[0]:,}",
        f"Features per frame      : {X_shape[1]}",
        f"Cross-validation        : GroupKFold, {n_splits} folds (recording-level)",
        "",
        "Model Performance",
        "-" * 40,
        f"Random Forest  -- mean acc: {rf_summary['mean_acc']:.4f}"
        f" +/- {rf_summary['std_acc']:.4f}",
        f"SVM            -- mean acc: {svm_summary['mean_acc']:.4f}"
        f" +/- {svm_summary['std_acc']:.4f}",
        f"XGBoost        -- mean acc: {xgb_summary['mean_acc']:.4f}"
        f" +/- {xgb_summary['std_acc']:.4f}",
        "",
        "Label Distribution",
        "-" * 40,
    ]
    for lbl, cnt in zip(unique, counts):
        lines.append(f"  {lbl:10s}: {cnt:7,}  ({100 * cnt / total:.1f}%)")

    lines += [
        "",
        "Feature Set  (librosa_extractor.py)",
        "-" * 40,
        "  MFCC               40 coefficients  (N_FFT=256, HOP=64, N_MELS=128)",
        "  Delta-MFCC         40  (HTK regression filter, width=9)",
        "  Delta-delta MFCC   40",
        "  Spectral centroid   1  (normalised by Nyquist)",
        "  Spectral flatness   1",
        "  Spectral rolloff    1  (normalised by Nyquist, roll_percent=0.85)",
        "  ZCR                 1",
        "  ─────────────────────",
        "  Per-frame total   124",
        "",
        "  Windowing (feature_extractor.py)",
        "  Window size : 7 frames = 56 ms  (HOP=64, SR=8000)",
        "  Stride      : 1 frame",
        "  Label       : majority vote over window",
        "  Per-window  : 7 × 124 = 868 features",
        "",
        "Skipped recordings",
        "-" * 40,
    ]
    for name, reason in skipped:
        lines.append(f"  {name}: {reason}")

    text = "\n".join(lines)
    path = os.path.join(config.RESULTS_DIR, "summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


# ──────────────────────────────────────────────────────────────
# Standalone debug entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("evaluate.py -- reads cv_results.csv and reprints summary")
    print("=" * 60)

    path = os.path.join(config.RESULTS_DIR, "cv_results.csv")
    if not os.path.exists(path):
        print(f"  No cv_results.csv found at {path}")
        print("  Run run_pipeline.py first to generate it.")
    else:
        df = pd.read_csv(path)
        print(df.to_string(index=False))
