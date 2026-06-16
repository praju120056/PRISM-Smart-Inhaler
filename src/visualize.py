"""
visualize.py
------------
Generates and saves:
  - Confusion matrix comparison (RF vs SVM)  → confusion_matrix.png
  - Top-15 RF feature importances             → feature_importance.png

Run standalone to regenerate plots from saved cv_results.csv:
    python src/visualize.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import config


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrices(rf_cm, svm_cm, xgb_cm, le,
                            run_rf=True, run_svm=True):
    """
    Plot confusion matrices for enabled models side by side.
    """
    models = []
    if run_rf:
        models.append((rf_cm, "Random Forest (all folds)"))
    if run_svm:
        models.append((svm_cm, "SVM (all folds)"))
    models.append((xgb_cm, "XGBoost (all folds)"))

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (cm, title) in zip(axes, models):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=le.classes_
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(fi_folds: list, top_n: int = 15, title_prefix: str = "RF"):
    """
    Compute mean feature importance across folds and save a bar chart.

    Parameters
    ----------
    fi_folds     : list[np.ndarray]   feature_importances_ per fold
    top_n        : int                how many features to display
    title_prefix : str                model name label in the chart title
    """
    feature_names = config.FEATURE_NAMES
    mean_fi       = np.mean(fi_folds, axis=0)
    top_idx       = np.argsort(mean_fi)[-top_n:][::-1]

    print(f"\n  Top {top_n} {title_prefix} features:")
    for rank, idx in enumerate(top_idx):
        print(
            f"  {rank + 1:2d}. {feature_names[idx]:20s}"
            f"  importance={mean_fi[idx]:.4f}"
        )

    # Bar chart — blue = raw feature, orange = delta
    colors = [
        "#4C72B0" if "delta" not in feature_names[i] else "#DD8452"
        for i in top_idx
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],
        mean_fi[top_idx[::-1]],
        color=colors[::-1],
    )
    ax.set_xlabel("Mean feature importance (across folds)")
    ax.set_title(
        f"Top {top_n} {title_prefix} Feature Importances\n(blue=raw, orange=delta)",
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, f"feature_importance_{title_prefix.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_class_metrics(rf_summary, svm_summary, xgb_summary, le,
                       run_rf=True, run_svm=True):
    """
    Plot per-class precision, recall, and F1-score for enabled models.
    """
    classes = le.classes_
    metrics = ['precision', 'recall', 'f1']

    active = []
    if run_rf  and rf_summary:  active.append(('RF',      rf_summary,  '#4C72B0'))
    if run_svm and svm_summary: active.append(('SVM',     svm_summary, '#DD8452'))
    active.append(('XGBoost', xgb_summary, '#55A868'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(classes))
    n = len(active)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for ax, metric, title in zip(axes, metrics, ['Precision', 'Recall', 'F1-Score']):
        for (label, summary, color), offset in zip(active, offsets):
            vals = [summary[c][metric] for c in classes]
            ax.bar(x + offset, vals, width, label=label, color=color)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_drug_stats(rf_summary, svm_summary, xgb_summary,
                    run_rf=True, run_svm=True):
    """
    Plot Drug-specific metrics for enabled models.
    """
    if 'Drug' not in xgb_summary:
        print("  [WARN] 'Drug' class not found in XGB summary, skipping drug stats plot.")
        return

    metrics = ['precision', 'recall', 'f1']
    active = []
    if run_rf  and rf_summary  and 'Drug' in rf_summary:  active.append(('RF',      rf_summary,  '#4C72B0'))
    if run_svm and svm_summary and 'Drug' in svm_summary: active.append(('SVM',     svm_summary, '#DD8452'))
    active.append(('XGBoost', xgb_summary, '#55A868'))

    x = np.arange(len(metrics))
    n = len(active)
    width = 0.6 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(6, 5))
    for (label, summary, color), offset in zip(active, offsets):
        vals = [summary['Drug'][m] for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_title("Drug Class Performance", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "drug_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_noise_confusion(rf_cm, svm_cm, xgb_cm, le,
                         run_rf=True, run_svm=True):
    """
    Plot Noise confusion breakdown for enabled models.
    """
    if "Noise" not in le.classes_:
        print("  [WARN] 'Noise' class not found, skipping noise confusion plot.")
        return
        
    noise_idx = list(le.classes_).index("Noise")
    classes = le.classes_
    other_classes = [c for c in classes if c != "Noise"]
    other_idx = [i for i, c in enumerate(classes) if c != "Noise"]

    active = []
    if run_rf:  active.append(("RF",      rf_cm,  "#4C72B0"))
    if run_svm: active.append(("SVM",     svm_cm, "#DD8452"))
    active.append(("XGBoost", xgb_cm, "#55A868"))

    n = len(active)
    x = np.arange(len(other_classes))
    width = 0.6 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (row_sel, col_sel, panel_title) in zip(
        axes,
        [
            (noise_idx, other_idx, "True = Noise, Predicted as Other"),
            (other_idx, noise_idx, "True = Other, Predicted as Noise"),
        ],
    ):
        for (label, cm, color), offset in zip(active, offsets):
            vals = cm[row_sel, col_sel]
            ax.bar(x + offset, vals, width, label=label, color=color)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(other_classes)
        ax.set_ylabel("Number of instances")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.suptitle("Noise Confusion Breakdown", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "noise_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



# ──────────────────────────────────────────────────────────────
# Standalone debug entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("visualize.py -- standalone check")
    print("=" * 60)

    csv_path = os.path.join(config.RESULTS_DIR, "cv_results.csv")
    if not os.path.exists(csv_path):
        print(f"  No cv_results.csv found at {csv_path}")
        print("  Run run_pipeline.py first to generate it.")
    else:
        df = pd.read_csv(csv_path)
        print(df[["fold", "model", "accuracy"]].to_string(index=False))
        print("\n  (Re-run run_pipeline.py to regenerate plots with full data.)")
