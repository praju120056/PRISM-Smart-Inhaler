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

def plot_confusion_matrices(rf_cm: np.ndarray, svm_cm: np.ndarray, xgb_cm: np.ndarray, le):
    """
    Plot RF, SVM, and XGBoost confusion matrices side by side and save to results/.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, cm, title in zip(
        axes,
        [rf_cm, svm_cm, xgb_cm],
        ["Random Forest (all folds)", "SVM (all folds)", "XGBoost (all folds)"],
    ):
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


def plot_feature_importance(rf_fi_folds: list, top_n: int = 15):
    """
    Compute mean RF feature importance across folds and save a bar chart.

    Parameters
    ----------
    rf_fi_folds : list[np.ndarray]   feature_importances_ per fold
    top_n       : int                how many features to display
    """
    feature_names = config.FEATURE_NAMES
    mean_fi       = np.mean(rf_fi_folds, axis=0)
    top_idx       = np.argsort(mean_fi)[-top_n:][::-1]

    print(f"\n  Top {top_n} features:")
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
        f"Top {top_n} RF Feature Importances\n(blue=raw, orange=delta)",
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_class_metrics(rf_summary: dict, svm_summary: dict, xgb_summary: dict, le):
    """
    Plot per-class precision, recall, and F1-score for RF vs SVM vs XGBoost.
    """
    classes = le.classes_
    metrics = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(classes))
    width = 0.25
    
    for ax, metric, title in zip(axes, metrics, ['Precision', 'Recall', 'F1-Score']):
        rf_vals = [rf_summary[c][metric] for c in classes]
        svm_vals = [svm_summary[c][metric] for c in classes]
        xgb_vals = [xgb_summary[c][metric] for c in classes]
        
        ax.bar(x - width, rf_vals, width, label='RF', color='#4C72B0')
        ax.bar(x, svm_vals, width, label='SVM', color='#DD8452')
        ax.bar(x + width, xgb_vals, width, label='XGBoost', color='#55A868')
        
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


def plot_drug_stats(rf_summary: dict, svm_summary: dict, xgb_summary: dict):
    """
    Plot Drug-specific metrics (Precision, Recall, F1) for RF vs SVM vs XGBoost.
    """
    if 'Drug' not in rf_summary or 'Drug' not in svm_summary or 'Drug' not in xgb_summary:
        print("  [WARN] 'Drug' class not found in summary, skipping drug stats plot.")
        return

    metrics = ['precision', 'recall', 'f1']
    rf_vals = [rf_summary['Drug'][m] for m in metrics]
    svm_vals = [svm_summary['Drug'][m] for m in metrics]
    xgb_vals = [xgb_summary['Drug'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width, rf_vals, width, label='RF', color='#4C72B0')
    bars2 = ax.bar(x, svm_vals, width, label='SVM', color='#DD8452')
    bars3 = ax.bar(x + width, xgb_vals, width, label='XGBoost', color='#55A868')
    
    ax.set_title("Drug Class Performance", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "drug_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_noise_confusion(rf_cm: np.ndarray, svm_cm: np.ndarray, xgb_cm: np.ndarray, le):
    """
    Plot Noise confusion breakdown (what Noise is predicted as, and what is falsely predicted as Noise).
    """
    if "Noise" not in le.classes_:
        print("  [WARN] 'Noise' class not found, skipping noise confusion plot.")
        return
        
    noise_idx = list(le.classes_).index("Noise")
    classes = le.classes_
    other_classes = [c for c in classes if c != "Noise"]
    other_idx = [i for i, c in enumerate(classes) if c != "Noise"]
    
    # 1. What Noise is predicted as (True: Noise, Pred: Other)
    rf_noise_as_other = rf_cm[noise_idx, other_idx]
    svm_noise_as_other = svm_cm[noise_idx, other_idx]
    xgb_noise_as_other = xgb_cm[noise_idx, other_idx]
    
    # 2. What is predicted as Noise (True: Other, Pred: Noise)
    rf_other_as_noise = rf_cm[other_idx, noise_idx]
    svm_other_as_noise = svm_cm[other_idx, noise_idx]
    xgb_other_as_noise = xgb_cm[other_idx, noise_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(other_classes))
    width = 0.25
    
    # Plot 1
    axes[0].bar(x - width, rf_noise_as_other, width, label='RF', color='#4C72B0')
    axes[0].bar(x, svm_noise_as_other, width, label='SVM', color='#DD8452')
    axes[0].bar(x + width, xgb_noise_as_other, width, label='XGBoost', color='#55A868')
    axes[0].set_title("True = Noise, Predicted as Other", fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(other_classes)
    axes[0].set_ylabel("Number of instances")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2
    axes[1].bar(x - width, rf_other_as_noise, width, label='RF', color='#4C72B0')
    axes[1].bar(x, svm_other_as_noise, width, label='SVM', color='#DD8452')
    axes[1].bar(x + width, xgb_other_as_noise, width, label='XGBoost', color='#55A868')
    axes[1].set_title("True = Other, Predicted as Noise", fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(other_classes)
    axes[1].set_ylabel("Number of instances")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
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
