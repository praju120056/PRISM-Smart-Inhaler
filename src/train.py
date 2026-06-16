"""
train.py
--------
Recording-level GroupKFold cross-validation with Random Forest, SVM,
and XGBoost.

Run standalone to train and print fold-level accuracy:
    python src/train.py
"""

import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

import config
from loader import load_annotation, load_all_recordings
from dataset import build_dataset


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _subsample_balanced(X, y, n_per_class: int, seed: int = 42):
    """
    Return a balanced subsample: up to n_per_class frames per class.
    Used to keep SVM training tractable (O(n²) kernel cost).
    """
    rng     = np.random.default_rng(seed)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen  = rng.choice(
            cls_idx,
            size=min(n_per_class, len(cls_idx)),
            replace=False,
        )
        indices.extend(chosen.tolist())
    return np.array(indices)


def _make_xgb_classifier(n_classes: int):
    """
    Build the XGBoost classifier lazily so the rest of the project still imports
    cleanly when xgboost is not installed yet.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from exc

    params = dict(config.XGB_PARAMS)
    params["num_class"] = n_classes
    return XGBClassifier(**params)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_cross_validation(X, y, groups, le):
    """
    Run GroupKFold CV, fitting RF, SVM, and XGBoost on each fold.

    Parameters
    ----------
    X, y, groups : as returned by dataset.build_dataset()
    le           : fitted LabelEncoder

    Returns
    -------
    dict with keys:
        rf_results    : list[dict]  per-fold RF metrics
        svm_results   : list[dict]  per-fold SVM metrics
        xgb_results   : list[dict]  per-fold XGBoost metrics
        rf_cm_total   : np.ndarray  summed confusion matrix (RF)
        svm_cm_total  : np.ndarray  summed confusion matrix (SVM)
        xgb_cm_total  : np.ndarray  summed confusion matrix (XGBoost)
        rf_fi_folds   : list[np.ndarray]  per-fold feature importances (RF)
        xgb_fi_folds  : list[np.ndarray]  per-fold feature importances (XGBoost)
    """
    gkf          = GroupKFold(n_splits=config.N_SPLITS)
    n_classes    = len(le.classes_)
    rf_results   = []
    svm_results  = []
    xgb_results  = []
    rf_cm_total  = np.zeros((n_classes, n_classes), dtype=int)
    svm_cm_total = np.zeros((n_classes, n_classes), dtype=int)
    xgb_cm_total = np.zeros((n_classes, n_classes), dtype=int)
    rf_fi_folds  = []
    xgb_fi_folds = []
    label_order  = le.transform(le.classes_)   # for consistent CM axis order

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):
        print(
            f"\n  Fold {fold + 1}/{config.N_SPLITS} -- "
            f"train={len(train_idx):,} | test={len(test_idx):,} frames"
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── normalise INSIDE fold — no leakage ──────────────────
        scaler      = StandardScaler()
        X_train_sc  = scaler.fit_transform(X_train)
        X_test_sc   = scaler.transform(X_test)

        # ── Random Forest ────────────────────────────────────────
        rf      = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(X_train_sc, y_train)
        rf_pred = rf.predict(X_test_sc)

        rf_acc = accuracy_score(y_test, rf_pred)
        rf_rep = classification_report(
            y_test, rf_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        rf_cm_total += confusion_matrix(y_test, rf_pred, labels=label_order)
        rf_fi_folds.append(rf.feature_importances_)
        rf_results.append({
            "fold": fold + 1, "model": "RF",
            "accuracy": rf_acc, "report": rf_rep,
        })
        print(f"    RF  accuracy: {rf_acc:.4f}")

        # ── SVM on balanced subsample ────────────────────────────
        sub_idx    = _subsample_balanced(
            X_train_sc, y_train, config.SVM_SUBSAMPLE
        )
        svm        = LinearSVC(**config.SVM_PARAMS)
        svm.fit(X_train_sc[sub_idx], y_train[sub_idx])
        svm_pred   = svm.predict(X_test_sc)

        svm_acc    = accuracy_score(y_test, svm_pred)
        svm_rep    = classification_report(
            y_test, svm_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        svm_cm_total += confusion_matrix(y_test, svm_pred, labels=label_order)
        svm_results.append({
            "fold": fold + 1, "model": "SVM",
            "accuracy": svm_acc, "report": svm_rep,
        })
        print(f"    SVM accuracy: {svm_acc:.4f}")

        # ── XGBoost ───────────────────────────────────────────────
        xgb      = _make_xgb_classifier(n_classes)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_rep = classification_report(
            y_test, xgb_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        xgb_cm_total += confusion_matrix(y_test, xgb_pred, labels=label_order)
        xgb_fi_folds.append(xgb.feature_importances_)
        xgb_results.append({
            "fold": fold + 1, "model": "XGBoost",
            "accuracy": xgb_acc, "report": xgb_rep,
        })
        print(f"    XGB accuracy: {xgb_acc:.4f}")

    return {
        "rf_results":    rf_results,
        "svm_results":   svm_results,
        "xgb_results":   xgb_results,
        "rf_cm_total":   rf_cm_total,
        "svm_cm_total":  svm_cm_total,
        "xgb_cm_total":  xgb_cm_total,
        "rf_fi_folds":   rf_fi_folds,
        "xgb_fi_folds":  xgb_fi_folds,
    }


# ──────────────────────────────────────────────────────────────
# Standalone debug entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("train.py — standalone CV run")
    print("=" * 60)

    ann                               = load_annotation()
    all_X, all_y, all_groups, skipped = load_all_recordings(ann)
    X, y, y_str, groups, le           = build_dataset(all_X, all_y, all_groups)

    print(f"\nDataset: {X.shape[0]:,} frames, {X.shape[1]} features, "
          f"{len(np.unique(groups))} recordings\n")

    cv = run_cross_validation(X, y, groups, le)

    print("\n\nFold summary:")
    for r in cv["rf_results"]:
        print(f"  Fold {r['fold']}  RF={r['accuracy']:.4f}")
    for r in cv["svm_results"]:
        print(f"  Fold {r['fold']} SVM={r['accuracy']:.4f}")
    for r in cv["xgb_results"]:
        print(f"  Fold {r['fold']} XGB={r['accuracy']:.4f}")
