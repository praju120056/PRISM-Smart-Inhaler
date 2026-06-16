from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import os
import sys
import hashlib
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from loader import load_annotation
from librosa_extractor import load_all_recordings_librosa
from feature_extractor import trim_to_events, create_windows, balance_dataset


def make_xgb_classifier(n_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        n_estimators=CONFIG["xgb_estimators"],
        max_depth=CONFIG["xgb_max_depth"],
        learning_rate=CONFIG["xgb_learning_rate"],
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

# =========================
# CONFIG
# =========================
DEV_MODE = True

CONFIG = {
    "window_size": 5,
    "stride": 2,
    "noise_buffer": 20,
    "drug_multiplier": 2.0,
    "subset_ratio": 0.4 if DEV_MODE else 1.0,
    "cv_splits": 3 if DEV_MODE else 5,
    "n_estimators": 50 if DEV_MODE else 200,
    "max_depth": 10 if DEV_MODE else None,
    "xgb_estimators": 100 if DEV_MODE else 300,
    "xgb_max_depth": 4 if DEV_MODE else 6,
    "xgb_learning_rate": 0.07 if DEV_MODE else 0.05,
    "use_feature_subset": True,
}

# =========================
# CACHE SETUP
# =========================
_cfg_hash = hashlib.md5(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:8]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "data", f"windowed_cache_{_cfg_hash}.npz")

# =========================
# FEATURE REDUCTION
# =========================
N_MFCC_KEEP = 13

def reduce_features(X):
    if not CONFIG["use_feature_subset"]:
        return X

    mfcc = X[:, :N_MFCC_KEEP]
    delta = X[:, 40:40+N_MFCC_KEEP]
    delta2 = X[:, 80:80+N_MFCC_KEEP]

    return np.hstack([mfcc, delta, delta2])

# =========================
# BUILD DATASET
# =========================
def build_dataset(all_X, all_y):
    le = LabelEncoder()
    le.fit(config.LABEL_NAMES)

    if os.path.exists(CACHE_PATH):
        print(f"[INFO] Loading cached dataset: {os.path.basename(CACHE_PATH)}")
        data = np.load(CACHE_PATH)
        return data["X"], data["y_enc"], data["groups"], le

    print("[INFO] Building dataset...")

    # subset recordings
    if CONFIG["subset_ratio"] < 1.0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_X), int(len(all_X)*CONFIG["subset_ratio"]), replace=False)
        all_X = [all_X[i] for i in idx]
        all_y = [all_y[i] for i in idx]

    Xs, ys, gs = [], [], []

    for rec_idx, (X_rec, y_rec) in enumerate(zip(all_X, all_y)):

        X_rec = reduce_features(X_rec)

        X_rec, y_rec = trim_to_events(
            X_rec, y_rec,
            noise_label="Noise",
            buffer_frames=CONFIG["noise_buffer"]
        )

        if X_rec.shape[0] < CONFIG["window_size"]:
            continue

        X_win, y_win = create_windows(
            X_rec, y_rec,
            window_size=CONFIG["window_size"],
            stride=CONFIG["stride"]
        )

        Xs.append(X_win)
        ys.append(y_win)
        gs.append(np.full(len(X_win), rec_idx))

    X = np.vstack(Xs)
    y_str = np.concatenate(ys)
    groups = np.concatenate(gs)

    # balance
    G = groups.reshape(-1, 1)
    X_aug = np.hstack([X, G]).astype(np.float64)

    X_aug, y_str = balance_dataset(
        X_aug,
        y_str,
        drug_multiplier=CONFIG["drug_multiplier"]
    )

    groups = X_aug[:, -1].astype(int)
    X = X_aug[:, :-1].astype(np.float32)

    y_enc = le.transform(y_str)

    np.savez(CACHE_PATH, X=X, y_enc=y_enc, groups=groups)
    return X, y_enc, groups, le

# =========================
# CROSS VALIDATION
# =========================
def run_cv(X, y_enc, groups, le):

    gkf = GroupKFold(n_splits=CONFIG["cv_splits"])

    rf_scores = []
    svm_scores = []
    xgb_scores = []

    labels = le.classes_
    n_classes = len(labels)
    total_cm = np.zeros((n_classes, n_classes), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, groups)):

        print(f"\nFold {fold_idx+1}/{CONFIG['cv_splits']}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # Scale ONLY for SVM
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Random Forest (NO scaling)
        rf = RandomForestClassifier(
            n_estimators=CONFIG["n_estimators"],
            max_depth=CONFIG["max_depth"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        rf_scores.append(accuracy_score(y_test, rf_pred))

        # SVM (scaled)
        svm = LinearSVC(class_weight="balanced", max_iter=200, dual=False)
        svm.fit(X_train_sc, y_train)
        svm_pred = svm.predict(X_test_sc)

        svm_scores.append(accuracy_score(y_test, svm_pred))

        # XGBoost (NO scaling)
        xgb = make_xgb_classifier(n_classes)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

        xgb_scores.append(accuracy_score(y_test, xgb_pred))

        # Confusion matrix (ONLY RF)
        total_cm += confusion_matrix(y_test, rf_pred, labels=range(n_classes))

    # ---- after CV ----
    print("\n=== CONFUSION MATRIX (NORMALIZED) ===")
    cm_norm = total_cm / total_cm.sum(axis=1, keepdims=True)

    for i, row in enumerate(cm_norm):
        print(f"{labels[i]:>6} " + " ".join(f"{v:6.2f}" for v in row))

    print("\n==============================")
    print(f"RF mean acc: {np.mean(rf_scores):.4f}")
    print(f"SVM mean acc: {np.mean(svm_scores):.4f}")
    print(f"XGB mean acc: {np.mean(xgb_scores):.4f}")

    
# =========================
# MAIN
# =========================
def main():

    print("Running pipeline...")

    ann = load_annotation()
    all_X, all_y, _, _ = load_all_recordings_librosa(ann)

    X, y_enc, groups, le = build_dataset(all_X, all_y)

    print(f"\nDataset: {X.shape}")

    run_cv(X, y_enc, groups, le)

if __name__ == "__main__":
    main()
