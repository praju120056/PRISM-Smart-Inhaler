"""
train_cnn.py
------------
GroupKFold cross-validation for InhalerCNN (1D CNN).

Mirrors the interface of train.run_cross_validation() so the rest of
the pipeline (evaluate.py, visualize.py, run_pipeline.py) works unchanged.

Key design decisions
--------------------
* Reshapes flat (N, 868) -> (N, 7, 124) before feeding the CNN
* Adam + cosine LR decay (standard for small CNNs)
* Mixed precision (torch.amp) for ~2x GPU speedup on RTX A1000
* Class-weighted cross-entropy to handle Drug imbalance
* Early stopping per fold (patience=7) to prevent overfitting
* Frees GPU memory between folds with torch.cuda.empty_cache()
* Saves best fold model + final ONNX export after last fold

Run standalone:
    python src/train_cnn.py
    python src/train_cnn.py --fast     (3-fold, 20 epochs)
"""

import os
import sys
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model_cnn import build_model

# ---------------------------------------------------------------------------
# Device selection  (CUDA > MPS > CPU, in priority order)
# ---------------------------------------------------------------------------

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    torch.device("cpu")
)

USE_AMP = DEVICE.type == "cuda"   # mixed precision only on CUDA

# ---------------------------------------------------------------------------
# Hyperparameters  (full mode)
# ---------------------------------------------------------------------------

CNN_PARAMS = dict(
    epochs       = 40,
    batch_size   = 512,    # fits comfortably in 4 GB VRAM
    lr           = 1e-3,
    weight_decay = 1e-4,
    patience     = 7,      # early stopping: epochs without improvement
    base_ch      = 128,
    dropout      = 0.25,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        pin_memory  = DEVICE.type == "cuda",
        num_workers = 0,   # keep 0 on Windows to avoid multiprocessing issues
        drop_last   = False,
    )


def _class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights so Drug (rare) is treated equally to Noise (common).
    Normalised so the weights sum to n_classes (no overall scale change).
    """
    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    counts  = np.maximum(counts, 1)           # avoid /0
    weights = n_classes / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def _train_one_fold(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    n_classes:  int,
    n_frames:   int,
    n_features: int,
    params:     dict,
) -> tuple:
    """
    Train InhalerCNN for one CV fold.

    Returns
    -------
    model      : best checkpoint loaded back onto DEVICE
    best_acc   : float  best validation accuracy achieved
    history    : list[float]  per-epoch val accuracy
    """
    model = build_model(
        n_classes  = n_classes,
        n_features = n_features,
        n_frames   = n_frames,
        base_ch    = params["base_ch"],
        dropout    = params["dropout"],
    ).to(DEVICE)

    weights   = _class_weights(y_train, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["epochs"], eta_min=1e-5,
    )

    train_loader = _make_loader(X_train, y_train, params["batch_size"], shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   params["batch_size"], shuffle=False)

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    best_acc   = 0.0
    best_state = None
    patience   = 0
    history    = []

    from tqdm import tqdm
    pbar = tqdm(range(1, params["epochs"] + 1), desc="      Training Epochs", leave=False)
    for epoch in pbar:

        # ── train ─────────────────────────────────────────────────
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = criterion(model(X_b), y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # ── validate ─────────────────────────────────────────────
        model.eval()
        preds_list = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                X_b = X_b.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    logits = model(X_b)
                preds_list.append(logits.argmax(1).cpu())

        val_pred = torch.cat(preds_list).numpy()
        val_acc  = accuracy_score(y_val, val_pred)
        history.append(val_acc)
        
        pbar.set_postfix(val_acc=f"{val_acc:.4f}", best_acc=f"{best_acc:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        # progress print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"      epoch {epoch:3d}/{params['epochs']}  "
                f"val_acc={val_acc:.4f}  best={best_acc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # early stopping
        if val_acc > best_acc + 1e-5:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= params["patience"]:
                print(f"      early stop @ epoch {epoch}  best={best_acc:.4f}")
                break

    # restore best checkpoint
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_acc, history


# ---------------------------------------------------------------------------
# Public API  (mirrors train.run_cross_validation)
# ---------------------------------------------------------------------------

def run_cnn_cv(
    X:       np.ndarray,
    y:       np.ndarray,
    groups:  np.ndarray,
    le,
    n_splits: int  = None,
    params:   dict = None,
    save_onnx: bool = True,
) -> dict:
    """
    GroupKFold CV for InhalerCNN.

    Accepts the same (X, y, groups, le) produced by run_pipeline.py.
    X is the flat (N, WINDOW_SIZE * LIBROSA_N_FEATURES) array — this
    function reshapes it to (N, n_frames, n_features) internally.

    Returns
    -------
    dict  -- same schema as train.run_cross_validation so evaluate.py
             and visualize.py work without modification.
             CNN results are stored in the 'xgb_results' / 'xgb_cm_total'
             slots since we treat CNN as the primary deployment model.
    """
    if params is None:
        params = CNN_PARAMS

    k_splits   = n_splits if n_splits is not None else config.N_SPLITS
    n_classes  = len(le.classes_)
    n_features = config.LIBROSA_N_FEATURES   # 124
    n_frames   = config.WINDOW_SIZE          # 7

    expected_dim = n_frames * n_features
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Expected X.shape[1]={expected_dim} "
            f"({n_frames} frames x {n_features} features), got {X.shape[1]}. "
            "Check config.WINDOW_SIZE and config.LIBROSA_N_FEATURES."
        )

    print(f"\n  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
    print(f"  AMP    : {USE_AMP}")
    print(f"  Params : {params}")

    # Reshape flat -> (N, n_frames, n_features)
    X_3d = X.reshape(-1, n_frames, n_features)

    # Print model summary once
    _tmp = build_model(n_classes=n_classes, n_features=n_features,
                       n_frames=n_frames, base_ch=params["base_ch"])
    print(f"  CNN params: {_tmp.count_parameters():,}")
    del _tmp

    gkf          = GroupKFold(n_splits=k_splits)
    label_order  = le.transform(le.classes_)
    cnn_results  = []
    cnn_cm_total = np.zeros((n_classes, n_classes), dtype=int)
    best_model_overall = None
    best_acc_overall   = 0.0

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_3d, y, groups=groups)):
        print(
            f"\n  Fold {fold + 1}/{k_splits} -- "
            f"train={len(train_idx):,} | test={len(test_idx):,}"
        )

        X_train, X_test = X_3d[train_idx], X_3d[test_idx]
        y_train, y_test = y[train_idx],    y[test_idx]

        model, fold_best_acc, _ = _train_one_fold(
            X_train, y_train, X_test, y_test,
            n_classes, n_frames, n_features, params,
        )

        # ── test evaluation ────────────────────────────────────────
        model.eval()
        test_loader = _make_loader(X_test, y_test, params["batch_size"], shuffle=False)
        preds_list  = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                X_b = X_b.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    logits = model(X_b)
                preds_list.append(logits.argmax(1).cpu())

        cnn_pred = torch.cat(preds_list).numpy()
        cnn_acc  = accuracy_score(y_test, cnn_pred)
        cnn_rep  = classification_report(
            y_test, cnn_pred,
            target_names  = le.classes_,
            output_dict   = True,
            zero_division = 0,
        )
        cnn_cm_total += confusion_matrix(y_test, cnn_pred, labels=label_order)
        cnn_results.append({
            "fold": fold + 1, "model": "CNN",
            "accuracy": cnn_acc, "report": cnn_rep,
        })
        print(f"    CNN test accuracy: {cnn_acc:.4f}")

        if cnn_acc > best_acc_overall:
            best_acc_overall   = cnn_acc
            best_model_overall = model

        # free VRAM between folds
        if model is not best_model_overall:
            del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── ONNX export (best fold model) ─────────────────────────────
    if save_onnx and best_model_overall is not None:
        onnx_path = os.path.join(config.RESULTS_DIR, "inhaler_cnn.onnx")
        try:
            best_model_overall.export_onnx(onnx_path)
        except Exception as e:
            print(f"  [WARN] ONNX export failed: {e}")
            print("         Install onnx: pip install onnx")
        del best_model_overall

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Return in the same dict schema as run_cross_validation.
    # We reuse the 'xgb_*' slots so evaluate.py / visualize.py require no changes.
    return {
        "rf_results":    [],
        "svm_results":   [],
        "xgb_results":   cnn_results,
        "rf_cm_total":   np.zeros((n_classes, n_classes), dtype=int),
        "svm_cm_total":  np.zeros((n_classes, n_classes), dtype=int),
        "xgb_cm_total":  cnn_cm_total,
        "rf_fi_folds":   [],
        "xgb_fi_folds":  [],   # CNNs don't expose sklearn feature_importances_
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _p = argparse.ArgumentParser()
    _p.add_argument("--fast", action="store_true",
                    help="3-fold, 20 epochs -- quick smoke test")
    _args = _p.parse_args()

    fast_params = dict(CNN_PARAMS, epochs=20, patience=5)
    run_params  = fast_params if _args.fast else CNN_PARAMS
    k           = 3 if _args.fast else config.N_SPLITS

    print("=" * 60)
    print(f"train_cnn.py  --  {'FAST (3-fold, 20 ep)' if _args.fast else 'FULL'}")
    print("=" * 60)

    from sklearn.preprocessing import LabelEncoder
    from loader import load_annotation
    from librosa_extractor import load_all_recordings_librosa
    from feature_extractor import trim_to_events, create_windows, balance_dataset
    from evaluate import aggregate_metrics, save_cv_csv

    ann = load_annotation()
    all_X, all_y, _, skipped = load_all_recordings_librosa(
        ann, data_dir=config.DATA_DIR
    )
    print(f"  Loaded {len(all_X)} recordings, skipped {len(skipped)}")

    Xs, ys, gs = [], [], []
    for rec_idx, (X_rec, y_rec) in enumerate(zip(all_X, all_y)):
        X_rec, y_rec = trim_to_events(X_rec, y_rec, noise_label="Noise",
                                      buffer_frames=config.NOISE_BUFFER)
        if X_rec.shape[0] < config.WINDOW_SIZE:
            continue
        X_win, y_win = create_windows(X_rec, y_rec,
                                      window_size=config.WINDOW_SIZE,
                                      stride=config.WINDOW_STRIDE)
        Xs.append(X_win)
        ys.append(y_win)
        gs.append(np.full(X_win.shape[0], rec_idx, dtype=int))

    X_all  = np.vstack(Xs)
    y_all  = np.concatenate(ys)
    groups = np.concatenate(gs)

    G_col = groups.reshape(-1, 1).astype(np.float64)
    X_aug = np.hstack([X_all.astype(np.float64), G_col])
    rng   = np.random.default_rng(42)
    X_aug, y_all = balance_dataset(
        X_aug, y_all,
        drug_multiplier=config.DRUG_MULTIPLIER,
        noise_label="Noise", drug_label="Drug", rng=rng,
    )
    groups = X_aug[:, -1].astype(int)
    X_all  = X_aug[:, :-1].astype(np.float32)

    le = LabelEncoder()
    le.fit(config.LABEL_NAMES)
    y_int = le.transform(y_all)

    print(f"\n  Dataset: {X_all.shape}  classes={list(le.classes_)}")

    cv = run_cnn_cv(X_all, y_int, groups, le, n_splits=k, params=run_params)
    summary = aggregate_metrics(cv["xgb_results"], "CNN", le)
    save_cv_csv([], [], cv["xgb_results"], le)

    print("\n" + "=" * 60)
    print(f"CNN mean accuracy: {summary['mean_acc']:.4f} +/- {summary['std_acc']:.4f}")
    print("=" * 60)
