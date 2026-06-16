"""
dataset.py
----------
Stacks per-recording arrays into a single dataset (X, y, groups)
and applies label encoding.

Run standalone to inspect dataset statistics:
    python src/dataset.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config
from loader import load_annotation, load_all_recordings


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def build_dataset(all_X, all_y, all_groups):
    """
    Stack per-recording lists into numpy arrays and encode labels.

    Parameters
    ----------
    all_X      : list[np.ndarray]
    all_y      : list[np.ndarray]  string labels
    all_groups : list[np.ndarray]  int group ids

    Returns
    -------
    X      : np.ndarray  (total_frames, n_features)
    y      : np.ndarray  (total_frames,) int encoded labels
    y_str  : np.ndarray  (total_frames,) original string labels
    groups : np.ndarray  (total_frames,) recording group ids
    le     : LabelEncoder  fitted encoder (use le.classes_ for names)
    """
    X      = np.vstack(all_X)
    y_str  = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    le = LabelEncoder()
    le.fit(config.LABEL_NAMES)  # fixed order — deterministic across runs
    y  = le.transform(y_str)

    return X, y, y_str, groups, le


def print_dataset_summary(X, y_str, groups, loaded: int):
    """Print a human-readable summary of the assembled dataset."""
    total        = len(y_str)
    unique_lbls, counts = np.unique(y_str, return_counts=True)

    print(f"  Recordings   : {loaded}")
    print(f"  Total frames : {X.shape[0]:,}")
    print(f"  Features     : {X.shape[1]}")
    print(f"  Groups       : {len(np.unique(groups))}")
    print("\n  Label distribution:")
    for lbl, cnt in zip(unique_lbls, counts):
        print(f"    {lbl:10s}: {cnt:7,}  ({100 * cnt / total:.1f}%)")


# ──────────────────────────────────────────────────────────────
# Standalone debug entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py — standalone check")
    print("=" * 60)

    ann                              = load_annotation()
    all_X, all_y, all_groups, skipped = load_all_recordings(ann)

    print(f"\nLoaded: {len(all_X)}, Skipped: {len(skipped)}")

    X, y, y_str, groups, le = build_dataset(all_X, all_y, all_groups)

    print("\nDataset summary:")
    print_dataset_summary(X, y_str, groups, loaded=len(all_X))
    print(f"\n  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
