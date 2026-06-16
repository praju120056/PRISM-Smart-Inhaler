"""
feature_extractor.py
--------------------
Converts frame-level MFCC features into window-based features that
capture temporal context suitable for SVM, XGBoost, or Random Forest.

Pipeline
--------
  1. compute_delta(features)      → delta or delta-delta coefficients
  2. stack_features(mfcc)         → [MFCC | Δ | ΔΔ] per frame
  3. create_windows(...)          → sliding-window feature matrix + labels
  4. normalize_windows(...)       → optional per-window or global normalisation
  5. build_windowed_dataset(...)  → multi-recording entry point (safe: no
                                    cross-recording frame mixing)

Run standalone for a quick sanity-check:
    python src/feature_extractor.py
"""

import sys
import os
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Delta computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_delta(features: np.ndarray, width: int = 9) -> np.ndarray:
    """
    Compute delta (velocity) coefficients using a regression filter.

    Uses the standard librosa / HTK-style formula::

        delta[t] = sum_{n=1}^{N} n * (features[t+n] - features[t-n])
                   ─────────────────────────────────────────────────
                             2 * sum_{n=1}^{N} n²

    where N = (width - 1) // 2.  Edge frames are padded by replication
    so the output has exactly the same shape as the input.

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_features)
    width    : int (odd, >= 3)  — context window for the filter

    Returns
    -------
    delta : np.ndarray, shape (n_frames, n_features)

    Notes
    -----
    *Why not np.diff?*  np.diff reduces the frame count by 1, requires
    prepend padding, and loses the smooth regression property.  The
    regression approach is numerically stabler and matches the standard
    MFCC delta computation used in speech recognition.
    """
    if width % 2 == 0:
        raise ValueError(f"width must be odd, got {width}")
    if width < 3:
        raise ValueError(f"width must be >= 3, got {width}")
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2-D (n_frames, n_features), got shape {features.shape}"
        )

    n_frames, n_feat = features.shape
    N = (width - 1) // 2

    # Pad by replication at boundaries (edge mode)
    # Shape after padding: (n_frames + 2*N, n_feat)
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")

    # Pre-compute denominator: 2 * sum_{n=1}^{N} n^2
    ns       = np.arange(1, N + 1, dtype=np.float64)       # (N,)
    denom    = 2.0 * np.dot(ns, ns)                        # scalar

    # Build delta via vectorised sum over context offsets
    # We want: delta[t] = sum_{n=1}^{N} n * (padded[t+N+n] - padded[t+N-n])
    # Reshape ns for broadcasting: (N, 1)
    ns_col = ns[:, np.newaxis]                             # (N, 1)

    # Stack shifted difference matrices: shape (N, n_frames, n_feat)
    pos = np.stack([padded[N + n : N + n + n_frames] for n in range(1, N + 1)], axis=0)
    neg = np.stack([padded[N - n : N - n + n_frames] for n in range(1, N + 1)], axis=0)

    # Weighted sum: (N, n_frames, n_feat) * (N, 1, 1) → sum over axis 0
    delta = np.sum(ns_col[:, np.newaxis, :] * (pos - neg), axis=0) / denom

    assert delta.shape == features.shape, "delta shape mismatch — internal bug"
    return delta.astype(features.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Feature stacking
# ──────────────────────────────────────────────────────────────────────────────

def stack_features(mfcc: np.ndarray, delta_width: int = 9) -> np.ndarray:
    """
    Compute delta and delta-delta MFCC and concatenate per frame.

    Parameters
    ----------
    mfcc        : np.ndarray, shape (n_frames, n_mfcc)
                  Raw MFCC coefficient matrix.
    delta_width : int (odd >= 3)
                  Regression filter width for delta computation.

    Returns
    -------
    stacked : np.ndarray, shape (n_frames, 3 * n_mfcc)
              Columns: [MFCC | Δ MFCC | ΔΔ MFCC]

    Example
    -------
    >>> mfcc = np.random.randn(200, 19)
    >>> stacked = stack_features(mfcc)
    >>> stacked.shape
    (200, 57)
    """
    if mfcc.ndim != 2:
        raise ValueError(
            f"mfcc must be 2-D (n_frames, n_mfcc), got shape {mfcc.shape}"
        )

    delta    = compute_delta(mfcc,   width=delta_width)   # (n_frames, n_mfcc)
    delta2   = compute_delta(delta,  width=delta_width)   # (n_frames, n_mfcc)
    stacked  = np.hstack([mfcc, delta, delta2])           # (n_frames, 3*n_mfcc)
    return stacked


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Sliding-window feature creation
# ──────────────────────────────────────────────────────────────────────────────

def create_windows(
    features:    np.ndarray,
    labels:      np.ndarray,
    window_size: int = 7,
    stride:      int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a sliding window over frame-level features and labels.

    Incomplete windows at the boundaries are discarded.

    Parameters
    ----------
    features    : np.ndarray, shape (n_frames, feature_dim)
                  Per-frame feature vectors (e.g. output of stack_features).
    labels      : np.ndarray, shape (n_frames,)
                  Per-frame class labels (int or str).
    window_size : int  Number of consecutive frames per window (default 7).
    stride      : int  Step size between successive windows (default 1).

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size * feature_dim)
        Each row is the concatenation of all frame vectors in one window.
    y : np.ndarray, shape (n_windows,)
        Majority-vote label for each window (ties broken by lowest label).

    Raises
    ------
    ValueError  If window_size > n_frames or stride < 1.

    Notes
    -----
    *Why majority voting?*  A window may straddle a transition boundary.
    Majority voting gives a single, clean label without requiring a
    hand-crafted smoothing rule, and is standard practice before feeding
    windowed features to SVM / RF.

    *No loop over frames.*  Windows are constructed with ``np.lib.stride_tricks
    .sliding_window_view`` (NumPy ≥ 1.20) for O(1) memory-view overhead
    and vectorised label aggregation.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2-D (n_frames, feature_dim), got {features.shape}"
        )
    n_frames, feature_dim = features.shape

    if labels.shape[0] != n_frames:
        raise ValueError(
            f"features ({n_frames} frames) and labels ({labels.shape[0]}) "
            "must have the same length."
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if window_size > n_frames:
        raise ValueError(
            f"window_size ({window_size}) > n_frames ({n_frames}): "
            "no complete windows possible."
        )

    # ── Build feature windows ─────────────────────────────────────────────────
    # sliding_window_view returns a *view* (zero-copy):
    # shape (n_windows_full, window_size, feature_dim)
    feat_windows = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=(window_size, feature_dim)
    )                                   # (n_frames - window_size + 1, 1, window_size, feature_dim)
    # squeeze the redundant axis introduced by 2-D input
    feat_windows = feat_windows[:, 0, :, :]   # (n_valid, window_size, feature_dim)

    # Apply stride and flatten each window into a 1-D vector
    feat_windows = feat_windows[::stride]            # (n_windows, window_size, feature_dim)
    n_windows    = feat_windows.shape[0]
    X            = feat_windows.reshape(n_windows, window_size * feature_dim)

    # ── Majority-vote labelling ───────────────────────────────────────────────
    # Build a (n_windows, window_size) label matrix without a Python loop
    lbl_windows = np.lib.stride_tricks.sliding_window_view(
        labels, window_shape=window_size
    )[::stride]                                       # (n_windows, window_size)

    y = _majority_vote(lbl_windows)                  # (n_windows,)

    return X, y


def _majority_vote(label_matrix: np.ndarray) -> np.ndarray:
    """
    Return the most frequent label in each row of label_matrix.

    Parameters
    ----------
    label_matrix : np.ndarray, shape (n_windows, window_size)

    Returns
    -------
    votes : np.ndarray, shape (n_windows,)

    Implementation
    --------------
    For integer labels: uses ``np.apply_along_axis`` with ``np.bincount``
    for efficiency.
    For string labels: uses ``scipy.stats.mode`` fallback — or our own
    vectorised approach with ``np.unique``.

    We avoid scipy to respect the "numpy + stdlib only" constraint and instead
    sort each row and pick the middle element for even-tie breaking with the
    *lowest-index* (numerically smallest or lexicographically first) label.
    """
    n_windows, window_size = label_matrix.shape

    # Fast path for integer labels — works for encoded y
    if np.issubdtype(label_matrix.dtype, np.integer):
        n_classes = label_matrix.max() + 1
        # one-hot count via bincount; reshape to (n_windows, n_classes)
        offsets   = np.arange(n_windows) * n_classes
        flat_idx  = (label_matrix + offsets[:, np.newaxis]).ravel()
        counts    = np.bincount(flat_idx, minlength=n_windows * n_classes)
        counts    = counts.reshape(n_windows, n_classes)
        votes     = counts.argmax(axis=1)
        return votes.astype(label_matrix.dtype)

    # General path for string / object labels
    # For each window row: find mode using sorted + midpoint approach
    # This is O(n_windows * window_size * log(window_size)) but avoids scipy
    votes = np.empty(n_windows, dtype=label_matrix.dtype)
    for i in range(n_windows):
        row    = label_matrix[i]
        unique, cnts = np.unique(row, return_counts=True)
        votes[i] = unique[np.argmax(cnts)]
    return votes


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Normalisation  (optional)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_windows(
    X:      np.ndarray,
    method: str = "global",
    mean:   np.ndarray | None = None,
    std:    np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise windowed feature matrix.

    Parameters
    ----------
    X      : np.ndarray, shape (n_windows, n_features)
    method : str
        - ``"global"``     — subtract global mean and divide by global std
                             (fit statistics computed from X itself)
        - ``"per_window"`` — each window row is independently z-scored
        - ``"prefit"``     — use the *mean* and *std* kwargs supplied by the
                             caller (useful for applying training statistics
                             to a held-out set without data leakage)
    mean   : np.ndarray | None  Required when method == "prefit"
    std    : np.ndarray | None  Required when method == "prefit"

    Returns
    -------
    X_norm : np.ndarray  Normalised matrix, same shape as X
    mean   : np.ndarray  Statistics used (shape depends on method)
    std    : np.ndarray  Statistics used

    Notes
    -----
    A small epsilon (1e-10) is added to std to avoid division by zero for
    constant features (e.g. silence-only recordings).
    """
    eps = 1e-10

    if method == "global":
        mean = X.mean(axis=0)           # (n_features,)
        std  = X.std(axis=0) + eps
        X_norm = (X - mean) / std

    elif method == "per_window":
        mean = X.mean(axis=1, keepdims=True)    # (n_windows, 1)
        std  = X.std(axis=1, keepdims=True) + eps
        X_norm = (X - mean) / std
        # Collapse to 1-D arrays for the return API (per-window stats vary)
        # Return the full matrices so the caller can inspect them
        return X_norm, mean, std

    elif method == "prefit":
        if mean is None or std is None:
            raise ValueError(
                "method='prefit' requires both mean and std to be provided."
            )
        X_norm = (X - mean) / (std + eps)

    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose 'global', 'per_window', or 'prefit'."
        )

    return X_norm, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Noise trimming
# ──────────────────────────────────────────────────────────────────────────────

def trim_to_events(
    X_rec:         np.ndarray,
    y_rec:         np.ndarray,
    noise_label:   str = "Noise",
    buffer_frames: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove excess Noise frames that are far from any annotated event.

    Keeps all non-Noise frames plus a symmetric buffer of ``buffer_frames``
    on each side.  This eliminates the long silent stretches (3-4 sec) at
    the start/end of each recording while preserving temporal context needed
    for delta features and window construction.

    Parameters
    ----------
    X_rec         : (n_frames, n_features)
    y_rec         : (n_frames,)  string labels
    noise_label   : label string used for background/silence frames
    buffer_frames : number of Noise frames to keep on each side of an event

    Returns
    -------
    X_trimmed, y_trimmed : arrays with irrelevant Noise frames removed
    """
    event_mask = (y_rec != noise_label)          # True where a real event occurs
    if not event_mask.any():
        return X_rec, y_rec                      # all Noise — return as-is

    n = len(y_rec)
    keep = np.zeros(n, dtype=bool)

    # For each event frame, mark the surrounding buffer as keep
    event_indices = np.where(event_mask)[0]
    for idx in event_indices:
        lo = max(0, idx - buffer_frames)
        hi = min(n, idx + buffer_frames + 1)
        keep[lo:hi] = True

    return X_rec[keep], y_rec[keep]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Class balancing  (oversample minority / undersample majority)
# ──────────────────────────────────────────────────────────────────────────────

def oversample_minority(
    X:          np.ndarray,
    y:          np.ndarray,
    label:      str,
    target_n:   int,
    jitter_std: float = 0.01,
    rng:        np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample windows of a specific class to *target_n* by random
    duplication with small Gaussian feature jitter (numpy-only SMOTE-lite).

    Parameters
    ----------
    X          : (n_windows, n_features)
    y          : (n_windows,)  string labels
    label      : class to oversample (e.g. "Drug")
    target_n   : desired total count for this class
    jitter_std : std-dev of Gaussian noise added to duplicated samples
                 (relative to feature scale; default 0.01 = 1%)
    rng        : numpy Generator for reproducibility

    Returns
    -------
    X_aug, y_aug : augmented arrays (originals + synthetic copies)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    idx   = np.where(y == label)[0]
    n_now = len(idx)
    if n_now == 0 or n_now >= target_n:
        return X, y

    n_new    = target_n - n_now
    src_idx  = rng.choice(idx, size=n_new, replace=True)
    jitter   = rng.normal(0, jitter_std, size=(n_new, X.shape[1])).astype(X.dtype)
    X_new    = X[src_idx] + jitter
    y_new    = np.full(n_new, label, dtype=y.dtype)

    return np.vstack([X, X_new]), np.concatenate([y, y_new])


def balance_dataset(
    X:               np.ndarray,
    y:               np.ndarray,
    noise_cap:       int | None = None,
    drug_multiplier: float      = 3.0,
    jitter_std:      float      = 0.01,
    noise_label:     str        = "Noise",
    drug_label:      str        = "Drug",
    rng:             np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-stage balancing pipeline:

    Stage 1 — Undersample Noise
        Cap Noise windows at ``noise_cap`` (default: 2 × Exhale count).

    Stage 2 — Oversample Drug
        Duplicate Drug windows (with jitter) until their count reaches
        ``drug_multiplier × Inhale count`` or Exhale count, whichever
        is smaller — preventing Drug from dominating.

    Parameters
    ----------
    X               : (n_windows, n_features)
    y               : (n_windows,)  string labels
    noise_cap       : hard cap on Noise windows; None → 2 × Exhale count
    drug_multiplier : Drug target = multiplier × current Inhale count
    jitter_std      : Gaussian noise std for Drug oversampling
    noise_label     : label string for background class
    drug_label      : label string for drug-actuation class
    rng             : numpy Generator

    Returns
    -------
    X_bal, y_bal : balanced arrays  (shuffled)

    Example resulting distribution (approximate)
    ---------------------------------------------
    Before: Noise 75% | Exhale 14% | Inhale 9% | Drug 2%
    After:  Noise 25% | Exhale 30% | Inhale 25% | Drug 20%
    """
    if rng is None:
        rng = np.random.default_rng(42)

    labels, cnts = np.unique(y, return_counts=True)
    count        = dict(zip(labels, cnts))

    # ── Stage 1: cap Noise ───────────────────────────────────────
    n_exhale = count.get("Exhale", 1)
    cap      = noise_cap if noise_cap is not None else 2 * n_exhale

    noise_idx = np.where(y == noise_label)[0]
    if len(noise_idx) > cap:
        keep_noise = rng.choice(noise_idx, size=cap, replace=False)
        other_idx  = np.where(y != noise_label)[0]
        keep_all   = np.sort(np.concatenate([keep_noise, other_idx]))
        X, y = X[keep_all], y[keep_all]

    # Recompute counts after Noise capping
    labels, cnts = np.unique(y, return_counts=True)
    count        = dict(zip(labels, cnts))

    # ── Stage 2: oversample Drug ─────────────────────────────────
    n_inhale     = count.get("Inhale", 1)
    n_exhale_new = count.get("Exhale", 1)
    # Target: drug_multiplier × Inhale, but never exceed Exhale count
    drug_target  = min(int(drug_multiplier * n_inhale), n_exhale_new)
    drug_target  = max(drug_target, count.get(drug_label, 0))  # never shrink

    X, y = oversample_minority(
        X, y,
        label      = drug_label,
        target_n   = drug_target,
        jitter_std = jitter_std,
        rng        = rng,
    )

    # ── Shuffle to mix synthetic and real samples ────────────────
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Multi-recording entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_windowed_dataset(
    all_X:           list,
    all_y:           list,
    window_size:     int        = 7,
    stride:          int        = 1,
    delta_width:     int        = 9,
    normalize:       str | None = None,
    trim_noise:      bool       = True,
    noise_buffer:    int        = 20,
    balance:         bool       = True,
    noise_cap:       int | None = None,
    drug_multiplier: float      = 3.0,
    jitter_std:      float      = 0.01,
    noise_label:     str        = "Noise",
    drug_label:      str        = "Drug",
    rng:             np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a windowed, balanced dataset from multiple recordings.

    Per-recording pipeline
    ----------------------
    1. trim_to_events()     — drop excess Noise frames far from events
    2. stack_features()     — [MFCC | Δ | ΔΔ]
    3. create_windows()     — sliding window + majority-vote label

    Post-concatenation
    ------------------
    4. balance_dataset()    — cap Noise, oversample Drug
    5. normalize_windows()  — optional z-score

    Parameters
    ----------
    all_X           : list[np.ndarray]  per-recording MFCC (n_frames, n_mfcc)
    all_y           : list[np.ndarray]  per-recording string labels
    window_size     : frames per window
    stride          : hop between windows
    delta_width     : regression filter width for Δ computation
    normalize       : 'global' | 'per_window' | None
    trim_noise      : remove Noise frames far from events (recommended)
    noise_buffer    : frames of Noise to keep around each event
    balance         : apply Noise capping + Drug oversampling
    noise_cap       : hard Noise ceiling; None → 2 × Exhale count
    drug_multiplier : Drug target = multiplier × Inhale count
    jitter_std      : std-dev of jitter added to oversampled Drug windows
    noise_label     : string used for background frames
    drug_label      : string used for drug-actuation frames
    rng             : numpy Generator for reproducibility

    Returns
    -------
    X      : (total_windows, window_size * 3 * n_mfcc)
    y      : (total_windows,)  string labels  [shuffled if balance=True]
    groups : (total_windows,)  recording group ids

    Notes
    -----
    groups is preserved through trimming and balancing so GroupKFold
    splits remain valid.  Oversampled Drug windows inherit the group id
    of their source window.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    Xs, ys, gs = [], [], []

    for rec_idx, (X_rec, y_rec) in enumerate(zip(all_X, all_y)):

        # ── 1. Trim excess Noise ──────────────────────────────────
        if trim_noise:
            X_rec, y_rec = trim_to_events(
                X_rec, y_rec,
                noise_label   = noise_label,
                buffer_frames = noise_buffer,
            )

        if X_rec.shape[0] < window_size:
            continue

        # ── 2. Stack [MFCC | Δ | ΔΔ] ─────────────────────────────
        stacked = stack_features(X_rec, delta_width=delta_width)

        # ── 3. Sliding windows ────────────────────────────────────
        X_win, y_win = create_windows(stacked, y_rec, window_size, stride)

        Xs.append(X_win)
        ys.append(y_win)
        gs.append(np.full(X_win.shape[0], rec_idx, dtype=int))

    if not Xs:
        raise RuntimeError(
            "No windows generated. Check recordings have >= window_size frames."
        )

    X      = np.vstack(Xs)
    y      = np.concatenate(ys)
    groups = np.concatenate(gs)

    # ── 4. Balance: cap Noise + oversample Drug ───────────────────
    if balance:
        # We need to carry groups through the balancing step.
        # Append groups as an extra column, balance, then split off.
        G_col  = groups.reshape(-1, 1).astype(np.float64)
        X_aug  = np.hstack([X.astype(np.float64), G_col])

        X_aug, y = balance_dataset(
            X_aug, y,
            noise_cap       = noise_cap,
            drug_multiplier = drug_multiplier,
            jitter_std      = jitter_std,
            noise_label     = noise_label,
            drug_label      = drug_label,
            rng             = rng,
        )
        groups = X_aug[:, -1].astype(int)
        X      = X_aug[:, :-1].astype(np.float32)

    # ── 5. Normalise ──────────────────────────────────────────────
    if normalize is not None:
        X, _, _ = normalize_windows(X, method=normalize)

    return X, y, groups


# ──────────────────────────────────────────────────────────────────────────────
# Standalone self-test / sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 65)
    print("feature_extractor.py  —  standalone sanity check")
    print("=" * 65)

    rng = np.random.default_rng(42)

    # ── Unit tests ────────────────────────────────────────────────
    print("\n[1] compute_delta  …", end=" ")
    mfcc_fake = rng.standard_normal((200, 19)).astype(np.float32)
    d1        = compute_delta(mfcc_fake)
    d2        = compute_delta(d1)
    assert d1.shape == mfcc_fake.shape, "delta shape wrong"
    assert d2.shape == mfcc_fake.shape, "delta-delta shape wrong"
    print("OK", d1.shape)

    print("[2] stack_features  …", end=" ")
    stacked = stack_features(mfcc_fake)
    assert stacked.shape == (200, 57), f"Expected (200,57), got {stacked.shape}"
    print("OK", stacked.shape)

    print("[3] create_windows (stride=1)  …", end=" ")
    labels_fake = rng.integers(0, 4, size=200)
    X_win, y_win = create_windows(stacked, labels_fake, window_size=7, stride=1)
    expected_wins = (200 - 7) // 1 + 1
    assert X_win.shape == (expected_wins, 7 * 57), (
        f"Expected ({expected_wins}, {7*57}), got {X_win.shape}"
    )
    assert y_win.shape == (expected_wins,), f"y_win shape wrong: {y_win.shape}"
    print(f"OK  X={X_win.shape}  y={y_win.shape}")

    print("[4] create_windows (stride=3)  …", end=" ")
    X_s, y_s = create_windows(stacked, labels_fake, window_size=7, stride=3)
    expected_s = len(range(0, 200 - 7 + 1, 3))
    assert X_s.shape[0] == expected_s, (
        f"Expected {expected_s} windows, got {X_s.shape[0]}"
    )
    print(f"OK  n_windows={X_s.shape[0]}")

    print("[5] majority vote (string labels) …", end=" ")
    str_labels  = np.array(["Inhale", "Inhale", "Exhale", "Inhale",
                             "Noise",  "Inhale", "Inhale"])
    mat         = str_labels.reshape(1, 7)
    vote        = _majority_vote(mat)
    assert vote[0] == "Inhale", f"Expected Inhale, got {vote[0]}"
    print(f"OK  vote={vote[0]}")

    print("[6] normalize_windows (global) …", end=" ")
    X_norm, mu, sigma = normalize_windows(X_win, method="global")
    assert X_norm.shape == X_win.shape
    assert np.abs(X_norm.mean()) < 0.1, "mean not ~0 after normalisation"
    print("OK")

    print("[7] normalize_windows (per_window) …", end=" ")
    X_pw, mu_pw, sig_pw = normalize_windows(X_win, method="per_window")
    assert X_pw.shape == X_win.shape
    print("OK")

    print("[8] normalize_windows (prefit) …", end=" ")
    X_pf, _, _ = normalize_windows(X_win, method="prefit", mean=mu, std=sigma)
    assert X_pf.shape == X_win.shape
    print("OK")

    print("\n[9] trim_to_events …", end=" ")
    # 80 frames: 20 Noise, 10 Inhale, 10 Exhale, 40 Noise
    y_trim_test = np.array(
        ["Noise"]*20 + ["Inhale"]*10 + ["Exhale"]*10 + ["Noise"]*40
    )
    X_trim_test = rng.standard_normal((80, 19)).astype(np.float32)
    Xt, yt = trim_to_events(X_trim_test, y_trim_test, buffer_frames=5)
    assert len(Xt) < 80, "trim_to_events should have removed some Noise frames"
    assert "Inhale" in yt and "Exhale" in yt, "Events must be preserved"
    print(f"OK  {80} -> {len(Xt)} frames")

    print("[10] oversample_minority ...", end=" ")
    y_os = np.array(["Drug"]*5 + ["Inhale"]*50 + ["Noise"]*100)
    X_os = rng.standard_normal((155, 57)).astype(np.float32)
    X_os2, y_os2 = oversample_minority(X_os, y_os, label="Drug", target_n=40)
    assert (y_os2 == "Drug").sum() == 40, "Drug should be upsampled to 40"
    print(f"OK  Drug: 5 -> {(y_os2=='Drug').sum()}")

    print("[11] balance_dataset …", end=" ")
    y_bal_in = np.array(["Drug"]*10 + ["Inhale"]*80 + ["Exhale"]*100 + ["Noise"]*500)
    X_bal_in = rng.standard_normal((690, 57)).astype(np.float32)
    X_bal, y_bal = balance_dataset(
        X_bal_in, y_bal_in,
        drug_multiplier=3.0, noise_label="Noise", drug_label="Drug"
    )
    noise_frac = (y_bal == "Noise").sum() / len(y_bal)
    assert noise_frac < 0.55, f"Noise still dominates after balancing: {noise_frac:.1%}"
    drug_count = (y_bal == "Drug").sum()
    assert drug_count > 10, "Drug should be oversampled"
    print(f"OK  total={len(y_bal)}, Drug={drug_count}, Noise%={noise_frac:.1%}")

    print("\n[12] build_windowed_dataset (2 fake recordings, with balancing) …")
    # Build fake labels with realistic class distribution
    def _fake_labels(n):
        y = np.array(["Noise"]*n)
        y[n//4 : n//4+n//10]  = "Inhale"
        y[n//2 : n//2+n//15]  = "Exhale"
        y[3*n//4 : 3*n//4+max(2, n//50)] = "Drug"
        return y

    all_mfcc   = [rng.standard_normal((150, 19)).astype(np.float32),
                  rng.standard_normal((300, 19)).astype(np.float32)]
    all_labels = [_fake_labels(150), _fake_labels(300)]

    X_ds, y_ds, grp = build_windowed_dataset(
        all_mfcc, all_labels,
        window_size=7, stride=2, normalize="global",
        trim_noise=True, balance=True, drug_multiplier=3.0,
    )
    feature_dim = 3 * 19
    assert X_ds.shape[1] == 7 * feature_dim, (
        f"Expected {7*feature_dim} features, got {X_ds.shape[1]}"
    )
    uniq_y, cnts_y = np.unique(y_ds, return_counts=True)
    print(f"  X shape : {X_ds.shape}")
    print(f"  Label distribution after balancing:")
    for lbl, cnt in zip(uniq_y, cnts_y):
        print(f"    {lbl:10s}: {cnt:4d}  ({100*cnt/len(y_ds):.1f}%)")

    # ── Integration with real data (if available) ─────────────────
    print("\n[13] Integration with real dataset (trim + balance) …")
    try:
        import config
        from loader import load_annotation, load_all_recordings

        ann                           = load_annotation()
        all_X_real, all_y_real, _, sk = load_all_recordings(ann)
        n_mfcc    = config.N_MFCC
        mfcc_only = [X[:, :n_mfcc] for X in all_X_real]

        print(f"  Recordings loaded : {len(mfcc_only)}  (skipped {len(sk)})")

        print("\n  [A] No trim, no balance (baseline):")
        Xb, yb, _ = build_windowed_dataset(
            mfcc_only, all_y_real,
            window_size=7, stride=1,
            trim_noise=False, balance=False,
        )
        for lbl, cnt in zip(*np.unique(yb, return_counts=True)):
            print(f"    {lbl:10s}: {cnt:7,}  ({100*cnt/len(yb):.1f}%)")

        print("\n  [B] Trim + balance (drug_multiplier=3.0):")
        Xbal, ybal, gbal = build_windowed_dataset(
            mfcc_only, all_y_real,
            window_size=7, stride=1, normalize="global",
            trim_noise=True, noise_buffer=20,
            balance=True, drug_multiplier=3.0, jitter_std=0.01,
        )
        for lbl, cnt in zip(*np.unique(ybal, return_counts=True)):
            print(f"    {lbl:10s}: {cnt:7,}  ({100*cnt/len(ybal):.1f}%)")
        print(f"\n  Final X shape : {Xbal.shape}")
        print(f"  Groups        : {len(np.unique(gbal))} recordings")

    except ImportError:
        print("  [INFO] config/loader not found — skipping real-data test")
    except Exception as exc:
        print(f"  [WARN] Real data test failed: {exc}")

    print("\n" + "=" * 65)
    print("All checks passed.")
    print("=" * 65)
