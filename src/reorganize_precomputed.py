"""
reorganize_precomputed.py
-------------------------
One-time script. Moves all precomputed CSV files from data/ into:

    data/precomputed/<recording_base>/<file>.csv

WAV files and annotation.csv remain in data/.

Safe to run multiple times (idempotent).

Usage:
    python src/reorganize_precomputed.py
"""

import os
import sys
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────────
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(_SRC_DIR)
DATA_DIR      = os.path.join(ROOT_DIR, "data")
PRECOMP_DIR   = os.path.join(DATA_DIR, "precomputed")

CSV_SUFFIXES  = ("_mfcc.csv", "_zcr.csv", "_spect.csv", "_cwt.csv", "_cepst.csv")


def base_from_csv(name: str) -> str | None:
    """Extract recording base name from a CSV filename."""
    for suffix in CSV_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def main() -> None:
    csv_files = [
        f for f in os.listdir(DATA_DIR)
        if any(f.endswith(s) for s in CSV_SUFFIXES)
    ]

    if not csv_files:
        print("No CSV files found in data/ — already reorganized or nothing to move.")
        return

    print(f"Found {len(csv_files)} CSV files.")
    os.makedirs(PRECOMP_DIR, exist_ok=True)

    moved   = 0
    skipped = 0

    for fname in sorted(csv_files):
        base = base_from_csv(fname)
        if base is None:
            skipped += 1
            continue

        dest_dir = os.path.join(PRECOMP_DIR, base)
        os.makedirs(dest_dir, exist_ok=True)

        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(dest_dir, fname)

        if os.path.exists(dst):
            # Already moved in a previous run; remove the source if it still exists
            if os.path.exists(src):
                os.remove(src)
            skipped += 1
            continue

        shutil.move(src, dst)
        moved += 1

    print(f"Moved   : {moved} files")
    print(f"Skipped : {skipped} files (already in place or unrecognised)")
    print(f"Target  : {PRECOMP_DIR}")
    print("WAV files and annotation.csv remain in data/")

    # Quick verification
    subdirs = [d for d in os.listdir(PRECOMP_DIR)
               if os.path.isdir(os.path.join(PRECOMP_DIR, d))]
    print(f"\nRecording subfolders created: {len(subdirs)}")


if __name__ == "__main__":
    main()
