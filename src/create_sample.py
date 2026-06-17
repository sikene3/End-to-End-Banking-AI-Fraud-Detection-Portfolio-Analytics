"""Stratified downsample of ``train_features.parquet`` to 250K rows.

GitHub rejects files > 100 MB, but ``train_features.parquet`` is ~500 MB.
This script produces a deployable artefact -- ``train_features_sample.parquet``
-- that:

  * Contains exactly 250,000 rows.
  * Preserves the ~0.15% fraud rate of the source frame (stratified split
    on ``is_fraud``).
  * Is small enough to ship in a GitHub repo (< 100 MB) and to load into
    Streamlit Community Cloud in a single read.

Usage
-----
    python create_sample.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("create_sample")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE       = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
DEST         = PROJECT_ROOT / "data" / "sample" / "train_features_sample.parquet"
TARGET_ROWS  = 250_000


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source parquet not found: {SOURCE}. "
            f"Run `engineer_banking_features.py` first."
        )

    # 1. Load the full labelled training frame. ``read_parquet`` is the
    #    only meaningful cost in this script; the rest is just slicing.
    logger.info("Loading %s", SOURCE)
    df = pd.read_parquet(SOURCE)
    n_rows = len(df)
    fraud_rate_full = float(df["is_fraud"].mean())
    logger.info("Loaded %s rows (fraud rate = %.4f%%)",
                f"{n_rows:,}", fraud_rate_full * 100.0)

    # 2. Stratified downsample. ``train_test_split`` does not have a
    #    "give me N rows" argument, but it does have ``train_size``,
    #    which we set to ``TARGET_ROWS``. The split is *deterministic*
    #    thanks to ``random_state=42`` so the same artefact is produced
    #    on every run.
    sample, _ = train_test_split(
        df,
        train_size=TARGET_ROWS,
        stratify=df["is_fraud"],
        random_state=42,
    )

    # 3. Reset the index. ``train_test_split`` preserves the original
    #    integer index, which would otherwise expose row numbers that
    #    are meaningful only in the source frame. A fresh 0..N-1
    #    RangeIndex is the honest default for a deployable artefact.
    sample = sample.reset_index(drop=True)

    # 4. Persist. Snappy is the Parquet default; on this slice it
    #    produces a file well under 30 MB, which is comfortably inside
    #    GitHub's 100 MB ceiling.
    sample.to_parquet(DEST, index=False)
    size_mb = DEST.stat().st_size / (1024 * 1024)

    # 5. Validation report. The two numbers to watch are *shape* and
    #    *fraud rate*: the shape must be exactly (250_000, 44) and the
    #    fraud rate must match the source to 4 decimal places. Anything
    #    looser than that means the stratification didn't work.
    fraud_rate_sample = float(sample["is_fraud"].mean())
    print("=" * 72)
    print("  Stratified downsample -- validation report")
    print("=" * 72)
    print(f"  Source path          : {SOURCE}")
    print(f"  Output path          : {DEST}")
    print(f"  Source shape         : {n_rows:>12,} rows  x  {df.shape[1]:>4} cols")
    print(f"  Sample shape         : {len(sample):>12,} rows  x  {sample.shape[1]:>4} cols")
    print(f"  Output file size     : {size_mb:>12.2f} MB")
    print(f"  Source fraud rate    : {fraud_rate_full*100:>11.4f}%")
    print(f"  Sample fraud rate    : {fraud_rate_sample*100:>11.4f}%")
    print(f"  Fraud rows in sample : {int(sample['is_fraud'].sum()):>12,}")
    print("=" * 72)


if __name__ == "__main__":
    main()

