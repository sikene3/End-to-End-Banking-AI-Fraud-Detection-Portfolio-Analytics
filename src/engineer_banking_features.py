"""Feature engineering + train/inference split for the CaixaBank-style ABT.

Pipeline:

  1. Load ``master_banking_data.parquet``.
  2. Coerce the three "dirty" money columns (``per_capita_income``,
     ``yearly_income``, ``total_debt``) from ``$xxx,xxx`` strings into
     ``float32`` -- in a single vectorised pass.
  3. Build the ML-grade features called out in the brief:
        - ``tx_to_limit_ratio``   = amount / credit_limit
        - ``debt_to_income_ratio``= total_debt / yearly_income
        - ``available_credit``    = credit_limit - amount
        - ``transaction_hour``    = 0..23 from the timestamp
        - ``transaction_day_of_week`` = 0..6 from the timestamp
  4. Split the frame into a labelled training set and an unlabelled
     inference set, persist both as Parquet, and print a short report.

The script is importable (``from engineer_banking_features import
engineer_banking_features``) and runnable as a CLI:

    python engineer_banking_features.py <path-to-master.parquet>
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("engineer_banking_features")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Money cleaner
# ---------------------------------------------------------------------------
# Same idea as the helper in ``clean_banking_tables`` -- a single regex that
# keeps only digits, sign, dot, and exponent characters, then a vectorised
# ``to_numeric`` cast. We re-implement it here (rather than importing from
# the cleaner) so this module has no hidden dependencies.
_MONEY_RE = re.compile(r"[^0-9eE+\-\.]")


def _clean_money_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Strip ``$``/``€``/commas and cast the listed columns to ``float32``."""
    for col in cols:
        if col not in df.columns:
            logger.warning("Column %s not found; skipping money clean", col)
            continue
        s = df[col]
        if s.dtype.kind in ("f", "i"):
            # Already numeric, just enforce the requested dtype.
            df[col] = s.astype("float32")
            continue

        # ``astype(str)`` on a pandas ``category`` returns the *display* form
        # (the original string), which is what we want here.
        cleaned = (
            s.astype(str)
             .str.strip()
             .str.replace(_MONEY_RE, "", regex=True)
             .replace({"": np.nan, "-": np.nan})
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce").astype("float32")
        logger.info("Cleaned money column %s -> float32", col)
    return df


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _add_ratio(df: pd.DataFrame, num: str, den: str, out: str) -> None:
    """Add ``out = num / den`` with a safe denominator.

    Using ``np.divide(..., out=..., where=...)`` is materially faster than
    ``df[num] / df[den]`` for a 13M-row frame because it skips the
    intermediate ``Series`` allocation and avoids the implicit float64
    upcast.
    """
    if num not in df.columns or den not in df.columns:
        logger.warning("Skipping %s: %s or %s missing", out, num, den)
        return
    num_vals = df[num].to_numpy(dtype="float64", copy=False)
    den_vals = df[den].to_numpy(dtype="float64", copy=False)
    safe_den = np.where(den_vals == 0, np.nan, den_vals)
    out_vals = num_vals / safe_den
    df[out] = out_vals.astype("float32")


def _add_difference(df: pd.DataFrame, a: str, b: str, out: str) -> None:
    """Add ``out = a - b`` as ``float32``."""
    if a not in df.columns or b not in df.columns:
        logger.warning("Skipping %s: %s or %s missing", out, a, b)
        return
    df[out] = (df[a].to_numpy("float64") - df[b].to_numpy("float64")).astype("float32")


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> str | None:
    """Coerce ``col`` to datetime, return the resolved column name."""
    if col in df.columns:
        if df[col].dtype.kind != "M":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return col
    # The ABT uses ``date``; if a downstream frame ever renames it, fall
    # back to a small set of sensible names.
    for cand in ("transaction_date", "timestamp", "created_at"):
        if cand in df.columns:
            if df[cand].dtype.kind != "M":
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
            return cand
    return None


def _add_temporal(df: pd.DataFrame, ts_col: str) -> None:
    """Add ``transaction_hour`` and ``transaction_day_of_week`` as int8."""
    if ts_col is None:
        logger.warning("No timestamp column found; skipping temporal features")
        return
    # ``.dt`` accessors are already vectorised and C-backed.
    df["transaction_hour"] = df[ts_col].dt.hour.astype("int8")
    df["transaction_day_of_week"] = df[ts_col].dt.dayofweek.astype("int8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def engineer_banking_features(filepath: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer features on the master ABT and split into train/inference.

    Parameters
    ----------
    filepath
        Path to ``master_banking_data.parquet``.

    Returns
    -------
    (train_data, inference_data)
        The two split DataFrames, *also* persisted to disk.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    logger.info("Loading %s", filepath)
    df = pd.read_parquet(filepath)
    initial_rows = len(df)
    logger.info("Loaded %s rows x %s cols", f"{initial_rows:,}", df.shape[1])

    # -----------------------------------------------------------------------
    # 2. Clean the dirty money columns
    # -----------------------------------------------------------------------
    df = _clean_money_columns(
        df,
        cols=["per_capita_income", "yearly_income", "total_debt"],
    )

    # -----------------------------------------------------------------------
    # 3. Engineer features
    # -----------------------------------------------------------------------
    _add_ratio(df, num="amount",          den="credit_limit", out="tx_to_limit_ratio")
    _add_ratio(df, num="total_debt",      den="yearly_income", out="debt_to_income_ratio")
    _add_difference(df, a="credit_limit", b="amount",          out="available_credit")

    # Temporal features. We re-resolve the timestamp column because the
    # ABT sometimes renames ``date`` to ``transaction_date`` after a
    # downstream operation; the helper above is defensive.
    ts_col = _ensure_datetime(df, col="date")
    _add_temporal(df, ts_col)

    # -----------------------------------------------------------------------
    # 4. Split
    # -----------------------------------------------------------------------
    # A boolean mask is the cheapest and clearest way to split, and it
    # preserves the original row order (no need to sort/index trickery).
    fraud_col = "is_fraud" if "is_fraud" in df.columns else None
    if fraud_col is None:
        raise KeyError("is_fraud column not present in master_banking_data.parquet")

    labeled_mask   = df[fraud_col].notna().to_numpy()
    unlabeled_mask = ~labeled_mask

    # Cast is_fraud to plain int8 for the training set; NaN cannot survive
    # the cast, which is exactly the contract we want here (the mask has
    # already filtered them out).
    train_data = df.loc[labeled_mask].copy()
    train_data[fraud_col] = train_data[fraud_col].astype("int8")

    # Inference set must not leak the label.
    inference_data = df.loc[unlabeled_mask].copy()
    inference_data = inference_data.drop(columns=[fraud_col])

    # -----------------------------------------------------------------------
    # 5. Report
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("  Feature engineering + train/inference split")
    print("=" * 72)
    print(f"  Loaded shape          : {initial_rows:>12,} rows  x  {df.shape[1]:>4} cols")
    print(f"  train_data shape      : {train_data.shape[0]:>12,} rows  x  {train_data.shape[1]:>4} cols")
    print(f"  inference_data shape  : {inference_data.shape[0]:>12,} rows  x  {inference_data.shape[1]:>4} cols")
    print(f"  train fraud=1 / 0     : "
          f"{int((train_data[fraud_col] == 1).sum()):>10,} / "
          f"{int((train_data[fraud_col] == 0).sum()):>10,}")
    print("-" * 72)
    print("  Engineered feature dtypes (.info() summary):")
    print("-" * 72)
    # Build a tiny frame with just the new columns for a focused info()
    new_cols = [
        "tx_to_limit_ratio", "debt_to_income_ratio", "available_credit",
        "transaction_hour", "transaction_day_of_week",
    ]
    existing = [c for c in new_cols if c in df.columns]
    df[existing].info(memory_usage="deep")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 6. Persist
    # -----------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "data" / "processed"
    train_path = out_dir / "train_features.parquet"
    infer_path = out_dir / "inference_features.parquet"

    logger.info("Writing %s", train_path)
    train_data.to_parquet(train_path, index=False)
    logger.info("Writing %s", infer_path)
    inference_data.to_parquet(infer_path, index=False)

    return train_data, inference_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "processed" / "master_banking_data.parquet")
    engineer_banking_features(target)



