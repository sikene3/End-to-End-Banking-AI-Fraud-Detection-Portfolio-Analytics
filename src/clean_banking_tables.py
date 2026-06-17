"""Production-grade cleaner for the CaixaBank-style banking extracts.

The script ships a single public entry point, ``clean_banking_tables(data_dir)``,
that ingests the three raw CSV exports -- transactions, cards, and users --
applies defensive data-quality fixes in-memory, prints a per-file Data Quality
Report, and persists the cleaned DataFrames as Parquet files ready for an
analytics / ML downstream.

Why Parquet? Columnar binary, schema-preserving, ~5-10x smaller on financial
data, and the de-facto input format for Spark / DuckDB / BigQuery.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# Resolve the project root regardless of where the script is invoked from
# so the cleaned outputs and the raw inputs always live in the canonical
# ``data/raw`` and ``data/processed`` folders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# A single, module-level logger keeps the script library-friendly (importable
# without side effects) and easy to silence / reconfigure from the outside.
logger = logging.getLogger("clean_banking_tables")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanedArtifacts:
    """Bundle of cleaned frames returned by :func:`clean_banking_tables`."""
    transactions: pd.DataFrame
    cards: pd.DataFrame
    users: pd.DataFrame


# ---------------------------------------------------------------------------
# Helper: monetary cleaning
# ---------------------------------------------------------------------------
# Some upstream sources prepend a currency symbol ($, €, £) and/or thousand
# separators (commas) before the numeric value. ``to_numeric`` would raise on
# those. Stripping them up-front gives us a single, well-defined numeric path.
_MONEY_RE = re.compile(r"[^0-9eE+\-\.]")             # keep digits, sign, dot, exp


def _to_money(series: pd.Series, *, col: str) -> pd.Series:
    """Vectorised, type-stable money coercion.

    * Strips currency symbols and thousand separators.
    * Coerces to ``float32`` (the task contract).
    * Leaves NaNs in place; the caller is expected to decide whether to drop
      them (transactions: yes, cards: yes for the limit, etc.).
    """
    if series.dtype.kind in ("f", "i"):
        # Already numeric -- only enforce the requested dtype.
        return series.astype("float32")

    # ``str`` is the common case in raw CSV exports.
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(_MONEY_RE, "", regex=True)        # strip $, €, , etc.
              .replace({"": np.nan, "-": np.nan})            # treat lone '-' as NA
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float32")


# ---------------------------------------------------------------------------
# Helper: Data Quality Report
# ---------------------------------------------------------------------------
def _dq_report(label: str, df: pd.DataFrame, *,
               initial_rows: int,
               missing_dropped: int,
               duplicate_dropped: int) -> None:
    """Pretty-print a compact DQ report. Stdout == "log-friendly"."""
    print("=" * 72)
    print(f"  Data Quality Report :: {label}")
    print("=" * 72)
    print(f"  Initial shape   : {initial_rows:>12,} rows  x  {df.shape[1]:>4} cols")
    print(f"  Final shape     : {df.shape[0]:>12,} rows  x  {df.shape[1]:>4} cols")
    print(f"  Missing dropped : {missing_dropped:>12,}")
    print(f"  Duplicate drop. : {duplicate_dropped:>12,}")
    print(f"  Remaining NaN   : {int(df.isna().sum().sum()):>12,} (informational)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Helper: defensive column resolution
# ---------------------------------------------------------------------------
# The task references the join key as ``card_id`` in transactions and cards,
# but the upstream extract actually uses ``id`` in cards / users and stores
# the FK under ``client_id`` for transactions. We resolve canonical names
# *first* so the cleaning rules read like prose.
def _ensure_columns(df: pd.DataFrame, mapping: dict[str, Iterable[str]]) -> pd.DataFrame:
    """Rename the first matching candidate per canonical key.

    Example: ``_ensure_columns(df, {"card_id": ["card_id", "id"]})`` ensures
    ``df["card_id"]`` exists, taking the value from whichever of ``"card_id"``
    or ``"id"`` is present.
    """
    rename = {}
    for canonical, candidates in mapping.items():
        for cand in candidates:
            if cand in df.columns and cand != canonical:
                rename[cand] = canonical
                break
    return df.rename(columns=rename) if rename else df


# ---------------------------------------------------------------------------
# Cleaners
# ---------------------------------------------------------------------------
def _clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply transaction-specific DQ rules."""
    initial_rows = len(df)
    # 1. Canonical column names: upstream may call the timestamp column "date"
    #    or "timestamp"; the join key may be "id" instead of "transaction_id".
    df = _ensure_columns(df, {
        "transaction_id": ["transaction_id", "id", "txn_id"],
        "card_id":        ["card_id"],
        "client_id":      ["client_id", "user_id"],
    })

    # 2. Resolve the timestamp column dynamically so the function is robust
    #    to header drift between exports.
    ts_col = next(
        (c for c in ("date", "timestamp", "transaction_date", "created_at") if c in df.columns),
        None,
    )
    if ts_col is None:
        raise KeyError("No timestamp column found in transactions_data.csv")

    # 3. Primary key hygiene -- rows with no transaction_id or no card_id are
    #    useless for a relational join and must be discarded up-front.
    missing_pk_mask = df[["transaction_id", "card_id"]].isna().any(axis=1)
    missing_dropped = int(missing_pk_mask.sum())
    df = df.loc[~missing_pk_mask].copy()

    # 4. Type coercion. ``errors="coerce"`` is intentional: it forces bad
    #    timestamps to NaT, which is far easier to diagnose than a parser
    #    crash. We then drop the unparseable rows because they cannot be
    #    placed in time.
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    bad_ts = df[ts_col].isna()
    if bad_ts.any():
        logger.warning("Dropping %d transactions with unparseable timestamps", int(bad_ts.sum()))
        df = df.loc[~bad_ts]
    df[ts_col] = df[ts_col].astype("datetime64[ns]")   # canonical, tz-naive

    # 5. Amount hygiene: strip currency symbols, cast to float32.
    df["amount"] = _to_money(df["amount"], col="amount")

    # 6. Exact-row deduplication. A *subset* dedup (e.g. on transaction_id) is
    #    intentionally avoided: a true duplicate is *all* columns equal, which
    #    is a safe signal of an upstream replay / load bug.
    pre = len(df)
    df = df.drop_duplicates()
    duplicate_dropped = pre - len(df)

    # 7. Downcast integer columns to save memory. We only touch columns that
    #    are *guaranteed* to be NaN-free here -- `zip` and `errors` are
    #    genuinely missing on a real share of rows, so leaving them nullable
    #    is the honest thing to do.
    for col in ("transaction_id", "card_id", "client_id", "merchant_id", "mcc"):
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].astype("Int64").astype("int64")

    return df, {
        "label":            "transactions_data.csv",
        "initial_rows":     initial_rows,
        "missing_dropped":  missing_dropped,
        "duplicate_dropped": duplicate_dropped,
    }


def _clean_cards(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply card-specific DQ rules.

    Note: the upstream extract uses ``id`` for ``card_id`` and ``client_id`` for
    ``user_id``. The cleaner canonicalises the names so the merge step is
    explicit and self-documenting.
    """
    initial_rows = len(df)
    df = _ensure_columns(df, {
        "card_id":   ["card_id", "id"],
        "user_id":   ["user_id", "client_id"],
    })

    # 1. Primary key hygiene.
    missing_pk_mask = df[["card_id", "user_id"]].isna().any(axis=1)
    missing_dropped = int(missing_pk_mask.sum())
    df = df.loc[~missing_pk_mask].copy()

    # 2. Deduplicate on card_id, keeping the *first* occurrence. Without this
    #    step a merge on card_id would explode the row count (cartesian
    #    product of transactions x duplicate-card rows).
    pre = len(df)
    df = df.drop_duplicates(subset=["card_id"], keep="first")
    duplicate_dropped = pre - len(df)

    # 3. Credit limit: numeric coercion (defensive -- some exports store it
    #    as a string with a leading '$').
    if "credit_limit" in df.columns:
        df["credit_limit"] = _to_money(df["credit_limit"], col="credit_limit")
    elif "card_limit" in df.columns:
        df["card_limit"] = _to_money(df["card_limit"], col="card_limit")

    # 4. Activation / open-date parsing. ``errors="coerce"`` converts the
    #    occasional unparseable value to NaT rather than aborting the run.
    #    ``acct_open_date`` and ``expires`` arrive as ``MM/YYYY`` strings;
    #    ``year_pin_last_changed`` is a 4-digit year integer.
    for col in ("acct_open_date", "expires"):
        if col in df.columns and df[col].dtype != "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], format="%m/%Y", errors="coerce")
    for col in ("year_pin_last_changed", "activation_date"):
        if col in df.columns and df[col].dtype != "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 5. Downcast the integer keys.
    for col in ("card_id", "user_id"):
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].astype("Int64").astype("int64")

    return df, {
        "label":            "cards_data.csv",
        "initial_rows":     initial_rows,
        "missing_dropped":  missing_dropped,
        "duplicate_dropped": duplicate_dropped,
    }


def _clean_users(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply user-specific DQ rules."""
    initial_rows = len(df)
    df = _ensure_columns(df, {
        "user_id": ["user_id", "id", "client_id"],
    })

    # 1. Primary key hygiene.
    missing_pk_mask = df["user_id"].isna()
    missing_dropped = int(missing_pk_mask.sum())
    df = df.loc[~missing_pk_mask].copy()

    # 2. Strict dedup on user_id.
    pre = len(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    duplicate_dropped = pre - len(df)

    # 3. Categorical clean-up. Lowercasing and stripping prevents
    #    "MALE"/"male"/" Male" from becoming three different one-hot columns
    #    in a downstream ML pipeline.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()

    # 4. Downcast the integer key.
    if df["user_id"].dtype == "float64":
        df["user_id"] = df["user_id"].astype("Int64").astype("int64")

    return df, {
        "label":            "users_data.csv",
        "initial_rows":     initial_rows,
        "missing_dropped":  missing_dropped,
        "duplicate_dropped": duplicate_dropped,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def clean_banking_tables(data_dir: str | os.PathLike) -> CleanedArtifacts:
    """Clean the three raw extracts and persist them as Parquet.

    Parameters
    ----------
    data_dir
        Directory that contains ``transactions_data.csv``, ``cards_data.csv``,
        and ``users_data.csv``.

    Returns
    -------
    CleanedArtifacts
        In-memory DataFrames ready for the merge step.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = {
        "transactions": PROJECT_ROOT / "data" / "raw" / "transactions_data.csv",
        "cards":        PROJECT_ROOT / "data" / "raw" / "cards_data.csv",
        "users":        PROJECT_ROOT / "data" / "raw" / "users_data.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------
    # ``low_memory=False`` is deliberate: the transactions file is wide and
    # we want pandas to read the whole slice before guessing dtypes, which
    # avoids the "mixed types in column X" warning that often masks a real
    # dirty-data problem.
    logger.info("Loading %s", paths["transactions"].name)
    tx = pd.read_csv(paths["transactions"], low_memory=False)
    logger.info("Loading %s", paths["cards"].name)
    cards = pd.read_csv(paths["cards"], low_memory=False)
    logger.info("Loading %s", paths["users"].name)
    users = pd.read_csv(paths["users"], low_memory=False)

    # ---------------------------------------------------------------
    # Clean
    # ---------------------------------------------------------------
    tx,     tx_meta     = _clean_transactions(tx)
    cards,  cards_meta  = _clean_cards(cards)
    users,  users_meta  = _clean_users(users)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    for df, meta in ((tx, tx_meta), (cards, cards_meta), (users, users_meta)):
        _dq_report(
            meta["label"], df,
            initial_rows=meta["initial_rows"],
            missing_dropped=meta["missing_dropped"],
            duplicate_dropped=meta["duplicate_dropped"],
        )

    # ---------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------
    # Parquet preserves the (downcasted) dtypes exactly, unlike CSV, which is
    # the whole point of doing the cleaning *before* the merge.
    out_paths = {
        "transactions": PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.parquet",
        "cards":        PROJECT_ROOT / "data" / "processed" / "cleaned_cards.parquet",
        "users":        PROJECT_ROOT / "data" / "processed" / "cleaned_users.parquet",
    }
    for name, df in (("transactions", tx), ("cards", cards), ("users", users)):
        target = out_paths[name]
        logger.info("Writing %s -> %s", name, target)
        df.to_parquet(target, index=False)

    return CleanedArtifacts(transactions=tx, cards=cards, users=users)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT)
    clean_banking_tables(directory)







