"""Master ABT builder for the CaixaBank-style banking extract.

This module is the second half of the ETL:

  1. ``clean_banking_tables`` produces three Parquet files that already
     honour the primary-key uniqueness contract.
  2. ``build_master_abt`` consumes those Parquets *plus* the two JSON
     enrichment files (fraud labels, MCC dictionary) and produces a single
     denormalised Analytical Base Table.

Design priorities, in order:

* **Strictness.** Every join is a left join on the *child* side, so we
  never silently drop a transaction. The function raises loudly the
  instant the row count drifts, which is the canonical "fan-out"
  symptom of a dirty primary key.
* **Memory discipline.** The transactions frame is ~13M rows; we use
  category dtypes for high-cardinality join keys when we can, and
  ``pyarrow``-backed Parquet I/O for both read and write.
* **Operational visibility.** Every step logs to ``logging`` *and* prints
  a compact summary at the end, so a smoke-test in a notebook is just
  one call.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("build_master_abt")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------
def _load_fraud_labels(path: Path) -> pd.Series:
    """Return ``transaction_id`` -> ``is_fraud`` (0/1) as a pandas Series.

    The upstream extract is ``{"target": {"<txn_id>": "Yes"|"No", ...}}``.
    We unwrap the wrapper, normalise the value to an ``Int8`` (NaN for
    rows that have no label) and index on the *string* transaction id so
    ``.map`` is fast and unambiguous.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Defensive: accept either the wrapped or the flat shape.
    inner: dict[str, Any] = raw.get("target", raw) if isinstance(raw, dict) else {}

    fraud_s = pd.Series(inner, dtype=object)
    if fraud_s.empty:
        # No labels at all -- still produce a correctly-typed empty series.
        return pd.Series(dtype="Int8", name="is_fraud")

    # Normalise the *string* label to 0/1 with pd.NA preserved for the rest.
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0, True: 1, False: 0}
    fraud_s = fraud_s.map(mapping).astype("Int8")
    fraud_s.index = fraud_s.index.astype(str)         # str keys for safe .map
    fraud_s.name = "is_fraud"
    return fraud_s


def _load_mcc_codes(path: Path) -> pd.Series:
    """Return ``mcc`` -> ``description`` as a pandas Series."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    s = pd.Series(raw, dtype=object)
    if s.empty:
        return pd.Series(dtype="string", name="mcc_description")

    # MCC keys arrive as strings; cast both keys and the mcc column on the
    # transactions frame to str for the .map() call, so we don't lose any
    # codes that look like ``0742`` vs ``742``.
    s.index = s.index.astype(str)
    s.name = "mcc_description"
    return s


# ---------------------------------------------------------------------------
# Pre-join safety net
# ---------------------------------------------------------------------------
def _assert_unique_keys(df: pd.DataFrame, keys: list[str], *, label: str) -> None:
    """Raise a descriptive error if any key column has duplicates.

    The "row count explodes after a left join" failure mode almost always
    traces back to a non-unique right-side key. Catching it *before* the
    join turns a 30-second mystery into a one-line fix.
    """
    for k in keys:
        if k not in df.columns:
            raise KeyError(f"[{label}] expected key column '{k}' not found")
        dup = int(df[k].duplicated().sum())
        if dup:
            raise ValueError(
                f"[{label}] key column '{k}' has {dup:,} duplicate values; "
                f"left-join would fan out and inflate the master ABT. "
                f"Re-run clean_banking_tables() on {label}."
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_master_abt(data_dir: str | os.PathLike = ".") -> pd.DataFrame:
    """Build the master Analytical Base Table.

    Parameters
    ----------
    data_dir
        Directory that contains the three cleaned Parquet files and the
        two enrichment JSON files. Defaults to the current working
        directory.

    Returns
    -------
    pandas.DataFrame
        The denormalised ABT, *also* persisted as ``master_banking_data.parquet``
        in ``data_dir``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # -----------------------------------------------------------------------
    # 1. Load the three Parquets
    # -----------------------------------------------------------------------
    paths = {
        "transactions": PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.parquet",
        "cards":        PROJECT_ROOT / "data" / "processed" / "cleaned_cards.parquet",
        "users":        PROJECT_ROOT / "data" / "processed" / "cleaned_users.parquet",
        "fraud":        PROJECT_ROOT / "data" / "raw" / "train_fraud_labels.json",
        "mcc":          PROJECT_ROOT / "data" / "raw" / "mcc_codes.json",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    logger.info("Loading cleaned_transactions.parquet")
    tx = pd.read_parquet(paths["transactions"])
    logger.info("Loading cleaned_cards.parquet")
    cards = pd.read_parquet(paths["cards"])
    logger.info("Loading cleaned_users.parquet")
    users = pd.read_parquet(paths["users"])

    initial_row_count = len(tx)                       # the invariant we will defend
    logger.info("Initial transactions row count: %s", f"{initial_row_count:,}")

    # -----------------------------------------------------------------------
    # 2. Pre-join sanity: the right-hand side of every join MUST be unique
    #    on its join key, otherwise the row count explodes.
    # -----------------------------------------------------------------------
    _assert_unique_keys(cards, ["card_id"], label="cleaned_cards.parquet")
    _assert_unique_keys(users, ["user_id"], label="cleaned_users.parquet")
    # ``transaction_id`` should also be unique on the transactions frame --
    # if it isn't, even the .map() call would be ambiguous. Surface the
    # problem here rather than letting it manifest later as silent label
    # collisions.
    _assert_unique_keys(tx, ["transaction_id"], label="cleaned_transactions.parquet")

    # -----------------------------------------------------------------------
    # 3. Map JSON enrichment onto the transactions frame
    # -----------------------------------------------------------------------
    # 3a. Fraud labels: ``is_fraud`` becomes a nullable Int8 column.
    fraud = _load_fraud_labels(paths["fraud"])
    # ``.map`` aligns on the index of ``fraud``; any transaction_id that
    # is not in the JSON keeps its NaN (the task spec: "fill with NaN").
    tx["transaction_id_str"] = tx["transaction_id"].astype(str)
    tx["is_fraud"] = tx["transaction_id_str"].map(fraud).astype("Int8")
    tx = tx.drop(columns=["transaction_id_str"])

    # 3b. MCC descriptions: dict-like .map on the mcc column (cast to str
    #     so 742 == "742").
    mcc_desc = _load_mcc_codes(paths["mcc"])
    tx["mcc_description"] = tx["mcc"].astype(str).map(mcc_desc).astype("string")

    # -----------------------------------------------------------------------
    # 4. Strict left joins
    # -----------------------------------------------------------------------
    # Join 1: transactions LEFT JOIN cards ON card_id
    logger.info("Left-joining cards on card_id ...")
    abt = tx.merge(cards, on="card_id", how="left", validate="many_to_one")
    if len(abt) != initial_row_count:
        raise RuntimeError(
            f"Row count drifted after cards join: {len(abt):,} "
            f"(expected {initial_row_count:,}). This indicates a fan-out -- "
            f"check the uniqueness of card_id in cleaned_cards.parquet."
        )

    # Join 2: result LEFT JOIN users ON user_id
    logger.info("Left-joining users on user_id ...")
    abt = abt.merge(users, on="user_id", how="left", validate="many_to_one")
    if len(abt) != initial_row_count:
        raise RuntimeError(
            f"Row count drifted after users join: {len(abt):,} "
            f"(expected {initial_row_count:,}). This indicates a fan-out -- "
            f"check the uniqueness of user_id in cleaned_users.parquet."
        )

    # -----------------------------------------------------------------------
    # 5. Friendly dtypes: shrink the (now wider) frame.
    # -----------------------------------------------------------------------
    # The two string-heavy columns from the cards/users frames can be
    # converted to ``category`` to save a lot of memory; downstream code
    # can opt back into ``object`` if it really wants to.
    for col in abt.select_dtypes(include="string").columns:
        nunique = abt[col].nunique(dropna=True)
        # Category beats string only when the cardinality is materially
        # smaller than the row count; otherwise the indirection costs more
        # than it saves.
        if 0 < nunique < max(64, len(abt) // 50):
            abt[col] = abt[col].astype("category")

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    labeled   = int(abt["is_fraud"].notna().sum())
    unlabeled = int(abt["is_fraud"].isna().sum())
    fraud_pos = int((abt["is_fraud"] == 1).sum())
    fraud_neg = int((abt["is_fraud"] == 0).sum())
    mem_mb    = abt.memory_usage(deep=True).sum() / (1024 ** 2)

    print("=" * 72)
    print("  Master Analytical Base Table -- build summary")
    print("=" * 72)
    print(f"  Final shape     : {abt.shape[0]:>12,} rows  x  {abt.shape[1]:>4} cols")
    print(f"  Memory usage    : {mem_mb:>12,.1f} MB")
    print(f"  Labeled   (0/1) : {labeled:>12,}  (fraud=1: {fraud_pos:,}, fraud=0: {fraud_neg:,})")
    print(f"  Unlabeled (NaN) : {unlabeled:>12,}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 7. Persist
    # -----------------------------------------------------------------------
    out_path = PROJECT_ROOT / "data" / "processed" / "master_banking_data.parquet"
    logger.info("Writing master_banking_data.parquet -> %s", out_path)
    abt.to_parquet(out_path, index=False)

    return abt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT)
    build_master_abt(directory)






