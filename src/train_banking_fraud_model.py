"""LightGBM fraud classifier trainer for the CaixaBank-style ABT.

Pipeline
========

1. Load ``train_features.parquet`` (the labelled split produced by
   ``engineer_banking_features``).
2. Add two high-signal V2 features as ``int8``:
       - ``is_chip_present``  (chip-on-card binary flag)
       - ``has_error``        (presence of any error / declined code)
3. Drop non-predictive identifier columns that would otherwise let the
   model memorise rows (transaction_id, card_id, card_number, cvv,
   account dates, raw errors text, etc.).
4. Cast remaining object columns to ``category`` for LightGBM's native
   categorical handling.
5. Stratified 80/20 train/validation split.
6. Train an ``lgb.LGBMClassifier`` with class-imbalance handling.
7. Evaluate on the validation set with AUPRC, ROC-AUC, F1, precision,
   and recall -- accuracy is intentionally omitted because it is
   meaningless on a 0.15% positive class.
8. Save the model to ``lgbm_fraud_model.txt`` and print the Top-10
   LightGBM feature importances.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_banking_fraud_model")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Identifier / non-predictive columns
# ---------------------------------------------------------------------------
# These columns would let a tree trivially memorise the row -> label mapping
# (think ``card_number`` + ``cvv``), and the date columns leak "future" info
# that wouldn't be available at training time in a real system. We drop
# them before fitting.
_DROP_COLS = [
    "is_fraud",          # target
    "transaction_id",    # unique per row
    "date",              # raw timestamp; we keep the engineered hour/dow
    "expires",
    "acct_open_date",
    "year_pin_last_changed",
    "card_number",       # 1:1 with card_id
    "cvv",               # secret-like, would invite leakage
    "errors",            # raw text; the binary ``has_error`` is what we keep
    "address",           # 1:1-ish with user demographics
]


# ---------------------------------------------------------------------------
# V2 feature builders
# ---------------------------------------------------------------------------
def _add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject the two V2 indicators as int8 columns.

    LightGBM will gladly consume ``Int8``/``int8`` directly; the only
    subtlety is that pandas ``NaN`` in the ``errors`` column must be
    detected as missing, not as the string ``"nan"``. We do that with
    ``.isna()`` which works on both object and categorical dtypes.
    """
    # 1. Chip presence: the cards table has ``has_chip`` as ``YES``/``NO``.
    #    Treat anything that lower-cases to ``yes`` as 1, everything else
    #    (including NaN) as 0 -- the absence of a chip is itself a signal.
    if "has_chip" in df.columns:
        chip_lower = df["has_chip"].astype("string").str.strip().str.lower()
        df["is_chip_present"] = chip_lower.eq("yes").fillna(False).astype("int8")
    else:
        # Some exports use ``use_chip`` as the chip-on-transaction signal
        # (e.g. "Chip Transaction" vs "Swipe Transaction"). Fall back to
        # that so the function is robust to header drift.
        if "use_chip" in df.columns:
            ut = df["use_chip"].astype("string").str.lower()
            df["is_chip_present"] = ut.str.contains("chip", na=False).astype("int8")
        else:
            logger.warning("No chip column found; is_chip_present = 0")
            df["is_chip_present"] = np.int8(0)

    # 2. Error flag: any non-null value in ``errors`` counts as 1.
    if "errors" in df.columns:
        df["has_error"] = df["errors"].notna().astype("int8")
    else:
        logger.warning("No errors column found; has_error = 0")
        df["has_error"] = np.int8(0)

    return df


# ---------------------------------------------------------------------------
# Categorical preparation
# ---------------------------------------------------------------------------
def _prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object/string columns to ``category`` for LightGBM.

    LightGBM has first-class categorical support (``categorical_feature``
    parameter) and benefits hugely from the 1-byte-per-row storage of
    ``category`` on high-cardinality columns. We exclude the engineered
    money columns and the binary flags which are already numeric.
    """
    SKIP = set(_DROP_COLS) | {
        "is_chip_present", "has_error", "is_fraud",
        "tx_to_limit_ratio", "debt_to_income_ratio", "available_credit",
        "transaction_hour", "transaction_day_of_week",
    }
    for col in df.columns:
        if col in SKIP:
            continue
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            # ``astype("category")`` is the right primitive; it materialises
            # the dictionary once and reuses the codes.
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def train_banking_fraud_model(filepath: str | os.PathLike) -> lgb.LGBMClassifier:
    """Train the LightGBM fraud model end-to-end.

    Parameters
    ----------
    filepath
        Path to ``train_features.parquet``.

    Returns
    -------
    lightgbm.LGBMClassifier
        The fitted model. *Also* saved to ``lgbm_fraud_model.txt``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    logger.info("Loading %s", filepath)
    df = pd.read_parquet(filepath)
    logger.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])

    # -----------------------------------------------------------------------
    # 2. V2 features
    # -----------------------------------------------------------------------
    df = _add_v2_features(df)

    # -----------------------------------------------------------------------
    # 3. X / y
    # -----------------------------------------------------------------------
    y = df["is_fraud"].astype("int8")
    # ``errors="ignore"`` is a safety net in case a future export happens
    # to add one of the drop columns under a slightly different name.
    X = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")

    # LightGBM categorical contract: every categorical column must be
    # marked as such, and pandas ``category`` is the canonical way.
    X = _prepare_categoricals(X)

    # Collect the categorical column names for the LightGBM constructor.
    cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c])]
    logger.info("Feature matrix: %d cols, %d categorical", X.shape[1], len(cat_cols))

    # -----------------------------------------------------------------------
    # 4. Train / validation split
    # -----------------------------------------------------------------------
    # ``stratify=y`` is non-negotiable on a 0.15% positive class: without
    # it, a 50/50 split can produce a validation set with zero fraud
    # rows and silently break the AUPRC.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    logger.info(
        "Split sizes: train=%s (fraud=%.4f%%), val=%s (fraud=%.4f%%)",
        f"{len(X_train):,}",
        100.0 * y_train.mean(),
        f"{len(X_val):,}",
        100.0 * y_val.mean(),
    )

    # -----------------------------------------------------------------------
    # 5. Class imbalance
    # -----------------------------------------------------------------------
    # We *dynamically* compute scale_pos_weight, which is the
    # recommended pattern for LightGBM: ``is_unbalance=True`` is a
    # convenient shortcut for binary problems but is hard-coded to
    # ``neg/pos``; computing it explicitly makes the intent obvious and
    # matches the spec ("or dynamically calculate scale_pos_weight").
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.2f (pos=%d, neg=%d)",
                scale_pos_weight, n_pos, n_neg)

    # -----------------------------------------------------------------------
    # 6. LightGBM model
    # -----------------------------------------------------------------------
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=200,        # larger leaves -> more conservative
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,                    # use all cores
        objective="binary",
        metric="average_precision",   # primary metric
        verbosity=-1,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_secs = time.perf_counter() - t0
    logger.info("Training finished in %.1fs, best iter = %s",
                train_secs, model.best_iteration_)

    # -----------------------------------------------------------------------
    # 7. Evaluation -- strictly no accuracy
    # -----------------------------------------------------------------------
    # We pull *probabilities* for the positive class, not the binary
    # predictions: AUPRC and ROC-AUC need scores, and F1/precision/recall
    # use a 0.5 threshold (the spec asks for "binary labels" too, so we
    # use the same threshold for both views).
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype("int8")

    auprc  = average_precision_score(y_val, y_proba)
    rocauc = roc_auc_score(y_val, y_proba)
    f1     = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
    prec   = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
    rec    = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

    print("=" * 72)
    print("  LightGBM Fraud Model -- Evaluation Report")
    print("=" * 72)
    print(f"  Train rows           : {len(X_train):>12,}")
    print(f"  Validation rows      : {len(X_val):>12,}")
    print(f"  Train fraud rate     : {100*y_train.mean():>11.4f}%")
    print(f"  Val   fraud rate     : {100*y_val.mean():>11.4f}%")
    print("-" * 72)
    print(f"  AUPRC   (primary)    : {auprc:>11.4f}")
    print(f"  ROC-AUC              : {rocauc:>11.4f}")
    print(f"  F1  (fraud class)    : {f1:>11.4f}")
    print(f"  Precision  (class 1) : {prec:>11.4f}")
    print(f"  Recall     (class 1) : {rec:>11.4f}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 8. Top-10 feature importances (gain-based)
    # -----------------------------------------------------------------------
    # ``booster_.feature_importance(importance_type="gain")`` is the
    # standard for LightGBM: it sums the loss reduction brought by each
    # feature across all trees, which correlates with what the model
    # actually *uses* to split.
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
        name="gain",
    ).sort_values(ascending=False)

    print("\n  Top 10 Most Important Features (by gain):")
    print("-" * 72)
    for i, (name, gain) in enumerate(importance.head(10).items(), start=1):
        print(f"   {i:>2}. {name:<30s} {gain:>14,.0f}")
    print("-" * 72)

    # -----------------------------------------------------------------------
    # 9. Persist the model
    # -----------------------------------------------------------------------
    out_path = PROJECT_ROOT / "models" / "lgbm_fraud_model.txt"
    logger.info("Saving model to %s", out_path)
    model.booster_.save_model(str(out_path))

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "processed" / "train_features.parquet")
    train_banking_fraud_model(target)



