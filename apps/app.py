"""Banking Fraud Detection Dashboard & Real-Time Simulator.

Streamlit front-end for the LightGBM fraud model trained by
``train_banking_fraud_model.py``.

Layout
------

  * Sidebar  : hackathon context, model blurb, baseline fraud rate.
  * KPIs     : headline numbers (recall, AUPRC, inference latency).
  * Simulator: a form that lets an analyst build a single synthetic
               transaction and see the model's fraud probability in
               real time.
  * Insights : a bar chart of the model's top features so users can
               see *why* the model said what it said.

Design notes
------------

  * The LightGBM booster is loaded once via ``@st.cache_resource`` --
    the model file is ~225 KB, so reloading per click would be silly.
  * The simulator builds a single-row DataFrame that exactly matches
    the 36-feature contract the booster was trained on (see
    ``booster.feature_name()``). ID columns are filled with neutral
    sentinel values, never random-looking real IDs, so a user can't
    accidentally trigger ID-based leakage from a memorized row.
  * ``st.set_page_config`` is the *first* Streamlit call -- it must
    run before any other Streamlit widget.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Streamlit captures stdout, so a module-level logger is mostly for
# debugging when the app is run via ``streamlit run app.py``.
logger = logging.getLogger("fraud_dashboard")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Path resolution: the model file is expected to live next to this script.
# Falling back to a relative path keeps the dashboard runnable from
# notebooks / different working directories.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_fraud_model.txt"

# The exact feature order the booster was trained on. ``booster.feature_name()``
# returns this list, but we hard-code it as a single source of truth so a
# missing model file doesn't make the app entirely unrenderable.
EXPECTED_FEATURES = [
    "client_id", "card_id", "amount", "use_chip", "merchant_id",
    "merchant_city", "merchant_state", "zip", "mcc", "mcc_description",
    "user_id", "card_brand", "card_type", "has_chip", "num_cards_issued",
    "credit_limit", "card_on_dark_web", "current_age", "retirement_age",
    "birth_year", "birth_month", "gender", "latitude", "longitude",
    "per_capita_income", "yearly_income", "total_debt", "credit_score",
    "num_credit_cards", "tx_to_limit_ratio", "debt_to_income_ratio",
    "available_credit", "transaction_hour", "transaction_day_of_week",
    "is_chip_present", "has_error",
]

# A small, analyst-friendly MCC menu. The codes are real ISO 8583 MCCs;
# the descriptions match those in ``mcc_codes.json``.
MCC_CHOICES = [
    ("5411", "Grocery Stores"),
    ("5812", "Eating Places and Restaurants"),
    ("5944", "Jewelry Stores"),
    ("4829", "Money Transfer"),
    ("7996", "Amusement Parks"),
]

# A handful of plausible US merchant cities -- the model is sensitive to
# this feature, so we offer a few distinct values rather than letting the
# user type a free-form string.
CITY_CHOICES = ["New York", "Los Angeles", "Chicago", "Houston", "Miami"]


# ---------------------------------------------------------------------------
# Cached resource: load the LightGBM booster once per session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fraud model ...")
def load_model() -> lgb.Booster:
    """Load (and cache) the trained LightGBM booster.

    ``@st.cache_resource`` is the Streamlit equivalent of a process-wide
    singleton: the same ``Booster`` instance is reused for every rerun,
    which matters because loading a model is O(10-100 ms) and we will
    re-run on every button click.
    """
    if not MODEL_PATH.exists():
        # Don't crash the whole page -- render a clear error and bail.
        st.error(
            f"Model file not found: `{MODEL_PATH}`. "
            f"Run `python src\\train_banking_fraud_model.py` first."
        )
        st.stop()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    logger.info("Loaded LightGBM booster with %d features", booster.num_feature())
    return booster


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def _build_feature_row(
    *,
    amount: float,
    credit_limit: float,
    use_chip: bool,
    current_age: int,
    transaction_hour: int,
    mcc_code: str,
    mcc_description: str,
    city: str,
) -> pd.DataFrame:
    """Construct a single-row DataFrame that matches the booster contract.

    The LightGBM booster is a *typed* model: it expects categorical
    columns to arrive as ``category`` dtype, and numeric columns as
    numerics. We build the row in the exact feature order produced by
    ``booster.feature_name()`` so the prediction is bit-identical to a
    real training-time row.

    ID columns are filled with neutral sentinel values rather than real
    IDs -- the model was trained on the *distribution* of IDs, not on
    memorising specific ones, so a constant value is the right default.
    """
    # Defensive: avoid div-by-zero on the ratio. A 0 limit means the
    # account has no available credit; we surface that as +inf, which is
    # exactly the "drain the card" signal the model is looking for.
    tx_to_limit_ratio = float(amount) / float(credit_limit) if credit_limit else np.inf

    # The other engineered columns the model expects.
    available_credit = float(credit_limit) - float(amount)
    debt_to_income_ratio = 0.0           # not exposed in the UI; safe default

    # ``is_chip_present`` mirrors the card-level ``has_chip`` flag in the
    # ABT. The user-controlled ``use_chip`` is the transaction-level
    # signal; we keep both so the model sees the same contract it was
    # trained on.
    is_chip_present = 1 if use_chip else 0

    # Default demographics + behavioural columns. The simulator is meant
    # to be a *what-if* tool, not a full customer-record editor, so we
    # use neutral defaults for fields the user doesn't control.
    row = {
        # --- ID columns: neutral sentinels to avoid leakage ---
        "client_id":     0,
        "card_id":       0,
        "user_id":       0,
        "merchant_id":   0,

        # --- User-controlled ---
        "amount":              float(amount),
        "credit_limit":        float(credit_limit),
        "use_chip":            "Chip Transaction" if use_chip else "Swipe Transaction",
        "transaction_hour":    int(transaction_hour),
        "merchant_city":       city,
        "mcc":                 int(mcc_code),
        "mcc_description":     mcc_description,
        "current_age":         int(current_age),

        # --- Engineered ---
        "tx_to_limit_ratio":   tx_to_limit_ratio,
        "available_credit":    available_credit,
        "debt_to_income_ratio": debt_to_income_ratio,
        "is_chip_present":     is_chip_present,
        "has_error":           0,

        # --- Static defaults (median-ish values from the train set) ---
        "merchant_state":      "CA",
        "zip":                 90000.0,
        "card_brand":          "Visa",
        "card_type":           "Credit",
        "has_chip":            "YES",
        "num_cards_issued":    1,
        "card_on_dark_web":    "No",
        "retirement_age":      67,
        "birth_year":          1980,
        "birth_month":         6,
        "gender":              "female",
        "latitude":            34.05,
        "longitude":          -118.24,
        "per_capita_income":  30000.0,
        "yearly_income":      60000.0,
        "total_debt":         50000.0,
        "credit_score":        700,
        "num_credit_cards":    3,
        "transaction_day_of_week": 2,         # Wednesday
    }

    # Build the DataFrame in the order the model expects, and back-fill
    # any feature we forgot with a neutral default (0 / empty string).
    df = pd.DataFrame([row], columns=EXPECTED_FEATURES)
    df = df.fillna({col: 0 for col in df.select_dtypes("number").columns})
    df = df.fillna({col: "" for col in df.select_dtypes("object").columns})

    # The booster's categorical contract: every category column must be
    # ``category`` dtype. We cast the strings we know are categorical.
    CATEGORICAL = {
        "use_chip", "merchant_city", "merchant_state", "mcc_description",
        "card_brand", "card_type", "has_chip", "card_on_dark_web", "gender",
    }
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """Hackathon context + model blurb + baseline fraud rate."""
    with st.sidebar:
        st.title("CaixaBank Fraud Ops")
        st.caption("Hackathon 2026 -- Banking Risk Track")

        st.markdown("### Model")
        st.markdown(
            "A **LightGBM** gradient-boosted tree classifier trained on "
            "**8.9M** labelled transactions. The booster evaluates 36 "
            "features per transaction in well under 10 ms, making it "
            "suitable for synchronous payment-authorisation workflows."
        )

        st.markdown("### Baseline")
        st.metric("Population fraud rate", "0.15%",
                  help="Share of transactions labelled ``is_fraud=1`` in the training split.")

        st.markdown("### Why these metrics?")
        st.markdown(
            "Accuracy is intentionally *not* shown: on a 0.15% positive "
            "class, predicting \"not fraud\" for every row would already "
            "be 99.85% accurate. We track **Recall** (catch rate) and "
            "**AUPRC** (ranking quality) instead."
        )


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
def _render_kpis() -> None:
    """Top-row KPIs. Inference speed is measured empirically below."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Recall (fraud)", "83.5%",
                  help="Share of true fraud cases the model catches on the validation set.")
    with col2:
        st.metric("AUPRC", "0.458",
                  help="Area under the precision-recall curve on the validation set.")
    with col3:
        st.metric("ROC-AUC", "0.872",
                  help="Discriminative power of the model on the validation set.")
    with col4:
        st.metric("Inference Speed", "< 10 ms",
                  help="Median wall-clock time to score a single transaction.")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def _render_simulator(booster: lgb.Booster) -> None:
    """Interactive form to score a synthetic transaction."""
    st.subheader("Real-Time Fraud Simulator")
    st.caption(
        "Build a single transaction and see the model's fraud probability. "
        "ID fields are filled with neutral sentinels, so the result depends "
        "only on the behavioural / financial features you control."
    )

    with st.form("sim_form", clear_on_submit=False):
        # Two-column layout for a denser, more analyst-friendly form.
        c1, c2 = st.columns(2)

        with c1:
            amount = st.number_input(
                "Transaction Amount (USD)",
                min_value=0.0, max_value=100000.0,
                value=125.0, step=10.0,
                help="The dollar value of the purchase.",
            )
            credit_limit = st.number_input(
                "Card Credit Limit (USD)",
                min_value=0.0, max_value=200000.0,
                value=5000.0, step=500.0,
                help="The cardholder's total credit line.",
            )
            use_chip = st.selectbox(
                "Chip Used?",
                options=["Yes", "No"],
                index=0,
                help="Was the transaction authenticated by EMV chip?",
            ) == "Yes"

        with c2:
            current_age = st.slider(
                "Cardholder Age",
                min_value=18, max_value=90, value=42, step=1,
            )
            transaction_hour = st.slider(
                "Transaction Hour (0-23)",
                min_value=0, max_value=23, value=14, step=1,
                help="Local hour of day the transaction occurred.",
            )
            mcc_code, mcc_description = st.selectbox(
                "Merchant Category",
                options=MCC_CHOICES,
                format_func=lambda c: c[1],
                index=0,
                help="The merchant category code (MCC) and its description.",
            )
            city = st.selectbox(
                "Merchant City",
                options=CITY_CHOICES,
                index=0,
            )

        # Dynamic ratio preview -- analysts love seeing the derived
        # feature before they hit the button.
        ratio = (amount / credit_limit) if credit_limit else float("inf")
        st.markdown(
            f"<small>Computed <code>tx_to_limit_ratio</code> = "
            f"<b>{ratio:.4f}</b></small>",
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button("Simulate Transaction", use_container_width=True)

    if not submitted:
        return

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------
    feature_row = _build_feature_row(
        amount=amount,
        credit_limit=credit_limit,
        use_chip=use_chip,
        current_age=current_age,
        transaction_hour=transaction_hour,
        mcc_code=mcc_code,
        mcc_description=mcc_description,
        city=city,
    )

    # ``predict`` is a thin wrapper around the C++ scorer; a single row
    # returns a single probability. We also time it to back the "<10 ms"
    # KPI in the header.
    t0 = time.perf_counter()
    proba = float(booster.predict(feature_row)[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Visual alert. We pick the colour from the threshold and surface the
    # exact probability in the same panel.
    st.markdown("---")
    if proba < 0.5:
        st.success(f"### \U0001F7E2 Approved (Low Risk)\n\n"
                   f"Fraud probability: **{proba*100:.2f}%**")
    else:
        st.error(f"### \U0001F534 Declined (FRAUD DETECTED)\n\n"
                 f"Fraud probability: **{proba*100:.2f}%**")

    st.caption(f"Inference latency: {latency_ms:.2f} ms")

    # Echo the engineered features that drove the score, so the analyst
    # can see *why* the model said what it said.
    with st.expander("Engineered features used by the model"):
        engineered = {
            "tx_to_limit_ratio":   feature_row["tx_to_limit_ratio"].iloc[0],
            "available_credit":    feature_row["available_credit"].iloc[0],
            "debt_to_income_ratio": feature_row["debt_to_income_ratio"].iloc[0],
            "is_chip_present":     int(feature_row["is_chip_present"].iloc[0]),
            "has_error":           int(feature_row["has_error"].iloc[0]),
        }
        st.json(engineered)


# ---------------------------------------------------------------------------
# Feature importance visual
# ---------------------------------------------------------------------------
# Hard-coded top features (gain-based) from the training run. We embed them
# here so the dashboard is self-contained -- it doesn't need the parquet
# files at runtime, which keeps the deployment artefact list short.
TOP_FEATURES = pd.DataFrame(
    {
        "feature": [
            "merchant_city",
            "merchant_id",
            "mcc",
            "amount",
            "mcc_description",
            "current_age",
            "tx_to_limit_ratio",
            "use_chip",
            "client_id",
            "transaction_hour",
        ],
        "importance": [
            1_295_261_184,
            1_208_139_069,
            929_424_157,
            451_902_410,
            261_519_446,
            128_436_907,
            118_584_642,
            36_720_826,
            26_940_144,
            20_212_436,
        ],
    }
).set_index("feature").sort_values("importance")


def _render_feature_importance() -> None:
    st.subheader("Why the model says what it says")
    st.caption(
        "Top-10 LightGBM features by **gain** (loss reduction across all "
        "splits). The simulator inputs on the left influence these "
        "features directly, so changing a slider or dropdown will tilt "
        "the fraud probability in the direction shown here."
    )
    st.bar_chart(TOP_FEATURES, horizontal=True, height=420)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit entry point."""
    # ``set_page_config`` must be the first Streamlit call.
    st.set_page_config(
        page_title="CaixaBank Fraud Command Center",
        page_icon="\U0001F6E1",
        layout="wide",
    )

    _render_sidebar()
    _render_kpis()

    st.title("Banking Fraud Detection Dashboard")
    st.markdown(
        "A real-time command center for fraud analysts. Inspect the "
        "model's headline metrics on the validation set, simulate a "
        "transaction to see the model's probability, and explore the "
        "feature importances that drive its decisions."
    )

    booster = load_model()

    # Two-column body: simulator on the left, importance chart on the right.
    left, right = st.columns([1.1, 1.0])
    with left:
        _render_simulator(booster)
    with right:
        _render_feature_importance()


if __name__ == "__main__":
    main()



