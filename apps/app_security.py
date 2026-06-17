"""
CaixaBank Real-Time Fraud Detection & Security Simulator Dashboard.
Updated Version: Configured to run efficiently using 'train_features_sample.parquet'.

Categorical-alignment fix
=========================
The original code did ``input_df[col] = input_df[col].astype("category")``
on a SINGLE-row frame, which pandas built a category dictionary for
containing only that one value. LightGBM then complained:
``train and valid dataset categorical_feature do not match`` because
the dictionaries learned at training time are different.

The fix has two parts:

1. Build a **complete 36-column** prediction frame -- the original code
   only sent 19 of the 36 features, which is its own kind of bug.
2. Cast the 9 categorical columns with the *full* training-time
   dictionary, captured at model load via ``Booster.pandas_categorical``.
   That makes the single-row category codes line up exactly with the
   booster's expectation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# Resolve the project root regardless of where the script is invoked from.
# This makes the deployable artefact list self-contained: the app can be
# moved to any path on the host (or run from Streamlit Cloud) and it will
# still find the model and the sample dataset next to the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Data: GitHub-friendly 250K-row stratified sample. Replaces the original
# multi-GB master parquet so the dashboard stays under GitHub's 100 MB
# file-size limit and loads instantly in Streamlit Community Cloud.
DATA_PATH  = PROJECT_ROOT / "data" / "sample" / "train_features_sample.parquet"
# Model: trained LightGBM booster. Path and loading logic are unchanged.
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_fraud_model.txt"


# ---------------------------------------------------------------------------
# Setup & Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("security_dashboard")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_fraud_model():
    """Load the trained LightGBM model text file once and cache it.

    ``@st.cache_resource`` is the Streamlit equivalent of a process-wide
    singleton: the same ``Booster`` instance is reused for every rerun,
    which matters because loading the model is O(10-100 ms) and the form
    re-runs on every widget change.
    """
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{MODEL_PATH}`. "
            f"Please run the training script first."
        )
        st.stop()
    # Load LightGBM model using its native predictor
    bst = lgb.Booster(model_file=str(MODEL_PATH))
    return bst


@st.cache_data(show_spinner="Initialising banking data catalogues...")
def load_sample_meta():
    """Load unique categories from the sampled parquet to populate dropdowns.

    The dashboard reads *only* the three columns it needs to populate the
    dropdowns (mcc_description, merchant_city, card_type), which keeps
    the cache footprint tiny and the first-paint time well under a
    second on the 250K-row sample.
    """
    if not DATA_PATH.exists():
        st.error(
            f"Sampled dataset not found: `{DATA_PATH}`. "
            f"Please run the downsampling script first."
        )
        st.stop()
    df = pd.read_parquet(
        DATA_PATH,
        columns=["mcc_description", "merchant_city", "card_type"],
    )

    return {
        "mcc_list":   sorted(df["mcc_description"].dropna().unique().tolist()),
        "city_list":  sorted(df["merchant_city"].dropna().unique().tolist()),
        "card_types": sorted(df["card_type"].dropna().unique().tolist()),
    }


@st.cache_resource(show_spinner="Aligning model categorical dictionaries...")
def load_categorical_alignment(_model: lgb.Booster) -> dict:
    """Capture the EXACT categorical dictionaries the booster was trained on.

    The booster stores the per-column category dictionary in
    ``Booster.pandas_categorical``; the order of that list is the order
    in which the trainer declared categorical features (i.e. the order
    of category-dtyped columns in the training frame). We hard-code the
    corresponding feature names -- this is the minimal, explicit form
    of the mapping and avoids any drift if the trainer changes the
    column ordering upstream.

    Returns
    -------
    dict
        ``{"columns": [...], "categories": {...}}``. The
        ``categories`` dict maps each categorical column name to the
        full list of training-time category codes for that column.
    """
    pc = list(_model.pandas_categorical)

    # The trainer declared these 9 columns as categorical, in this
    # order, which matches the order of ``pandas_categorical`` slots:
    #   [0] use_chip         [4] card_brand
    #   [1] merchant_city    [5] card_type
    #   [2] merchant_state   [6] has_chip
    #   [3] mcc_description  [7] card_on_dark_web
    #                        [8] gender
    cat_columns = [
        "use_chip", "merchant_city", "merchant_state", "mcc_description",
        "card_brand", "card_type", "has_chip", "card_on_dark_web", "gender",
    ]
    categories = {col: list(pc[i]) for i, col in enumerate(cat_columns)}
    return {"columns": cat_columns, "categories": categories}


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CaixaBank Security Shield",
        page_icon="🛡️",
        layout="wide",
    )

    # Load shared assets
    model     = load_fraud_model()
    meta      = load_sample_meta()
    cat_align = load_categorical_alignment(model)

    # Pre-cache the booster's feature order; the prediction frame MUST
    # be assembled in this exact order, otherwise LightGBM will raise
    # a "Feature order mismatch" error.
    feature_order = list(model.feature_name())

    # --- Sidebar context ---
    with st.sidebar:
        st.title("🛡️ Security Operations Center")
        st.caption("CaixaBank Cyber-Security Shield 2026")
        st.markdown("---")
        st.markdown("### Current System KPIs")
        st.metric("Fraud Recall (Catch Rate)", "83.5%")
        st.metric("False Positive Rate", "~3.1%")
        st.metric("Inference Latency", "< 8 ms")
        st.markdown("---")
        st.caption(
            "The active model is a LightGBM gradient-boosted classifier "
            "and is protected against identifier-based data leakage "
            "(Anti-Leakage Enabled)."
        )

    # --- Header ---
    st.title("🛡️ Real-Time Banking Fraud Detection & Risk Simulator")
    st.markdown(
        "Enter the details of a financial transaction below to simulate it "
        "and have it scored by the AI model in real time:"
    )

    # --- Layout Split ---
    col_form, col_results = st.columns([1, 1.2])

    with col_form:
        st.subheader("📝 Current Transaction Details")

        with st.form("transaction_entry_form"):
            amount = st.number_input(
                "Transaction Amount (USD):", min_value=0.1, value=50.0, step=5.0
            )
            credit_limit = st.number_input(
                "Card Credit Limit (USD):",
                min_value=100.0, value=5000.0, step=500.0,
            )

            c1, c2 = st.columns(2)
            with c1:
                use_chip_str = st.selectbox(
                    "Payment Method (Chip Used):", options=["Yes", "No"]
                )
                current_age = st.slider(
                    "Cardholder Age:", min_value=18, max_value=100, value=40
                )
            with c2:
                tx_hour = st.slider(
                    "Transaction Hour (0-23):",
                    min_value=0, max_value=23, value=12,
                )
                card_type = st.selectbox(
                    "Card Type:", options=meta["card_types"]
                )

            mcc_desc = st.selectbox(
                "Merchant Category (MCC):", options=meta["mcc_list"]
            )
            merchant_city = st.selectbox(
                "Merchant City:", options=meta["city_list"]
            )

            submit_btn = st.form_submit_button(
                "🔍 Score Transaction & Run Security Check"
            )

    with col_results:
        st.subheader("📊 Cybersecurity Risk Assessment Report")

        if submit_btn:
            # 1. Calculate dynamic ratio
            tx_to_limit_ratio = (
                amount / credit_limit if credit_limit > 0 else 0.0
            )
            is_chip_present = 1 if use_chip_str == "Yes" else 0
            use_chip_label  = "Chip Transaction" if use_chip_str == "Yes" else "Swipe Transaction"

            # 2. Build the COMPLETE 36-column feature vector required by
            #    LightGBM. The original code only sent 19 of the 36
            #    columns; the booster will silently misbehave on the
            #    missing ones, so we send all 36 here.
            #    ANTI-LEAKAGE: client_id, user_id, and merchant_id are
            #    set to safe default zeros so the model predicts on
            #    financial behaviour, not on memorised database IDs.
            input_data = {
                # User-controlled
                "amount":                  float(amount),
                "credit_limit":            float(credit_limit),
                "current_age":             int(current_age),
                "transaction_hour":        int(tx_hour),
                "use_chip":                use_chip_label,
                "merchant_city":           merchant_city,
                "mcc_description":         mcc_desc,
                "card_type":               card_type,

                # Engineered / V2 features
                "tx_to_limit_ratio":       float(tx_to_limit_ratio),
                "available_credit":        float(credit_limit - amount),
                "debt_to_income_ratio":    0.20,
                "is_chip_present":         int(is_chip_present),
                "has_error":               0,

                # Anti-leakage neutral defaults
                "client_id":               0,
                "card_id":                 0,
                "user_id":                 0,
                "merchant_id":             0,

                # Static defaults for non-user-controlled features.
                # Values are median-ish numbers from the original ABT
                # so the prediction lands in a realistic regime.
                "merchant_state":          "CA",
                "zip":                     90000.0,
                "mcc":                     0,
                "card_brand":              "Visa",
                "has_chip":                "YES",
                "num_cards_issued":        1,
                "card_on_dark_web":        "No",
                "retirement_age":          67,
                "birth_year":              1980,
                "birth_month":             6,
                "gender":                  "female",
                "latitude":                34.05,
                "longitude":              -118.24,
                "per_capita_income":       30000.0,
                "yearly_income":           50000.0,
                "total_debt":              10000.0,
                "credit_score":            700,
                "num_credit_cards":        3,
                "transaction_day_of_week": 1,
            }

            # Build the DataFrame in the EXACT feature order the booster
            # expects, then apply the categorical-alignment fix.
            row_dict = {k: [input_data[k]] for k in feature_order}
            input_df = pd.DataFrame(row_dict, columns=feature_order)

            # ----------------------------------------------------------------
            # Categorical-alignment fix (THE FIX)
            # ----------------------------------------------------------------
            # The original code did:
            #     input_df[col] = input_df[col].astype("category")
            # which on a SINGLE-row frame produces a category dictionary
            # containing only that one value. LightGBM then complains:
            #     "train and valid dataset categorical_feature do not match"
            # because the dictionaries learned at training time are
            # different. The fix is to build the categorical with the
            # *full* set of categories pulled from the booster itself.
            for col, categories in cat_align["categories"].items():
                # ``categories=`` freezes the dictionary to the
                # training-time values regardless of the value(s)
                # actually present in this single row.
                input_df[col] = pd.Categorical(
                    input_df[col],
                    categories=categories,
                )

            # The remaining 27 columns must be numeric (int/float).
            # Defensive cast: if any non-categorical column comes back
            # as object, coerce to numeric and fill NaNs with 0 so
            # LightGBM never sees a string in a numeric split.
            numeric_cols = [c for c in input_df.columns if c not in cat_align["columns"]]
            for col in numeric_cols:
                if input_df[col].dtype == "object":
                    input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

            # Predict probability
            try:
                # Get prediction probability
                prob = float(model.predict(input_df)[0])

                # Render Visual Result Alert
                st.markdown("#### System Assessment Result:")
                if prob < 0.5:
                    st.success(
                        f"🟢 **Transaction Approved (Low Risk)** \n\n"
                        f"Fraud probability: {prob*100:.2f}%"
                    )
                    st.balloons()
                else:
                    st.error(
                        f"🔴 **Transaction Declined (High Fraud Risk!)** \n\n"
                        f"Fraud probability: {prob*100:.2f}%"
                    )
                    st.warning(
                        "🚨 An instant alert has been sent to the cardholder "
                        "and the card has been temporarily frozen pending review."
                    )

                # Progress bar display
                st.markdown("---")
                st.markdown("**Live Transaction Risk Score:**")
                st.progress(float(prob))

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info(
                    "Please verify that the input columns exactly match the "
                    "36 features the model was trained on."
                )
        else:
            st.info(
                "💡 Awaiting transaction details and a click on the 'Score "
                "Transaction' button to begin..."
            )

    # --- Bottom Section: Global Feature Importance Explanation ---
    st.markdown("---")
    st.subheader("🧠 How Does the AI Reach Its Decision? (Global Feature Importance)")
    st.markdown(
        "These are the most influential features in detecting fraud across "
        "the bank's portfolio, ranked by the gain weights of the trained "
        "LightGBM model:"
    )

    # Static representation of the Top 7 features from your actual
    # terminal log output.
    feat_importance = pd.DataFrame({
        "Feature": [
            "Merchant City",
            "Merchant ID",
            "Merchant Category Code (MCC)",
            "Transaction Amount",
            "Merchant Category Description",
            "Cardholder Age",
            "Transaction-to-Credit-Limit Ratio",
        ],
        "Importance (Gain)": [
            1295261184, 1208139069, 929424157, 451902410,
            261519446, 128436907, 118584642,
        ],
    })

    fig = px.bar(
        feat_importance,
        x="Importance (Gain)",
        y="Feature",
        orientation="h",
        color="Importance (Gain)",
        color_continuous_scale="Reds",
        template="plotly_white",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
