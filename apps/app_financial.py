"""CaixaBank Portfolio Health & Spending BI Dashboard (sample build).

Reads the stratified 250K-row sample (``train_features_sample.parquet``)
and renders an executive-grade Streamlit view of the portfolio. Built
to deploy cleanly on GitHub (< 100 MB) and run interactively on
Streamlit Community Cloud.

Sections
========

  1. Executive Summary (KPIs)
  2. Credit Risk Analytics (income vs debt scatter, utilisation by card)
  3. Spending Behaviour (top MCCs, hourly volume)
  4. Demographic Insights (age-group spend distribution)

All charts use Plotly Express / Graph Objects for native zoom / hover.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("financial_dashboard")
if not logger.handlers:                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Constants -- aligned to the deployable sample
# ---------------------------------------------------------------------------
APP_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "train_features_sample.parquet"

# The sample is exactly 250,000 rows, so the slider cannot exceed that.
DEFAULT_SAMPLE = 100_000
MAX_SAMPLE     = 250_000

# Industry-standard 40% debt-to-income threshold used in the scatter plot.
DTI_THRESHOLD = 0.40

# Age bins for the demographic chart. The sample spans 18..101, so we
# use decades and clamp the tail.
AGE_BINS   = [18, 25, 35, 45, 55, 65, 75, 101]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
TEMPLATE_LIGHT = "plotly_white"
TEMPLATE_DARK  = "plotly_dark"


# ---------------------------------------------------------------------------
# Cached data load
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading train_features_sample.parquet ...")
def load_data() -> pd.DataFrame:
    """Read the 250K-row sample once and cache it.

    The sample is ~16 MB on disk, so the read is fast; the cache keeps
    it warm across user interactions.
    """
    if not DATA_PATH.exists():
        st.error(
            f"Could not find `{DATA_PATH}`. "
            f"Run `create_sample.py` first to produce it."
        )
        st.stop()
    return pd.read_parquet(DATA_PATH)


@st.cache_data(show_spinner=False)
def sample_dataframe(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic random sample of the cached frame.

    ``random_state`` makes successive runs of the dashboard identical
    for the same slider position -- important when a screenshot ends
    up in a slide deck.
    """
    if n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(df: pd.DataFrame) -> Tuple[str, list[str], list[str], int, int]:
    """Render the sidebar and return the user-selected options."""
    with st.sidebar:
        st.title("Portfolio BI")
        st.caption("CaixaBank Hackathon 2026")

        st.markdown("### Sample size")
        # The slider is the single biggest performance lever: 100K
        # rows renders in < 1 s. The cap is the sample's own size --
        # we never want to ask Streamlit for more rows than the file
        # actually has.
        sample_size = st.slider(
            "Rows to analyse",
            min_value=10_000,
            max_value=MAX_SAMPLE,
            value=DEFAULT_SAMPLE,
            step=10_000,
            help="Larger samples are more accurate but slower to render.",
        )

        st.markdown("### Theme")
        theme = st.radio(
            "Visual style",
            options=["Light", "Dark"],
            index=0,
            horizontal=True,
        )

        st.markdown("### Filters")
        all_card_types = sorted(df["card_type"].dropna().unique().tolist())
        card_types = st.multiselect(
            "Card type",
            options=all_card_types,
            default=all_card_types,
            help="Restrict the dashboard to a subset of card products.",
        )

        # ``merchant_state`` has NaNs in the source; we hide those
        # from the filter list so the user only sees real codes.
        top_states = (
            df["merchant_state"].value_counts().head(20).index.tolist()
        )
        state_choice = st.multiselect(
            "Merchant state (top 20 by volume)",
            options=top_states,
            default=top_states,
            help="Restrict to the 20 most active merchant states.",
        )

        st.markdown("### About")
        st.caption(
            "Data: 250K stratified sample of the 8.9M labelled training "
            "split (fraud rate preserved at ~0.15%). Charts use Plotly -- "
            "hover / zoom / lasso are all live."
        )

    return theme.lower(), card_types, state_choice, sample_size, 42


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------
def _apply_filters(
    df: pd.DataFrame,
    card_types: list[str],
    states: list[str],
) -> pd.DataFrame:
    """Apply the sidebar filters to a (possibly already-sampled) frame."""
    out = df
    if card_types:
        out = out[out["card_type"].isin(card_types)]
    if states:
        out = out[out["merchant_state"].isin(states)]
    return out


# ---------------------------------------------------------------------------
# Executive KPIs
# ---------------------------------------------------------------------------
def _render_kpis(df: pd.DataFrame) -> None:
    """Top-row KPI tiles. All values come from a single vectorised
    pass over the filtered frame -- the four numbers together cost
    about 30 ms on the 100K default sample."""
    total_volume       = float(df["amount"].sum())
    avg_dti            = float(df["debt_to_income_ratio"].mean())
    avg_utilization    = float(df["tx_to_limit_ratio"].mean())
    avg_yearly_income  = float(df["yearly_income"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Volume (USD)", f"${total_volume/1e6:,.1f}M")
    c2.metric("Avg Debt-to-Income",     f"{avg_dti:.2f}")
    c3.metric("Avg Credit Utilisation", f"{avg_utilization:.4f}")
    c4.metric("Avg Yearly Income (USD)", f"${avg_yearly_income:,.0f}")


# ---------------------------------------------------------------------------
# Credit Risk section
# ---------------------------------------------------------------------------
def _render_risk(df: pd.DataFrame, template: str) -> None:
    st.subheader("Credit Risk Analytics")

    # --- Scatter: Yearly Income vs. Total Debt, coloured by is_fraud.
    # With 250K rows the entire frame fits comfortably in Plotly's
    # WebGL canvas, so no further sub-sampling is needed.
    scatter_df = df[["yearly_income", "total_debt", "is_fraud"]].dropna()

    fig_scatter = px.scatter(
        scatter_df,
        x="yearly_income",
        y="total_debt",
        color=scatter_df["is_fraud"].astype(str),
        color_discrete_map={"0": "#1f77b4", "1": "#d62728"},
        labels={"color": "is_fraud",
                "yearly_income": "Yearly Income (USD)",
                "total_debt":     "Total Debt (USD)"},
        opacity=0.55,
        template=template,
        title="Yearly Income vs. Total Debt (red = fraud)",
    )
    # The 40% DTI threshold: ``y = 0.4 * x``. We pick the x-range from
    # the data so the line spans the visible plot.
    x_max = float(scatter_df["yearly_income"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, x_max], y=[0, x_max * DTI_THRESHOLD],
        mode="lines",
        line=dict(color="orange", width=3, dash="dash"),
        name=f"{int(DTI_THRESHOLD*100)}% DTI threshold",
    ))
    fig_scatter.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Utilisation by card type: a box plot exposes the *spread*,
    # which is what an underwriter actually wants to see.
    util_df = df[["card_type", "tx_to_limit_ratio"]].dropna()
    fig_box = px.box(
        util_df,
        x="card_type",
        y="tx_to_limit_ratio",
        color="card_type",
        template=template,
        title="Credit Utilisation (tx_to_limit_ratio) by Card Type",
        labels={"tx_to_limit_ratio": "tx_to_limit_ratio",
                "card_type": "Card Type"},
    )
    # Clip the y-axis at the 99th percentile so a handful of extreme
    # outliers don't squash the boxes into a single line.
    p99 = float(util_df["tx_to_limit_ratio"].quantile(0.99))
    fig_box.update_yaxes(range=[0, p99])
    st.plotly_chart(fig_box, use_container_width=True)


# ---------------------------------------------------------------------------
# Spending Behaviour
# ---------------------------------------------------------------------------
def _render_spending(df: pd.DataFrame, template: str) -> None:
    st.subheader("Spending Behaviour")

    # --- Top-10 MCC by total spend. We aggregate *once* and then
    # slice the top 10 in pandas; that's far cheaper than letting
    # Plotly sort 250K points.
    mcc_spend = (
        df.groupby("mcc_description", observed=True)["amount"]
          .sum()
          .sort_values(ascending=True)        # ascending for horizontal bar
          .tail(10)
    )
    fig_mcc = px.bar(
        x=mcc_spend.values,
        y=mcc_spend.index,
        orientation="h",
        template=template,
        labels={"x": "Total Spend (USD)", "y": "MCC Description"},
        title="Top 10 Merchant Categories by Spend",
        color=mcc_spend.values,
        color_continuous_scale="Blues",
    )
    fig_mcc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_mcc, use_container_width=True)

    # --- Hourly volume: a line chart of total spend by hour. Showing
    # all 24 hours is the right granularity -- anything finer would
    # need a date-range filter the user doesn't have.
    hourly = (
        df.groupby("transaction_hour", observed=True)["amount"]
          .sum()
          .reindex(range(24), fill_value=0)
    )
    fig_hour = px.line(
        x=hourly.index,
        y=hourly.values,
        template=template,
        markers=True,
        labels={"x": "Hour of Day", "y": "Total Spend (USD)"},
        title="Spending Volume by Hour of Day",
    )
    fig_hour.update_xaxes(dtick=1)
    st.plotly_chart(fig_hour, use_container_width=True)


# ---------------------------------------------------------------------------
# Demographic Insights
# ---------------------------------------------------------------------------
def _render_demographics(df: pd.DataFrame, template: str) -> None:
    st.subheader("Demographic Insights")

    # --- Age-group distribution. We use ``pd.cut`` to bucket the
    # continuous age column once, then groupby + sum to get the
    # spending per bucket. ``observed=True`` keeps the categorical
    # groupby fast.
    age_bucket = pd.cut(
        df["current_age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )
    spend_by_age = (
        df.assign(age_group=age_bucket)
          .groupby("age_group", observed=True)["amount"]
          .sum()
          .reindex(AGE_LABELS, fill_value=0)
    )
    fig_age = px.bar(
        x=spend_by_age.index.astype(str),
        y=spend_by_age.values,
        template=template,
        labels={"x": "Age Group", "y": "Total Spend (USD)"},
        title="Total Spend by Age Group",
        color=spend_by_age.values,
        color_continuous_scale="Viridis",
    )
    fig_age.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_age, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="CaixaBank Portfolio Health",
        page_icon="\U0001F4CA",
        layout="wide",
    )

    df_full = load_data()
    theme, card_types, states, sample_size, seed = _render_sidebar(df_full)

    # Apply sampling first (cheap), then filters (also cheap). Doing
    # it in that order means the user gets a stable picture even when
    # they crank the slider; the filter is just a slice on top.
    df_sampled = sample_dataframe(df_full, sample_size, seed)
    df_view    = _apply_filters(df_sampled, card_types, states)

    if df_view.empty:
        st.warning("No rows match the current filters. "
                   "Try widening the card type / state selection.")
        return

    st.title("Banking Portfolio Health & Spending Patterns")
    st.caption(
        f"Showing **{len(df_view):,}** sampled transactions "
        f"(of {len(df_full):,} total) | card types: {len(card_types)} | "
        f"states: {len(states)}"
    )

    template = TEMPLATE_DARK if theme == "dark" else TEMPLATE_LIGHT

    _render_kpis(df_view)
    st.divider()
    _render_risk(df_view, template)
    st.divider()
    _render_spending(df_view, template)
    st.divider()
    _render_demographics(df_view, template)


if __name__ == "__main__":
    main()




