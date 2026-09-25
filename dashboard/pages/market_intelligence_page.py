"""
Market Intelligence Dashboard Page (Phase 10.3).

Provides interactive exploration of the Sri Lankan vehicle listing market
using collected public listing data.

Methodological Notice:
- This is a MARKET INTELLIGENCE dashboard, not a valuation-confidence dashboard.
- Summarizes advertised asking prices, which may differ from negotiated selling prices.
- Statistical associations are descriptive and do NOT imply mathematical causality.
- Observed listing disappearance does NOT by itself confirm a vehicle sale.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.market.market_analytics import (
    apply_filters,
    compute_market_overview,
    get_brand_summary,
    get_category_summary,
    get_filter_options,
    get_model_summary,
)
from eda.dataset import EDADatasetLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme Helper for Plotly Charts
# ---------------------------------------------------------------------------

def _apply_dark_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Applies unified dark glassmorphism styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.7)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(99, 102, 241, 0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(99, 102, 241, 0.15)")
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_lkr(amount: Optional[float]) -> str:
    """Format a LKR value as '12,500,000' or 'N/A'."""
    if amount is None or pd.isna(amount):
        return "N/A"
    return f"{amount:,.0f} LKR"


def _fmt_num(val: Optional[float | int]) -> str:
    """Format an integer or number with commas."""
    if val is None or pd.isna(val):
        return "0"
    return f"{val:,.0f}"


# ---------------------------------------------------------------------------
# Cached Data Loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Loading vehicle market dataset...")
def load_market_data() -> pd.DataFrame:
    """
    Loads vehicle listings with joined attributes, latest prices, and observations.
    Uses EDADatasetLoader with Streamlit in-memory caching to avoid redundant queries.
    """
    try:
        loader = EDADatasetLoader()
        df = loader.load_listings()
        return df
    except Exception as e:
        logger.error(f"Failed to load market listings: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main Page Renderer
# ---------------------------------------------------------------------------

def render_market_intelligence_page(api_url: str = "http://localhost:8000") -> None:
    """Renders the Market Intelligence Dashboard page."""

    # ── Page Header / Hero ───────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 1.5rem 0 1rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2.4rem;
                        font-weight:700; color:#ffffff; line-height:1.2;
                        text-shadow: 0 0 30px rgba(99,102,241,0.5);">
                📈 Market Intelligence Dashboard
            </div>
            <div style="font-size:1.05rem; color:#94a3b8; margin-top:0.5rem; line-height:1.5;">
                Explore real-time descriptive statistics, category distributions, brand pricing,
                and geographic patterns from the Sri Lankan vehicle market.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Market Scope & Limitation Notice ─────────────────────────────────
    st.markdown(
        """
        <div class="info-card" style="border-left: 4px solid #6366f1; margin-bottom: 1.2rem;">
            <div style="font-weight: 600; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 0.2rem;">
                📌 Market Scope & Asking Price Notice
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
                This dashboard describes advertised vehicle listings collected from configured public sources (Riyasewana).
                <strong>Advertised asking prices may differ from negotiated final transaction prices.</strong>
                Observed listing disappearance does not by itself confirm a vehicle sale.
                Statistical correlations reflect observed market patterns and do not imply mathematical causality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load Dataset ─────────────────────────────────────────────────────
    df_raw = load_market_data()

    if df_raw.empty:
        st.warning("No market listing records found in the database. Please verify the database connection.")
        return

    # Extract dynamic options from current dataset
    opts = get_filter_options(df_raw)

    # ── Global Filters ───────────────────────────────────────────────────
    with st.expander("🔍 **Global Market Filters** (Click to expand/collapse)", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            sel_categories = st.multiselect(
                "Vehicle Category",
                options=opts["categories"],
                default=[],
                placeholder="All Categories",
                help="Filter by vehicle categories present in the dataset.",
                key="filter_categories",
            )

            # Available brands dynamic filter
            cat_filtered_df = (
                df_raw[df_raw["canonical_category"].isin(sel_categories)]
                if sel_categories
                else df_raw
            )
            brand_options = sorted(
                [str(b) for b in cat_filtered_df["brand"].dropna().unique() if str(b).strip()]
            )

            sel_brands = st.multiselect(
                "Brand",
                options=brand_options,
                default=[],
                placeholder="All Brands",
                help="Filter by brands available in the selected categories.",
                key="filter_brands",
            )

            # Available models dynamic filter
            brand_filtered_df = (
                cat_filtered_df[cat_filtered_df["brand"].isin(sel_brands)]
                if sel_brands
                else cat_filtered_df
            )
            model_options = sorted(
                [str(m) for m in brand_filtered_df["model"].dropna().unique() if str(m).strip()]
            )

            sel_models = st.multiselect(
                "Model",
                options=model_options,
                default=[],
                placeholder="All Models",
                help="Filter by models available for the selected brands.",
                key="filter_models",
            )

        with col2:
            sel_districts = st.multiselect(
                "District",
                options=opts["districts"],
                default=[],
                placeholder="All Districts",
                help="Filter by Sri Lankan administrative districts.",
                key="filter_districts",
            )

            sel_fuels = st.multiselect(
                "Fuel Type",
                options=opts["fuel_types"],
                default=[],
                placeholder="All Fuel Types",
                help="Filter by fuel type (Petrol, Diesel, Hybrid, etc.).",
                key="filter_fuels",
            )

            sel_transmissions = st.multiselect(
                "Transmission",
                options=opts["transmissions"],
                default=[],
                placeholder="All Transmissions",
                help="Filter by transmission (Automatic, Manual).",
                key="filter_transmissions",
            )

        with col3:
            sel_conditions = st.multiselect(
                "Condition",
                options=opts["conditions"],
                default=[],
                placeholder="All Conditions",
                help="Filter by condition (e.g. Registered (Used), Unregistered).",
                key="filter_conditions",
            )

            # Year range slider
            y_min, y_max = opts["year_min"], opts["year_max"]
            if y_min < y_max:
                sel_year_range = st.slider(
                    "Manufacture Year Range",
                    min_value=y_min,
                    max_value=y_max,
                    value=(y_min, y_max),
                    key="filter_year_range",
                )
            else:
                sel_year_range = (y_min, y_max)
                st.caption(f"Manufacture Year: {y_min}")

            # Price range slider
            p_min, p_max = opts["price_min"], opts["price_max"]
            if p_min < p_max:
                sel_price_range = st.slider(
                    "Asking Price Range (LKR)",
                    min_value=p_min,
                    max_value=p_max,
                    value=(p_min, p_max),
                    step=max(50_000, (p_max - p_min) // 100),
                    format="%d",
                    key="filter_price_range",
                )
            else:
                sel_price_range = (p_min, p_max)
                st.caption(f"Asking Price: {_fmt_lkr(p_min)}")

        # Active filter count indicator
        active_filters = []
        if sel_categories:
            active_filters.append(f"Categories ({len(sel_categories)})")
        if sel_brands:
            active_filters.append(f"Brands ({len(sel_brands)})")
        if sel_models:
            active_filters.append(f"Models ({len(sel_models)})")
        if sel_districts:
            active_filters.append(f"Districts ({len(sel_districts)})")
        if sel_fuels:
            active_filters.append(f"Fuel ({len(sel_fuels)})")
        if sel_transmissions:
            active_filters.append(f"Transmission ({len(sel_transmissions)})")
        if sel_conditions:
            active_filters.append(f"Condition ({len(sel_conditions)})")
        if sel_year_range != (y_min, y_max):
            active_filters.append(f"Years ({sel_year_range[0]}–{sel_year_range[1]})")
        if sel_price_range != (p_min, p_max):
            active_filters.append(f"Price ({_fmt_lkr(sel_price_range[0])}–{_fmt_lkr(sel_price_range[1])})")

        if active_filters:
            st.markdown(
                f"<div style='font-size:0.85rem; color:#a5b4fc; padding-top:0.4rem;'>"
                f"⚡ <strong>Active Filters:</strong> {', '.join(active_filters)}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Apply Filters to In-Memory DataFrame ─────────────────────────────
    df_filtered = apply_filters(
        df=df_raw,
        categories=sel_categories,
        brands=sel_brands,
        models=sel_models,
        districts=sel_districts,
        fuel_types=sel_fuels,
        transmissions=sel_transmissions,
        conditions=sel_conditions,
        year_range=sel_year_range,
        price_range=sel_price_range,
    )

    st.caption(
        f"Displaying **{len(df_filtered):,}** of **{len(df_raw):,}** total listings in dataset."
    )

    if df_filtered.empty:
        st.info("⚠️ No listings match the selected filters. Please adjust your filter criteria.")
        return

    # ── Market Overview KPIs ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Market Overview KPIs</div>", unsafe_allow_html=True)
    overview = compute_market_overview(df_filtered)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(
            label="Total Listings",
            value=_fmt_num(overview["total_listings"]),
            help="Total advertised market listing records in the filtered view.",
        )
    with kpi_col2:
        st.metric(
            label="Vehicle Specs",
            value=_fmt_num(overview["total_vehicles"]),
            help="Vehicle specification records associated with listings in the current schema.",
        )
    with kpi_col3:
        st.metric(
            label="Median Asking Price",
            value=_fmt_lkr(overview["median_asking_price"]),
            help="50th percentile of advertised asking prices in LKR. Robust against extreme values.",
        )
    with kpi_col4:
        st.metric(
            label="Average Asking Price",
            value=_fmt_lkr(overview["average_asking_price"]),
            help="Arithmetic mean of advertised asking prices in LKR.",
        )

    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
    with kpi_col5:
        st.metric(
            label="Categories",
            value=_fmt_num(overview["distinct_categories"]),
            help="Number of distinct vehicle categories in view.",
        )
    with kpi_col6:
        st.metric(
            label="Brands",
            value=_fmt_num(overview["distinct_brands"]),
            help="Number of distinct vehicle makes/brands in view.",
        )
    with kpi_col7:
        st.metric(
            label="Models",
            value=_fmt_num(overview["distinct_models"]),
            help="Number of distinct vehicle models in view.",
        )
    with kpi_col8:
        st.metric(
            label="ML Eligible",
            value=f"{overview['ml_eligible_pct']}% ({overview['ml_eligible_listings']})",
            help="Listings meeting quality and attribute requirements for machine learning modeling.",
        )

    st.markdown(
        """
        <div style="font-size:0.78rem; color:#64748b; margin-top:-0.5rem; margin-bottom:1.5rem;">
            ℹ️ <em>Note on Listings vs Vehicles:</em> In the current database schema, each collected listing corresponds to an associated vehicle record.
            Listing count reflects individual advertised market postings, not necessarily distinct physical vehicles across repeat postings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Category Analysis ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🚗 Vehicle Category Analysis</div>", unsafe_allow_html=True)
    st.caption("Compare listing volumes, market share, and advertised asking price levels across vehicle categories.")

    cat_df = get_category_summary(df_filtered)
    if cat_df.empty or cat_df["listing_count"].sum() == 0:
        st.info("No category data available for the current filter selection.")
    else:
        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            # Bar chart: Listings by Category
            fig_vol = go.Figure(
                data=[
                    go.Bar(
                        x=cat_df["category"],
                        y=cat_df["listing_count"],
                        marker=dict(
                            color=cat_df["listing_count"],
                            colorscale="Viridis",
                            line=dict(color="#6366f1", width=1),
                        ),
                        text=cat_df["listing_count"],
                        textposition="auto",
                        hovertemplate="<b>%{x}</b><br>Listings: %{y}<br>Share: %{customdata:.1f}%<extra></extra>",
                        customdata=cat_df["pct_of_total"],
                    )
                ]
            )
            fig_vol.update_layout(
                title="Listings by Category",
                xaxis_title="Category",
                yaxis_title="Listing Count",
            )
            _apply_dark_theme(fig_vol)
            st.plotly_chart(fig_vol, use_container_width=True)

        with cat_col2:
            # Grouped Bar chart: Median vs Average Asking Price by Category
            fig_price = go.Figure(
                data=[
                    go.Bar(
                        name="Median Asking Price",
                        x=cat_df["category"],
                        y=cat_df["median_asking_price"],
                        marker_color="#6366f1",
                        hovertemplate="<b>%{x}</b><br>Median Asking Price: %{y:,.0f} LKR<extra></extra>",
                    ),
                    go.Bar(
                        name="Average Asking Price",
                        x=cat_df["category"],
                        y=cat_df["mean_asking_price"],
                        marker_color="#a5b4fc",
                        hovertemplate="<b>%{x}</b><br>Average Asking Price: %{y:,.0f} LKR<extra></extra>",
                    ),
                ]
            )
            fig_price.update_layout(
                title="Advertised Asking Price by Category (Median vs Mean)",
                xaxis_title="Category",
                yaxis_title="Advertised Asking Price (LKR)",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            _apply_dark_theme(fig_price)
            st.plotly_chart(fig_price, use_container_width=True)

        # Category Breakdown Table
        with st.expander("📋 View Category Summary Table", expanded=False):
            display_cat_df = cat_df.copy()
            display_cat_df["median_asking_price"] = display_cat_df["median_asking_price"].apply(
                lambda p: _fmt_lkr(p) if pd.notna(p) else "N/A"
            )
            display_cat_df["mean_asking_price"] = display_cat_df["mean_asking_price"].apply(
                lambda p: _fmt_lkr(p) if pd.notna(p) else "N/A"
            )
            display_cat_df["median_mileage"] = display_cat_df["median_mileage"].apply(
                lambda m: f"{m:,.0f} km" if pd.notna(m) else "N/A"
            )
            display_cat_df["pct_of_total"] = display_cat_df["pct_of_total"].apply(
                lambda pct: f"{pct:.1f}%"
            )
            display_cat_df["ml_eligible_pct"] = display_cat_df["ml_eligible_pct"].apply(
                lambda pct: f"{pct:.1f}%"
            )
            display_cat_df = display_cat_df.rename(
                columns={
                    "category": "Category",
                    "listing_count": "Listings",
                    "pct_of_total": "Market Share",
                    "ml_eligible_count": "ML Eligible",
                    "ml_eligible_pct": "ML Eligible %",
                    "median_asking_price": "Median Asking Price",
                    "mean_asking_price": "Mean Asking Price",
                    "median_mileage": "Median Mileage",
                    "median_yom": "Median YOM",
                }
            )
            st.dataframe(display_cat_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Brand & Model Analysis ───────────────────────────────────────────
    st.markdown("<div class='section-title'>🏷️ Brand & Model Market Analysis</div>", unsafe_allow_html=True)
    st.caption(
        "Descriptive statistics for vehicle makes and models. Not a quality or reliability ranking. "
        "Models shown respect the selected category and brand filters."
    )

    brand_col_ctl1, brand_col_ctl2 = st.columns([1, 3])
    with brand_col_ctl1:
        top_n_brands = st.slider(
            "Top N Brands to Display",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
            key="slider_top_n_brands",
        )

    brand_df = get_brand_summary(df_filtered, min_sample=1, top_n=top_n_brands)
    model_df = get_model_summary(df_filtered, min_sample=1, top_n=15)

    if brand_df.empty or brand_df["listing_count"].sum() == 0:
        st.info("No brand data available for the current filter selection.")
    else:
        bm_col1, bm_col2 = st.columns(2)

        with bm_col1:
            # Top Brands by Listing Count (Horizontal bar)
            sorted_brands_cnt = brand_df.sort_values(by="listing_count", ascending=True)
            fig_brands = go.Figure(
                data=[
                    go.Bar(
                        y=sorted_brands_cnt["brand"],
                        x=sorted_brands_cnt["listing_count"],
                        orientation="h",
                        marker=dict(
                            color=sorted_brands_cnt["listing_count"],
                            colorscale="Blues",
                            line=dict(color="#3b82f6", width=1),
                        ),
                        text=sorted_brands_cnt["listing_count"],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Listings: %{x}<br>Share: %{customdata:.1f}%<extra></extra>",
                        customdata=sorted_brands_cnt["pct_of_total"],
                    )
                ]
            )
            fig_brands.update_layout(
                title=f"Top {len(brand_df)} Brands by Listing Count",
                xaxis_title="Listing Count",
                yaxis_title="Brand",
            )
            _apply_dark_theme(fig_brands)
            st.plotly_chart(fig_brands, use_container_width=True)

        with bm_col2:
            # Median Asking Price by Top Brands
            sorted_brands_price = brand_df.dropna(subset=["median_asking_price"]).sort_values(
                by="median_asking_price", ascending=True
            )
            fig_brand_price = go.Figure(
                data=[
                    go.Bar(
                        y=sorted_brands_price["brand"],
                        x=sorted_brands_price["median_asking_price"],
                        orientation="h",
                        marker=dict(
                            color=sorted_brands_price["median_asking_price"],
                            colorscale="Purples",
                            line=dict(color="#8b5cf6", width=1),
                        ),
                        text=[_fmt_lkr(p) for p in sorted_brands_price["median_asking_price"]],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Median Asking Price: %{x:,.0f} LKR<extra></extra>",
                    )
                ]
            )
            fig_brand_price.update_layout(
                title=f"Median Asking Price by Brand (Top {len(sorted_brands_price)})",
                xaxis_title="Advertised Asking Price (LKR)",
                yaxis_title="Brand",
            )
            _apply_dark_theme(fig_brand_price)
            st.plotly_chart(fig_brand_price, use_container_width=True)

        # Asking Price Box Plot for Major Brands
        top_brand_names = list(brand_df["brand"].unique()[:top_n_brands])
        df_top_brands = df_filtered[df_filtered["brand"].isin(top_brand_names) & df_filtered["asking_price"].notna()]
        if not df_top_brands.empty and len(top_brand_names) > 0:
            fig_box = go.Figure()
            for b_name in top_brand_names:
                b_prices = df_top_brands[df_top_brands["brand"] == b_name]["asking_price"]
                if not b_prices.empty:
                    fig_box.add_trace(
                        go.Box(
                            y=b_prices,
                            name=b_name,
                            boxpoints="outliers",
                            jitter=0.3,
                            pointpos=-1.8,
                            marker_color="#818cf8",
                            hovertemplate=f"<b>{b_name}</b><br>Asking Price: %{{y:,.0f}} LKR<extra></extra>",
                        )
                    )
            fig_box.update_layout(
                title=f"Asking Price Distribution by Brand (Top {len(top_brand_names)})",
                xaxis_title="Brand",
                yaxis_title="Advertised Asking Price (LKR)",
                showlegend=False,
            )
            _apply_dark_theme(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Model Breakdown ──────────────────────────────────────────────────
    if not model_df.empty and model_df["listing_count"].sum() > 0:
        mod_col1, mod_col2 = st.columns(2)

        with mod_col1:
            sorted_models = model_df.head(12).sort_values(by="listing_count", ascending=True)
            labels = [f"{b} {m}" for b, m in zip(sorted_models["brand"], sorted_models["model"])]
            fig_models = go.Figure(
                data=[
                    go.Bar(
                        y=labels,
                        x=sorted_models["listing_count"],
                        orientation="h",
                        marker=dict(
                            color=sorted_models["listing_count"],
                            colorscale="Teal",
                            line=dict(color="#14b8a6", width=1),
                        ),
                        text=sorted_models["listing_count"],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Listings: %{x}<extra></extra>",
                    )
                ]
            )
            fig_models.update_layout(
                title=f"Top Models by Listing Count",
                xaxis_title="Listing Count",
                yaxis_title="Model",
            )
            _apply_dark_theme(fig_models)
            st.plotly_chart(fig_models, use_container_width=True)

        with mod_col2:
            models_with_price = model_df.dropna(subset=["median_asking_price"]).head(12)
            sorted_m_price = models_with_price.sort_values(by="median_asking_price", ascending=True)
            labels_p = [f"{b} {m}" for b, m in zip(sorted_m_price["brand"], sorted_m_price["model"])]
            fig_m_price = go.Figure(
                data=[
                    go.Bar(
                        y=labels_p,
                        x=sorted_m_price["median_asking_price"],
                        orientation="h",
                        marker=dict(
                            color=sorted_m_price["median_asking_price"],
                            colorscale="Viridis",
                            line=dict(color="#10b981", width=1),
                        ),
                        text=[_fmt_lkr(p) for p in sorted_m_price["median_asking_price"]],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Median Asking Price: %{x:,.0f} LKR<extra></extra>",
                    )
                ]
            )
            fig_m_price.update_layout(
                title=f"Median Asking Price by Model",
                xaxis_title="Advertised Asking Price (LKR)",
                yaxis_title="Model",
            )
            _apply_dark_theme(fig_m_price)
            st.plotly_chart(fig_m_price, use_container_width=True)

        # Brand and Model Expanders
        with st.expander("📋 View Brand & Model Summary Tables", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Brand Summary**")
                disp_b = brand_df.copy()
                disp_b["median_asking_price"] = disp_b["median_asking_price"].apply(lambda p: _fmt_lkr(p) if pd.notna(p) else "N/A")
                disp_b["mean_asking_price"] = disp_b["mean_asking_price"].apply(lambda p: _fmt_lkr(p) if pd.notna(p) else "N/A")
                disp_b = disp_b.rename(columns={
                    "brand": "Brand",
                    "listing_count": "Listings",
                    "median_asking_price": "Median Asking Price",
                    "mean_asking_price": "Mean Asking Price",
                    "sample_flag": "Sample Status",
                })
                st.dataframe(disp_b[["Brand", "Listings", "Median Asking Price", "Mean Asking Price", "Sample Status"]], use_container_width=True, hide_index=True)

            with t_col2:
                st.markdown("**Model Summary**")
                disp_m = model_df.copy()
                disp_m["median_asking_price"] = disp_m["median_asking_price"].apply(lambda p: _fmt_lkr(p) if pd.notna(p) else "N/A")
                disp_m["median_mileage"] = disp_m["median_mileage"].apply(lambda m: f"{m:,.0f} km" if pd.notna(m) else "N/A")
                disp_m = disp_m.rename(columns={
                    "brand": "Brand",
                    "model": "Model",
                    "listing_count": "Listings",
                    "median_asking_price": "Median Asking Price",
                    "median_mileage": "Median Mileage",
                    "sample_flag": "Sample Status",
                })
                st.dataframe(disp_m[["Brand", "Model", "Listings", "Median Asking Price", "Median Mileage", "Sample Status"]], use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
