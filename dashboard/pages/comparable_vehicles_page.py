"""
Comparable Vehicles Search Page (Phase 10.4).

Provides a dedicated interactive search workflow for finding similar market
listings from the existing vehicle database using the existing
ComparableVehicleEngine (multi-attribute weighted similarity scoring).

Methodological Notices:
- Similarity score represents multi-attribute SPECIFICATION DISTANCE between
  the entered vehicle and candidate listings. It is NOT:
  * Prediction confidence
  * Valuation accuracy
  * Probability of correctness
  * Match certainty
  * Price guarantee
- All prices are advertised ASKING PRICES observed on market listings.
  They do NOT represent confirmed transaction or negotiated selling prices.
- Statistical associations are descriptive and do NOT imply mathematical causality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.comparables.comparable_engine import (
    ComparableVehicle,
    ComparableVehicleEngine,
)
from analytics.comparables.market_summary import create_comparable_market_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain constants (mirroring canonical values in cleaners.py / valuation_page.py)
# ---------------------------------------------------------------------------

VEHICLE_CATEGORIES = [
    "Cars",
    "SUVs",
    "Vans",
    "Motorbikes",
    "Pickups",
    "Lorries",
    "Three Wheelers",
    "Heavy-Duty",
]

FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric", "LPG", "CNG", "Gas"]

TRANSMISSIONS = ["Automatic", "Manual"]

CONDITIONS = ["Registered (Used)", "Unregistered", "Brand New"]

SRI_LANKA_DISTRICTS = [
    "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo",
    "Galle", "Gampaha", "Hambantota", "Jaffna", "Kalutara",
    "Kandy", "Kegalle", "Kilinochchi", "Kurunegala", "Mannar",
    "Matale", "Matara", "Monaragala", "Mullaitivu", "Nuwara Eliya",
    "Polonnaruwa", "Puttalam", "Ratnapura", "Trincomalee", "Vavuniya",
]

CURRENT_YEAR = 2026


# ---------------------------------------------------------------------------
# Plotly dark theme helper (consistent with Phase 10.3 market_intelligence_page)
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
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_lkr(amount: Optional[float]) -> str:
    """Format a LKR amount as a readable string, or 'N/A' if missing."""
    if amount is None or pd.isna(amount):
        return "N/A"
    return f"{amount:,.0f} LKR"


def _fmt_km(mileage: Optional[float]) -> str:
    """Format mileage in km with commas."""
    if mileage is None or pd.isna(mileage):
        return "N/A"
    return f"{mileage:,.0f} km"


def _fmt_cc(cc: Optional[float]) -> str:
    """Format engine displacement in cc."""
    if cc is None or pd.isna(cc):
        return "N/A"
    return f"{cc:,.0f} cc"


def _fmt_pct(pct: Optional[float]) -> str:
    """Format similarity percentage."""
    if pct is None or pd.isna(pct):
        return "N/A"
    return f"{pct:.1f}%"


def _safe_val(val: Any, fallback: str = "N/A") -> str:
    """Return val as string or fallback if None/nan."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return fallback
    s = str(val).strip()
    return s if s else fallback


# ---------------------------------------------------------------------------
# Comparable engine (cached at session level to avoid re-instantiation)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_comparable_engine() -> ComparableVehicleEngine:
    """Cached singleton of the ComparableVehicleEngine."""
    return ComparableVehicleEngine()


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_comparable_vehicles_page(api_url: str = "http://localhost:8000") -> None:
    """Renders the Comparable Vehicles Search page (Phase 10.4)."""

    # Page Header
    st.markdown(
        """
        <div style="padding: 1.5rem 0 1rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2.4rem;
                        font-weight:700; color:#ffffff; line-height:1.2;
                        text-shadow: 0 0 30px rgba(99,102,241,0.5);">
                🔎 Comparable Vehicles
            </div>
            <div style="font-size:1.05rem; color:#94a3b8; margin-top:0.5rem; line-height:1.5;">
                Find similar vehicles in the market database using the multi-attribute similarity engine.
                Enter your target vehicle's specification to retrieve the closest matching market listings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodological Scope Notice
    st.markdown(
        """
        <div class="info-card" style="border-left: 4px solid #6366f1; margin-bottom: 1.2rem;">
            <div style="font-weight: 600; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 0.4rem;">
                📌 Comparable Search — Scope &amp; Similarity Notice
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
                The <strong>Similarity</strong> score measures multi-attribute specification distance between
                your entered vehicle and each candidate listing based on weighted feature alignment
                (Brand, Model, Year, Mileage, Engine CC, Fuel, Transmission, District, Condition).
                <br><br>
                Similarity does <strong>NOT</strong> represent prediction confidence, valuation accuracy,
                probability of correctness, or match certainty.
                All prices shown are <strong>advertised asking prices</strong> — they are not confirmed
                transaction or negotiated selling prices.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Target Vehicle Input Form
    _render_target_vehicle_form()


def _render_target_vehicle_form() -> None:
    """Renders the target vehicle input form and orchestrates the search workflow."""

    st.markdown(
        '<div class="section-title" style="font-size:1.3rem; margin-bottom:0.8rem;">'
        "🚗 Target Vehicle Specification"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form(key="comparable_search_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            category = st.selectbox(
                "Vehicle Category *",
                options=VEHICLE_CATEGORIES,
                index=0,
                help="Select the canonical vehicle category.",
                key="comp_category",
            )
            brand = st.text_input(
                "Brand / Make *",
                value="Toyota",
                placeholder="e.g. Toyota, Honda, Nissan",
                help="Vehicle manufacturer / brand.",
                key="comp_brand",
            )
            model = st.text_input(
                "Model *",
                value="Premio",
                placeholder="e.g. Premio, Fit, Vezel",
                help="Vehicle model name.",
                key="comp_model",
            )
            fuel_type = st.selectbox(
                "Fuel Type *",
                options=FUEL_TYPES,
                index=0,
                help="Primary fuel type.",
                key="comp_fuel",
            )

        with col2:
            manufacture_year = st.number_input(
                "Manufacture Year *",
                min_value=1950,
                max_value=CURRENT_YEAR,
                value=2016,
                step=1,
                help="Year the vehicle was manufactured.",
                key="comp_year",
            )
            mileage = st.number_input(
                "Odometer Mileage (km)",
                min_value=0,
                max_value=2_000_000,
                value=85_000,
                step=1_000,
                help="Odometer reading in kilometres. Leave 0 if unknown.",
                key="comp_mileage",
            )
            engine_cc = st.number_input(
                "Engine Capacity (cc)",
                min_value=0,
                max_value=25_000,
                value=1500,
                step=50,
                help="Engine displacement in cubic centimetres. Leave 0 if unknown.",
                key="comp_cc",
            )
            transmission = st.selectbox(
                "Transmission *",
                options=TRANSMISSIONS,
                index=0,
                help="Gear transmission type.",
                key="comp_transmission",
            )

        with col3:
            district = st.selectbox(
                "District",
                options=SRI_LANKA_DISTRICTS,
                index=SRI_LANKA_DISTRICTS.index("Colombo"),
                help="Sri Lankan administrative district.",
                key="comp_district",
            )
            condition = st.selectbox(
                "Condition *",
                options=CONDITIONS,
                index=0,
                help="Vehicle registration/condition status.",
                key="comp_condition",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                "<div style='font-size:0.82rem; color:#64748b; font-weight:600; "
                "text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;'>"
                "Result Controls</div>",
                unsafe_allow_html=True,
            )
            top_k = st.slider(
                "Maximum Comparables",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                help="Maximum number of comparable listings to retrieve from the database.",
                key="comp_top_k",
            )
            min_sim_display = st.slider(
                "Min. Similarity Display (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help=(
                    "UI display filter: hide results below this similarity threshold. "
                    "Does NOT affect the underlying similarity calculation."
                ),
                key="comp_min_sim",
            )
            sort_by = st.selectbox(
                "Sort By",
                options=[
                    "Similarity (highest first)",
                    "Asking Price (lowest first)",
                    "Asking Price (highest first)",
                    "Year (newest first)",
                ],
                index=0,
                key="comp_sort_by",
            )

        submitted = st.form_submit_button(
            "Find Comparable Vehicles",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        _run_comparable_search(
            category=category,
            brand=brand.strip(),
            model=model.strip(),
            manufacture_year=int(manufacture_year),
            mileage=float(mileage) if mileage > 0 else None,
            engine_cc=float(engine_cc) if engine_cc > 0 else None,
            fuel_type=fuel_type,
            transmission=transmission,
            district=district,
            condition=condition,
            top_k=int(top_k),
            min_sim_display=int(min_sim_display),
            sort_by=sort_by,
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_target_vehicle_input(query: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validates target vehicle input against domain constraints.
    
    The target vehicle is an in-memory input object only. It is never inserted
    or modified in the PostgreSQL database.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    cat = query.get("category")
    if not cat or cat not in VEHICLE_CATEGORIES:
        errors.append(f"Vehicle Category must be one of: {', '.join(VEHICLE_CATEGORIES)}.")

    brand = str(query.get("brand", "")).strip()
    if not brand:
        errors.append("Brand / Make cannot be empty.")

    model = str(query.get("model", "")).strip()
    if not model:
        errors.append("Model cannot be empty.")

    year = query.get("manufacture_year")
    try:
        y_int = int(year)  # type: ignore[arg-type]
        if y_int < 1950 or y_int > CURRENT_YEAR:
            errors.append(f"Manufacture Year must be between 1950 and {CURRENT_YEAR}.")
    except (ValueError, TypeError):
        errors.append(f"Manufacture Year must be a valid integer between 1950 and {CURRENT_YEAR}.")

    fuel = query.get("fuel_type")
    if fuel and fuel not in FUEL_TYPES:
        errors.append(f"Fuel Type must be one of: {', '.join(FUEL_TYPES)}.")

    trans = query.get("transmission")
    if trans and trans not in TRANSMISSIONS:
        errors.append(f"Transmission must be one of: {', '.join(TRANSMISSIONS)}.")

    dist = query.get("district")
    if dist and dist not in SRI_LANKA_DISTRICTS:
        errors.append("District must be a valid Sri Lankan district.")

    cond = query.get("condition")
    if cond and cond not in CONDITIONS:
        errors.append(f"Condition must be one of: {', '.join(CONDITIONS)}.")

    mileage = query.get("mileage")
    if mileage is not None:
        try:
            m_val = float(mileage)
            if m_val < 0:
                errors.append("Mileage cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Mileage must be a positive number.")

    engine_cc = query.get("engine_cc")
    if engine_cc is not None:
        try:
            cc_val = float(engine_cc)
            if cc_val < 0:
                errors.append("Engine CC cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Engine CC must be a positive number.")

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
def execute_comparable_search(
    query: Dict[str, Any],
    top_k: int = 10,
    match_category_strictly: bool = True,
    exclude_listing_id: Optional[str] = None,
    candidate_pool: Optional[pd.DataFrame] = None,
    engine: Optional[ComparableVehicleEngine] = None,
) -> List[ComparableVehicle]:
    """
    Executes comparable vehicle search using the existing ComparableVehicleEngine.

    Reuses existing parameters and methodology:
    - Multi-attribute weighted similarity scoring
    - Category strict matching (default: True)
    - ML-eligible listings only
    - Excludes private seller contact data

    Returns:
        List of ComparableVehicle objects sorted by similarity score descending.
    """
    if engine is None:
        engine = _get_comparable_engine()
    return engine.find_comparables(
        query=query,
        top_k=top_k,
        match_category_strictly=match_category_strictly,
        exclude_listing_id=exclude_listing_id,
        candidate_pool=candidate_pool,
    )


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

def _run_comparable_search(
    category: str,
    brand: str,
    model: str,
    manufacture_year: int,
    mileage: Optional[float],
    engine_cc: Optional[float],
    fuel_type: str,
    transmission: str,
    district: str,
    condition: str,
    top_k: int,
    min_sim_display: int,
    sort_by: str,
) -> None:
    """Validates input, calls the existing comparable engine, and renders results."""

    query: Dict[str, Any] = {
        "category": category,
        "brand": brand,
        "model": model,
        "manufacture_year": manufacture_year,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "district": district,
        "condition": condition,
    }
    if mileage is not None:
        query["mileage"] = mileage
    if engine_cc is not None:
        query["engine_cc"] = engine_cc

    is_valid, errors = validate_target_vehicle_input(query)
    if not is_valid:
        for e in errors:
            st.error(f"⚠️ {e}")
        return

    with st.spinner("Searching comparable vehicles in the database…"):
        try:
            comparables: List[ComparableVehicle] = execute_comparable_search(
                query=query,
                top_k=top_k,
                match_category_strictly=True,
                exclude_listing_id=None,
            )
        except Exception as exc:
            logger.error("Comparable engine error: %s", exc, exc_info=True)
            st.error(
                "⚠️ The comparable search engine encountered an error. "
                "Please check that the database is reachable and try again."
            )
            return

    if min_sim_display > 0:
        threshold = min_sim_display / 100.0
        comparables = [c for c in comparables if c.similarity_score >= threshold]

    comparables = _apply_sort(comparables, sort_by)

    _render_results(
        comparables=comparables,
        query=query,
        manufacture_year=manufacture_year,
        mileage=mileage,
        engine_cc=engine_cc,
        min_sim_display=min_sim_display,
        sort_by=sort_by,
    )


def _apply_sort(
    comparables: List[ComparableVehicle],
    sort_by: str,
) -> List[ComparableVehicle]:
    """Applies UI-level sort to comparable results. Does NOT affect similarity scores."""
    if sort_by == "Asking Price (lowest first)":
        return sorted(comparables, key=lambda c: c.asking_price)
    elif sort_by == "Asking Price (highest first)":
        return sorted(comparables, key=lambda c: c.asking_price, reverse=True)
    elif sort_by == "Year (newest first)":
        return sorted(
            comparables,
            key=lambda c: c.manufacture_year if c.manufacture_year else 0,
            reverse=True,
        )
    # Default: Similarity (highest first)
    return sorted(comparables, key=lambda c: c.similarity_score, reverse=True)


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _render_results(
    comparables: List[ComparableVehicle],
    query: Dict[str, Any],
    manufacture_year: int,
    mileage: Optional[float],
    engine_cc: Optional[float],
    min_sim_display: int,
    sort_by: str,
) -> None:
    """Renders the complete comparable results section."""

    st.divider()

    count = len(comparables)
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.8rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem;
                        font-weight:700; color:#a5b4fc;">
                🏷️ Comparable Vehicle Results
            </div>
            <div style="font-size:0.85rem; color:#64748b;">
                Eligible comparables: <strong style="color:#e2e8f0;">{count}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if count == 0:
        st.markdown(
            """
            <div class="info-card" style="border-left:4px solid #ef4444; text-align:center; padding:2rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🚫</div>
                <div style="font-size:1rem; color:#fca5a5; font-weight:600; margin-bottom:0.4rem;">
                    No comparable vehicles were found for the selected vehicle.
                </div>
                <div style="font-size:0.85rem; color:#94a3b8;">
                    Try expanding your search by adjusting the category, brand, or model,
                    or by lowering the minimum similarity display threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if count <= 2:
        st.info(
            f"⚠️ Limited comparable data is available ({count} result{'s' if count > 1 else ''}). "
            "Interpret the market summary with caution — statistical figures are less reliable "
            "with very few comparables."
        )

    _render_results_table(comparables)

    st.divider()

    _render_market_summary(comparables)

    st.divider()

    _render_visualizations(
        comparables=comparables,
        query=query,
        manufacture_year=manufacture_year,
        mileage=mileage,
    )

    _render_data_quality(comparables, query)


def _render_results_table(comparables: List[ComparableVehicle]) -> None:
    """Renders the comparable vehicles as a formatted Streamlit dataframe."""

    st.markdown(
        '<div class="section-title" style="font-size:1.1rem; margin-bottom:0.5rem;">'
        "📋 Comparable Listings"
        "</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for c in comparables:
        rows.append(
            {
                "Brand": _safe_val(c.brand),
                "Model": _safe_val(c.model),
                "Category": _safe_val(c.category),
                "Year": str(c.manufacture_year) if c.manufacture_year else "N/A",
                "Mileage": _fmt_km(c.mileage),
                "Engine CC": _fmt_cc(c.engine_cc),
                "Fuel": _safe_val(c.fuel_type),
                "Transmission": _safe_val(c.transmission),
                "District": _safe_val(c.district),
                "Condition": _safe_val(c.condition),
                "Asking Price (LKR)": _fmt_lkr(c.asking_price),
                "Similarity": _fmt_pct(c.similarity_percentage),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Market summary rendering
# ---------------------------------------------------------------------------

def _render_market_summary(comparables: List[ComparableVehicle]) -> None:
    """Renders the comparable market summary using create_comparable_market_summary."""

    st.markdown(
        '<div class="section-title" style="font-size:1.1rem; margin-bottom:0.5rem;">'
        "📊 Comparable Market Summary"
        "</div>",
        unsafe_allow_html=True,
    )

    summary = create_comparable_market_summary(comparables)

    if summary.comparable_count == 0:
        st.info("No comparable vehicles were found for the selected target.")
        return

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.metric("Comparable Count", f"{summary.comparable_count}")
    with col2:
        st.metric("Minimum Asking Price", _fmt_lkr(summary.min_asking_price))
    with col3:
        st.metric("Maximum Asking Price", _fmt_lkr(summary.max_asking_price))
    with col4:
        st.metric("Median Asking Price", _fmt_lkr(summary.median_asking_price))
    with col5:
        st.metric("Average Asking Price", _fmt_lkr(summary.average_asking_price))
    with col6:
        st.metric("Price Spread", _fmt_lkr(summary.price_spread))

    st.markdown(
        '<div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem; font-style:italic;">'
        "ℹ️ Summary statistics are based on advertised asking prices of the retrieved comparable"
        " vehicles. They are not verified transaction prices."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _render_visualizations(
    comparables: List[ComparableVehicle],
    query: Dict[str, Any],
    manufacture_year: int,
    mileage: Optional[float],
) -> None:
    """Renders comparison visualizations based on available comparable data."""

    st.markdown(
        '<div class="section-title" style="font-size:1.1rem; margin-bottom:0.5rem;">'
        "📈 Comparison Visualizations"
        "</div>",
        unsafe_allow_html=True,
    )

    count = len(comparables)
    records = [c.to_dict() for c in comparables]
    df = pd.DataFrame(records)

    target_brand = _safe_val(query.get("brand"), "Target Vehicle")
    target_model = _safe_val(query.get("model"), "")
    target_label = f"Target: {target_brand} {target_model}".strip()

    col_left, col_right = st.columns(2)

    # Chart 1: Comparable Asking Prices bar chart
    with col_left:
        st.markdown(
            "<div style='font-size:0.9rem; color:#94a3b8; margin-bottom:0.4rem;'>"
            "Comparable Advertised Asking Price (LKR)</div>",
            unsafe_allow_html=True,
        )
        if count >= 1 and "asking_price" in df.columns:
            labels = [f"{_safe_val(c.brand)} {_safe_val(c.model)}" for c in comparables]
            prices = [c.asking_price for c in comparables]
            sims = [c.similarity_percentage for c in comparables]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=prices,
                    marker_color=[
                        f"rgba(99,102,241,{max(0.35, s / 100.0):.2f})" for s in sims
                    ],
                    text=[_fmt_lkr(p) for p in prices],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Asking Price: %{y:,.0f} LKR<br><extra></extra>",
                    name="Comparable Asking Price",
                )
            )
            fig = _apply_dark_theme(fig, height=380)
            fig.update_layout(
                xaxis_title="Vehicle",
                yaxis_title="Advertised Asking Price (LKR)",
                xaxis_tickangle=-35,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No asking price data available for visualization.")

    # Chart 2: Year vs Asking Price scatter
    with col_right:
        st.markdown(
            "<div style='font-size:0.9rem; color:#94a3b8; margin-bottom:0.4rem;'>"
            "Manufacture Year vs Advertised Asking Price (LKR)</div>",
            unsafe_allow_html=True,
        )
        scatter_df = (
            df.dropna(subset=["manufacture_year", "asking_price"]) if not df.empty else df
        )
        if len(scatter_df) >= 1:
            valid_records = [
                r for r in records
                if pd.notna(r.get("manufacture_year")) and pd.notna(r.get("asking_price"))
            ]
            valid_comparables = [
                c for c in comparables
                if c.manufacture_year is not None and c.asking_price is not None
            ]
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=scatter_df["manufacture_year"].tolist(),
                    y=scatter_df["asking_price"].tolist(),
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=[c.similarity_percentage for c in valid_comparables],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Similarity %"),
                        opacity=0.85,
                    ),
                    text=[
                        f"{_safe_val(r.get('brand'))} {_safe_val(r.get('model'))}"
                        for r in valid_records
                    ],
                    textposition="top center",
                    textfont=dict(size=9, color="#94a3b8"),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Year: %{x}<br>"
                        "Asking Price: %{y:,.0f} LKR<br>"
                        "<extra></extra>"
                    ),
                    name="Comparables",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=[manufacture_year],
                    y=[None],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color="#f59e0b"),
                    name=target_label,
                    hovertemplate=(
                        f"<b>{target_label}</b><br>"
                        f"Year: {manufacture_year}<br>"
                        "<extra></extra>"
                    ),
                )
            )
            fig2 = _apply_dark_theme(fig2, height=380)
            fig2.update_layout(
                xaxis_title="Manufacture Year",
                yaxis_title="Advertised Asking Price (LKR)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient year/price data for this visualization.")

    # Chart 3: Mileage vs Asking Price (only if >= 2 comparables with mileage)
    if count >= 2 and mileage is not None:
        st.markdown(
            "<div style='font-size:0.9rem; color:#94a3b8; margin-bottom:0.4rem;'>"
            "Odometer Mileage vs Advertised Asking Price (LKR)</div>",
            unsafe_allow_html=True,
        )
        mileage_df = (
            df.dropna(subset=["mileage", "asking_price"]) if not df.empty else df
        )
        if len(mileage_df) >= 2:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=mileage_df["mileage"].tolist(),
                    y=mileage_df["asking_price"].tolist(),
                    mode="markers",
                    marker=dict(size=10, color="#6366f1", opacity=0.8),
                    hovertemplate=(
                        "Mileage: %{x:,.0f} km<br>"
                        "Asking Price: %{y:,.0f} LKR<br>"
                        "<extra></extra>"
                    ),
                    name="Comparables",
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=[mileage],
                    y=[None],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color="#f59e0b"),
                    name=target_label,
                    hovertemplate=(
                        f"<b>{target_label}</b><br>"
                        f"Mileage: {mileage:,.0f} km<br>"
                        "<extra></extra>"
                    ),
                )
            )
            fig3 = _apply_dark_theme(fig3, height=360)
            fig3.update_layout(
                xaxis_title="Odometer Mileage (km)",
                yaxis_title="Advertised Asking Price (LKR)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            col_m, _ = st.columns([2, 1])
            with col_m:
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Insufficient mileage/price data for this visualization.")

    # Chart 4: Asking Price Distribution (only if >= 3 comparables)
    if count >= 3:
        st.markdown(
            "<div style='font-size:0.9rem; color:#94a3b8; margin-bottom:0.4rem;'>"
            "Comparable Advertised Asking Price Distribution</div>",
            unsafe_allow_html=True,
        )
        price_series = [c.asking_price for c in comparables if c.asking_price > 0]
        if len(price_series) >= 3:
            fig4 = go.Figure()
            fig4.add_trace(
                go.Histogram(
                    x=price_series,
                    nbinsx=min(len(price_series), 10),
                    marker_color="rgba(99,102,241,0.65)",
                    marker_line=dict(color="rgba(99,102,241,1.0)", width=1),
                    name="Asking Price Distribution",
                    hovertemplate="Price Range: %{x:,.0f} LKR<br>Count: %{y}<extra></extra>",
                )
            )
            fig4 = _apply_dark_theme(fig4, height=340)
            fig4.update_layout(
                xaxis_title="Advertised Asking Price (LKR)",
                yaxis_title="Count",
                showlegend=False,
            )
            col_d, _ = st.columns([2, 1])
            with col_d:
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Insufficient comparable price data for distribution chart.")

    st.markdown(
        '<div style="font-size:0.79rem; color:#475569; margin-top:0.8rem; font-style:italic;">'
        "Visualizations show observed descriptive associations in the retrieved comparable listings."
        " They do not imply mathematical causality and do not represent prediction accuracy."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data quality summary
# ---------------------------------------------------------------------------

def _render_data_quality(
    comparables: List[ComparableVehicle],
    query: Dict[str, Any],
) -> None:
    """Renders a lightweight data quality notice for the comparable search results."""

    with st.expander("🔬 Data Quality & Search Scope", expanded=False):
        count = len(comparables)

        cols_provided = [
            k for k, v in query.items()
            if v is not None and str(v).strip() not in ("", "nan")
        ]
        total_attrs = 9  # brand, model, year, mileage, engine_cc, fuel, transmission, district, condition

        st.markdown(
            f"""
            <div style="font-size:0.88rem; color:#94a3b8; line-height:1.7;">
                <b style="color:#e2e8f0;">Comparable Count:</b> {count}<br>
                <b style="color:#e2e8f0;">Input Attributes Provided:</b>
                    {len(cols_provided)} of {total_attrs}<br>
                <b style="color:#e2e8f0;">Category Strict Matching:</b> Enabled (only same-category listings considered)<br>
                <b style="color:#e2e8f0;">Similarity Basis:</b> Multi-attribute weighted feature alignment
                    (Brand 25%, Model 25%, Year 14%, Mileage 10%, Engine CC 8%,
                    Transmission 6%, Fuel 4%, District 4%, Condition 4%)<br>
                <b style="color:#e2e8f0;">Price Type:</b> Advertised Asking Price (LKR)
                    — not verified transaction prices
            </div>
            """,
            unsafe_allow_html=True,
        )

        if count < 3:
            st.warning(
                "⚠️ Fewer than 3 comparables were retrieved. "
                "Statistical summary figures (median, average) are less reliable with very limited data. "
                "Consider broadening the search parameters."
            )
