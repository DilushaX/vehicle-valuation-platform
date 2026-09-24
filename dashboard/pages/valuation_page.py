"""
Vehicle Valuation Page (Phase 10.2).

Displays a structured input form, calls POST /api/valuation/predict,
then renders all result sections:
  1. Price Estimate Hero + Range
  2. SHAP Factor Attribution
  3. Data Quality
  4. Comparable Listings
  5. Market Summary
  6. Audit & Reproducibility
  7. Limitations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import APIClientError, ValuationAPIClient

# ---------------------------------------------------------------------------
# Domain constants (mirroring canonical values in cleaners.py)
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
# Helpers
# ---------------------------------------------------------------------------

def _fmt_lkr(amount: float) -> str:
    """Format a LKR value as '12,500,000'."""
    return f"{amount:,.0f}"


def _direction_color(direction: str) -> str:
    return "#34d399" if direction == "positive" else "#f87171"


def _build_payload(
    category: str,
    brand: str,
    model_name: str,
    manufacture_year: int,
    mileage: Optional[float],
    engine_cc: Optional[float],
    fuel_type: str,
    transmission: str,
    district: str,
    condition: str,
    top_k_factors: int,
    top_k_comparables: int,
    percentile_lower: int,
    percentile_upper: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "category": category,
        "brand": brand.strip(),
        "model": model_name.strip(),
        "manufacture_year": manufacture_year,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "district": district,
        "condition": condition,
        "top_k_factors": top_k_factors,
        "top_k_comparables": top_k_comparables,
        "percentile_lower": percentile_lower,
        "percentile_upper": percentile_upper,
    }
    if mileage is not None and mileage >= 0:
        payload["mileage"] = mileage
    if engine_cc is not None and engine_cc > 0:
        payload["engine_cc"] = engine_cc
    return payload


# ---------------------------------------------------------------------------
# Result section renderers
# ---------------------------------------------------------------------------

def _render_price_hero(result: Dict[str, Any]) -> None:
    price = result.get("estimated_asking_price_lkr", 0)
    rng = result.get("prediction_range_lkr", {})
    lower = rng.get("lower", 0)
    upper = rng.get("upper", 0)
    p_lower = rng.get("percentile_lower", 10)
    p_upper = rng.get("percentile_upper", 90)
    currency = result.get("currency", "LKR")

    st.markdown(
        f"""
        <div class="price-hero">
            <div class="label">Estimated Market Asking Price</div>
            <h1>{currency} {_fmt_lkr(price)}</h1>
            <div class="range-label">
                Indicative Range ({p_lower}th–{p_upper}th percentile):
                <strong>{currency} {_fmt_lkr(lower)}</strong>
                —
                <strong>{currency} {_fmt_lkr(upper)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Range chart using Plotly
    spread = rng.get("spread", upper - lower)
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=price,
            number={"prefix": "LKR ", "valueformat": ",.0f", "font": {"size": 28, "color": "#a5b4fc"}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [max(0, lower * 0.8), upper * 1.2], "visible": False},
                "bar": {"color": "#7c3aed", "thickness": 0.6},
                "steps": [
                    {"range": [lower, upper], "color": "rgba(109,40,217,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#10b981", "width": 3},
                    "thickness": 0.9,
                    "value": price,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=90,
        margin={"t": 0, "b": 0, "l": 10, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Range metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(f"Lower ({p_lower}th pct)", f"LKR {_fmt_lkr(lower)}")
    with c2:
        st.metric("Estimate", f"LKR {_fmt_lkr(price)}")
    with c3:
        st.metric(f"Upper ({p_upper}th pct)", f"LKR {_fmt_lkr(upper)}")
    with c4:
        st.metric("Spread", f"LKR {_fmt_lkr(spread)}")


def _render_explanation(explanation: List[Dict[str, Any]]) -> None:
    st.markdown(
        "<div class='section-title'>📊 SHAP Factor Attribution</div>",
        unsafe_allow_html=True,
    )
    if not explanation:
        st.info("No explanation factors were returned by the model.")
        return

    # Plotly horizontal bar chart
    features = [f.get("feature", "") for f in explanation]
    contributions = [f.get("contribution", 0) for f in explanation]
    colors = ["#34d399" if c >= 0 else "#f87171" for c in contributions]
    values = [f.get("value", "") for f in explanation]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=[f"{feat} ({val})" for feat, val in zip(features, values)],
            orientation="h",
            marker_color=colors,
            text=[f"{'+' if c >= 0 else ''}{c:,.0f}" for c in contributions],
            textposition="outside",
            textfont={"color": "#e2e8f0", "size": 11},
            hovertemplate="<b>%{y}</b><br>Contribution: LKR %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(280, len(explanation) * 48 + 80),
        xaxis={
            "title": "Contribution to Price (LKR)",
            "color": "#94a3b8",
            "gridcolor": "rgba(99,102,241,0.1)",
            "zerolinecolor": "rgba(99,102,241,0.4)",
        },
        yaxis={"color": "#94a3b8", "autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.4)",
        margin={"t": 10, "b": 40, "l": 20, "r": 60},
        font={"family": "Inter", "color": "#e2e8f0"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Detailed factor table
    with st.expander("📋 Detailed Factor Breakdown", expanded=False):
        for f in explanation:
            direction = f.get("direction", "positive")
            contrib = f.get("contribution", 0)
            color = _direction_color(direction)
            sign = "+" if contrib >= 0 else ""
            st.markdown(
                f"""
                <div class="factor-row">
                    <span class="factor-label">{f.get('feature', '—')}</span>
                    <span class="factor-value">{f.get('value', '—')}</span>
                    <span style="color:{color}; font-weight:600; min-width:120px;">
                        {sign}{_fmt_lkr(contrib)} LKR
                    </span>
                    <span style="font-size:0.82rem; color:#64748b; flex:1;">
                        {f.get('description', '')}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_data_quality(dq: Dict[str, Any]) -> None:
    st.markdown(
        "<div class='section-title'>✅ Data Quality</div>",
        unsafe_allow_html=True,
    )

    status = dq.get("status", "UNKNOWN")
    provided = dq.get("provided_features", 0)
    expected = dq.get("expected_features", 0)
    missing = dq.get("missing_features", [])
    comp_count = dq.get("comparable_count", 0)

    badge_class = "badge-green" if status == "COMPLETE" else "badge-yellow"
    badge_label = "✅ COMPLETE" if status == "COMPLETE" else "⚠️ PARTIAL"

    st.markdown(
        f"<span class='badge {badge_class}'>{badge_label}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Provided Features", f"{provided} / {expected}")
    with c2:
        st.metric("Comparable Listings Found", comp_count)
    with c3:
        st.metric("Missing Features", len(missing) if missing else 0)

    if missing:
        st.warning(f"⚠️ Missing features: **{', '.join(missing)}**. "
                   "The estimate may be less accurate.")


def _render_comparables(comparables: List[Dict[str, Any]]) -> None:
    st.markdown(
        "<div class='section-title'>🔗 Comparable Market Listings</div>",
        unsafe_allow_html=True,
    )

    if not comparables:
        st.info(
            "No comparable listings were found in the database for this vehicle specification. "
            "This may occur for rare categories or configurations."
        )
        return

    import pandas as pd

    rows = []
    for c in comparables:
        rows.append(
            {
                "Brand": c.get("brand", "—"),
                "Model": c.get("model", "—"),
                "Year": c.get("manufacture_year") or "—",
                "Fuel": c.get("fuel_type") or "—",
                "Trans.": c.get("transmission") or "—",
                "Mileage (km)": f"{c.get('mileage', 0):,.0f}" if c.get("mileage") else "—",
                "Engine (cc)": f"{c.get('engine_cc', 0):,.0f}" if c.get("engine_cc") else "—",
                "District": c.get("district") or "—",
                "Condition": c.get("condition") or "—",
                "Asking Price (LKR)": f"{c.get('asking_price', 0):,.0f}",
                "Similarity": f"{c.get('similarity_percentage', 0):.1f}%",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def _render_market_summary(cms: Optional[Dict[str, Any]]) -> None:
    if not cms:
        return

    st.markdown(
        "<div class='section-title'>📈 Comparable Market Summary</div>",
        unsafe_allow_html=True,
    )

    count = cms.get("comparable_count", 0)
    if count == 0:
        st.info("No comparables available for market summary.")
        return

    min_p = cms.get("min_asking_price")
    max_p = cms.get("max_asking_price")
    median_p = cms.get("median_asking_price")
    avg_p = cms.get("average_asking_price")
    spread = cms.get("price_spread")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Comparables", count)
    with c2:
        st.metric("Min Price (LKR)", f"{_fmt_lkr(min_p)}" if min_p else "—")
    with c3:
        st.metric("Median Price (LKR)", f"{_fmt_lkr(median_p)}" if median_p else "—")
    with c4:
        st.metric("Average Price (LKR)", f"{_fmt_lkr(avg_p)}" if avg_p else "—")
    with c5:
        st.metric("Max Price (LKR)", f"{_fmt_lkr(max_p)}" if max_p else "—")

    # Bar chart for price distribution
    if min_p and max_p and median_p and avg_p:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Min", "Average", "Median", "Max"],
                y=[min_p, avg_p, median_p, max_p],
                marker_color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"],
                text=[f"LKR {_fmt_lkr(v)}" for v in [min_p, avg_p, median_p, max_p]],
                textposition="outside",
                textfont={"color": "#e2e8f0", "size": 11},
                hovertemplate="<b>%{x}</b>: LKR %{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.4)",
            yaxis={
                "title": "Asking Price (LKR)",
                "color": "#94a3b8",
                "gridcolor": "rgba(99,102,241,0.1)",
            },
            xaxis={"color": "#94a3b8"},
            margin={"t": 20, "b": 10, "l": 10, "r": 10},
            font={"family": "Inter", "color": "#e2e8f0"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_audit(audit: Optional[Dict[str, Any]], repro: Optional[Dict[str, Any]]) -> None:
    if not audit and not repro:
        return

    with st.expander("🛡️ Audit & Reproducibility", expanded=False):
        if audit:
            st.markdown(
                "<div class='section-title' style='font-size:1rem;'>Valuation Audit</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            audit_fields = [
                ("Model Name", audit.get("model_name", "—")),
                ("Model Version", audit.get("model_version") or "—"),
                ("Model Status", audit.get("model_status", "—")),
                ("Target Variable", audit.get("target", "—")),
                ("Generated At (UTC)", audit.get("generated_at", "—")),
                ("Feature Schema Version", audit.get("feature_schema_version", "—")),
                ("Valuation Method", audit.get("valuation_method", "—")),
                ("Comparable Method", audit.get("comparable_method", "—")),
                ("Range Method", audit.get("prediction_range_method", "—")),
            ]
            for i, (label, value) in enumerate(audit_fields):
                with cols[i % 2]:
                    st.markdown(
                        f"<div style='margin-bottom:0.5rem;'>"
                        f"<span style='color:#64748b; font-size:0.8rem;'>{label}</span><br>"
                        f"<span style='color:#e2e8f0; font-size:0.9rem; font-weight:500;'>{value}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        if repro:
            st.divider()
            st.markdown(
                "<div class='section-title' style='font-size:1rem;'>Reproducibility</div>",
                unsafe_allow_html=True,
            )
            algo = repro.get("algorithm", "SHA-256")
            fp = repro.get("fingerprint", "—")
            st.markdown(
                f"<div style='color:#64748b; font-size:0.8rem;'>Algorithm: {algo}</div>"
                f"<code style='color:#a5b4fc; font-size:0.82rem; word-break:break-all;'>{fp}</code>",
                unsafe_allow_html=True,
            )


def _render_model_meta(model_meta: Dict[str, Any]) -> None:
    with st.expander("🤖 Model Metadata", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model", model_meta.get("name", "—"))
        with c2:
            st.metric("Training Records", model_meta.get("training_records", "—"))
        with c3:
            st.metric("Test Samples", model_meta.get("test_samples", "—"))

        status = model_meta.get("status", "")
        badge_class = "badge-yellow" if "EXPERIMENTAL" in status.upper() else "badge-green"
        st.markdown(
            f"<span class='badge {badge_class}'>{status}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#64748b; font-size:0.8rem; margin-top:0.5rem;'>"
            f"Target: <strong style='color:#a5b4fc;'>{model_meta.get('target_variable', '—')}</strong>"
            f" | Transform: <strong style='color:#a5b4fc;'>{model_meta.get('target_transform', '—')}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_limitations(limitations: List[str]) -> None:
    with st.expander("⚠️ Limitations & Disclaimers", expanded=False):
        st.markdown(
            "<div class='disclaimer-box'>",
            unsafe_allow_html=True,
        )
        for lim in limitations:
            st.markdown(
                f"<p>• {lim}</p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_valuation_page(api_url: str = "http://localhost:8000") -> None:
    """Renders the full Vehicle Valuation page."""

    st.markdown(
        """
        <div style="padding-bottom:0.5rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                        font-weight:700; color:#ffffff;">
                🔍 Vehicle Valuation
            </div>
            <div style="font-size:0.92rem; color:#64748b; margin-top:0.2rem;">
                Enter vehicle details to get an ML-powered asking price estimate with
                SHAP attribution, comparable listings, and full audit trail.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    client = ValuationAPIClient(base_url=api_url)

    # ── Input Form ────────────────────────────────────────────────────────
    with st.form("valuation_form", clear_on_submit=False):
        st.markdown(
            "<div class='section-title'>🚗 Vehicle Details</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            category = st.selectbox(
                "Vehicle Category *",
                options=VEHICLE_CATEGORIES,
                index=0,
                help="Select the canonical vehicle category",
            )

        with col2:
            brand = st.text_input(
                "Brand / Make *",
                placeholder="e.g. Toyota, Honda, Nissan",
                help="Vehicle manufacturer brand",
            )

        with col3:
            model_name = st.text_input(
                "Model *",
                placeholder="e.g. Premio, Vezel, Fit",
                help="Vehicle model name",
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            manufacture_year = st.number_input(
                "Manufacture Year *",
                min_value=1950,
                max_value=CURRENT_YEAR,
                value=2015,
                step=1,
                help="Year the vehicle was manufactured",
            )

        with col5:
            fuel_type = st.selectbox(
                "Fuel Type *",
                options=FUEL_TYPES,
                index=0,
                help="Vehicle fuel type",
            )

        with col6:
            transmission = st.selectbox(
                "Transmission *",
                options=TRANSMISSIONS,
                index=0,
                help="Transmission type",
            )

        col7, col8, col9 = st.columns(3)

        with col7:
            district = st.selectbox(
                "District *",
                options=SRI_LANKA_DISTRICTS,
                index=SRI_LANKA_DISTRICTS.index("Colombo"),
                help="Sri Lankan administrative district",
            )

        with col8:
            condition = st.selectbox(
                "Condition *",
                options=CONDITIONS,
                index=0,
                help="Vehicle registration / usage condition",
            )

        with col9:
            mileage = st.number_input(
                "Mileage (km)",
                min_value=0,
                max_value=2_000_000,
                value=80_000,
                step=1_000,
                help="Odometer reading in kilometres (leave 0 to omit)",
            )

        col10, col11 = st.columns(2)
        with col10:
            engine_cc = st.number_input(
                "Engine Capacity (cc)",
                min_value=0,
                max_value=25_000,
                value=1500,
                step=50,
                help="Engine displacement in cubic centimetres (leave 0 to omit)",
            )

        # ── Advanced options ──────────────────────────────────────────────
        with st.expander("⚙️ Advanced Options", expanded=False):
            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
            with adv_col1:
                top_k_factors = st.slider(
                    "Top SHAP Factors",
                    min_value=1,
                    max_value=20,
                    value=7,
                    help="Number of SHAP explanation factors to display",
                )
            with adv_col2:
                top_k_comparables = st.slider(
                    "Comparable Listings",
                    min_value=0,
                    max_value=20,
                    value=5,
                    help="Number of comparable market listings to retrieve",
                )
            with adv_col3:
                percentile_lower = st.slider(
                    "Lower Percentile",
                    min_value=1,
                    max_value=49,
                    value=10,
                    help="Lower bound percentile for indicative price range",
                )
            with adv_col4:
                percentile_upper = st.slider(
                    "Upper Percentile",
                    min_value=51,
                    max_value=99,
                    value=90,
                    help="Upper bound percentile for indicative price range",
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Get Valuation Estimate",
            use_container_width=True,
        )

    # ── Form submission & API call ────────────────────────────────────────
    if submitted:
        # Client-side validation
        errors: List[str] = []
        if not brand.strip():
            errors.append("Brand / Make is required.")
        if not model_name.strip():
            errors.append("Model is required.")
        if percentile_lower >= percentile_upper:
            errors.append(
                f"Lower percentile ({percentile_lower}) must be less than "
                f"upper percentile ({percentile_upper})."
            )

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            return

        payload = _build_payload(
            category=category,
            brand=brand,
            model_name=model_name,
            manufacture_year=int(manufacture_year),
            mileage=float(mileage) if mileage > 0 else None,
            engine_cc=float(engine_cc) if engine_cc > 0 else None,
            fuel_type=fuel_type,
            transmission=transmission,
            district=district,
            condition=condition,
            top_k_factors=top_k_factors,
            top_k_comparables=top_k_comparables,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
        )

        with st.spinner("🤖 Computing valuation — contacting API…"):
            try:
                result = client.predict(payload)
            except APIClientError as exc:
                st.error(f"❌ Valuation failed: {exc.user_message}")
                if exc.status_code:
                    st.caption(f"HTTP status: {exc.status_code}")
                return
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                return

        st.success("✅ Valuation complete!")
        st.divider()

        # ── Section 1: Price Estimate ─────────────────────────────────────
        _render_price_hero(result)

        st.divider()

        # ── Section 2 & 3: Explanation + Data Quality (side by side) ─────
        exp_col, dq_col = st.columns([3, 1])

        with exp_col:
            _render_explanation(result.get("explanation", []))

        with dq_col:
            _render_data_quality(result.get("data_quality", {}))

        st.divider()

        # ── Section 4: Comparable Listings ───────────────────────────────
        _render_comparables(result.get("comparables", []))

        st.divider()

        # ── Section 5: Market Summary ─────────────────────────────────────
        _render_market_summary(result.get("comparable_market_summary"))

        st.divider()

        # ── Sections 6, 7, 8: Audit, Model Meta, Limitations ─────────────
        _render_model_meta(result.get("model", {}))
        _render_audit(result.get("audit"), result.get("reproducibility"))
        _render_limitations(result.get("limitations", []))

        st.divider()

        # ── Raw JSON (debug) ──────────────────────────────────────────────
        with st.expander("🔎 Raw API Response (JSON)", expanded=False):
            st.json(result)
