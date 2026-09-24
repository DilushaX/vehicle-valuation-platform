"""
Overview page — platform introduction and API status at a glance.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIClientError, ValuationAPIClient


def render_overview_page(api_url: str = "http://localhost:8000") -> None:
    """Renders the Overview / home page."""

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1.5rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:3rem;
                        font-weight:700; color:#ffffff; line-height:1.15;
                        text-shadow: 0 0 40px rgba(109,40,217,0.6);">
                🚗 Sri Lankan Vehicle<br>Valuation Platform
            </div>
            <div style="font-size:1.15rem; color:#94a3b8; margin-top:0.9rem; max-width:600px;
                        margin-left:auto; margin-right:auto; line-height:1.6;">
                Explainable machine learning valuations for the Sri Lankan used vehicle market,
                powered by <strong style="color:#a5b4fc;">RandomForest</strong> and
                <strong style="color:#a5b4fc;">Tree SHAP</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Feature highlights ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">🤖</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    ML Asking-Price Estimate
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    RandomForest regressor trained on verified Sri Lankan
                    listing data. Predicts market asking prices in LKR.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">📊</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    Tree SHAP Attribution
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    Feature-level contribution factors explain why the
                    model produced each estimate — fully transparent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">🔗</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    Comparable Listings
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    Multi-attribute similarity search surfaces real market
                    listings comparable to the vehicle being valued.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">📐</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    Indicative Price Range
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    Ensemble tree dispersion provides a lower/upper
                    indicative band around the central estimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">🛡️</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    Audit & Reproducibility
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    Every valuation carries a SHA-256 reproducibility
                    fingerprint and full methodology audit trail.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem;">✅</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                            font-weight:600; color:#a5b4fc; margin:0.5rem 0;">
                    Data Quality Indicator
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    Reports feature completeness and flags partial inputs
                    so users understand the confidence context.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── API status section ────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>⚡ API Status</div>",
        unsafe_allow_html=True,
    )

    client = ValuationAPIClient(base_url=api_url)

    c1, c2, c3 = st.columns(3)

    # Health check
    with c1:
        try:
            resp = client.health()
            st.metric("API Liveness", "✅ Online", help="GET /health")
        except APIClientError as e:
            st.metric("API Liveness", "❌ Offline", help=str(e))

    # Readiness check
    with c2:
        try:
            resp = client.ready()
            st.metric("Model Readiness", "✅ Ready", help="GET /ready")
        except APIClientError as e:
            code = e.status_code
            label = "⚠️ Not Ready" if code == 503 else "❌ Error"
            st.metric("Model Readiness", label, help=str(e))

    # API endpoint
    with c3:
        st.metric("API Endpoint", api_url)

    st.divider()

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="disclaimer-box">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.9rem;
                        font-weight:600; color:#f87171; margin-bottom:0.5rem;">
                ⚠️ Important Disclaimers
            </div>
            <p>• This platform estimates <strong>advertised asking prices</strong> observed on Riyasewana.
               It does <strong>not</strong> predict verified transaction or final selling prices.</p>
            <p>• The valuation model is an <strong>experimental research benchmark</strong> trained on 113
               verified ML-eligible records. Further data collection is required for production-grade claims.</p>
            <p>• Actual vehicle value may differ due to physical condition, accident history, mechanical state,
               battery/engine health, and registration documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;'>"
        "<span style='color:#64748b; font-size:0.82rem;'>"
        "Use the sidebar to navigate to <strong style='color:#a5b4fc;'>🔍 Vehicle Valuation</strong> "
        "to get a valuation estimate."
        "</span></div>",
        unsafe_allow_html=True,
    )
