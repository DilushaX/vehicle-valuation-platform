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
import streamlit as st

from eda.dataset import EDADatasetLoader

logger = logging.getLogger(__name__)


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

    # ── Dashboard Structure Placeholders ─────────────────────────────────
    filters_container = st.container()
    overview_container = st.container()
    category_container = st.container()
    brand_model_container = st.container()
    price_container = st.container()
    characteristics_container = st.container()
    geographic_container = st.container()
    trends_container = st.container()
    quality_scope_container = st.container()

    with filters_container:
        st.markdown("<div class='section-title'>🔍 Global Market Filters</div>", unsafe_allow_html=True)
        st.caption("Filters apply dynamically across all analytical views.")
