"""
Vehicle Valuation Platform — Streamlit Dashboard (Phase 10.2).

Entry point: ``streamlit run dashboard/app.py``

Sections
--------
- 🏠 Overview          : Platform introduction and API status
- 🔍 Vehicle Valuation : Interactive valuation form + full result display
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when running as:
#   streamlit run dashboard/app.py
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard.api_client import APIClientError, ValuationAPIClient
from dashboard.pages.valuation_page import render_valuation_page
from dashboard.pages.overview_page import render_overview_page

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lankan Vehicle Valuation Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Vehicle Market Intelligence & Valuation Platform**\n\n"
            "Explainable ML valuations for Sri Lankan used vehicles. "
            "Powered by RandomForest + Tree SHAP.\n\n"
            "_Asking-price estimates only. Not a verified transaction price._"
        )
    },
)

# ---------------------------------------------------------------------------
# Global custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base & typography ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit header decoration ────────────── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Main background ─────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0b0c1e 0%, #111827 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* ── Metric cards ─────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(30,41,59,0.9));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    /* ── Section headings ─────────────────────────────────────── */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Price hero card ─────────────────────────────────────── */
    .price-hero {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #7c3aed 100%);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(109,40,217,0.4);
        margin-bottom: 1.2rem;
    }
    .price-hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .price-hero .label {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.75);
        margin-bottom: 0.3rem;
    }
    .price-hero .range-label {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.6rem;
    }

    /* ── Info cards ──────────────────────────────────────────── */
    .info-card {
        background: linear-gradient(145deg, rgba(17,24,39,0.95), rgba(30,41,59,0.9));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    }

    /* ── Factor bar ──────────────────────────────────────────── */
    .factor-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.7rem;
        font-size: 0.9rem;
    }
    .factor-label { color: #94a3b8; min-width: 160px; }
    .factor-value { color: #e2e8f0; font-weight: 500; min-width: 90px; }
    .factor-bar-outer {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .factor-bar-pos { background: linear-gradient(90deg, #10b981, #34d399); height: 100%; border-radius: 4px; }
    .factor-bar-neg { background: linear-gradient(90deg, #ef4444, #f87171); height: 100%; border-radius: 4px; }
    .factor-contrib { font-size: 0.85rem; min-width: 90px; text-align: right; }

    /* ── Comparable table ────────────────────────────────────── */
    .comp-table th {
        background: rgba(99,102,241,0.15) !important;
        color: #a5b4fc !important;
    }
    .comp-table td {
        color: #cbd5e1 !important;
    }

    /* ── Disclaimer box ──────────────────────────────────────── */
    .disclaimer-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
    }
    .disclaimer-box p {
        color: #fca5a5;
        font-size: 0.85rem;
        margin: 0.25rem 0;
    }

    /* ── Sidebar nav pills ───────────────────────────────────── */
    .nav-pill {
        display: block;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.3rem;
        cursor: pointer;
        transition: background 0.2s ease;
        color: #cbd5e1;
        text-decoration: none;
        font-size: 0.92rem;
    }
    .nav-pill:hover     { background: rgba(99,102,241,0.2); color: #fff; }
    .nav-pill.active    { background: rgba(99,102,241,0.35); color: #a5b4fc; font-weight: 600; }

    /* ── Dividers ────────────────────────────────────────────── */
    hr { border-color: rgba(99,102,241,0.15) !important; }

    /* ── Streamlit buttons ───────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.8rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(109,40,217,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        box-shadow: 0 6px 24px rgba(109,40,217,0.5);
        transform: translateY(-1px);
    }

    /* ── Input widgets ───────────────────────────────────────── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(17,24,39,0.8) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Expander ────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(30,41,59,0.7) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
    }

    /* ── Status badges ───────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }
    .badge-yellow { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
    .badge-red { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
    .badge-blue { background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem;
                        font-weight: 700; color: #a5b4fc; line-height: 1.3;">
                🚗 Vehicle<br>Valuation Platform
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.3rem;">
                Sri Lankan Market Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        options=["🏠 Overview", "🔍 Vehicle Valuation"],
        label_visibility="collapsed",
    )

    st.divider()

    # API base URL configuration
    st.markdown(
        "<div style='font-size:0.8rem; color:#64748b; font-weight:600;"
        " letter-spacing:0.05em; text-transform:uppercase;'>API Connection</div>",
        unsafe_allow_html=True,
    )
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        label_visibility="collapsed",
        placeholder="http://localhost:8000",
        key="api_base_url",
    )

    # Quick API status indicator
    client = ValuationAPIClient(base_url=api_url)
    try:
        health_resp = client.health()
        api_status = health_resp.get("status", "ok")
        st.markdown(
            f"<span class='badge badge-green'>● API Online</span>",
            unsafe_allow_html=True,
        )
    except APIClientError:
        st.markdown(
            "<span class='badge badge-red'>● API Offline</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem; color:#475569;'>"
        "Run the FastAPI backend first:<br>"
        "<code style='color:#7c3aed;'>uvicorn api.main:app --port 8000</code>"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    render_overview_page(api_url=api_url)
else:
    render_valuation_page(api_url=api_url)
