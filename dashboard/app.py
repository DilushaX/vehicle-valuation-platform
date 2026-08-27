"""
Sri Lankan Vehicle Market Intelligence & ML Valuation Platform
Interactive Streamlit Multi-Tab Dashboard
"""
import sys
from pathlib import Path
# Ensure project root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import settings
from database.connection import init_db, SessionLocal
from database.models import Listing, ScrapeRun, ModelVersion, DataQualityRecord
from analytics.market.market_analytics import MarketAnalyticsEngine
from analytics.trends.trend_engine import MarketTrendEngine
from analytics.comparables.comparable_engine import ComparableEngine
from ml.prediction.predictor import VehicleValuationPredictor
from ml.explainability.explainer import ValuationExplainer
from ml.negotiation.insights import NegotiationInsightEngine
from data_pipeline.pipeline_runner import PipelineRunner

# Initialize Database Schema
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="Sri Lankan Vehicle Market Intelligence & ML Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin: 15px 0;
    }
    .badge-within {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-below {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-above {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper DB Session
db = SessionLocal()
analytics_engine = MarketAnalyticsEngine(db)
trend_engine = MarketTrendEngine(db)
comparable_engine = ComparableEngine(db)
predictor = VehicleValuationPredictor(db)
explainer = ValuationExplainer()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/isometric/100/car.png", width=70)
st.sidebar.title("Market Filters")

selected_category = st.sidebar.selectbox(
    "Vehicle Category",
    options=["All"] + [c.capitalize() for c in settings.SUPPORTED_CATEGORIES],
    index=0
)
filter_cat = None if selected_category == "All" else selected_category

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Actions")
if st.sidebar.button("🔄 Refresh Market Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Sri Lankan Vehicle Intelligence v1.0.0\nData Source: Riyasewana & Market Observations")

# Header Section
st.markdown('<div class="main-header">🇱🇰 Sri Lankan Vehicle Market Intelligence & ML Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated data collection, market trends, comparable vehicle discovery, and explainable ML asking value estimation.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
  ⚠️ <strong>Important Notice:</strong> The system estimates market asking values based on observed vehicle listings. It does not claim to predict actual transaction prices, negotiated prices, or guaranteed vehicle values.
</div>
""", unsafe_allow_html=True)

# 10 Main Tabs
tabs = st.tabs([
    "1. Overview",
    "2. Vehicle Categories",
    "3. Market Analytics",
    "4. Market Trends",
    "5. Vehicle Analysis",
    "6. Comparable Vehicles",
    "7. AI Valuation",
    "8. Asking Price Assessment",
    "9. Data Quality",
    "10. System Monitoring"
])

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tabs[0]:
    st.subheader("Market Summary & High-Level KPIs")
    overview = analytics_engine.get_market_overview(category=filter_cat)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Listings", f"{overview['total_listings']:,}")
    with col2:
        st.metric("Active Listings", f"{overview['active_listings']:,}")
    with col3:
        st.metric("Valid Clean Records", f"{overview['valid_listings']:,}")
    with col4:
        st.metric("Median Asking Price", f"Rs. {overview['median_price']:,.0f}")
    with col5:
        st.metric("Average Asking Price", f"Rs. {overview['avg_price']:,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Most Listed Brands")
        brands_data = analytics_engine.get_brand_analysis(category=filter_cat, top_n=7)
        if brands_data:
            df_b = pd.DataFrame(brands_data)
            fig_b = px.bar(
                df_b, x="make", y="listing_count",
                color="median_price",
                color_continuous_scale="Blues",
                labels={"make": "Brand", "listing_count": "Listings", "median_price": "Median Price (Rs.)"},
                text="listing_count"
            )
            fig_b.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("No brand data found for selected category.")

    with c2:
        st.markdown("#### 🚘 Top Vehicle Models")
        models_data = analytics_engine.get_model_analysis(category=filter_cat, top_n=7)
        if models_data:
            df_m = pd.DataFrame(models_data)
            df_m["full_model"] = df_m["make"] + " " + df_m["model"]
            fig_m = px.bar(
                df_m, x="listing_count", y="full_model",
                orientation="h",
                color="median_price",
                color_continuous_scale="Teal",
                labels={"listing_count": "Listings", "full_model": "Model", "median_price": "Median Price (Rs.)"},
                text="listing_count"
            )
            fig_m.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No model data found for selected category.")

# ==========================================
# TAB 2: VEHICLE CATEGORIES
# ==========================================
with tabs[1]:
    st.subheader("Category-Specific Market Distribution")
    cat_records = db.query(Listing.category, Listing.current_price).filter(Listing.data_quality_status == "VALID").all()
    if cat_records:
        df_cat = pd.DataFrame([{"category": r[0], "price": r[1]} for r in cat_records])
        cat_summary = df_cat.groupby("category").agg(
            Volume=("price", "count"),
            Median_Price=("price", "median"),
            Average_Price=("price", "mean")
        ).reset_index()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Listing Volume by Category")
            fig_pie = px.pie(cat_summary, names="category", values="Volume", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            st.markdown("#### Median Asking Price by Category (Rs.)")
            fig_bar = px.bar(cat_summary, x="category", y="Median_Price", text_auto=".2s", color="category")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            cat_summary.style.format({
                "Volume": "{:,}",
                "Median_Price": "Rs. {:,.0f}",
                "Average_Price": "Rs. {:,.0f}"
            }),
            use_container_width=True
        )

# ==========================================
# TAB 3: MARKET ANALYTICS
# ==========================================
with tabs[2]:
    st.subheader("Deep-Dive Market Analytics")
    
    analytics_subtab = st.radio("Select Dimension", ["Brand & Model", "Age & Depreciation", "Fuel & Transmission", "District / Geographic"], horizontal=True)

    if analytics_subtab == "Brand & Model":
        brands = analytics_engine.get_brand_analysis(category=filter_cat, top_n=12)
        if brands:
            st.dataframe(pd.DataFrame(brands).style.format({
                "listing_count": "{:,}",
                "median_price": "Rs. {:,.0f}",
                "avg_price": "Rs. {:,.0f}",
                "min_price": "Rs. {:,.0f}",
                "max_price": "Rs. {:,.0f}",
                "avg_mileage": "{:,.0f} km",
                "market_share_pct": "{:.1f}%"
            }), use_container_width=True)

    elif analytics_subtab == "Age & Depreciation":
        st.markdown("#### Vehicle Age & Depreciation Trends")
        dep_data = analytics_engine.get_depreciation_analysis(category=filter_cat or "Cars")
        if dep_data:
            df_dep = pd.DataFrame(dep_data)
            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(
                x=df_dep["yom"], y=df_dep["median_price"],
                mode="lines+markers", name="Median Price",
                line=dict(color="#2563EB", width=3)
            ))
            fig_dep.add_trace(go.Scatter(
                x=df_dep["yom"], y=df_dep["p75_price"],
                mode="lines", name="75th Percentile",
                line=dict(color="#93C5FD", dash="dot")
            ))
            fig_dep.add_trace(go.Scatter(
                x=df_dep["yom"], y=df_dep["p25_price"],
                mode="lines", name="25th Percentile",
                line=dict(color="#93C5FD", dash="dot"),
                fill="tonexty", fillcolor="rgba(147, 197, 253, 0.2)"
            ))
            fig_dep.update_layout(xaxis_title="Year of Manufacture (YOM)", yaxis_title="Price (LKR)", hovermode="x unified")
            st.plotly_chart(fig_dep, use_container_width=True)

    elif analytics_subtab == "Fuel & Transmission":
        ft_data = analytics_engine.get_fuel_and_transmission_breakdown(category=filter_cat)
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.markdown("#### Asking Price by Fuel Type")
            if ft_data["fuel"]:
                df_fuel = pd.DataFrame(ft_data["fuel"])
                fig_fuel = px.bar(df_fuel, x="fuel_type", y="median_price", color="fuel_type", text_auto=".2s")
                st.plotly_chart(fig_fuel, use_container_width=True)
        with c_f2:
            st.markdown("#### Asking Price by Transmission")
            if ft_data["transmission"]:
                df_trans = pd.DataFrame(ft_data["transmission"])
                fig_trans = px.bar(df_trans, x="transmission", y="median_price", color="transmission", text_auto=".2s")
                st.plotly_chart(fig_trans, use_container_width=True)

    elif analytics_subtab == "District / Geographic":
        st.markdown("#### Price & Listing Distribution by District")
        dist_data = analytics_engine.get_district_pricing_heatmap(category=filter_cat)
        if dist_data:
            df_dist = pd.DataFrame(dist_data)
            fig_dist = px.bar(
                df_dist, x="district", y="listing_count",
                color="median_price", color_continuous_scale="Viridis",
                labels={"district": "District", "listing_count": "Listing Volume", "median_price": "Median Price (Rs.)"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 4: MARKET TRENDS
# ==========================================
with tabs[3]:
    st.subheader("Historical Market Trends & Listing Dynamics")
    movements = trend_engine.get_price_movement_summary(days=60)
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Observed Price Changes", f"{movements['total_price_changes']:,}")
    with col_t2:
        st.metric("Price Reductions 📉", f"{movements['price_drops_count']:,}")
    with col_t3:
        st.metric("Price Hikes 📈", f"{movements['price_hikes_count']:,}")
    with col_t4:
        st.metric("Avg Price Reduction", f"Rs. {movements['avg_price_drop_amount']:,.0f}")

    st.markdown("---")
    st.markdown("#### Listing Volume Trends Over Time")
    vol_trends = trend_engine.get_listing_volume_trends()
    if vol_trends:
        df_vol = pd.DataFrame(vol_trends)
        fig_vol = px.line(df_vol, x="period", y="listing_count", color="category", markers=True)
        st.plotly_chart(fig_vol, use_container_width=True)

# ==========================================
# TAB 5: VEHICLE ANALYSIS
# ==========================================
with tabs[4]:
    st.subheader("Granular Vehicle Model Deep-Dive")
    makes = [r[0] for r in db.query(Listing.make).distinct().filter(Listing.make.isnot(None)).order_by(Listing.make).all()]
    if makes:
        sel_make = st.selectbox("Select Make", options=makes, index=0)
        models = [r[0] for r in db.query(Listing.model).distinct().filter(Listing.make == sel_make).order_by(Listing.model).all()]
        if models:
            sel_model = st.selectbox("Select Model", options=models, index=0)
            
            listings_for_model = db.query(Listing).filter(
                Listing.make == sel_make,
                Listing.model == sel_model,
                Listing.data_quality_status == "VALID"
            ).all()

            if listings_for_model:
                df_spec = pd.DataFrame([{
                    "YOM": l.yom,
                    "Condition": l.condition,
                    "Mileage (km)": l.mileage_km,
                    "Fuel": l.fuel_type,
                    "Transmission": l.transmission,
                    "District": l.district,
                    "Price (Rs.)": l.current_price,
                    "Status": l.current_status
                } for l in listings_for_model])

                st.markdown(f"#### Market Overview for **{sel_make} {sel_model}** ({len(df_spec)} observed listings)")
                k1, k2, k3 = st.columns(3)
                k1.metric("Median Price", f"Rs. {df_spec['Price (Rs.)'].median():,.0f}")
                k2.metric("Lowest Observed", f"Rs. {df_spec['Price (Rs.)'].min():,.0f}")
                k3.metric("Highest Observed", f"Rs. {df_spec['Price (Rs.)'].max():,.0f}")

                fig_sc = px.scatter(
                    df_spec, x="Mileage (km)", y="Price (Rs.)",
                    color="YOM", size="Price (Rs.)", hover_data=["District", "Condition"],
                    title=f"Price vs Mileage for {sel_make} {sel_model}"
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                st.dataframe(df_spec.style.format({"Price (Rs.)": "Rs. {:,.0f}", "Mileage (km)": "{:,.0f} km"}), use_container_width=True)

# ==========================================
# TAB 6: COMPARABLE VEHICLES
# ==========================================
with tabs[5]:
    st.subheader("🔍 Comparable Vehicle Discovery Engine")
    st.caption("Enter vehicle specifications to discover similar active and historical listings with similarity scores.")

    with st.form("comp_form"):
        col_cp1, col_cp2, col_cp3 = st.columns(3)
        with col_cp1:
            cp_cat = st.selectbox("Category", options=["Cars", "Vans", "SUVs", "Three-Wheel", "Motorbikes", "Lorries", "Buses"], index=0)
            cp_make = st.text_input("Make", value="Toyota")
        with col_cp2:
            cp_model = st.text_input("Model", value="Aqua")
            cp_yom = st.number_input("Year of Manufacture", min_value=1980, max_value=2026, value=2018)
        with col_cp3:
            cp_mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=65000, step=5000)
            cp_fuel = st.selectbox("Fuel Type", options=["Hybrid", "Petrol", "Diesel", "Electric"], index=0)

        col_cp4, col_cp5 = st.columns(2)
        with col_cp4:
            cp_trans = st.selectbox("Transmission", options=["Automatic", "Manual", "Tiptronic"], index=0)
        with col_cp5:
            cp_district = st.selectbox("District", options=settings.BASE_DIR and [
                "Colombo", "Gampaha", "Kalutara", "Kandy", "Galle", "Kurunegala", "Matara", "Anuradhapura"
            ], index=0)

        submit_comp = st.form_submit_button("🔎 Find Comparable Vehicles", use_container_width=True)

    if submit_comp:
        with st.spinner("Finding comparable vehicles..."):
            comp_res = comparable_engine.find_comparables(
                category=cp_cat,
                make=cp_make,
                model=cp_model,
                yom=int(cp_yom),
                mileage_km=int(cp_mileage),
                fuel_type=cp_fuel,
                transmission=cp_trans,
                district=cp_district,
                top_k=6
            )

        if comp_res["comparables"]:
            st.success(f"Found {comp_res['count']} comparable listings in the market dataset.")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median Comparable Price", f"Rs. {comp_res['median_price']:,.0f}")
            m2.metric("25th - 75th Percentile", f"Rs. {comp_res['p25_price']:,.0f} – {comp_res['p75_price']:,.0f}")
            m3.metric("Lowest Comparable", f"Rs. {comp_res['min_price']:,.0f}")
            m4.metric("Highest Comparable", f"Rs. {comp_res['max_price']:,.0f}")

            st.markdown("#### Top Matching Comparable Listings")
            for c in comp_res["comparables"]:
                with st.expander(f"🚗 {c['title']} — Rs. {c['price_rs']:,.0f} (Similarity: {c['similarity_score']}%)", expanded=True):
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.write(f"**Year:** {c['yom']}")
                    col_m2.write(f"**Mileage:** {c['mileage_km']:,} km")
                    col_m3.write(f"**Fuel / Gear:** {c['fuel_type']} / {c['transmission']}")
                    col_m4.write(f"**Location:** {c['district']}")

# ==========================================
# TAB 7: AI VALUATION & SHAP EXPLAINABILITY
# ==========================================
with tabs[6]:
    st.subheader("🤖 AI Market Asking Value Estimation & Explainable AI")
    st.caption("Estimate the fair market asking range using Category-Trained Machine Learning Models and view SHAP factor influences.")

    with st.form("val_form"):
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            v_cat = st.selectbox("Category", options=["Cars", "Vans", "SUVs", "Three-Wheel", "Motorbikes", "Lorries", "Buses"], index=0, key="val_cat")
            v_make = st.text_input("Make", value="Toyota", key="val_make")
        with v_col2:
            v_model = st.text_input("Model", value="Aqua", key="val_model")
            v_yom = st.number_input("Year of Manufacture", min_value=1980, max_value=2026, value=2018, key="val_yom")
        with v_col3:
            v_mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=65000, step=5000, key="val_km")
            v_fuel = st.selectbox("Fuel Type", options=["Hybrid", "Petrol", "Diesel", "Electric"], index=0, key="val_fuel")

        v_col4, v_col5, v_col6 = st.columns(3)
        with v_col4:
            v_trans = st.selectbox("Transmission", options=["Automatic", "Manual", "Tiptronic"], index=0, key="val_trans")
        with v_col5:
            v_district = st.selectbox("District", options=["Colombo", "Gampaha", "Kalutara", "Kandy", "Galle", "Kurunegala", "Matara", "Anuradhapura"], index=0, key="val_dist")
        with v_col6:
            v_condition = st.selectbox("Condition", options=["Used", "Reconditioned", "Brand New"], index=0, key="val_cond")

        submit_val = st.form_submit_button("⚡ Estimate Market Asking Value", use_container_width=True)

    if submit_val:
        with st.spinner("Calculating ML Market Asking Value and SHAP Attribution..."):
            val_res = predictor.predict_value(
                category=v_cat,
                make=v_make,
                model=v_model,
                yom=int(v_yom),
                mileage_km=int(v_mileage),
                fuel_type=v_fuel,
                transmission=v_trans,
                condition=v_condition,
                district=v_district
            )

        if val_res.get("status") == "SUCCESS":
            est_val = val_res["estimated_market_asking_value"]
            r_low = val_res["estimated_market_range"]["low"]
            r_high = val_res["estimated_market_range"]["high"]
            conf = val_res["confidence"]

            st.markdown("---")
            st.markdown("### 🏷️ Estimated Market Asking Value")

            res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 1])
            with res_c1:
                st.metric("Estimated Market Asking Value", f"Rs. {est_val:,.0f}")
            with res_c2:
                st.metric("Estimated Market Range", f"Rs. {r_low:,.0f} – {r_high:,.0f}")
            with res_c3:
                conf_color = "green" if conf == "High" else ("orange" if conf == "Medium" else "red")
                st.markdown(f"**Confidence Rating:**")
                st.markdown(f"<span style='color:{conf_color}; font-size:1.4rem; font-weight:bold;'>{conf}</span>", unsafe_allow_html=True)
                st.caption(f"Based on {val_res['similar_listings_count']} similar market listings.")

            # Explainable AI Section
            st.markdown("---")
            st.markdown("### 🔍 Explainable AI (SHAP Factor Influence Breakdown)")
            st.caption("How each vehicle attribute influenced the estimated price relative to the market baseline:")

            shap_exp = explainer.explain_prediction(
                category=v_cat,
                vehicle_dict={
                    "make": v_make,
                    "model": v_model,
                    "yom": v_yom,
                    "mileage_km": v_mileage,
                    "fuel_type": v_fuel,
                    "transmission": v_trans,
                    "condition": v_condition,
                    "district": v_district
                }
            )

            if shap_exp.get("status") == "SUCCESS" and shap_exp.get("contributions"):
                df_shap = pd.DataFrame(shap_exp["contributions"])
                fig_shap = px.bar(
                    df_shap,
                    x="impact_rs",
                    y="feature",
                    orientation="h",
                    color="impact_rs",
                    color_continuous_scale="RdBu",
                    labels={"impact_rs": "Price Impact (Rs. in LKR)", "feature": "Vehicle Attribute"},
                    text_auto=".2s"
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_shap, use_container_width=True)

                st.dataframe(
                    df_shap[["feature", "impact_rs", "impact_direction", "impact_strength"]].style.format({"impact_rs": "Rs. {:,.0f}"}),
                    use_container_width=True
                )

        elif val_res.get("status") == "INSUFFICIENT_DATA":
            st.warning(f"⚠️ {val_res.get('message')}")

# ==========================================
# TAB 8: ASKING PRICE ASSESSMENT
# ==========================================
with tabs[7]:
    st.subheader("⚖️ Seller Asking Price Assessment")
    st.caption("Compare a seller's asking price against the data-driven market range to assess market positioning.")

    with st.form("assess_form"):
        as_c1, as_c2 = st.columns(2)
        with as_c1:
            as_cat = st.selectbox("Category", options=["Cars", "Vans", "SUVs", "Three-Wheel", "Motorbikes", "Lorries", "Buses"], index=0, key="as_cat")
            as_make = st.text_input("Make", value="Toyota", key="as_make")
            as_model = st.text_input("Model", value="Aqua", key="as_model")
            as_yom = st.number_input("Year", min_value=1980, max_value=2026, value=2018, key="as_yom")
        with as_c2:
            as_mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=65000, step=5000, key="as_km")
            as_fuel = st.selectbox("Fuel Type", options=["Hybrid", "Petrol", "Diesel", "Electric"], index=0, key="as_fuel")
            as_trans = st.selectbox("Transmission", options=["Automatic", "Manual", "Tiptronic"], index=0, key="as_trans")
            as_asking_price = st.number_input("Seller Asking Price (Rs.)", min_value=100000.0, value=8500000.0, step=50000.0)

        submit_assess = st.form_submit_button("📊 Assess Asking Price", use_container_width=True)

    if submit_assess:
        with st.spinner("Evaluating price assessment..."):
            as_res = predictor.predict_value(
                category=as_cat,
                make=as_make,
                model=as_model,
                yom=int(as_yom),
                mileage_km=int(as_mileage),
                fuel_type=as_fuel,
                transmission=as_trans,
                seller_asking_price=float(as_asking_price)
            )

        if as_res.get("status") == "SUCCESS":
            est_p = as_res["estimated_market_asking_value"]
            r_lo = as_res["estimated_market_range"]["low"]
            r_hi = as_res["estimated_market_range"]["high"]
            assess_info = as_res["price_assessment"]
            classification = assess_info["classification"]
            diff_pct = assess_info["difference_percent"]

            st.markdown("---")
            st.markdown("### Assessment Result")

            badge_class = "badge-within"
            if classification == "BELOW MARKET RANGE":
                badge_class = "badge-below"
            elif classification == "ABOVE MARKET RANGE":
                badge_class = "badge-above"

            st.markdown(f"<h4>Positioning: <span class='{badge_class}'>{classification}</span></h4>", unsafe_allow_html=True)

            k_a1, k_a2, k_a3, k_a4 = st.columns(4)
            k_a1.metric("Seller Asking Price", f"Rs. {as_asking_price:,.0f}")
            k_a2.metric("Estimated Market Value", f"Rs. {est_p:,.0f}")
            k_a3.metric("Price Difference", f"{diff_pct:+.1f}%")
            k_a4.metric("Estimated Range", f"Rs. {r_lo:,.0f} – {r_hi:,.0f}")

            # Optional Negotiation Insights
            insights_res = NegotiationInsightEngine.generate_negotiation_insights(
                estimated_value=est_p,
                range_low=r_lo,
                range_high=r_hi,
                seller_asking_price=as_asking_price
            )

            if insights_res.get("available"):
                st.markdown("---")
                st.markdown("#### 💡 Data-Driven Market Perspectives (Negotiation Context)")
                for obs in insights_res["observations"]:
                    st.write(f"• {obs}")
                st.info(f"**Reference Market Range:** {insights_res['suggested_negotiation_reference_range']}")
                st.caption(f"_{insights_res['disclaimer']}_")

# ==========================================
# TAB 9: DATA QUALITY
# ==========================================
with tabs[8]:
    st.subheader("🛡️ Data Quality & Anomaly Detection Audit")
    
    total_q = db.query(Listing).count()
    valid_q = db.query(Listing).filter(Listing.data_quality_status == "VALID").count()
    susp_q = db.query(Listing).filter(Listing.data_quality_status == "SUSPICIOUS").count()
    inv_q = db.query(Listing).filter(Listing.data_quality_status == "INVALID").count()
    miss_q = db.query(Listing).filter(Listing.data_quality_status == "MISSING").count()

    q_score = (valid_q / total_q * 100.0) if total_q > 0 else 100.0

    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    qc1.metric("Overall Quality Score", f"{q_score:.1f}%")
    qc2.metric("Valid Records ✅", f"{valid_q:,}")
    qc3.metric("Suspicious Records ⚠️", f"{susp_q:,}")
    qc4.metric("Invalid Records ❌", f"{inv_q:,}")
    qc5.metric("Missing Critical ❓", f"{miss_q:,}")

    st.markdown("---")
    st.markdown("#### Data Quality Score Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=q_score,
        title={'text': "Platform Data Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2563EB"},
            'steps': [
                {'range': [0, 60], 'color': "#FEE2E2"},
                {'range': [60, 85], 'color': "#FEF3C7"},
                {'range': [85, 100], 'color': "#D1FAE5"}
            ],
            'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### Recent Data Quality Audit Records")
    recent_dq = db.query(DataQualityRecord).order_by(DataQualityRecord.evaluated_at.desc()).limit(15).all()
    if recent_dq:
        df_dq = pd.DataFrame([{
            "Listing ID": r.listing_id,
            "Quality Status": r.status,
            "Quality Score": r.score,
            "Is Outlier": "Yes" if r.is_outlier else "No",
            "Issues Detected": r.issues_json,
            "Evaluated At": r.evaluated_at
        } for r in recent_dq])
        st.dataframe(df_dq, use_container_width=True)

# ==========================================
# TAB 10: SYSTEM MONITORING
# ==========================================
with tabs[9]:
    st.subheader("⚙️ System Monitoring & Automated Pipeline")
    
    st.markdown("#### Run Pipeline / Seed Dataset")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("🌱 Seed Realistic Sri Lankan Vehicle Dataset (1,500 listings)", use_container_width=True):
            with st.spinner("Seeding database and running quality validation..."):
                runner = PipelineRunner(db=db)
                seed_res = runner.run_pipeline(use_mock_data=True, mock_count=1500)
                st.success(f"Pipeline executed successfully! Run ID: {seed_res['run_id']}")
                st.rerun()

    with p_col2:
        if st.button("🧠 Retrain All ML Valuation Models", use_container_width=True):
            with st.spinner("Retraining Category Models & updating model registry..."):
                from ml.training.trainer import ModelTrainer
                tr = ModelTrainer(db=db)
                t_res = tr.train_all_supported_categories()
                st.success("All vehicle models successfully retrained and registered!")
                st.rerun()

    st.markdown("---")
    st.markdown("#### Historical Scrape & Pipeline Runs")
    runs = db.query(ScrapeRun).order_by(ScrapeRun.started_at.desc()).limit(10).all()
    if runs:
        df_runs = pd.DataFrame([{
            "Run ID": r.run_id,
            "Category": r.category or "All",
            "Started At": r.started_at,
            "Total Scraped": r.total_scraped,
            "New Listings": r.new_count,
            "Updated": r.updated_count,
            "Price Changes": r.price_change_count,
            "Inactive Detected": r.no_longer_observed_count,
            "Status": r.status
        } for r in runs])
        st.dataframe(df_runs, use_container_width=True)

    st.markdown("#### Active Machine Learning Model Registry")
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).all()
    if mv:
        df_mv = pd.DataFrame([{
            "Category": m.category,
            "Algorithm": m.algorithm,
            "Trained At": m.trained_at,
            "Metrics": m.metrics_json,
            "Artifact Path": m.artifact_path
        } for m in mv])
        st.dataframe(df_mv, use_container_width=True)

db.close()
