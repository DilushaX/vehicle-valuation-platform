# 📈 Market Intelligence & Dashboard Documentation

Welcome to the **Sri Lankan Vehicle Market Intelligence Dashboard** (Phase 10.3).

This dashboard provides an interactive, descriptive exploration of the Sri Lankan vehicle market using publicly collected listing data from [Riyasewana](https://riyasewana.com).

---

## 🏛️ Core Principles & Methodological Governance

1. **Market Intelligence, Not Valuation Confidence**:
   - This dashboard visualizes the advertised marketplace landscape.
   - It is strictly separated from individual vehicle valuation confidence scoring.

2. **Advertised Asking Prices vs. Transaction Prices**:
   - All monetary figures reflect **Advertised Asking Prices in Sri Lankan Rupees (LKR)**.
   - Asking prices represent seller listing expectations and do **not** equal confirmed transaction or negotiated selling prices.

3. **Non-Causal Descriptive Interpretation**:
   - Visualized associations (e.g., vehicle age vs. price, mileage vs. price, district vs. price) describe observed patterns in the current dataset.
   - They do **not** assert mathematical causality (e.g., the dashboard does not claim that a location or year causes price changes).

4. **Lifecycle & Disappearance Semantics**:
   - Observed listing disappearance is recorded as `NO_LONGER_OBSERVED`.
   - Listing removal does **not** by itself confirm that a vehicle was sold or confirm the final price.

5. **Historical Integrity & Non-Fabrication**:
   - Longitudinal trend analysis strictly requires a minimum observation depth of **60 days**.
   - If historical depth is insufficient, the dashboard explicitly reports:
     > *"Insufficient historical observations for this analysis."*
   - The engine never fabricates, synthesizes, or extrapolates time-series data.

6. **Listings vs. Physical Vehicles**:
   - The dashboard explicitly distinguishes between **Listings** (advertised market postings) and **Vehicles** (unique vehicle technical specifications). In the current database schema, each listing maps to a specification record; repeat advertisements are documented accordingly.

---

## 🧭 Dashboard Architecture & Sections

The Market Intelligence page (`dashboard/pages/market_intelligence_page.py`) is structured into nine cohesive analytical sections:

```
┌─────────────────────────────────────────────────────────────┐
│                 Market Intelligence Page                    │
├─────────────────────────────────────────────────────────────┤
│  1. Market Scope & Legal Disclaimers Banner                 │
├─────────────────────────────────────────────────────────────┤
│  2. Global Interactive Filters (Sidebar / In-Page)          │
├─────────────────────────────────────────────────────────────┤
│  3. Market Overview KPIs (Listings, Specs, Medians, ML-El)  │
├─────────────────────────────────────────────────────────────┤
│  4. Category Analysis (Distribution & Median/Mean Prices)   │
├─────────────────────────────────────────────────────────────┤
│  5. Brand & Model Analysis (Top-N Volumes & Distributions)  │
├─────────────────────────────────────────────────────────────┤
│  6. Price Analysis (Log/Linear Distribution & Correlations) │
├─────────────────────────────────────────────────────────────┤
│  7. Vehicle Characteristics (Fuel, Transmission, Age, Odo)  │
├─────────────────────────────────────────────────────────────┤
│  8. Geographic Analysis (Observed Differences by District)  │
├─────────────────────────────────────────────────────────────┤
│  9. Historical Market Trends (Depth-Gated Longitudinal)     │
├─────────────────────────────────────────────────────────────┤
│ 10. Data Quality Indicators & Attribute Completeness        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Interactive Global Filters

The dashboard provides dynamic, multi-factor filtering powered by `analytics/market/market_analytics.py`:

| Filter | Type | Description |
| :--- | :--- | :--- |
| **Category** | Multi-select | Filter by available categories (`Cars`, `Vans`, `SUVs`, `Motorbikes`, `Pickups`, `Heavy-Duty`, `Lorries`, `Three Wheelers`). |
| **Brand** | Multi-select | Filter by vehicle make (dynamically populated from current data). |
| **Model** | Multi-select | Cascades based on selected brand. |
| **District** | Multi-select | Filter across 25 Sri Lankan administrative districts. |
| **Fuel Type** | Multi-select | `Petrol`, `Diesel`, `Hybrid`, `Electric`, etc. |
| **Transmission** | Multi-select | `Automatic`, `Manual`. |
| **Condition** | Multi-select | `Used`, `Reconditioned`, `Brand New`. |
| **Manufacture Year** | Range Slider | Filter by minimum and maximum manufacture year. |
| **Asking Price (LKR)**| Range Slider | Filter by minimum and maximum asking price. |

**Performance Optimization**: All global filters execute in-memory against a single cached dataset (`@st.cache_data(ttl=300)`). No redundant database queries are executed when toggling filters.

---

## 📊 Detailed Analytics Modules

### 1. Market Overview KPIs
- **Total Advertised Listings**: Count of active and tracked marketplace postings.
- **Unique Vehicle Specifications**: Count of distinct vehicle technical records.
- **Distinct Categories, Brands & Models**: Marketplace catalog depth.
- **Median Asking Price (LKR)**: Robust non-parametric central tendency.
- **Average Asking Price (LKR)**: Parametric arithmetic mean.
- **ML-Eligible Ratio (%)**: Proportion of listings meeting strict ML data validation rules.

### 2. Category Analysis
- Bar chart comparing total listing volume across vehicle classes.
- Grouped bar comparison of **Median vs. Mean Advertised Asking Price** per category.
- Comprehensive category summary table with percentage shares and low-sample indicators.

### 3. Brand & Model Analysis
- Configurable **Top-N** selector (Top 5, 10, 15, 20 brands/models).
- Horizontal bar charts of highest-volume brands and models.
- Box plot distributions illustrating asking price dispersion per major brand.
- Sample size flags (`< 3` listings flagged as low sample) to prevent over-generalization.

### 4. Asking Price Analysis
- Price distribution histogram with **Linear / Logarithmic (log10)** scale toggle.
- Parametric metrics: Mean, Standard Deviation, Minimum, Maximum.
- Non-parametric metrics: Median, First Quartile (Q1), Third Quartile (Q3), Interquartile Range (IQR).
- Bivariate scatter visualizations:
  - Advertised Asking Price vs. Vehicle Age
  - Advertised Asking Price vs. Odometer Mileage
  - Advertised Asking Price vs. Engine Capacity (CC)
- Correlation summary matrix reporting Spearman ($\rho$) and Pearson ($r$) coefficients.

### 5. Vehicle Characteristics
- **Fuel Type Analysis**: Listing counts, median asking price, and mean asking price across Petrol, Diesel, and Hybrid vehicles.
- **Transmission Analysis**: Comparative market share and median prices for Automatic vs. Manual transmissions.
- **Vehicle Age & Manufacture Year**: Age distribution, median asking price curve by age, and manufacture year percentiles.
- **Odometer Mileage Analysis**: Distribution across standard mileage brackets (`< 25k`, `25k–50k`, `50k–100k`, `100k–150k`, `150k–200k`, `> 200k km`).

### 6. Geographic Analysis
- Listing volume distribution across Sri Lankan districts.
- Observed asking-price differences by district (clearly labeled as non-causal observations).
- Summary table with listing counts, percentage of total, and price statistics.

### 7. Historical Market Trends (Depth Gated)
- Powered by `analytics/trends/trend_engine.py`.
- Computes monthly listing activity, median asking price changes, and category-level dynamics.
- **Depth Gate**: Requires at least **60 days** of longitudinal history. If current observation span is less than 60 days, displays an informative notice explaining that historical data is accumulating and will not be synthetically generated.

### 8. Data Quality & Market Scope Indicators
- Compact audit card summarizing data cleanliness:
  - Total records audited
  - Valid / ML-Eligible records vs. records with validation issues
  - Attribute completeness percentages (Year, Price, Mileage, Fuel, Transmission, District)
  - Common validation issues breakdown (e.g. non-numeric mileage, extreme outliers, missing fields)
- Reminder that data quality metrics describe input completeness, **not** valuation accuracy.

---

## 🧪 Testing & Verification

The market intelligence analytics layer is thoroughly tested with unit and integration tests:

```bash
# Run all market intelligence tests:
.venv/bin/pytest tests/test_market_intelligence.py -q

# Run full project test suite:
.venv/bin/pytest -q
```

All 375 tests pass with zero regressions.

---

## 🚀 Running the Dashboard Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Streamlit
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` and select **📈 Market Intelligence** from the sidebar navigation.
