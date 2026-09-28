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

## 🔎 Comparable Vehicles Search UI (Phase 10.4)

The **Comparable Vehicles** workflow (`dashboard/pages/comparable_vehicles_page.py`) provides an interactive, descriptive search interface for identifying similar market listings from the existing database.

```
Target Vehicle Input
        ↓
Existing Comparable Engine (analytics/comparables/comparable_engine.py)
        ↓
Comparable Vehicles Ranking
        ↓
Comparable Market Summary (analytics/comparables/market_summary.py)
        ↓
User-Friendly Visual Comparison
```

### 1. Purpose & Methodological Scope
- **Descriptive Reference**: Allows users to inspect active, verified market listings with physical specifications closest to a target vehicle.
- **Strictly Descriptive / Comparative**: Designed to contextualize market asking prices. It does **not** generate point valuations or appraisals.
- **No Confirmed Transaction Prices**: All observed monetary figures are **advertised asking prices** from market postings. They do not represent verified or final transaction prices.
- **Semantic Definition of Similarity**:
  - The **Similarity** score represents multi-attribute specification distance based on implemented feature weights.
  - It does **NOT** represent prediction confidence, probability, valuation accuracy, match certainty, or vehicle identity.

### 2. Target Vehicle Input Fields
The search form accepts the standard 10 vehicle specification attributes:
| Field | UI Control | Canonical Validation Rules |
| :--- | :--- | :--- |
| **Category** | Selectbox | Must match canonical categories (`Cars`, `SUVs`, `Vans`, `Motorbikes`, etc.) |
| **Brand / Make** | Text Input | Non-empty manufacturer name (e.g., `Toyota`, `Honda`) |
| **Model** | Text Input | Non-empty model name (e.g., `Premio`, `Fit`, `Vezel`) |
| **Manufacture Year** | Number Input | Valid integer between 1950 and current calendar year (2026) |
| **Mileage (km)** | Number Input | Non-negative numeric odometer reading (optional; leave 0 if unknown) |
| **Engine CC** | Number Input | Non-negative engine displacement in cubic centimetres (optional) |
| **Fuel Type** | Selectbox | Allowed fuels: `Petrol`, `Diesel`, `Hybrid`, `Electric`, `LPG`, `CNG`, `Gas` |
| **Transmission** | Selectbox | `Automatic`, `Manual` |
| **District** | Selectbox | 25 administrative districts of Sri Lanka |
| **Condition** | Selectbox | `Registered (Used)`, `Unregistered`, `Brand New` |

> **Data Integrity Guarantee**: The target vehicle is strictly an in-memory input query. It is never inserted, updated, or written to PostgreSQL.

### 3. Comparable Engine Integration
- **Engine Reuse**: Directly invokes `ComparableVehicleEngine` (`analytics/comparables/comparable_engine.py`).
- **No Second Algorithm**: Retains existing attribute similarity weights:
  - Brand (Make): **0.25**
  - Model: **0.25**
  - Manufacture Year / Age: **0.14**
  - Mileage: **0.10**
  - Engine CC: **0.08**
  - Transmission: **0.06**
  - Fuel Type: **0.04**
  - District / Location: **0.04**
  - Condition: **0.04**
- **Strict Category Matching**: Only listings belonging to the same canonical vehicle category are retrieved.
- **Data Quality & Privacy**: Read-only extraction via `MLDatasetLoader` enforces `ml_eligible = True` and strictly excludes seller phone, email, and private contact information.

### 4. Results Presentation & Formatting
- **Clean Table**: Displays Brand, Model, Category, Manufacture Year, Mileage (`km`), Engine CC (`cc`), Fuel Type, Transmission, District, Condition, Advertised Asking Price (`LKR`), and Similarity (`%`).
- **Privacy Enforcement**: No seller contact details are ever exposed in the table or DOM.
- **Missing Value Handling**: Incomplete optional specifications are formatted as `N/A` without synthetic imputation.

### 5. Comparable Market Summary
- **Module Reuse**: Direct call to `create_comparable_market_summary` (`analytics/comparables/market_summary.py`).
- **Descriptive Statistics**:
  - **Comparable Count**
  - **Minimum Advertised Asking Price (LKR)**
  - **Maximum Advertised Asking Price (LKR)**
  - **Median Advertised Asking Price (LKR)**
  - **Average Advertised Asking Price (LKR)**
  - **Advertised Asking Price Spread (LKR)**
- **Methodological Disclaimer**: Prominently notes that figures reflect advertised asking prices of retrieved comparables, not verified transaction prices.
- **Zero / Low-Data Handling**:
  - Zero comparables: Displays *"No comparable vehicles were found for the selected target."* with null fields safely represented.
  - Low-sample caution: Displays *"Limited comparable data is available. Interpret the market summary with caution."* when 1–2 comparables are returned.

### 6. Comparison Visualizations
Four dynamic charts contextualize the retrieved comparables:
1. **Target Vehicle vs. Comparable Asking Prices**: Bar chart displaying asking prices per listing with a horizontal median reference line.
2. **Manufacture Year vs. Advertised Asking Price**: Scatter plot with target vehicle year marked by a vertical dashed reference line.
3. **Odometer Mileage vs. Advertised Asking Price**: Scatter plot with target vehicle mileage marked by a vertical dashed reference line (rendered when $\ge 2$ comparables possess mileage data).
4. **Comparable Asking Price Distribution**: Frequency histogram of asking prices (rendered only when $\ge 3$ comparables are available to prevent misleading small-sample distributions).

### 7. Result Controls & Display Filters
- **Maximum Comparables**: User-configurable retrieval cap (1 to 20 listings).
- **Minimum Similarity Display (%)**: Visual threshold slider filtering displayed results. Does not alter underlying similarity calculations or engine weights.
- **Multi-Field Sorting**:
  - Similarity (highest first)
  - Asking Price (lowest first)
  - Asking Price (highest first)
  - Year (newest first)

---

## 🧪 Testing & Verification

The platform is thoroughly covered by automated test suites:

```bash
# Run Phase 10.4 Comparable Vehicles Dashboard tests:
.venv/bin/pytest tests/test_comparable_vehicles_dashboard.py -v

# Run existing comparable engine and market summary tests:
.venv/bin/pytest tests/test_comparable_engine.py tests/test_comparable_market_summary.py -v

# Run Phase 10.3 Market Intelligence tests:
.venv/bin/pytest tests/test_market_intelligence.py -q

# Run full project test suite:
.venv/bin/pytest -q
```

All **389 tests pass** with zero regressions.

---

## 🚀 Running the Dashboard Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Streamlit
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` and select:
- **🔎 Comparable Vehicles** for the multi-attribute comparable search workflow (Phase 10.4).
- **📈 Market Intelligence** for marketplace analytics (Phase 10.3).
- **🔍 Vehicle Valuation** for ML-driven asking price predictions (Phase 10.2).
- **🏠 Overview** for platform introduction and API connectivity status.
