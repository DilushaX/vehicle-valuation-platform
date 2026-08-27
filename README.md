# 🇱🇰 Sri Lankan Vehicle Market Intelligence & ML Valuation Platform

A production-grade, end-to-end data-driven market intelligence and machine learning valuation platform for Sri Lankan used vehicles.

The system continuously tracks vehicle listings (primarily from **Riyasewana**), audits data quality, detects suspicious anomalies, maintains historical price and status timelines, performs multi-factor comparable matching, and estimates market asking values using category-specific ML models with **Explainable AI (SHAP)**.

---

## 📌 Key Capabilities

1. **Polite Automated Crawler**:
   - Multi-category support (Cars, Vans, SUVs, Motorbikes, Three-Wheelers, Lorries, Buses).
   - Polite rate-limiting (1.5s – 3.0s delay), rotating headers, and exponential backoff retry.
   - Incremental delta scraping (stops when known listing IDs are encountered).
   - Status tracking: Unseen listings marked as `NO_LONGER_OBSERVED` (never assumed sold).

2. **Data Quality & Anomaly Engine**:
   - Rule-based and pattern validation (detects dummy mileage like `123, 111111`, dummy prices like `123, 111111`).
   - Grouped IQR statistical outlier detection per `(Category, Make, Model)`.
   - Classification: `VALID`, `SUSPICIOUS`, `INVALID`, `MISSING` (raw data preserved).
   - Attribute fingerprint deduplication.

3. **Historical Price & Status Tracking**:
   - Tracks price hikes and drops with delta amounts and percentages.
   - Maintains full lifecycle audit from first seen to removal.

4. **Comparable Vehicle Engine**:
   - Multi-attribute weighted similarity scoring (Make, Model, Year decay, Mileage distance, Fuel, Transmission, District).
   - Returns Top-K matching vehicles with similarity score percentages (e.g. 94%, 91%).

5. **Machine Learning Valuation & Explainable AI (SHAP)**:
   - Category-specific regression models (Gradient Boosting, Random Forest, Ridge, XGBoost).
   - Outputs: **Estimated Market Asking Value**, **Estimated Market Range (Confidence Interval)**, and **Confidence Rating** (High/Medium/Low).
   - "Insufficient Market Data" safety fallback for sparse models.
   - **SHAP TreeExplainer** factor breakdown in Sri Lankan Rupees.
   - **Asking Price Assessment**: Classifies seller price as `BELOW MARKET RANGE`, `WITHIN MARKET RANGE`, or `ABOVE MARKET RANGE`.
   - Data-driven negotiation context with explicit legal disclaimers.

6. **Interactive 10-Tab Streamlit Dashboard**:
   - Overview KPIs, Category Comparison, Market Analytics, Market Trends, Model Deep-Dive, Comparable Finder, AI Valuation & SHAP Waterfall, Price Assessment, Data Quality Audit, and System Monitoring.

7. **FastAPI REST Backend**:
   - Complete RESTful endpoints for valuation, comparables, market analytics, trends, and quality reports.

---

## 🏗️ Architecture

```
                    Riyasewana Listing Pages
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │  Polite Async/Sync Scraper (HTTPX / BS4)  │
        │  - Delta scraping (stop on seen IDs)      │
        │  - Exponential backoff & rate limiting    │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │           Data Quality Engine             │
        │  - Missing / Type / Pattern Validation    │
        │  - Statistical Outlier Detection (IQR)    │
        │  - Categorization: VALID/SUSPICIOUS/      │
        │    INVALID/MISSING (Raw Data Preserved)   │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │      PostgreSQL / SQLite Database         │
        │  - Vehicles, Listings, Price History      │
        │  - Status History, Scrape Runs, Quality   │
        └─────────────────────┬─────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Market Trend &  │  │   Comparable    │  │  ML Valuation   │
│ Analytics Engine│  │ Vehicle Engine  │  │ & SHAP Explainer│
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │            FastAPI Backend                │
        │  - REST Endpoints (/predict, /compare,    │
        │    /analytics, /trends, /quality, /health)│
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │   Streamlit Interactive Dashboard (10 Tab)│
        └───────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup

```bash
# Clone repository
cd vehicle-valuation-platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Seed Sample Market Data

```bash
# Populate database with 1,500 realistic Sri Lankan vehicle records
python3 -m data_pipeline.pipeline_runner --seed-samples --count 1500
```

### 3. Train ML Valuation Models

```bash
# Train and register category-specific ML models
python3 -m ml.training.trainer --train-all
```

### 4. Run Automated Test Suite

```bash
python3 -m pytest tests/ -v
```

### 5. Launch FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 6. Launch Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```
- Dashboard UI: `http://localhost:8501`

---

## 🐳 Docker Deployment

To launch PostgreSQL, FastAPI Backend, and Streamlit Dashboard simultaneously:

```bash
docker-compose up --build
```

---

## 📑 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/valuation/predict` | ML Price Valuation, Range, Confidence & SHAP Factor Breakdown |
| `POST` | `/api/v1/comparables/search` | Multi-attribute similarity search returning top matching listings |
| `GET` | `/api/v1/analytics/overview` | High-level market KPIs (Median/Average prices, top makes & models) |
| `GET` | `/api/v1/analytics/brands` | Brand listing volumes, price statistics, market share % |
| `GET` | `/api/v1/analytics/models` | Model listing volumes, prices, and year distributions |
| `GET` | `/api/v1/analytics/depreciation` | Vehicle age vs median asking price depreciation curves |
| `GET` | `/api/v1/analytics/fuel-transmission` | Fuel type and transmission price breakdown |
| `GET` | `/api/v1/analytics/districts` | Geographic district price and listing distribution |
| `GET` | `/api/v1/analytics/trends/price-movements` | Historical price reduction and price increase summary |
| `GET` | `/api/v1/quality/summary` | Data quality audit (Valid, Suspicious, Invalid, Missing) |
| `GET` | `/api/v1/quality/runs` | Historical scraper run telemetry and delta execution logs |

---

## ⚠️ Important Legal & Technical Disclaimers

1. **Asking Price vs Transaction Price**: Listing prices observed on Riyasewana are seller asking prices, not confirmed settlement prices.
2. **Observation Lifecycle**: An ad no longer appearing on Riyasewana is classified as `NO_LONGER_OBSERVED` and is never assumed to be a completed transaction.
3. **Statistical Estimates**: ML predictions represent observed market asking ranges and do not constitute financial appraisals or guarantees.
