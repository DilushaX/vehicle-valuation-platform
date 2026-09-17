# Vehicle Valuation Model Training & Evaluation Report

**Evaluation Date**: 2026-09-17 05:50:17 UTC  
**Target Variable**: Observed Seller Asking Price (`asking_price`) in LKR  
**Evaluation Scale**: Original Sri Lankan Rupees (LKR)  

> [!IMPORTANT]
> **Asking Price Notice**: The platform models seller advertised asking prices from Riyasewana, 
> NOT finalized transaction prices. Predictions represent listed market expectations.

> [!NOTE]
> **Sample Size Limitation**: The current dataset contains **113 ML-eligible records** across 8 vehicle categories. 
> This is an experimental, research-ready evaluation benchmark, NOT a production-validated valuation engine.

---

## 1. Experimental Setup & Partitions
- **Total ML-Eligible Records**: 113
- **Training Set (80%)**: 90 records
- **Holdout Test Set (20%)**: 23 records (completely untouched during model selection)
- **Active Model Features**: 11 features (including derived vehicle age, CC, mileage, categorical one-hot, brand_model interaction)
- **Cross-Validation**: 5-Fold Cross-Validation on training partitions only
- **Model Selection Criterion**: Lowest Mean Cross-Validation MAE on training folds

---

## 2. Model Comparison Table

| Model | Target Transform | Selected | CV MAE (Mean ± Std) | CV RMSE (Mean) | CV R² (Mean) | Test MAE (LKR) | Test RMSE (LKR) | Test R² | Test MedAE (LKR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LinearRegression** | `log1p` | Yes (Best CV) | LKR 0 ± 0 | LKR 1 | 0.8138 | LKR 0 | LKR 0 | 0.9269 | LKR 0 |
| **RandomForestRegressor** | `log1p` | No | LKR 0 ± 0 | LKR 1 | 0.8103 | LKR 0 | LKR 0 | 0.9311 | LKR 0 |
| **HistGradientBoostingRegressor** | `log1p` | No | LKR 1 ± 0 | LKR 1 | 0.6461 | LKR 0 | LKR 0 | 0.8848 | LKR 0 |
| **MedianBaseline** | `log1p` | No | LKR 1 ± 0 | LKR 1 | -0.0523 | LKR 1 | LKR 1 | -0.0242 | LKR 1 |
| **MeanBaseline** | `log1p` | No | LKR 1 ± 0 | LKR 1 | -0.0548 | LKR 1 | LKR 1 | -0.0005 | LKR 1 |

---

## 3. Selected Model Performance: `LinearRegression`
- **Holdout Test MAE**: LKR 0
- **Holdout Test RMSE**: LKR 0
- **Holdout Test R²**: 0.9269
- **Holdout Test Median Absolute Error**: LKR 0

---

## 4. Error Analysis & Diagnostics

### Category Error Breakdown
| Category | Test Count | Median Actual (LKR) | Median Predicted (LKR) | Mean MAE (LKR) | Median Error % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cars | 5 | LKR 16 | LKR 16 | LKR 0 | 1.8% |
| Heavy-Duty | 3 | LKR 15 | LKR 15 | LKR 0 | 0.7% |
| Lorries | 3 | LKR 14 | LKR 14 | LKR 0 | 1.3% |
| Motorbikes | 3 | LKR 12 | LKR 13 | LKR 1 | 4.1% |
| SUVs | 3 | LKR 17 | LKR 16 | LKR 0 | 0.7% |
| Pickups | 2 | LKR 15 | LKR 15 | LKR 0 | 0.2% |
| Three Wheelers | 2 | LKR 13 | LKR 14 | LKR 0 | 1.1% |
| Vans | 2 | LKR 15 | LKR 16 | LKR 0 | 2.8% |

### Price Bracket Error Breakdown
| Price Bracket | Test Count | Mean MAE (LKR) | Median MAE (LKR) | Median Error % |
| :--- | :--- | :--- | :--- | :--- |
| < 2M | 23 | LKR 0 | LKR 0 | 1.3% |
| 2M - 5M | 0 | LKR nan | LKR nan | nan% |
| 5M - 10M | 0 | LKR nan | LKR nan | nan% |
| 10M - 25M | 0 | LKR nan | LKR nan | nan% |
| > 25M | 0 | LKR nan | LKR nan | nan% |

### Top 5 Largest Absolute Errors (Holdout Test Set)
| Listing ID | Category | Make / Model | Age | Mileage | Actual Asking | Predicted | Absolute Error (LKR) | Error % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `12268740` | Cars | Daihatsu Mira L SA3 LTD SAFETY | 2 yrs | 3,255 km | LKR 16 | LKR 17 | LKR 1 | 5.0% |
| `12294837` | Motorbikes | Bajaj N160 | 1 yrs | 10,123 km | LKR 12 | LKR 13 | LKR 1 | 6.3% |
| `12311860` | Cars | Honda N WGN | 1 yrs | 17,063 km | LKR 16 | LKR 16 | LKR 1 | 3.9% |
| `12294409` | Lorries | Isuzu Isuzu | 46 yrs | 355 km | LKR 14 | LKR 14 | LKR 1 | 3.9% |
| `12294924` | Motorbikes | Bajaj Pulsar 135 | 15 yrs | 138,000 km | LKR 12 | LKR 13 | LKR 1 | 4.1% |

---

## 5. Methodological Limitations & Future Collection Roadmap
1. **Small Sample Volume (113 Records)**: With only 11-25 listings per category, statistical power is constrained. Models must be retrained as scheduled collection accumulates 1,000+ observations.
2. **Asking Price Premise**: Listing prices reflect advertised seller demands which typically contain negotiation margins not captured in public online classifieds.
3. **Extreme Luxury / Heavy-Duty Outliers**: High-value commercial and luxury vehicles create large absolute residuals, highlighting the need for category-stratified or category-specific sub-models in future iterations.

### Readiness Verdict
**EXPERIMENTAL / RESEARCH-READY** (Not production-grade).