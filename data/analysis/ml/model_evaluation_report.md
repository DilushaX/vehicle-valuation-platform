# Vehicle Valuation Model Training & Evaluation Report

**Evaluation Date**: 2026-09-17 05:51:20 UTC  
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
| **RandomForestRegressor** | `log1p` | Yes (Best CV) | LKR 4,290,840 ± 3,502,448 | LKR 11,916,938 | 0.5949 | LKR 1,228,489 | LKR 1,916,199 | 0.8959 | LKR 766,519 |
| **LinearRegression** | `log1p` | No | LKR 4,406,413 ± 3,625,003 | LKR 12,078,357 | 0.5764 | LKR 1,439,241 | LKR 2,491,244 | 0.8241 | LKR 374,050 |
| **HistGradientBoostingRegressor** | `log1p` | No | LKR 5,026,914 ± 3,755,265 | LKR 12,829,184 | 0.5029 | LKR 1,740,768 | LKR 2,554,045 | 0.8151 | LKR 949,604 |
| **MedianBaseline** | `log1p` | No | LKR 7,287,272 ± 3,666,484 | LKR 16,186,246 | -0.1490 | LKR 4,292,569 | LKR 6,323,888 | -0.1334 | LKR 3,089,920 |
| **MeanBaseline** | `log1p` | No | LKR 7,291,784 ± 3,650,468 | LKR 16,341,717 | -0.1810 | LKR 4,331,421 | LKR 6,487,456 | -0.1927 | LKR 2,888,264 |

---

## 3. Selected Model Performance: `RandomForestRegressor`
- **Holdout Test MAE**: LKR 1,228,489
- **Holdout Test RMSE**: LKR 1,916,199
- **Holdout Test R²**: 0.8959
- **Holdout Test Median Absolute Error**: LKR 766,519

---

## 4. Error Analysis & Diagnostics

### Category Error Breakdown
| Category | Test Count | Median Actual (LKR) | Median Predicted (LKR) | Mean MAE (LKR) | Median Error % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cars | 5 | LKR 7,850,000 | LKR 9,013,965 | LKR 2,496,411 | 25.3% |
| Heavy-Duty | 3 | LKR 3,600,000 | LKR 3,692,403 | LKR 459,436 | 2.6% |
| Lorries | 3 | LKR 1,975,000 | LKR 2,076,602 | LKR 1,282,805 | 56.4% |
| Motorbikes | 3 | LKR 240,000 | LKR 407,536 | LKR 177,473 | 69.8% |
| SUVs | 3 | LKR 15,500,000 | LKR 12,968,703 | LKR 2,444,827 | 16.3% |
| Pickups | 2 | LKR 4,050,000 | LKR 4,435,012 | LKR 385,012 | 8.6% |
| Three Wheelers | 2 | LKR 716,500 | LKR 743,513 | LKR 123,279 | 18.3% |
| Vans | 2 | LKR 5,012,500 | LKR 5,334,912 | LKR 831,500 | 16.0% |

### Price Bracket Error Breakdown
| Price Bracket | Test Count | Mean MAE (LKR) | Median MAE (LKR) | Median Error % |
| :--- | :--- | :--- | :--- | :--- |
| < 2M | 7 | LKR 298,216 | LKR 150,291 | 65.5% |
| 2M - 5M | 7 | LKR 731,367 | LKR 509,088 | 12.1% |
| 5M - 10M | 4 | LKR 2,615,961 | LKR 1,551,042 | 24.4% |
| 10M - 25M | 5 | LKR 2,116,866 | LKR 2,531,297 | 16.3% |
| > 25M | 0 | LKR nan | LKR nan | nan% |

### Top 5 Largest Absolute Errors (Holdout Test Set)
| Listing ID | Category | Make / Model | Age | Mileage | Actual Asking | Predicted | Absolute Error (LKR) | Error % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `12293566` | Cars | Honda Fit GP5 | 10 yrs | 136,035 km | LKR 8,645,000 | LKR 14,852,849 | LKR 6,207,849 | 71.8% |
| `12336170` | SUVs | BYD Sealion 6 | 1 yrs | 6,000 km | LKR 21,950,000 | LKR 18,161,093 | LKR 3,788,907 | 17.3% |
| `12336155` | Cars | Honda Jade | 11 yrs | 98,500 km | LKR 11,000,000 | LKR 13,781,250 | LKR 2,781,250 | 25.3% |
| `12336121` | Lorries | Ashok-Leyland Leyland | 16 yrs | 12,856 km | LKR 4,500,000 | LKR 1,960,122 | LKR 2,539,878 | 56.4% |
| `12294908` | SUVs | Honda CRV | 8 yrs | 125,000 km | LKR 15,500,000 | LKR 12,968,703 | LKR 2,531,297 | 16.3% |

---

## 5. Methodological Limitations & Future Collection Roadmap
1. **Small Sample Volume (113 Records)**: With only 11-25 listings per category, statistical power is constrained. Models must be retrained as scheduled collection accumulates 1,000+ observations.
2. **Asking Price Premise**: Listing prices reflect advertised seller demands which typically contain negotiation margins not captured in public online classifieds.
3. **Extreme Luxury / Heavy-Duty Outliers**: High-value commercial and luxury vehicles create large absolute residuals, highlighting the need for category-stratified or category-specific sub-models in future iterations.

### Readiness Verdict
**EXPERIMENTAL / RESEARCH-READY** (Not production-grade).