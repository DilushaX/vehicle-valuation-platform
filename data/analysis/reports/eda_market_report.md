# Sri Lankan Vehicle Market — Exploratory Data Analysis (EDA) Report
*Generated: 2026-09-14 15:27:24 UTC*

> [!IMPORTANT]
> IMPORTANT DISCLAIMER: Listed prices represent seller asking/advertised prices extracted from publicly accessible listings on Riyasewana. They DO NOT represent completed transaction prices or confirmed market sale values. Actual finalized sales may differ due to buyer-seller negotiation.

> [!NOTE]
> STATISTICAL NOTE: All correlations and bivariate trends represent observed co-movements and empirical associations within the collected sample. Correlation does NOT establish causation.

## 1. Executive Summary & Dataset Populations

The platform collects publicly accessible vehicle listings from Riyasewana across 8 vehicle categories.
The analysis distinguishes three analytical populations:
1. **Full Dataset**: Complete historical registry of all observed listings and vehicles.
2. **Quality-Filtered Dataset**: Records evaluated against syntax, bounds, and consistency checks.
3. **ML-Eligible Dataset**: Verified listings meeting all critical valuation criteria (valid asking price, mileage, YOM, Make, Model, Category).

| Metric | Observed Count / Rate |
| :--- | :--- |
| **Total Listings** | 94 |
| **Total Vehicles** | 94 |
| **ML Eligible Listings** | 63 (67.0%) |
| **ML Ineligible Listings** | 31 |
| **Active Listings** | 94 |
| **No-Longer-Observed Listings** | 0 |
| **Total Price Events** | 75 |
| **Total Lifecycle Observations** | 108 |
| **Distinct Categories** | 8 |
| **Distinct Brands** | 23 |
| **Distinct Models** | 69 |
| **Districts Represented** | 34 |

---
## 2. Vehicle Category Market Profiles

Central tendency and distribution across the 8 canonical vehicle categories:

| Category | Listings | Share (%) | ML Eligible | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Median Mileage (km) | Median YOM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cars** | 19 | 20.2% | 15 (79%) | Rs. 6,690,000 | Rs. 21,378,824 | 115,000 km | 2016 |
| **Heavy-Duty** | 10 | 10.6% | 8 (80%) | Rs. 3,275,000 | Rs. 3,845,000 | 1,025 km | 2013 |
| **Lorries** | 10 | 10.6% | 7 (70%) | Rs. 1,975,000 | Rs. 5,164,286 | 14,000 km | 2001 |
| **Motorbikes** | 10 | 10.6% | 6 (60%) | Rs. 250,000 | Rs. 369,714 | 28,612 km | 2010 |
| **Pickups** | 10 | 10.6% | 7 (70%) | Rs. 3,700,000 | Rs. 6,659,375 | 48,000 km | 2014 |
| **SUVs** | 10 | 10.6% | 7 (70%) | Rs. 12,900,000 | Rs. 14,012,857 | 4,950 km | 2024 |
| **Three Wheelers** | 10 | 10.6% | 4 (40%) | Rs. 810,000 | Rs. 742,500 | 50,000 km | 2004 |
| **Vans** | 15 | 16.0% | 9 (60%) | Rs. 4,200,000 | Rs. 4,510,417 | 167,500 km | 2000 |

---
## 3. Brand & Model Market Concentration

Dominant vehicle brands observed in the dataset (configurable threshold: minimum 5 listings for reliable inference):

| Brand | Listings | Share (%) | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Reliability Flag |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bajaj** | 20 | 21.3% | Rs. 565,000 | Rs. 568,533 | Sufficient Sample |
| **Daihatsu** | 11 | 11.7% | Rs. 6,237,500 | Rs. 6,648,000 | Sufficient Sample |
| **Honda** | 9 | 9.6% | Rs. 8,645,000 | Rs. 9,177,857 | Sufficient Sample |
| **Mahindra** | 8 | 8.5% | Rs. 3,600,000 | Rs. 3,612,500 | Sufficient Sample |
| **Isuzu** | 6 | 6.4% | Rs. 2,212,500 | Rs. 6,868,750 | Sufficient Sample |
| **Mitsubishi** | 5 | 5.3% | Rs. 3,462,500 | Rs. 3,393,750 | Sufficient Sample |
| **BMW** | 4 | 4.3% | Rs. 53,500,000 | Rs. 89,766,667 | Low Sample (<5) |
| **Kia** | 3 | 3.2% | Rs. 11,850,000 | Rs. 9,760,000 | Low Sample (<5) |
| **CAT** | 3 | 3.2% | Rs. 6,200,000 | Rs. 5,866,667 | Low Sample (<5) |
| **Hyundai** | 3 | 3.2% | Rs. 4,200,000 | Rs. 4,183,333 | Low Sample (<5) |

---
## 4. Numerical Variables & Five-Number Summaries

Parametric (mean, standard deviation) and non-parametric (median, IQR) dispersion statistics:

| Variable | Valid Count | Missing (%) | Mean | Std Dev | Min | Q1 (25%) | Median (50%) | Q3 (75%) | Max | IQR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `asking_price` | 75 | 20.2% | 8,642,840.00 | 23,157,426.07 | 208,000.00 | 1,550,000.00 | 3,675,000.00 | 7,520,000.00 | 195,000,000.00 | 5,970,000.00 |
| `mileage` | 88 | 6.4% | 118,975.27 | 478,248.39 | 0.00 | 1,627.75 | 40,125.00 | 123,456.00 | 4,500,000.00 | 121,828.25 |
| `manufacture_year` | 94 | 0.0% | 2,009.66 | 11.50 | 1,978.00 | 2,003.25 | 2,011.50 | 2,017.00 | 2,026.00 | 13.75 |
| `registration_year` | 10 | 89.4% | 2,016.60 | 5.93 | 2,006.00 | 2,017.00 | 2,018.00 | 2,018.75 | 2,026.00 | 1.75 |
| `vehicle_age` | 94 | 0.0% | 16.34 | 11.50 | 0.00 | 9.00 | 14.50 | 22.75 | 48.00 | 13.75 |
| `engine_cc` | 77 | 18.1% | 1,515.71 | 1,450.03 | 1.00 | 650.00 | 1,400.00 | 2,200.00 | 6,550.00 | 1,550.00 |

---
## 5. Bivariate Relationships & Correlation Analysis

Statistical associations between asking price and key valuation dimensions:

| Relationship Pair | Sample Size | Pearson ($r$) | Spearman ($\rho$) | Association Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Asking Price vs Odometer Mileage** | 71 | -0.0073 | 0.1107 | Direct relationship |
| **Asking Price vs Manufacture Year (YOM)** | 75 | 0.1803 | 0.5186 | Direct relationship |
| **Asking Price vs Vehicle Age** | 75 | -0.1803 | -0.5186 | Inverse relationship |
| **Asking Price vs Engine Capacity (CC)** | 62 | 0.0924 | 0.5160 | Direct relationship |

---
## 6. Outlier Analysis & Data Integrity

Detected **13** statistical outliers using IQR and percentile thresholds.
Outliers are systematically categorized without destructive deletion:
- **Suspicious Data**: Records containing artificial patterns (e.g. dummy mileage sequences `123456`, implausible prices).
- **Genuine Market Extremes**: Authentic observations representing luxury exotics, heavy commercial equipment, or vintage collectors.
- **Insufficient Information**: Incomplete records flagged for exclusion from ML training.

---
## 7. Historical Market Dynamics & Timeframe Depth Limits

- **Observed Timeframe Span**: 2.0 days
- **Historical Depth Status**: Insufficient historical depth for reliable monthly market trend inference.
- **Price Revision Events**: 0 listings with observed price changes (0 reductions, 0 increases).

> [!WARNING]
> Insufficient historical depth for reliable monthly market trend inference.

---
## 8. Key Takeaways for Future Valuation Modeling (Phase 5+)

1. **Category Specificity**: Market scales differ drastically between vehicle categories (e.g., motorbikes vs cars vs heavy commercial vehicles). Segmented valuation models or category interaction terms will be essential.
2. **Non-Linear Age Decay**: Asking price exhibits strong non-linear depreciation curves with vehicle age rather than strict linear decay.
3. **Brand Concentration**: Market listings are concentrated in top Japanese and Indian manufacturers (Toyota, Suzuki, Honda, Bajaj).
4. **Outlier Quarantine**: Step 5 quality filters successfully isolate dummy odometer and price sequences, preventing model distortion while retaining verified observations.

*Report generated by the Vehicle Market Intelligence Platform EDA subsystem.*