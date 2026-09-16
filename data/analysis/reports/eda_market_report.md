# Sri Lankan Vehicle Market — Exploratory Data Analysis (EDA) Report
*Generated: 2026-09-16 16:11:06 UTC*

> [!IMPORTANT]
> IMPORTANT DISCLAIMER: Listed prices represent seller asking/advertised prices extracted from publicly accessible listings on Riyasewana. They DO NOT represent completed transaction prices or confirmed market sale values. Actual finalized sales may differ due to buyer-seller negotiation.

> [!NOTE]
> STATISTICAL NOTE: All correlations and bivariate trends represent observed co-movements and empirical associations within the collected sample. Correlation does NOT establish causation.

## 1. Executive Summary & Dataset Populations

The current dataset represents publicly accessible Riyasewana listings collected within the configured collection scope, rather than a full census of the entire Sri Lankan vehicle market.
The analysis distinguishes three analytical populations:
1. **Full Dataset**: Complete historical registry of all observed listings and vehicles.
2. **Quality-Filtered Dataset**: Records evaluated against syntax, bounds, and consistency checks.
3. **ML-Eligible Dataset**: Verified listings meeting all critical valuation criteria (valid asking price, mileage, YOM, Make, Model, Category).

| Metric | Observed Count / Rate |
| :--- | :--- |
| **Total Listings** | 171 |
| **Total Vehicles** | 171 |
| **ML Eligible Listings** | 113 (66.1%) |
| **ML Ineligible Listings** | 58 |
| **Active Listings** | 171 |
| **No-Longer-Observed Listings** | 0 |
| **Total Price Events** | 135 |
| **Total Lifecycle Observations** | 188 |
| **Distinct Categories** | 8 |
| **Distinct Brands** | 33 |
| **Distinct Models** | 112 |
| **Districts Represented** | 40 |

The asking-price distribution is strongly right-skewed. The mean asking price is substantially higher than the median because a small number of high-priced observations pull the distribution upward. Median asking price is often a more representative measure of central tendency for the current skewed sample. High-priced observations represent genuine market vehicles (such as luxury passenger vehicles and commercial equipment) and are retained in the dataset as authentic market observations rather than deleted.

---
## 2. Vehicle Category Market Profiles

Central tendency and distribution across the 8 canonical vehicle categories:

| Category | Listings | Share (%) | ML Eligible | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Median Mileage (km) | Median YOM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cars** | 29 | 17.0% | 25 (86%) | Rs. 7,850,000 | Rs. 17,946,852 | 111,500 km | 2016 |
| **Heavy-Duty** | 20 | 11.7% | 14 (70%) | Rs. 3,087,500 | Rs. 4,612,969 | 1,244 km | 2015 |
| **Lorries** | 20 | 11.7% | 13 (65%) | Rs. 2,075,000 | Rs. 3,680,000 | 34,691 km | 2003 |
| **Motorbikes** | 20 | 11.7% | 14 (70%) | Rs. 245,000 | Rs. 323,625 | 19,850 km | 2009 |
| **Pickups** | 19 | 11.1% | 12 (63%) | Rs. 4,562,500 | Rs. 6,940,714 | 50,600 km | 2016 |
| **SUVs** | 19 | 11.1% | 13 (68%) | Rs. 13,325,000 | Rs. 15,462,857 | 7,250 km | 2024 |
| **Three Wheelers** | 19 | 11.1% | 11 (58%) | Rs. 730,000 | Rs. 745,375 | 45,000 km | 2003 |
| **Vans** | 25 | 14.6% | 11 (44%) | Rs. 4,150,000 | Rs. 4,145,882 | 136,728 km | 1999 |

Vehicle categories operate at substantially different price scales. Category-level comparisons should therefore be interpreted within category context rather than as a single homogeneous market. Observed category extremes represent a combination of genuine market variation (e.g., heavy industrial machinery vs commuter two-wheelers), distinct operational segments, and records subject to data validation.

Observed geographic and operational dimensions across the sample:
- **Districts**: Colombo has the largest number of observed listings in the current sample. Observed asking-price differences by district should not be interpreted as causal geographic price effects; observed district differences may reflect differences in inventory composition, vehicle categories, brands, models, and sample sizes.
- **Fuel & Transmission**: Petrol is the most frequently observed fuel type in the current sample, and manual listings are more common than automatic listings. These distributions reflect observed listing composition in the sample rather than an intrinsic price premium caused by fuel or transmission type in isolation.

---
## 3. Brand & Model Market Concentration

Dominant vehicle brands observed in the dataset (minimum sample thresholds: 5 listings for brands, 3 listings for models):

| Brand | Listings | Share (%) | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Reliability Flag |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bajaj** | 38 | 22.2% | Rs. 448,000 | Rs. 499,484 | Sufficient Sample |
| **Honda** | 20 | 11.7% | Rs. 10,225,000 | Rs. 11,758,438 | Sufficient Sample |
| **Daihatsu** | 15 | 8.8% | Rs. 6,237,500 | Rs. 6,352,500 | Sufficient Sample |
| **Mahindra** | 15 | 8.8% | Rs. 4,500,000 | Rs. 4,160,909 | Sufficient Sample |
| **Mitsubishi** | 11 | 6.4% | Rs. 3,250,000 | Rs. 3,245,000 | Sufficient Sample |
| **Ashok-Leyland** | 7 | 4.1% | Rs. 2,815,000 | Rs. 3,146,667 | Sufficient Sample |
| **Isuzu** | 6 | 3.5% | Rs. 2,212,500 | Rs. 6,868,750 | Sufficient Sample |
| **Hyundai** | 5 | 2.9% | Rs. 4,175,000 | Rs. 3,862,500 | Sufficient Sample |
| **Mazda** | 5 | 2.9% | Rs. 3,265,000 | Rs. 2,755,000 | Sufficient Sample |
| **BMW** | 4 | 2.3% | Rs. 53,500,000 | Rs. 89,766,667 | Low Sample (<5) |

Where sample sizes fall below minimum reliability thresholds (5 for brands, 3 for models), statistics are flagged as low-sample and should be interpreted with caution. Low-sample records are fully retained in the underlying dataset without exclusion.

---
## 4. Numerical Variables & Five-Number Summaries

Parametric (mean, standard deviation) and non-parametric (median, IQR) dispersion statistics:

| Variable | Valid Count | Missing (%) | Mean | Std Dev | Min | Q1 (25%) | Median (50%) | Q3 (75%) | Max | IQR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `asking_price` | 135 | 21.1% | 7,517,085.19 | 17,891,973.60 | 169,000.00 | 1,175,000.00 | 3,600,000.00 | 8,170,000.00 | 195,000,000.00 | 6,995,000.00 |
| `mileage` | 157 | 8.2% | 92,689.13 | 361,435.21 | 0.00 | 1,244.00 | 40,250.00 | 115,000.00 | 4,500,000.00 | 113,756.00 |
| `manufacture_year` | 171 | 0.0% | 2,009.82 | 11.34 | 1,978.00 | 2,003.00 | 2,011.00 | 2,018.00 | 2,026.00 | 15.00 |
| `registration_year` | 18 | 89.5% | 2,016.00 | 6.74 | 2,002.00 | 2,017.00 | 2,018.00 | 2,019.00 | 2,026.00 | 2.00 |
| `vehicle_age` | 171 | 0.0% | 16.18 | 11.34 | 0.00 | 8.00 | 15.00 | 23.00 | 48.00 | 15.00 |
| `engine_cc` | 146 | 14.6% | 1,540.80 | 1,473.14 | 1.00 | 316.25 | 1,484.00 | 2,200.00 | 6,550.00 | 1,883.75 |

---
## 5. Bivariate Relationships & Correlation Analysis

Statistical associations between asking price and key valuation dimensions:

| Relationship Pair | Sample Size | Pearson ($r$) | Spearman ($\rho$) | Association Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Asking Price vs Odometer Mileage** | 126 | -0.0013 | 0.1620 | Weak monotonic association in isolated sample; confounded by age, category, brand |
| **Asking Price vs Manufacture Year (YOM)** | 135 | 0.2196 | 0.5066 | Moderate direct monotonic association in observed sample |
| **Asking Price vs Vehicle Age** | 135 | -0.2196 | -0.5066 | Moderate inverse monotonic association in observed sample |
| **Asking Price vs Engine Capacity (CC)** | 119 | 0.0920 | 0.5069 | Moderate direct monotonic association across mixed categories |

> [!NOTE]
> **Mathematical Dependency Notice**: Manufacture year and vehicle age are mathematically derived from one another in this dataset (`vehicle_age = current_year - manufacture_year`), producing a perfect inverse correlation (-1.00). This is not an independent market relationship. This identity must be explicitly handled during feature engineering to avoid exact collinearity.

The current sample does not show a strong monotonic association between mileage and asking price when considered in isolation (Pearson $r \approx -0.01$, Spearman $\rho \approx +0.11$). This isolated observation does not imply that mileage has no effect on vehicle price, nor that mileage is useless or should be removed. In a heterogeneous sample, category differences, vehicle age, brand/model prestige, extreme mileage observations, and sample size interact with odometer readings. Feature selection decisions should not be made from bivariate EDA in isolation.

---
## 6. Outlier Analysis & Data Integrity

Detected **24** statistical outliers using IQR and percentile thresholds.
Outliers are systematically categorized without destructive deletion:
- **Suspicious Data**: Records containing artificial patterns (e.g. dummy mileage sequences `123456`, implausible prices) as flagged by Step 5 validation rules.
- **Possible Genuine Market Observations**: Authentic observations representing luxury exotics, heavy commercial equipment, or vintage collectors that naturally sit in market distribution tails.
- **Insufficient Information**: Incomplete records or boundary anomalies flagged for qualification.

Statistical outliers are not automatically classified as 'bad data'. All records are preserved in the underlying registry to maintain uncompromised data integrity.

---
## 7. Historical Market Dynamics & Timeframe Depth Limits

- **Observed Timeframe Span**: 8.0 days
- **Historical Depth Status**: Insufficient historical depth for reliable monthly market trend inference.
- **Price Revision Events**: 0 listings with observed price changes (0 reductions, 0 increases).

No asking-price changes were observed during the current collection window. The approximately two-day observation window is insufficient to infer reliable monthly, seasonal, or long-term market price trends. These data points represent a snapshot of active market listings rather than an established temporal price trend.

> [!WARNING]
> Insufficient historical depth for reliable monthly market trend inference.

---
## 8. Key Takeaways for Future Valuation Modeling (Phase 5+)

1. **Category Specificity**: Market scales differ substantially between vehicle categories (e.g., motorbikes vs cars vs heavy commercial vehicles). Segmented valuation models or category interaction terms will be essential.
2. **Vehicle Age Association**: The observed sample shows a moderate negative monotonic association between vehicle age and asking price. More historical and larger cross-sectional data is required to establish a reliable depreciation pattern. OLS trend lines serve strictly as simple linear trend references rather than true depreciation models.
3. **Brand Concentration**: Market listings are concentrated in top Japanese and Indian manufacturers (Toyota, Suzuki, Honda, Bajaj).
4. **Outlier Quarantine**: Step 5 quality filters successfully isolate dummy odometer and price sequences, preventing model distortion while retaining verified observations.

---
## 9. EDA Interpretation & Limitations

To maintain methodological rigor, subsequent feature engineering (Step 7) and valuation modeling (Phase 5+) must incorporate the following constraints:

1. **Asking Price vs. Transaction Price**: Advertised prices represent seller asking figures on Riyasewana and do not reflect finalized transaction values or negotiated discounts.
2. **Small Current Sample**: The current dataset contains an initial collection volume (94 listings across 8 categories), meaning parameter estimates are preliminary.
3. **Short Historical Timeframe**: The active observation window spans approximately two days, preventing longitudinal, seasonal, or macroeconomic price drift inference.
4. **Category Heterogeneity**: Aggregating diverse categories into unstratified metrics distorts market reality; each canonical category operates on distinct pricing mechanics.
5. **Outlier Leverage**: Genuine luxury high-value records and extreme mileage entries (e.g. ~4.5M km) exert strong leverage on parametric estimators like mean and standard deviation.
6. **Low-Sample Strata**: Brands with fewer than 5 listings and models with fewer than 3 listings carry wide estimation uncertainty and cannot support standalone regression weights.
7. **Correlation Does Not Imply Causation**: Bivariate co-movements (such as district differences or fuel type distributions) reflect sample inventory composition rather than proven causal value drivers.
8. **Mathematical Identity of YOM and Age**: Manufacture year and vehicle age are collinear identities ($age = current\_year - YOM$), producing a perfect $-1.00$ correlation that must not be entered simultaneously into linear modeling without regularization.
9. **Observed Public Listings Scope**: The data captures publicly accessible Riyasewana listings within the scraper's collection scope, not a complete census of the national vehicle fleet.
10. **Absence of Confirmed Sold Prices**: The platform monitors listing lifecycle removals and asking price revisions, but has no visibility into finalized cash settlements.

*Report generated by the Vehicle Market Intelligence Platform EDA subsystem.*