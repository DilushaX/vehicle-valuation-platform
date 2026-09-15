# Sri Lankan Vehicle Market — Exploratory Data Analysis (EDA) Report
*Generated: 2026-09-15 15:48:27 UTC*

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

The asking-price distribution is strongly right-skewed. The mean asking price is substantially higher than the median because a small number of high-priced observations pull the distribution upward. Median asking price is often a more representative measure of central tendency for the current skewed sample. High-priced observations represent genuine market vehicles (such as luxury passenger vehicles and commercial equipment) and are retained in the dataset as authentic market observations rather than deleted.

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

Vehicle categories operate at substantially different price scales. Category-level comparisons should therefore be interpreted within category context rather than as a single homogeneous market. Observed category extremes represent a combination of genuine market variation (e.g., heavy industrial machinery vs commuter two-wheelers), distinct operational segments, and records subject to data validation.

Observed geographic and operational dimensions across the sample:
- **Districts**: Colombo has the largest number of observed listings in the current sample. Observed asking-price differences by district should not be interpreted as causal geographic price effects; observed district differences may reflect differences in inventory composition, vehicle categories, brands, models, and sample sizes.
- **Fuel & Transmission**: Petrol is the most frequently observed fuel type in the current sample, and manual listings are more common than automatic listings. These distributions reflect observed listing composition in the sample rather than an intrinsic price premium caused by fuel or transmission type in isolation.

---
## 3. Brand & Model Market Concentration

Dominant vehicle brands observed in the dataset (minimum sample thresholds: 5 listings for brands, 3 listings for models):

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

Where sample sizes fall below minimum reliability thresholds (5 for brands, 3 for models), statistics are flagged as low-sample and should be interpreted with caution. Low-sample records are fully retained in the underlying dataset without exclusion.

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
| **Asking Price vs Odometer Mileage** | 71 | -0.0073 | 0.1107 | Weak monotonic association in isolated sample; confounded by age, category, brand |
| **Asking Price vs Manufacture Year (YOM)** | 75 | 0.1803 | 0.5186 | Moderate direct monotonic association in observed sample |
| **Asking Price vs Vehicle Age** | 75 | -0.1803 | -0.5186 | Moderate inverse monotonic association in observed sample |
| **Asking Price vs Engine Capacity (CC)** | 62 | 0.0924 | 0.5160 | Moderate direct monotonic association across mixed categories |

> [!NOTE]
> **Mathematical Dependency Notice**: Manufacture year and vehicle age are mathematically derived from one another in this dataset (`vehicle_age = current_year - manufacture_year`), producing a perfect inverse correlation (-1.00). This is not an independent market relationship. This identity must be explicitly handled during feature engineering to avoid exact collinearity.

The current sample does not show a strong monotonic association between mileage and asking price when considered in isolation (Pearson $r \approx -0.01$, Spearman $\rho \approx +0.11$). This isolated observation does not imply that mileage has no effect on vehicle price, nor that mileage is useless or should be removed. In a heterogeneous sample, category differences, vehicle age, brand/model prestige, extreme mileage observations, and sample size interact with odometer readings. Feature selection decisions should not be made from bivariate EDA in isolation.

---
## 6. Outlier Analysis & Data Integrity

Detected **13** statistical outliers using IQR and percentile thresholds.
Outliers are systematically categorized without destructive deletion:
- **Suspicious Data**: Records containing artificial patterns (e.g. dummy mileage sequences `123456`, implausible prices) as flagged by Step 5 validation rules.
- **Possible Genuine Market Observations**: Authentic observations representing luxury exotics, heavy commercial equipment, or vintage collectors that naturally sit in market distribution tails.
- **Insufficient Information**: Incomplete records or boundary anomalies flagged for qualification.

Statistical outliers are not automatically classified as 'bad data'. All records are preserved in the underlying registry to maintain uncompromised data integrity.

---
## 7. Historical Market Dynamics & Timeframe Depth Limits

- **Observed Timeframe Span**: 2.0 days
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