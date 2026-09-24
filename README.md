# 🚗 Sri Lankan Vehicle Market Intelligence & Valuation Platform

An end-to-end data-driven market intelligence and explainable machine learning valuation platform for Sri Lankan used vehicles.

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
   - Category-specific regression models (e.g. Random Forest with log target transform).
   - Outputs: **Estimated Market Asking Price**, **Indicative Model-Based Prediction Range**, and **Factor Attribution (Tree SHAP)**.
   - Robust fallback mechanisms for sparse categories.
   - **SHAP TreeExplainer** feature contribution breakdown in Sri Lankan Rupees.
   - Explicit disclaimers regarding unobserved physical conditions and asking price scope.

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

## 🛡️ Production Market Data Collection (Phase 4)

The platform includes a robust, polite, and production-ready data collection engine designed for long-term historical tracking of the Sri Lankan automotive market (primarily **Riyasewana**).

### Collection Commands

```bash
# Dry run: Inspect discovery and parse without modifying PostgreSQL or exporting CSV
python scripts/scrape_riyasewana.py --category Cars --max-pages 1 --max-listings 5 --dry-run

# Controlled live collection for specific categories:
python scripts/scrape_riyasewana.py --category Cars --max-pages 5 --max-listings 100

# Multi-category collection (space or comma-separated):
python scripts/scrape_riyasewana.py --category Cars SUVs Vans --max-pages 2 --max-listings 20
python scripts/scrape_riyasewana.py --category Cars,SUVs,Vans --max-pages 2 --max-listings 20

# Run all 8 discovered categories with polite rate-limiting:
python scripts/scrape_riyasewana.py --max-pages 2 --max-listings 50 --request-delay 1.5
```

### Volume Controls & Safeguards

- **`--max-pages <N>`**: Caps the maximum number of pagination pages inspected per category. Pagination traversal automatically tracks visited URLs and stops cleanly if cyclical pagination or missing links are encountered.
- **`--max-listings <N>`**: Caps the maximum number of unique listing detail pages fetched per category. If early pagination pages satisfy the requested listing volume, subsequent pagination discovery terminates early to prevent redundant HTTP requests.
- **`--dry-run`**: Performs complete category discovery and detail parsing in memory, displaying a full audit report while guaranteeing **zero mutations** to PostgreSQL tables, observations, or price histories.

### Request Rate & Retry Policy

- **Rate-Limiting**: Strictly sequential requests with a default polite delay of **`1.5 seconds`** (`--request-delay 1.5`). Bursts, aggressive concurrency, and parallel scraping are prohibited.
- **Retry Logic & Backoff**: Transient errors (connection resets, read timeouts, HTTP 5xx, HTTP 429) are retried with exponential backoff (default: 3 retries, 2.0x backoff). Non-transient errors (HTTP 400, 403, 404, 410) fail immediately without retrying.
- **HTTP 429 & `Retry-After`**: If upstream signals `Retry-After <= 30.0s`, the crawler respects the requested wait time. If `Retry-After > 30.0s`, retries safely abort to prevent blocking execution.

### Incremental Collection & Listing Lifecycle (Phase 4 Step 3)

The platform implements an audit-safe, incremental lifecycle engine that manages listing states while strictly preserving historical integrity in PostgreSQL:

#### 1. NEW vs. EXISTING Listing Detection
Every discovered listing is evaluated against PostgreSQL by `(source, listing_id)`:
- **NEW Listing**: Full detail scrape is executed. Creates new `Vehicle`, `Listing`, initial `PriceHistory` (if price present), and initial `ListingObservation`.
- **EXISTING Listing**: Full detail scrape is executed. Preserves original primary key and `first_seen_at`, updates `last_seen_at`, refreshes vehicle attributes, and records a new point-in-time `ListingObservation`.

#### 2. Price History & Observation Behavior
- **Point-in-Time Observations**: Every successful observation across runs records a row in `listing_observations` linked to the active `ScrapeRun`.
- **Price Change Audit**: If the seller's asking price has not changed, **no duplicate row** is created in `price_history`. A new `PriceHistory` row is recorded **only** when the price changes, chronologically preserving all previous price points.

#### 3. Scope-Aware Disappearance Protection
> [!IMPORTANT]
> **Definitive Disappearance Rule**:
> **"Listing disappearance is only inferred from a sufficiently complete collection scope. A listing missing from a limited/partial scrape is not considered disappeared."**

- A collection run is considered **complete scope** for a category **only** if:
  1. The crawl was not constrained by `--max-pages` or `--max-listings` before pagination reached its natural end (`pagination_exhausted == True`).
  2. All pagination pages were scraped with zero failures (`failed_pages == 0`).
  3. All listing detail pages were scraped with zero failures (`failed_listings == 0`).
  4. Spider run status is `COMPLETED`.
- **Why a partial scrape cannot determine disappearance**: If a user runs `--category Cars --max-pages 1 --max-listings 5`, only 5 listings are observed out of thousands. Missing from those 5 listings is **not** evidence of disappearance. In all partial/scoped runs, disappearance detection is **automatically skipped**, preserving all active listings in `ACTIVE` status without false transitions.
- **Never Infer Sold**: When a genuinely complete collection confirms a listing is missing, it is transitioned to **`NO_LONGER_OBSERVED`**, never `SOLD`.

#### 4. Reappearance Handling
- If a listing previously marked `NO_LONGER_OBSERVED` is observed again on Riyasewana:
  - Status transitions back: `NO_LONGER_OBSERVED` → `ACTIVE`.
  - The original listing identity, database primary key, and immutable `first_seen_at` are preserved.
  - `last_seen_at` is updated to the new observation timestamp.
  - A new `ListingObservation` is recorded.
  - A new `PriceHistory` row is recorded only if the price changed since its last observation.
  - Exactly **one** `Listing` record exists—no duplicates are ever created.

#### 5. Multi-Category Isolation
- Lifecycle updates and disappearance transitions are strictly scoped per category. A collection for `Cars` will never affect listings belonging to `Vans`, `SUVs`, or other categories.

> [!WARNING]
> **Scoped Completeness vs. Website Completeness**: Scrape reports and metrics reflect completeness **within the requested collection scope** (`max_pages`, `max_listings`). A status of `COMPLETED` confirms that all requested pages and attempted listings succeeded without errors; it does **not** claim to have scraped the entirety of Riyasewana.

### Automated Scheduling & Long-Term Operational Pipeline (Phase 4 Step 4)

To transition from ad-hoc manual runs into a repeatable, automated system that continuously builds the historical Sri Lankan vehicle market dataset, the platform includes a dedicated scheduled collection pipeline.

> [!IMPORTANT]
> **Long-Term Market Dataset Rule**:
> **"Scheduled collection is designed to continuously accumulate historical market data. A listing disappearing from a failed or partial run is not considered disappeared."**

#### 1. Scheduling Mechanism & Platform Selection
- **macOS Native LaunchAgent (`launchd`)**:
  - The primary scheduling mechanism on macOS is a LaunchAgent property list installed at `~/Library/LaunchAgents/com.vehicle_valuation.collection.plist`.
  - *Rationale*: Unlike legacy `cron`, `launchd` operates seamlessly under modern macOS security (TCC privacy controls), handles system sleep/wake calendar intervals reliably (`StartCalendarInterval`), and redirects stdout/stderr into dedicated log files (`logs/scheduled_collection.log` and `logs/scheduled_collection_error.log`).
- **Standard POSIX Crontab Alternative**:
  - For standard Linux servers or environments preferring cron, standard 5-field cron entries are fully supported:
    ```bash
    # Run daily at 02:00 AM Sri Lanka / local time
    0 2 * * * cd /path/to/vehicle-valuation-platform && PYTHONPATH=. .venv/bin/python scripts/scheduled_collection.py >> logs/scheduled_collection.log 2>&1
    ```

#### 2. Configuration & Schedule Time
Configure daily collection timing in your `.env` file or environment variables:
```env
COLLECTION_SCHEDULE=daily
COLLECTION_TIME=02:00
```
- `COLLECTION_SCHEDULE`: Execution frequency (default: `daily`).
- `COLLECTION_TIME`: 24-hour time format `HH:MM` (default: `02:00` AM).

#### 3. Management CLI Commands
The scheduled collection pipeline provides dedicated CLI commands for complete lifecycle management:

```bash
# Check current LaunchAgent installation and active daemon status
python scripts/scheduled_collection.py --status-launchd

# Install and activate the LaunchAgent plist (runs daily at configured COLLECTION_TIME)
python scripts/scheduled_collection.py --install-launchd

# Install with a custom time override (e.g., 03:30 AM)
python scripts/scheduled_collection.py --install-launchd --time 03:30

# Generate and view LaunchAgent XML plist without installing
python scripts/scheduled_collection.py --generate-plist

# Unload and remove the LaunchAgent from ~/Library/LaunchAgents
python scripts/scheduled_collection.py --uninstall-launchd
```

#### 4. Manual Execution & Verification
Operators can trigger on-demand runs using the same entry point:

```bash
# Test run in Dry-Run mode across Cars (no PostgreSQL mutations, no CSV export)
python scripts/scheduled_collection.py --dry-run --category Cars --max-pages 1 --max-listings 5

# Test run across all 8 canonical categories in dry-run mode
python scripts/scheduled_collection.py --dry-run --max-pages 1 --max-listings 5

# Execute live collection across all 8 categories sequentially
python scripts/scheduled_collection.py
```

#### 5. Non-Overlapping Process Locking
To eliminate race conditions and avoid double-scraping if a scheduled run exceeds 24 hours or overlaps with a manual run:
- A non-blocking filesystem lock (`data/.collection.lock`) is acquired via `fcntl.flock(LOCK_EX | LOCK_NB)`.
- If another collection process is already running, the new process detects the lock, logs a safe notice displaying the holding process PID and start time, and exits cleanly with return code `0` (preventing scheduler error cascades).
- The lock file is atomically unlinked upon process termination or unhandled exceptions, and the OS kernel automatically releases the `flock` descriptor if a process terminates unexpectedly.

#### 6. Category Failure Isolation
- The 8 canonical categories (`Cars`, `Heavy-Duty`, `Lorries`, `Motorbikes`, `Pickups`, `SUVs`, `Three Wheelers`, `Vans`) are processed **strictly sequentially**.
- If one category experiences an upstream network glitch, parsing error, or HTTP failure, it is marked as `FAILED` and recorded in `ScrapeRun`.
- The pipeline **isolates the failure and continues processing the remaining categories**, ensuring that temporary issues in one vehicle class do not prevent data collection for others.

#### 7. Operational Logging & Telemetry
Every scheduled run logs comprehensive metrics to stdout, `logs/scheduled_collection.log`, and PostgreSQL `scrape_runs`:
- Start timestamp and duration
- Per-category outcome (`COMPLETED`, `FAILED`, or `INCOMPLETE`)
- Discovered and scraped pagination pages
- Discovered and scraped listing URLs
- Inserted `new_listings` and updated `existing_listings`
- Created `listing_observations`
- Detected `price_changes`
- `reactivated_listings` (`NO_LONGER_OBSERVED` → `ACTIVE`)
- `disappeared_listings` (`ACTIVE` → `NO_LONGER_OBSERVED` — only when scope is complete)
- Failure reasons and stack traces if any
- **Zero credential leaks**: Database connection strings, passwords, and sensitive environment variables are strictly withheld from logs.

### Data Quality & Historical Dataset Readiness (Phase 4 Step 5)

To prepare the accumulated historical vehicle dataset for future Exploratory Data Analysis (EDA) and Machine Learning (ML) valuation modeling, the platform includes a comprehensive data quality assurance and dataset readiness framework.

> [!IMPORTANT]
> **Fundamental Market Valuation Rule**:
> **"Listing prices observed on Riyasewana are seller asking prices, NOT confirmed transaction or settlement prices. The ML system estimates market asking value and range. Asking price must never be described as actual transaction or sold price."**

#### 1. Data Quality Rules & Issue Codes
Every listing is evaluated against a structured data quality rulebook without destructive deletion:

| Dimension | Issue Code | Severity | Description |
| :--- | :--- | :--- | :--- |
| **Price** | `missing_price` | **CRITICAL** | Asking price is absent or empty. |
| | `invalid_price` | **CRITICAL** | Price is zero or negative. |
| | `price_out_of_range` | **CRITICAL** | Price is outside plausible category boundaries (e.g. Motorbike > 25M, Car > 350M). |
| | `suspicious_price_pattern` | **CRITICAL** | Obvious dummy sequence (e.g. `111111`, `123456`, `999999`). |
| **Mileage** | `missing_mileage` | **CRITICAL** | Odometer mileage is absent for motorized road vehicles. |
| | `invalid_mileage` | **CRITICAL** | Odometer mileage is negative. |
| | `mileage_out_of_range` | **CRITICAL** | Mileage exceeds physical plausible limit (> 1,500,000 km). |
| | `suspicious_mileage_pattern` | **CRITICAL** | Obvious dummy sequence (e.g. `111111`, `123456`, `222222`). |
| **Year (YOM)** | `missing_yom` | **CRITICAL** | Year of manufacture is missing (unanchored depreciation). |
| | `invalid_yom` | **CRITICAL** | Year of manufacture is prior to 1950 historical threshold. |
| | `future_yom` | **CRITICAL** | Year of manufacture is in the future (> current calendar year). |
| **Year (YOR)** | `missing_yor` | NON-CRITICAL | Registration year absent (brand new / unregistered vehicles). |
| | `invalid_yor` | NON-CRITICAL | Registration year prior to 1950. |
| | `future_yor` | NON-CRITICAL | Registration year in the future. |
| **Relationship** | `registration_before_manufacture` | NON-CRITICAL | Clerical paperwork anomaly (`YOR < YOM`). Flagged, not deleted. |
| **Engine CC** | `invalid_engine_cc` | NON-CRITICAL | Engine CC is negative or zero on non-EV vehicle. |
| | `implausible_engine_cc` | NON-CRITICAL | Engine CC < 25 cc (unless Electric) or > 16,000 cc. |
| **Category** | `missing_category` | **CRITICAL** | Category is missing. |
| | `invalid_category` | **CRITICAL** | Category not in the 8 canonical vehicle classes. |
| **Identity** | `missing_brand` / `missing_model` | **CRITICAL** | Vehicle make or model is absent. |

#### 2. ML Training Eligibility (`ml_eligible` vs `is_valid`)
The platform explicitly decouples **data cleanliness** from **valuation model usability**:
- **`is_valid`**: `True` only when a listing has **zero** validation issues of any kind.
- **`ml_eligible`**: `True` when a listing has **zero CRITICAL** issues.
  - Non-critical issues (e.g. missing optional `engine_cc`, unlisted `registration_year`, or `registration_before_manufacture` paperwork anomalies) do **not** disqualify a listing from valuation training, provided its core valuation features (asking price, YOM, mileage, brand, model, category) are reliable.
  - Ineligible listings are preserved in PostgreSQL with human-readable `ml_exclusion_reasons` for auditing.

#### 3. Attribute Normalization & Canonicalization
- **Fuel Type**: Standardizes variations (`gasoline` → `Petrol`, `super diesel` → `Diesel`, `phev` → `Hybrid`, `ev` → `Electric`) while preserving the exact raw text in `raw_data`.
- **Transmission**: Standardizes variations (`auto`, `cvt`, `tiptronic` → `Automatic`; `mt` → `Manual`).
- **Category**: Canonicalizes plural and singular variants to the 8 canonical classes (`Cars`, `Heavy-Duty`, `Lorries`, `Motorbikes`, `Pickups`, `SUVs`, `Three Wheelers`, `Vans`).

#### 4. Historical Dataset Consistency Auditing
The `DatasetQualityEngine` automatically audits 11 historical integrity invariants:
1. **Duplicate Listings**: Identical `(source, listing_id)`.
2. **Duplicate URLs**: Repeated listing URLs.
3. **Duplicate Observations**: Multiple observations for the same listing within the same `ScrapeRun`.
4. **Consecutive Duplicate Prices**: Redundant identical price points in `PriceHistory`.
5. **Observation Timestamps**: `observed_at >= first_seen_at`.
6. **Timestamp Ordering**: `first_seen_at <= last_seen_at`.
7. **Invalid Lifecycle Statuses**: Status values outside `ACTIVE` or `NO_LONGER_OBSERVED`.
8. **Orphan Price History**: Price records referencing missing listing IDs.
9. **Orphan Observations**: Observations referencing missing listings or scrape runs.
10. **Orphan Vehicles**: Vehicle rows without any associated listings.
11. **Broken Vehicle References**: Listing rows with non-existent `vehicle_id`.

#### 5. Data Quality Reporting CLI
Execute quality audits and consistency checks directly from the command line:

```bash
# Run full data quality & consistency report against live PostgreSQL
python scripts/data_quality_report.py

# Run report as a module with consistency check validation
python -m scripts.data_quality_report --check-consistency

# Filter quality report to a specific category
python scripts/data_quality_report.py --category Cars

# Export machine-readable JSON quality audit
python scripts/data_quality_report.py --json

# Fail with non-zero exit code if any consistency errors exist (CI/CD pipeline check)
python scripts/data_quality_report.py --fail-on-inconsistency
```

#### Example Output:
```
============================================================================
 VEHICLE VALUATION PLATFORM — DATA QUALITY & READINESS REPORT
 Phase 4 Step 5: Historical Dataset Quality & Integrity Audit
============================================================================

1. GLOBAL DATASET INVENTORY
----------------------------------------------------------------------------
  Total Vehicle Entities    : 94
  Total Market Listings     : 94
  Total Price History Events: 75
  Total Observations        : 108
  Total Scrape Runs Tracked : 15
  Status Breakdown          : ACTIVE=94, NO_LONGER_OBSERVED=0

2. ML VALUATION TRAINING ELIGIBILITY
----------------------------------------------------------------------------
  ML Eligible Listings      : 63 (67.02%)
  ML Ineligible Listings    : 31
  * Note: Ineligible listings are preserved with flags and excluded from training.

3. CATEGORY ELIGIBILITY BREAKDOWN
----------------------------------------------------------------------------
  Category         Total    Eligible   Ineligible   Rate %   Active  
  ---------------- -------- ---------- ------------ -------- --------
  Cars             19       15         4            78.95%   19      
  Heavy-Duty       10       8          2            80.0%    10      
  Lorries          10       7          3            70.0%    10      
  Motorbikes       10       6          4            60.0%    10      
  Pickups          10       7          3            70.0%    10      
  SUVs             10       7          3            70.0%    10      
  Three Wheelers   10       4          6            40.0%    10      
  Vans             15       9          6            60.0%    15      

4. DATA QUALITY ISSUES AUDIT
----------------------------------------------------------------------------
  [CRITICAL ISSUES — Excludes from ML Training]
    - missing_price                   : 19 listings
    - suspicious_mileage_pattern      : 7 listings
    - missing_mileage                 : 6 listings
    - suspicious_mileage              : 1 listings

  [NON-CRITICAL ISSUES — Audited & Preserved, Training Eligible]
    None detected.

5. HISTORICAL LIFECYCLE CONSISTENCY AUDIT
----------------------------------------------------------------------------
  Overall Consistency Status: PASSED (Zero Anomalies)
    [OK]           Duplicate Listings
    [OK]           Duplicate Listing Urls
    [OK]           Duplicate Observations Same Run
    [OK]           Consecutive Duplicate Prices
    [OK]           First Seen After Last Seen
    [OK]           Observation Before First Seen
    [OK]           Invalid Lifecycle Statuses
    [OK]           Orphan Price Histories
    [OK]           Orphan Observations
    [OK]           Orphan Vehicles
    [OK]           Broken Vehicle References

============================================================================
 BUSINESS & LEGAL NOTICE:
 - Asking prices observed on Riyasewana are seller asking prices, NOT confirmed
   transaction or sold prices.
 - Inactive listings are classified as NO_LONGER_OBSERVED and never marked as SOLD.
 - Quality analysis performs auditing and classification without destructive deletion.
============================================================================
```

---

### 6. Exploratory Data Analysis (EDA) & Market Intelligence (Phase 4 Step 6)

The **Exploratory Data Analysis (EDA)** engine provides reproducible, non-destructive analytical pipelines to understand empirical distributions, structural relationships, price spreads, category divergence, and data limitations across the collected Sri Lankan vehicle market data.

> [!IMPORTANT]
> **Asking Price Disclaimer**: Prices analyzed throughout the platform represent seller asking/listed prices observed on Riyasewana. They do **not** represent confirmed transaction prices or finalized sale contracts.

> [!NOTE]
> **Correlation vs. Causation**: Observed statistical associations (e.g. price vs. mileage or district variations) measure empirical co-movement and do **not** imply causality.

#### 1. CLI Execution & Reproducible Reporting
Execute the full EDA pipeline against the production PostgreSQL database:

```bash
# Execute standard EDA runner (loads PostgreSQL, generates figures, tables, and report)
python scripts/run_eda.py

# Alternatively, execute via module syntax
python -m eda.report

# Optional category-specific EDA
python scripts/run_eda.py --category Cars

# Run analysis restricted to verified ML-eligible listings
python scripts/run_eda.py --ml-eligible-only

# Custom output directory
python scripts/run_eda.py --output-dir data/analysis
```

#### 2. Generated Artifacts & Directory Structure
Generated artifacts are isolated in `data/analysis/` (never placed in `data/raw/`):

```
data/analysis/
├── figures/
│   ├── category_price_comparison.png       # Boxplot of asking prices across all 8 categories
│   ├── correlation_matrix.png              # Spearman rank correlation heatmap
│   ├── district_distribution.png           # Listing volume by administrative district
│   ├── fuel_transmission_distribution.png  # Fuel type & transmission distribution bar charts
│   ├── mileage_distribution.png            # Operational range histogram & log-scale dispersion boxplot
│   ├── price_distribution.png              # Linear histogram & log-scale boxplot
│   ├── price_vs_age.png                    # Scatter plot with simple linear trend reference
│   ├── price_vs_mileage.png                # Dual-panel scatter (full dataset with extreme tail + analytical operational range)
│   └── yom_age_distribution.png            # Manufacture year and vehicle age histograms
├── tables/
│   ├── category_summary.csv / .json        # Median/mean price, mileage, YOM across 8 categories
│   ├── brand_summary.csv / .json           # Top brands, market share, asking prices, sample flags
│   ├── model_summary.csv / .json           # Model-level statistics, spreads, and sample reliability
│   ├── numerical_summary.csv / .json       # Parametric & 5-number non-parametric statistics
│   ├── fuel_summary.csv / .json            # Fuel breakdown & median prices
│   ├── transmission_summary.csv / .json    # Automatic vs Manual proportions & prices
│   ├── district_summary.csv / .json        # Geographic distribution (non-causal observations)
│   ├── correlations.json                   # Pearson & Spearman matrices, bivariate diagnostics
│   ├── flagged_outliers.csv / .json        # IQR/percentile outliers with categorization
│   └── dataset_overview.json               # Top-level entity, listing, and attribute counts
└── reports/
    └── eda_market_report.md                # Comprehensive Markdown market intelligence report
```

#### 3. Analytical Populations
The EDA pipeline explicitly distinguishes three operational populations:
- **Full Dataset**: Total historical corpus (94 listings, 94 vehicles, 108 observations).
- **Quality-Filtered Dataset**: Records audited for data integrity, field syntax, and plausible physical limits.
- **ML-Eligible Dataset**: Verified listings meeting all critical valuation criteria (valid asking price, mileage, manufacture year, brand, model, and category) suitable for future valuation modeling (63 listings, 67.0% eligibility rate).

#### 4. Sample Size Protections & Historical Depth Limits
- **Sample Size Safeguards**: Configurable minimum sample thresholds (`MIN_BRAND_SAMPLE = 5`, `MIN_MODEL_SAMPLE = 3`, `MIN_DISTRICT_SAMPLE = 3`) flag low-volume groups as `Low Sample` to prevent misleading rankings while retaining records.
- **Extreme-Value Handling**: Extreme observations (such as 4.5M km mileage or high-value luxury exotics) are fully retained in the underlying registry to maintain uncompromised data integrity. Visualizations provide both full-data and analytical views.
- **Historical Depth Enforcement**: The system strictly checks the observation time span. Because a multi-day scrape window does not constitute a long-term trend, the historical engine explicitly reports:
  > *"Insufficient historical depth for reliable monthly market trend inference."*
  Fabricated or extrapolated monthly trends are strictly prevented.

---

### 7. Feature Engineering & ML-Ready Feature Preparation (Phase 5 Step 7)

The **Feature Engineering and ML-Ready Feature Preparation** subsystem (`feature_engineering/`) transforms audited market listings into scikit-learn compatible, leakage-safe training features for valuation models.

> [!IMPORTANT]
> **Asking Price Limitation**: The platform predicts observed seller asking prices, not confirmed transaction prices.
> Listing prices reflect seller advertised expectations on Riyasewana and do not account for unrecorded offline negotiation discounts or settlement concessions.

> [!NOTE]
> **Dataset Size Limitation**: The current verified dataset (63 ML-eligible listings across 8 categories) is still small for reliable production model training. It serves as an initial foundation and should continue growing through scheduled data collection.

#### 1. Core Principles & Architecture
- **Read-Only Database Guarantee**: Operates strictly via read-only queries. PostgreSQL remains the single source of truth and is never modified during dataset preparation.
- **ML Eligibility Enforcement**: Loads listings with `ml_eligible = True` by default. Ineligible records (31 listings) remain securely stored in PostgreSQL.
- **Target Separation**: The target variable $y$ (`asking_price`) is strictly decoupled from the feature matrix $X$. Supports both `"raw"` (default) and `"log1p"` transformations.
- **Leakage Prevention**: An automated `LeakageValidator` actively inspects features and blocks target variables, lifecycle outcomes (`current_status`), temporal observations, and private seller contact details from entering $X$.
- **Unfitted Preprocessor Architecture**: Preprocessing components (`ColumnTransformer`, `RareCategoryGrouper`, `OneHotEncoder`) are created unfitted. They are fitted exclusively on training splits during future model training, preventing data leakage across test splits.

#### 2. Feature Policy & Derivations

| Feature | Type | Source | Policy / Transformation | Default | Collinearity / Leakage Mitigation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vehicle_age` | Numerical | `manufacture_year` | `reference_year (2026) - manufacture_year` | **Yes** | Replaces `manufacture_year` to eliminate exact collinearity ($\rho = -1.00$). Uses explicit reference year to prevent temporal leakage. |
| `manufacture_year` | Numerical | `vehicles.manufacture_year` | Retained in metadata only | No | Excluded from active features due to perfect collinearity with `vehicle_age`. Preserved in metadata for full traceability. |
| `mileage` | Numerical | `listing_observations.observed_mileage` | Median imputation (natural scale) | **Yes** | Supports optional `log1p(mileage)` transformation via `--mileage-transform log1p` for high-skewness models. |
| `engine_cc` | Numerical | `vehicles.engine_cc` | Median imputation | **Yes** | Never imputed with zero; missing values filled with median during training fit. |
| `registration_year` | Numerical | `vehicles.registration_year` | Optional median imputation + missing indicator | No | Excluded by default due to high missingness (~70%). Available via `--include-registration-year`. |
| `category` | Categorical | `vehicles.category` | Canonical string → OneHotEncoder | **Yes** | Retained across all 8 canonical vehicle classes to support global and category-specific architectures. |
| `brand` | Categorical | `vehicles.brand` | Rare grouping (`min_freq=5`) → OneHotEncoder | **Yes** | Infrequent brands mapped to `"Other"`. Rare luxury/exotic records are retained, not deleted. |
| `model` | Categorical | `vehicles.model` | Rare grouping (`min_freq=5`) → OneHotEncoder | **Yes** | Low-frequency models mapped to `"Other"` to avoid high-cardinality overfitting. |
| `brand_model` | Categorical | `brand + "_" + model` | Concatenation → rare grouping → OneHotEncoder | **Yes** | Captures domain hierarchy (e.g. Toyota Corolla vs Toyota Prado). |
| `fuel_type` | Categorical | `vehicles.fuel_type` | Standardized string → OneHotEncoder | **Yes** | Normalized to Petrol, Diesel, Hybrid, Electric. |
| `transmission` | Categorical | `vehicles.transmission` | Standardized string → OneHotEncoder | **Yes** | Normalized to Automatic, Manual, Tiptronic. |
| `district` | Categorical | `listings.district` | String → OneHotEncoder | **Yes** | Administrative district of vehicle location. Unknown categories handled safely via `handle_unknown="ignore"`. |
| `condition` | Categorical | `vehicles.condition` | String → OneHotEncoder | **Yes** | Registered (Used), Unregistered, Brand New. |

#### 3. CLI Execution & Dataset Export
Generate reproducible ML-ready artifacts:

```bash
# Prepare standard ML dataset with default parameters (raw target, vehicle_age, no mileage log)
python scripts/prepare_ml_dataset.py

# Prepare dataset with log1p target and log1p mileage transformation
python scripts/prepare_ml_dataset.py --target-transform log1p --mileage-transform log1p

# Include registration year and customize rare category grouping threshold
python scripts/prepare_ml_dataset.py --include-registration-year --min-frequency 3

# Specify custom artifact export directory
python scripts/prepare_ml_dataset.py --output-dir data/analysis/ml
```

#### 4. Output Artifacts (`data/analysis/ml/`)
- `ml_dataset.csv`: Combined ML-ready dataset containing traceability IDs (`listing_id`, `vehicle_id`, `manufacture_year`), active engineered features ($X$), and target ($y$).
- `feature_schema.json`: Machine-readable metadata schema defining data types, sources, transformations, and leakage classifications for every feature.
- `dataset_summary.json`: Statistical profile including record counts, category breakdown, target quantiles, and configuration parameters.

---

### 8. Model Training & Evaluation (Phase 5 Step 8)

The **Model Training & Evaluation** subsystem (`ml/`) implements a leakage-safe regression benchmark and inference interface for Sri Lankan vehicle asking-price prediction.

> [!IMPORTANT]
> **Observed Asking Price Disclaimer**: The target variable is `asking_price` — the seller's advertised price on Riyasewana.
> It does **NOT** represent confirmed transaction or settlement price. Predictions estimate expected advertised asking prices.

> [!NOTE]
> **Sample Size & Research Benchmark Status**: With 113 ML-eligible listings across 8 vehicle categories, all models are designated as **EXPERIMENTAL RESEARCH BENCHMARKS**.
> They validate mathematical rigor, pipeline isolation, and leakage safety, but must **NOT** be used as a production-grade valuation authority until large-scale crawling expands the dataset.

#### 1. Methodology & Validation Architecture
- **Holdout Partitioning (80/20)**: 90 training records, 23 holdout test records (`random_state=42`). Category stratification is applied where category counts support it.
- **Strict Leakage Prevention**: Feature transformers (imputation, rare category grouping, one-hot encoding) are fitted **only** on the training partition (or training folds within CV). The holdout test set remains completely untouched until final evaluation.
- **5-Fold Cross-Validation**: Candidate models are cross-validated on the 90-sample training split. Model selection is based exclusively on mean CV MAE.
- **LKR Scale Integrity**: When using `--target-transform log1p`, scikit-learn's `TransformedTargetRegressor` manages `log1p` on fit and inverts predictions via `expm1`. All evaluation metrics (MAE, RMSE, $R^2$, MedAE) are calculated and reported strictly in **original Sri Lankan Rupees (LKR)**.

#### 2. Model Comparison Table (Holdout Test Set)

| Model | Target Transform | Selected | CV MAE (Mean ± Std) | CV RMSE (Mean) | CV R² (Mean) | Test MAE (LKR) | Test RMSE (LKR) | Test R² | Test MedAE (LKR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomForestRegressor** | `log1p` | **Yes (Best CV)** | LKR 4,290,840 ± 3,502,448 | LKR 11,916,938 | 0.5949 | **LKR 1,228,489** | **LKR 1,916,199** | **0.8959** | **LKR 766,519** |
| **LinearRegression** | `log1p` | No | LKR 4,406,413 ± 3,625,003 | LKR 12,078,357 | 0.5764 | LKR 1,439,241 | LKR 2,491,244 | 0.8241 | LKR 374,050 |
| **HistGradientBoostingRegressor** | `log1p` | No | LKR 5,026,914 ± 3,755,265 | LKR 12,829,184 | 0.5029 | LKR 1,740,768 | LKR 2,554,045 | 0.8151 | LKR 949,604 |
| **MedianBaseline** | `log1p` | No | LKR 7,287,272 ± 3,666,484 | LKR 16,186,246 | -0.1490 | LKR 4,292,569 | LKR 6,323,888 | -0.1334 | LKR 3,089,920 |
| **MeanBaseline** | `log1p` | No | LKR 7,291,784 ± 3,650,468 | LKR 16,341,717 | -0.1810 | LKR 4,331,421 | LKR 6,487,456 | -0.1927 | LKR 2,888,264 |

*Raw target run comparison: With `--target-transform raw`, RandomForestRegressor achieved Test MAE of LKR 1,524,263, RMSE of LKR 2,223,011, and $R^2$ of 0.8600. Log-target transformation improves performance across all candidates by dampening extreme price outliers.*

#### 3. CLI Execution

```bash
# Run model training and evaluation with log1p target transformation (recommended)
python scripts/train_valuation_model.py --target-transform log1p

# Run training with raw LKR target
python scripts/train_valuation_model.py --target-transform raw

# Custom test split size and cross-validation folds
python scripts/train_valuation_model.py --test-size 0.20 --cv-folds 5 --seed 42
```

#### 4. Generated Artifacts & Evaluation Reports
- Model pipeline artifact: [`data/analysis/ml/models/model.joblib`](file:///Users/dilusha/Documents/vehicle-valuation-platform/data/analysis/ml/models/model.joblib)
- Serialization metadata: [`data/analysis/ml/models/model_metadata.json`](file:///Users/dilusha/Documents/vehicle-valuation-platform/data/analysis/ml/models/model_metadata.json)
- Model comparison table: [`data/analysis/ml/model_comparison.csv`](file:///Users/dilusha/Documents/vehicle-valuation-platform/data/analysis/ml/model_comparison.csv)
- Detailed evaluation report: [`data/analysis/ml/model_evaluation_report.md`](file:///Users/dilusha/Documents/vehicle-valuation-platform/data/analysis/ml/model_evaluation_report.md)
- Diagnostic figures:
  - Actual vs Predicted: `data/analysis/ml/figures/actual_vs_predicted.png`
  - Residual Distribution: `data/analysis/ml/figures/residual_distribution.png`
  - Absolute Error Distribution: `data/analysis/ml/figures/absolute_error_distribution.png`

---

### 9. Explainable Valuation Prediction Layer (Phase 5 Step 9)

The **Explainable Valuation Prediction Layer** (`ml/valuation/`, `ml/explainability/`, `ml/prediction/`, `analytics/comparables/`, `api/`) turns the trained machine learning pipeline into an end-to-end explainable valuation service with model-based uncertainty ranges, Tree SHAP factor attribution, comparable vehicle retrieval, and REST API endpoints.

> [!IMPORTANT]
> **Asking Price vs. Actual Transaction Price**:
> This valuation estimates market asking prices observed on Riyasewana. It is not a confirmed transaction or final selling price.
> Actual negotiated selling prices may differ from advertised asking prices because the collected dataset does not contain verified final transaction prices.
> Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.
> The system must **never** be interpreted as providing a guaranteed resale price or binding appraisal.

> [!NOTE]
> **Dataset Size & Experimental Research Benchmark Status**:
> The underlying model is an **experimental research benchmark** trained on **113 ML-eligible listings** across 8 vehicle categories (`Cars`, `Heavy-Duty`, `Lorries`, `Motorbikes`, `Pickups`, `SUVs`, `Three Wheelers`, `Vans`).
> Technically implemented valuation layer; further large-scale data collection is required before production-grade valuation claims. Current dataset size limits generalization across sparse categories.

#### 1. System Architecture & Capabilities
```
                  Incoming Vehicle Specification
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        VehiclePricePredictor Validation      │
        │  - Strict domain & schema checks             │
        │  - Zero leakage guard (blocks asking_price)  │
        │  - Vehicle age derivation (2026 - mfg_year)  │
        │  - Canonical category normalization          │
        └──────────────────────┬───────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  ML Asking    │  │  Model-Based  │  │   Tree SHAP   │
    │  Price Point  │  │  Prediction   │  │  Explainable  │
    │   Estimate    │  │     Range     │  │  Attribution  │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        ComparableVehicleEngine Search        │
        │  - Multi-attribute weighted similarity model │
        │  - Strictly same category matching           │
        │  - Filter ML-eligible & valid prices only    │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       Unified VehicleValuationService        │
        │  - REST API: POST /api/valuation/predict     │
        │  - ValuationReportFormatter (Text/Markdown)  │
        └──────────────────────────────────────────────┘
```

#### 2. Methodology & Component Details
1. **Model Used**: Scikit-Learn `TransformedTargetRegressor` wrapping `ColumnTransformer` (imputation, rare category grouping, one-hot encoding) and `RandomForestRegressor` (300 estimators, `min_samples_leaf=2`, `random_state=42`) trained on `log1p(y)` and inverted via `expm1` to original LKR.
2. **Prediction Range Methodology**:
   - Computes empirical quantiles across the individual predictions of all 300 decision trees in the Random Forest.
   - Evaluated at the 10th percentile (lower bound) and 90th percentile (upper bound).
   - Bounds are strictly non-negative ($\ge \text{Rs. } 10,000$) and mathematically ordered ($\text{lower} \le \text{estimate} \le \text{upper}$).
   - **Disclaimer**: This is an *indicative model-based prediction range* representing tree ensemble dispersion; it is NOT a statistically guaranteed confidence interval.
3. **Explainability Methodology**:
   - Implements **Tree SHAP** (`shap.TreeExplainer`) on the underlying forest.
   - Aggregates high-dimensional one-hot encoded dummy features back to canonical input attributes (`vehicle_age`, `mileage`, `engine_cc`, `transmission`, `brand`, `model`, `fuel_type`, `district`, `condition`).
   - Returns structured records with numerical contributions, direction (`"positive"` vs `"negative"`), and clear human-readable descriptions.
4. **Comparable Vehicle Retrieval Methodology**:
   - Queries verified PostgreSQL market listings matching the requested vehicle category.
   - Evaluates multi-attribute weighted similarity: Brand (0.25), Model (0.25), Manufacture Year/Age (0.14), Mileage (0.10), Engine CC (0.08), Transmission (0.06), Fuel Type (0.04), District (0.04), Condition (0.04).
   - Excludes ML-ineligible records, unpriced listings, and query vehicle self-matches.
   - Returns similarity scores normalized between 0.0 and 1.0 (0% to 100%).
5. **Valuation Data Quality Indicators**:
   - Assesses feature completeness across core and optional input attributes.
   - Does NOT report arbitrary confidence percentages or statistical guarantees.
6. **Comparable Market Summary**:
   - Computes distribution metrics (min, median, max, average, count) across verified comparable listings.
7. **Valuation Audit & Reproducibility**:
   - **Auditable via Model Metadata**: Every valuation result embeds non-sensitive metadata (`model_name`, `model_version`, `model_status`, `target`, `generated_at`, `feature_schema_version`, `valuation_method`, `comparable_method`, `prediction_range_method`) sourced directly from `model_metadata.json`.
   - **Deterministic Reproducibility Fingerprint**: Implements SHA-256 hashing over canonicalized, key-sorted vehicle inputs and model configuration. Identical inputs and model artifacts generate the exact same fingerprint, while changing input parameters alters the fingerprint.
   - **Important System Clarifications & Limitations**:
     - **Not a Security Signature**: The reproducibility fingerprint is an algorithmic input/model configuration identifier for provenance tracking, NOT a cryptographic security signature, appraisal guarantee, or anti-tamper seal.
     - **Experimental Research Benchmark**: The valuation pipeline is an experimental research benchmark trained on verified ML-eligible listings; it is not a production enterprise appraisal service.
     - **Asking Price vs Transaction Price**: The model predicts advertised asking price observed on online listings, NOT negotiated or verified final transaction prices.
     - **Indicative Range vs Confidence Interval**: The prediction range represents empirical tree dispersion across the Random Forest ensemble (10th–90th percentiles), NOT a statistical confidence interval.

#### 3. REST API Reference: `POST /api/valuation/predict`

##### Example Request:
```bash
curl -X POST http://localhost:8000/api/valuation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Cars",
    "brand": "Toyota",
    "model": "Premio",
    "manufacture_year": 2016,
    "mileage": 85000,
    "engine_cc": 1500,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "district": "Colombo",
    "condition": "Registered (Used)",
    "top_k_factors": 3,
    "top_k_comparables": 3
  }'
```

##### Example Response:
```json
{
  "estimated_asking_price_lkr": 15916610.12,
  "prediction_range_lkr": {
    "estimate": 15916610.12,
    "lower": 8723171.12,
    "upper": 24122604.89,
    "spread": 15399433.77,
    "percentile_lower": 10,
    "percentile_upper": 90,
    "method": "RandomForest 300-tree empirical dispersion (10th–90th percentiles)"
  },
  "currency": "LKR",
  "model": {
    "name": "RandomForestRegressor",
    "target_variable": "asking_price",
    "target_transform": "log1p",
    "training_records": 90,
    "test_samples": 23,
    "status": "EXPERIMENTAL_RESEARCH_BENCHMARK"
  },
  "explanation": [
    {
      "feature": "transmission",
      "value": "Automatic",
      "contribution": 0.6673,
      "direction": "positive",
      "description": "Transmission (Automatic) contributed positively to the model prediction."
    },
    {
      "feature": "brand",
      "value": "Toyota",
      "contribution": 0.4541,
      "direction": "positive",
      "description": "Brand (Toyota) contributed positively to the model prediction."
    },
    {
      "feature": "engine_cc",
      "value": "1,500 cc",
      "contribution": 0.1681,
      "direction": "positive",
      "description": "Engine Cc (1,500 cc) contributed positively to the model prediction."
    }
  ],
  "comparables": [
    {
      "listing_id": "12293566",
      "category": "Cars",
      "brand": "Toyota",
      "model": "Allion",
      "manufacture_year": 2015,
      "mileage": 90000.0,
      "engine_cc": 1500.0,
      "fuel_type": "Petrol",
      "transmission": "Automatic",
      "district": "Colombo",
      "condition": "Registered (Used)",
      "asking_price": 14500000.0,
      "similarity_score": 0.88,
      "similarity_percentage": 88.0
    }
  ],
  "limitations": [
    "This valuation estimates market asking prices observed on Riyasewana. It is not a confirmed transaction or final selling price.",
    "Actual negotiated selling prices may differ from advertised asking prices because the collected dataset does not contain verified final transaction prices.",
    "The underlying valuation model was trained on an experimental benchmark dataset of 113 verified ML-eligible records across 8 vehicle categories; current dataset size limits generalization across sparse categories.",
    "The indicative prediction range reflects ensemble decision tree dispersion, NOT a legally or financially guaranteed appraisal.",
    "Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation."
  ]
}
```

##### 4. Health & Readiness Endpoints

###### Service Liveness Check: `GET /health`
Verifies that the FastAPI process is active and receiving requests.
```bash
curl -X GET http://localhost:8000/health
```
Response (`200 OK`):
```json
{
  "status": "ok"
}
```

###### Model Readiness Check: `GET /ready`
Verifies that the serialized valuation model pipeline (`model.joblib`) is present on disk and ready to serve live inferences without executing database writes.
```bash
curl -X GET http://localhost:8000/ready
```
Response (`200 OK` when ready):
```json
{
  "status": "ready"
}
```
If the trained model artifact is missing or unavailable, the endpoint returns `503 Service Unavailable`:
```json
{
  "status": "unavailable",
  "detail": "Valuation model artifact is unavailable.",
  "error_type": "ServiceUnavailable"
}
```

##### 5. Validation & Error Handling Behavior

The Valuation API enforces production hardening across all entry points:
- **Request Validation (`422 Unprocessable Entity`)**: Rejects missing required attributes (`category`, `brand`, `model`, `fuel_type`, `transmission`, `district`, `condition`, and age/year), malformed numeric values, negative values (`mileage < 0`, `engine_cc < 0`, `vehicle_age < 0`), and physically implausible manufacture years (`< 1920` or `> 2026`).
- **Domain Validation (`400 Bad Request`)**: Rejects unrecognized vehicle categories outside canonical classes.
- **Model Availability (`404 Not Found`)**: Returns a structured message if model artifacts have not yet been trained or registered.
- **Internal Error Safety (`500 Internal Server Error`)**: Unhandled exceptions are logged with full server-side tracebacks while returning a sanitized response (`detail: "An unexpected error occurred during vehicle valuation."`). Stack traces, absolute file paths, database credentials, passwords, and private seller contact details are strictly withheld from client responses.
- **Explicit CORS**: Restricts cross-origin resource sharing to designated local development environments (`localhost:8501`, `127.0.0.1:8501`, `localhost:8000`, `localhost:3000`) or custom origins via `CORS_ALLOWED_ORIGINS`, preventing accidental wildcard exposure.

##### 6. Methodological Scope & Limitations
- **Asking Price vs. Transaction Price**: The model predicts advertised asking prices observed on Riyasewana. It does **not** predict verified final transaction or negotiated selling prices.
- **Experimental Research Benchmark**: The underlying valuation pipeline is an experimental research benchmark trained on 113 verified listings across 8 vehicle categories. It serves as a benchmark and technical foundation, not a certified appraisal guarantee.

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
| `GET` | `/health` | API service liveness check (`{"status": "ok"}`) |
| `GET` | `/ready` | Model artifact availability and inference readiness check (`{"status": "ready"}`) |
| `POST` | `/api/valuation/predict` | ML Asking Price Valuation, Indicative Range & Tree SHAP Attribution |
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

1. **Asking Price vs Transaction Price**: This valuation estimates market asking prices observed on Riyasewana. It is not a confirmed transaction or final selling price.
2. **Unobserved Vehicle Factors**: Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.
3. **Observation Lifecycle**: An ad no longer appearing on Riyasewana is classified as `NO_LONGER_OBSERVED` and is never assumed to be a completed transaction.
4. **Statistical Estimates**: Indicative prediction ranges reflect ensemble decision tree dispersion, NOT legally or financially guaranteed appraisals.
5. **Dataset Scope**: The current valuation model is an **experimental research benchmark** trained on 113 verified records. It is a technically implemented valuation layer; further large-scale data collection is required before production-grade valuation claims.
