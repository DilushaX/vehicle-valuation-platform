# Production Collection Readiness Report: Sri Lankan Vehicle Market Intelligence Platform

**Document Version**: 1.0  
**Audit Date**: September 16, 2026  
**Evaluation Scope**: Pre-Model Training Full-System Validation (Phase 5 Step 7.5)  
**Target Platform**: Riyasewana (riyasewana.com)  

---

## Executive Summary

Before transitioning to Machine Learning Model Training (Phase 5 Step 8), an end-to-end audit and controlled real-world validation was conducted across the entire vehicle market valuation stack. 

The validation confirmed that the system operates harmoniously from HTTP ingestion to feature dataset preparation:
- **Controlled Live Ingestion**: Successfully executed live collection across all **8 canonical categories** (`Cars`, `Heavy-Duty`, `Lorries`, `Motorbikes`, `Pickups`, `SUVs`, `Three Wheelers`, `Vans`), scraping 80 live listings with 0 HTTP or parsing failures.
- **Deduplication & Observation Tracking**: Existing listings were correctly matched by `(source, listing_id)`, recording new timestamped observations without creating duplicate listing rows or redundant price history points.
- **Lifecycle & Disappearance Safety**: Partial collection safely skipped unobserved listing deactivation. Zero active listings were erroneously transitioned to `NO_LONGER_OBSERVED`, and disappeared listings are never assumed to be `SOLD`.
- **Data Quality & ML Dataset**: The data quality engine correctly classified 113 listings as ML-eligible (66.08%) and 58 as ML-ineligible. The ML dataset preparation pipeline cleanly extracted all 113 eligible records into a leakage-free feature matrix with zero target contamination.
- **Overall Verdict**: **PRODUCTION-COLLECTION READY** for phased, rate-limited batch collection. (Large-scale sustained execution of 100,000+ records remains to be verified under extended operational schedules).

---

## Technical Readiness Assessment (14 Evaluation Areas)

### 1. Scraper Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - `RiyasewanaClient` cleanly extracts raw HTML payloads using `httpx` with desktop browser user-agent emulation.
  - Parsing engine (`RiyasewanaParser`, `VehicleExtractor`) accurately parses title, make, model, manufacture year, registration year, price, mileage, fuel type, transmission, engine capacity, condition, and district.
  - Zero private contact information (seller phone numbers, email addresses, personal IDs) is extracted or stored.
- **Large-Scale Consideration**: Target site DOM changes over months could require parser adjustments. Robust error logging and fallback selectors are active.

### 2. Pagination Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - `RiyasewanaCategoryDiscovery` discovers pagination page links (`?page=2`, `?page=3`) dynamically from navigation controls.
  - Early termination triggers accurately when `max_pages` or `max_listings` limits are reached.
  - `pagination_exhausted` flag correctly signals when the final page of a category has been retrieved.
- **Large-Scale Consideration**: For categories with hundreds of pages (e.g. Cars), deep pagination requires sustained HTTP sessions with polite request pacing.

### 3. Retry Readiness & Fault Tolerance
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - `retry_with_backoff` handles transient network blurs, connection drops, and HTTP 429/5xx status codes.
  - Configurable parameters: `MAX_RETRIES = 3`, exponential backoff multiplier `2.0x`, base delay `1.0s`.
  - Respects HTTP `Retry-After` headers if returned by the remote server.

### 4. Rate-Limit Safety & Politeness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - Mandatory delay of `1.5s` enforced between outbound HTTP requests via `time.sleep()`.
  - No concurrent or asynchronous bombardment of riyasewana.com.
  - CAPTCHAs, access controls, and anti-bot systems are respected; the crawler strictly avoids circumvention or aggressive request bursts.

### 5. Deduplication Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - PostgreSQL unique constraint on `(source, listing_id)` prevents listing identity duplicates.
  - URL uniqueness audited; zero duplicate URLs across the registry.
  - Existing listings receive updated `last_seen_at` timestamps and append new rows to `ListingObservation` without duplicate `Listing` insertions.

### 6. PostgreSQL Database Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - Verified relational structure: `Vehicle` (1-to-many) `Listing` (1-to-many) `PriceHistory` and `ListingObservation`.
  - Zero orphan listings, zero orphan price histories, zero orphan observations, and zero orphan vehicles.
  - Clean transaction rollback management ensures failed category batches do not leave uncommitted or corrupted states.

### 7. Historical Tracking Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - Price changes trigger append-only insertions into `PriceHistory`.
  - Observations append-only into `ListingObservation`, capturing price and mileage trajectory over time.
  - Every scrape run records execution metrics in `ScrapeRun`.

### 8. Data Quality Engine Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - Audits 20 syntax and domain boundary rules.
  - Critical vs. non-critical classification decouples physical data validity from ML modeling eligibility.
  - Ineligible records are preserved with auditable failure codes (`missing_price`, `missing_mileage`, `suspicious_mileage_pattern`), never destructively deleted.

### 9. ML Dataset Preparation Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - `MLDatasetLoader` extracts only verified `ml_eligible = True` listings.
  - `TargetTransformer` strictly separates `asking_price` as $y$, with reversible `"raw"` and `"log1p"` transformations.
  - `NumericalFeatureEngineer` calculates `vehicle_age = reference_year (2026) - manufacture_year` and excludes `manufacture_year` to eliminate $\rho = -1.00$ collinearity.
  - `RareCategoryGrouper` groups low-frequency categories without dropping records.
  - `LeakageValidator` actively verifies that no target, price clone, lifecycle status, observation counts, or seller contact details enter $X$.
  - Generates reproducible artifacts: `ml_dataset.csv`, `feature_schema.json`, `dataset_summary.json`.

### 10. Scheduler Readiness
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - POSIX `fcntl.flock` atomic file locking (`CollectionLock`) prevents overlapping or concurrent collection processes.
  - Dry-run mode (`--dry-run`) enables end-to-end verification without database mutations.
  - macOS launchd agent generator and status reporter (`scripts/scheduled_collection.py`) tested and functional.

### 11. Failure Isolation
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - Category-level isolation: an exception or network timeout in one category rolls back only that category's transaction and allows remaining categories to proceed uninterrupted.
  - Listing-level isolation: a single malformed listing HTML page is logged to `failed_listings` while other listings on the page are scraped successfully.

### 12. Completeness Verification
- **Status**: **VERIFIED & READY**
- **Verified Capabilities**:
  - `CategoryCompleteness` tracks pages attempted/scraped, listing URLs discovered, listings attempted/scraped, and completeness percentages.
  - Disappearance transitions are strictly gated behind `is_complete_scope`.

### 13. Known Limitations (What Is NOT Yet Verified at Large Scale)
1. **Unverified at 100,000+ Scale**: The crawler has been rigorously verified on controlled runs up to 100 listings per run. Long-duration crawls spanning tens of thousands of pages across multiple days have not yet been executed.
2. **IP Rate Limit Thresholds**: While `1.5s` politeness is effective for batches, Riyasewana's ISP-level daily bandwidth threshold is unmeasured.
3. **Historical Trend Depth**: Current database history spans several calendar days; multi-month macroeconomic price inflation or seasonal depreciation curves require sustained daily collection over 3–6 months.
4. **Asking Price Limitation**: Observed asking prices represent seller listing quotes, not final negotiated transaction prices.

---

## Recommended Safe Production Collection Procedure

For future large-scale expansion without risking IP blocking or database instability:

1. **Phased Batching**:
   - Phase A: Collect first 5 pages per category (approx. 200 listings/category = 1,600 total listings).
   - Phase B: Review error rates, network latency, and memory consumption.
   - Phase C: Increase to scheduled daily runs of 10–20 pages per category during off-peak hours (02:00 AM Sri Lanka Time / UTC 20:30).
2. **Politeness Settings**:
   - Maintain `REQUEST_DELAY = 1.5s` (minimum 1.2s). Do not lower below 1.0s.
   - Use `REQUEST_TIMEOUT = 20.0s` with exponential backoff on retries.
3. **Database Maintenance**:
   - Periodic index reindexing on `idx_listing_source`, `idx_observation_listing_date`, and `idx_price_history_listing_date`.
4. **Monitoring**:
   - Run `python scripts/data_quality_report.py --fail-on-inconsistency` after each batch to guarantee referential integrity.
