# Data Dictionary — Sri Lankan Vehicle Market Intelligence Platform

This document defines the schema, data types, descriptions, example values, and validation rules for normalized vehicle records collected from online market sources (primarily Riyasewana).

---

| Field | Type | Description | Example | Validation Rule |
| :--- | :--- | :--- | :--- | :--- |
| `listing_id` | `String` | Unique source platform listing identifier extracted from listing URL. | `"12217083"` | Must be non-empty numeric string. Unique key constraint. |
| `listing_url` | `String` | Canonical target listing page URL. | `"https://riyasewana.com/buy/toyota-corolla-axio-sale-homagama-12217083"` | Valid HTTP/HTTPS URI format. |
| `title` | `String` | Raw listing headline text. | `"Toyota Corolla Axio Wxb 2016 Car (Used)"` | Trimmed string. |
| `category` | `String` | High-level vehicle classification category (Car, SUV, Van, Motorbike, Three Wheel, Lorry, Bus, Pickup). | `"Car"` | Standardized string category. |
| `brand` | `String` | Vehicle manufacturer / make. | `"Toyota"` | Required for valid classification. `missing_brand` flag if null. |
| `model` | `String` | Specific vehicle model name / trim. | `"Corolla Axio Wxb"` | Required for valid classification. `missing_model` flag if null. |
| `year` | `Integer` | Legacy / General vehicle model year (mapped from manufacture_year or registration_year). | `2016` | Must be integer between `1950` and `Current Year + 1`. Flagged `invalid_year` or `missing_year`. |
| `manufacture_year` | `Integer` | Year of vehicle manufacture (YOM). Extracted from structured fields or listing description (e.g. YOM, Year of Manufacture). | `2016` | Integer between `1950` and `Current Year + 1`. Kept as `None` if not available. |
| `registration_year` | `Integer` | Year of vehicle first registration (YOR). Extracted from structured fields or listing description (e.g. YOR, First Registration, Reg Year). | `2017` | Integer between `1950` and `Current Year + 1`. Kept as `None` if not available. |
| `mileage` | `Integer` | Odometer reading in kilometers (km). | `118000` | Non-negative integer $\le 2,000,000$. Flagged `missing_mileage`, `suspicious_mileage`, or `suspicious_mileage_pattern`. |
| `price` | `Integer` | Asking price in Sri Lankan Rupees (LKR). | `12300000` | Integer between $50,000$ and $500,000,000$. Flagged `missing_price`, `suspicious_price`, or `suspicious_price_pattern`. |
| `fuel_type` | `String` | Engine fuel technology (Petrol, Diesel, Hybrid, Electric). | `"Hybrid"` | Normalized string identifier. |
| `transmission` | `String` | Gearbox type (Automatic, Manual, Tiptronic). | `"Automatic"` | Normalized string identifier. |
| `engine_cc` | `Integer` | Engine displacement in cubic centimeters (cc). | `1500` | Non-negative integer. |
| `condition` | `String` | Vehicle registration status / condition (Registered (Used), Unregistered (Reconditioned), Brand New). | `"Registered (Used)"` | Standardized text descriptor. |
| `location` | `String` | Local town or area specified by the seller. | `"Homagama"` | Raw location text string. |
| `district` | `String` | Inferred Sri Lankan administrative district derived from seller location. | `"Colombo"` | Standardized Sri Lankan administrative district name. |
| `ad_date` | `String` | Original posting timestamp string as reported by the source website. | `"2026 Aug 27, 12:03 pm"` | Text string representing ad publication date. |
| `description` | `Text` | Full free-text seller notes, specifications, and ad copy. | `"3rd owner, Hybrid battery replaced with warranty..."` | Full text preserve. |
| `source` | `String` | Source platform identifier code. | `"riyasewana"` | Constant string identifier. |
| `scraped_at` | `String` | UTC ISO-8601 timestamp when the listing was scraped. | `"2026-09-03T08:21:50Z"` | Valid ISO-8601 UTC timestamp string. |
| `raw_data` | `JSON / Dict` | Complete un-mutated raw parsed attributes for auditing. | `{...}` | Complete dictionary preservation. |
| `validation_issues` | `List[String]` | Array of detected data quality issue flags. | `["suspicious_price_pattern"]` | List of flag string tokens. Empty list if 100% valid. |
| `is_valid` | `Boolean` | Overall data quality validation status flag. | `True` | `True` if `validation_issues` is empty, `False` otherwise. |

---

## Data Quality Issue Flags Reference

| Issue Flag | Trigger Criteria | Action Taken |
| :--- | :--- | :--- |
| `missing_price` | Price field is missing or null. | Record flagged `is_valid = False`; preserved in DB. |
| `suspicious_price_pattern` | Price contains dummy pattern (e.g., 123, 111111, 123456). | Record flagged `is_valid = False`; excluded from ML training. |
| `suspicious_price` | Price $< 50,000$ or $> 500,000,000$ LKR. | Record flagged `is_valid = False`. |
| `missing_mileage` | Mileage field is missing or null. | Record flagged `is_valid = False`. |
| `suspicious_mileage_pattern` | Mileage contains dummy pattern (e.g., 123456, 999999). | Record flagged `is_valid = False`. |
| `suspicious_mileage` | Mileage $> 2,000,000$ km. | Record flagged `is_valid = False`. |
| `missing_year` | Neither `manufacture_year`, `registration_year`, nor `year` is present. | Record flagged `is_valid = False`. |
| `invalid_year` | Manufacture Year $< 1950$ or $> \text{Current Year} + 1$. | Record flagged `is_valid = False`. |
| `invalid_registration_year` | Registration Year $< 1950$ or $> \text{Current Year} + 1$. | Record flagged `is_valid = False`. |
| `registration_precedes_manufacture` | Registration Year is strictly earlier than Manufacture Year. | Record flagged `is_valid = False`. |
| `missing_brand` | Brand / Make field is null or empty. | Record flagged `is_valid = False`. |
| `missing_model` | Model field is null or empty. | Record flagged `is_valid = False`. |
