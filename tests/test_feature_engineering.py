"""
Comprehensive test suite for Feature Engineering & ML-Ready Feature Preparation.
Validates all 23 core requirements:
1. Only ML-eligible records included by default
2. Missing asking price is not converted to zero
3. Invalid target values excluded safely
4. Vehicle age calculation is correct
5. Negative vehicle age is rejected
6. Manufacture year not included together with vehicle_age by default
7. Missing mileage is handled correctly (not blindly zero)
8. Missing engine CC is handled correctly (not blindly zero)
9. Missing registration year does not become zero
10. OneHotEncoder handles unknown categories safely
11. Rare categories are handled deterministically
12. Target is never present inside X
13. Asking price cannot appear in transformed feature columns
14. No seller contact information enters the dataset
15. Input DataFrame is not mutated
16. PostgreSQL remains unchanged
17. Feature schema is deterministic
18. Re-running preparation produces consistent results
19. Raw feature values remain traceable
20. Target transformation works for raw and log1p (with inverse)
21. Leakage validation catches intentionally injected target columns
22. Empty/insufficient datasets fail clearly
23. All 8 categories are supported
"""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from database.connection import get_engine, get_sessionmaker
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    ScrapeRun,
    Vehicle,
)
from feature_engineering.categorical import (
    CategoricalFeatureEngineer,
    RareCategoryGrouper,
)
from feature_engineering.dataset import MLDatasetLoader, RAW_FEATURE_COLUMNS
from feature_engineering.numerical import (
    DEFAULT_REFERENCE_YEAR,
    NumericalFeatureEngineer,
)
from feature_engineering.pipeline import FeaturePipeline, FeaturePipelineConfig
from feature_engineering.schema import FEATURE_CATALOG, FeatureSchema
from feature_engineering.target import TargetTransformer
from feature_engineering.validation import DataLeakageError, LeakageValidator


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provides a synthetic DataFrame mimicking raw database extraction."""
    return pd.DataFrame(
        {
            "listing_id": ["L001", "L002", "L003", "L004", "L005", "L006"],
            "vehicle_id": [1, 2, 3, 4, 5, 6],
            "category": ["Cars", "Cars", "Vans", "SUVs", "Motorbikes", "Cars"],
            "brand": ["Toyota", "Toyota", "Nissan", "Toyota", "Honda", "RareBrand"],
            "model": ["Corolla", "Corolla", "Caravan", "Prado", "Dio", "RareModel"],
            "manufacture_year": [2018, 2020, 2015, 2022, 2019, 2010],
            "registration_year": [2019, np.nan, 2016, 2022, np.nan, 2011],
            "mileage": [45000.0, np.nan, 120000.0, 30000.0, 15000.0, 80000.0],
            "engine_cc": [1500.0, 1800.0, np.nan, 2800.0, 110.0, 1300.0],
            "fuel_type": ["Petrol", "Petrol", "Diesel", "Diesel", "Petrol", "Petrol"],
            "transmission": ["Automatic", "Automatic", "Manual", "Automatic", "Manual", "Automatic"],
            "district": ["Colombo", "Gampaha", "Kandy", "Colombo", "Kalutara", "Kurunegala"],
            "condition": ["Registered (Used)", "Unregistered", "Registered (Used)", "Registered (Used)", "Registered (Used)", "Registered (Used)"],
            "asking_price": [8500000.0, 12000000.0, 6500000.0, 32000000.0, 450000.0, 3000000.0],
            "ml_eligible": [True, True, True, True, True, True],
            "first_seen_at": [datetime(2026, 9, 1, tzinfo=timezone.utc)] * 6,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Only ML-eligible records included by default
# ---------------------------------------------------------------------------
def test_only_ml_eligible_included_by_default(db_session: Session):
    v = Vehicle(category="Cars", brand="Toyota", model="Vitz", manufacture_year=2018)
    db_session.add(v)
    db_session.flush()

    l_elig = Listing(listing_id="ELIG_1", vehicle_id=v.id, listing_url="http://x/1", ml_eligible=True)
    l_inelig = Listing(listing_id="INELIG_1", vehicle_id=v.id, listing_url="http://x/2", ml_eligible=False)
    db_session.add_all([l_elig, l_inelig])
    db_session.flush()

    p1 = PriceHistory(listing_id=l_elig.id, price=5000000)
    p2 = PriceHistory(listing_id=l_inelig.id, price=5000000)
    o1 = ListingObservation(listing_id=l_elig.id, scrape_run_id=1, observed_price=5000000, observed_mileage=50000)
    o2 = ListingObservation(listing_id=l_inelig.id, scrape_run_id=1, observed_price=5000000, observed_mileage=50000)
    db_session.add_all([p1, p2, o1, o2])
    db_session.commit()

    loader = MLDatasetLoader(session_factory=lambda: db_session)
    df_default = loader.load_raw_dataset(session=db_session, ml_eligible_only=True)
    assert len(df_default) == 1
    assert df_default.iloc[0]["listing_id"] == "ELIG_1"

    df_all = loader.load_raw_dataset(session=db_session, ml_eligible_only=False)
    assert len(df_all) == 2


# ---------------------------------------------------------------------------
# Test 2: Missing asking price is not converted to zero
# ---------------------------------------------------------------------------
def test_missing_asking_price_not_converted_to_zero():
    df = pd.DataFrame({
        "asking_price": [5000000.0, None, np.nan],
        "brand": ["Toyota", "Nissan", "Honda"],
    })
    df_valid, y_valid = TargetTransformer.extract_valid_target(df)
    assert len(df_valid) == 1
    assert len(y_valid) == 1
    assert y_valid.iloc[0] == 5000000.0
    assert 0 not in y_valid.values


# ---------------------------------------------------------------------------
# Test 3: Invalid target values excluded safely
# ---------------------------------------------------------------------------
def test_invalid_target_values_excluded_safely():
    df = pd.DataFrame({
        "asking_price": [-100.0, 0.0, np.nan, "INVALID", 4500000.0],
        "brand": ["A", "B", "C", "D", "E"],
    })
    df_valid, y_valid = TargetTransformer.extract_valid_target(df)
    assert len(df_valid) == 1
    assert len(y_valid) == 1
    assert y_valid.iloc[0] == 4500000.0


# ---------------------------------------------------------------------------
# Test 4: Vehicle age calculation is correct
# ---------------------------------------------------------------------------
def test_vehicle_age_calculation():
    df = pd.DataFrame({"manufacture_year": [2020, 2016, 2000]})
    nfe = NumericalFeatureEngineer(reference_year=2026)
    ages = nfe.compute_vehicle_age(df)
    assert ages.tolist() == [6.0, 10.0, 26.0]


# ---------------------------------------------------------------------------
# Test 5: Negative vehicle age is rejected
# ---------------------------------------------------------------------------
def test_negative_vehicle_age_rejected():
    df = pd.DataFrame({"manufacture_year": [2030, 2020]})
    nfe = NumericalFeatureEngineer(reference_year=2026)
    with pytest.raises(ValueError, match="Negative vehicle age detected"):
        nfe.compute_vehicle_age(df)


# ---------------------------------------------------------------------------
# Test 6: Manufacture year is not included together with vehicle_age by default
# ---------------------------------------------------------------------------
def test_manufacture_year_not_included_with_vehicle_age_by_default(sample_raw_df):
    pipe = FeaturePipeline(FeaturePipelineConfig(age_representation="vehicle_age"))
    X, y, meta = pipe.prepare_features(sample_raw_df)

    assert "vehicle_age" in X.columns
    assert "manufacture_year" not in X.columns
    # Traceability is preserved in meta
    assert "manufacture_year" in meta.columns


# ---------------------------------------------------------------------------
# Test 7: Missing mileage is handled correctly
# ---------------------------------------------------------------------------
def test_missing_mileage_handled_correctly(sample_raw_df):
    nfe = NumericalFeatureEngineer()
    df_eng = nfe.engineer_features(sample_raw_df)
    # Missing mileage must remain NaN in df_eng (never converted to 0)
    assert pd.isna(df_eng.loc[1, "mileage"])
    assert df_eng.loc[1, "mileage"] != 0


# ---------------------------------------------------------------------------
# Test 8: Missing engine CC is handled correctly
# ---------------------------------------------------------------------------
def test_missing_engine_cc_handled_correctly(sample_raw_df):
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(sample_raw_df)
    # Row 2 had NaN engine_cc in sample_raw_df
    assert pd.isna(X.loc[2, "engine_cc"])
    assert X.loc[2, "engine_cc"] != 0

    # Test preprocessor imputes median
    preprocessor = pipe.build_preprocessor()
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)
    assert not np.isnan(X_trans).any()


# ---------------------------------------------------------------------------
# Test 9: Missing registration year does not become zero
# ---------------------------------------------------------------------------
def test_missing_registration_year_does_not_become_zero(sample_raw_df):
    nfe = NumericalFeatureEngineer(include_registration_year=True)
    df_eng = nfe.engineer_features(sample_raw_df)
    assert "registration_year_missing" in df_eng.columns
    # Row 1 had NaN registration_year
    assert pd.isna(df_eng.loc[1, "registration_year"])
    assert df_eng.loc[1, "registration_year_missing"] == 1
    # Row 0 had 2019
    assert df_eng.loc[0, "registration_year_missing"] == 0


# ---------------------------------------------------------------------------
# Test 10: OneHotEncoder handles unknown categories safely
# ---------------------------------------------------------------------------
def test_onehot_encoder_handles_unknown_categories(sample_raw_df):
    pipe = FeaturePipeline()
    X_train, y_train, _ = pipe.prepare_features(sample_raw_df.iloc[:4])
    preprocessor = pipe.build_preprocessor()
    preprocessor.fit(X_train)

    # Test data with completely unknown category, brand, model, district
    unseen_df = pd.DataFrame(
        {
            "listing_id": ["UNSEEN_1"],
            "vehicle_id": [999],
            "category": ["UnseenCategory"],
            "brand": ["AstonMartin"],
            "model": ["Vantage"],
            "manufacture_year": [2017],
            "registration_year": [2018],
            "mileage": [10000.0],
            "engine_cc": [4000.0],
            "fuel_type": ["Petrol"],
            "transmission": ["Automatic"],
            "district": ["Monaragala"],
            "condition": ["Registered (Used)"],
            "asking_price": [50000000.0],
            "ml_eligible": [True],
            "first_seen_at": [datetime.now(timezone.utc)],
        }
    )
    X_test, y_test, _ = pipe.prepare_features(unseen_df)
    # Must not raise error
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape[0] == 1
    assert not np.isnan(X_test_trans).any()


# ---------------------------------------------------------------------------
# Test 11: Rare categories are handled deterministically
# ---------------------------------------------------------------------------
def test_rare_categories_handled_deterministically():
    df = pd.DataFrame({
        "brand": ["Toyota"] * 6 + ["RareBrand"] * 2,
        "model": ["Corolla"] * 6 + ["RareModel"] * 2,
    })
    grouper = RareCategoryGrouper(min_frequency=5, columns=["brand", "model"])
    grouper.fit(df)
    df_trans = grouper.transform(df)

    assert df_trans["brand"].tolist() == ["Toyota"] * 6 + ["Other"] * 2
    assert df_trans["model"].tolist() == ["Corolla"] * 6 + ["Other"] * 2


# ---------------------------------------------------------------------------
# Test 12: Target is never present inside X
# ---------------------------------------------------------------------------
def test_target_never_present_inside_X(sample_raw_df):
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(sample_raw_df)

    assert "asking_price" not in X.columns
    assert "price" not in X.columns
    assert "raw_asking_price" not in X.columns


# ---------------------------------------------------------------------------
# Test 13: Asking price cannot appear in transformed feature columns
# ---------------------------------------------------------------------------
def test_asking_price_cannot_appear_in_transformed_feature_columns(sample_raw_df):
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(sample_raw_df)
    preprocessor = pipe.build_preprocessor()
    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()
    for name in feature_names:
        assert "price" not in name.lower()
        assert "asking" not in name.lower()


# ---------------------------------------------------------------------------
# Test 14: No seller contact information enters the dataset
# ---------------------------------------------------------------------------
def test_no_seller_contact_information_enters_dataset(sample_raw_df):
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(sample_raw_df)

    forbidden_terms = ["phone", "email", "contact", "description", "title"]
    for col in list(X.columns) + list(meta.columns):
        for term in forbidden_terms:
            assert term not in col.lower()


# ---------------------------------------------------------------------------
# Test 15: Input DataFrame is not mutated
# ---------------------------------------------------------------------------
def test_input_dataframe_not_mutated(sample_raw_df):
    original_df = sample_raw_df.copy(deep=True)
    pipe = FeaturePipeline()
    pipe.prepare_features(sample_raw_df)

    pd.testing.assert_frame_equal(sample_raw_df, original_df)


# ---------------------------------------------------------------------------
# Test 16: PostgreSQL remains unchanged
# ---------------------------------------------------------------------------
def test_postgresql_remains_unchanged():
    engine = get_engine()
    with Session(engine) as session:
        v_before = session.query(Vehicle).count()
        l_before = session.query(Listing).count()
        ph_before = session.query(PriceHistory).count()
        lo_before = session.query(ListingObservation).count()
        sr_before = session.query(ScrapeRun).count()

    # Run extraction and pipeline
    loader = MLDatasetLoader()
    raw_df = loader.load_raw_dataset(ml_eligible_only=True)
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(raw_df)

    with Session(engine) as session:
        v_after = session.query(Vehicle).count()
        l_after = session.query(Listing).count()
        ph_after = session.query(PriceHistory).count()
        lo_after = session.query(ListingObservation).count()
        sr_after = session.query(ScrapeRun).count()

    assert v_before == v_after == 94
    assert l_before == l_after == 94
    assert ph_before == ph_after == 75
    assert lo_before == lo_after == 108
    assert sr_before == sr_after == 15


# ---------------------------------------------------------------------------
# Test 17: Feature schema is deterministic
# ---------------------------------------------------------------------------
def test_feature_schema_deterministic():
    schema1 = FeatureSchema()
    schema2 = FeatureSchema()

    assert schema1.to_dict() == schema2.to_dict()
    assert schema1.to_json() == schema2.to_json()
    assert schema1.to_markdown_table() == schema2.to_markdown_table()


# ---------------------------------------------------------------------------
# Test 18: Re-running preparation produces consistent results
# ---------------------------------------------------------------------------
def test_rerunning_preparation_produces_consistent_results(sample_raw_df):
    pipe1 = FeaturePipeline()
    X1, y1, meta1 = pipe1.prepare_features(sample_raw_df)

    pipe2 = FeaturePipeline()
    X2, y2, meta2 = pipe2.prepare_features(sample_raw_df)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    pd.testing.assert_frame_equal(meta1, meta2)


# ---------------------------------------------------------------------------
# Test 19: Raw feature values remain traceable
# ---------------------------------------------------------------------------
def test_raw_feature_values_remain_traceable(sample_raw_df):
    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(sample_raw_df)

    assert "listing_id" in meta.columns
    assert "vehicle_id" in meta.columns
    assert "manufacture_year" in meta.columns
    assert "raw_asking_price" in meta.columns
    assert list(meta["listing_id"]) == list(sample_raw_df["listing_id"])


# ---------------------------------------------------------------------------
# Test 20: Target transformation works for raw and log1p
# ---------------------------------------------------------------------------
def test_target_transformation_raw_and_log1p():
    prices = np.array([1000000.0, 5000000.0, 25000000.0])

    # Raw
    tt_raw = TargetTransformer(transform="raw")
    trans_raw = tt_raw.transform(prices)
    np.testing.assert_array_equal(trans_raw, prices)
    inv_raw = tt_raw.inverse_transform(trans_raw)
    np.testing.assert_array_equal(inv_raw, prices)

    # Log1p
    tt_log = TargetTransformer(transform="log1p")
    trans_log = tt_log.transform(prices)
    np.testing.assert_allclose(trans_log, np.log1p(prices))
    inv_log = tt_log.inverse_transform(trans_log)
    np.testing.assert_allclose(inv_log, prices, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 21: Leakage validation catches intentionally injected target columns
# ---------------------------------------------------------------------------
def test_leakage_validation_catches_injected_columns(sample_raw_df):
    validator = LeakageValidator()

    # Exact forbidden column
    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        validator.validate_features(["vehicle_age", "mileage", "asking_price"])

    # Lifecycle outcome column
    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        validator.validate_features(["vehicle_age", "current_status"])

    # Seller contact column
    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        validator.validate_features(["vehicle_age", "seller_phone"])

    # Direct target duplicate
    X = sample_raw_df[["vehicle_age", "mileage"]].copy() if "vehicle_age" in sample_raw_df else sample_raw_df[["mileage", "engine_cc"]].copy()
    X["leaked_target"] = sample_raw_df["asking_price"]
    with pytest.raises(DataLeakageError):
        validator.validate_matrix_against_target(X, sample_raw_df["asking_price"])


# ---------------------------------------------------------------------------
# Test 22: Empty or insufficient datasets fail clearly
# ---------------------------------------------------------------------------
def test_empty_or_insufficient_datasets_fail_clearly():
    pipe = FeaturePipeline()

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input raw_df is empty"):
        pipe.prepare_features(pd.DataFrame())

    # All invalid prices
    df_invalid_prices = pd.DataFrame({
        "asking_price": [None, np.nan, -500],
        "category": ["Cars", "Cars", "Cars"],
        "brand": ["Toyota", "Toyota", "Toyota"],
        "model": ["Corolla", "Corolla", "Corolla"],
        "manufacture_year": [2020, 2021, 2019],
    })
    with pytest.raises(ValueError, match="No valid records remain after target validation"):
        pipe.prepare_features(df_invalid_prices)


# ---------------------------------------------------------------------------
# Test 23: All 8 categories are supported
# ---------------------------------------------------------------------------
def test_all_eight_categories_supported():
    all_categories = [
        "Cars",
        "Heavy-Duty",
        "Lorries",
        "Motorbikes",
        "Pickups",
        "SUVs",
        "Three Wheelers",
        "Vans",
    ]
    records = []
    for i, cat in enumerate(all_categories):
        records.append({
            "listing_id": f"CAT_{i}",
            "vehicle_id": i + 1,
            "category": cat,
            "brand": "TestBrand",
            "model": "TestModel",
            "manufacture_year": 2020,
            "registration_year": 2020,
            "mileage": 50000.0,
            "engine_cc": 1500.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Colombo",
            "condition": "Registered (Used)",
            "asking_price": 5000000.0,
            "ml_eligible": True,
            "first_seen_at": datetime.now(timezone.utc),
        })
    df_all_cats = pd.DataFrame(records)

    pipe = FeaturePipeline()
    X, y, meta = pipe.prepare_features(df_all_cats)
    assert len(X) == 8
    assert set(X["category"]) == set(all_categories)
    preprocessor = pipe.build_preprocessor()
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)
    assert X_trans.shape[0] == 8
