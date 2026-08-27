"""
Data Quality and System Monitoring API Routes Module
Endpoints for data quality audits, outlier summaries, and scraper runs.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.connection import get_db
from database.models import Listing, ScrapeRun, DataQualityRecord
from api.schemas.schemas import DataQualitySummaryResponse

router = APIRouter(prefix="/api/v1/quality", tags=["Data Quality & Monitoring"])

@router.get("/summary", response_model=DataQualitySummaryResponse)
def get_data_quality_summary(db: Session = Depends(get_db)):
    """Provides platform-wide data quality KPIs."""
    total = db.query(Listing).count()
    valid = db.query(Listing).filter(Listing.data_quality_status == "VALID").count()
    suspicious = db.query(Listing).filter(Listing.data_quality_status == "SUSPICIOUS").count()
    invalid = db.query(Listing).filter(Listing.data_quality_status == "INVALID").count()
    missing = db.query(Listing).filter(Listing.data_quality_status == "MISSING").count()
    
    # Calculate overall quality score
    score = round((valid / total * 100.0), 2) if total > 0 else 100.0

    last_run = db.query(ScrapeRun).order_by(desc(ScrapeRun.started_at)).first()
    last_run_dict = None
    if last_run:
        last_run_dict = {
            "run_id": last_run.run_id,
            "category": last_run.category,
            "started_at": last_run.started_at.isoformat() if last_run.started_at else None,
            "ended_at": last_run.ended_at.isoformat() if last_run.ended_at else None,
            "total_scraped": last_run.total_scraped,
            "new_count": last_run.new_count,
            "updated_count": last_run.updated_count,
            "status": last_run.status
        }

    return DataQualitySummaryResponse(
        total_records=total,
        valid_records=valid,
        suspicious_records=suspicious,
        invalid_records=invalid,
        missing_records=missing,
        duplicates_detected=0,
        overall_quality_score=score,
        last_scrape_run=last_run_dict
    )

@router.get("/runs")
def get_scrape_runs(limit: int = Query(default=10, ge=1, le=50), db: Session = Depends(get_db)):
    """Lists historical scraper runs with telemetry."""
    runs = db.query(ScrapeRun).order_by(desc(ScrapeRun.started_at)).limit(limit).all()
    return [
        {
            "run_id": r.run_id,
            "category": r.category,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            "total_scraped": r.total_scraped,
            "new_count": r.new_count,
            "updated_count": r.updated_count,
            "price_change_count": r.price_change_count,
            "no_longer_observed_count": r.no_longer_observed_count,
            "invalid_count": r.invalid_count,
            "status": r.status,
            "error_message": r.error_message
        }
        for r in runs
    ]
