"""
Vehicle Valuation REST API Endpoints.
Provides explainable market asking price predictions with model-based ranges and comparables.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from api.schemas.schemas import ErrorResponse, VehicleValuationRequest, VehicleValuationResponse
from ml.prediction.predictor import ValidationError
from ml.valuation.valuation_service import VehicleValuationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/valuation", tags=["Valuation"])

# Global service instance cache
_service_instance: Optional[VehicleValuationService] = None


def get_valuation_service() -> VehicleValuationService:
    global _service_instance
    if _service_instance is None:
        _service_instance = VehicleValuationService()
    return _service_instance


@router.post(
    "/predict",
    response_model=VehicleValuationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid vehicle specification"},
        404: {"model": ErrorResponse, "description": "Model artifact unavailable"},
        422: {"description": "Request validation failure"},
        500: {"model": ErrorResponse, "description": "Internal prediction failure"},
    },
    summary="Estimate Vehicle Asking Price",
    description=(
        "Estimates market asking prices observed on Sri Lankan vehicle listings (Riyasewana) with indicative "
        "model-based prediction range, Tree SHAP factor attribution, and comparable vehicle matches.\n\n"
        "**Important Operational Disclaimers**:\n"
        "- The model estimates advertised asking price, NOT verified transaction or settlement price.\n"
        "- It does NOT predict verified transaction price or final sale contracts.\n"
        "- The valuation model is an experimental research benchmark trained on verified benchmark listings."
    ),
)
def predict_vehicle_valuation(
    request: VehicleValuationRequest,
) -> VehicleValuationResponse:
    """
    Computes explainable valuation for a vehicle listing.
    
    This valuation estimates market asking prices observed on Riyasewana. It does not predict
    verified transaction or final selling prices. The underlying valuation model is an experimental
    research benchmark. Actual vehicle value may differ due to factors not captured by the dataset,
    including physical condition, accident history, mechanical condition, battery/engine health,
    and registration documentation.
    """
    try:
        service = get_valuation_service()
    except FileNotFoundError as e:
        logger.error(f"Valuation model artifact unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Valuation model artifact is currently unavailable. Ensure model training has completed.",
        )
    except Exception as e:
        logger.error(f"Failed to initialize valuation service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Valuation service initialization failed.",
        )

    # Convert Pydantic request to dictionary
    data_dict = request.model_dump()
    top_k_factors = data_dict.pop("top_k_factors", 5)
    top_k_comps = data_dict.pop("top_k_comparables", 5)
    p_lower = data_dict.pop("percentile_lower", 10)
    p_upper = data_dict.pop("percentile_upper", 90)

    try:
        result = service.valuate(
            vehicle_data=data_dict,
            top_k_factors=top_k_factors,
            top_k_comparables=top_k_comps,
            percentile_lower=p_lower,
            percentile_upper=p_upper,
        )
        return VehicleValuationResponse(**result.to_dict())

    except ValidationError as e:
        logger.warning(f"Validation error in valuation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid vehicle input: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Internal valuation processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during vehicle valuation.",
        )
