import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    InferenceRequest,
    InferenceResponse,
)
from app.inference.pipeline import InferencePipeline


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1",
    tags=["Inference"],
)


# ==============================================================
# Load models once when the API process starts.
# ==============================================================

pipeline = InferencePipeline()


@router.post(
    "/inference",
    response_model=InferenceResponse,
    summary="Run engine health inference",
    description=(
        "Runs the complete AI inference pipeline on the supplied "
        "engine telemetry window. Anomaly and fault predictions "
        "can be generated from a single sample. RUL prediction "
        "becomes available when at least 8 samples are present."
    ),
    responses={
        400: {
            "description": "Invalid telemetry or missing required fields."
        },
        500: {
            "description": "Internal inference failure."
        },
    },
)
def run_inference(
    payload: InferenceRequest,
) -> InferenceResponse:

    try:
        telemetry = [
            sample.model_dump()
            for sample in payload.telemetry
        ]

        result = pipeline.run(
            telemetry
        )

        return InferenceResponse(
            success=True,
            data=result,
        )

    except ValueError as exc:
        logger.exception(
            "Inference validation error"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Inference pipeline failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc