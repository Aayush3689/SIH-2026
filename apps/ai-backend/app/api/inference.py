import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

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
) -> InferenceResponse | JSONResponse:

    try:
        sample_count = len(payload.telemetry)
        latest_sample = payload.telemetry[-1]
        logger.info(
            "Inference request received: engine_id=%s flight_id=%s samples=%d",
            latest_sample.engine_id,
            latest_sample.flight_id,
            sample_count,
        )

        telemetry = [
            sample.model_dump()
            for sample in payload.telemetry
        ]

        result = pipeline.run(
            telemetry
        )

        logger.info(
            "Inference completed successfully: engine_id=%s flight_id=%s samples=%d",
            latest_sample.engine_id,
            latest_sample.flight_id,
            sample_count,
        )

        return InferenceResponse(
            success=True,
            data=result,
        )

    except ValueError as exc:
        logger.warning(
            "Inference validation failed: %s",
            exc,
        )

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_TELEMETRY",
                    "message": str(exc),
                },
            },
        )

    except Exception:
        logger.exception(
            "Inference pipeline failed"
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": (
                        "An internal error occurred while processing "
                        "the inference request."
                    ),
                },
            },
        )
