import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.inference import router as inference_router


logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Engine Health Monitoring Service",
    description=(
        "AI inference service for real-time health monitoring, "
        "fault prediction and remaining useful life estimation "
        "of aero piston engines used in MALE UAVs."
    ),
    version="1.0.0",
)


# ==============================================================
# Global Error Handling
# ==============================================================

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    is_empty_telemetry = any(
        error.get("loc") == ("body", "telemetry")
        and error.get("type") == "too_short"
        and error.get("input") == []
        for error in errors
    )

    if is_empty_telemetry:
        logger.warning(
            "Inference validation failed: empty telemetry payload path=%s",
            request.url.path,
        )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_TELEMETRY",
                    "message": "Telemetry payload cannot be empty.",
                },
            },
        )

    details = [
        {
            "loc": list(error["loc"]),
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in errors
    ]
    logger.warning(
        "Request validation failed: method=%s path=%s errors=%d",
        request.method,
        request.url.path,
        len(details),
    )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": details,
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed."
    logger.warning(
        "HTTP exception: method=%s path=%s status_code=%d",
        request.method,
        request.url.path,
        exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": message,
            },
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled exception: method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
            },
        },
    )


# ==============================================================
# API Routers
# ==============================================================

app.include_router(
    inference_router
)


# ==============================================================
# Health Check
# ==============================================================

@app.get(
    "/health",
    tags=["Health"],
    summary="Check AI backend health",
)
def health_check():
    return {
        "status": "ok",
        "service": "ai-backend",
    }
