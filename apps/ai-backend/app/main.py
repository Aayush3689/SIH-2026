from fastapi import FastAPI

from app.api.inference import router as inference_router


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