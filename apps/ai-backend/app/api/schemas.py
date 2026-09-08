from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TelemetrySample(BaseModel):
    """
    Single telemetry sample received by the AI backend.
    """

    model_config = ConfigDict(
        extra="allow"
    )

    timestamp: str = Field(
        ...,
        description="Telemetry timestamp in ISO-8601 format.",
        examples=["2024-02-06T01:07:42"],
    )

    engine_id: str = Field(
        ...,
        description="Unique engine identifier.",
        examples=["ENG-2023009"],
    )

    flight_id: str = Field(
        ...,
        description="Flight identifier.",
        examples=["ENG-2023009-F010"],
    )

    elapsed_s: float = Field(
        ...,
        ge=0,
        description="Elapsed time from flight start in seconds.",
        examples=[1.0],
    )

    rpm: float
    throttle_position_pct: float
    cht_c: float
    egt_c: float
    oil_pressure_kpa: float
    oil_temperature_c: float
    fuel_flow_lph: float
    fuel_pressure_kpa: float
    intake_manifold_pressure_kpa: float
    engine_vibration_mm_s: float
    battery_voltage_v: float
    alternator_output_a: float
    injection_timing_deg_btdc: float
    engine_load_pct: float
    ambient_temp_c: float
    planned_altitude_ft: float
    airspeed_kts: float


class InferenceRequest(BaseModel):
    """
    Request payload for engine health inference.
    """

    telemetry: List[TelemetrySample] = Field(
        ...,
        min_length=1,
        description=(
            "One or more telemetry samples from the current "
            "engine and flight."
        ),
    )


class AnomalyPrediction(BaseModel):
    state: str = Field(
        ...,
        description="Predicted anomaly state.",
        examples=["NORMAL"],
    )

    raw_label: int = Field(
        ...,
        description="Raw classifier label.",
        examples=[0],
    )

    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability for each anomaly state.",
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence of the predicted anomaly state.",
    )


class FaultPrediction(BaseModel):
    fault: str = Field(
        ...,
        description="Predicted fault mode.",
        examples=["misfire"],
    )

    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability for each fault mode.",
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence of the predicted fault mode.",
    )


class InferenceData(BaseModel):
    engine_id: str = Field(
        ...,
        examples=["ENG-2023009"],
    )

    flight_id: str = Field(
        ...,
        examples=["ENG-2023009-F010"],
    )

    timestamp: str = Field(
        ...,
        description="Timestamp of the latest telemetry sample.",
        examples=["2024-02-06T01:07:45"],
    )

    anomaly: List[AnomalyPrediction]

    fault: List[FaultPrediction]

    rul: Optional[List[float]] = Field(
        default=None,
        description=(
            "Predicted remaining useful life in hours. "
            "Returns null until the minimum RUL window is available."
        ),
        examples=[[6.31]],
    )


class InferenceResponse(BaseModel):
    success: bool = Field(
        ...,
        description="Whether inference completed successfully.",
        examples=[True],
    )

    data: InferenceData