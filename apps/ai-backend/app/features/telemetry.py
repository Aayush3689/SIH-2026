from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_TELEMETRY_COLUMNS = [
    "timestamp",

    "rpm",
    "throttle_position_pct",
    "cht_c",
    "egt_c",

    "oil_pressure_kpa",
    "oil_temperature_c",

    "fuel_flow_lph",
    "fuel_pressure_kpa",

    "intake_manifold_pressure_kpa",

    "engine_vibration_mm_s",

    "battery_voltage_v",
    "alternator_output_a",

    "injection_timing_deg_btdc",
    "engine_load_pct",

    "ambient_temp_c",
    "planned_altitude_ft",
    "airspeed_kts",
]


def normalize_telemetry(
    telemetry: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert incoming telemetry into a normalized DataFrame.

    The returned rows are chronological.
    """

    if isinstance(telemetry, pd.DataFrame):

        df = telemetry.copy()

    elif isinstance(telemetry, list):

        if not telemetry:
            raise ValueError(
                "Telemetry list is empty."
            )

        df = pd.DataFrame(
            telemetry
        )

    else:

        raise TypeError(
            "telemetry must be a list of dictionaries "
            "or pandas DataFrame."
        )

    missing = [
        col
        for col in REQUIRED_TELEMETRY_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing telemetry columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Telemetry contains invalid timestamps."
        )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    numeric_columns = [
        col
        for col in REQUIRED_TELEMETRY_COLUMNS
        if col != "timestamp"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def validate_sample_count(
    df: pd.DataFrame,
    minimum_samples: int,
) -> None:
    """
    Ensure enough historical samples exist for a model.
    """

    if len(df) < minimum_samples:

        raise ValueError(
            f"Need at least {minimum_samples} telemetry "
            f"samples, received {len(df)}."
        )