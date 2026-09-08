from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Features calculated over the most recent 30 seconds.
ROLLING_30S_COLUMNS = [
    "vibration",
    "rpm",
]

# Features calculated over the most recent 60 seconds.
ROLLING_60S_COLUMNS = [
    "egt",
    "cht",
    "oil_pressure",
    "oil_temperature",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "timestamp" not in result.columns:
        raise ValueError("Missing required column: timestamp")

    if "engine_id" not in result.columns:
        raise ValueError("Missing required column: engine_id")

    if "flight_id" not in result.columns:
        raise ValueError("Missing required column: flight_id")

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found in telemetry."
        )

    result = result.sort_values(
        ["engine_id", "flight_id", "timestamp"]
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------
# Rolling feature generation
# ---------------------------------------------------------

def _add_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    required = [
        "engine_vibration_mm_s",
        "rpm",
        "egt_c",
        "cht_c",
        "oil_pressure_kpa",
        "oil_temperature_c",
    ]

    _validate_columns(result, required)

    result = result.sort_values(
        ["engine_id", "flight_id", "timestamp"]
    ).reset_index(drop=True)

    feature_frames = []

    for _, group in result.groupby(
        ["engine_id", "flight_id"],
        sort=False,
    ):
        group = group.sort_values("timestamp").copy()

        indexed = group.set_index("timestamp")

        indexed["vibration_mean_30s"] = (
            indexed["engine_vibration_mm_s"]
            .rolling("30s", min_periods=1)
            .mean()
        )

        indexed["vibration_std_30s"] = (
            indexed["engine_vibration_mm_s"]
            .rolling("30s", min_periods=2)
            .std()
        )

        indexed["rpm_mean_30s"] = (
            indexed["rpm"]
            .rolling("30s", min_periods=1)
            .mean()
        )

        indexed["rpm_std_30s"] = (
            indexed["rpm"]
            .rolling("30s", min_periods=2)
            .std()
        )

        indexed["egt_mean_60s"] = (
            indexed["egt_c"]
            .rolling("60s", min_periods=1)
            .mean()
        )

        indexed["egt_std_60s"] = (
            indexed["egt_c"]
            .rolling("60s", min_periods=2)
            .std()
        )

        indexed["cht_mean_60s"] = (
            indexed["cht_c"]
            .rolling("60s", min_periods=1)
            .mean()
        )

        indexed["cht_std_60s"] = (
            indexed["cht_c"]
            .rolling("60s", min_periods=2)
            .std()
        )

        indexed["oil_pressure_mean_60s"] = (
            indexed["oil_pressure_kpa"]
            .rolling("60s", min_periods=1)
            .mean()
        )

        indexed["oil_pressure_std_60s"] = (
            indexed["oil_pressure_kpa"]
            .rolling("60s", min_periods=2)
            .std()
        )

        indexed["oil_temperature_mean_60s"] = (
            indexed["oil_temperature_c"]
            .rolling("60s", min_periods=1)
            .mean()
        )

        indexed["oil_temperature_std_60s"] = (
            indexed["oil_temperature_c"]
            .rolling("60s", min_periods=2)
            .std()
        )

        feature_frames.append(
            indexed.reset_index()
        )

    return pd.concat(
        feature_frames,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Rate-of-change features
# ---------------------------------------------------------

def _add_rate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    grouped = result.groupby(
        ["engine_id", "flight_id"],
        group_keys=False,
    )

    # -----------------------------------------------------
    # Per-sample change
    # -----------------------------------------------------

    result["egt_delta"] = (
        grouped["egt_c"]
        .diff()
    )

    result["cht_delta"] = (
        grouped["cht_c"]
        .diff()
    )

    result["oil_pressure_delta"] = (
        grouped["oil_pressure_kpa"]
        .diff()
    )

    result["vibration_delta"] = (
        grouped["engine_vibration_mm_s"]
        .diff()
    )

    # -----------------------------------------------------
    # Change per second
    # -----------------------------------------------------

    time_delta = (
        grouped["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # Avoid division by zero.
    time_delta = time_delta.replace(0, pd.NA)

    result["egt_rate"] = (
        result["egt_delta"] / time_delta
    )

    result["cht_rate"] = (
        result["cht_delta"] / time_delta
    )

    result["oil_pressure_rate"] = (
        result["oil_pressure_delta"] / time_delta
    )

    result["vibration_rate"] = (
        result["vibration_delta"] / time_delta
    )

    return result


# ---------------------------------------------------------
# Public function
# ---------------------------------------------------------

def add_temporal_features(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add rolling and temporal features to telemetry data.

    Rolling calculations are performed independently for
    each engine and flight so that data from one flight
    never leaks into another flight.

    Parameters
    ----------
    telemetry:
        Raw telemetry DataFrame.

    Returns
    -------
    pd.DataFrame
        Original telemetry plus temporal features.
    """

    result = _prepare_dataframe(telemetry)

    result = _add_rolling_features(result)

    result = _add_rate_features(result)

    return result