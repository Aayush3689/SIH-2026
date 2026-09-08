from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CANONICAL BASE COLUMNS
# ============================================================

BASE_COLUMNS = [
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


# ============================================================
# HELPERS
# ============================================================

def _safe_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def _last_value(
    df: pd.DataFrame,
    column: str,
) -> float:

    series = _safe_series(
        df,
        column,
    )

    if series.empty:
        return np.nan

    return float(
        series.iloc[-1]
    )


def _mean(
    df: pd.DataFrame,
    column: str,
) -> float:

    series = _safe_series(
        df,
        column,
    )

    return float(
        series.mean()
    )


def _std(
    df: pd.DataFrame,
    column: str,
) -> float:

    series = _safe_series(
        df,
        column,
    )

    return float(
        series.std(
            ddof=0
        )
    )


def _rate(
    df: pd.DataFrame,
    column: str,
) -> float:

    if len(df) < 2:
        return 0.0

    values = _safe_series(
        df,
        column,
    ).to_numpy(
        dtype=float
    )

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    dt = (
        timestamps.diff()
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    valid = (
        np.isfinite(values)
        &
        np.isfinite(dt)
        &
        (dt > 0)
    )

    if valid.sum() < 1:
        return 0.0

    value_diff = np.diff(
        values
    )

    time_diff = dt[1:]

    valid_diff = (
        np.isfinite(value_diff)
        &
        np.isfinite(time_diff)
        &
        (time_diff > 0)
    )

    if valid_diff.sum() < 1:
        return 0.0

    rates = (
        value_diff[valid_diff]
        /
        time_diff[valid_diff]
    )

    if len(rates) == 0:
        return 0.0

    return float(
        np.mean(rates)
    )


def _rolling_mean(
    df: pd.DataFrame,
    column: str,
    window: int,
) -> float:

    series = _safe_series(
        df,
        column,
    )

    return float(
        series
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
        .iloc[-1]
    )


def _rolling_std(
    df: pd.DataFrame,
    column: str,
    window: int,
) -> float:

    series = _safe_series(
        df,
        column,
    )

    return float(
        series
        .rolling(
            window=window,
            min_periods=1,
        )
        .std(
            ddof=0
        )
        .iloc[-1]
    )


# ============================================================
# GENERIC WINDOW FEATURES
# ============================================================

def build_window_features(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row feature representation for the latest
    telemetry window.

    This is the common feature layer.

    Model-specific feature selection happens later.
    """

    if telemetry.empty:
        raise ValueError(
            "Cannot build features from empty telemetry."
        )

    row: dict[str, Any] = {}

    # --------------------------------------------------------
    # Last / mean / std
    # --------------------------------------------------------

    for column in BASE_COLUMNS:

        row[
            f"{column}_mean"
        ] = _mean(
            telemetry,
            column,
        )

        row[
            f"{column}_std"
        ] = _std(
            telemetry,
            column,
        )

        row[
            f"{column}_last"
        ] = _last_value(
            telemetry,
            column,
        )

    # --------------------------------------------------------
    # Standard rolling statistics used by the project
    # --------------------------------------------------------

    row[
        "vibration_mean_30s"
    ] = _rolling_mean(
        telemetry,
        "engine_vibration_mm_s",
        2,
    )

    row[
        "vibration_std_30s"
    ] = _rolling_std(
        telemetry,
        "engine_vibration_mm_s",
        2,
    )

    row[
        "rpm_mean_30s"
    ] = _rolling_mean(
        telemetry,
        "rpm",
        2,
    )

    row[
        "rpm_std_30s"
    ] = _rolling_std(
        telemetry,
        "rpm",
        2,
    )

    row[
        "egt_mean_60s"
    ] = _rolling_mean(
        telemetry,
        "egt_c",
        4,
    )

    row[
        "egt_std_60s"
    ] = _rolling_std(
        telemetry,
        "egt_c",
        4,
    )

    row[
        "cht_mean_60s"
    ] = _rolling_mean(
        telemetry,
        "cht_c",
        4,
    )

    row[
        "cht_std_60s"
    ] = _rolling_std(
        telemetry,
        "cht_c",
        4,
    )

    row[
        "oil_pressure_mean_60s"
    ] = _rolling_mean(
        telemetry,
        "oil_pressure_kpa",
        4,
    )

    row[
        "oil_pressure_std_60s"
    ] = _rolling_std(
        telemetry,
        "oil_pressure_kpa",
        4,
    )

    row[
        "oil_temperature_mean_60s"
    ] = _rolling_mean(
        telemetry,
        "oil_temperature_c",
        4,
    )

    row[
        "oil_temperature_std_60s"
    ] = _rolling_std(
        telemetry,
        "oil_temperature_c",
        4,
    )

    # --------------------------------------------------------
    # Rate features
    # --------------------------------------------------------

    row[
        "egt_rate"
    ] = _rate(
        telemetry,
        "egt_c",
    )

    row[
        "cht_rate"
    ] = _rate(
        telemetry,
        "cht_c",
    )

    row[
        "oil_pressure_rate"
    ] = _rate(
        telemetry,
        "oil_pressure_kpa",
    )

    row[
        "vibration_rate"
    ] = _rate(
        telemetry,
        "engine_vibration_mm_s",
    )

    return pd.DataFrame(
        [row]
    )