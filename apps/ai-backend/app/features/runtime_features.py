from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# REQUIRED RAW TELEMETRY FIELDS
# =============================================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "engine_id",
    "flight_id",

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

    "elapsed_s",
]


# =============================================================================
# RAW FEATURES
# =============================================================================

RAW_FEATURES = [
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
    "elapsed_s",
]


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

TEMPORAL_FEATURES = [
    "vibration_mean_30s",
    "vibration_std_30s",

    "rpm_mean_30s",
    "rpm_std_30s",

    "egt_mean_60s",
    "egt_std_60s",

    "cht_mean_60s",
    "cht_std_60s",

    "oil_pressure_mean_60s",
    "oil_pressure_std_60s",

    "oil_temperature_mean_60s",
    "oil_temperature_std_60s",

    "egt_delta",
    "cht_delta",
    "oil_pressure_delta",
    "vibration_delta",

    "egt_rate",
    "cht_rate",
    "oil_pressure_rate",
    "vibration_rate",
]


# =============================================================================
# PHYSICS FEATURES
# =============================================================================

PHYSICS_FEATURES = [
    "egt_expected",
    "egt_residual",

    "cht_expected",
    "cht_residual",

    "oil_temperature_expected",
    "oil_temperature_residual",
]


# =============================================================================
# TRAINING-TIME ROLLING CONTRACT
# =============================================================================
#
# These definitions reproduce the common temporal feature logic used by the
# training pipeline.
#
# Important:
#   - Rolling is performed separately for each engine + flight.
#   - Time windows are timestamp-based.
#   - mean uses min_periods=1
#   - std uses min_periods=2
#

ROLLING_SPECS = {
    "vibration_mean_30s": (
        "engine_vibration_mm_s",
        "mean",
        "30s",
        1,
    ),

    "vibration_std_30s": (
        "engine_vibration_mm_s",
        "std",
        "30s",
        2,
    ),

    "rpm_mean_30s": (
        "rpm",
        "mean",
        "30s",
        1,
    ),

    "rpm_std_30s": (
        "rpm",
        "std",
        "30s",
        2,
    ),

    "egt_mean_60s": (
        "egt_c",
        "mean",
        "60s",
        1,
    ),

    "egt_std_60s": (
        "egt_c",
        "std",
        "60s",
        2,
    ),

    "cht_mean_60s": (
        "cht_c",
        "mean",
        "60s",
        1,
    ),

    "cht_std_60s": (
        "cht_c",
        "std",
        "60s",
        2,
    ),

    "oil_pressure_mean_60s": (
        "oil_pressure_kpa",
        "mean",
        "60s",
        1,
    ),

    "oil_pressure_std_60s": (
        "oil_pressure_kpa",
        "std",
        "60s",
        2,
    ),

    "oil_temperature_mean_60s": (
        "oil_temperature_c",
        "mean",
        "60s",
        1,
    ),

    "oil_temperature_std_60s": (
        "oil_temperature_c",
        "std",
        "60s",
        2,
    ),
}


# =============================================================================
# DELTA FEATURES
# =============================================================================

DELTA_SPECS = {
    "egt_c": "egt_delta",
    "cht_c": "cht_delta",
    "oil_pressure_kpa": "oil_pressure_delta",
    "engine_vibration_mm_s": "vibration_delta",
}


# =============================================================================
# RATE FEATURES
# =============================================================================

RATE_SPECS = {
    "egt_delta": "egt_rate",
    "cht_delta": "cht_rate",
    "oil_pressure_delta": "oil_pressure_rate",
    "vibration_delta": "vibration_rate",
}


# =============================================================================
# RUNTIME FEATURE BUILDER
# =============================================================================

@dataclass
class RuntimeFeatureBuilder:
    """
    Build shared runtime temporal features.

    Important:
        - Input must contain the required telemetry fields.
        - Rows are sorted by engine_id, flight_id and timestamp.
        - Rolling features reset at every engine + flight boundary.
        - Physics features are generated separately.
    """

    min_samples: int = 1

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(
        self,
        df: pd.DataFrame,
    ) -> None:

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "Telemetry must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Telemetry dataframe is empty."
            )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing telemetry columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

        if df["engine_id"].isna().any():
            raise ValueError(
                "engine_id contains missing values."
            )

        if df["flight_id"].isna().any():
            raise ValueError(
                "flight_id contains missing values."
            )

        timestamp = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if timestamp.isna().any():
            raise ValueError(
                "timestamp contains invalid values."
            )

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    def build_temporal(
        self,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build temporal features using the training-time definitions.
        """

        self.validate(
            telemetry
        )

        df = telemetry.copy()

        # ---------------------------------------------------------------------
        # Timestamp normalization
        # ---------------------------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        # ---------------------------------------------------------------------
        # Stable chronological ordering
        # ---------------------------------------------------------------------

        df = (
            df.sort_values(
                [
                    "engine_id",
                    "flight_id",
                    "timestamp",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # Rolling features
        # ---------------------------------------------------------------------
        #
        # We calculate the rolling Series and then use positional assignment
        # with to_numpy().
        #
        # Why?
        # timestamp values can be duplicated across different engines/flights.
        # Direct pandas index alignment can then fail with:
        #
        #   ValueError: cannot reindex on an axis with duplicate labels
        #
        # Positional assignment avoids that problem because the dataframe and
        # rolling result are already in the same sorted row order.
        #

        grouped = (
            df.set_index(
                "timestamp"
            )
            .groupby(
                [
                    "engine_id",
                    "flight_id",
                ],
                group_keys=False,
            )
        )

        for (
            feature_name,
            (
                source_column,
                aggregation,
                window,
                min_periods,
            ),
        ) in ROLLING_SPECS.items():

            rolling = grouped[
                source_column
            ].rolling(
                window,
                min_periods=min_periods,
            )

            if aggregation == "mean":

                values = rolling.mean()

            elif aggregation == "std":

                values = rolling.std()

            else:

                raise ValueError(
                    f"Unsupported aggregation: "
                    f"{aggregation}"
                )

            values = values.reset_index(
                level=[
                    0,
                    1,
                ],
                drop=True,
            )

            if len(values) != len(df):
                raise RuntimeError(
                    f"Rolling feature length mismatch "
                    f"for '{feature_name}': "
                    f"values={len(values)}, "
                    f"rows={len(df)}"
                )

            # IMPORTANT:
            # Positional assignment.
            df[feature_name] = (
                values.to_numpy()
            )

        # ---------------------------------------------------------------------
        # Delta features
        # ---------------------------------------------------------------------

        group = df.groupby(
            [
                "engine_id",
                "flight_id",
            ],
            sort=False,
        )

        for (
            source_column,
            output_column,
        ) in DELTA_SPECS.items():

            df[output_column] = (
                group[source_column].diff()
            )

        # ---------------------------------------------------------------------
        # Rate features
        # ---------------------------------------------------------------------

        delta_t = (
            group["elapsed_s"]
            .diff()
        )

        delta_t = delta_t.replace(
            0,
            np.nan,
        )

        for (
            source_column,
            output_column,
        ) in RATE_SPECS.items():

            df[output_column] = (
                df[source_column]
                / delta_t
            )

        return df


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def build_temporal_features(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function used by inference.
    """

    builder = RuntimeFeatureBuilder()

    return builder.build_temporal(
        telemetry
    )