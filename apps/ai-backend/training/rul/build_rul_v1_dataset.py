from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

TRAINING_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "rul_features.csv"
)

OUTPUT_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "rul_v1"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rul_flight_level_v1.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

ENGINE = "engine_id"
FLIGHT = "flight_id"
TIMESTAMP = "timestamp"
TARGET = "rul_hours_remaining"

# Telemetry features that are meaningful for RUL.
BASE_FEATURES = [
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
    "egt_rate",
    "cht_rate",
    "oil_pressure_rate",
    "vibration_rate",
    "egt_expected",
    "egt_residual",
    "cht_expected",
    "cht_residual",
    "oil_temperature_expected",
    "oil_temperature_residual",
    "total_cycles_at_start",
    "last_overhaul_days_ago",
    "last_major_maintenance_days_ago",
]


# =============================================================================
# HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("BUILD RUL V1 FLIGHT-LEVEL DATASET")
    print("=" * 78)

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nInput rows   : {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns):,}"
    )

    # -------------------------------------------------------------------------
    # Validate columns
    # -------------------------------------------------------------------------

    required = {
        ENGINE,
        FLIGHT,
        TIMESTAMP,
        TARGET,
        *BASE_FEATURES,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        fail(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # -------------------------------------------------------------------------
    # Parse timestamp
    # -------------------------------------------------------------------------

    df["_time"] = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    invalid_timestamp = int(
        df["_time"].isna().sum()
    )

    if invalid_timestamp:
        fail(
            f"Invalid timestamps: "
            f"{invalid_timestamp}"
        )

    # -------------------------------------------------------------------------
    # Normalize IDs
    # -------------------------------------------------------------------------

    df[ENGINE] = (
        df[ENGINE]
        .astype(str)
        .str.strip()
    )

    df[FLIGHT] = (
        df[FLIGHT]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    df = (
        df.sort_values(
            [
                ENGINE,
                "_time",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    for feature in BASE_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if df[TARGET].isna().any():
        fail(
            f"Invalid target values in {TARGET}."
        )

    # -------------------------------------------------------------------------
    # Verify target is constant within each flight
    # -------------------------------------------------------------------------

    section("FLIGHT TARGET VALIDATION")

    target_nunique = (
        df.groupby(FLIGHT)[TARGET]
        .nunique()
    )

    non_constant = (
        target_nunique[
            target_nunique > 1
        ]
    )

    print(
        f"Total flights: "
        f"{len(target_nunique):,}"
    )

    print(
        f"Constant-target flights: "
        f"{(target_nunique == 1).sum():,}"
    )

    print(
        f"Non-constant-target flights: "
        f"{len(non_constant):,}"
    )

    if len(non_constant) > 0:
        fail(
            "RUL target is not constant within every flight."
        )

    # -------------------------------------------------------------------------
    # Flight-level aggregation
    # -------------------------------------------------------------------------

    section("FLIGHT-LEVEL AGGREGATION")

    rows = []

    grouped = df.groupby(
        [
            ENGINE,
            FLIGHT,
        ],
        sort=False,
    )

    for (
        engine_id,
        flight_id,
    ), group in grouped:

        group = group.sort_values(
            "_time"
        )

        row = {
            ENGINE: engine_id,
            FLIGHT: flight_id,
            "flight_start": group[
                "_time"
            ].iloc[0],
            "flight_end": group[
                "_time"
            ].iloc[-1],
            "sample_count": len(group),
            TARGET: float(
                group[TARGET].iloc[0]
            ),
        }

        # ---------------------------------------------------------------------
        # Aggregate each telemetry feature
        #
        # mean captures normal operating level
        # std captures instability / variability
        # last captures end-of-flight state
        # ---------------------------------------------------------------------

        for feature in BASE_FEATURES:

            series = group[
                feature
            ].dropna()

            if len(series) == 0:
                row[
                    f"{feature}_mean"
                ] = np.nan

                row[
                    f"{feature}_std"
                ] = np.nan

                row[
                    f"{feature}_last"
                ] = np.nan

            else:
                row[
                    f"{feature}_mean"
                ] = float(
                    series.mean()
                )

                row[
                    f"{feature}_std"
                ] = float(
                    series.std(
                        ddof=0
                    )
                )

                row[
                    f"{feature}_last"
                ] = float(
                    series.iloc[-1]
                )

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Flight duration
    # -------------------------------------------------------------------------

    result["flight_duration_min"] = (
        (
            result["flight_end"]
            - result["flight_start"]
        )
        .dt.total_seconds()
        / 60.0
    )

    # -------------------------------------------------------------------------
    # Engine-wise flight ordering
    # -------------------------------------------------------------------------

    result = result.sort_values(
        [
            ENGINE,
            "flight_start",
            FLIGHT,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    result["engine_flight_number"] = (
        result.groupby(
            ENGINE
        ).cumcount()
        + 1
    )

    # This field is for split generation / analysis only.
    # It will NOT be used as a model feature.
    result["engine_total_flights"] = (
        result.groupby(
            ENGINE
        )[FLIGHT]
        .transform("count")
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    section("OUTPUT VALIDATION")

    if result[FLIGHT].duplicated().any():
        fail(
            "Duplicate flight IDs in output."
        )

    if result[
        TARGET
    ].isna().any():
        fail(
            "Missing RUL target in output."
        )

    if (
        result[TARGET] < 0
    ).any():
        fail(
            "Negative RUL found."
        )

    feature_columns = [
        column
        for column in result.columns
        if column.endswith(
            (
                "_mean",
                "_std",
                "_last",
            )
        )
    ]

    if len(feature_columns) == 0:
        fail(
            "No aggregated feature columns generated."
        )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------

    print(
        f"Flights generated : "
        f"{len(result):,}"
    )

    print(
        f"Engines           : "
        f"{result[ENGINE].nunique():,}"
    )

    print(
        f"Aggregated features: "
        f"{len(feature_columns):,}"
    )

    print(
        f"RUL minimum       : "
        f"{result[TARGET].min():.2f} h"
    )

    print(
        f"RUL maximum       : "
        f"{result[TARGET].max():.2f} h"
    )

    print(
        f"RUL mean          : "
        f"{result[TARGET].mean():.2f} h"
    )

    print(
        f"RUL median        : "
        f"{result[TARGET].median():.2f} h"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print(
        "\nVerification: PASSED"
    )

    print(
        "\nRUL V1 DATASET BUILD COMPLETE."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("RUL V1 DATASET BUILD FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)