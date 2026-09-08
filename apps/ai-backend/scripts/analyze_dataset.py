import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.dataset_loader import load_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TELEMETRY_NUMERIC_COLUMNS = [
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
]

TARGET_COLUMNS = [
    "fault_mode_ground_truth",
    "fault_severity_score",
    "fault_active_label",
    "health_index",
    "rul_hours_remaining",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_statistics(df: pd.DataFrame, columns: list[str]) -> None:
    available_columns = [
        column for column in columns if column in df.columns
    ]

    stats = df[available_columns].describe().T

    print(
        stats[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ].round(3).to_string()
    )


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def main() -> None:
    print_section("DRDO UAV ENGINE DATASET ANALYSIS")

    dataset = load_dataset()

    metadata = dataset["metadata"]
    telemetry = dataset["telemetry"]
    missions = dataset["missions"]

    # -----------------------------------------------------
    # Dataset overview
    # -----------------------------------------------------
    print_section("DATASET OVERVIEW")

    print(f"Metadata rows   : {len(metadata):,}")
    print(f"Telemetry rows  : {len(telemetry):,}")
    print(f"Mission rows    : {len(missions):,}")

    print(f"Engines         : {metadata['engine_id'].nunique()}")
    print(f"UAVs            : {metadata['uav_id'].nunique()}")
    print(f"Flights         : {telemetry['flight_id'].nunique()}")

    # -----------------------------------------------------
    # Timestamp information
    # -----------------------------------------------------
    print_section("TELEMETRY TIME RANGE")

    timestamps = pd.to_datetime(
        telemetry["timestamp"],
        errors="coerce",
    )

    print(f"Start : {timestamps.min()}")
    print(f"End   : {timestamps.max()}")
    print(f"Span  : {timestamps.max() - timestamps.min()}")

    # -----------------------------------------------------
    # Numeric statistics
    # -----------------------------------------------------
    print_section("ENGINE TELEMETRY STATISTICS")

    print_statistics(
        telemetry,
        TELEMETRY_NUMERIC_COLUMNS,
    )

    # -----------------------------------------------------
    # Fault distribution
    # -----------------------------------------------------
    print_section("FAULT MODE DISTRIBUTION")

    fault_distribution = (
        telemetry["fault_mode_ground_truth"]
        .value_counts(dropna=False)
    )

    for fault_mode, count in fault_distribution.items():
        percentage = count / len(telemetry) * 100

        print(
            f"{str(fault_mode):30} "
            f"{count:8,} "
            f"({percentage:6.2f}%)"
        )

    # -----------------------------------------------------
    # Fault severity
    # -----------------------------------------------------
    print_section("FAULT SEVERITY")

    if "fault_severity_score" in telemetry.columns:
        print(
            telemetry["fault_severity_score"]
            .describe()
            .round(4)
            .to_string()
        )

    # -----------------------------------------------------
    # Health index
    # -----------------------------------------------------
    print_section("HEALTH INDEX")

    if "health_index" in telemetry.columns:
        print(
            telemetry["health_index"]
            .describe()
            .round(4)
            .to_string()
        )

    # -----------------------------------------------------
    # RUL
    # -----------------------------------------------------
    print_section("RUL DISTRIBUTION")

    if "rul_hours_remaining" in telemetry.columns:
        print(
            telemetry["rul_hours_remaining"]
            .describe()
            .round(4)
            .to_string()
        )

    # -----------------------------------------------------
    # Flight phase
    # -----------------------------------------------------
    print_section("FLIGHT PHASE DISTRIBUTION")

    if "flight_phase" in telemetry.columns:
        phase_distribution = (
            telemetry["flight_phase"]
            .value_counts(dropna=False)
        )

        for phase, count in phase_distribution.items():
            percentage = count / len(telemetry) * 100

            print(
                f"{str(phase):20} "
                f"{count:8,} "
                f"({percentage:6.2f}%)"
            )

    # -----------------------------------------------------
    # Engine-wise telemetry count
    # -----------------------------------------------------
    print_section("ENGINE-WISE TELEMETRY")

    engine_counts = (
        telemetry["engine_id"]
        .value_counts()
        .sort_index()
    )

    for engine_id, count in engine_counts.items():
        print(f"{engine_id:15} {count:8,}")

    # -----------------------------------------------------
    # Engine-wise fault mode
    # -----------------------------------------------------
    print_section("ENGINE-WISE ASSIGNED FAULT")

    engine_faults = (
        metadata[
            [
                "engine_id",
                "uav_id",
                "assigned_fault_mode",
            ]
        ]
        .sort_values("engine_id")
    )

    print(engine_faults.to_string(index=False))

    # -----------------------------------------------------
    # Target / reference columns
    # -----------------------------------------------------
    print_section("TARGET / REFERENCE COLUMNS")

    for column in TARGET_COLUMNS:
        if column in telemetry.columns:
            print(f"  - {column}")

    print_section("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()