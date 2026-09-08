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
# Columns we want to investigate
# ---------------------------------------------------------
SENSOR_COLUMNS = [
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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_fault_summary(
    telemetry: pd.DataFrame,
    fault: str,
) -> None:
    fault_df = telemetry[
        telemetry["fault_mode_ground_truth"] == fault
    ]

    stats = (
        fault_df[SENSOR_COLUMNS]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
        .round(3)
    )

    print(f"\nFault: {fault}")
    print(f"Samples: {len(fault_df):,}")
    print("-" * 90)

    print(stats.to_string())


# ---------------------------------------------------------
# Compare fault vs normal
# ---------------------------------------------------------
def compare_with_normal(
    telemetry: pd.DataFrame,
    fault: str,
) -> pd.DataFrame:

    normal_df = telemetry[
        telemetry["fault_mode_ground_truth"] == "none"
    ]

    fault_df = telemetry[
        telemetry["fault_mode_ground_truth"] == fault
    ]

    normal_mean = normal_df[SENSOR_COLUMNS].mean()
    fault_mean = fault_df[SENSOR_COLUMNS].mean()

    comparison = pd.DataFrame(
        {
            "normal_mean": normal_mean,
            "fault_mean": fault_mean,
        }
    )

    comparison["difference"] = (
        comparison["fault_mean"]
        - comparison["normal_mean"]
    )

    # Percentage change relative to normal.
    comparison["pct_change"] = (
        comparison["difference"]
        / comparison["normal_mean"].replace(0, pd.NA)
        * 100
    )

    return comparison.round(3)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:

    print_section("DRDO FAULT SIGNATURE ANALYSIS")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    # -----------------------------------------------------
    # Fault distribution
    # -----------------------------------------------------
    print_section("AVAILABLE FAULT MODES")

    fault_modes = (
        telemetry["fault_mode_ground_truth"]
        .dropna()
        .unique()
    )

    for fault in sorted(fault_modes):
        count = (
            telemetry["fault_mode_ground_truth"] == fault
        ).sum()

        print(f"{fault:30} {count:8,}")

    # -----------------------------------------------------
    # Individual fault statistics
    # -----------------------------------------------------
    print_section("FAULT-WISE SENSOR STATISTICS")

    for fault in sorted(fault_modes):
        print_fault_summary(
            telemetry,
            fault,
        )

    # -----------------------------------------------------
    # Compare each fault with normal
    # -----------------------------------------------------
    print_section("FAULT vs NORMAL COMPARISON")

    for fault in sorted(fault_modes):

        if fault == "none":
            continue

        comparison = compare_with_normal(
            telemetry,
            fault,
        )

        print(f"\n\nFAULT: {fault}")
        print("-" * 90)

        print(
            comparison.to_string()
        )

    print_section("FAULT SIGNATURE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()