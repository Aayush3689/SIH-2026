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

# Variables that describe current operating conditions.
OPERATING_COLUMNS = [
    "rpm",
    "throttle_position_pct",
    "engine_load_pct",
    "intake_manifold_pressure_kpa",
    "ambient_temp_c",
    "planned_altitude_ft",
    "airspeed_kts",
    "fuel_flow_lph",
    "fuel_pressure_kpa",
    "injection_timing_deg_btdc",
]

# Sensor outputs that we may want to estimate using
# the operating conditions.
OUTPUT_COLUMNS = [
    "egt_c",
    "cht_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_correlation(
    telemetry: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
) -> None:

    available_inputs = [
        column
        for column in input_columns
        if column in telemetry.columns
    ]

    available_outputs = [
        column
        for column in output_columns
        if column in telemetry.columns
    ]

    selected = available_inputs + available_outputs

    correlation = telemetry[selected].corr()

    print(
        correlation.loc[
            available_inputs,
            available_outputs,
        ]
        .round(3)
        .to_string()
    )


def print_top_relationships(
    telemetry: pd.DataFrame,
    input_columns: list[str],
    output_column: str,
) -> None:

    available_inputs = [
        column
        for column in input_columns
        if column in telemetry.columns
        and column != output_column
    ]

    correlations = (
        telemetry[available_inputs + [output_column]]
        .corr(numeric_only=True)[output_column]
        .drop(labels=[output_column])
    )

    correlations = (
        correlations
        .to_frame("correlation")
        .assign(abs_correlation=lambda df: df["correlation"].abs())
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
    )

    print(f"\nTarget: {output_column}")
    print("-" * 60)

    for feature, row in correlations.iterrows():
        print(
            f"{feature:35} "
            f"{row['correlation']:8.3f}"
        )


def print_faultwise_relationships(
    telemetry: pd.DataFrame,
    output_column: str,
) -> None:

    print(
        f"\nOutput: {output_column}"
    )

    summary = (
        telemetry
        .groupby("fault_mode_ground_truth")[
            output_column
        ]
        .agg(
            mean="mean",
            std="std",
            minimum="min",
            median="median",
            maximum="max",
        )
        .round(3)
    )

    print(summary.to_string())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print_section(
        "DRDO PHYSICS RELATIONSHIP ANALYSIS"
    )

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    # -----------------------------------------------------
    # Basic overview
    # -----------------------------------------------------

    print_section("ANALYSIS VARIABLES")

    print("Operating variables:")

    for column in OPERATING_COLUMNS:
        if column in telemetry.columns:
            print(f"  - {column}")

    print("\nPotential output variables:")

    for column in OUTPUT_COLUMNS:
        if column in telemetry.columns:
            print(f"  - {column}")

    # -----------------------------------------------------
    # Overall correlation
    # -----------------------------------------------------

    print_section(
        "OPERATING CONDITIONS vs ENGINE OUTPUTS"
    )

    print_correlation(
        telemetry,
        OPERATING_COLUMNS,
        OUTPUT_COLUMNS,
    )

    # -----------------------------------------------------
    # Top relationships for each output
    # -----------------------------------------------------

    print_section(
        "STRONGEST RELATIONSHIPS"
    )

    available_outputs = [
        column
        for column in OUTPUT_COLUMNS
        if column in telemetry.columns
    ]

    for output_column in available_outputs:

        print_top_relationships(
            telemetry,
            OPERATING_COLUMNS,
            output_column,
        )

    # -----------------------------------------------------
    # Fault-wise output behavior
    # -----------------------------------------------------

    print_section(
        "OUTPUT BEHAVIOR BY FAULT MODE"
    )

    for output_column in available_outputs:

        print_faultwise_relationships(
            telemetry,
            output_column,
        )

    print_section(
        "PHYSICS RELATIONSHIP ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()