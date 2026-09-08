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
# Helpers
# ---------------------------------------------------------
def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_engine_summary(
    engine_id: str,
    engine_df: pd.DataFrame,
) -> None:
    engine_df = engine_df.sort_values("timestamp").copy()

    # Approximate operating time from telemetry timestamps.
    # We use elapsed_s when available because it represents
    # time elapsed within a flight.
    operating_hours = (
        engine_df.groupby("flight_id")["elapsed_s"]
        .max()
        .sum()
        / 3600
    )

    # Flight-level values
    flight_df = (
        engine_df
        .groupby("flight_id", as_index=False)
        .agg(
            flight_start=("timestamp", "min"),
            flight_end=("timestamp", "max"),
            starting_rul=("rul_hours_remaining", "first"),
            ending_rul=("rul_hours_remaining", "last"),
            starting_health=("health_index", "first"),
            ending_health=("health_index", "last"),
            starting_severity=("fault_severity_score", "first"),
            ending_severity=("fault_severity_score", "last"),
        )
        .sort_values("flight_start")
    )

    print(f"\nEngine: {engine_id}")
    print("-" * 90)

    print(f"Flights                    : {len(flight_df)}")
    print(
        f"Approx. operating hours   : "
        f"{operating_hours:.2f}"
    )

    print(
        f"First recorded RUL        : "
        f"{flight_df.iloc[0]['starting_rul']:.2f} h"
    )

    print(
        f"Last recorded RUL         : "
        f"{flight_df.iloc[-1]['ending_rul']:.2f} h"
    )

    print(
        f"Minimum recorded RUL      : "
        f"{flight_df['starting_rul'].min():.2f} h"
    )

    print(
        f"Maximum recorded RUL      : "
        f"{flight_df['starting_rul'].max():.2f} h"
    )

    print(
        f"First health              : "
        f"{flight_df.iloc[0]['starting_health']:.2f}"
    )

    print(
        f"Last health               : "
        f"{flight_df.iloc[-1]['ending_health']:.2f}"
    )

    print(
        f"First severity            : "
        f"{flight_df.iloc[0]['starting_severity']:.3f}"
    )

    print(
        f"Last severity             : "
        f"{flight_df.iloc[-1]['ending_severity']:.3f}"
    )

    # -----------------------------------------------------
    # RUL movement between chronological flights
    # -----------------------------------------------------
    flight_rul = flight_df["starting_rul"]

    rul_decreases = int((flight_rul.diff() < 0).sum())
    rul_increases = int((flight_rul.diff() > 0).sum())
    rul_unchanged = int((flight_rul.diff() == 0).sum())

    print("\nFlight-to-flight RUL changes")
    print(f"  Decreases                : {rul_decreases}")
    print(f"  Increases                : {rul_increases}")
    print(f"  Unchanged                : {rul_unchanged}")

    # -----------------------------------------------------
    # Correlations
    # -----------------------------------------------------
    correlation_columns = [
        "rul_hours_remaining",
        "health_index",
        "fault_severity_score",
    ]

    corr = engine_df[correlation_columns].corr()

    print("\nCorrelation matrix")
    print(corr.round(3).to_string())

    # -----------------------------------------------------
    # Flight-level table
    # -----------------------------------------------------
    print("\nFlight progression")

    display_df = flight_df[
        [
            "flight_start",
            "starting_rul",
            "ending_rul",
            "starting_health",
            "ending_health",
            "starting_severity",
            "ending_severity",
        ]
    ].copy()

    print(
        display_df
        .round(3)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Global RUL analysis
# ---------------------------------------------------------
def global_rul_analysis(
    telemetry: pd.DataFrame,
) -> None:

    print_section("GLOBAL RUL RELATIONSHIPS")

    columns = [
        "rul_hours_remaining",
        "health_index",
        "fault_severity_score",
    ]

    print("Overall correlations:")
    print(
        telemetry[columns]
        .corr()
        .round(3)
        .to_string()
    )

    print("\nAverage RUL by fault mode:")

    rul_by_fault = (
        telemetry
        .groupby("fault_mode_ground_truth")[
            "rul_hours_remaining"
        ]
        .agg(
            count="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
        )
        .sort_values("mean")
        .round(3)
    )

    print(rul_by_fault.to_string())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:

    print_section("DRDO RUL INVESTIGATION")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Engine-wise analysis
    # -----------------------------------------------------
    print_section("ENGINE-WISE RUL INVESTIGATION")

    for engine_id, engine_df in telemetry.groupby("engine_id"):
        print_engine_summary(
            engine_id,
            engine_df,
        )

    # -----------------------------------------------------
    # Global relationships
    # -----------------------------------------------------
    global_rul_analysis(telemetry)

    print_section("RUL INVESTIGATION COMPLETE")


if __name__ == "__main__":
    main()