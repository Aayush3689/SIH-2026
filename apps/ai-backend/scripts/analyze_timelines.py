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


def safe_round(value, decimals: int = 2):
    if pd.isna(value):
        return None
    return round(float(value), decimals)


# ---------------------------------------------------------
# Engine timeline analysis
# ---------------------------------------------------------
def analyze_engine_timeline(
    engine_id: str,
    engine_df: pd.DataFrame,
) -> None:

    engine_df = engine_df.sort_values("timestamp").copy()

    first_row = engine_df.iloc[0]
    last_row = engine_df.iloc[-1]

    first_time = first_row["timestamp"]
    last_time = last_row["timestamp"]

    flights = engine_df["flight_id"].nunique()

    first_rul = first_row["rul_hours_remaining"]
    last_rul = last_row["rul_hours_remaining"]

    first_health = first_row["health_index"]
    last_health = last_row["health_index"]

    first_severity = first_row["fault_severity_score"]
    last_severity = last_row["fault_severity_score"]

    fault_states = (
        engine_df["fault_mode_ground_truth"]
        .value_counts()
        .to_dict()
    )

    print(f"\nEngine: {engine_id}")
    print("-" * 90)

    print(f"Telemetry start      : {first_time}")
    print(f"Telemetry end        : {last_time}")
    print(f"Number of flights    : {flights}")

    print("\nRUL")
    print(f"  First             : {safe_round(first_rul)} hours")
    print(f"  Last              : {safe_round(last_rul)} hours")
    print(
        f"  Change            : "
        f"{safe_round(last_rul - first_rul)} hours"
    )

    print("\nHealth Index")
    print(f"  First             : {safe_round(first_health)}")
    print(f"  Last              : {safe_round(last_health)}")
    print(
        f"  Change            : "
        f"{safe_round(last_health - first_health)}"
    )

    print("\nFault Severity")
    print(f"  First             : {safe_round(first_severity, 4)}")
    print(f"  Last              : {safe_round(last_severity, 4)}")
    print(
        f"  Change            : "
        f"{safe_round(last_severity - first_severity, 4)}"
    )

    print("\nFault Distribution")

    for fault, count in fault_states.items():
        percentage = count / len(engine_df) * 100

        print(
            f"  {str(fault):30} "
            f"{count:8,} "
            f"({percentage:6.2f}%)"
        )


# ---------------------------------------------------------
# Flight-level analysis
# ---------------------------------------------------------
def analyze_flights(engine_df: pd.DataFrame) -> None:

    print_section("FLIGHT-LEVEL TIMELINE")

    flight_summary = (
        engine_df
        .groupby(
            ["engine_id", "flight_id"],
            as_index=False,
        )
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            telemetry_points=("timestamp", "count"),
            starting_rul=("rul_hours_remaining", "first"),
            ending_rul=("rul_hours_remaining", "last"),
            starting_health=("health_index", "first"),
            ending_health=("health_index", "last"),
            starting_severity=("fault_severity_score", "first"),
            ending_severity=("fault_severity_score", "last"),
        )
        .sort_values(
            ["engine_id", "start_time"]
        )
    )

    print(
        flight_summary[
            [
                "engine_id",
                "flight_id",
                "start_time",
                "end_time",
                "telemetry_points",
                "starting_rul",
                "ending_rul",
                "starting_health",
                "ending_health",
                "starting_severity",
                "ending_severity",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# RUL monotonicity check
# ---------------------------------------------------------
def check_rul_trend(
    engine_id: str,
    engine_df: pd.DataFrame,
) -> None:

    engine_df = engine_df.sort_values("timestamp")

    rul = engine_df["rul_hours_remaining"]

    decreases = (rul.diff() < 0).sum()
    increases = (rul.diff() > 0).sum()
    unchanged = (rul.diff() == 0).sum()

    print(
        f"{engine_id:15} "
        f"decreases={decreases:6,}  "
        f"increases={increases:6,}  "
        f"unchanged={unchanged:6,}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:

    print_section("DRDO ENGINE TIMELINE ANALYSIS")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Engine-wise analysis
    # -----------------------------------------------------
    print_section("ENGINE-WISE TIMELINE")

    for engine_id, engine_df in telemetry.groupby("engine_id"):

        analyze_engine_timeline(
            engine_id,
            engine_df,
        )

    # -----------------------------------------------------
    # RUL behavior
    # -----------------------------------------------------
    print_section("RUL TEMPORAL BEHAVIOR")

    print(
        f"{'Engine':15} "
        f"{'Decreases':>12} "
        f"{'Increases':>12} "
        f"{'Unchanged':>12}"
    )

    print("-" * 55)

    for engine_id, engine_df in telemetry.groupby("engine_id"):

        check_rul_trend(
            engine_id,
            engine_df,
        )

    # -----------------------------------------------------
    # Flight-level information
    # -----------------------------------------------------
    analyze_flights(telemetry)

    print_section("TIMELINE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()