from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.dataset_loader import load_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "egt_c"

TRAIN_ENGINES = [
    "ENG-2023000",
    "ENG-2023001",
    "ENG-2023002",
    "ENG-2023003",
    "ENG-2023004",
    "ENG-2023005",
    "ENG-2023006",
]

TEST_ENGINES = [
    "ENG-2023007",
    "ENG-2023008",
    "ENG-2023009",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def validate_split(telemetry: pd.DataFrame) -> None:
    available_engines = set(telemetry["engine_id"].unique())

    unknown_train = set(TRAIN_ENGINES) - available_engines
    unknown_test = set(TEST_ENGINES) - available_engines

    if unknown_train:
        raise ValueError(
            f"Unknown training engines: {sorted(unknown_train)}"
        )

    if unknown_test:
        raise ValueError(
            f"Unknown test engines: {sorted(unknown_test)}"
        )

    overlap = set(TRAIN_ENGINES) & set(TEST_ENGINES)

    if overlap:
        raise ValueError(
            f"Train/test engine overlap detected: {sorted(overlap)}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print_section("EGT PHYSICS RESIDUAL ANALYSIS")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    validate_split(telemetry)

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    columns = [
        "engine_id",
        "fault_mode_ground_truth",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    model_data = telemetry[columns].dropna().copy()

    train_df = model_data[
        model_data["engine_id"].isin(TRAIN_ENGINES)
    ].copy()

    test_df = model_data[
        model_data["engine_id"].isin(TEST_ENGINES)
    ].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Training samples : {len(train_df):,}")
    print(f"Test samples     : {len(test_df):,}")

    # -----------------------------------------------------
    # Train expected-EGT model
    # -----------------------------------------------------

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    print("\nExpected EGT model trained.")

    # -----------------------------------------------------
    # Predict expected EGT
    # -----------------------------------------------------

    test_df["expected_egt"] = model.predict(X_test)

    test_df["egt_residual"] = (
        test_df["egt_c"]
        - test_df["expected_egt"]
    )

    test_df["absolute_egt_residual"] = (
        test_df["egt_residual"].abs()
    )

    # -----------------------------------------------------
    # Overall residual statistics
    # -----------------------------------------------------

    print_section("OVERALL RESIDUAL STATISTICS")

    overall = test_df["egt_residual"].describe()

    print(
        overall[
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
        ]
        .round(3)
        .to_string()
    )

    # -----------------------------------------------------
    # Fault-wise residual statistics
    # -----------------------------------------------------

    print_section("FAULT-WISE RESIDUAL STATISTICS")

    fault_summary = (
        test_df
        .groupby("fault_mode_ground_truth")
        .agg(
            samples=("egt_residual", "count"),
            mean_residual=("egt_residual", "mean"),
            median_residual=("egt_residual", "median"),
            std_residual=("egt_residual", "std"),
            min_residual=("egt_residual", "min"),
            max_residual=("egt_residual", "max"),
            mean_absolute_residual=(
                "absolute_egt_residual",
                "mean",
            ),
            p95_absolute_residual=(
                "absolute_egt_residual",
                lambda x: x.quantile(0.95),
            ),
        )
        .sort_values("mean_absolute_residual", ascending=False)
        .round(3)
    )

    print(
        fault_summary.to_string()
    )

    # -----------------------------------------------------
    # Positive residual analysis
    # -----------------------------------------------------
    #
    # Positive residual means:
    #
    # Actual EGT > Expected EGT
    #
    # Negative residual means:
    #
    # Actual EGT < Expected EGT
    # -----------------------------------------------------

    print_section("POSITIVE RESIDUAL RATE")

    positive_rate = (
        test_df
        .groupby("fault_mode_ground_truth")["egt_residual"]
        .apply(lambda x: (x > 0).mean() * 100)
        .sort_values(ascending=False)
        .round(2)
    )

    for fault, rate in positive_rate.items():
        print(
            f"{fault:30} "
            f"{rate:6.2f}% positive residual"
        )

    # -----------------------------------------------------
    # Large residual rate
    # -----------------------------------------------------
    #
    # Threshold is intentionally just a diagnostic value.
    # We are NOT defining 20°C as a fault threshold.
    # -----------------------------------------------------

    RESIDUAL_THRESHOLD_C = 20.0

    print_section(
        f"LARGE RESIDUAL RATE (>|RESIDUAL| > {RESIDUAL_THRESHOLD_C}°C)"
    )

    large_residual_rate = (
        test_df
        .groupby("fault_mode_ground_truth")[
            "absolute_egt_residual"
        ]
        .apply(
            lambda x: (x > RESIDUAL_THRESHOLD_C).mean()
            * 100
        )
        .sort_values(ascending=False)
        .round(2)
    )

    for fault, rate in large_residual_rate.items():
        print(
            f"{fault:30} "
            f"{rate:6.2f}% large residual"
        )

    # -----------------------------------------------------
    # Sample residuals
    # -----------------------------------------------------

    print_section("SAMPLE RESIDUALS")

    print(
        test_df[
            [
                "engine_id",
                "fault_mode_ground_truth",
                "egt_c",
                "expected_egt",
                "egt_residual",
            ]
        ]
        .head(20)
        .round(3)
        .to_string(index=False)
    )

    print_section("EGT PHYSICS RESIDUAL ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()