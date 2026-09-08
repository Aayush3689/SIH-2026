from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
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

# ---------------------------------------------------------
# IMPORTANT
# These are ENGINE IDs, not feature values.
#
# The test engines must never appear in training.
# ---------------------------------------------------------

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


def validate_engine_split(
    telemetry: pd.DataFrame,
) -> None:

    available_engines = set(
        telemetry["engine_id"].unique()
    )

    train_engines = set(TRAIN_ENGINES)
    test_engines = set(TEST_ENGINES)

    unknown_train = train_engines - available_engines
    unknown_test = test_engines - available_engines

    if unknown_train:
        raise ValueError(
            f"Unknown training engines: {sorted(unknown_train)}"
        )

    if unknown_test:
        raise ValueError(
            f"Unknown test engines: {sorted(unknown_test)}"
        )

    overlap = train_engines & test_engines

    if overlap:
        raise ValueError(
            f"Engine leakage detected. "
            f"Train/test overlap: {sorted(overlap)}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print_section(
        "ENGINE-WISE EGT PHYSICS BASELINE EVALUATION"
    )

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    # -----------------------------------------------------
    # Validate split
    # -----------------------------------------------------

    validate_engine_split(telemetry)

    print("Training engines:")
    for engine in TRAIN_ENGINES:
        print(f"  - {engine}")

    print("\nUnseen test engines:")
    for engine in TEST_ENGINES:
        print(f"  - {engine}")

    # -----------------------------------------------------
    # Select data
    # -----------------------------------------------------

    model_data = telemetry[
        [
            "engine_id",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].dropna()

    train_df = model_data[
        model_data["engine_id"].isin(TRAIN_ENGINES)
    ].copy()

    test_df = model_data[
        model_data["engine_id"].isin(TEST_ENGINES)
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print_section("DATA SPLIT")

    print(
        f"Training samples : {len(train_df):,}"
    )

    print(
        f"Test samples     : {len(test_df):,}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    print_section("TRAINING")

    model.fit(
        X_train,
        y_train,
    )

    print("EGT baseline trained.")

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    print_section("UNSEEN-ENGINE RESULTS")

    print(f"MAE : {mae:.3f} °C")
    print(f"R²  : {r2:.4f}")

    # -----------------------------------------------------
    # Residuals
    # -----------------------------------------------------

    results = test_df[
        [
            "engine_id",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    results["expected_egt"] = y_pred

    results["egt_residual"] = (
        results[TARGET_COLUMN]
        - results["expected_egt"]
    )

    print_section("RESIDUAL STATISTICS")

    print(
        results["egt_residual"]
        .describe()
        .round(3)
        .to_string()
    )

    # -----------------------------------------------------
    # Per-engine performance
    # -----------------------------------------------------

    print_section(
        "PER-ENGINE TEST PERFORMANCE"
    )

    for engine_id in TEST_ENGINES:

        engine_results = results[
            results["engine_id"] == engine_id
        ]

        if engine_results.empty:
            continue

        engine_mae = mean_absolute_error(
            engine_results[TARGET_COLUMN],
            engine_results["expected_egt"],
        )

        engine_r2 = r2_score(
            engine_results[TARGET_COLUMN],
            engine_results["expected_egt"],
        )

        print(
            f"{engine_id:15}"
            f"MAE = {engine_mae:8.3f} °C   "
            f"R² = {engine_r2:8.4f}"
        )

    # -----------------------------------------------------
    # Sample predictions
    # -----------------------------------------------------

    print_section(
        "SAMPLE UNSEEN-ENGINE PREDICTIONS"
    )

    print(
        results[
            [
                "engine_id",
                TARGET_COLUMN,
                "expected_egt",
                "egt_residual",
            ]
        ]
        .head(20)
        .round(3)
        .to_string(index=False)
    )

    print_section(
        "ENGINE-WISE EGT BASELINE EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()