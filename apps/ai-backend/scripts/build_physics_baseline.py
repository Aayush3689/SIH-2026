import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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
# Helpers
# ---------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def validate_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print_section("EGT PHYSICS BASELINE")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    validate_columns(
        telemetry,
        FEATURE_COLUMNS + [TARGET_COLUMN],
    )

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    model_data = telemetry[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].copy()

    # Remove rows containing NaN/inf values.
    model_data = model_data.dropna()

    X = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN]

    print(f"Total usable samples : {len(model_data):,}")

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------
    #
    # This is only a BASELINE experiment.
    #
    # Later we will replace random split with
    # engine-wise / time-aware validation.
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training samples      : {len(X_train):,}")
    print(f"Testing samples       : {len(X_test):,}")

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

    print("EGT baseline training complete.")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    print_section("BASELINE RESULTS")

    print(f"MAE : {mae:.3f} °C")
    print(f"R²  : {r2:.4f}")

    # -----------------------------------------------------
    # Example predictions
    # -----------------------------------------------------

    results = X_test.copy()

    results["actual_egt"] = y_test
    results["expected_egt"] = y_pred
    results["egt_residual"] = (
        results["actual_egt"]
        - results["expected_egt"]
    )

    print_section("SAMPLE PREDICTIONS")

    print(
        results[
            [
                "actual_egt",
                "expected_egt",
                "egt_residual",
            ]
        ]
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Residual statistics
    # -----------------------------------------------------

    print_section("RESIDUAL STATISTICS")

    print(
        results["egt_residual"]
        .describe()
        .round(3)
        .to_string()
    )

    print_section("EGT PHYSICS BASELINE COMPLETE")


if __name__ == "__main__":
    main()