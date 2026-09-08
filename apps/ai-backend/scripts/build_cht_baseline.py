from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.dataset_loader import load_dataset


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

TARGET_COLUMN = "cht_c"

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


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    print_section("ENGINE-WISE CHT PHYSICS BASELINE")

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    columns = [
        "engine_id",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    data = telemetry[columns].dropna().copy()

    train_df = data[
        data["engine_id"].isin(TRAIN_ENGINES)
    ].copy()

    test_df = data[
        data["engine_id"].isin(TEST_ENGINES)
    ].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Training samples : {len(train_df):,}")
    print(f"Test samples     : {len(test_df):,}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    print_section("TRAINING")

    model.fit(X_train, y_train)

    print("CHT baseline trained.")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print_section("UNSEEN-ENGINE RESULTS")

    print(f"MAE : {mae:.3f} °C")
    print(f"R²  : {r2:.4f}")

    results = test_df[
        [
            "engine_id",
            TARGET_COLUMN,
        ]
    ].copy()

    results["expected_cht"] = y_pred

    results["cht_residual"] = (
        results["cht_c"]
        - results["expected_cht"]
    )

    print_section("RESIDUAL STATISTICS")

    print(
        results["cht_residual"]
        .describe()
        .round(3)
        .to_string()
    )

    print_section("PER-ENGINE PERFORMANCE")

    for engine_id in TEST_ENGINES:

        engine_results = results[
            results["engine_id"] == engine_id
        ]

        if engine_results.empty:
            continue

        engine_mae = mean_absolute_error(
            engine_results["cht_c"],
            engine_results["expected_cht"],
        )

        engine_r2 = r2_score(
            engine_results["cht_c"],
            engine_results["expected_cht"],
        )

        print(
            f"{engine_id:15}"
            f"MAE = {engine_mae:8.3f} °C   "
            f"R² = {engine_r2:8.4f}"
        )

    print_section("SAMPLE PREDICTIONS")

    print(
        results[
            [
                "engine_id",
                "cht_c",
                "expected_cht",
                "cht_residual",
            ]
        ]
        .head(20)
        .round(3)
        .to_string(index=False)
    )

    print_section("CHT BASELINE COMPLETE")


if __name__ == "__main__":
    main()