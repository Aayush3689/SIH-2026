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

TARGET_COLUMN = "cht_c"

REFERENCE_COLUMNS = [
    "fault_mode_ground_truth",
    "fault_severity_score",
    "health_index",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def validate_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print_section(
        "ALL-FAULT CHT RESIDUAL ANALYSIS"
    )

    dataset = load_dataset()

    telemetry = dataset["telemetry"].copy()

    validate_columns(
        telemetry,
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + REFERENCE_COLUMNS,
    )

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    columns = (
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "fault_mode_ground_truth",
            "fault_severity_score",
            "health_index",
        ]
    )

    data = telemetry[columns].dropna().copy()

    print(f"Usable samples: {len(data):,}")

    # -----------------------------------------------------
    # Train expected CHT baseline
    #
    # This is exploratory residual analysis only.
    # It is NOT final model evaluation.
    # -----------------------------------------------------

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X, y)

    data["expected_cht"] = model.predict(X)

    data["cht_residual"] = (
        data["cht_c"]
        - data["expected_cht"]
    )

    data["absolute_cht_residual"] = (
        data["cht_residual"].abs()
    )

    # -----------------------------------------------------
    # Fault-wise residual statistics
    # -----------------------------------------------------

    print_section(
        "FAULT-WISE CHT RESIDUALS"
    )

    fault_summary = (
        data
        .groupby("fault_mode_ground_truth")
        .agg(
            samples=("cht_residual", "count"),
            mean_residual=("cht_residual", "mean"),
            median_residual=("cht_residual", "median"),
            std_residual=("cht_residual", "std"),
            mean_absolute_residual=(
                "absolute_cht_residual",
                "mean",
            ),
            p95_absolute_residual=(
                "absolute_cht_residual",
                lambda x: x.quantile(0.95),
            ),
            max_absolute_residual=(
                "absolute_cht_residual",
                "max",
            ),
        )
        .sort_values(
            "mean_absolute_residual",
            ascending=False,
        )
        .round(3)
    )

    print(
        fault_summary.to_string()
    )

    # -----------------------------------------------------
    # Residual vs severity
    # -----------------------------------------------------

    print_section(
        "CHT RESIDUAL vs FAULT SEVERITY"
    )

    data["severity_bin"] = pd.cut(
        data["fault_severity_score"],
        bins=[
            -0.0001,
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        labels=[
            "0",
            "0-0.1",
            "0.1-0.25",
            "0.25-0.5",
            "0.5-0.75",
            "0.75-1.0",
        ],
    )

    severity_summary = (
        data
        .groupby(
            "severity_bin",
            observed=False,
        )
        .agg(
            samples=("cht_residual", "count"),
            mean_residual=("cht_residual", "mean"),
            mean_absolute_residual=(
                "absolute_cht_residual",
                "mean",
            ),
            p95_absolute_residual=(
                "absolute_cht_residual",
                lambda x: x.quantile(0.95),
            ),
        )
        .round(3)
    )

    print(
        severity_summary.to_string()
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    print_section(
        "CORRELATION WITH CHT RESIDUAL"
    )

    correlation_columns = [
        "cht_residual",
        "absolute_cht_residual",
        "fault_severity_score",
        "health_index",
    ]

    correlation_matrix = (
        data[correlation_columns]
        .corr()
        .round(3)
    )

    print(
        correlation_matrix.to_string()
    )

    # -----------------------------------------------------
    # Fault-wise severity relationship
    # -----------------------------------------------------

    print_section(
        "FAULT-WISE RESIDUAL vs SEVERITY CORRELATION"
    )

    faults = sorted(
        data["fault_mode_ground_truth"]
        .dropna()
        .unique()
    )

    for fault in faults:

        fault_data = data[
            data["fault_mode_ground_truth"] == fault
        ]

        residual_severity_corr = (
            fault_data[
                [
                    "cht_residual",
                    "fault_severity_score",
                ]
            ]
            .corr()
            .loc[
                "cht_residual",
                "fault_severity_score",
            ]
        )

        absolute_residual_severity_corr = (
            fault_data[
                [
                    "absolute_cht_residual",
                    "fault_severity_score",
                ]
            ]
            .corr()
            .loc[
                "absolute_cht_residual",
                "fault_severity_score",
            ]
        )

        print(
            f"{fault:30}"
            f" residual↔severity = "
            f"{residual_severity_corr:7.3f}   "
            f"|residual|↔severity = "
            f"{absolute_residual_severity_corr:7.3f}"
        )

    # -----------------------------------------------------
    # Actual vs expected CHT
    # -----------------------------------------------------

    print_section(
        "ACTUAL vs EXPECTED CHT BY FAULT"
    )

    cht_summary = (
        data
        .groupby("fault_mode_ground_truth")
        .agg(
            actual_cht=("cht_c", "mean"),
            expected_cht=("expected_cht", "mean"),
            mean_residual=("cht_residual", "mean"),
        )
        .round(3)
    )

    print(
        cht_summary.to_string()
    )

    print_section(
        "ALL-FAULT CHT RESIDUAL ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()