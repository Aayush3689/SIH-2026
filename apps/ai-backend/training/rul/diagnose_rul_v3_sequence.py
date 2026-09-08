from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "training/datasets/processed/rul_v1/rul_flight_level_v1.csv"
)

SPLIT_DIR = Path(
    "training/datasets/processed/rul_v1/split"
)

OUTPUT_DIR = Path(
    "training/datasets/processed/rul_v1/v3_diagnosis"
)

TARGET = "rul_hours_remaining"


# ============================================================
# CURRENT HEALTH FEATURES
# ============================================================

HEALTH_FEATURES = [
    "rpm_mean",
    "rpm_std",
    "rpm_last",

    "cht_c_mean",
    "cht_c_std",
    "cht_c_last",

    "egt_c_mean",
    "egt_c_std",
    "egt_c_last",

    "oil_pressure_kpa_mean",
    "oil_pressure_kpa_std",
    "oil_pressure_kpa_last",

    "oil_temperature_c_mean",
    "oil_temperature_c_std",
    "oil_temperature_c_last",

    "fuel_flow_lph_mean",
    "fuel_flow_lph_std",
    "fuel_flow_lph_last",

    "fuel_pressure_kpa_mean",
    "fuel_pressure_kpa_std",
    "fuel_pressure_kpa_last",

    "engine_vibration_mm_s_mean",
    "engine_vibration_mm_s_std",
    "engine_vibration_mm_s_last",

    "egt_rate_mean",
    "cht_rate_mean",
    "oil_pressure_rate_mean",
    "vibration_rate_mean",

    "egt_residual_mean",
    "cht_residual_mean",
    "oil_temperature_residual_mean",

    "total_cycles_at_start_last",
    "last_overhaul_days_ago_last",
    "last_major_maintenance_days_ago_last",
]


# ============================================================
# HELPERS
# ============================================================

def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    bias = np.mean(
        y_pred - y_true
    )

    return {
        "MAE_hours": float(mae),
        "RMSE_hours": float(rmse),
        "R2": float(r2),
        "Mean_Bias_hours": float(bias),
    }


def print_metrics(
    title: str,
    result: dict,
):
    print(f"\n{title}")

    print(
        f"  MAE       : "
        f"{result['MAE_hours']:.4f} h"
    )

    print(
        f"  RMSE      : "
        f"{result['RMSE_hours']:.4f} h"
    )

    print(
        f"  R²        : "
        f"{result['R2']:.4f}"
    )

    print(
        f"  Mean Bias : "
        f"{result['Mean_Bias_hours']:.4f} h"
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["flight_start"] = pd.to_datetime(
        df["flight_start"],
        errors="raise",
    )

    df = df.sort_values(
        [
            "engine_id",
            "flight_start",
            "flight_id",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE SEQUENCE FEATURES
# ============================================================

def build_sequence_features(
    df: pd.DataFrame,
):

    df = df.copy()

    # --------------------------------------------------------
    # Previous flight target
    # --------------------------------------------------------

    df["previous_rul_hours"] = (
        df.groupby("engine_id")[TARGET]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous flight health state
    # --------------------------------------------------------

    for feature in HEALTH_FEATURES:

        previous_name = (
            f"previous_{feature}"
        )

        delta_name = (
            f"{feature}_change"
        )

        df[previous_name] = (
            df.groupby("engine_id")[feature]
            .shift(1)
        )

        df[delta_name] = (
            df[feature]
            - df[previous_name]
        )

    # --------------------------------------------------------
    # RUL transition
    #
    # Diagnostic only.
    # This is the target change from previous -> current.
    # --------------------------------------------------------

    df["rul_change_hours"] = (
        df[TARGET]
        - df["previous_rul_hours"]
    )

    # --------------------------------------------------------
    # Flight sequence number
    # --------------------------------------------------------

    df["engine_flight_number"] = (
        df.groupby("engine_id")
        .cumcount()
        + 1
    )

    return df


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlations(
    df: pd.DataFrame,
    columns,
):

    rows = []

    for col in columns:

        if col not in df.columns:
            continue

        subset = df[
            [col, TARGET]
        ].dropna()

        if len(subset) < 3:
            continue

        if (
            subset[col].nunique() <= 1
            or subset[TARGET].nunique() <= 1
        ):
            continue

        corr = subset[col].corr(
            subset[TARGET]
        )

        if pd.isna(corr):
            continue

        rows.append(
            {
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(
                    corr
                ),
                "samples": len(subset),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "RUL V3 SEQUENCE / DEGRADATION DIAGNOSIS"
    )

    df = load_data()

    print(
        f"Flights : {len(df)}"
    )

    print(
        f"Engines : {df['engine_id'].nunique()}"
    )

    # ========================================================
    # BUILD SEQUENCE FEATURES
    # ========================================================

    section(
        "BUILD SEQUENCE FEATURES"
    )

    df = build_sequence_features(
        df
    )

    first_flight_mask = (
        df["previous_rul_hours"]
        .isna()
    )

    sequence_df = df[
        ~first_flight_mask
    ].copy()

    print(
        f"Total flights              : {len(df)}"
    )

    print(
        f"First flights / no previous : "
        f"{first_flight_mask.sum()}"
    )

    print(
        f"Sequential transitions      : "
        f"{len(sequence_df)}"
    )

    # ========================================================
    # PREVIOUS RUL RELATIONSHIP
    # ========================================================

    section(
        "PREVIOUS RUL → CURRENT RUL"
    )

    previous_rul_corr = (
        sequence_df[
            "previous_rul_hours"
        ]
        .corr(
            sequence_df[TARGET]
        )
    )

    print(
        f"Correlation(previous RUL, current RUL): "
        f"{previous_rul_corr:.4f}"
    )

    # ========================================================
    # RUL CHANGE
    # ========================================================

    section(
        "RUL CHANGE ANALYSIS"
    )

    rul_change = (
        sequence_df[
            "rul_change_hours"
        ]
    )

    print(
        f"Mean change   : {rul_change.mean():.4f} h"
    )

    print(
        f"Median change : {rul_change.median():.4f} h"
    )

    print(
        f"Min change    : {rul_change.min():.4f} h"
    )

    print(
        f"Max change    : {rul_change.max():.4f} h"
    )

    print(
        f"Increases     : "
        f"{(rul_change > 0).sum()}"
    )

    print(
        f"Decreases     : "
        f"{(rul_change < 0).sum()}"
    )

    # ========================================================
    # PREVIOUS RUL BASELINE
    # ========================================================
    #
    # Prediction:
    # current RUL = previous RUL
    #
    # This is a very important baseline.
    #
    # ========================================================

    section(
        "SEQUENCE BASELINE — PREVIOUS RUL"
    )

    y_true = (
        sequence_df[TARGET]
        .to_numpy()
    )

    previous_prediction = (
        sequence_df[
            "previous_rul_hours"
        ]
        .to_numpy()
    )

    previous_baseline_metrics = metrics(
        y_true,
        previous_prediction,
    )

    print_metrics(
        "Previous-RUL baseline",
        previous_baseline_metrics,
    )

    # ========================================================
    # CORRELATION OF HEALTH CHANGES WITH RUL CHANGE
    # ========================================================

    section(
        "HEALTH CHANGE → RUL CHANGE"
    )

    change_columns = [
        col
        for col in sequence_df.columns
        if col.endswith("_change")
    ]

    health_change_corr = []

    for col in change_columns:

        subset = sequence_df[
            [col, "rul_change_hours"]
        ].dropna()

        if len(subset) < 3:
            continue

        if subset[col].nunique() <= 1:
            continue

        corr = subset[col].corr(
            subset["rul_change_hours"]
        )

        if pd.isna(corr):
            continue

        health_change_corr.append(
            {
                "feature": col,
                "correlation":
                    corr,
                "abs_correlation":
                    abs(corr),
                "samples":
                    len(subset),
            }
        )

    health_change_corr_df = (
        pd.DataFrame(
            health_change_corr
        )
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
    )

    print(
        health_change_corr_df.head(
            30
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # PREVIOUS RUL + HEALTH CHANGE CORRELATIONS
    # ========================================================

    section(
        "SEQUENCE FEATURE CORRELATION WITH CURRENT RUL"
    )

    sequence_candidate_features = [
        "previous_rul_hours",
        "engine_flight_number",
    ]

    sequence_candidate_features.extend(
        change_columns
    )

    sequence_candidate_features.extend(
        HEALTH_FEATURES
    )

    sequence_corr_df = (
        calculate_correlations(
            sequence_df,
            sequence_candidate_features,
        )
    )

    print(
        sequence_corr_df.head(
            40
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # RUL CHANGE PREDICTABILITY
    # ========================================================
    #
    # Can health changes predict:
    #
    # current_rul - previous_rul
    #
    # This is the most important diagnostic.
    #
    # ========================================================

    section(
        "RUL CHANGE MODEL DIAGNOSTIC"
    )

    change_features = [
        col
        for col in change_columns
        if col in sequence_df.columns
    ]

    # Include previous RUL separately
    model_features = [
        "previous_rul_hours"
    ] + change_features

    model_df = sequence_df[
        [
            TARGET,
            "rul_change_hours",
            *model_features,
            "engine_id",
            "flight_id",
            "flight_start",
        ]
    ].copy()

    model_df = model_df.dropna(
        subset=[
            "previous_rul_hours"
        ]
    )

    # --------------------------------------------------------
    # Prepare X/y
    # --------------------------------------------------------

    X_change = model_df[
        model_features
    ]

    y_change = model_df[
        "rul_change_hours"
    ].to_numpy()

    imputer = SimpleImputer(
        strategy="median"
    )

    X_change_imp = (
        imputer.fit_transform(
            X_change
        )
    )

    # --------------------------------------------------------
    # Simple temporal split for diagnostic
    #
    # Last 20% of transitions are held out.
    #
    # This is diagnostic only. Final V3 split will be
    # explicitly engineered later.
    # --------------------------------------------------------

    n = len(model_df)

    split_index = max(
        1,
        int(n * 0.80)
    )

    X_train = X_change_imp[
        :split_index
    ]

    X_test = X_change_imp[
        split_index:
    ]

    y_train = y_change[
        :split_index
    ]

    y_test_change = y_change[
        split_index:
    ]

    if len(y_test_change) < 2:
        raise ValueError(
            "Not enough samples for diagnostic holdout."
        )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predicted_change = (
        model.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Predict current RUL:
    #
    # previous RUL + predicted RUL change
    # --------------------------------------------------------

    previous_rul_test = (
        model_df[
            "previous_rul_hours"
        ]
        .to_numpy()
        [split_index:]
    )

    actual_current_rul = (
        model_df[
            TARGET
        ]
        .to_numpy()
        [split_index:]
    )

    predicted_current_rul = (
        previous_rul_test
        + predicted_change
    )

    predicted_current_rul = np.maximum(
        predicted_current_rul,
        0.0,
    )

    current_rul_metrics = metrics(
        actual_current_rul,
        predicted_current_rul,
    )

    change_metrics = metrics(
        y_test_change,
        predicted_change,
    )

    print_metrics(
        "RUL-change model",
        change_metrics,
    )

    print_metrics(
        "Reconstructed current-RUL prediction",
        current_rul_metrics,
    )

    # ========================================================
    # PREVIOUS-RUL BASELINE ON SAME HOLDOUT
    # ========================================================

    section(
        "SAME-HOLDOUT BASELINE"
    )

    previous_test_baseline = (
        previous_rul_test
    )

    previous_test_metrics = metrics(
        actual_current_rul,
        previous_test_baseline,
    )

    print_metrics(
        "Previous-RUL baseline",
        previous_test_metrics,
    )

    improvement = (
        previous_test_metrics[
            "MAE_hours"
        ]
        - current_rul_metrics[
            "MAE_hours"
        ]
    )

    print(
        f"\nMAE improvement over previous-RUL baseline: "
        f"{improvement:.4f} h"
    )

    # ========================================================
    # ABSOLUTE RUL MODEL USING SEQUENCE FEATURES
    # ========================================================
    #
    # This is another diagnostic:
    #
    # previous RUL + current health + current changes
    # -> current RUL
    #
    # ========================================================

    section(
        "ABSOLUTE RUL — SEQUENCE FEATURES"
    )

    absolute_features = [
        "previous_rul_hours",
    ]

    absolute_features.extend(
        HEALTH_FEATURES
    )

    absolute_features.extend(
        change_columns
    )

    absolute_df = sequence_df[
        [
            TARGET,
            *absolute_features,
        ]
    ].dropna(
        subset=[
            "previous_rul_hours"
        ]
    )

    X_absolute = absolute_df[
        absolute_features
    ]

    y_absolute = absolute_df[
        TARGET
    ].to_numpy()

    absolute_imputer = SimpleImputer(
        strategy="median"
    )

    X_absolute_imp = (
        absolute_imputer.fit_transform(
            X_absolute
        )
    )

    n_abs = len(
        absolute_df
    )

    split_abs = max(
        1,
        int(n_abs * 0.80)
    )

    X_abs_train = (
        X_absolute_imp[
            :split_abs
        ]
    )

    X_abs_test = (
        X_absolute_imp[
            split_abs:
        ]
    )

    y_abs_train = (
        y_absolute[
            :split_abs
        ]
    )

    y_abs_test = (
        y_absolute[
            split_abs:
        ]
    )

    abs_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    abs_model.fit(
        X_abs_train,
        y_abs_train,
    )

    abs_pred = (
        abs_model.predict(
            X_abs_test
        )
    )

    abs_pred = np.maximum(
        abs_pred,
        0.0,
    )

    absolute_metrics = metrics(
        y_abs_test,
        abs_pred,
    )

    print_metrics(
        "Sequence-feature absolute-RUL model",
        absolute_metrics,
    )

    # ========================================================
    # ENGINE-WISE SEQUENCE BEHAVIOR
    # ========================================================

    section(
        "ENGINE-WISE SEQUENCE SIGNAL"
    )

    engine_rows = []

    for engine_id, group in (
        sequence_df.groupby(
            "engine_id",
            sort=True,
        )
    ):

        prev_rul = (
            group[
                "previous_rul_hours"
            ]
        )

        current_rul = (
            group[TARGET]
        )

        corr = (
            prev_rul.corr(
                current_rul
            )
        )

        mean_abs_change = (
            group[
                "rul_change_hours"
            ]
            .abs()
            .mean()
        )

        engine_rows.append(
            {
                "engine_id": engine_id,
                "transitions": len(group),
                "previous_current_rul_corr":
                    corr,
                "mean_abs_rul_change_hours":
                    mean_abs_change,
                "mean_previous_rul":
                    prev_rul.mean(),
                "mean_current_rul":
                    current_rul.mean(),
            }
        )

        print(
            f"{engine_id}: "
            f"transitions={len(group):2d}  "
            f"corr={corr:.4f}  "
            f"mean_abs_change="
            f"{mean_abs_change:.2f} h"
        )

    engine_df = pd.DataFrame(
        engine_rows
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    section(
        "SEQUENCE MODEL FEATURE IMPORTANCE"
    )

    importance_df = pd.DataFrame(
        {
            "feature": model_features,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance_df.head(
            30
        ).to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    health_change_path = (
        OUTPUT_DIR
        / "health_change_vs_rul_change.csv"
    )

    sequence_corr_path = (
        OUTPUT_DIR
        / "sequence_feature_correlation.csv"
    )

    engine_path = (
        OUTPUT_DIR
        / "engine_sequence_signal.csv"
    )

    importance_path = (
        OUTPUT_DIR
        / "sequence_model_feature_importance.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "sequence_diagnostic_predictions.csv"
    )

    metrics_path = (
        OUTPUT_DIR
        / "sequence_diagnostic_metrics.csv"
    )

    health_change_corr_df.to_csv(
        health_change_path,
        index=False,
    )

    sequence_corr_df.to_csv(
        sequence_corr_path,
        index=False,
    )

    engine_df.to_csv(
        engine_path,
        index=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    prediction_output = pd.DataFrame(
        {
            "actual_rul_hours":
                actual_current_rul,
            "previous_rul_hours":
                previous_rul_test,
            "actual_rul_change_hours":
                y_test_change,
            "predicted_rul_change_hours":
                predicted_change,
            "predicted_rul_hours":
                predicted_current_rul,
        }
    )

    prediction_output.to_csv(
        predictions_path,
        index=False,
    )

    metrics_output = pd.DataFrame(
        [
            {
                "experiment":
                    "previous_rul_baseline",
                **previous_baseline_metrics,
            },
            {
                "experiment":
                    "previous_rul_baseline_same_holdout",
                **previous_test_metrics,
            },
            {
                "experiment":
                    "rul_change_model",
                **change_metrics,
            },
            {
                "experiment":
                    "reconstructed_current_rul",
                **current_rul_metrics,
            },
            {
                "experiment":
                    "sequence_feature_absolute_rul",
                **absolute_metrics,
            },
        ]
    )

    metrics_output.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {health_change_path}"
    )

    print(
        f"  {sequence_corr_path}"
    )

    print(
        f"  {engine_path}"
    )

    print(
        f"  {importance_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        "\nRUL V3 SEQUENCE DIAGNOSIS COMPLETE."
    )


if __name__ == "__main__":
    main()