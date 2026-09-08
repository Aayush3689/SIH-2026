from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

RAW_PATH = Path(
    "training/datasets/raw/dt_engine_telemetry_full.csv"
)

OUTPUT_DIR = Path(
    "training/datasets/processed/rul_v4"
)

MODEL_DIR = Path(
    "models/rul_v4"
)


TARGET = "rul_hours_remaining"


# ============================================================
# CONFIG
# ============================================================

WINDOW_SAMPLES = 8
WINDOWS_PER_FLIGHT = 5

RANDOM_STATE = 42


# ============================================================
# SENSOR FEATURES
# ============================================================

SENSOR_FEATURES = [
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
    "ambient_temp_c",
    "planned_altitude_ft",
    "airspeed_kts",
]


# ============================================================
# MISSION / LIFECYCLE CONTEXT
# ============================================================

MISSION_NUMERIC = [
    "planned_mission_duration_min",
    "takeoff_duration_min",
    "cruise_duration_min",
    "loiter_duration_min",
    "return_to_base_duration_min",
    "total_cycles_at_start",
    "last_overhaul_days_ago",
    "last_major_maintenance_days_ago",
    "flight_index",
]


MISSION_CATEGORICAL = [
    "mission_type",
    "environmental_profile",
    "throttle_profile",
    "mission_priority",
    "engine_model_variant",
]


# ============================================================
# HELPERS
# ============================================================

def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def calculate_metrics(
    y_true,
    y_pred,
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


def low_rul_metrics(
    y_true,
    y_pred,
    threshold=10.0,
):
    mask = y_true <= threshold

    if not mask.any():
        return {
            "count": 0,
            "MAE_hours": None,
            "RMSE_hours": None,
            "Mean_Bias_hours": None,
        }

    return {
        "count": int(mask.sum()),
        "MAE_hours": float(
            mean_absolute_error(
                y_true[mask],
                y_pred[mask],
            )
        ),
        "RMSE_hours": float(
            np.sqrt(
                mean_squared_error(
                    y_true[mask],
                    y_pred[mask],
                )
            )
        ),
        "Mean_Bias_hours": float(
            np.mean(
                y_pred[mask]
                - y_true[mask]
            )
        ),
    }


def print_metrics(
    name,
    overall,
    low,
):
    print(f"\n{name}")

    print(
        f"  MAE        : "
        f"{overall['MAE_hours']:.4f} h"
    )

    print(
        f"  RMSE       : "
        f"{overall['RMSE_hours']:.4f} h"
    )

    print(
        f"  R²         : "
        f"{overall['R2']:.4f}"
    )

    print(
        f"  Mean Bias  : "
        f"{overall['Mean_Bias_hours']:.4f} h"
    )

    if low["count"] > 0:
        print(
            f"  Low-RUL MAE: "
            f"{low['MAE_hours']:.4f} h "
            f"(n={low['count']})"
        )


# ============================================================
# ENGINE-WISE CHRONOLOGICAL SPLIT
# ============================================================

def build_flight_split(
    df: pd.DataFrame,
):
    """
    Create an engine-wise chronological split at FLIGHT level.

    Important:
    The raw telemetry dataframe contains many rows per flight.
    Therefore we MUST first create one unique row per flight
    before assigning flights to train / validation / test.

    For every engine:
        early flights  -> TRAIN
        middle flights -> VALIDATION
        late flights   -> TEST
    """

    # --------------------------------------------------------
    # Create unique flight-level table first
    # --------------------------------------------------------

    flight_table = (
        df[
            [
                "engine_id",
                "flight_id",
                "flight_start_time",
            ]
        ]
        .drop_duplicates(
            subset=["flight_id"]
        )
        .sort_values(
            [
                "engine_id",
                "flight_start_time",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    total_flights = (
        flight_table["flight_id"].nunique()
    )

    raw_flights = (
        df["flight_id"].nunique()
    )

    if total_flights != raw_flights:
        raise ValueError(
            "Flight deduplication failed: "
            f"raw={raw_flights}, "
            f"unique={total_flights}"
        )

    train_ids = []
    val_ids = []
    test_ids = []

    summary_rows = []

    # --------------------------------------------------------
    # Split each engine independently
    # --------------------------------------------------------

    for engine_id, group in flight_table.groupby(
        "engine_id",
        sort=True,
    ):

        group = group.sort_values(
            [
                "flight_start_time",
                "flight_id",
            ]
        ).reset_index(drop=True)

        n = len(group)

        if n < 3:
            raise ValueError(
                f"{engine_id} has only {n} flights. "
                "At least 3 are required."
            )

        # Approximate 60 / 20 / 20
        n_train = max(
            1,
            int(n * 0.60),
        )

        n_val = max(
            1,
            int(n * 0.20),
        )

        n_test = (
            n
            - n_train
            - n_val
        )

        # Guarantee at least one test flight.
        while n_test < 1:

            if (
                n_train > n_val
                and n_train > 1
            ):
                n_train -= 1

            elif n_val > 1:
                n_val -= 1

            else:
                raise ValueError(
                    f"Could not create valid split "
                    f"for {engine_id}"
                )

            n_test = (
                n
                - n_train
                - n_val
            )

        train_group = group.iloc[
            :n_train
        ]

        val_group = group.iloc[
            n_train:n_train + n_val
        ]

        test_group = group.iloc[
            n_train + n_val:
        ]

        # ----------------------------------------------------
        # Store UNIQUE flight IDs
        # ----------------------------------------------------

        train_ids.extend(
            train_group[
                "flight_id"
            ].tolist()
        )

        val_ids.extend(
            val_group[
                "flight_id"
            ].tolist()
        )

        test_ids.extend(
            test_group[
                "flight_id"
            ].tolist()
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_rows.append(
            {
                "engine_id": engine_id,

                "total_flights": n,

                "train_flights":
                    len(train_group),

                "validation_flights":
                    len(val_group),

                "test_flights":
                    len(test_group),

                "train_first":
                    train_group[
                        "flight_start_time"
                    ].min(),

                "train_last":
                    train_group[
                        "flight_start_time"
                    ].max(),

                "validation_first":
                    val_group[
                        "flight_start_time"
                    ].min(),

                "validation_last":
                    val_group[
                        "flight_start_time"
                    ].max(),

                "test_first":
                    test_group[
                        "flight_start_time"
                    ].min(),

                "test_last":
                    test_group[
                        "flight_start_time"
                    ].max(),
            }
        )

    # --------------------------------------------------------
    # Global sanity checks
    # --------------------------------------------------------

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:
        raise ValueError(
            "Train / validation flight overlap detected."
        )

    if train_set & test_set:
        raise ValueError(
            "Train / test flight overlap detected."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation / test flight overlap detected."
        )

    all_split_ids = (
        train_set
        | val_set
        | test_set
    )

    if len(all_split_ids) != total_flights:
        raise ValueError(
            "Not all flights were assigned exactly once."
        )

    if (
        len(train_set)
        + len(val_set)
        + len(test_set)
        != total_flights
    ):
        raise ValueError(
            "Split flight counts do not sum to "
            "total flight count."
        )

    return (
        train_ids,
        val_ids,
        test_ids,
        pd.DataFrame(summary_rows),
    )


# ============================================================
# WINDOW FEATURE CREATION
# ============================================================

def create_window_row(
    flight: pd.DataFrame,
    end_index: int,
):
    start_index = max(
        0,
        end_index
        - WINDOW_SAMPLES
        + 1,
    )

    window = flight.iloc[
        start_index:end_index + 1
    ]

    row = {
        "engine_id":
            flight["engine_id"].iloc[0],

        "flight_id":
            flight["flight_id"].iloc[0],

        "timestamp":
            flight["timestamp"].iloc[
                end_index
            ],

        TARGET:
            float(
                flight[TARGET].iloc[0]
            ),
    }

    # --------------------------------------------------------
    # Current-window statistical features
    # --------------------------------------------------------

    for col in SENSOR_FEATURES:

        values = window[
            col
        ].astype(float)

        row[
            f"{col}_mean"
        ] = float(
            values.mean()
        )

        row[
            f"{col}_std"
        ] = float(
            values.std(
                ddof=0
            )
        )

        row[
            f"{col}_last"
        ] = float(
            values.iloc[-1]
        )

    # --------------------------------------------------------
    # Window trend / slope
    # --------------------------------------------------------

    slope_features = [
        "cht_c",
        "egt_c",
        "oil_pressure_kpa",
        "oil_temperature_c",
        "engine_vibration_mm_s",
        "rpm",
    ]

    x = np.arange(
        len(window),
        dtype=float,
    )

    for col in slope_features:

        y = window[
            col
        ].astype(float).to_numpy()

        if len(y) >= 2:
            slope = np.polyfit(
                x,
                y,
                1
            )[0]
        else:
            slope = 0.0

        row[
            f"{col}_slope"
        ] = float(slope)

    # --------------------------------------------------------
    # Mission context
    # --------------------------------------------------------

    for col in MISSION_NUMERIC:

        row[col] = (
            flight[col].iloc[0]
        )

    for col in MISSION_CATEGORICAL:

        row[col] = (
            flight[col].iloc[0]
        )

    return row


def create_windows(
    df: pd.DataFrame,
    allowed_flights,
):
    """
    Create 5 windows per flight.

    IMPORTANT:
    Multiple windows from a flight share the same RUL target.
    Final evaluation will therefore also be reported at
    flight-level after averaging the window predictions.
    """

    allowed = set(
        allowed_flights
    )

    subset = df[
        df["flight_id"].isin(
            allowed
        )
    ].copy()

    rows = []

    for flight_id, flight in subset.groupby(
        "flight_id",
        sort=False,
    ):

        flight = flight.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

        n = len(flight)

        if n < WINDOW_SAMPLES:
            continue

        endpoints = np.linspace(
            WINDOW_SAMPLES - 1,
            n - 1,
            WINDOWS_PER_FLIGHT,
        )

        endpoints = sorted(
            set(
                int(round(x))
                for x in endpoints
            )
        )

        # Safety: make sure we have exactly
        # WINDOWS_PER_FLIGHT windows.
        if len(endpoints) < WINDOWS_PER_FLIGHT:

            for candidate in range(
                WINDOW_SAMPLES - 1,
                n,
            ):

                if candidate not in endpoints:
                    endpoints.append(
                        candidate
                    )

                if (
                    len(endpoints)
                    == WINDOWS_PER_FLIGHT
                ):
                    break

        for endpoint in endpoints:

            rows.append(
                create_window_row(
                    flight,
                    endpoint,
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "RUL V4 WINDOW-LEVEL PROTOTYPE"
    )

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_PATH}"
        )

    df = pd.read_csv(
        RAW_PATH
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    section(
        "RAW DATASET"
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"Engines : {df['engine_id'].nunique()}"
    )

    print(
        f"Flights : {df['flight_id'].nunique()}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target: {TARGET}"
        )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df["flight_start_time"] = pd.to_datetime(
        df["flight_start_time"],
        errors="raise",
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = {
        "engine_id",
        "flight_id",
        "timestamp",
        "flight_start_time",
        TARGET,
        *SENSOR_FEATURES,
        *MISSION_NUMERIC,
        *MISSION_CATEGORICAL,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    # ========================================================
    # FLIGHT TARGET VALIDATION
    # ========================================================

    section(
        "FLIGHT TARGET VALIDATION"
    )

    target_counts = (
        df.groupby(
            "flight_id"
        )[TARGET]
        .nunique()
    )

    print(
        f"Total flights: "
        f"{len(target_counts)}"
    )

    print(
        f"Constant-target flights: "
        f"{(target_counts == 1).sum()}"
    )

    print(
        f"Non-constant-target flights: "
        f"{(target_counts > 1).sum()}"
    )

    if (
        target_counts > 1
    ).any():
        raise ValueError(
            "RUL is not constant within every flight."
        )

    # ========================================================
    # FLIGHT SPLIT
    # ========================================================

    section(
        "ENGINE-WISE CHRONOLOGICAL SPLIT"
    )

    (
        train_flights,
        val_flights,
        test_flights,
        split_summary,
    ) = build_flight_split(
        df
    )

    print(
        f"TRAIN flights      : "
        f"{len(train_flights)}"
    )

    print(
        f"VALIDATION flights : "
        f"{len(val_flights)}"
    )

    print(
        f"TEST flights       : "
        f"{len(test_flights)}"
    )

    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    section(
        "WINDOW DATASET"
    )

    train = create_windows(
        df,
        train_flights,
    )

    val = create_windows(
        df,
        val_flights,
    )

    test = create_windows(
        df,
        test_flights,
    )

    print(
        f"TRAIN windows      : {len(train)}"
    )

    print(
        f"VALIDATION windows : {len(val)}"
    )

    print(
        f"TEST windows       : {len(test)}"
    )

    print(
        f"Windows per train flight ≈ "
        f"{len(train) / len(train_flights):.2f}"
    )

    # ========================================================
    # WINDOW VALIDATION
    # ========================================================

    section(
        "WINDOW VALIDATION"
    )

    assert (
        set(train["flight_id"])
        .isdisjoint(
            set(val["flight_id"])
        )
    )

    assert (
        set(train["flight_id"])
        .isdisjoint(
            set(test["flight_id"])
        )
    )

    assert (
        set(val["flight_id"])
        .isdisjoint(
            set(test["flight_id"])
        )
    )

    print(
        "Flight leakage : PASSED"
    )

    # ========================================================
    # SAVE WINDOW DATASETS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        OUTPUT_DIR
        / "rul_v4_train_windows.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT_DIR
        / "rul_v4_validation_windows.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR
        / "rul_v4_test_windows.csv",
        index=False,
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "rul_v4_split_summary.csv",
        index=False,
    )

    # ========================================================
    # MODEL MATRICES
    # ========================================================

    drop_columns = [
        "engine_id",
        "flight_id",
        "timestamp",
        TARGET,
    ]

    X_train = train.drop(
        columns=drop_columns
    )

    y_train = train[
        TARGET
    ].astype(float).to_numpy()

    X_val = val.drop(
        columns=drop_columns
    )

    y_val = val[
        TARGET
    ].astype(float).to_numpy()

    X_test = test.drop(
        columns=drop_columns
    )

    y_test = test[
        TARGET
    ].astype(float).to_numpy()

    numeric_columns = [
        col
        for col in X_train.columns
        if col not in MISSION_CATEGORICAL
    ]

    categorical_columns = [
        col
        for col in X_train.columns
        if col in MISSION_CATEGORICAL
    ]

    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy=(
                                    "most_frequent"
                                )
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown=(
                                    "ignore"
                                ),
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "hist_gradient_boosting":
            HistGradientBoostingRegressor(
                max_iter=350,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=8,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
    }

    validation_results = []
    trained_models = {}

    # ========================================================
    # TRAIN + VALIDATION
    # ========================================================

    section(
        "TRAINING + VALIDATION"
    )

    for model_name, model in models.items():

        print(
            f"\nTraining: "
            f"{model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        val_pred = pipeline.predict(
            X_val
        )

        val_pred = np.maximum(
            val_pred,
            0.0,
        )

        overall = calculate_metrics(
            y_val,
            val_pred,
        )

        low = low_rul_metrics(
            y_val,
            val_pred,
        )

        selection_score = (
            0.70
            * overall["MAE_hours"]
            +
            0.30
            * (
                low["MAE_hours"]
                if low["MAE_hours"]
                is not None
                else overall["MAE_hours"]
            )
        )

        validation_results.append(
            {
                "model": model_name,
                **overall,
                "low_rul_count":
                    low["count"],
                "low_rul_MAE_hours":
                    low["MAE_hours"],
                "low_rul_RMSE_hours":
                    low["RMSE_hours"],
                "low_rul_bias_hours":
                    low["Mean_Bias_hours"],
                "selection_score":
                    float(selection_score),
            }
        )

        trained_models[
            model_name
        ] = pipeline

        print_metrics(
            "Validation",
            overall,
            low,
        )

        print(
            f"  Selection score : "
            f"{selection_score:.4f}"
        )

    validation_results = (
        pd.DataFrame(
            validation_results
        )
        .sort_values(
            "selection_score"
        )
        .reset_index(
            drop=True
        )
    )

    section(
        "VALIDATION MODEL RANKING"
    )

    print(
        validation_results.to_string(
            index=False
        )
    )

    best_model_name = (
        validation_results.iloc[0][
            "model"
        ]
    )

    best_model = trained_models[
        best_model_name
    ]

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    # ========================================================
    # WINDOW-LEVEL TEST
    # ========================================================

    section(
        "WINDOW-LEVEL TEST"
    )

    test_window_pred = (
        best_model.predict(
            X_test
        )
    )

    test_window_pred = np.maximum(
        test_window_pred,
        0.0,
    )

    window_metrics = calculate_metrics(
        y_test,
        test_window_pred,
    )

    window_low = low_rul_metrics(
        y_test,
        test_window_pred,
    )

    print_metrics(
        "Window-level TEST",
        window_metrics,
        window_low,
    )

    # ========================================================
    # FLIGHT-LEVEL TEST
    # ========================================================
    #
    # Multiple windows belong to the same flight.
    # Average their predictions to obtain ONE RUL prediction
    # per flight.
    #
    # This is the more honest evaluation.
    #
    # ========================================================

    section(
        "FLIGHT-LEVEL TEST"
    )

    test_output = test[
        [
            "engine_id",
            "flight_id",
            "timestamp",
            TARGET,
        ]
    ].copy()

    test_output[
        "predicted_rul_window"
    ] = test_window_pred

    flight_predictions = (
        test_output
        .groupby(
            [
                "engine_id",
                "flight_id",
            ]
        )
        .agg(
            actual_rul=(
                TARGET,
                "first",
            ),
            predicted_rul=(
                "predicted_rul_window",
                "mean",
            ),
            min_predicted_rul=(
                "predicted_rul_window",
                "min",
            ),
            max_predicted_rul=(
                "predicted_rul_window",
                "max",
            ),
        )
        .reset_index()
    )

    flight_y = (
        flight_predictions[
            "actual_rul"
        ].to_numpy()
    )

    flight_pred = (
        flight_predictions[
            "predicted_rul"
        ].to_numpy()
    )

    flight_metrics = calculate_metrics(
        flight_y,
        flight_pred,
    )

    flight_low = low_rul_metrics(
        flight_y,
        flight_pred,
    )

    print_metrics(
        "Flight-level TEST",
        flight_metrics,
        flight_low,
    )

    print(
        f"\nTest flights evaluated: "
        f"{len(flight_predictions)}"
    )

    print(
        f"Actual RUL mean    : "
        f"{flight_y.mean():.2f} h"
    )

    print(
        f"Predicted RUL mean : "
        f"{flight_pred.mean():.2f} h"
    )

    print(
        f"Prediction min     : "
        f"{flight_pred.min():.2f} h"
    )

    print(
        f"Prediction max     : "
        f"{flight_pred.max():.2f} h"
    )

    # ========================================================
    # RUL RANGE PERFORMANCE
    # ========================================================

    section(
        "FLIGHT-LEVEL TEST BY RUL RANGE"
    )

    ranges = [
        ("0-5h", 0.0, 5.0),
        ("5-10h", 5.0, 10.0),
        ("10-20h", 10.0, 20.0),
        ("20-40h", 20.0, 40.0),
        ("40h+", 40.0, np.inf),
    ]

    range_rows = []

    for label, low, high in ranges:

        if np.isinf(high):

            mask = (
                flight_y >= low
            )

        else:

            mask = (
                (flight_y >= low)
                &
                (flight_y < high)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            range_rows.append(
                {
                    "rul_range": label,
                    "samples": 0,
                    "MAE_hours": None,
                    "RMSE_hours": None,
                    "bias_hours": None,
                }
            )

            continue

        range_rows.append(
            {
                "rul_range": label,
                "samples": count,
                "MAE_hours": float(
                    mean_absolute_error(
                        flight_y[mask],
                        flight_pred[mask],
                    )
                ),
                "RMSE_hours": float(
                    np.sqrt(
                        mean_squared_error(
                            flight_y[mask],
                            flight_pred[mask],
                        )
                    )
                ),
                "bias_hours": float(
                    np.mean(
                        flight_pred[mask]
                        - flight_y[mask]
                    )
                ),
            }
        )

    range_df = pd.DataFrame(
        range_rows
    )

    print(
        range_df.to_string(
            index=False
        )
    )

    # ========================================================
    # ENGINE-WISE TEST PERFORMANCE
    # ========================================================

    section(
        "ENGINE-WISE FLIGHT-LEVEL TEST"
    )

    engine_rows = []

    for engine_id, group in (
        flight_predictions.groupby(
            "engine_id"
        )
    ):

        actual = group[
            "actual_rul"
        ].to_numpy()

        pred = group[
            "predicted_rul"
        ].to_numpy()

        overall = calculate_metrics(
            actual,
            pred,
        )

        low = low_rul_metrics(
            actual,
            pred,
        )

        engine_rows.append(
            {
                "engine_id":
                    engine_id,
                "samples":
                    len(group),
                **overall,
                "low_rul_count":
                    low["count"],
                "low_rul_MAE_hours":
                    low["MAE_hours"],
            }
        )

        print(
            f"{engine_id:<14}"
            f"n={len(group):>2}  "
            f"MAE={overall['MAE_hours']:.2f} h  "
            f"RMSE={overall['RMSE_hours']:.2f} h  "
            f"R²={overall['R2']:.3f}"
        )

    engine_df = pd.DataFrame(
        engine_rows
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_output.to_csv(
        OUTPUT_DIR
        / "rul_v4_window_test_predictions.csv",
        index=False,
    )

    flight_predictions.to_csv(
        OUTPUT_DIR
        / "rul_v4_flight_test_predictions.csv",
        index=False,
    )

    validation_results.to_csv(
        OUTPUT_DIR
        / "rul_v4_validation_results.csv",
        index=False,
    )

    range_df.to_csv(
        OUTPUT_DIR
        / "rul_v4_test_ranges.csv",
        index=False,
    )

    engine_df.to_csv(
        OUTPUT_DIR
        / "rul_v4_engine_metrics.csv",
        index=False,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_bundle = {
        "pipeline": best_model,
        "target": TARGET,
        "window_samples": WINDOW_SAMPLES,
        "windows_per_flight":
            WINDOWS_PER_FLIGHT,
        "sensor_features":
            SENSOR_FEATURES,
        "mission_numeric_features":
            MISSION_NUMERIC,
        "mission_categorical_features":
            MISSION_CATEGORICAL,
        "split_strategy":
            "engine-wise chronological",
        "evaluation_unit":
            "flight-level averaged prediction",
        "warning":
            (
                "Multiple windows share the same flight-level "
                "RUL target. Independent target count equals "
                "number of flights, not number of windows."
            ),
    }

    model_path = (
        MODEL_DIR
        / "final_rul_v4.joblib"
    )

    joblib.dump(
        model_bundle,
        model_path,
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metrics_json = {
        "selected_model":
            best_model_name,

        "train_flights":
            len(train_flights),

        "validation_flights":
            len(val_flights),

        "test_flights":
            len(test_flights),

        "train_windows":
            len(train),

        "validation_windows":
            len(val),

        "test_windows":
            len(test),

        "window_samples":
            WINDOW_SAMPLES,

        "windows_per_flight":
            WINDOWS_PER_FLIGHT,

        "validation_results":
            validation_results.to_dict(
                orient="records"
            ),

        "window_test":
            {
                **window_metrics,
                "low_rul":
                    window_low,
            },

        "flight_test":
            {
                **flight_metrics,
                "low_rul":
                    flight_low,
            },

        "prediction_range": {
            "actual_min":
                float(flight_y.min()),
            "actual_max":
                float(flight_y.max()),
            "predicted_min":
                float(flight_pred.min()),
            "predicted_max":
                float(flight_pred.max()),
            "actual_mean":
                float(flight_y.mean()),
            "predicted_mean":
                float(flight_pred.mean()),
        },
    }

    with open(
        MODEL_DIR
        / "rul_v4_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_json,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "OUTPUT"
    )

    print(
        f"Model:"
    )

    print(
        f"  {model_path}"
    )

    print(
        "\nDatasets/results:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nRUL V4 COMPLETE."
    )


if __name__ == "__main__":
    main()