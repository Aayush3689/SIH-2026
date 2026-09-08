from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "training/datasets/processed/rul_v1/split"
)

MODEL_DIR = Path(
    "models/rul_v3"
)

RESULT_DIR = Path(
    "training/datasets/processed/rul_v3"
)

TRAIN_PATH = BASE_DIR / "rul_train_v1.csv"
VAL_PATH = BASE_DIR / "rul_validation_v1.csv"
TEST_PATH = BASE_DIR / "rul_test_v1.csv"

TARGET = "rul_hours_remaining"

LOW_RUL_THRESHOLD = 10.0


# ============================================================
# BASE HEALTH FEATURES
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
# IMPORTANT CHANGE FEATURES
# ============================================================
#
# These are computed from:
#
#     current flight - previous flight
#
# They must be created only after sorting chronologically
# inside each engine.
#
# ============================================================

IMPORTANT_CHANGE_FEATURES = [
    "cht_rate_mean",
    "egt_rate_mean",
    "cht_c_std",
    "rpm_std",
    "fuel_flow_lph_std",
    "egt_c_std",
    "cht_c_last",
    "oil_temperature_c_last",
    "oil_pressure_rate_mean",
    "oil_temperature_c_std",
    "cht_residual_mean",
    "oil_temperature_residual_mean",
    "egt_residual_mean",
    "engine_vibration_mm_s_std",
    "engine_vibration_mm_s_mean",
    "vibration_rate_mean",
    "rpm_last",
    "fuel_flow_lph_last",
]


# ============================================================
# HELPERS
# ============================================================

def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def regression_metrics(y_true, y_pred):
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

    median_ae = np.median(
        np.abs(y_true - y_pred)
    )

    bias = np.mean(
        y_pred - y_true
    )

    return {
        "MAE_hours": float(mae),
        "RMSE_hours": float(rmse),
        "R2": float(r2),
        "Median_AE_hours": float(median_ae),
        "Mean_Bias_hours": float(bias),
    }


def low_rul_metrics(y_true, y_pred):
    mask = (
        y_true
        <= LOW_RUL_THRESHOLD
    )

    count = int(mask.sum())

    if count == 0:
        return {
            "low_rul_count": 0,
            "low_rul_MAE_hours": None,
            "low_rul_RMSE_hours": None,
            "low_rul_bias_hours": None,
        }

    actual = y_true[mask]
    pred = y_pred[mask]

    return {
        "low_rul_count": count,
        "low_rul_MAE_hours": float(
            mean_absolute_error(
                actual,
                pred,
            )
        ),
        "low_rul_RMSE_hours": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    pred,
                )
            )
        ),
        "low_rul_bias_hours": float(
            np.mean(
                pred - actual
            )
        ),
    }


def print_metrics(title, overall, low):
    print(f"\n{title}")

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
        f"  Median AE  : "
        f"{overall['Median_AE_hours']:.4f} h"
    )

    print(
        f"  Mean Bias  : "
        f"{overall['Mean_Bias_hours']:.4f} h"
    )

    if low["low_rul_count"] > 0:

        print(
            f"  Low-RUL MAE: "
            f"{low['low_rul_MAE_hours']:.4f} h"
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    for path in [
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{path}"
            )

    train = pd.read_csv(
        TRAIN_PATH
    )

    val = pd.read_csv(
        VAL_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    for df in [
        train,
        val,
        test,
    ]:
        df["flight_start"] = pd.to_datetime(
            df["flight_start"],
            errors="raise",
        )

        df["flight_end"] = pd.to_datetime(
            df["flight_end"],
            errors="raise",
        )

    return train, val, test


# ============================================================
# BUILD SEQUENTIAL FEATURES
# ============================================================

def add_sequence_features(df):
    """
    Build previous-flight and current-vs-previous-flight
    health features.

    IMPORTANT:
    Rows are sorted by engine + flight_start first.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "engine_id",
            "flight_start",
            "flight_id",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Previous RUL
    # --------------------------------------------------------

    df["previous_rul_hours"] = (
        df.groupby("engine_id")[TARGET]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous health features
    # --------------------------------------------------------

    for feature in HEALTH_FEATURES:

        previous_col = (
            f"previous_{feature}"
        )

        change_col = (
            f"{feature}_change"
        )

        df[previous_col] = (
            df.groupby("engine_id")[feature]
            .shift(1)
        )

        df[change_col] = (
            df[feature]
            - df[previous_col]
        )

    # --------------------------------------------------------
    # Flight sequence
    # --------------------------------------------------------

    df["engine_flight_number"] = (
        df.groupby("engine_id")
        .cumcount()
        + 1
    )

    return df


# ============================================================
# SELECT FEATURES
# ============================================================

def get_model_features(
    include_previous_rul: bool,
):
    """
    Two experiment variants:

    Variant A:
        current health
        +
        selected health changes

    Variant B:
        Variant A
        +
        previous RUL
    """

    features = list(
        HEALTH_FEATURES
    )

    for base_feature in IMPORTANT_CHANGE_FEATURES:

        change_name = (
            f"{base_feature}_change"
        )

        if change_name not in features:
            features.append(
                change_name
            )

    if include_previous_rul:
        features.append(
            "previous_rul_hours"
        )

    return features


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    train,
    val,
    test,
    features,
):
    required = set(features)
    required.add(TARGET)

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:

        missing = sorted(
            required - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} missing columns:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in missing
                )
            )

    return True


# ============================================================
# PREPARE SPLIT
# ============================================================

def prepare_xy(
    df,
    features,
    imputer=None,
    fit_imputer=False,
):

    X = df[
        features
    ].copy()

    y = df[
        TARGET
    ].astype(float).to_numpy()

    if fit_imputer:

        imputer = SimpleImputer(
            strategy="median"
        )

        X_array = (
            imputer.fit_transform(
                X
            )
        )

    else:

        if imputer is None:
            raise ValueError(
                "Imputer required when "
                "fit_imputer=False."
            )

        X_array = (
            imputer.transform(
                X
            )
        )

    return X_array, y, imputer


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
):
    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "RUL V3 SEQUENCE-AWARE MODEL TRAINING"
    )

    # ========================================================
    # LOAD
    # ========================================================

    (
        train_df,
        val_df,
        test_df,
    ) = load_data()

    print(
        f"Train rows      : {len(train_df)}"
    )

    print(
        f"Validation rows : {len(val_df)}"
    )

    print(
        f"Test rows       : {len(test_df)}"
    )

    # ========================================================
    # CREATE SEQUENCE FEATURES
    # ========================================================

    section(
        "BUILD SEQUENCE FEATURES"
    )

    train_df = add_sequence_features(
        train_df
    )

    val_df = add_sequence_features(
        val_df
    )

    test_df = add_sequence_features(
        test_df
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Since the split was already created chronologically,
    # previous-flight information must come from within each
    # split only.
    #
    # However, the first validation/test flight of an engine
    # may need the immediately preceding train flight.
    #
    # Therefore we will separately construct a FULL historical
    # frame later for the production-like variant.
    #
    # For this initial V3 comparison, rows without previous
    # flight information are dropped.
    # --------------------------------------------------------

    train_df = train_df[
        train_df[
            "previous_rul_hours"
        ].notna()
    ].copy()

    val_df = val_df[
        val_df[
            "previous_rul_hours"
        ].notna()
    ].copy()

    test_df = test_df[
        test_df[
            "previous_rul_hours"
        ].notna()
    ].copy()

    print(
        f"Train sequential rows      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation sequential rows : "
        f"{len(val_df)}"
    )

    print(
        f"Test sequential rows       : "
        f"{len(test_df)}"
    )

    if len(train_df) < 10:
        raise ValueError(
            "Too few sequential training rows."
        )

    # ========================================================
    # EXPERIMENT VARIANTS
    # ========================================================

    experiments = {

        "health_change_only": {
            "include_previous_rul": False,
        },

        "health_change_plus_previous_rul": {
            "include_previous_rul": True,
        },
    }

    # ========================================================
    # MODEL CANDIDATES
    # ========================================================

    model_factories = {

        "random_forest": lambda: RandomForestRegressor(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": lambda:
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=4,
                l2_regularization=1.0,
                random_state=42,
            ),
    }

    all_validation_results = []
    trained_candidates = {}

    # ========================================================
    # TRAIN + VALIDATION
    # ========================================================

    section(
        "TRAINING + VALIDATION"
    )

    for experiment_name, config in experiments.items():

        include_previous_rul = config[
            "include_previous_rul"
        ]

        features = get_model_features(
            include_previous_rul
        )

        validate_features(
            train_df,
            val_df,
            test_df,
            features,
        )

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"EXPERIMENT: {experiment_name}"
        )

        print(
            f"Features: {len(features)}"
        )

        # ----------------------------------------------------
        # Build X/y
        # ----------------------------------------------------

        X_train, y_train, imputer = (
            prepare_xy(
                train_df,
                features,
                fit_imputer=True,
            )
        )

        X_val, y_val, _ = (
            prepare_xy(
                val_df,
                features,
                imputer=imputer,
                fit_imputer=False,
            )
        )

        for model_name, factory in (
            model_factories.items()
        ):

            print(
                f"\nTraining: "
                f"{model_name}"
            )

            model = factory()

            model = train_model(
                model,
                X_train,
                y_train,
            )

            val_pred = model.predict(
                X_val
            )

            val_pred = np.maximum(
                val_pred,
                0.0,
            )

            overall = regression_metrics(
                y_val,
                val_pred,
            )

            low = low_rul_metrics(
                y_val,
                val_pred,
            )

            # ------------------------------------------------
            # Selection score
            #
            # Overall MAE is primary.
            # Low-RUL MAE gets 30% weight.
            # ------------------------------------------------

            if low[
                "low_rul_MAE_hours"
            ] is not None:

                selection_score = (
                    0.70
                    * overall[
                        "MAE_hours"
                    ]
                    +
                    0.30
                    * low[
                        "low_rul_MAE_hours"
                    ]
                )

            else:

                selection_score = (
                    overall[
                        "MAE_hours"
                    ]
                )

            result = {
                "experiment":
                    experiment_name,

                "model":
                    model_name,

                "feature_count":
                    len(features),

                **overall,

                **low,

                "selection_score":
                    float(selection_score),
            }

            all_validation_results.append(
                result
            )

            trained_candidates[
                (
                    experiment_name,
                    model_name,
                )
            ] = {
                "model": model,
                "imputer": imputer,
                "features": features,
            }

            print_metrics(
                "Validation",
                overall,
                low,
            )

            print(
                f"  Selection score : "
                f"{selection_score:.4f}"
            )

    # ========================================================
    # RANKING
    # ========================================================

    validation_results = (
        pd.DataFrame(
            all_validation_results
        )
        .sort_values(
            "selection_score",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    section(
        "VALIDATION MODEL RANKING"
    )

    columns = [
        "experiment",
        "model",
        "feature_count",
        "MAE_hours",
        "RMSE_hours",
        "R2",
        "low_rul_MAE_hours",
        "low_rul_bias_hours",
        "selection_score",
    ]

    print(
        validation_results[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    best_row = (
        validation_results.iloc[0]
    )

    best_key = (
        best_row[
            "experiment"
        ],
        best_row[
            "model"
        ],
    )

    best_candidate = (
        trained_candidates[
            best_key
        ]
    )

    best_model = (
        best_candidate["model"]
    )

    best_imputer = (
        best_candidate["imputer"]
    )

    best_features = (
        best_candidate["features"]
    )

    best_experiment = (
        best_row[
            "experiment"
        ]
    )

    best_model_name = (
        best_row["model"]
    )

    section(
        "SELECTED MODEL"
    )

    print(
        f"Experiment : "
        f"{best_experiment}"
    )

    print(
        f"Model      : "
        f"{best_model_name}"
    )

    print(
        f"Features   : "
        f"{len(best_features)}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    section(
        "FINAL CHRONOLOGICAL TEST"
    )

    X_test, y_test, _ = (
        prepare_xy(
            test_df,
            best_features,
            imputer=best_imputer,
            fit_imputer=False,
        )
    )

    test_pred = best_model.predict(
        X_test
    )

    test_pred = np.maximum(
        test_pred,
        0.0,
    )

    test_overall = regression_metrics(
        y_test,
        test_pred,
    )

    test_low = low_rul_metrics(
        y_test,
        test_pred,
    )

    print_metrics(
        "TEST",
        test_overall,
        test_low,
    )

    # ========================================================
    # TEST RUL RANGES
    # ========================================================

    section(
        "TEST PERFORMANCE BY RUL RANGE"
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
                y_test >= low
            )

        else:

            mask = (
                (y_test >= low)
                & (y_test < high)
            )

        count = int(mask.sum())

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

        actual = y_test[mask]
        pred = test_pred[mask]

        range_rows.append(
            {
                "rul_range": label,
                "samples": count,
                "MAE_hours": float(
                    mean_absolute_error(
                        actual,
                        pred,
                    )
                ),
                "RMSE_hours": float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            pred,
                        )
                    )
                ),
                "bias_hours": float(
                    np.mean(
                        pred - actual
                    )
                ),
            }
        )

    range_results = pd.DataFrame(
        range_rows
    )

    print(
        range_results.to_string(
            index=False
        )
    )

    # ========================================================
    # ENGINE-WISE TEST PERFORMANCE
    # ========================================================

    section(
        "ENGINE-WISE TEST PERFORMANCE"
    )

    test_predictions = test_df[
        [
            "engine_id",
            "flight_id",
            "flight_start",
            TARGET,
        ]
    ].copy()

    test_predictions[
        "predicted_rul_hours"
    ] = test_pred

    test_predictions[
        "error_hours"
    ] = (
        test_predictions[
            "predicted_rul_hours"
        ]
        - test_predictions[
            TARGET
        ]
    )

    test_predictions[
        "absolute_error_hours"
    ] = np.abs(
        test_predictions[
            "error_hours"
        ]
    )

    engine_rows = []

    for engine_id, group in (
        test_predictions.groupby(
            "engine_id"
        )
    ):

        actual = group[
            TARGET
        ].to_numpy()

        pred = group[
            "predicted_rul_hours"
        ].to_numpy()

        overall = regression_metrics(
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

                **low,
            }
        )

        if low[
            "low_rul_count"
        ] > 0:

            low_text = (
                f"LowRUL-MAE="
                f"{low['low_rul_MAE_hours']:.2f} h"
            )

        else:

            low_text = (
                "LowRUL-MAE=N/A"
            )

        print(
            f"{engine_id:<14}"
            f"n={len(group):>2}  "
            f"MAE={overall['MAE_hours']:.2f} h  "
            f"RMSE={overall['RMSE_hours']:.2f} h  "
            f"R²={overall['R2']:.3f}  "
            f"{low_text}"
        )

    engine_results = pd.DataFrame(
        engine_rows
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    section(
        "FEATURE IMPORTANCE"
    )

    feature_importance = None

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        feature_importance = (
            pd.DataFrame(
                {
                    "feature":
                        best_features,

                    "importance":
                        best_model
                        .feature_importances_,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
        )

        print(
            feature_importance.head(
                30
            ).to_string(
                index=False
            )
        )

    else:

        print(
            "Selected model does not expose "
            "feature_importances_."
        )

    # ========================================================
    # PREVIOUS RUL USAGE CHECK
    # ========================================================

    section(
        "SEQUENCE FEATURE CHECK"
    )

    print(
        f"Previous RUL included: "
        f"{'YES' if best_experiment == 'health_change_plus_previous_rul' else 'NO'}"
    )

    if (
        "previous_rul_hours"
        in best_features
    ):
        print(
            "WARNING: previous_rul_hours is "
            "an observed-target-derived feature."
        )

        print(
            "Use it for experimental sequence analysis, "
            "not as a direct production input."
        )

    # ========================================================
    # SAVE
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "final_rul_v3.joblib"
    )

    bundle = {
        "model": best_model,
        "imputer": best_imputer,
        "feature_names":
            best_features,
        "target": TARGET,
        "model_name":
            best_model_name,
        "experiment":
            best_experiment,
        "low_rul_threshold_hours":
            LOW_RUL_THRESHOLD,
        "prediction_postprocess":
            "max(prediction, 0.0)",
        "split_strategy":
            "engine-wise chronological",
        "production_warning":
            (
                "previous_rul_hours is target-derived "
                "and must be replaced by a previous "
                "prediction/state before production use"
            ),
    }

    joblib.dump(
        bundle,
        model_path,
    )

    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    validation_path = (
        RESULT_DIR
        / "rul_v3_validation_results.csv"
    )

    validation_results.to_csv(
        validation_path,
        index=False,
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_predictions_path = (
        RESULT_DIR
        / "rul_v3_test_predictions.csv"
    )

    test_predictions.to_csv(
        test_predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Engine metrics
    # --------------------------------------------------------

    engine_metrics_path = (
        RESULT_DIR
        / "rul_v3_engine_test_metrics.csv"
    )

    engine_results.to_csv(
        engine_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # RUL range metrics
    # --------------------------------------------------------

    ranges_path = (
        RESULT_DIR
        / "rul_v3_test_rul_ranges.csv"
    )

    range_results.to_csv(
        ranges_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if feature_importance is not None:

        importance_path = (
            RESULT_DIR
            / "rul_v3_feature_importance.csv"
        )

        feature_importance.to_csv(
            importance_path,
            index=False,
        )

    else:

        importance_path = None

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics_path = (
        RESULT_DIR
        / "rul_v3_metrics.json"
    )

    metrics_json = {
        "selected_model":
            best_model_name,

        "selected_experiment":
            best_experiment,

        "feature_count":
            len(best_features),

        "features":
            best_features,

        "validation_results":
            validation_results
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "test_overall":
            test_overall,

        "test_low_rul":
            test_low,

        "test_prediction_range": {
            "actual_min":
                float(y_test.min()),

            "actual_max":
                float(y_test.max()),

            "predicted_min":
                float(test_pred.min()),

            "predicted_max":
                float(test_pred.max()),

            "predicted_mean":
                float(test_pred.mean()),
        },

        "previous_rul_warning":
            (
                "previous_rul_hours is target-derived "
                "and is not production-safe as-is."
            ),
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_json,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    metadata_path = (
        MODEL_DIR
        / "rul_v3_model_metadata.json"
    )

    metadata = {
        "model_type":
            type(best_model).__name__,

        "model_name":
            best_model_name,

        "experiment":
            best_experiment,

        "target":
            TARGET,

        "feature_count":
            len(best_features),

        "features":
            best_features,

        "train_samples":
            len(train_df),

        "validation_samples":
            len(val_df),

        "test_samples":
            len(test_df),

        "split_strategy":
            "engine-wise chronological",

        "low_rul_threshold_hours":
            LOW_RUL_THRESHOLD,

        "sequence_features":
            True,

        "previous_rul_used":
            (
                "previous_rul_hours"
                in best_features
            ),

        "production_warning":
            (
                "Observed previous RUL is suitable "
                "for sequence diagnostics but should "
                "be replaced by previous prediction/state "
                "before deployment."
            ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
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
        f"\nResults:"
    )

    print(
        f"  {validation_path}"
    )

    print(
        f"  {test_predictions_path}"
    )

    print(
        f"  {engine_metrics_path}"
    )

    print(
        f"  {ranges_path}"
    )

    print(
        f"  {metrics_path}"
    )

    if importance_path is not None:
        print(
            f"  {importance_path}"
        )

    print(
        f"\nMetadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nRUL V3 TRAINING COMPLETE."
    )


if __name__ == "__main__":
    main()