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
    "models/rul_v2"
)

RESULT_DIR = Path(
    "training/datasets/processed/rul_v2"
)

TRAIN_PATH = BASE_DIR / "rul_train_v1.csv"
VAL_PATH = BASE_DIR / "rul_validation_v1.csv"
TEST_PATH = BASE_DIR / "rul_test_v1.csv"

FEATURE_PATH = BASE_DIR / "rul_v1_features.txt"

TARGET = "rul_hours_remaining"

LOW_RUL_THRESHOLD = 10.0


# ============================================================
# HELPERS
# ============================================================

def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def regression_metrics(
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


def low_rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = LOW_RUL_THRESHOLD,
):
    mask = y_true <= threshold

    count = int(mask.sum())

    if count == 0:
        return {
            "count": 0,
            "MAE_hours": None,
            "RMSE_hours": None,
            "Mean_Bias_hours": None,
        }

    y_low = y_true[mask]
    p_low = y_pred[mask]

    return {
        "count": count,
        "MAE_hours": float(
            mean_absolute_error(
                y_low,
                p_low,
            )
        ),
        "RMSE_hours": float(
            np.sqrt(
                mean_squared_error(
                    y_low,
                    p_low,
                )
            )
        ),
        "Mean_Bias_hours": float(
            np.mean(p_low - y_low)
        ),
    }


def print_metrics(
    name: str,
    metrics: dict,
):
    print(f"\n{name}")

    print(
        f"  MAE        : "
        f"{metrics['MAE_hours']:.4f} h"
    )

    print(
        f"  RMSE       : "
        f"{metrics['RMSE_hours']:.4f} h"
    )

    print(
        f"  R²         : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"  Median AE  : "
        f"{metrics['Median_AE_hours']:.4f} h"
    )

    print(
        f"  Mean Bias  : "
        f"{metrics['Mean_Bias_hours']:.4f} h"
    )


def print_low_metrics(
    metrics: dict,
):
    if metrics["count"] == 0:
        print("\nLow-RUL metrics: no samples")
        return

    print(
        f"\nLow-RUL (<= {LOW_RUL_THRESHOLD:.0f} h)"
    )

    print(
        f"  Samples    : "
        f"{metrics['count']}"
    )

    print(
        f"  MAE        : "
        f"{metrics['MAE_hours']:.4f} h"
    )

    print(
        f"  RMSE       : "
        f"{metrics['RMSE_hours']:.4f} h"
    )

    print(
        f"  Mean Bias  : "
        f"{metrics['Mean_Bias_hours']:.4f} h"
    )


def load_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_PATH}"
        )

    features = [
        line.strip()
        for line in FEATURE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not features:
        raise ValueError(
            "Feature list is empty."
        )

    return features


def load_data():
    for path in [
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset:\n{path}"
            )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def build_low_rul_weights(
    y: np.ndarray,
):
    """
    Give more training weight to low-RUL samples.

    Weight:
        RUL >= 20h -> 1.0
        RUL = 10h  -> 2.0
        RUL = 0h   -> 3.0

    This is deliberately mild; we do not want the model
    to ignore the higher-RUL region.
    """

    weights = np.ones(
        len(y),
        dtype=float,
    )

    low_mask = y < 20.0

    weights[low_mask] = (
        1.0
        + 2.0
        * (
            (20.0 - y[low_mask])
            / 20.0
        )
    )

    return weights


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "RUL V2 HEALTH-FOCUSED REGRESSION"
    )

    # ========================================================
    # LOAD
    # ========================================================

    features = load_features()

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

    print(
        f"Features        : {len(features)}"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = set(
        features
    )
    required_columns.add(TARGET)

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        missing = sorted(
            required_columns - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} missing columns:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in missing
                )
            )

    # ========================================================
    # SORT
    # ========================================================

    for df in [
        train_df,
        val_df,
        test_df,
    ]:
        df["flight_start"] = pd.to_datetime(
            df["flight_start"],
            errors="raise",
        )

    # ========================================================
    # X / y
    # ========================================================

    X_train = train_df[
        features
    ].copy()

    y_train = train_df[
        TARGET
    ].astype(float).to_numpy()

    X_val = val_df[
        features
    ].copy()

    y_val = val_df[
        TARGET
    ].astype(float).to_numpy()

    X_test = test_df[
        features
    ].copy()

    y_test = test_df[
        TARGET
    ].astype(float).to_numpy()

    # ========================================================
    # PREPROCESSING
    # ========================================================

    section(
        "FEATURE PREPROCESSING"
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_val_imp = imputer.transform(
        X_val
    )

    X_test_imp = imputer.transform(
        X_test
    )

    print(
        "Median imputation fitted on TRAIN only."
    )

    print(
        f"Train matrix : {X_train_imp.shape}"
    )

    print(
        f"Val matrix   : {X_val_imp.shape}"
    )

    print(
        f"Test matrix  : {X_test_imp.shape}"
    )

    # ========================================================
    # TRAIN TARGET DISTRIBUTION
    # ========================================================

    section(
        "TRAINING TARGET DISTRIBUTION"
    )

    print(
        f"Min    : {y_train.min():.2f} h"
    )

    print(
        f"Max    : {y_train.max():.2f} h"
    )

    print(
        f"Mean   : {y_train.mean():.2f} h"
    )

    print(
        f"Median : {np.median(y_train):.2f} h"
    )

    print(
        f"RUL <= 10 h : "
        f"{(y_train <= 10).sum()}"
    )

    print(
        f"RUL <= 20 h : "
        f"{(y_train <= 20).sum()}"
    )

    # ========================================================
    # SAMPLE WEIGHTS
    # ========================================================

    sample_weights = build_low_rul_weights(
        y_train
    )

    print(
        "\nLow-RUL weighted training enabled:"
    )

    print(
        f"Weight min : {sample_weights.min():.2f}"
    )

    print(
        f"Weight max : {sample_weights.max():.2f}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    section(
        "MODEL CANDIDATES"
    )

    models = {

        "random_forest": (
            RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            False,
        ),

        "extra_trees": (
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            False,
        ),

        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=4,
                l2_regularization=1.5,
                random_state=42,
            ),
            False,
        ),

        "random_forest_low_rul_weighted": (
            RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            True,
        ),
    }

    trained_models = {}
    validation_records = []

    # ========================================================
    # TRAIN + VALIDATION
    # ========================================================

    section(
        "TRAINING + VALIDATION"
    )

    for name, (
        model,
        use_weights,
    ) in models.items():

        print(
            f"\nTraining: {name}"
        )

        if use_weights:

            model.fit(
                X_train_imp,
                y_train,
                sample_weight=sample_weights,
            )

        else:

            model.fit(
                X_train_imp,
                y_train,
            )

        val_pred = model.predict(
            X_val_imp
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

        # ----------------------------------------------------
        # Model-selection score
        #
        # 60% overall MAE
        # 40% low-RUL MAE
        # ----------------------------------------------------

        if low["MAE_hours"] is None:
            selection_score = overall[
                "MAE_hours"
            ]
        else:
            selection_score = (
                0.60
                * overall["MAE_hours"]
                + 0.40
                * low["MAE_hours"]
            )

        validation_records.append(
            {
                "model": name,
                **overall,
                "low_rul_count": low[
                    "count"
                ],
                "low_rul_MAE_hours": low[
                    "MAE_hours"
                ],
                "low_rul_RMSE_hours": low[
                    "RMSE_hours"
                ],
                "low_rul_bias_hours": low[
                    "Mean_Bias_hours"
                ],
                "selection_score": (
                    float(selection_score)
                ),
            }
        )

        trained_models[name] = model

        print_metrics(
            "Validation",
            overall,
        )

        print_low_metrics(
            low
        )

        print(
            f"\n  Selection score : "
            f"{selection_score:.4f}"
        )

    # ========================================================
    # RANK MODELS
    # ========================================================

    validation_results = pd.DataFrame(
        validation_records
    ).sort_values(
        "selection_score"
    )

    section(
        "VALIDATION MODEL RANKING"
    )

    display_columns = [
        "model",
        "MAE_hours",
        "RMSE_hours",
        "R2",
        "low_rul_MAE_hours",
        "low_rul_bias_hours",
        "selection_score",
    ]

    print(
        validation_results[
            display_columns
        ].to_string(
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
    # FINAL TEST
    # ========================================================

    section(
        "FINAL CHRONOLOGICAL TEST"
    )

    test_pred = best_model.predict(
        X_test_imp
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
    )

    print_low_metrics(
        test_low
    )

    # ========================================================
    # TEST RUL BINS
    # ========================================================

    section(
        "TEST PERFORMANCE BY RUL RANGE"
    )

    bin_definitions = [
        ("0-5h", 0.0, 5.0),
        ("5-10h", 5.0, 10.0),
        ("10-20h", 10.0, 20.0),
        ("20-40h", 20.0, 40.0),
        ("40h+", 40.0, np.inf),
    ]

    bin_records = []

    for label, lower, upper in bin_definitions:

        if np.isinf(upper):

            mask = (
                y_test >= lower
            )

        else:

            mask = (
                (y_test >= lower)
                & (y_test < upper)
            )

        count = int(mask.sum())

        if count == 0:
            bin_records.append(
                {
                    "rul_range": label,
                    "samples": 0,
                    "MAE_hours": None,
                    "RMSE_hours": None,
                    "Mean_Bias_hours": None,
                }
            )
            continue

        actual = y_test[mask]
        predicted = test_pred[mask]

        bin_records.append(
            {
                "rul_range": label,
                "samples": count,
                "MAE_hours": float(
                    mean_absolute_error(
                        actual,
                        predicted,
                    )
                ),
                "RMSE_hours": float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            predicted,
                        )
                    )
                ),
                "Mean_Bias_hours": float(
                    np.mean(
                        predicted - actual
                    )
                ),
            }
        )

    test_bins_df = pd.DataFrame(
        bin_records
    )

    print(
        test_bins_df.to_string(
            index=False
        )
    )

    # ========================================================
    # ENGINE-WISE TEST PERFORMANCE
    # ========================================================

    section(
        "ENGINE-WISE TEST PERFORMANCE"
    )

    test_output = test_df[
        [
            "engine_id",
            "flight_id",
            "flight_start",
            TARGET,
        ]
    ].copy()

    test_output[
        "predicted_rul_hours"
    ] = test_pred

    test_output[
        "error_hours"
    ] = (
        test_output[
            "predicted_rul_hours"
        ]
        - test_output[TARGET]
    )

    test_output[
        "absolute_error_hours"
    ] = np.abs(
        test_output[
            "error_hours"
        ]
    )

    engine_records = []

    for engine_id, group in (
        test_output.groupby(
            "engine_id"
        )
    ):

        actual = group[
            TARGET
        ].to_numpy()

        predicted = group[
            "predicted_rul_hours"
        ].to_numpy()

        metrics = regression_metrics(
            actual,
            predicted,
        )

        low = low_rul_metrics(
            actual,
            predicted,
        )

        engine_records.append(
            {
                "engine_id": engine_id,
                "samples": len(group),
                "MAE_hours": metrics[
                    "MAE_hours"
                ],
                "RMSE_hours": metrics[
                    "RMSE_hours"
                ],
                "R2": metrics["R2"],
                "low_rul_count": low[
                    "count"
                ],
                "low_rul_MAE_hours": low[
                    "MAE_hours"
                ],
            }
        )

        print(
            f"{engine_id:<14}"
            f"n={len(group):>2}  "
            f"MAE={metrics['MAE_hours']:.2f} h  "
            f"RMSE={metrics['RMSE_hours']:.2f} h  "
            f"R2={metrics['R2']:.3f}  "
            f"LowRUL-MAE="
            f"{low['MAE_hours']:.2f} h"
            if low["MAE_hours"] is not None
            else
            f"{engine_id:<14}"
            f"n={len(group):>2}  "
            f"MAE={metrics['MAE_hours']:.2f} h  "
            f"RMSE={metrics['RMSE_hours']:.2f} h  "
            f"R2={metrics['R2']:.3f}"
        )

    engine_results_df = pd.DataFrame(
        engine_records
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    section(
        "FEATURE IMPORTANCE"
    )

    feature_importance_df = None

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        feature_importance_df = (
            pd.DataFrame(
                {
                    "feature": features,
                    "importance":
                        best_model.feature_importances_,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
        )

        print(
            feature_importance_df.head(
                20
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
    # PREDICTION DISTRIBUTION
    # ========================================================

    section(
        "PREDICTION RANGE"
    )

    print(
        f"Actual min       : "
        f"{y_test.min():.2f} h"
    )

    print(
        f"Actual max       : "
        f"{y_test.max():.2f} h"
    )

    print(
        f"Predicted min    : "
        f"{test_pred.min():.2f} h"
    )

    print(
        f"Predicted max    : "
        f"{test_pred.max():.2f} h"
    )

    print(
        f"Predicted mean   : "
        f"{test_pred.mean():.2f} h"
    )

    print(
        f"Actual mean      : "
        f"{y_test.mean():.2f} h"
    )

    print(
        f"Mean bias        : "
        f"{test_overall['Mean_Bias_hours']:.2f} h"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    section(
        "BASELINE COMPARISON"
    )

    train_mean = y_train.mean()
    train_median = np.median(
        y_train
    )

    mean_pred = np.full(
        len(y_test),
        train_mean,
    )

    median_pred = np.full(
        len(y_test),
        train_median,
    )

    mean_baseline_mae = (
        mean_absolute_error(
            y_test,
            mean_pred,
        )
    )

    median_baseline_mae = (
        mean_absolute_error(
            y_test,
            median_pred,
        )
    )

    print(
        f"Train-mean baseline MAE   : "
        f"{mean_baseline_mae:.4f} h"
    )

    print(
        f"Train-median baseline MAE : "
        f"{median_baseline_mae:.4f} h"
    )

    print(
        f"RUL V2 model MAE           : "
        f"{test_overall['MAE_hours']:.4f} h"
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
        / "final_rul_v2.joblib"
    )

    bundle = {
        "model": best_model,
        "imputer": imputer,
        "feature_names": features,
        "target": TARGET,
        "model_name": best_model_name,
        "low_rul_threshold_hours":
            LOW_RUL_THRESHOLD,
        "prediction_postprocess":
            "max(prediction, 0.0)",
        "split_strategy":
            "engine-wise chronological",
        "feature_policy":
            "health-focused V1 feature set",
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
        / "rul_v2_validation_results.csv"
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
        / "rul_v2_test_predictions.csv"
    )

    test_output.to_csv(
        test_predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Engine metrics
    # --------------------------------------------------------

    engine_metrics_path = (
        RESULT_DIR
        / "rul_v2_engine_test_metrics.csv"
    )

    engine_results_df.to_csv(
        engine_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # RUL-bin metrics
    # --------------------------------------------------------

    bins_path = (
        RESULT_DIR
        / "rul_v2_test_rul_bins.csv"
    )

    test_bins_df.to_csv(
        bins_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if feature_importance_df is not None:

        importance_path = (
            RESULT_DIR
            / "rul_v2_feature_importance.csv"
        )

        feature_importance_df.to_csv(
            importance_path,
            index=False,
        )

    else:

        importance_path = None

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics_json_path = (
        RESULT_DIR
        / "rul_v2_metrics.json"
    )

    metrics_json = {
        "selected_model":
            best_model_name,

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

        "baseline": {
            "train_mean_MAE_hours":
                float(mean_baseline_mae),
            "train_median_MAE_hours":
                float(median_baseline_mae),
        },

        "test_prediction_range": {
            "min":
                float(test_pred.min()),
            "max":
                float(test_pred.max()),
            "mean":
                float(test_pred.mean()),
        },
    }

    with open(
        metrics_json_path,
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
        / "rul_v2_model_metadata.json"
    )

    metadata = {
        "model_type":
            type(best_model).__name__,

        "model_name":
            best_model_name,

        "target":
            TARGET,

        "feature_count":
            len(features),

        "features":
            features,

        "train_samples":
            len(train_df),

        "validation_samples":
            len(val_df),

        "test_samples":
            len(test_df),

        "split_strategy":
            "engine-wise chronological",

        "model_selection":
            "60% overall validation MAE + "
            "40% validation low-RUL MAE",

        "low_rul_threshold_hours":
            LOW_RUL_THRESHOLD,

        "imputation":
            "median fitted on training set",

        "negative_prediction_handling":
            "clip to zero",
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
        f"  {bins_path}"
    )

    print(
        f"  {metrics_json_path}"
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
        "\nRUL V2 TRAINING COMPLETE."
    )


if __name__ == "__main__":
    main()