from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

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

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "training/datasets/processed/rul_v1/split"
)

MODEL_DIR = Path(
    "models/rul_v1"
)

RESULT_DIR = Path(
    "training/datasets/processed/rul_v1/results"
)


TRAIN_PATH = BASE_DIR / "rul_train_v1.csv"
VAL_PATH = BASE_DIR / "rul_validation_v1.csv"
TEST_PATH = BASE_DIR / "rul_test_v1.csv"

FEATURE_PATH = BASE_DIR / "rul_v1_features.txt"

TARGET = "rul_hours_remaining"


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
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
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


def print_metrics(
    name: str,
    metrics: dict,
):
    print(f"\n{name}")

    print(
        f"  MAE           : "
        f"{metrics['MAE_hours']:.4f} h"
    )

    print(
        f"  RMSE          : "
        f"{metrics['RMSE_hours']:.4f} h"
    )

    print(
        f"  R²            : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"  Median AE     : "
        f"{metrics['Median_AE_hours']:.4f} h"
    )

    print(
        f"  Mean Bias     : "
        f"{metrics['Mean_Bias_hours']:.4f} h"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_names():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found:\n{FEATURE_PATH}"
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
# MAIN
# ============================================================

def main():

    section(
        "RUL V1 BASELINE MODEL TRAINING"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    features = load_feature_names()

    train_df, val_df, test_df = load_data()

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

    # --------------------------------------------------------
    # REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    required_columns = set(features)
    required_columns.add(TARGET)

    for split_name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        missing = sorted(
            required_columns - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{split_name} is missing columns:\n"
                + "\n".join(
                    f"  - {x}"
                    for x in missing
                )
            )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------
    #
    # Fit medians ONLY on training data.
    #
    # --------------------------------------------------------

    section("FEATURE PREPROCESSING")

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

    train_medians = dict(
        zip(
            features,
            imputer.statistics_
        )
    )

    print(
        "Median imputation fitted on TRAIN only."
    )

    print(
        "Train matrix shape :",
        X_train_imp.shape
    )

    print(
        "Validation shape    :",
        X_val_imp.shape
    )

    print(
        "Test shape          :",
        X_test_imp.shape
    )

    # --------------------------------------------------------
    # TARGET SUMMARY
    # --------------------------------------------------------

    section("TARGET SUMMARY")

    for name, y in [
        ("TRAIN", y_train),
        ("VALIDATION", y_val),
        ("TEST", y_test),
    ]:

        print(
            f"{name:<12}"
            f"n={len(y):>3}  "
            f"min={y.min():>6.2f}  "
            f"max={y.max():>6.2f}  "
            f"mean={y.mean():>6.2f}  "
            f"median={np.median(y):>6.2f}"
        )

    # ========================================================
    # MODEL CANDIDATES
    # ========================================================

    section("MODEL CANDIDATES")

    models = {

        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=4,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    validation_results = {}

    trained_models = {}

    # ========================================================
    # TRAIN + VALIDATION
    # ========================================================

    section(
        "TRAINING + VALIDATION"
    )

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        model.fit(
            X_train_imp,
            y_train
        )

        val_pred = model.predict(
            X_val_imp
        )

        metrics = regression_metrics(
            y_val,
            val_pred
        )

        validation_results[name] = metrics

        trained_models[name] = model

        print_metrics(
            "Validation metrics",
            metrics
        )

    # ========================================================
    # MODEL SELECTION
    # ========================================================
    #
    # Primary criterion:
    #   lowest validation MAE
    #
    # Secondary:
    #   lowest validation RMSE
    #
    # R² is NOT the primary selection metric because
    # validation set contains only 22 samples.
    #
    # ========================================================

    section("MODEL SELECTION")

    ranked_models = sorted(
        validation_results.keys(),
        key=lambda name: (
            validation_results[name]["MAE_hours"],
            validation_results[name]["RMSE_hours"],
        )
    )

    best_model_name = ranked_models[0]

    best_model = trained_models[
        best_model_name
    ]

    print(
        f"Selected model : {best_model_name}"
    )

    print(
        f"Validation MAE : "
        f"{validation_results[best_model_name]['MAE_hours']:.4f} h"
    )

    print(
        f"Validation RMSE: "
        f"{validation_results[best_model_name]['RMSE_hours']:.4f} h"
    )

    # ========================================================
    # VALIDATION PREDICTION TABLE
    # ========================================================

    val_predictions = val_df[
        [
            "engine_id",
            "flight_id",
            "flight_start",
            TARGET,
        ]
    ].copy()

    val_predictions[
        "predicted_rul_hours"
    ] = best_model.predict(
        X_val_imp
    )

    val_predictions[
        "prediction_error_hours"
    ] = (
        val_predictions[
            "predicted_rul_hours"
        ]
        - val_predictions[TARGET]
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    section(
        "FINAL TEST EVALUATION"
    )

    test_pred = best_model.predict(
        X_test_imp
    )

    # --------------------------------------------------------
    # Avoid negative RUL predictions for final presentation.
    # --------------------------------------------------------

    test_pred_clipped = np.maximum(
        test_pred,
        0.0
    )

    test_metrics_raw = regression_metrics(
        y_test,
        test_pred
    )

    test_metrics_clipped = regression_metrics(
        y_test,
        test_pred_clipped
    )

    print_metrics(
        "TEST metrics - raw",
        test_metrics_raw
    )

    print_metrics(
        "TEST metrics - clipped at 0 h",
        test_metrics_clipped
    )

    # --------------------------------------------------------
    # Select clipped predictions for production output.
    # --------------------------------------------------------

    final_test_metrics = (
        test_metrics_clipped
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions = test_df[
        [
            "engine_id",
            "flight_id",
            "flight_start",
            TARGET,
        ]
    ].copy()

    test_predictions[
        "predicted_rul_hours_raw"
    ] = test_pred

    test_predictions[
        "predicted_rul_hours"
    ] = test_pred_clipped

    test_predictions[
        "prediction_error_hours"
    ] = (
        test_predictions[
            "predicted_rul_hours"
        ]
        - test_predictions[TARGET]
    )

    test_predictions[
        "absolute_error_hours"
    ] = np.abs(
        test_predictions[
            "prediction_error_hours"
        ]
    )

    test_predictions[
        "percent_error"
    ] = np.where(
        test_predictions[TARGET] > 0,
        (
            test_predictions[
                "absolute_error_hours"
            ]
            / test_predictions[TARGET]
            * 100
        ),
        np.nan,
    )

    # ========================================================
    # ENGINE-WISE TEST PERFORMANCE
    # ========================================================

    section(
        "ENGINE-WISE TEST PERFORMANCE"
    )

    engine_results = []

    for engine_id, group in (
        test_predictions.groupby(
            "engine_id"
        )
    ):

        y_true_engine = group[
            TARGET
        ].to_numpy()

        y_pred_engine = group[
            "predicted_rul_hours"
        ].to_numpy()

        engine_metrics = (
            regression_metrics(
                y_true_engine,
                y_pred_engine
            )
        )

        engine_results.append(
            {
                "engine_id": engine_id,
                "samples": len(group),
                **engine_metrics,
            }
        )

        print(
            f"{engine_id:<14}"
            f"n={len(group):>2}  "
            f"MAE={engine_metrics['MAE_hours']:.2f} h  "
            f"RMSE={engine_metrics['RMSE_hours']:.2f} h  "
            f"R2={engine_metrics['R2']:.3f}"
        )

    engine_results_df = pd.DataFrame(
        engine_results
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

        importance = (
            best_model.feature_importances_
        )

        feature_importance_df = pd.DataFrame(
            {
                "feature": features,
                "importance": importance,
            }
        ).sort_values(
            "importance",
            ascending=False
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
    # PREDICTION RANGE CHECK
    # ========================================================

    section(
        "PREDICTION RANGE"
    )

    print(
        f"Raw prediction min : "
        f"{test_pred.min():.4f} h"
    )

    print(
        f"Raw prediction max : "
        f"{test_pred.max():.4f} h"
    )

    print(
        f"Clipped min        : "
        f"{test_pred_clipped.min():.4f} h"
    )

    print(
        f"Clipped max        : "
        f"{test_pred_clipped.max():.4f} h"
    )

    negative_predictions = (
        test_pred < 0
    ).sum()

    print(
        f"Negative predictions: "
        f"{negative_predictions}"
    )

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / "final_rul_v1.joblib"
    )

    bundle = {
        "model": best_model,
        "imputer": imputer,
        "feature_names": features,
        "target": TARGET,
        "model_name": best_model_name,
        "prediction_postprocess": (
            "max(prediction, 0.0)"
        ),
        "metadata": {
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "feature_count": len(features),
        },
    }

    joblib.dump(
        bundle,
        model_path
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_output = {
        "selected_model": best_model_name,
        "validation": validation_results,
        "test_raw": test_metrics_raw,
        "test_clipped": test_metrics_clipped,
    }

    metrics_path = (
        RESULT_DIR
        / "rul_v1_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    val_predictions_path = (
        RESULT_DIR
        / "rul_v1_validation_predictions.csv"
    )

    test_predictions_path = (
        RESULT_DIR
        / "rul_v1_test_predictions.csv"
    )

    val_predictions.to_csv(
        val_predictions_path,
        index=False
    )

    test_predictions.to_csv(
        test_predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Save engine results
    # --------------------------------------------------------

    engine_results_path = (
        RESULT_DIR
        / "rul_v1_engine_test_metrics.csv"
    )

    engine_results_df.to_csv(
        engine_results_path,
        index=False
    )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    if feature_importance_df is not None:

        importance_path = (
            RESULT_DIR
            / "rul_v1_feature_importance.csv"
        )

        feature_importance_df.to_csv(
            importance_path,
            index=False
        )

    # --------------------------------------------------------
    # Save model metadata
    # --------------------------------------------------------

    metadata_path = (
        MODEL_DIR
        / "rul_v1_model_metadata.json"
    )

    metadata = {
        "model_type": type(
            best_model
        ).__name__,
        "model_name": best_model_name,
        "target": TARGET,
        "feature_count": len(features),
        "features": features,
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "split_strategy": (
            "engine-wise chronological"
        ),
        "imputation": "median fitted on train",
        "negative_prediction_handling": (
            "clip to zero"
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    section("OUTPUT")

    print(
        f"Model saved:"
    )

    print(
        f"  {model_path}"
    )

    print(
        f"\nResults saved:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {val_predictions_path}"
    )

    print(
        f"  {test_predictions_path}"
    )

    print(
        f"  {engine_results_path}"
    )

    if feature_importance_df is not None:
        print(
            f"  {importance_path}"
        )

    print(
        f"\nMetadata saved:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nRUL V1 BASELINE TRAINING COMPLETE."
    )


if __name__ == "__main__":
    main()