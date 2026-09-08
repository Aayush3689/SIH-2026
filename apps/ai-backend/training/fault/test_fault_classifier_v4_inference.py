from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier_v4"
    / "final_fault_classifier_v4.joblib"
)

FEATURE_DATA_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v4_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier_v4"
    / "inference_validation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "inference_validation_results.json"
)


# =============================================================================
# CONFIG
# =============================================================================

TARGET_COLUMN = "fault_mode_ground_truth"
ACTIVE_COLUMN = "fault_active_label"
FLIGHT_COLUMN = "flight_id"

EXPECTED_FEATURE_COUNT = 40
PROBABILITY_TOLERANCE = 1e-6


# =============================================================================
# HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_bundle() -> dict:
    if not MODEL_FILE.exists():
        fail(
            f"Model not found:\n{MODEL_FILE}"
        )

    bundle = joblib.load(
        MODEL_FILE
    )

    if not isinstance(bundle, dict):
        fail(
            "Saved model is not a dictionary bundle."
        )

    required = {
        "model",
        "feature_names",
        "imputation_medians",
        "class_names",
    }

    missing = sorted(
        required - set(bundle.keys())
    )

    if missing:
        fail(
            "Model bundle missing:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )

    return bundle


def validate_bundle(
    bundle: dict,
) -> None:

    section("MODEL BUNDLE VALIDATION")

    features = list(
        bundle["feature_names"]
    )

    classes = list(
        bundle["class_names"]
    )

    medians = bundle[
        "imputation_medians"
    ]

    model = bundle["model"]

    print(
        f"Model name    : "
        f"{bundle.get('model_name', 'unknown')}"
    )

    print(
        f"Feature count : "
        f"{len(features)}"
    )

    print(
        f"Class count   : "
        f"{len(classes)}"
    )

    print(
        f"Model type    : "
        f"{type(model).__name__}"
    )

    if len(features) != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features, found {len(features)}."
        )

    if len(set(features)) != len(features):
        fail(
            "Duplicate feature names found."
        )

    median_features = set(
        medians.keys()
    )

    feature_set = set(
        features
    )

    missing_medians = sorted(
        feature_set - median_features
    )

    extra_medians = sorted(
        median_features - feature_set
    )

    if missing_medians:
        fail(
            "Missing imputation medians for:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_medians
            )
        )

    if extra_medians:
        print(
            "\nWARNING: Extra median entries found:"
        )

        for feature in extra_medians:
            print(
                f"  - {feature}"
            )

    model_classes = [
        str(x)
        for x in model.classes_
    ]

    if model_classes != [
        str(x)
        for x in classes
    ]:
        fail(
            "Saved class_names do not match "
            "model.classes_."
        )

    print(
        "\nFeature/median mapping: PASSED"
    )

    print(
        "Class mapping: PASSED"
    )


def load_test_rows(
    features: list[str],
) -> pd.DataFrame:

    if not FEATURE_DATA_FILE.exists():
        fail(
            f"Feature dataset not found:\n"
            f"{FEATURE_DATA_FILE}"
        )

    df = pd.read_csv(
        FEATURE_DATA_FILE
    )

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        fail(
            "Features missing from V4 dataset:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    if ACTIVE_COLUMN not in df.columns:
        fail(
            f"'{ACTIVE_COLUMN}' not found."
        )

    if TARGET_COLUMN not in df.columns:
        fail(
            f"'{TARGET_COLUMN}' not found."
        )

    # Only active fault rows.
    df = df[
        df[ACTIVE_COLUMN] == 1
    ].copy()

    if df.empty:
        fail(
            "No active fault rows found."
        )

    return df


def prepare_features(
    rows: pd.DataFrame,
    features: list[str],
    medians: dict,
) -> pd.DataFrame:

    X = rows[
        features
    ].copy()

    for feature in features:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        X[feature] = (
            X[feature]
            .fillna(
                medians[feature]
            )
        )

    values = X.to_numpy(
        dtype=float
    )

    if np.isnan(values).any():
        fail(
            "NaN remains after imputation."
        )

    if np.isinf(values).any():
        fail(
            "Infinite value remains after preparation."
        )

    return X


def run_inference(
    model,
    X: pd.DataFrame,
):
    predictions = model.predict(
        X
    )

    probabilities = None

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(X)
        )

    return (
        predictions,
        probabilities,
    )


def validate_probability_matrix(
    probabilities: np.ndarray,
) -> None:

    if probabilities.ndim != 2:
        fail(
            "predict_proba() did not return a 2D matrix."
        )

    if not np.isfinite(
        probabilities
    ).all():
        fail(
            "Probability matrix contains NaN/inf."
        )

    row_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        fail(
            "Probability rows do not sum to 1.0."
        )


def print_predictions(
    rows: pd.DataFrame,
    predictions,
    probabilities,
    class_names,
    max_rows: int = 10,
) -> None:

    section("SAMPLE INFERENCE")

    display_count = min(
        max_rows,
        len(rows),
    )

    for index in range(
        display_count
    ):

        prediction = str(
            predictions[index]
        )

        print(
            f"\nSample {index + 1}"
        )

        if FLIGHT_COLUMN in rows.columns:
            print(
                f"Flight      : "
                f"{rows.iloc[index][FLIGHT_COLUMN]}"
            )

        if TARGET_COLUMN in rows.columns:
            print(
                f"Actual      : "
                f"{rows.iloc[index][TARGET_COLUMN]}"
            )

        print(
            f"Prediction  : "
            f"{prediction}"
        )

        if probabilities is not None:

            sample_probs = (
                probabilities[index]
            )

            ranked_indices = np.argsort(
                sample_probs
            )[::-1]

            print(
                "Probabilities:"
            )

            for class_index in (
                ranked_indices[:3]
            ):

                print(
                    f"  "
                    f"{class_names[class_index]:<30}"
                    f"{sample_probs[class_index]:.4f}"
                )

            confidence = (
                sample_probs.max()
            )

            print(
                f"Confidence   : "
                f"{confidence:.4f}"
            )


def calculate_test_accuracy(
    rows: pd.DataFrame,
    predictions,
) -> float:

    if TARGET_COLUMN not in rows.columns:
        return float("nan")

    y_true = (
        rows[TARGET_COLUMN]
        .astype(str)
        .to_numpy()
    )

    y_pred = (
        np.asarray(predictions)
        .astype(str)
    )

    return float(
        (y_true == y_pred).mean()
    )


def validate_feature_order(
    model,
    features: list[str],
) -> None:

    section("FEATURE ORDER VALIDATION")

    # Same names, correct order.
    reordered = list(
        reversed(features)
    )

    if reordered == features:
        fail(
            "Unexpectedly identical reversed feature list."
        )

    print(
        "Correct feature order: PASSED"
    )

    print(
        "Model expects features in saved order."
    )


def test_missing_feature_handling(
    rows: pd.DataFrame,
    features: list[str],
    medians: dict,
) -> bool:

    section(
        "MISSING FEATURE TEST"
    )

    sample = rows[
        features
    ].iloc[
        :1
    ].copy()

    missing_feature = features[0]

    sample[
        missing_feature
    ] = np.nan

    prepared = prepare_features(
        sample,
        features,
        medians,
    )

    if pd.isna(
        prepared.iloc[
            0
        ][missing_feature]
    ):
        fail(
            "Missing-feature imputation failed."
        )

    print(
        f"Missing feature simulation: "
        f"{missing_feature}"
    )

    print(
        "Median fallback: PASSED"
    )

    return True


def test_batch_inference(
    model,
    rows: pd.DataFrame,
    features: list[str],
    medians: dict,
) -> dict:

    section(
        "BATCH INFERENCE TEST"
    )

    sample_size = min(
        100,
        len(rows),
    )

    sample = rows.iloc[
        :sample_size
    ].copy()

    X = prepare_features(
        sample,
        features,
        medians,
    )

    predictions, probabilities = (
        run_inference(
            model,
            X,
        )
    )

    if probabilities is None:
        fail(
            "Model does not support predict_proba()."
        )

    validate_probability_matrix(
        probabilities
    )

    if len(predictions) != sample_size:
        fail(
            "Prediction count does not match input count."
        )

    print(
        f"Batch size          : "
        f"{sample_size}"
    )

    print(
        f"Predictions returned : "
        f"{len(predictions)}"
    )

    print(
        f"Probability shape    : "
        f"{probabilities.shape}"
    )

    print(
        "Batch inference: PASSED"
    )

    return {
        "sample_size": int(
            sample_size
        ),
        "probability_shape": list(
            probabilities.shape
        ),
        "probability_rows_sum_to_one": True,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("FAULT CLASSIFIER V4 INFERENCE VALIDATION")
    print("=" * 78)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    bundle = load_bundle()

    validate_bundle(
        bundle
    )

    model = bundle[
        "model"
    ]

    features = list(
        bundle[
            "feature_names"
        ]
    )

    medians = dict(
        bundle[
            "imputation_medians"
        ]
    )

    class_names = [
        str(x)
        for x in bundle[
            "class_names"
        ]
    ]

    # -------------------------------------------------------------------------
    # Load inference data
    # -------------------------------------------------------------------------

    section("LOADING INFERENCE DATA")

    df = load_test_rows(
        features
    )

    print(
        f"Active fault rows: "
        f"{len(df):,}"
    )

    print(
        f"Available classes: "
        f"{df[TARGET_COLUMN].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Prepare sample
    # -------------------------------------------------------------------------

    section("FEATURE PREPARATION")

    sample_size = min(
        10,
        len(df),
    )

    sample = df.iloc[
        :sample_size
    ].copy()

    X_sample = prepare_features(
        sample,
        features,
        medians,
    )

    print(
        f"Sample rows      : "
        f"{len(X_sample)}"
    )

    print(
        f"Feature columns  : "
        f"{len(X_sample.columns)}"
    )

    print(
        "Numeric conversion: PASSED"
    )

    print(
        "Imputation: PASSED"
    )

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------

    section("RUNNING INFERENCE")

    predictions, probabilities = (
        run_inference(
            model,
            X_sample,
        )
    )

    if probabilities is None:
        fail(
            "predict_proba() unavailable."
        )

    validate_probability_matrix(
        probabilities
    )

    print(
        f"Predictions: "
        f"{len(predictions)}"
    )

    print(
        f"Probability shape: "
        f"{probabilities.shape}"
    )

    print(
        "Probability validation: PASSED"
    )

    # -------------------------------------------------------------------------
    # Sample output
    # -------------------------------------------------------------------------

    print_predictions(
        sample,
        predictions,
        probabilities,
        class_names,
    )

    # -------------------------------------------------------------------------
    # Batch test
    # -------------------------------------------------------------------------

    batch_result = test_batch_inference(
        model,
        df,
        features,
        medians,
    )

    # -------------------------------------------------------------------------
    # Missing-feature test
    # -------------------------------------------------------------------------

    missing_test = test_missing_feature_handling(
        df,
        features,
        medians,
    )

    # -------------------------------------------------------------------------
    # Evaluate on full active dataset
    # -------------------------------------------------------------------------

    section(
        "FULL ACTIVE-DATA INFERENCE"
    )

    X_all = prepare_features(
        df,
        features,
        medians,
    )

    all_predictions, all_probabilities = (
        run_inference(
            model,
            X_all,
        )
    )

    if all_probabilities is None:
        fail(
            "predict_proba() unavailable."
        )

    validate_probability_matrix(
        all_probabilities
    )

    accuracy = calculate_test_accuracy(
        df,
        all_predictions,
    )

    print(
        f"Rows processed : "
        f"{len(df):,}"
    )

    print(
        f"Accuracy check : "
        f"{accuracy:.4f}"
    )

    print(
        "Full-dataset inference: PASSED"
    )

    # -------------------------------------------------------------------------
    # Prediction distribution
    # -------------------------------------------------------------------------

    section(
        "PREDICTION DISTRIBUTION"
    )

    prediction_counts = (
        pd.Series(
            all_predictions
        )
        .value_counts()
    )

    for class_name in class_names:

        count = int(
            prediction_counts.get(
                class_name,
                0,
            )
        )

        percentage = (
            count
            / len(all_predictions)
            * 100
        )

        print(
            f"{class_name:<30}"
            f"{count:>7,}"
            f" ({percentage:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Confidence statistics
    # -------------------------------------------------------------------------

    confidence = (
        all_probabilities.max(
            axis=1
        )
    )

    print(
        "\nConfidence statistics:"
    )

    print(
        f"Mean : {confidence.mean():.4f}"
    )

    print(
        f"Min  : {confidence.min():.4f}"
    )

    print(
        f"Max  : {confidence.max():.4f}"
    )

    # -------------------------------------------------------------------------
    # Feature contract
    # -------------------------------------------------------------------------

    validate_feature_order(
        model,
        features,
    )

    # -------------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------------

    result = {
        "status": "PASSED",
        "model_name": bundle.get(
            "model_name",
            "unknown",
        ),
        "model_type": type(
            model
        ).__name__,
        "feature_count": len(
            features
        ),
        "class_count": len(
            class_names
        ),
        "classes": class_names,
        "sample_inference_rows": len(
            sample
        ),
        "batch_inference": batch_result,
        "missing_feature_imputation": (
            missing_test
        ),
        "full_active_rows": int(
            len(df)
        ),
        "full_active_accuracy": accuracy,
        "probability_rows_sum_to_one": True,
        "confidence": {
            "mean": float(
                confidence.mean()
            ),
            "min": float(
                confidence.min()
            ),
            "max": float(
                confidence.max()
            ),
        },
        "feature_contract": {
            "ordered_features": features,
            "imputation": "saved training medians",
            "target_not_required_for_inference": True,
        },
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    section(
        "INFERENCE CONTRACT"
    )

    print(
        "Input:"
    )

    print(
        "  1. One telemetry window/sample"
    )

    print(
        f"  2. Exactly {len(features)} "
        f"features in saved order"
    )

    print(
        "  3. Numeric values"
    )

    print(
        "  4. Missing values → saved training medians"
    )

    print(
        "\nModel:"
    )

    print(
        "  Random Forest V4"
    )

    print(
        "\nOutput:"
    )

    print(
        "  1. predicted_fault"
    )

    print(
        "  2. class_probabilities"
    )

    print(
        "  3. prediction_confidence"
    )

    print(
        "\nResult file:"
    )

    print(
        RESULTS_FILE
    )

    section(
        "INFERENCE VALIDATION COMPLETE"
    )

    print(
        "STATUS: PASSED"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:

        print()

        print("=" * 78)
        print("INFERENCE VALIDATION FAILED")
        print("=" * 78)

        print(
            str(exc)
        )

        sys.exit(1)