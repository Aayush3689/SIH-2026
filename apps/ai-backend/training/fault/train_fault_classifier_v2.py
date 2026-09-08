from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# PATHS
# =============================================================================

TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent

FEATURES_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_features.csv"
)

FEATURE_LIST_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v2_features.txt"
)

SPLIT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_flight_split_final.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier_v2"
)

MODEL_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v2.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v2_metadata.json"
)

RESULTS_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v2_results.json"
)

CONFUSION_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v2_confusion_matrix.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

TARGET_COLUMN = "fault_mode_ground_truth"
ACTIVE_COLUMN = "fault_active_label"
FLIGHT_COLUMN = "flight_id"

TRAIN_SPLIT = "TRAIN"
VALIDATION_SPLIT = "VALIDATION"
TEST_SPLIT = "TEST"

RANDOM_STATE = 42


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


def load_features() -> list[str]:

    if not FEATURE_LIST_FILE.exists():
        fail(
            f"V2 feature file not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    features = [
        line.strip()
        for line in FEATURE_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != 27:
        fail(
            f"Expected 27 V2 features, found {len(features)}."
        )

    return features


def load_split() -> pd.DataFrame:

    if not SPLIT_FILE.exists():
        fail(
            f"Split file not found:\n"
            f"{SPLIT_FILE}"
        )

    split = pd.read_csv(SPLIT_FILE)

    split_column = None

    for column in split.columns:
        if column.lower() in {
            "split",
            "dataset_split",
            "set",
        }:
            split_column = column
            break

    if split_column is None:
        fail(
            "Cannot identify split column."
        )

    split = split.rename(
        columns={
            split_column: "dataset_split"
        }
    )

    split[FLIGHT_COLUMN] = (
        split[FLIGHT_COLUMN]
        .astype(str)
        .str.strip()
    )

    split["dataset_split"] = (
        split["dataset_split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return split[
        [
            FLIGHT_COLUMN,
            "dataset_split",
        ]
    ].drop_duplicates()


def load_data(
    features: list[str],
) -> pd.DataFrame:

    if not FEATURES_FILE.exists():
        fail(
            f"Feature file not found:\n"
            f"{FEATURES_FILE}"
        )

    df = pd.read_csv(FEATURES_FILE)

    required = {
        TARGET_COLUMN,
        ACTIVE_COLUMN,
        FLIGHT_COLUMN,
        *features,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        fail(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # Active faults only.
    df = df[
        df[ACTIVE_COLUMN] == 1
    ].copy()

    df[FLIGHT_COLUMN] = (
        df[FLIGHT_COLUMN]
        .astype(str)
        .str.strip()
    )

    split = load_split()

    df = df.merge(
        split,
        on=FLIGHT_COLUMN,
        how="left",
        validate="many_to_one",
    )

    if df["dataset_split"].isna().any():
        fail(
            "Some active-fault rows are missing dataset split."
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    return df


def validate_split(df: pd.DataFrame) -> None:

    section("SPLIT VALIDATION")

    train_flights = set(
        df.loc[
            df["dataset_split"] == TRAIN_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    val_flights = set(
        df.loc[
            df["dataset_split"] == VALIDATION_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    test_flights = set(
        df.loc[
            df["dataset_split"] == TEST_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    print(
        f"TRAIN flights      : {len(train_flights)}"
    )

    print(
        f"VALIDATION flights : {len(val_flights)}"
    )

    print(
        f"TEST flights       : {len(test_flights)}"
    )

    if train_flights & val_flights:
        fail("TRAIN/VALIDATION flight leakage.")

    if train_flights & test_flights:
        fail("TRAIN/TEST flight leakage.")

    if val_flights & test_flights:
        fail("VALIDATION/TEST flight leakage.")

    print("\nFlight leakage check: PASSED")


def build_xy(
    df: pd.DataFrame,
    features: list[str],
):
    X = df[features].copy()
    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .copy()
    )

    if np.isinf(
        X.to_numpy(dtype=float)
    ).any():
        fail(
            "Infinite values found in V2 feature matrix."
        )

    return X, y


def fit_imputation(
    X_train: pd.DataFrame,
) -> pd.Series:

    medians = X_train.median()

    missing = [
        feature
        for feature in X_train.columns
        if pd.isna(medians[feature])
    ]

    if missing:
        fail(
            "Cannot calculate training medians for:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    return medians


def apply_imputation(
    X: pd.DataFrame,
    medians: pd.Series,
) -> pd.DataFrame:

    result = X.copy()

    for feature in result.columns:
        result[feature] = (
            result[feature]
            .fillna(medians[feature])
        )

    return result


def build_model() -> RandomForestClassifier:

    return RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(
    model,
    X,
    y,
    labels,
) -> dict:

    pred = model.predict(X)

    metrics = {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "macro_precision": float(
            precision_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "predictions": pred.tolist(),
        "classification_report": classification_report(
            y,
            pred,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=labels,
        ).tolist(),
    }

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("FAULT CLASSIFIER V2")
    print("HEALTH-FOCUSED FEATURE MODEL")
    print("=" * 78)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    section("LOADING")

    features = load_features()

    print(
        f"V2 feature count: {len(features)}"
    )

    df = load_data(features)

    print(
        f"Active fault rows: {len(df):,}"
    )

    print(
        f"Active fault flights: "
        f"{df[FLIGHT_COLUMN].nunique()}"
    )

    validate_split(df)

    # -------------------------------------------------------------------------
    # Target classes
    # -------------------------------------------------------------------------

    labels = sorted(
        df[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    print(
        f"\nFault classes: {len(labels)}"
    )

    for label in labels:
        count = int(
            (
                df[TARGET_COLUMN] == label
            ).sum()
        )

        print(
            f"  {label:<30} {count:>6,}"
        )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    train = df[
        df["dataset_split"] == TRAIN_SPLIT
    ].copy()

    val = df[
        df["dataset_split"] == VALIDATION_SPLIT
    ].copy()

    test = df[
        df["dataset_split"] == TEST_SPLIT
    ].copy()

    X_train_raw, y_train = build_xy(
        train,
        features,
    )

    X_val_raw, y_val = build_xy(
        val,
        features,
    )

    X_test_raw, y_test = build_xy(
        test,
        features,
    )

    # -------------------------------------------------------------------------
    # TRAIN-only imputation
    # -------------------------------------------------------------------------

    section("IMPUTATION")

    train_medians = fit_imputation(
        X_train_raw
    )

    X_train = apply_imputation(
        X_train_raw,
        train_medians,
    )

    X_val = apply_imputation(
        X_val_raw,
        train_medians,
    )

    X_test = apply_imputation(
        X_test_raw,
        train_medians,
    )

    print(
        "Strategy: TRAINING MEDIAN"
    )

    # -------------------------------------------------------------------------
    # Validation model
    # -------------------------------------------------------------------------

    section("VALIDATION")

    validation_model = build_model()

    validation_model.fit(
        X_train,
        y_train,
    )

    validation_metrics = evaluate(
        validation_model,
        X_val,
        y_val,
        labels,
    )

    print(
        f"Validation Accuracy : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1 : "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Validation Macro Rec: "
        f"{validation_metrics['macro_recall']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Final train + validation
    # -------------------------------------------------------------------------

    section("FINAL REFIT")

    X_train_val_raw = pd.concat(
        [
            X_train_raw,
            X_val_raw,
        ],
        ignore_index=True,
    )

    y_train_val = pd.concat(
        [
            y_train,
            y_val,
        ],
        ignore_index=True,
    )

    final_medians = fit_imputation(
        X_train_val_raw
    )

    X_train_val = apply_imputation(
        X_train_val_raw,
        final_medians,
    )

    X_test_final = apply_imputation(
        X_test_raw,
        final_medians,
    )

    final_model = build_model()

    final_model.fit(
        X_train_val,
        y_train_val,
    )

    # -------------------------------------------------------------------------
    # FINAL TEST
    # -------------------------------------------------------------------------

    section("FINAL TEST")

    test_metrics = evaluate(
        final_model,
        X_test_final,
        y_test,
        labels,
    )

    print(
        f"Accuracy        : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision : "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    section("CLASSIFICATION REPORT")

    report_df = pd.DataFrame(
        test_metrics["classification_report"]
    ).transpose()

    print(
        report_df.to_string()
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    section("CONFUSION MATRIX")

    cm = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=labels,
        columns=labels,
    )

    print(
        cm.to_string()
    )

    cm.to_csv(
        CONFUSION_FILE
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    section("FEATURE IMPORTANCE")

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": final_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format
            },
        )
    )

    importance.to_csv(
        MODEL_DIR / "feature_importance_v2.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save model bundle
    # -------------------------------------------------------------------------

    bundle = {
        "model": final_model,
        "model_name": "random_forest_v2_health_focused",
        "feature_names": features,
        "imputation_medians": final_medians.to_dict(),
        "class_names": labels,
        "target_column": TARGET_COLUMN,
        "active_filter": f"{ACTIVE_COLUMN} == 1",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        bundle,
        MODEL_FILE,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "model_name": "random_forest_v2_health_focused",
        "task": "multi_class_fault_classification",
        "feature_count": len(features),
        "feature_names": features,
        "class_names": labels,
        "target_column": TARGET_COLUMN,
        "active_filter": f"{ACTIVE_COLUMN} == 1",
        "split_strategy": (
            "Flight-wise chronological split within each fault class."
        ),
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "train_flights": int(
            train[FLIGHT_COLUMN].nunique()
        ),
        "validation_flights": int(
            val[FLIGHT_COLUMN].nunique()
        ),
        "test_flights": int(
            test[FLIGHT_COLUMN].nunique()
        ),
        "validation_metrics": {
            "accuracy": validation_metrics[
                "accuracy"
            ],
            "macro_precision": validation_metrics[
                "macro_precision"
            ],
            "macro_recall": validation_metrics[
                "macro_recall"
            ],
            "macro_f1": validation_metrics[
                "macro_f1"
            ],
            "weighted_f1": validation_metrics[
                "weighted_f1"
            ],
        },
        "test_metrics": {
            "accuracy": test_metrics[
                "accuracy"
            ],
            "macro_precision": test_metrics[
                "macro_precision"
            ],
            "macro_recall": test_metrics[
                "macro_recall"
            ],
            "macro_f1": test_metrics[
                "macro_f1"
            ],
            "weighted_f1": test_metrics[
                "weighted_f1"
            ],
        },
        "design_change": (
            "Removed operating-context features and direct expected-value "
            "features. Retained health-oriented sensor signals, rolling "
            "statistics, rates and physics residuals."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Result JSON
    # -------------------------------------------------------------------------

    results = {
        "validation": metadata[
            "validation_metrics"
        ],
        "test": metadata[
            "test_metrics"
        ],
        "classification_report": (
            test_metrics[
                "classification_report"
            ]
        ),
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    section("TRAINING COMPLETE")

    print(
        f"Model saved to:\n{MODEL_FILE}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{METADATA_FILE}"
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_FILE}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("FAULT CLASSIFIER V2 FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)