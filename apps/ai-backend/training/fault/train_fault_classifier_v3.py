from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

FEATURE_DATA_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v3_features.csv"
)

FEATURE_LIST_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v3_features.txt"
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
    / "fault_classifier_v3"
)

MODEL_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v3.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v3_metadata.json"
)

RESULTS_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v3_results.json"
)

CONFUSION_FILE = (
    MODEL_DIR
    / "final_fault_classifier_v3_confusion_matrix.csv"
)

IMPORTANCE_FILE = (
    MODEL_DIR
    / "feature_importance_v3.csv"
)

PERMUTATION_FILE = (
    MODEL_DIR
    / "permutation_importance_v3_test.csv"
)

EXPECTED_V3_FEATURE_COUNT = 53


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
            f"V3 feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    features = [
        line.strip()
        for line in FEATURE_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != EXPECTED_V3_FEATURE_COUNT:
        fail(
            f"Expected "
            f"{EXPECTED_V3_FEATURE_COUNT} V3 features, "
            f"found {len(features)}."
        )

    return features


def load_split() -> pd.DataFrame:

    if not SPLIT_FILE.exists():
        fail(
            f"Split file not found:\n"
            f"{SPLIT_FILE}"
        )

    split = pd.read_csv(
        SPLIT_FILE
    )

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
            "Could not identify split column."
        )

    split = split.rename(
        columns={
            split_column: "dataset_split"
        }
    ).copy()

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

    if not FEATURE_DATA_FILE.exists():
        fail(
            f"V3 feature dataset not found:\n"
            f"{FEATURE_DATA_FILE}"
        )

    df = pd.read_csv(
        FEATURE_DATA_FILE
    )

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
            "Missing columns in V3 feature dataset:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # Only active fault population.
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
            "Some active fault rows have no dataset split."
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    return df


def make_xy(
    frame: pd.DataFrame,
    features: list[str],
):
    X = frame[
        features
    ].copy()

    y = (
        frame[TARGET_COLUMN]
        .astype(str)
        .copy()
    )

    if np.isinf(
        X.to_numpy(
            dtype=float
        )
    ).any():
        fail(
            "Infinite values detected."
        )

    return X, y


def fit_medians(
    X: pd.DataFrame,
) -> pd.Series:

    medians = X.median()

    invalid = [
        feature
        for feature in X.columns
        if pd.isna(
            medians[feature]
        )
    ]

    if invalid:
        fail(
            "Cannot calculate medians for:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in invalid
            )
        )

    return medians


def apply_medians(
    X: pd.DataFrame,
    medians: pd.Series,
) -> pd.DataFrame:

    result = X.copy()

    for feature in result.columns:
        result[feature] = (
            result[feature]
            .fillna(
                medians[feature]
            )
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

    predictions = model.predict(
        X
    )

    return {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                y,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": (
            classification_report(
                y,
                predictions,
                labels=labels,
                target_names=labels,
                output_dict=True,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y,
                predictions,
                labels=labels,
            ).tolist()
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("FAULT CLASSIFIER V3")
    print("PHYSICS / RELATIONSHIP-BASED MODEL")
    print("=" * 78)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    section("LOADING")

    features = load_features()

    print(
        f"Feature count: {len(features)}"
    )

    df = load_data(
        features
    )

    print(
        f"Active fault rows: {len(df):,}"
    )

    print(
        f"Active fault flights: "
        f"{df[FLIGHT_COLUMN].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Flight split validation
    # -------------------------------------------------------------------------

    train_flights = set(
        df.loc[
            df["dataset_split"]
            == TRAIN_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    validation_flights = set(
        df.loc[
            df["dataset_split"]
            == VALIDATION_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    test_flights = set(
        df.loc[
            df["dataset_split"]
            == TEST_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    if train_flights & validation_flights:
        fail(
            "TRAIN/VALIDATION flight leakage."
        )

    if train_flights & test_flights:
        fail(
            "TRAIN/TEST flight leakage."
        )

    if validation_flights & test_flights:
        fail(
            "VALIDATION/TEST flight leakage."
        )

    print(
        "\nFlight leakage check: PASSED"
    )

    # -------------------------------------------------------------------------
    # Target labels
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
                df[TARGET_COLUMN]
                == label
            ).sum()
        )

        print(
            f"  {label:<30} "
            f"{count:>6,}"
        )

    # -------------------------------------------------------------------------
    # Dataset split
    # -------------------------------------------------------------------------

    train = df[
        df["dataset_split"]
        == TRAIN_SPLIT
    ].copy()

    validation = df[
        df["dataset_split"]
        == VALIDATION_SPLIT
    ].copy()

    test = df[
        df["dataset_split"]
        == TEST_SPLIT
    ].copy()

    X_train_raw, y_train = make_xy(
        train,
        features,
    )

    X_val_raw, y_val = make_xy(
        validation,
        features,
    )

    X_test_raw, y_test = make_xy(
        test,
        features,
    )

    # -------------------------------------------------------------------------
    # Validation imputation
    # -------------------------------------------------------------------------

    section("VALIDATION IMPUTATION")

    train_medians = fit_medians(
        X_train_raw
    )

    X_train = apply_medians(
        X_train_raw,
        train_medians,
    )

    X_val = apply_medians(
        X_val_raw,
        train_medians,
    )

    print(
        "Imputation strategy: "
        "TRAINING MEDIAN"
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
    # Final refit
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

    final_medians = fit_medians(
        X_train_val_raw
    )

    X_train_val = apply_medians(
        X_train_val_raw,
        final_medians,
    )

    X_test = apply_medians(
        X_test_raw,
        final_medians,
    )

    final_model = build_model()

    final_model.fit(
        X_train_val,
        y_train_val,
    )

    # -------------------------------------------------------------------------
    # Final test
    # -------------------------------------------------------------------------

    section("FINAL TEST")

    test_metrics = evaluate(
        final_model,
        X_test,
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
        test_metrics[
            "classification_report"
        ]
    ).transpose()

    print(
        report_df.to_string()
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    section("CONFUSION MATRIX")

    confusion = pd.DataFrame(
        test_metrics[
            "confusion_matrix"
        ],
        index=labels,
        columns=labels,
    )

    print(
        confusion.to_string()
    )

    confusion.to_csv(
        CONFUSION_FILE
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    section("FEATURE IMPORTANCE")

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                final_model
                .feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False,
            formatters={
                "importance":
                    "{:.6f}".format
            },
        )
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Permutation importance
    # -------------------------------------------------------------------------

    section(
        "PERMUTATION IMPORTANCE"
    )

    permutation = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": (
                permutation.importances_mean
            ),
            "importance_std": (
                permutation.importances_std
            ),
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    print(
        permutation_df.head(
            20
        ).to_string(
            index=False,
            formatters={
                "importance_mean":
                    "{:.6f}".format,
                "importance_std":
                    "{:.6f}".format,
            },
        )
    )

    permutation_df.to_csv(
        PERMUTATION_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save model bundle
    # -------------------------------------------------------------------------

    section("SAVING")

    model_bundle = {
        "model": final_model,
        "model_name": (
            "random_forest_v3_physics_relationship"
        ),
        "feature_names": features,
        "imputation_medians": (
            final_medians.to_dict()
        ),
        "class_names": labels,
        "target_column": TARGET_COLUMN,
        "active_filter": (
            f"{ACTIVE_COLUMN} == 1"
        ),
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "model_name": (
            "random_forest_v3_physics_relationship"
        ),
        "task": (
            "multi_class_fault_classification"
        ),
        "feature_count": len(features),
        "feature_names": features,
        "class_names": labels,
        "target_column": TARGET_COLUMN,
        "active_filter": (
            f"{ACTIVE_COLUMN} == 1"
        ),
        "split_strategy": (
            "Flight-wise chronological split "
            "within each fault class."
        ),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_flights": int(len(train_flights)),
        "validation_flights": int(
            len(validation_flights)
        ),
        "test_flights": int(
            len(test_flights)
        ),
        "validation_metrics": {
            "accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),
            "macro_precision": (
                validation_metrics[
                    "macro_precision"
                ]
            ),
            "macro_recall": (
                validation_metrics[
                    "macro_recall"
                ]
            ),
            "macro_f1": (
                validation_metrics[
                    "macro_f1"
                ]
            ),
            "weighted_f1": (
                validation_metrics[
                    "weighted_f1"
                ]
            ),
        },
        "test_metrics": {
            "accuracy": (
                test_metrics[
                    "accuracy"
                ]
            ),
            "macro_precision": (
                test_metrics[
                    "macro_precision"
                ]
            ),
            "macro_recall": (
                test_metrics[
                    "macro_recall"
                ]
            ),
            "macro_f1": (
                test_metrics[
                    "macro_f1"
                ]
            ),
            "weighted_f1": (
                test_metrics[
                    "weighted_f1"
                ]
            ),
        },
        "design_change": (
            "V3 retains the V2 health-focused "
            "sensor features and adds derived "
            "physics/relationship features such "
            "as thermal ratios, residual ratios, "
            "pressure relationships, vibration "
            "relationships, variability ratios "
            "and cross-system residual features."
        ),
        "dataset_limitation": (
            "Each fault class is associated with "
            "one engine in the current synthetic "
            "dataset. Therefore the evaluation "
            "measures later-flight generalization "
            "within known engine/fault scenarios "
            "rather than unseen-engine fault-class "
            "generalization."
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    results = {
        "validation_metrics": (
            metadata[
                "validation_metrics"
            ]
        ),
        "test_metrics": (
            metadata[
                "test_metrics"
            ]
        ),
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
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print(
        f"\nModel saved to:\n"
        f"{MODEL_FILE}"
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

    print(
        f"\nFeature importance saved to:\n"
        f"{IMPORTANCE_FILE}"
    )

    print(
        f"\nPermutation importance saved to:\n"
        f"{PERMUTATION_FILE}"
    )

    section("V3 TRAINING COMPLETE")

    print(
        f"Validation Macro F1: "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Accuracy       : "
        f"{test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("V3 TRAINING FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)