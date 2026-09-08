from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
    / "fault_classifier_final_features.txt"
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
    / "fault_classifier"
)

MODEL_FILE = MODEL_DIR / "final_fault_classifier.joblib"
METADATA_FILE = MODEL_DIR / "final_fault_classifier_metadata.json"
RESULTS_FILE = MODEL_DIR / "final_fault_classifier_results.json"
CONFUSION_FILE = MODEL_DIR / "final_fault_classifier_confusion_matrix.csv"


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


def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_feature_list() -> list[str]:
    if not FEATURE_LIST_FILE.exists():
        fail(
            f"Feature list not found:\n"
            f"{FEATURE_LIST_FILE}"
        )

    features = [
        line.strip()
        for line in FEATURE_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not features:
        fail("Feature list is empty.")

    return features


def load_data() -> tuple[pd.DataFrame, list[str]]:
    if not FEATURES_FILE.exists():
        fail(
            f"Feature CSV not found:\n"
            f"{FEATURES_FILE}"
        )

    if not SPLIT_FILE.exists():
        fail(
            f"Flight split file not found:\n"
            f"{SPLIT_FILE}"
        )

    df = pd.read_csv(FEATURES_FILE)
    split_df = pd.read_csv(SPLIT_FILE)

    feature_names = load_feature_list()

    return df, split_df, feature_names


def validate_columns(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    feature_names: list[str],
) -> None:

    required_df_columns = {
        TARGET_COLUMN,
        ACTIVE_COLUMN,
        FLIGHT_COLUMN,
    }

    missing_df = sorted(
        required_df_columns - set(df.columns)
    )

    if missing_df:
        fail(
            "Required columns missing from fault_features.csv:\n"
            + "\n".join(f"  - {col}" for col in missing_df)
        )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        fail(
            "Final feature list contains columns missing from "
            "fault_features.csv:\n"
            + "\n".join(f"  - {col}" for col in missing_features)
        )

    if FLIGHT_COLUMN not in split_df.columns:
        fail(
            f"Split file does not contain '{FLIGHT_COLUMN}'.\n"
            f"Columns found: {list(split_df.columns)}"
        )

    split_column_candidates = [
        column
        for column in split_df.columns
        if column.lower() in {
            "split",
            "dataset_split",
            "set",
        }
    ]

    if not split_column_candidates:
        fail(
            "Could not identify split column in flight split file.\n"
            f"Columns found: {list(split_df.columns)}"
        )

    split_column = split_column_candidates[0]

    split_df.rename(
        columns={split_column: "dataset_split"},
        inplace=True,
    )

    # Store the normalized split name for downstream use.
    split_df["dataset_split"] = (
        split_df["dataset_split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed_splits = {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
    }

    unexpected = sorted(
        set(split_df["dataset_split"]) - allowed_splits
    )

    if unexpected:
        fail(
            "Unexpected split values found:\n"
            + str(unexpected)
        )

    return split_df


def load_and_prepare(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    split_df = split_df.copy()

    # -------------------------------------------------------------------------
    # Keep only active-fault population
    # -------------------------------------------------------------------------
    active_mask = df[ACTIVE_COLUMN] == 1

    active_df = df.loc[active_mask].copy()

    if active_df.empty:
        fail("No active-fault rows found.")

    # -------------------------------------------------------------------------
    # Normalize flight IDs to string for safe joining
    # -------------------------------------------------------------------------
    active_df[FLIGHT_COLUMN] = (
        active_df[FLIGHT_COLUMN]
        .astype(str)
        .str.strip()
    )

    split_df[FLIGHT_COLUMN] = (
        split_df[FLIGHT_COLUMN]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # Validate flight coverage
    # -------------------------------------------------------------------------
    active_flights = set(
        active_df[FLIGHT_COLUMN].unique()
    )

    split_flights = set(
        split_df[FLIGHT_COLUMN].unique()
    )

    missing_split_flights = sorted(
        active_flights - split_flights
    )

    if missing_split_flights:
        fail(
            "Some active-fault flights are missing from the split file:\n"
            + "\n".join(
                f"  - {flight}"
                for flight in missing_split_flights
            )
        )

    unexpected_split_flights = sorted(
        split_flights - active_flights
    )

    if unexpected_split_flights:
        print(
            "\nWARNING: Split file contains flights with no active-fault "
            "rows. They will not be used:"
        )

        for flight in unexpected_split_flights:
            print(f"  - {flight}")

    # -------------------------------------------------------------------------
    # Join flight split into telemetry rows
    # -------------------------------------------------------------------------
    split_lookup = split_df[
        [FLIGHT_COLUMN, "dataset_split"]
    ].drop_duplicates()

    if split_lookup[FLIGHT_COLUMN].duplicated().any():
        duplicated_flights = (
            split_lookup.loc[
                split_lookup[FLIGHT_COLUMN].duplicated(keep=False),
                FLIGHT_COLUMN,
            ]
            .unique()
            .tolist()
        )

        fail(
            "A flight appears multiple times in the split file:\n"
            + str(duplicated_flights)
        )

    merged = active_df.merge(
        split_lookup,
        on=FLIGHT_COLUMN,
        how="left",
        validate="many_to_one",
    )

    if merged["dataset_split"].isna().any():
        fail(
            "Some active-fault rows did not receive a dataset split."
        )

    return merged, split_df


def validate_target(merged: pd.DataFrame) -> None:
    if merged[TARGET_COLUMN].isna().any():
        fail(
            f"Missing target values found in '{TARGET_COLUMN}'."
        )

    target_values = sorted(
        merged[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    if len(target_values) != 8:
        print(
            f"\nWARNING: Expected 8 fault classes, found {len(target_values)}."
        )

    print("\nFault classes:")
    for target in target_values:
        count = int(
            (merged[TARGET_COLUMN].astype(str) == target).sum()
        )
        print(f"  {target:<30} {count:>6,}")


def validate_splits(merged: pd.DataFrame) -> None:
    print_section("SPLIT SUMMARY")

    split_counts = (
        merged.groupby("dataset_split")
        .size()
        .reindex(
            [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
            fill_value=0,
        )
    )

    for split_name, count in split_counts.items():
        print(f"{split_name:<15} {int(count):>8,} rows")

    # -------------------------------------------------------------------------
    # Flight counts
    # -------------------------------------------------------------------------
    print()

    flight_counts = (
        merged.groupby("dataset_split")[FLIGHT_COLUMN]
        .nunique()
        .reindex(
            [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT],
            fill_value=0,
        )
    )

    for split_name, count in flight_counts.items():
        print(f"{split_name:<15} {int(count):>8,} flights")

    # -------------------------------------------------------------------------
    # Flight leakage checks
    # -------------------------------------------------------------------------
    train_flights = set(
        merged.loc[
            merged["dataset_split"] == TRAIN_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    val_flights = set(
        merged.loc[
            merged["dataset_split"] == VALIDATION_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    test_flights = set(
        merged.loc[
            merged["dataset_split"] == TEST_SPLIT,
            FLIGHT_COLUMN,
        ]
    )

    if train_flights & val_flights:
        fail(
            "Flight leakage detected between TRAIN and VALIDATION."
        )

    if train_flights & test_flights:
        fail(
            "Flight leakage detected between TRAIN and TEST."
        )

    if val_flights & test_flights:
        fail(
            "Flight leakage detected between VALIDATION and TEST."
        )

    print("\nFlight leakage check: PASSED")

    # -------------------------------------------------------------------------
    # Class coverage
    # -------------------------------------------------------------------------
    print_section("CLASS COVERAGE BY SPLIT")

    coverage = pd.crosstab(
        merged["dataset_split"],
        merged[TARGET_COLUMN],
    )

    coverage = coverage.reindex(
        index=[
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            TEST_SPLIT,
        ],
        fill_value=0,
    )

    print(coverage.to_string())

    for split_name in [
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
    ]:
        split_targets = set(
            merged.loc[
                merged["dataset_split"] == split_name,
                TARGET_COLUMN,
            ]
            .astype(str)
        )

        all_targets = set(
            merged[TARGET_COLUMN].astype(str).unique()
        )

        missing = sorted(
            all_targets - split_targets
        )

        if missing:
            fail(
                f"{split_name} is missing fault classes:\n"
                + "\n".join(
                    f"  - {fault}"
                    for fault in missing
                )
            )

    print("\nClass coverage check: PASSED")


def make_xy(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[feature_names].copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    # Force numeric conversion.
    for feature in feature_names:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    # Reject infinite values.
    numeric_values = X.to_numpy(dtype=float)

    if np.isinf(numeric_values).any():
        fail("Infinite values detected in classifier features.")

    return X, y


def fit_medians(X_train: pd.DataFrame) -> pd.Series:
    medians = X_train.median(numeric_only=True)

    missing_medians = [
        feature
        for feature in X_train.columns
        if pd.isna(medians[feature])
    ]

    if missing_medians:
        fail(
            "Could not compute training median for:\n"
            + "\n".join(f"  - {feature}" for feature in missing_medians)
        )

    return medians


def apply_medians(
    X: pd.DataFrame,
    medians: pd.Series,
) -> pd.DataFrame:

    output = X.copy()

    for feature in output.columns:
        output[feature] = output[feature].fillna(
            medians[feature]
        )

    return output


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
) -> dict:

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    macro_precision = precision_score(
        y,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=labels,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions.tolist(),
    }


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_hist_gradient_boosting(
    class_names: list[str],
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def train_and_validate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
) -> tuple[dict, dict]:

    print_section("MODEL COMPARISON")

    models = {
        "random_forest": build_random_forest(),
        "hist_gradient_boosting": build_hist_gradient_boosting(
            labels
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            labels,
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics,
        }

        print(f"  Accuracy       : {metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
        print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Select by validation Macro F1
    # -------------------------------------------------------------------------
    best_name = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["macro_f1"],
            results[name]["metrics"]["macro_recall"],
            results[name]["metrics"]["accuracy"],
        ),
    )

    best_model = results[best_name]["model"]

    print_section("MODEL SELECTION")

    print(f"Selected model: {best_name}")
    print(
        f"Validation Macro F1: "
        f"{results[best_name]['metrics']['macro_f1']:.4f}"
    )

    return results, {
        "name": best_name,
        "model": best_model,
    }


def combine_train_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:

    X_combined = pd.concat(
        [X_train, X_val],
        axis=0,
        ignore_index=True,
    )

    y_combined = pd.concat(
        [y_train, y_val],
        axis=0,
        ignore_index=True,
    )

    return X_combined, y_combined


def build_final_model(model_name: str):
    if model_name == "random_forest":
        return build_random_forest()

    if model_name == "hist_gradient_boosting":
        return build_hist_gradient_boosting([])

    fail(f"Unknown selected model: {model_name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("FAULT CLASSIFICATION MODEL TRAINING")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print_section("LOADING DATA")

    df, split_df, feature_names = load_data()

    print(f"Feature dataset rows   : {len(df):,}")
    print(f"Feature dataset columns: {len(df.columns):,}")
    print(f"Final feature count    : {len(feature_names)}")

    # -------------------------------------------------------------------------
    # Validate feature count
    # -------------------------------------------------------------------------
    if len(feature_names) != 40:
        fail(
            f"Expected 40 final features, found {len(feature_names)}."
        )

    # -------------------------------------------------------------------------
    # Validate columns / split
    # -------------------------------------------------------------------------
    split_df = validate_columns(
        df,
        split_df,
        feature_names,
    )

    merged, split_df = load_and_prepare(
        df,
        split_df,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # Target validation
    # -------------------------------------------------------------------------
    print_section("ACTIVE FAULT POPULATION")

    print(f"Active fault rows: {len(merged):,}")
    print(
        f"Active fault flights: "
        f"{merged[FLIGHT_COLUMN].nunique():,}"
    )

    validate_target(merged)

    # -------------------------------------------------------------------------
    # Split validation
    # -------------------------------------------------------------------------
    validate_splits(merged)

    # -------------------------------------------------------------------------
    # Create train/validation/test datasets
    # -------------------------------------------------------------------------
    train_df = merged[
        merged["dataset_split"] == TRAIN_SPLIT
    ].copy()

    val_df = merged[
        merged["dataset_split"] == VALIDATION_SPLIT
    ].copy()

    test_df = merged[
        merged["dataset_split"] == TEST_SPLIT
    ].copy()

    if train_df.empty:
        fail("TRAIN split is empty.")

    if val_df.empty:
        fail("VALIDATION split is empty.")

    if test_df.empty:
        fail("TEST split is empty.")

    X_train_raw, y_train = make_xy(
        train_df,
        feature_names,
    )

    X_val_raw, y_val = make_xy(
        val_df,
        feature_names,
    )

    X_test_raw, y_test = make_xy(
        test_df,
        feature_names,
    )

    labels = sorted(
        merged[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    # -------------------------------------------------------------------------
    # Train-only imputation
    # -------------------------------------------------------------------------
    print_section("MISSING VALUE HANDLING")

    train_missing_before = int(
        X_train_raw.isna().sum().sum()
    )

    val_missing_before = int(
        X_val_raw.isna().sum().sum()
    )

    test_missing_before = int(
        X_test_raw.isna().sum().sum()
    )

    print(
        f"TRAIN missing values before imputation: "
        f"{train_missing_before:,}"
    )

    print(
        f"VALIDATION missing values before imputation: "
        f"{val_missing_before:,}"
    )

    print(
        f"TEST missing values before imputation: "
        f"{test_missing_before:,}"
    )

    train_medians = fit_medians(X_train_raw)

    X_train = apply_medians(
        X_train_raw,
        train_medians,
    )

    X_val = apply_medians(
        X_val_raw,
        train_medians,
    )

    X_test = apply_medians(
        X_test_raw,
        train_medians,
    )

    print("\nImputation strategy: TRAINING MEDIAN")
    print("Imputation leakage check: PASSED")

    # -------------------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------------------
    comparison_results, selected = train_and_validate(
        X_train,
        y_train,
        X_val,
        y_val,
        labels,
    )

    selected_model_name = selected["name"]

    # -------------------------------------------------------------------------
    # Refit selected model on TRAIN + VALIDATION
    # -------------------------------------------------------------------------
    print_section("FINAL MODEL REFIT")

    X_train_val, y_train_val = combine_train_validation(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # Recompute medians only from TRAIN + VALIDATION,
    # then apply those medians to TEST.
    final_medians = fit_medians(
        X_train_val
    )

    X_train_val_final = apply_medians(
        X_train_val,
        final_medians,
    )

    X_test_final = apply_medians(
        X_test_raw,
        final_medians,
    )

    final_model = build_final_model(
        selected_model_name
    )

    print(
        f"Final model: {selected_model_name}"
    )

    print(
        f"Training rows for final refit: "
        f"{len(X_train_val_final):,}"
    )

    final_model.fit(
        X_train_val_final,
        y_train_val,
    )

    # -------------------------------------------------------------------------
    # Final untouched test evaluation
    # -------------------------------------------------------------------------
    print_section("FINAL TEST EVALUATION")

    test_metrics = evaluate_model(
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
    print_section("TEST CLASSIFICATION REPORT")

    report_df = pd.DataFrame(
        test_metrics["classification_report"]
    ).transpose()

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(report_df)

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------
    print_section("TEST CONFUSION MATRIX")

    cm = np.array(
        test_metrics["confusion_matrix"]
    )

    confusion_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(confusion_df.to_string())

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------
    print_section("SAVING MODEL")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Important:
    # Save a bundle instead of only the sklearn estimator.
    #
    # This makes deployment easier:
    #
    # live telemetry
    #       ↓
    # feature engineering
    #       ↓
    # saved feature order
    #       ↓
    # saved medians
    #       ↓
    # model
    #       ↓
    # fault class + probabilities
    # -------------------------------------------------------------------------
    model_bundle = {
        "model": final_model,
        "model_name": selected_model_name,
        "feature_names": feature_names,
        "imputation_medians": final_medians.to_dict(),
        "target_column": TARGET_COLUMN,
        "active_column": ACTIVE_COLUMN,
        "class_names": labels,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE,
    )

    print(f"Model saved to:")
    print(MODEL_FILE)

    # -------------------------------------------------------------------------
    # Save confusion matrix
    # -------------------------------------------------------------------------
    confusion_df.to_csv(
        CONFUSION_FILE,
        index=True,
    )

    print(f"\nConfusion matrix saved to:")
    print(CONFUSION_FILE)

    # -------------------------------------------------------------------------
    # Build comparison summary
    # -------------------------------------------------------------------------
    validation_comparison = {}

    for model_name, result in comparison_results.items():
        metrics = result["metrics"]

        validation_comparison[model_name] = {
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    metadata = {
        "task": "fault_classification",
        "description": (
            "Multi-class fault classification for active-fault "
            "aero piston engine telemetry."
        ),
        "target_column": TARGET_COLUMN,
        "active_fault_filter": f"{ACTIVE_COLUMN} == 1",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "class_names": labels,
        "selected_model": selected_model_name,
        "random_state": RANDOM_STATE,
        "split_strategy": (
            "Flight-wise chronological split within each fault class."
        ),
        "split_file": str(SPLIT_FILE),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_validation_refit_rows": int(len(X_train_val_final)),
        "train_flights": int(
            train_df[FLIGHT_COLUMN].nunique()
        ),
        "validation_flights": int(
            val_df[FLIGHT_COLUMN].nunique()
        ),
        "test_flights": int(
            test_df[FLIGHT_COLUMN].nunique()
        ),
        "validation_model_comparison": validation_comparison,
        "final_test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "missing_values": {
            "train_before_imputation": train_missing_before,
            "validation_before_imputation": val_missing_before,
            "test_before_imputation": test_missing_before,
            "strategy": "median",
            "fit_scope_for_final_model": "train_plus_validation",
        },
        "known_dataset_limitation": (
            "Each fault class in the current synthetic dataset is "
            "associated with one specific engine. Therefore this "
            "flight-wise evaluation measures later-flight generalization "
            "within known engine/fault scenarios, not unseen-engine "
            "fault-class generalization."
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

    print(f"\nMetadata saved to:")
    print(METADATA_FILE)

    # -------------------------------------------------------------------------
    # Save result JSON
    # -------------------------------------------------------------------------
    results_payload = {
        "validation_comparison": validation_comparison,
        "selected_model": selected_model_name,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "classification_report": test_metrics[
            "classification_report"
        ],
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results_payload,
            f,
            indent=2,
        )

    print(f"\nResults saved to:")
    print(RESULTS_FILE)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print_section("TRAINING COMPLETE")

    print(f"Selected model : {selected_model_name}")
    print(
        f"Test Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test Macro F1   : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print("\nArtifacts:")
    print(f"  Model     : {MODEL_FILE}")
    print(f"  Metadata  : {METADATA_FILE}")
    print(f"  Results   : {RESULTS_FILE}")
    print(f"  Confusion : {CONFUSION_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("TRAINING FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)