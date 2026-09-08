from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score


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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier"
    / "final_fault_classifier.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier"
    / "diagnostics"
)


TARGET_COLUMN = "fault_mode_ground_truth"
ACTIVE_COLUMN = "fault_active_label"
FLIGHT_COLUMN = "flight_id"

TRAIN_SPLIT = "TRAIN"
VALIDATION_SPLIT = "VALIDATION"
TEST_SPLIT = "TEST"


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
        fail(f"Feature list not found:\n{FEATURE_LIST_FILE}")

    features = [
        line.strip()
        for line in FEATURE_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != 40:
        fail(
            f"Expected 40 final features, found {len(features)}."
        )

    return features


def load_split_table() -> pd.DataFrame:
    if not SPLIT_FILE.exists():
        fail(f"Split file not found:\n{SPLIT_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)

    split_col = None

    for col in split_df.columns:
        if col.lower() in {"split", "dataset_split", "set"}:
            split_col = col
            break

    if split_col is None:
        fail(
            "Could not identify split column in split file.\n"
            f"Columns: {list(split_df.columns)}"
        )

    split_df = split_df.rename(
        columns={split_col: "dataset_split"}
    ).copy()

    split_df[FLIGHT_COLUMN] = (
        split_df[FLIGHT_COLUMN]
        .astype(str)
        .str.strip()
    )

    split_df["dataset_split"] = (
        split_df["dataset_split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return split_df[
        [FLIGHT_COLUMN, "dataset_split"]
    ].drop_duplicates()


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    if not FEATURES_FILE.exists():
        fail(f"Feature file not found:\n{FEATURES_FILE}")

    if not MODEL_FILE.exists():
        fail(f"Model file not found:\n{MODEL_FILE}")

    df = pd.read_csv(FEATURES_FILE)

    features = load_feature_list()

    required_columns = {
        TARGET_COLUMN,
        ACTIVE_COLUMN,
        FLIGHT_COLUMN,
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        fail(
            "Required columns missing:\n"
            + "\n".join(f"  - {col}" for col in missing)
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

    split_df = load_split_table()

    df = df.merge(
        split_df,
        on=FLIGHT_COLUMN,
        how="left",
        validate="many_to_one",
    )

    if df["dataset_split"].isna().any():
        fail(
            "Some active-fault rows do not have a dataset split."
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    return df, features


def load_model_bundle():
    bundle = joblib.load(MODEL_FILE)

    if not isinstance(bundle, dict):
        fail(
            "Expected the saved model to be a dictionary bundle."
        )

    if "model" not in bundle:
        fail("Model bundle does not contain 'model'.")

    if "feature_names" not in bundle:
        fail("Model bundle does not contain 'feature_names'.")

    if "imputation_medians" not in bundle:
        fail(
            "Model bundle does not contain "
            "'imputation_medians'."
        )

    return bundle


def make_matrix(
    df: pd.DataFrame,
    features: list[str],
    medians: dict,
) -> pd.DataFrame:

    X = df[features].copy()

    for feature in features:
        X[feature] = X[feature].fillna(
            medians.get(feature, X[feature].median())
        )

    if np.isinf(
        X.to_numpy(dtype=float)
    ).any():
        fail("Infinite values found in feature matrix.")

    return X


# =============================================================================
# 1. BASIC DATASET DIAGNOSIS
# =============================================================================

def dataset_summary(
    df: pd.DataFrame,
) -> None:

    print_section("DATASET SUMMARY")

    print(f"Active rows      : {len(df):,}")
    print(
        f"Active flights   : "
        f"{df[FLIGHT_COLUMN].nunique():,}"
    )

    print("\nRows by split:")

    split_rows = (
        df.groupby("dataset_split")
        .size()
        .reindex(
            [
                TRAIN_SPLIT,
                VALIDATION_SPLIT,
                TEST_SPLIT,
            ],
            fill_value=0,
        )
    )

    for split, count in split_rows.items():
        print(f"  {split:<15} {int(count):>8,}")

    print("\nFlights by split:")

    split_flights = (
        df.groupby("dataset_split")[FLIGHT_COLUMN]
        .nunique()
        .reindex(
            [
                TRAIN_SPLIT,
                VALIDATION_SPLIT,
                TEST_SPLIT,
            ],
            fill_value=0,
        )
    )

    for split, count in split_flights.items():
        print(f"  {split:<15} {int(count):>8,}")


# =============================================================================
# 2. CLASS DISTRIBUTION
# =============================================================================

def class_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_section("CLASS DISTRIBUTION")

    table = pd.crosstab(
        df["dataset_split"],
        df[TARGET_COLUMN],
    )

    table = table.reindex(
        index=[
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            TEST_SPLIT,
        ],
        fill_value=0,
    )

    print(table.to_string())

    table.to_csv(
        OUTPUT_DIR / "class_distribution_by_split.csv"
    )

    return table


# =============================================================================
# 3. MODEL PREDICTIONS
# =============================================================================

def generate_predictions(
    df: pd.DataFrame,
    features: list[str],
    bundle: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    model = bundle["model"]

    model_features = list(
        bundle["feature_names"]
    )

    if model_features != features:
        fail(
            "Saved model feature order does not match "
            "current final feature list."
        )

    medians = bundle["imputation_medians"]

    X = make_matrix(
        df,
        features,
        medians,
    )

    predictions = model.predict(X)

    result = df[
        [
            FLIGHT_COLUMN,
            "dataset_split",
            TARGET_COLUMN,
        ]
    ].copy()

    result["prediction"] = predictions

    # Probability columns where available.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        class_names = list(
            model.classes_
        )

        for idx, class_name in enumerate(class_names):
            result[
                f"prob_{class_name}"
            ] = probabilities[:, idx]

        result["prediction_confidence"] = probabilities.max(
            axis=1
        )

    result.to_csv(
        OUTPUT_DIR / "row_level_predictions.csv",
        index=False,
    )

    return result, X


# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(
    bundle: dict,
    features: list[str],
) -> pd.DataFrame:

    print_section("MODEL FEATURE IMPORTANCE")

    model = bundle["model"]

    if not hasattr(model, "feature_importances_"):
        print(
            "Model does not expose tree feature_importances_."
        )
        return pd.DataFrame()

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
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
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    return importance


# =============================================================================
# 5. PERMUTATION IMPORTANCE
# =============================================================================

def analyze_permutation_importance(
    df: pd.DataFrame,
    X: pd.DataFrame,
    features: list[str],
    bundle: dict,
) -> pd.DataFrame:

    print_section(
        "PERMUTATION IMPORTANCE ON UNTOUCHED TEST SET"
    )

    test_mask = (
        df["dataset_split"] == TEST_SPLIT
    )

    X_test = X.loc[test_mask]
    y_test = df.loc[
        test_mask,
        TARGET_COLUMN,
    ]

    model = bundle["model"]

    # To keep runtime practical, use all test rows but
    # a limited number of repeats.
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    result = pd.DataFrame(
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
        result.head(20).to_string(
            index=False,
            formatters={
                "importance_mean": "{:.6f}".format,
                "importance_std": "{:.6f}".format,
            },
        )
    )

    result.to_csv(
        OUTPUT_DIR / "permutation_importance_test.csv",
        index=False,
    )

    return result


# =============================================================================
# 6. CLASS-WISE METRICS
# =============================================================================

def class_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    print_section("CLASS-WISE TEST PERFORMANCE")

    test = predictions[
        predictions["dataset_split"] == TEST_SPLIT
    ].copy()

    classes = sorted(
        test[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    rows = []

    for class_name in classes:
        y_true = (
            test[TARGET_COLUMN]
            .astype(str)
        )

        y_pred = (
            test["prediction"]
            .astype(str)
        )

        true_binary = (
            y_true == class_name
        ).astype(int)

        pred_binary = (
            y_pred == class_name
        ).astype(int)

        tp = int(
            (
                (true_binary == 1)
                & (pred_binary == 1)
            ).sum()
        )

        fp = int(
            (
                (true_binary == 0)
                & (pred_binary == 1)
            ).sum()
        )

        fn = int(
            (
                (true_binary == 1)
                & (pred_binary == 0)
            ).sum()
        )

        support = int(
            true_binary.sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "class": class_name,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    result.to_csv(
        OUTPUT_DIR / "class_metrics_test.csv",
        index=False,
    )

    return result


# =============================================================================
# 7. CONFUSION MATRIX + CONFUSION PAIRS
# =============================================================================

def confusion_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    print_section("CONFUSION ANALYSIS")

    test = predictions[
        predictions["dataset_split"] == TEST_SPLIT
    ].copy()

    labels = sorted(
        test[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    y_true = (
        test[TARGET_COLUMN]
        .astype(str)
    )

    y_pred = (
        test["prediction"]
        .astype(str)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"TRUE_{x}"
            for x in labels
        ],
        columns=[
            f"PRED_{x}"
            for x in labels
        ],
    )

    print("\nConfusion matrix:")
    print(cm_df.to_string())

    cm_df.to_csv(
        OUTPUT_DIR / "diagnostic_confusion_matrix.csv"
    )

    pairs = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):

            if i == j:
                continue

            count = int(cm[i, j])

            if count == 0:
                continue

            pairs.append(
                {
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "count": count,
                }
            )

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values(
            "count",
            ascending=False,
        )
        if pairs
        else pd.DataFrame(
            columns=[
                "true_class",
                "predicted_class",
                "count",
            ]
        )
    )

    print("\nTop confusion pairs:")

    if len(pairs_df) > 0:
        print(
            pairs_df.head(20).to_string(
                index=False
            )
        )
    else:
        print("No misclassifications.")

    pairs_df.to_csv(
        OUTPUT_DIR / "top_confusion_pairs.csv",
        index=False,
    )

    return pairs_df


# =============================================================================
# 8. FEATURE DISTRIBUTION DRIFT
# =============================================================================

def feature_distribution_drift(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    print_section(
        "TRAIN VS TEST FEATURE DISTRIBUTION"
    )

    train = df[
        df["dataset_split"] == TRAIN_SPLIT
    ]

    test = df[
        df["dataset_split"] == TEST_SPLIT
    ]

    rows = []

    for feature in features:

        train_values = train[feature].dropna()
        test_values = test[feature].dropna()

        train_mean = float(
            train_values.mean()
        )

        test_mean = float(
            test_values.mean()
        )

        train_std = float(
            train_values.std()
        )

        test_std = float(
            test_values.std()
        )

        train_median = float(
            train_values.median()
        )

        test_median = float(
            test_values.median()
        )

        # Symmetric standardized mean difference.
        pooled_std = np.sqrt(
            (
                train_std ** 2
                + test_std ** 2
            ) / 2
        )

        if pooled_std > 0:
            standardized_mean_diff = (
                test_mean - train_mean
            ) / pooled_std
        else:
            standardized_mean_diff = 0.0

        rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "train_median": train_median,
                "test_median": test_median,
                "standardized_mean_diff": (
                    standardized_mean_diff
                ),
                "abs_standardized_mean_diff": abs(
                    standardized_mean_diff
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "abs_standardized_mean_diff",
        ascending=False,
    )

    print(
        result.head(20).to_string(
            index=False,
            formatters={
                "train_mean": "{:.4f}".format,
                "test_mean": "{:.4f}".format,
                "train_std": "{:.4f}".format,
                "test_std": "{:.4f}".format,
                "train_median": "{:.4f}".format,
                "test_median": "{:.4f}".format,
                "standardized_mean_diff": "{:.4f}".format,
                "abs_standardized_mean_diff": "{:.4f}".format,
            },
        )
    )

    result.to_csv(
        OUTPUT_DIR / "train_test_feature_drift.csv",
        index=False,
    )

    return result


# =============================================================================
# 9. PER-CLASS FEATURE STATISTICS
# =============================================================================

def class_feature_statistics(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    print_section(
        "CLASS-WISE FEATURE SEPARATION"
    )

    test = df[
        df["dataset_split"] == TEST_SPLIT
    ].copy()

    classes = sorted(
        test[TARGET_COLUMN]
        .astype(str)
        .unique()
    )

    rows = []

    for feature in features:

        class_means = (
            test.groupby(TARGET_COLUMN)[feature]
            .mean()
        )

        class_stds = (
            test.groupby(TARGET_COLUMN)[feature]
            .std()
        )

        if class_means.empty:
            continue

        rows.append(
            {
                "feature": feature,
                "between_class_mean_range": (
                    float(
                        class_means.max()
                        - class_means.min()
                    )
                ),
                "max_class_mean": float(
                    class_means.max()
                ),
                "min_class_mean": float(
                    class_means.min()
                ),
                "average_within_class_std": float(
                    class_stds.mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["separation_ratio"] = (
            result["between_class_mean_range"]
            / result["average_within_class_std"].replace(
                0,
                np.nan,
            )
        )

        result = result.sort_values(
            "separation_ratio",
            ascending=False,
        )

        print(
            result.head(20).to_string(
                index=False,
                formatters={
                    "between_class_mean_range": "{:.4f}".format,
                    "max_class_mean": "{:.4f}".format,
                    "min_class_mean": "{:.4f}".format,
                    "average_within_class_std": "{:.4f}".format,
                    "separation_ratio": "{:.4f}".format,
                },
            )
        )

    result.to_csv(
        OUTPUT_DIR / "class_feature_separation.csv",
        index=False,
    )

    return result


# =============================================================================
# 10. OPERATING-CONTEXT FEATURE CHECK
# =============================================================================

def operating_context_analysis(
    df: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "OPERATING-CONTEXT FEATURE CHECK"
    )

    context_features = [
        "ambient_temp_c",
        "planned_altitude_ft",
        "airspeed_kts",
        "elapsed_s",
        "rpm",
        "rpm_mean_30s",
        "rpm_std_30s",
        "throttle_position_pct",
        "engine_load_pct",
        "intake_manifold_pressure_kpa",
    ]

    context_features = [
        feature
        for feature in context_features
        if feature in df.columns
    ]

    rows = []

    test = df[
        df["dataset_split"] == TEST_SPLIT
    ]

    for feature in context_features:

        means = (
            test.groupby(TARGET_COLUMN)[feature]
            .mean()
            .sort_values()
        )

        importance_value = np.nan

        if not importance.empty:
            match = importance[
                importance["feature"] == feature
            ]

            if not match.empty:
                importance_value = float(
                    match.iloc[0]["importance"]
                )

        rows.append(
            {
                "feature": feature,
                "importance": importance_value,
                "class_mean_range": float(
                    means.max() - means.min()
                ),
                "overall_mean": float(
                    test[feature].mean()
                ),
                "overall_std": float(
                    test[feature].std()
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "importance",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "importance": lambda x: (
                    "nan"
                    if pd.isna(x)
                    else f"{x:.6f}"
                ),
                "class_mean_range": "{:.4f}".format,
                "overall_mean": "{:.4f}".format,
                "overall_std": "{:.4f}".format,
            },
        )
    )

    result.to_csv(
        OUTPUT_DIR / "operating_context_analysis.csv",
        index=False,
    )

    return result


# =============================================================================
# 11. PER-FLIGHT PERFORMANCE
# =============================================================================

def per_flight_performance(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "TEST PERFORMANCE BY FLIGHT"
    )

    test = predictions[
        predictions["dataset_split"] == TEST_SPLIT
    ].copy()

    rows = []

    for flight_id, group in test.groupby(
        FLIGHT_COLUMN
    ):
        score = f1_score(
            group[TARGET_COLUMN],
            group["prediction"],
            average="macro",
            zero_division=0,
        )

        accuracy = (
            group[TARGET_COLUMN]
            == group["prediction"]
        ).mean()

        true_classes = (
            group[TARGET_COLUMN]
            .nunique()
        )

        rows.append(
            {
                "flight_id": flight_id,
                "rows": len(group),
                "true_classes": true_classes,
                "accuracy": float(accuracy),
                "macro_f1": float(score),
                "true_fault": (
                    group[TARGET_COLUMN]
                    .mode()
                    .iloc[0]
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "macro_f1"
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
            },
        )
    )

    result.to_csv(
        OUTPUT_DIR / "per_flight_test_performance.csv",
        index=False,
    )

    return result


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("FAULT CLASSIFIER DIAGNOSTIC ANALYSIS")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    df, features = prepare_data()
    bundle = load_model_bundle()

    print_section("LOADED ARTIFACTS")

    print(f"Feature count : {len(features)}")
    print(f"Model         : {bundle.get('model_name', 'unknown')}")
    print(f"Model file    : {MODEL_FILE}")

    dataset_summary(df)

    # -------------------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------------------
    class_distribution(df)

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------
    predictions, X = generate_predictions(
        df,
        features,
        bundle,
    )

    # -------------------------------------------------------------------------
    # Model feature importance
    # -------------------------------------------------------------------------
    importance = analyze_feature_importance(
        bundle,
        features,
    )

    # -------------------------------------------------------------------------
    # Permutation importance
    # -------------------------------------------------------------------------
    permutation = analyze_permutation_importance(
        df,
        X,
        features,
        bundle,
    )

    # -------------------------------------------------------------------------
    # Class metrics
    # -------------------------------------------------------------------------
    metrics = class_metrics(
        predictions
    )

    # -------------------------------------------------------------------------
    # Confusion
    # -------------------------------------------------------------------------
    confusion_pairs = confusion_analysis(
        predictions
    )

    # -------------------------------------------------------------------------
    # Distribution drift
    # -------------------------------------------------------------------------
    drift = feature_distribution_drift(
        df,
        features,
    )

    # -------------------------------------------------------------------------
    # Class separation
    # -------------------------------------------------------------------------
    separation = class_feature_statistics(
        df,
        features,
    )

    # -------------------------------------------------------------------------
    # Operating context
    # -------------------------------------------------------------------------
    context = operating_context_analysis(
        df,
        importance,
    )

    # -------------------------------------------------------------------------
    # Per flight
    # -------------------------------------------------------------------------
    flight_results = per_flight_performance(
        predictions
    )

    # -------------------------------------------------------------------------
    # Save overall diagnostic summary
    # -------------------------------------------------------------------------

    summary = {
        "model_name": bundle.get(
            "model_name",
            "unknown",
        ),
        "feature_count": len(features),
        "active_rows": int(len(df)),
        "active_flights": int(
            df[FLIGHT_COLUMN].nunique()
        ),
        "test_rows": int(
            (
                df["dataset_split"]
                == TEST_SPLIT
            ).sum()
        ),
        "test_flights": int(
            df.loc[
                df["dataset_split"] == TEST_SPLIT,
                FLIGHT_COLUMN,
            ].nunique()
        ),
        "top_model_features": (
            importance.head(10).to_dict(
                orient="records"
            )
            if not importance.empty
            else []
        ),
        "top_permutation_features": (
            permutation.head(10).to_dict(
                orient="records"
            )
            if not permutation.empty
            else []
        ),
        "top_confusion_pairs": (
            confusion_pairs.head(15).to_dict(
                orient="records"
            )
            if not confusion_pairs.empty
            else []
        ),
        "worst_feature_drift": (
            drift.head(10).to_dict(
                orient="records"
            )
            if not drift.empty
            else []
        ),
        "worst_test_classes": (
            metrics.sort_values(
                "f1"
            ).head(10).to_dict(
                orient="records"
            )
            if not metrics.empty
            else []
        ),
        "worst_test_flights": (
            flight_results.head(10).to_dict(
                orient="records"
            )
            if not flight_results.empty
            else []
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "diagnostic_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------
    print_section("DIAGNOSTIC OUTPUT")

    print(f"Output directory:")
    print(OUTPUT_DIR)

    print("\nCreated:")
    print("  - class_distribution_by_split.csv")
    print("  - row_level_predictions.csv")
    print("  - feature_importance.csv")
    print("  - permutation_importance_test.csv")
    print("  - class_metrics_test.csv")
    print("  - diagnostic_confusion_matrix.csv")
    print("  - top_confusion_pairs.csv")
    print("  - train_test_feature_drift.csv")
    print("  - class_feature_separation.csv")
    print("  - operating_context_analysis.csv")
    print("  - per_flight_test_performance.csv")
    print("  - diagnostic_summary.json")

    print("\nDIAGNOSTIC ANALYSIS COMPLETE.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("DIAGNOSTIC ANALYSIS FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)