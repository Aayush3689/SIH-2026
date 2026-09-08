"""
FAULT CLASSIFICATION FEATURE AUDIT
==================================

Purpose
-------
Audit the engineered features before training the Fault Classifier.

Inputs:
    1. fault_features.csv
    2. fault_flight_split_final.csv

Target:
    fault_mode_ground_truth

Training population:
    fault_active_label == 1

Split:
    flight-wise chronological split from
    fault_flight_split_final.csv

This script checks:

    1. Feature availability
    2. Target leakage
    3. Identifier leakage
    4. Constant / near-constant features
    5. Duplicate columns
    6. Highly correlated features
    7. Missing / infinite values
    8. Train-vs-test feature drift
    9. Final candidate feature set
    10. Feature importance using a simple
        non-final RF ranking

IMPORTANT:
    - No final model is trained.
    - engine_id and flight_id are NEVER candidate features.
    - fault target columns are NEVER candidate features.
    - health_index / RUL are excluded.
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# =====================================================================
# PATHS
# =====================================================================

# File:
# ai-backend/training/fault/audit_fault_features.py
#
# parents[0] -> training/fault
# parents[1] -> training
#
TRAINING_DIR = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_features.csv"
)

SPLIT_PATH = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_flight_split_final.csv"
)

OUTPUT_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
)

FEATURE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "fault_feature_audit.csv"
)

FINAL_FEATURE_PATH = (
    OUTPUT_DIR
    / "fault_classifier_candidate_features.txt"
)

DRIFT_PATH = (
    OUTPUT_DIR
    / "fault_feature_drift.csv"
)


# =====================================================================
# CONSTANTS
# =====================================================================

TARGET_COLUMN = "fault_mode_ground_truth"

ACTIVE_LABEL = 1

# We intentionally exclude identifiers.
IDENTIFIER_COLUMNS = {
    "engine_id",
    "uav_id",
    "flight_id",
    "timestamp",
}

# Target / potentially target-derived information.
TARGET_LEAKAGE_COLUMNS = {
    "fault_mode_ground_truth",
    "fault_active_label",
}

# These must not be used by the fault classifier.
OTHER_TARGET_COLUMNS = {
    "health_index",
    "rul_hours_remaining",
    "rul_hours",
    "rul",
}

# Features whose names clearly indicate labels / predictions.
LEAKAGE_KEYWORDS = [
    "label",
    "target",
    "ground_truth",
    "fault_mode",
    "fault_active",
    "health_index",
    "rul",
]

# Correlation threshold.
HIGH_CORRELATION_THRESHOLD = 0.98

# Near-constant threshold.
NEAR_CONSTANT_THRESHOLD = 0.999

# Drift warning threshold.
DRIFT_RATIO_THRESHOLD = 5.0

RANDOM_STATE = 42


# =====================================================================
# HELPERS
# =====================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


# =====================================================================
# LOAD
# =====================================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not FEATURE_PATH.exists():
        fail(
            "Feature file not found:\n"
            f"{FEATURE_PATH}"
        )

    if not SPLIT_PATH.exists():
        fail(
            "Final split file not found:\n"
            f"{SPLIT_PATH}"
        )

    print(
        "Loading fault features..."
    )

    features = pd.read_csv(
        FEATURE_PATH
    )

    print(
        "Loading final flight split..."
    )

    split = pd.read_csv(
        SPLIT_PATH
    )

    return features, split


# =====================================================================
# BASIC VALIDATION
# =====================================================================

def validate_inputs(
    features: pd.DataFrame,
    split: pd.DataFrame,
) -> None:

    required_feature_columns = {
        "engine_id",
        "flight_id",
        TARGET_COLUMN,
    }

    missing = (
        required_feature_columns
        - set(features.columns)
    )

    if missing:
        fail(
            "Feature file is missing required columns:\n"
            f"{sorted(missing)}"
        )

    required_split_columns = {
        "engine_id",
        "flight_id",
        "fault_mode",
        "split",
    }

    missing = (
        required_split_columns
        - set(split.columns)
    )

    if missing:
        fail(
            "Split file is missing required columns:\n"
            f"{sorted(missing)}"
        )

    if features.empty:
        fail(
            "Feature dataset is empty."
        )

    if split.empty:
        fail(
            "Split dataset is empty."
        )


# =====================================================================
# KEEP ACTIVE FAULT ROWS
# =====================================================================

def build_training_population(
    features: pd.DataFrame,
    split: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only the active-fault telemetry rows represented in
    the final fault flight split.

    The split was created from:
        fault_active_label == 1

    so joining against its flight IDs gives us the exact
    classification population.
    """

    keys = [
        "engine_id",
        "flight_id",
    ]

    split_keys = (
        split[keys]
        .drop_duplicates()
    )

    population = features.merge(
        split_keys,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    if population.empty:
        fail(
            "No feature rows matched the final "
            "fault flight split."
        )

    # -------------------------------------------------------------
    # Attach split metadata
    # -------------------------------------------------------------

    split_columns = [
        "engine_id",
        "flight_id",
        "fault_mode",
        "split",
    ]

    population = population.merge(
        split[
            split_columns
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if population["split"].isna().any():
        fail(
            "Some population rows do not have "
            "a split assignment."
        )

    # -------------------------------------------------------------
    # Target consistency
    # -------------------------------------------------------------

    feature_target = (
        population[
            TARGET_COLUMN
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_target = (
        population[
            "fault_mode"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mismatch = (
        feature_target
        != split_target
    )

    mismatch_count = int(
        mismatch.sum()
    )

    print()

    print(
        f"Target consistency mismatches: "
        f"{mismatch_count:,}"
    )

    if mismatch_count:
        fail(
            "fault_mode_ground_truth in features "
            "does not match fault_mode in split."
        )

    population[
        TARGET_COLUMN
    ] = feature_target

    return population


# =====================================================================
# IDENTIFY MODEL CANDIDATES
# =====================================================================

def identify_candidate_features(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    """
    Decide which columns are eligible for the fault classifier.

    Returns:
        candidates
        exclusion_reasons
    """

    candidates = []
    exclusions = {}

    for column in df.columns:

        # ---------------------------------------------------------
        # Explicit identifiers
        # ---------------------------------------------------------

        if column in IDENTIFIER_COLUMNS:

            exclusions[column] = (
                "identifier"
            )

            continue

        # ---------------------------------------------------------
        # Explicit targets
        # ---------------------------------------------------------

        if column in TARGET_LEAKAGE_COLUMNS:

            exclusions[column] = (
                "target/leakage"
            )

            continue

        # ---------------------------------------------------------
        # Other target-like signals
        # ---------------------------------------------------------

        if column in OTHER_TARGET_COLUMNS:

            exclusions[column] = (
                "other target / future information"
            )

            continue

        # ---------------------------------------------------------
        # Name-based leakage
        # ---------------------------------------------------------

        lower_name = (
            column.lower()
        )

        leakage_keyword = next(
            (
                keyword
                for keyword in LEAKAGE_KEYWORDS
                if keyword in lower_name
            ),
            None,
        )

        if leakage_keyword is not None:

            exclusions[column] = (
                f"name contains leakage keyword "
                f"'{leakage_keyword}'"
            )

            continue

        # ---------------------------------------------------------
        # Non-numeric columns
        # ---------------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            exclusions[column] = (
                "non-numeric feature"
            )

            continue

        candidates.append(
            column
        )

    return candidates, exclusions


# =====================================================================
# MISSING / INFINITE AUDIT
# =====================================================================

def audit_missing_values(
    df: pd.DataFrame,
    candidate_features: list[str],
) -> pd.DataFrame:

    rows = []

    for column in candidate_features:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        missing_count = int(
            series.isna().sum()
        )

        infinite_count = int(
            np.isinf(
                series.to_numpy(
                    dtype=float
                )
            ).sum()
        )

        rows.append(
            {
                "feature":
                    column,

                "missing_count":
                    missing_count,

                "missing_pct":
                    missing_count
                    / len(df)
                    * 100,

                "infinite_count":
                    infinite_count,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# CONSTANT / NEAR-CONSTANT AUDIT
# =====================================================================

def audit_constant_features(
    df: pd.DataFrame,
    candidate_features: list[str],
) -> pd.DataFrame:

    rows = []

    for column in candidate_features:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        unique_count = (
            series.nunique(
                dropna=True
            )
        )

        value_counts = (
            series
            .value_counts(
                normalize=True
            )
        )

        dominant_ratio = (
            float(
                value_counts.iloc[0]
            )
            if not value_counts.empty
            else 1.0
        )

        rows.append(
            {
                "feature":
                    column,

                "unique_values":
                    unique_count,

                "dominant_value_ratio":
                    dominant_ratio,

                "is_constant":
                    unique_count <= 1,

                "is_near_constant":
                    dominant_ratio
                    >= NEAR_CONSTANT_THRESHOLD,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# DUPLICATE FEATURE AUDIT
# =====================================================================

def find_duplicate_columns(
    df: pd.DataFrame,
    candidate_features: list[str],
) -> list[tuple[str, str]]:

    duplicates = []

    # Compare only numeric candidate columns.
    for i, left in enumerate(
        candidate_features
    ):

        left_values = (
            pd.to_numeric(
                df[left],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

        for right in candidate_features[
            i + 1:
        ]:

            right_values = (
                pd.to_numeric(
                    df[right],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy()
            )

            if np.array_equal(
                left_values,
                right_values,
            ):

                duplicates.append(
                    (
                        left,
                        right,
                    )
                )

    return duplicates


# =====================================================================
# HIGH CORRELATION AUDIT
# =====================================================================

def find_high_correlations(
    df: pd.DataFrame,
    candidate_features: list[str],
) -> pd.DataFrame:

    numeric = df[
        candidate_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    corr = numeric.corr()

    rows = []

    for i, left in enumerate(
        corr.columns
    ):

        for right in corr.columns[
            i + 1:
        ]:

            value = corr.loc[
                left,
                right,
            ]

            if pd.isna(value):
                continue

            if abs(value) >= HIGH_CORRELATION_THRESHOLD:

                rows.append(
                    {
                        "feature_a":
                            left,

                        "feature_b":
                            right,

                        "correlation":
                            float(value),
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = (
            result
            .sort_values(
                "correlation",
                key=lambda s:
                s.abs(),
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =====================================================================
# TRAIN / TEST DISTRIBUTION DRIFT
# =====================================================================

def compute_feature_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: list[str],
) -> pd.DataFrame:
    """
    Compare train and test distributions.

    This is NOT a statistical deployment drift test.

    It is a simple engineering audit using:
        train mean/std
        test mean/std
        mean ratio
        std ratio
    """

    rows = []

    for column in candidate_features:

        train_series = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        test_series = pd.to_numeric(
            test_df[column],
            errors="coerce",
        )

        train_mean = float(
            train_series.mean()
        )

        test_mean = float(
            test_series.mean()
        )

        train_std = float(
            train_series.std()
        )

        test_std = float(
            test_series.std()
        )

        if abs(train_mean) > 1e-9:

            mean_ratio = (
                abs(test_mean)
                / abs(train_mean)
            )

        else:

            mean_ratio = np.inf

        if abs(train_std) > 1e-9:

            std_ratio = (
                abs(test_std)
                / abs(train_std)
            )

        else:

            std_ratio = np.inf

        rows.append(
            {
                "feature":
                    column,

                "train_mean":
                    train_mean,

                "test_mean":
                    test_mean,

                "mean_ratio":
                    mean_ratio,

                "train_std":
                    train_std,

                "test_std":
                    test_std,

                "std_ratio":
                    std_ratio,

                "possible_drift":
                    (
                        mean_ratio
                        >= DRIFT_RATIO_THRESHOLD
                        or
                        mean_ratio
                        <= 1
                        / DRIFT_RATIO_THRESHOLD
                        or
                        std_ratio
                        >= DRIFT_RATIO_THRESHOLD
                        or
                        std_ratio
                        <= 1
                        / DRIFT_RATIO_THRESHOLD
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# QUICK RF FEATURE RANKING
# =====================================================================

def quick_feature_ranking(
    train_df: pd.DataFrame,
    candidate_features: list[str],
) -> pd.DataFrame:
    """
    Train a SMALL exploratory Random Forest only for feature ranking.

    This is NOT the final classifier.

    It is useful for identifying:
        - clearly useful features
        - suspicious leakage-like features
        - completely irrelevant features

    The model is not saved.
    """

    X = train_df[
        candidate_features
    ].copy()

    y = (
        train_df[
            TARGET_COLUMN
        ]
        .astype("category")
        .cat.codes
    )

    # Numeric conversion
    for column in candidate_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Median imputation using TRAIN only.
    medians = X.median()

    X = X.fillna(
        medians
    )

    if X.isna().any().any():

        fail(
            "NaN remains in exploratory ranking features."
        )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    model.fit(
        X,
        y,
    )

    ranking = pd.DataFrame(
        {
            "feature":
                candidate_features,

            "importance":
                model.feature_importances_,
        }
    )

    ranking = (
        ranking
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "rank"
    ] = (
        ranking.index
        + 1
    )

    return ranking


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 78)
    print(
        "FAULT CLASSIFICATION FEATURE AUDIT"
    )
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    features, split = load_inputs()

    validate_inputs(
        features,
        split,
    )

    print()
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )

    print(
        f"Feature columns: "
        f"{len(features.columns)}"
    )

    print(
        f"Split rows: "
        f"{len(split):,}"
    )

    # -------------------------------------------------------------
    # Build classification population
    # -------------------------------------------------------------

    population = build_training_population(
        features,
        split,
    )

    print()
    print(
        f"Fault classification rows: "
        f"{len(population):,}"
    )

    print(
        f"Fault classes: "
        f"{population[TARGET_COLUMN].nunique()}"
    )

    # -------------------------------------------------------------
    # Split distribution
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("POPULATION BY SPLIT")
    print("=" * 78)

    split_summary = (
        population
        .groupby(
            "split"
        )
        .agg(
            rows=(
                TARGET_COLUMN,
                "size",
            ),
            flights=(
                "flight_id",
                "nunique",
            ),
            engines=(
                "engine_id",
                "nunique",
            ),
        )
        .reindex(
            [
                "TRAIN",
                "VALIDATION",
                "TEST",
            ]
        )
    )

    print(
        split_summary.to_string()
    )

    # -------------------------------------------------------------
    # Candidate features
    # -------------------------------------------------------------

    candidates, exclusions = (
        identify_candidate_features(
            population
        )
    )

    print()
    print("=" * 78)
    print("FEATURE ELIGIBILITY")
    print("=" * 78)

    print(
        f"Total columns: "
        f"{len(population.columns)}"
    )

    print(
        f"Candidate numeric features: "
        f"{len(candidates)}"
    )

    print(
        f"Excluded columns: "
        f"{len(exclusions)}"
    )

    print()

    for column, reason in exclusions.items():

        print(
            f"EXCLUDE  {column:40s}"
            f" -> {reason}"
        )

    # -------------------------------------------------------------
    # Missing / infinity
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("MISSING / INFINITE VALUES")
    print("=" * 78)

    missing_df = audit_missing_values(
        population,
        candidates,
    )

    problematic_missing = (
        missing_df[
            (
                missing_df[
                    "missing_count"
                ]
                > 0
            )
            |
            (
                missing_df[
                    "infinite_count"
                ]
                > 0
            )
        ]
        .sort_values(
            "missing_count",
            ascending=False,
        )
    )

    if problematic_missing.empty:

        print(
            "PASS: No missing or infinite values "
            "in candidate features."
        )

    else:

        print(
            problematic_missing.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Constant features
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("CONSTANT / NEAR-CONSTANT FEATURES")
    print("=" * 78)

    constant_df = (
        audit_constant_features(
            population,
            candidates,
        )
    )

    constant_features = (
        constant_df[
            constant_df[
                "is_constant"
            ]
            |
            constant_df[
                "is_near_constant"
            ]
        ]
    )

    if constant_features.empty:

        print(
            "PASS: No constant / near-constant "
            "candidate features."
        )

    else:

        print(
            constant_features.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Duplicate columns
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("DUPLICATE FEATURE CHECK")
    print("=" * 78)

    duplicate_columns = find_duplicate_columns(
        population,
        candidates,
    )

    if not duplicate_columns:

        print(
            "PASS: No duplicate candidate feature columns."
        )

    else:

        print(
            f"Found "
            f"{len(duplicate_columns)} "
            f"duplicate feature pair(s):"
        )

        for left, right in duplicate_columns:

            print(
                f"  {left} == {right}"
            )

    # -------------------------------------------------------------
    # High correlation
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("HIGH-CORRELATION FEATURE CHECK")
    print("=" * 78)

    high_corr_df = (
        find_high_correlations(
            population,
            candidates,
        )
    )

    if high_corr_df.empty:

        print(
            "PASS: No feature pairs above "
            f"{HIGH_CORRELATION_THRESHOLD:.2f} correlation."
        )

    else:

        print(
            f"Found "
            f"{len(high_corr_df)} "
            f"high-correlation pairs:"
        )

        print(
            high_corr_df.head(30).to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Train / validation / test
    # -------------------------------------------------------------

    train_df = population[
        population["split"]
        == "TRAIN"
    ].copy()

    val_df = population[
        population["split"]
        == "VALIDATION"
    ].copy()

    test_df = population[
        population["split"]
        == "TEST"
    ].copy()

    if train_df.empty:
        fail("TRAIN split is empty.")

    if val_df.empty:
        fail("VALIDATION split is empty.")

    if test_df.empty:
        fail("TEST split is empty.")

    # -------------------------------------------------------------
    # Class coverage
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("CLASS COVERAGE")
    print("=" * 78)

    coverage = (
        population
        .groupby(
            [
                "split",
                TARGET_COLUMN,
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        coverage.to_string()
    )

    expected_classes = sorted(
        population[
            TARGET_COLUMN
        ]
        .unique()
        .tolist()
    )

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        present = set(
            population[
                population["split"]
                == split_name
            ][TARGET_COLUMN]
            .unique()
        )

        missing_classes = (
            set(expected_classes)
            - present
        )

        if missing_classes:

            fail(
                f"{split_name} is missing classes: "
                f"{sorted(missing_classes)}"
            )

    print()
    print(
        "PASS: All fault classes are present "
        "in TRAIN / VALIDATION / TEST."
    )

    # -------------------------------------------------------------
    # Feature drift
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAIN -> TEST FEATURE DRIFT AUDIT")
    print("=" * 78)

    drift_df = compute_feature_drift(
        train_df,
        test_df,
        candidates,
    )

    drift_df.to_csv(
        DRIFT_PATH,
        index=False,
    )

    possible_drift = (
        drift_df[
            drift_df[
                "possible_drift"
            ]
        ]
        .sort_values(
            "feature"
        )
    )

    if possible_drift.empty:

        print(
            "No extreme mean/std ratio drift detected."
        )

    else:

        print(
            f"Potentially drifting features: "
            f"{len(possible_drift)}"
        )

        print(
            possible_drift.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Quick exploratory feature ranking
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPLORATORY FEATURE IMPORTANCE")
    print("=" * 78)

    ranking = quick_feature_ranking(
        train_df,
        candidates,
    )

    print()

    print(
        ranking.head(25).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Build final candidate list
    # -------------------------------------------------------------

    remove_features = set()

    # Remove constants / near-constants
    for _, row in constant_df.iterrows():

        if (
            row["is_constant"]
            or row["is_near_constant"]
        ):

            remove_features.add(
                row["feature"]
            )

    # Remove exact duplicates
    for left, right in duplicate_columns:

        # Keep the first one, remove second.
        remove_features.add(
            right
        )

    final_candidates = [
        feature
        for feature in candidates
        if feature
        not in remove_features
    ]

    # -------------------------------------------------------------
    # Save candidate feature list
    # -------------------------------------------------------------

    with open(
        FINAL_FEATURE_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for feature in final_candidates:

            file.write(
                f"{feature}\n"
            )

    # -------------------------------------------------------------
    # Build feature audit table
    # -------------------------------------------------------------

    summary = (
        missing_df
        .merge(
            constant_df,
            on="feature",
            how="left",
        )
    )

    summary[
        "excluded_after_basic_audit"
    ] = summary[
        "feature"
    ].isin(
        remove_features
    )

    summary[
        "final_candidate"
    ] = summary[
        "feature"
    ].isin(
        final_candidates
    )

    summary = (
        summary
        .sort_values(
            [
                "final_candidate",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        FEATURE_SUMMARY_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL FEATURE AUDIT SUMMARY")
    print("=" * 78)

    print(
        f"Initial numeric candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Removed constant/duplicate  : "
        f"{len(remove_features)}"
    )

    print(
        f"Final candidate features    : "
        f"{len(final_candidates)}"
    )

    print()

    print(
        "Final candidate features:"
    )

    for index, feature in enumerate(
        final_candidates,
        start=1,
    ):

        print(
            f"{index:3d}. "
            f"{feature}"
        )

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    print()
    print(
        f"Feature audit:\n"
        f"{FEATURE_SUMMARY_PATH}"
    )

    print()
    print(
        f"Final candidate list:\n"
        f"{FINAL_FEATURE_PATH}"
    )

    print()
    print(
        f"Train/Test drift:\n"
        f"{DRIFT_PATH}"
    )

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    print(
        """
Next step:

    final candidate features
            ↓
    leakage review
            ↓
    Fault Classifier model comparison
            ↓
    validation
            ↓
    final unseen-flight TEST

No final classifier was trained by this script.
"""
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()