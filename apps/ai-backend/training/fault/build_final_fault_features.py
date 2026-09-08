from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent

INPUT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_features.csv"
)

OUTPUT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_final_features.txt"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features that MUST NOT be used by the fault classification model.
#
# Reasons:
#   - identifiers / metadata
#   - target / labels
#   - leakage-like / target-derived
#   - redundant delta features where corresponding rate feature is retained
#
EXCLUDED_FEATURES = {
    # -------------------------------------------------------------------------
    # Identifiers / metadata
    # -------------------------------------------------------------------------
    "timestamp",
    "engine_id",
    "uav_id",
    "flight_id",
    "flight_phase",
    "mission_type",
    "environmental_profile",

    # -------------------------------------------------------------------------
    # Target / label columns
    # -------------------------------------------------------------------------
    "fault_mode_ground_truth",
    "fault_active_label",

    "label_abnormal_vibration",
    "label_coking_degradation",
    "label_combustion_instability",
    "label_injector_abnormality",
    "label_lubrication_issue",
    "label_misfire",
    "label_overheating_trend",
    "label_sensor_drift",

    # -------------------------------------------------------------------------
    # Excluded after leakage / redundancy audit
    # -------------------------------------------------------------------------
    "flight_index",
    "fault_severity_score",

    # Delta features are redundant with rate features.
    "egt_delta",
    "cht_delta",
    "oil_pressure_delta",
    "vibration_delta",
}


# These are explicitly known target-related columns.
# Used as an additional safety check.
TARGET_COLUMNS = {
    "fault_mode_ground_truth",
    "fault_active_label",
    "label_abnormal_vibration",
    "label_coking_degradation",
    "label_combustion_instability",
    "label_injector_abnormality",
    "label_lubrication_issue",
    "label_misfire",
    "label_overheating_trend",
    "label_sensor_drift",
}


# Extra keywords that should trigger a manual review.
LEAKAGE_KEYWORDS = (
    "target",
    "ground_truth",
    "label_",
    "fault_severity",
)


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


def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("BUILD FINAL FAULT CLASSIFICATION FEATURE SET")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------
    if not INPUT_FILE.exists():
        fail(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows   : {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # Validate expected target columns exist
    # -------------------------------------------------------------------------
    missing_target_columns = sorted(
        col for col in TARGET_COLUMNS
        if col not in df.columns
    )

    if missing_target_columns:
        fail(
            "Expected target/label columns are missing from fault_features.csv:\n"
            + "\n".join(f"  - {col}" for col in missing_target_columns)
        )

    # -------------------------------------------------------------------------
    # Print exclusions
    # -------------------------------------------------------------------------
    print_section("EXCLUDED FEATURES")

    for column in df.columns:
        if column in EXCLUDED_FEATURES:
            if column in TARGET_COLUMNS:
                reason = "target / label"
            elif column in {
                "timestamp",
                "engine_id",
                "uav_id",
                "flight_id",
                "flight_phase",
                "mission_type",
                "environmental_profile",
            }:
                reason = "identifier / target / metadata"
            else:
                reason = "excluded after leakage/redundancy audit"

            print(f"{column:<35} -> {reason}")

    # -------------------------------------------------------------------------
    # Check for unknown leakage-like columns
    # -------------------------------------------------------------------------
    leakage_like_columns = []

    for column in df.columns:
        column_lower = column.lower()

        if column in EXCLUDED_FEATURES:
            continue

        for keyword in LEAKAGE_KEYWORDS:
            if keyword in column_lower:
                leakage_like_columns.append(column)
                break

    if leakage_like_columns:
        print_section("LEAKAGE-LIKE COLUMNS DETECTED")

        for column in sorted(leakage_like_columns):
            print(column)

        fail(
            "Potential leakage-like features remain:\n"
            + str(sorted(leakage_like_columns))
        )

    # -------------------------------------------------------------------------
    # Build numeric candidate feature set
    # -------------------------------------------------------------------------
    candidate_features = []

    for column in df.columns:
        if column in EXCLUDED_FEATURES:
            continue

        if not is_numeric_series(df[column]):
            continue

        candidate_features.append(column)

    # -------------------------------------------------------------------------
    # Validate that no target columns slipped through
    # -------------------------------------------------------------------------
    target_overlap = sorted(
        set(candidate_features).intersection(TARGET_COLUMNS)
    )

    if target_overlap:
        fail(
            "Target columns are still present in final feature candidates:\n"
            + str(target_overlap)
        )

    # -------------------------------------------------------------------------
    # Validate that no excluded features slipped through
    # -------------------------------------------------------------------------
    exclusion_overlap = sorted(
        set(candidate_features).intersection(EXCLUDED_FEATURES)
    )

    if exclusion_overlap:
        fail(
            "Excluded features are still present in final feature candidates:\n"
            + str(exclusion_overlap)
        )

    # -------------------------------------------------------------------------
    # Validate feature names
    # -------------------------------------------------------------------------
    duplicate_features = (
        pd.Series(candidate_features)[
            pd.Series(candidate_features).duplicated()
        ]
        .tolist()
    )

    if duplicate_features:
        fail(
            "Duplicate feature names detected:\n"
            + str(duplicate_features)
        )

    # -------------------------------------------------------------------------
    # Validate numeric values
    # -------------------------------------------------------------------------
    non_numeric = [
        feature
        for feature in candidate_features
        if not is_numeric_series(df[feature])
    ]

    if non_numeric:
        fail(
            "Non-numeric features found in final candidate set:\n"
            + str(non_numeric)
        )

    # -------------------------------------------------------------------------
    # Check infinity values
    # -------------------------------------------------------------------------
    inf_features = []

    for feature in candidate_features:
        values = pd.to_numeric(df[feature], errors="coerce")

        if np.isinf(values.to_numpy(dtype=float)).any():
            inf_features.append(feature)

    if inf_features:
        print_section("INFINITE VALUES")

        for feature in inf_features:
            print(feature)

        fail(
            "Infinite values detected in final candidate features:\n"
            + str(inf_features)
        )

    # -------------------------------------------------------------------------
    # Missing-value report
    # -------------------------------------------------------------------------
    missing_report = []

    for feature in candidate_features:
        missing_count = int(df[feature].isna().sum())

        if missing_count > 0:
            missing_report.append(
                {
                    "feature": feature,
                    "missing_count": missing_count,
                    "missing_pct": (
                        missing_count / len(df) * 100
                        if len(df) > 0
                        else 0.0
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Sort feature names for deterministic output
    # -------------------------------------------------------------------------
    candidate_features = sorted(candidate_features)

    # -------------------------------------------------------------------------
    # Expected final feature count
    # -------------------------------------------------------------------------
    expected_feature_count = 40

    if len(candidate_features) != expected_feature_count:
        print_section("FEATURE COUNT WARNING")

        print(f"Expected feature count : {expected_feature_count}")
        print(f"Actual feature count   : {len(candidate_features)}")

        print("\nActual features:")

        for idx, feature in enumerate(candidate_features, start=1):
            print(f"{idx:>2}. {feature}")

        fail(
            f"Final feature count is {len(candidate_features)}, "
            f"but expected {expected_feature_count}."
        )

    # -------------------------------------------------------------------------
    # Print final feature set
    # -------------------------------------------------------------------------
    print_section("FINAL FEATURE SET")

    for idx, feature in enumerate(candidate_features, start=1):
        print(f"{idx:>2}. {feature}")

    print(f"\nFINAL FEATURE COUNT: {len(candidate_features)}")

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------
    print_section("MISSING VALUE CHECK")

    if missing_report:
        for item in missing_report:
            print(
                f"{item['feature']:<35} "
                f"{item['missing_count']:>6,} "
                f"({item['missing_pct']:.6f}%)"
            )
    else:
        print("No missing values in final feature set.")

    # -------------------------------------------------------------------------
    # Save feature list
    # -------------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for feature in candidate_features:
            f.write(feature + "\n")

    print_section("OUTPUT")

    print(f"Feature list saved to:")
    print(OUTPUT_FILE)

    # -------------------------------------------------------------------------
    # Final verification from saved file
    # -------------------------------------------------------------------------
    saved_features = [
        line.strip()
        for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if saved_features != candidate_features:
        fail(
            "Saved feature list does not exactly match the final "
            "candidate feature list."
        )

    print(f"\nSaved features: {len(saved_features)}")
    print("Verification   : PASSED")
    print("\nBUILD COMPLETE.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()