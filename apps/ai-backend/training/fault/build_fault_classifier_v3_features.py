from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

TRAINING_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_features.csv"
)

V2_FEATURE_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v2_features.txt"
)

OUTPUT_FEATURE_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v3_features.txt"
)

OUTPUT_DATA_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v3_features.csv"
)

EXPECTED_V3_FEATURE_COUNT = 53


# =============================================================================
# TARGET / LEAKAGE COLUMNS
# =============================================================================

TARGET_COLUMNS = {
    "fault_mode_ground_truth",
    "fault_active_label",
    "fault_severity_score",
    "flight_index",
    "engine_id",
    "uav_id",
    "flight_id",
    "timestamp",
    "flight_phase",
    "mission_type",
    "environmental_profile",
}

LABEL_COLUMNS = {
    "label_abnormal_vibration",
    "label_coking_degradation",
    "label_combustion_instability",
    "label_injector_abnormality",
    "label_lubrication_issue",
    "label_misfire",
    "label_overheating_trend",
    "label_sensor_drift",
}


# =============================================================================
# HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = 1e-6,
) -> pd.Series:
    """
    Stable ratio calculation.

    Very small denominators are replaced with NaN.
    Infinite values are also converted to NaN.
    """
    denominator = denominator.astype(float)

    denominator = denominator.where(
        denominator.abs() >= eps,
        np.nan,
    )

    result = (
        numerator.astype(float)
        / denominator
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def read_v2_features() -> list[str]:

    if not V2_FEATURE_FILE.exists():
        fail(
            f"V2 feature file not found:\n"
            f"{V2_FEATURE_FILE}"
        )

    features = [
        line.strip()
        for line in V2_FEATURE_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(features) != 27:
        fail(
            f"Expected 27 V2 features, found {len(features)}."
        )

    return features


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("BUILD FAULT CLASSIFIER V3 FEATURE SET")
    print("PHYSICS / RELATIONSHIP-BASED FEATURES")
    print("=" * 78)

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nInput rows   : {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns):,}"
    )

    # -------------------------------------------------------------------------
    # Load V2 features
    # -------------------------------------------------------------------------

    v2_features = read_v2_features()

    missing_v2 = [
        feature
        for feature in v2_features
        if feature not in df.columns
    ]

    if missing_v2:
        fail(
            "V2 features missing from source data:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_v2
            )
        )

    # -------------------------------------------------------------------------
    # Derived V3 features
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("BUILDING V3 DERIVED FEATURES")
    print("=" * 78)

    derived = pd.DataFrame(
        index=df.index
    )

    # -------------------------------------------------------------------------
    # 1. Thermal relationships
    # -------------------------------------------------------------------------

    derived["egt_cht_ratio"] = safe_ratio(
        df["egt_c"],
        df["cht_c"],
    )

    derived["egt_cht_mean_ratio"] = safe_ratio(
        df["egt_mean_60s"],
        df["cht_mean_60s"],
    )

    derived["egt_minus_cht"] = (
        df["egt_c"]
        - df["cht_c"]
    )

    derived["egt_mean_minus_cht_mean"] = (
        df["egt_mean_60s"]
        - df["cht_mean_60s"]
    )

    derived["cht_residual_ratio"] = safe_ratio(
        df["cht_residual"],
        df["cht_expected"].abs(),
    )

    derived["egt_residual_ratio"] = safe_ratio(
        df["egt_residual"],
        df["egt_expected"].abs(),
    )

    # -------------------------------------------------------------------------
    # 2. Oil-system relationships
    # -------------------------------------------------------------------------

    derived["oil_pressure_rpm_ratio"] = safe_ratio(
        df["oil_pressure_kpa"],
        df["rpm_mean_30s"],
    )

    derived["oil_pressure_temperature_ratio"] = safe_ratio(
        df["oil_pressure_kpa"],
        df["oil_temperature_c"],
    )

    derived["oil_temperature_residual_ratio"] = safe_ratio(
        df["oil_temperature_residual"],
        df["oil_temperature_expected"].abs(),
    )

    derived["oil_pressure_change_to_rpm"] = safe_ratio(
        df["oil_pressure_rate"],
        df["rpm_mean_30s"].abs(),
    )

    # -------------------------------------------------------------------------
    # 3. Fuel-system relationships
    # -------------------------------------------------------------------------

    derived["fuel_pressure_to_flow"] = safe_ratio(
        df["fuel_pressure_kpa"],
        df["fuel_flow_lph"],
    )

    derived["fuel_flow_to_pressure"] = safe_ratio(
        df["fuel_flow_lph"],
        df["fuel_pressure_kpa"],
    )

    # -------------------------------------------------------------------------
    # 4. Vibration relationships
    # -------------------------------------------------------------------------

    derived["vibration_to_rpm"] = safe_ratio(
        df["engine_vibration_mm_s"],
        df["rpm_mean_30s"],
    )

    derived["vibration_growth_ratio"] = safe_ratio(
        df["vibration_rate"],
        df["vibration_mean_30s"].abs(),
    )

    derived["vibration_std_to_mean"] = safe_ratio(
        df["vibration_std_30s"],
        df["vibration_mean_30s"].abs(),
    )

    # -------------------------------------------------------------------------
    # 5. Variability relationships
    # -------------------------------------------------------------------------

    derived["egt_variability_ratio"] = safe_ratio(
        df["egt_std_60s"],
        df["egt_mean_60s"].abs(),
    )

    derived["cht_variability_ratio"] = safe_ratio(
        df["cht_std_60s"],
        df["cht_mean_60s"].abs(),
    )

    derived["oil_pressure_variability_ratio"] = safe_ratio(
        df["oil_pressure_std_60s"],
        df["oil_pressure_mean_60s"].abs(),
    )

    derived["oil_temperature_variability_ratio"] = safe_ratio(
        df["oil_temperature_std_60s"],
        df["oil_temperature_mean_60s"].abs(),
    )

    # -------------------------------------------------------------------------
    # 6. Cross-system residual relationships
    # -------------------------------------------------------------------------

    derived["thermal_residual_difference"] = (
        df["egt_residual"]
        - df["cht_residual"]
    )

    derived["thermal_residual_abs_sum"] = (
        df["egt_residual"].abs()
        + df["cht_residual"].abs()
    )

    derived["oil_thermal_residual_difference"] = (
        df["oil_temperature_residual"]
        - df["cht_residual"]
    )

    # -------------------------------------------------------------------------
    # 7. Trend relationships
    # -------------------------------------------------------------------------

    derived["thermal_trend_sum"] = (
        df["egt_rate"]
        + df["cht_rate"]
    )

    derived["thermal_trend_difference"] = (
        df["egt_rate"]
        - df["cht_rate"]
    )

    # Keep this relation local and deterministic.
    #
    # oil_pressure_rate
    #       vs
    # short rolling mean of oil_pressure_rate
    #
    # This avoids using a global future-looking quantity.
    rolling_oil_pressure_rate = (
        df["oil_pressure_rate"]
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    derived["pressure_trend_difference"] = (
        df["oil_pressure_rate"]
        - rolling_oil_pressure_rate
    )

    # -------------------------------------------------------------------------
    # 8. Electrical relationship
    # -------------------------------------------------------------------------

    derived["alternator_to_battery_ratio"] = safe_ratio(
        df["alternator_output_a"],
        df["battery_voltage_v"],
    )

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    derived = derived.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Final feature set
    # -------------------------------------------------------------------------

    final_features = (
        v2_features
        + list(derived.columns)
    )

    if len(final_features) != len(
        set(final_features)
    ):
        duplicates = pd.Series(
            final_features
        )[
            pd.Series(
                final_features
            ).duplicated()
        ].tolist()

        fail(
            "Duplicate features detected:\n"
            + str(duplicates)
        )

    # -------------------------------------------------------------------------
    # Leakage check
    # -------------------------------------------------------------------------

    leakage_features = []

    for feature in final_features:

        lower = feature.lower()

        if feature in TARGET_COLUMNS:
            leakage_features.append(feature)

        if feature in LABEL_COLUMNS:
            leakage_features.append(feature)

        if (
            "ground_truth" in lower
            or "target" in lower
            or "label_" in lower
            or "fault_severity" in lower
        ):
            leakage_features.append(feature)

    leakage_features = sorted(
        set(leakage_features)
    )

    if leakage_features:
        fail(
            "Potential leakage features detected:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in leakage_features
            )
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        f"\nV2 features        : {len(v2_features)}"
    )

    print(
        f"Derived V3 features: {len(derived.columns)}"
    )

    print(
        f"Final V3 features  : {len(final_features)}"
    )

    print("\nDerived features:")

    for index, feature in enumerate(
        derived.columns,
        start=1,
    ):
        print(
            f"{index:>2}. {feature}"
        )

    # -------------------------------------------------------------------------
    # Build output dataset
    # -------------------------------------------------------------------------

    output = df[
        [
            "timestamp",
            "engine_id",
            "uav_id",
            "flight_id",
            "fault_mode_ground_truth",
            "fault_active_label",
        ]
    ].copy()

    for feature in v2_features:
        output[feature] = df[feature]

    for feature in derived.columns:
        output[feature] = derived[feature]

    output = output[
        [
            "timestamp",
            "engine_id",
            "uav_id",
            "flight_id",
            "fault_mode_ground_truth",
            "fault_active_label",
            *final_features,
        ]
    ]

    # -------------------------------------------------------------------------
    # Validate values
    # -------------------------------------------------------------------------

    for feature in final_features:
        output[feature] = pd.to_numeric(
            output[feature],
            errors="coerce",
        )

    infinite_count = np.isinf(
        output[final_features]
        .to_numpy(
            dtype=float
        )
    ).sum()

    if infinite_count > 0:
        fail(
            f"Infinite values remain: {infinite_count}"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FEATURE_FILE.write_text(
        "\n".join(
            final_features
        ) + "\n",
        encoding="utf-8",
    )

    output.to_csv(
        OUTPUT_DATA_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Verify saved feature list
    # -------------------------------------------------------------------------

    saved_features = [
        line.strip()
        for line in OUTPUT_FEATURE_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if saved_features != final_features:
        fail(
            "Saved feature list does not match "
            "the final feature list."
        )

    if len(saved_features) != EXPECTED_V3_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_V3_FEATURE_COUNT} V3 features, "
            f"found {len(saved_features)}."
        )

    print()
    print("=" * 78)
    print("V3 FEATURE BUILD COMPLETE")
    print("=" * 78)

    print(
        f"\nFinal feature count: "
        f"{len(saved_features)}"
    )

    print(
        f"\nFeature list:\n"
        f"{OUTPUT_FEATURE_FILE}"
    )

    print(
        f"\nFeature dataset:\n"
        f"{OUTPUT_DATA_FILE}"
    )

    print(
        "\nLeakage check : PASSED"
    )

    print(
        "Verification  : PASSED"
    )

    print(
        "\nBUILD COMPLETE."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("V3 FEATURE BUILD FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)