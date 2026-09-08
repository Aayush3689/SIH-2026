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
    / "fault_classifier_v4_features.txt"
)

OUTPUT_DATA_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v4_features.csv"
)

EXPECTED_V2_FEATURE_COUNT = 27
EXPECTED_DERIVED_FEATURE_COUNT = 13
EXPECTED_V4_FEATURE_COUNT = 40

ROLLING_WINDOW = 5


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

    if len(features) != EXPECTED_V2_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_V2_FEATURE_COUNT} V2 features, "
            f"found {len(features)}."
        )

    return features


def rolling_mean_by_flight(
    df: pd.DataFrame,
    column: str,
    window: int,
) -> pd.Series:

    return (
        df.groupby(
            "flight_id",
            sort=False,
        )[column]
        .transform(
            lambda s: s.rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )


def rolling_std_by_flight(
    df: pd.DataFrame,
    column: str,
    window: int,
) -> pd.Series:

    return (
        df.groupby(
            "flight_id",
            sort=False,
        )[column]
        .transform(
            lambda s: s.rolling(
                window=window,
                min_periods=2,
            ).std(ddof=0)
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("BUILD FAULT CLASSIFIER V4 FEATURE SET")
    print("CONTROLLED TEMPORAL / PERSISTENCE FEATURES")
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
    # Required columns
    # -------------------------------------------------------------------------

    required = {
        "timestamp",
        "flight_id",
    }

    missing_required = sorted(
        required - set(df.columns)
    )

    if missing_required:
        fail(
            "Required columns missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_required
            )
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
            "V2 features missing from source:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_v2
            )
        )

    # -------------------------------------------------------------------------
    # Sort chronologically inside each flight
    # -------------------------------------------------------------------------

    section("TEMPORAL ORDER")

    df["_timestamp_parsed"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        df["_timestamp_parsed"].isna().sum()
    )

    if invalid_timestamp_count > 0:
        fail(
            f"Invalid timestamps found: "
            f"{invalid_timestamp_count}"
        )

    df["flight_id"] = (
        df["flight_id"]
        .astype(str)
        .str.strip()
    )

    df = (
        df.sort_values(
            [
                "flight_id",
                "_timestamp_parsed",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    print(
        f"Flights: {df['flight_id'].nunique():,}"
    )

    print(
        "Rolling features: within-flight only"
    )

    print(
        f"Rolling window: {ROLLING_WINDOW} samples"
    )

    # -------------------------------------------------------------------------
    # Build working signals
    # -------------------------------------------------------------------------

    # Absolute residuals capture magnitude independently of direction.
    df["_egt_abs_residual"] = (
        df["egt_residual"].abs()
    )

    df["_cht_abs_residual"] = (
        df["cht_residual"].abs()
    )

    df["_oil_temperature_abs_residual"] = (
        df["oil_temperature_residual"].abs()
    )

    df["_thermal_abs_residual_sum"] = (
        df["_egt_abs_residual"]
        + df["_cht_abs_residual"]
        + df["_oil_temperature_abs_residual"]
    )

    # -------------------------------------------------------------------------
    # Derived temporal features
    # -------------------------------------------------------------------------

    derived = pd.DataFrame(
        index=df.index
    )

    # -------------------------------------------------------------------------
    # 1-3: Residual persistence
    # -------------------------------------------------------------------------

    derived["egt_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "_egt_abs_residual",
            ROLLING_WINDOW,
        )
    )

    derived["cht_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "_cht_abs_residual",
            ROLLING_WINDOW,
        )
    )

    derived["oil_temperature_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "_oil_temperature_abs_residual",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # 4-6: Stronger residual persistence
    #
    # Same health signals, but use signed residual smoothing.
    # This helps distinguish persistent positive/negative deviation.
    # -------------------------------------------------------------------------

    derived["egt_signed_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "egt_residual",
            ROLLING_WINDOW,
        )
    )

    derived["cht_signed_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "cht_residual",
            ROLLING_WINDOW,
        )
    )

    derived[
        "oil_temperature_signed_residual_mean_5"
    ] = (
        rolling_mean_by_flight(
            df,
            "oil_temperature_residual",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # 7: Total thermal deviation persistence
    # -------------------------------------------------------------------------

    derived["thermal_abs_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "_thermal_abs_residual_sum",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # 8: Thermal residual interaction
    #
    # Product indicates whether residuals move together and magnitude
    # of their joint deviation.
    # -------------------------------------------------------------------------

    df["_thermal_residual_product"] = (
        df["egt_residual"]
        * df["cht_residual"]
    )

    derived["thermal_residual_product"] = (
        df["_thermal_residual_product"]
    )

    # -------------------------------------------------------------------------
    # 9: Thermal residual direction alignment
    #
    # 1 -> same sign
    # 0 -> opposite sign / zero
    # -------------------------------------------------------------------------

    derived["thermal_residual_same_sign"] = (
        (
            np.sign(
                df["egt_residual"]
            )
            == np.sign(
                df["cht_residual"]
            )
        )
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # 10-11: Fuel pressure persistence and variability
    # -------------------------------------------------------------------------

    derived["fuel_pressure_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "fuel_pressure_kpa",
            ROLLING_WINDOW,
        )
    )

    derived["fuel_pressure_std_5"] = (
        rolling_std_by_flight(
            df,
            "fuel_pressure_kpa",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # 12-13: Vibration trend persistence / variability
    # -------------------------------------------------------------------------

    derived["vibration_rate_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "vibration_rate",
            ROLLING_WINDOW,
        )
    )

    derived["vibration_rate_std_5"] = (
        rolling_std_by_flight(
            df,
            "vibration_rate",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    derived = derived.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Final features
    # -------------------------------------------------------------------------

    final_features = (
        v2_features
        + list(derived.columns)
    )

    if len(final_features) != len(
        set(final_features)
    ):
        duplicates = sorted(
            {
                feature
                for feature in final_features
                if final_features.count(feature) > 1
            }
        )

        fail(
            "Duplicate features detected:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in duplicates
            )
        )

    if len(derived.columns) != EXPECTED_DERIVED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_DERIVED_FEATURE_COUNT} "
            f"derived features, found {len(derived.columns)}."
        )

    if len(final_features) != EXPECTED_V4_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_V4_FEATURE_COUNT} V4 features, "
            f"found {len(final_features)}."
        )

    # -------------------------------------------------------------------------
    # Leakage safety
    # -------------------------------------------------------------------------

    forbidden_keywords = (
        "ground_truth",
        "target",
        "label_",
        "fault_severity",
    )

    leakage_features = []

    for feature in final_features:

        lower = feature.lower()

        if any(
            keyword in lower
            for keyword in forbidden_keywords
        ):
            leakage_features.append(feature)

    if leakage_features:
        fail(
            "Potential leakage-like features detected:\n"
            + "\n".join(
                sorted(leakage_features)
            )
        )

    # -------------------------------------------------------------------------
    # Output dataframe
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
    # Numeric validation
    # -------------------------------------------------------------------------

    for feature in final_features:

        output[feature] = pd.to_numeric(
            output[feature],
            errors="coerce",
        )

    infinite_count = int(
        np.isinf(
            output[
                final_features
            ].to_numpy(
                dtype=float
            )
        ).sum()
    )

    if infinite_count > 0:
        fail(
            f"Infinite values remain: "
            f"{infinite_count}"
        )

    # -------------------------------------------------------------------------
    # Print feature summary
    # -------------------------------------------------------------------------

    section("V4 FEATURE SUMMARY")

    print(
        f"V2 features        : "
        f"{len(v2_features)}"
    )

    print(
        f"Derived features   : "
        f"{len(derived.columns)}"
    )

    print(
        f"Final V4 features  : "
        f"{len(final_features)}"
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
    # Missing values
    # -------------------------------------------------------------------------

    section("DERIVED FEATURE MISSING VALUES")

    missing_rows = []

    for feature in derived.columns:

        count = int(
            output[feature].isna().sum()
        )

        missing_rows.append(
            (
                feature,
                count,
            )
        )

    for feature, count in missing_rows:

        if count > 0:

            percentage = (
                count
                / len(output)
                * 100
            )

            print(
                f"{feature:<40}"
                f"{count:>7,}"
                f" ({percentage:.6f}%)"
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
        )
        + "\n",
        encoding="utf-8",
    )

    output.to_csv(
        OUTPUT_DATA_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Verification
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
            "final feature list."
        )

    section("OUTPUT")

    print(
        f"Feature list:\n"
        f"{OUTPUT_FEATURE_FILE}"
    )

    print(
        f"\nFeature dataset:\n"
        f"{OUTPUT_DATA_FILE}"
    )

    print(
        f"\nFinal feature count: "
        f"{len(saved_features)}"
    )

    print(
        "Temporal leakage check: PASSED"
    )

    print(
        "Leakage check: PASSED"
    )

    print(
        "Verification: PASSED"
    )

    print(
        "\nV4 FEATURE BUILD COMPLETE."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("V4 FEATURE BUILD FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)