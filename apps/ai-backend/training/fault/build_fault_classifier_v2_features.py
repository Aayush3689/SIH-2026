from pathlib import Path
import sys

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

OUTPUT_FILE = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "fault_classifier_v2_features.txt"
)


# =============================================================================
# V2 HEALTH-FOCUSED FEATURES
# =============================================================================

V2_FEATURES = [
    # -------------------------------------------------------------------------
    # Cylinder Head Temperature
    # -------------------------------------------------------------------------
    "cht_c",
    "cht_mean_60s",
    "cht_rate",
    "cht_residual",
    "cht_std_60s",

    # -------------------------------------------------------------------------
    # Exhaust Gas Temperature
    # -------------------------------------------------------------------------
    "egt_c",
    "egt_mean_60s",
    "egt_rate",
    "egt_residual",
    "egt_std_60s",

    # -------------------------------------------------------------------------
    # Oil Pressure
    # -------------------------------------------------------------------------
    "oil_pressure_kpa",
    "oil_pressure_mean_60s",
    "oil_pressure_rate",
    "oil_pressure_std_60s",

    # -------------------------------------------------------------------------
    # Oil Temperature
    # -------------------------------------------------------------------------
    "oil_temperature_c",
    "oil_temperature_mean_60s",
    "oil_temperature_residual",
    "oil_temperature_std_60s",

    # -------------------------------------------------------------------------
    # Fuel System
    # -------------------------------------------------------------------------
    "fuel_flow_lph",
    "fuel_pressure_kpa",

    # -------------------------------------------------------------------------
    # Vibration
    # -------------------------------------------------------------------------
    "engine_vibration_mm_s",
    "vibration_mean_30s",
    "vibration_rate",
    "vibration_std_30s",

    # -------------------------------------------------------------------------
    # Electrical / Injection
    # -------------------------------------------------------------------------
    "battery_voltage_v",
    "alternator_output_a",
    "injection_timing_deg_btdc",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:

    print("=" * 78)
    print("BUILD FAULT CLASSIFIER V2 FEATURE SET")
    print("=" * 78)

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows   : {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # Validate all requested features
    # -------------------------------------------------------------------------

    missing = [
        feature
        for feature in V2_FEATURES
        if feature not in df.columns
    ]

    if missing:
        fail(
            "V2 features missing from fault_features.csv:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    # -------------------------------------------------------------------------
    # Validate numeric
    # -------------------------------------------------------------------------

    non_numeric = [
        feature
        for feature in V2_FEATURES
        if not pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    if non_numeric:
        fail(
            "Non-numeric V2 features detected:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in non_numeric
            )
        )

    # -------------------------------------------------------------------------
    # Check duplicates
    # -------------------------------------------------------------------------

    if len(V2_FEATURES) != len(set(V2_FEATURES)):
        fail("Duplicate feature names found in V2 feature list.")

    # -------------------------------------------------------------------------
    # Check infinite values
    # -------------------------------------------------------------------------

    numeric = df[V2_FEATURES]

    if numeric.isin([float("inf"), float("-inf")]).any().any():
        fail(
            "Infinite values detected in V2 features."
        )

    # -------------------------------------------------------------------------
    # Print final list
    # -------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("V2 FEATURE SET")
    print("=" * 78)

    for index, feature in enumerate(
        V2_FEATURES,
        start=1,
    ):
        print(
            f"{index:>2}. {feature}"
        )

    print(
        f"\nV2 FEATURE COUNT: {len(V2_FEATURES)}"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(V2_FEATURES) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    saved = [
        line.strip()
        for line in OUTPUT_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if saved != V2_FEATURES:
        fail(
            "Saved feature list does not match V2 feature list."
        )

    print("\n" + "=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Feature list saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nSaved features: {len(saved)}"
    )

    print("Verification   : PASSED")
    print("\nBUILD COMPLETE.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("FEATURE BUILD FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)