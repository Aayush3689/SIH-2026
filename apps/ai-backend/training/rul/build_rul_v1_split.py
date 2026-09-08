from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "training/datasets/processed/rul_v1/rul_flight_level_v1.csv"
)

OUTPUT_DIR = Path(
    "training/datasets/processed/rul_v1/split"
)

TARGET = "rul_hours_remaining"


# ============================================================
# FEATURE POLICY
# ============================================================
#
# Keep a compact health-focused feature set.
#
# Excluded deliberately:
# - engine_id / uav_id / flight_id
# - sample_count
# - flight_duration_min
# - engine_flight_number / engine_total_flights
# - planned altitude
# - airspeed
# - ambient temperature
#
# Reason:
# 1. Avoid identity leakage
# 2. Avoid flight-duration shortcut
# 3. Reduce operating-context dependence
# 4. Keep feature/sample ratio reasonable
#
# ============================================================


FEATURE_COLUMNS = [
    # --------------------------------------------------------
    # Core engine health
    # --------------------------------------------------------
    "rpm_mean",
    "rpm_std",
    "rpm_last",

    "cht_c_mean",
    "cht_c_std",
    "cht_c_last",

    "egt_c_mean",
    "egt_c_std",
    "egt_c_last",

    "oil_pressure_kpa_mean",
    "oil_pressure_kpa_std",
    "oil_pressure_kpa_last",

    "oil_temperature_c_mean",
    "oil_temperature_c_std",
    "oil_temperature_c_last",

    "fuel_flow_lph_mean",
    "fuel_flow_lph_std",
    "fuel_flow_lph_last",

    "fuel_pressure_kpa_mean",
    "fuel_pressure_kpa_std",
    "fuel_pressure_kpa_last",

    "engine_vibration_mm_s_mean",
    "engine_vibration_mm_s_std",
    "engine_vibration_mm_s_last",

    # --------------------------------------------------------
    # Engine dynamics / health trend
    # --------------------------------------------------------
    "egt_rate_mean",
    "cht_rate_mean",
    "oil_pressure_rate_mean",
    "vibration_rate_mean",

    # --------------------------------------------------------
    # Physics residuals
    # --------------------------------------------------------
    "egt_residual_mean",
    "cht_residual_mean",
    "oil_temperature_residual_mean",

    # --------------------------------------------------------
    # Lifecycle / maintenance state
    # --------------------------------------------------------
    "total_cycles_at_start_last",
    "last_overhaul_days_ago_last",
    "last_major_maintenance_days_ago_last",
]


# ============================================================
# SPLIT POLICY
# ============================================================

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def chronological_split(group: pd.DataFrame):
    """
    Split one engine chronologically into:
        TRAIN -> VALIDATION -> TEST

    Later flights are always kept for later splits.
    """

    n = len(group)

    if n < 3:
        raise ValueError(
            f"Engine {group['engine_id'].iloc[0]} has only "
            f"{n} flights. At least 3 are required."
        )

    # Start with proportional sizes.
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))

    # Guarantee at least one test flight.
    n_test = n - n_train - n_val

    # Adjust if rounding consumed too many samples.
    while n_test < 1:
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError(
                f"Unable to create valid split for engine "
                f"{group['engine_id'].iloc[0]}"
            )

        n_test = n - n_train - n_val

    train = group.iloc[:n_train].copy()
    val = group.iloc[n_train:n_train + n_val].copy()
    test = group.iloc[n_train + n_val:].copy()

    return train, val, test


def validate_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
):

    section("SPLIT VALIDATION")

    # --------------------------------------------------------
    # No overlap
    # --------------------------------------------------------

    train_ids = set(train["flight_id"])
    val_ids = set(val["flight_id"])
    test_ids = set(test["flight_id"])

    assert train_ids.isdisjoint(val_ids), \
        "Train/Validation flight leakage detected."

    assert train_ids.isdisjoint(test_ids), \
        "Train/Test flight leakage detected."

    assert val_ids.isdisjoint(test_ids), \
        "Validation/Test flight leakage detected."

    # --------------------------------------------------------
    # Engine-wise chronological ordering
    # --------------------------------------------------------

    for engine_id in sorted(train["engine_id"].unique()):

        tr = train[
            train["engine_id"] == engine_id
        ]

        va = val[
            val["engine_id"] == engine_id
        ]

        te = test[
            test["engine_id"] == engine_id
        ]

        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            raise AssertionError(
                f"Engine {engine_id} missing one of train/val/test."
            )

        tr_last = tr["flight_start"].max()
        va_first = va["flight_start"].min()

        va_last = va["flight_start"].max()
        te_first = te["flight_start"].min()

        assert tr_last <= va_first, (
            f"Chronology violation for {engine_id}: "
            f"train ends {tr_last}, val starts {va_first}"
        )

        assert va_last <= te_first, (
            f"Chronology violation for {engine_id}: "
            f"validation ends {va_last}, test starts {te_first}"
        )

    print("No flight leakage             : PASSED")
    print("Engine-wise chronological     : PASSED")
    print("Every engine in all 3 splits  : PASSED")


def main():

    section("BUILD RUL V1 FEATURE SET + CHRONOLOGICAL SPLIT")

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns):,}")

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "engine_id",
        "flight_id",
        "flight_start",
        "flight_end",
        TARGET,
        *FEATURE_COLUMNS,
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df["flight_start"] = pd.to_datetime(
        df["flight_start"],
        errors="raise"
    )

    df["flight_end"] = pd.to_datetime(
        df["flight_end"],
        errors="raise"
    )

    df = df.sort_values(
        ["engine_id", "flight_start", "flight_id"]
    ).reset_index(drop=True)

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    section("FEATURE VALIDATION")

    print(
        f"Selected features : {len(FEATURE_COLUMNS)}"
    )

    missing_values = (
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values : {missing_values:,}"
    )

    if missing_values > 0:
        raise ValueError(
            "Selected features contain missing values."
        )

    # ========================================================
    # REMOVE CONSTANT SELECTED FEATURES
    # ========================================================

    constant_features = [
        col
        for col in FEATURE_COLUMNS
        if df[col].nunique(dropna=False) <= 1
    ]

    if constant_features:

        print("\nWARNING: Constant selected features:")

        for col in constant_features:
            print(f"  - {col}")

        raise ValueError(
            "Remove constant features before training."
        )

    print("Feature validation : PASSED")

    # ========================================================
    # ENGINE DISTRIBUTION
    # ========================================================

    section("ENGINE FLIGHT COUNTS")

    engine_counts = (
        df.groupby("engine_id")
        .size()
        .sort_index()
    )

    print(
        engine_counts.to_string()
    )

    if (engine_counts < 3).any():
        raise ValueError(
            "Every engine needs at least 3 flights."
        )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    section("ENGINE-WISE CHRONOLOGICAL SPLIT")

    train_parts = []
    val_parts = []
    test_parts = []

    split_rows = []

    for engine_id, group in df.groupby(
        "engine_id",
        sort=True
    ):

        group = group.sort_values(
            ["flight_start", "flight_id"]
        ).reset_index(drop=True)

        train, val, test = chronological_split(
            group
        )

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

        split_rows.append(
            {
                "engine_id": engine_id,
                "total_flights": len(group),
                "train_flights": len(train),
                "validation_flights": len(val),
                "test_flights": len(test),
                "train_first": train["flight_start"].min(),
                "train_last": train["flight_start"].max(),
                "validation_first": val["flight_start"].min(),
                "validation_last": val["flight_start"].max(),
                "test_first": test["flight_start"].min(),
                "test_last": test["flight_start"].max(),
            }
        )

        print(
            f"{engine_id}: "
            f"TRAIN={len(train):2d}, "
            f"VAL={len(val):2d}, "
            f"TEST={len(test):2d}"
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    val_df = pd.concat(
        val_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_split(
        train_df,
        val_df,
        test_df,
    )

    # ========================================================
    # TARGET DISTRIBUTION PER SPLIT
    # ========================================================

    section("TARGET DISTRIBUTION BY SPLIT")

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        print(
            f"\n{name}"
        )

        print(
            f"Flights : {len(split_df)}"
        )

        print(
            f"RUL min : {split_df[TARGET].min():.2f} h"
        )

        print(
            f"RUL max : {split_df[TARGET].max():.2f} h"
        )

        print(
            f"RUL mean: {split_df[TARGET].mean():.2f} h"
        )

        print(
            f"RUL med : {split_df[TARGET].median():.2f} h"
        )

        print(
            f"RUL=0   : "
            f"{(split_df[TARGET] == 0).sum()}"
        )

    # ========================================================
    # SELECT ONLY MODEL + METADATA COLUMNS
    # ========================================================

    metadata_columns = [
        "engine_id",
        "flight_id",
        "flight_start",
        "flight_end",
        TARGET,
    ]

    final_columns = (
        metadata_columns +
        FEATURE_COLUMNS
    )

    train_out = train_df[
        final_columns
    ].copy()

    val_out = val_df[
        final_columns
    ].copy()

    test_out = test_df[
        final_columns
    ].copy()

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = (
        OUTPUT_DIR /
        "rul_train_v1.csv"
    )

    val_path = (
        OUTPUT_DIR /
        "rul_validation_v1.csv"
    )

    test_path = (
        OUTPUT_DIR /
        "rul_test_v1.csv"
    )

    split_summary_path = (
        OUTPUT_DIR /
        "rul_split_summary.csv"
    )

    feature_list_path = (
        OUTPUT_DIR /
        "rul_v1_features.txt"
    )

    metadata_path = (
        OUTPUT_DIR /
        "rul_v1_split_metadata.json"
    )

    # ========================================================
    # SAVE DATA
    # ========================================================

    train_out.to_csv(
        train_path,
        index=False
    )

    val_out.to_csv(
        val_path,
        index=False
    )

    test_out.to_csv(
        test_path,
        index=False
    )

    split_summary = pd.DataFrame(
        split_rows
    )

    split_summary.to_csv(
        split_summary_path,
        index=False
    )

    with open(
        feature_list_path,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in FEATURE_COLUMNS:
            f.write(feature + "\n")

    metadata = {
        "dataset": "RUL V1",
        "unit_of_sample": "flight",
        "total_samples": int(len(df)),
        "train_samples": int(len(train_out)),
        "validation_samples": int(len(val_out)),
        "test_samples": int(len(test_out)),
        "engine_count": int(df["engine_id"].nunique()),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "target": TARGET,
        "split_method": (
            "engine-wise chronological "
            "60/20/20 approximately"
        ),
        "engine_id_used_as_feature": False,
        "flight_id_used_as_feature": False,
        "flight_duration_used_as_feature": False,
        "sample_count_used_as_feature": False,
        "feature_policy": (
            "health-focused + dynamics + "
            "physics residuals + lifecycle state"
        ),
        "features": FEATURE_COLUMNS,
        "excluded_features": [
            "engine_id",
            "uav_id",
            "flight_id",
            "sample_count",
            "flight_duration_min",
            "engine_flight_number",
            "engine_total_flights",
            "planned_altitude_*",
            "airspeed_*",
            "ambient_temp_*",
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    section("FINAL OUTPUT VALIDATION")

    print(
        f"Train samples      : {len(train_out)}"
    )

    print(
        f"Validation samples : {len(val_out)}"
    )

    print(
        f"Test samples       : {len(test_out)}"
    )

    print(
        f"Features           : {len(FEATURE_COLUMNS)}"
    )

    assert (
        len(train_out)
        + len(val_out)
        + len(test_out)
        == len(df)
    )

    assert (
        len(set(train_out["flight_id"]) &
            set(val_out["flight_id"]))
        == 0
    )

    assert (
        len(set(train_out["flight_id"]) &
            set(test_out["flight_id"]))
        == 0
    )

    assert (
        len(set(val_out["flight_id"]) &
            set(test_out["flight_id"]))
        == 0
    )

    print("Row count conservation : PASSED")
    print("Flight leakage         : PASSED")
    print("Feature validation     : PASSED")

    # ========================================================
    # OUTPUT
    # ========================================================

    section("OUTPUT")

    print(f"Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {split_summary_path}")
    print(f"  {feature_list_path}")
    print(f"  {metadata_path}")

    print("\nRUL V1 SPLIT BUILD COMPLETE.")


if __name__ == "__main__":
    main()