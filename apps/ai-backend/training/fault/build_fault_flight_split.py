"""
FINAL FAULT CLASSIFICATION FLIGHT SPLIT
=======================================

Purpose
-------
Create the final flight-wise chronological split for Fault Classification.

Why not engine-wise?
--------------------
Each fault class is associated with exactly one engine in the current
dataset. Therefore an unseen-engine split would also create unseen fault
classes in validation/test.

So for Fault Classification we split at FLIGHT level.

Strategy
--------
For EACH fault class independently:

    4 active flights:
        Flight 1 -> TRAIN
        Flight 2 -> TRAIN
        Flight 3 -> VALIDATION
        Flight 4 -> TEST

    3 active flights:
        Flight 1 -> TRAIN
        Flight 2 -> VALIDATION
        Flight 3 -> TEST

This guarantees:
    - every fault class is present in TRAIN
    - every fault class is present in VALIDATION
    - every fault class is present in TEST
    - chronological ordering is preserved within each class
    - no flight leakage

IMPORTANT
---------
Only active-fault flights are used:

    fault_active_label == 1

Target:
    fault_mode_ground_truth

engine_id:
    used only for grouping / auditing
    NEVER use as a model feature

This script DOES NOT train a model.

Output:
    training/datasets/processed/fault_flight_split_final.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

# File:
# ai-backend/training/fault/build_fault_flight_split.py
#
# parents[0] -> training/fault
# parents[1] -> training
#
TRAINING_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = (
    TRAINING_DIR
    / "datasets"
    / "raw"
    / "dt_engine_telemetry_full.csv"
)

OUTPUT_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fault_flight_split_final.csv"
)


# =====================================================================
# CONSTANTS
# =====================================================================

ACTIVE_LABEL = 1

EXPECTED_FAULT_CLASSES = [
    "misfire",
    "injector_abnormality",
    "coking_degradation",
    "lubrication_issue",
    "sensor_drift",
    "combustion_instability",
    "overheating_trend",
    "abnormal_vibration",
]


# =====================================================================
# HELPERS
# =====================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


# =====================================================================
# LOAD
# =====================================================================

def load_raw_data() -> pd.DataFrame:

    if not RAW_PATH.exists():
        fail(
            "Raw dataset not found:\n"
            f"{RAW_PATH}"
        )

    print(
        "Loading raw telemetry..."
    )

    df = pd.read_csv(
        RAW_PATH
    )

    required = {
        "engine_id",
        "uav_id",
        "flight_id",
        "timestamp",
        "fault_active_label",
        "fault_mode_ground_truth",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        fail(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        fail(
            "Invalid timestamps found."
        )

    df["fault_active_label"] = (
        pd.to_numeric(
            df["fault_active_label"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["fault_mode_ground_truth"] = (
        df[
            "fault_mode_ground_truth"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# =====================================================================
# ACTIVE FAULT POPULATION
# =====================================================================

def build_active_population(
    df: pd.DataFrame,
) -> pd.DataFrame:

    active = (
        df[
            df["fault_active_label"]
            == ACTIVE_LABEL
        ]
        .copy()
    )

    if active.empty:
        fail(
            "No active-fault telemetry rows found."
        )

    # A row marked active must have a real fault mode.
    invalid_none = (
        active[
            "fault_mode_ground_truth"
        ]
        == "none"
    )

    if invalid_none.any():

        count = int(
            invalid_none.sum()
        )

        fail(
            "Found active-fault rows with "
            f"fault_mode_ground_truth='none': "
            f"{count}"
        )

    return active


# =====================================================================
# BUILD ACTIVE FLIGHT TABLE
# =====================================================================

def build_active_flight_table(
    active: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per active flight.

    Every active flight must have exactly one fault mode.
    """

    rows = []

    grouped = active.groupby(
        [
            "engine_id",
            "flight_id",
        ],
        sort=False,
    )

    for (
        engine_id,
        flight_id,
    ), group in grouped:

        modes = (
            group[
                "fault_mode_ground_truth"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if len(modes) != 1:

            fail(
                "Multiple fault modes found "
                "inside one active flight:\n"
                f"Engine: {engine_id}\n"
                f"Flight: {flight_id}\n"
                f"Modes: {modes}"
            )

        rows.append(
            {
                "engine_id":
                    engine_id,

                "flight_id":
                    flight_id,

                "fault_mode":
                    modes[0],

                "first_timestamp":
                    group[
                        "timestamp"
                    ].min(),

                "last_timestamp":
                    group[
                        "timestamp"
                    ].max(),

                "active_rows":
                    len(group),
            }
        )

    flights = pd.DataFrame(
        rows
    )

    if flights.empty:
        fail(
            "Active flight table is empty."
        )

    flights = (
        flights
        .sort_values(
            [
                "fault_mode",
                "first_timestamp",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    return flights


# =====================================================================
# SPLIT ONE FAULT CLASS
# =====================================================================

def split_fault_class(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split one fault class chronologically.

    For n >= 3:

        first ~ ceil((n - 2) / 2) flights -> TRAIN
        next flight                      -> VALIDATION
        last flight                      -> TEST

    For the current dataset n is 3 or 4, therefore:

        n=3 -> TRAIN / VAL / TEST
        n=4 -> TRAIN / TRAIN / VAL / TEST
    """

    group = (
        group
        .sort_values(
            [
                "first_timestamp",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    n = len(group)

    if n < 3:

        fail(
            f"Fault class '{group['fault_mode'].iloc[0]}' "
            f"has only {n} active flights. "
            "At least 3 are required."
        )

    # -------------------------------------------------------------
    # Fixed rule for current dataset
    # -------------------------------------------------------------

    if n == 3:

        train_count = 1

    else:

        # Everything except the final validation and test flights
        train_count = n - 2

    split_values = (
        ["TRAIN"] * train_count
        + ["VALIDATION"]
        + ["TEST"]
    )

    if len(split_values) != n:

        fail(
            "Internal split-size error for "
            f"fault class {group['fault_mode'].iloc[0]}"
        )

    group["split"] = split_values

    group[
        "flight_order_within_fault_class"
    ] = range(
        1,
        n + 1,
    )

    group[
        "total_active_flights_fault_class"
    ] = n

    return group


# =====================================================================
# BUILD FINAL SPLIT
# =====================================================================

def build_final_split(
    active_flights: pd.DataFrame,
) -> pd.DataFrame:

    split_groups = []

    for (
        fault_mode,
        group,
    ) in active_flights.groupby(
        "fault_mode",
        sort=False,
    ):

        split_group = split_fault_class(
            group
        )

        split_groups.append(
            split_group
        )

    result = pd.concat(
        split_groups,
        ignore_index=True,
    )

    # Chronological global ordering
    result = (
        result
        .sort_values(
            [
                "first_timestamp",
                "fault_mode",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    return result


# =====================================================================
# VALIDATE CLASS COVERAGE
# =====================================================================

def validate_class_coverage(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("CLASS COVERAGE CHECK")
    print("=" * 78)

    class_split_table = (
        pd.crosstab(
            split_df[
                "fault_mode"
            ],
            split_df[
                "split"
            ],
        )
        .reindex(
            index=EXPECTED_FAULT_CLASSES,
            columns=[
                "TRAIN",
                "VALIDATION",
                "TEST",
            ],
            fill_value=0,
        )
    )

    print()
    print(
        class_split_table.to_string()
    )

    missing_train = (
        class_split_table.index[
            class_split_table[
                "TRAIN"
            ] == 0
        ]
        .tolist()
    )

    missing_val = (
        class_split_table.index[
            class_split_table[
                "VALIDATION"
            ] == 0
        ]
        .tolist()
    )

    missing_test = (
        class_split_table.index[
            class_split_table[
                "TEST"
            ] == 0
        ]
        .tolist()
    )

    print()

    if missing_train:

        fail(
            "Fault classes missing from TRAIN:\n"
            f"{missing_train}"
        )

    if missing_val:

        fail(
            "Fault classes missing from VALIDATION:\n"
            f"{missing_val}"
        )

    if missing_test:

        fail(
            "Fault classes missing from TEST:\n"
            f"{missing_test}"
        )

    print(
        "PASS: All 8 fault classes are present "
        "in TRAIN / VALIDATION / TEST."
    )


# =====================================================================
# CHECK FLIGHT LEAKAGE
# =====================================================================

def validate_flight_leakage(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("FLIGHT LEAKAGE CHECK")
    print("=" * 78)

    sets = {}

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        sets[split_name] = set(
            split_df[
                split_df["split"]
                == split_name
            ]["flight_id"]
        )

    train_val = (
        sets["TRAIN"]
        & sets["VALIDATION"]
    )

    train_test = (
        sets["TRAIN"]
        & sets["TEST"]
    )

    val_test = (
        sets["VALIDATION"]
        & sets["TEST"]
    )

    print(
        f"TRAIN ∩ VALIDATION: "
        f"{len(train_val)}"
    )

    print(
        f"TRAIN ∩ TEST      : "
        f"{len(train_test)}"
    )

    print(
        f"VALIDATION ∩ TEST : "
        f"{len(val_test)}"
    )

    if (
        train_val
        or train_test
        or val_test
    ):

        fail(
            "Flight leakage detected."
        )

    print()
    print(
        "PASS: No flight appears in multiple splits."
    )


# =====================================================================
# CHECK CHRONOLOGY
# =====================================================================

def validate_chronology(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("CHRONOLOGICAL ORDER CHECK")
    print("=" * 78)

    problems = []

    split_rank = {
        "TRAIN": 0,
        "VALIDATION": 1,
        "TEST": 2,
    }

    for fault_mode, group in split_df.groupby(
        "fault_mode",
        sort=False,
    ):

        # ---------------------------------------------------------
        # IMPORTANT:
        # Use the SAME ordering rule that was used while creating
        # the split:
        #
        #   first_timestamp
        #   flight_id
        #
        # Some synthetic flights have the same first_timestamp,
        # so timestamp-only comparison is too strict.
        # ---------------------------------------------------------

        group = (
            group
            .sort_values(
                [
                    "first_timestamp",
                    "flight_id",
                ]
            )
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # Check split order
        # ---------------------------------------------------------

        ranks = [
            split_rank[split]
            for split in group["split"]
        ]

        if ranks != sorted(ranks):

            problems.append(
                f"{fault_mode}: split order"
            )

        # ---------------------------------------------------------
        # Check pair ordering using the same
        # timestamp + flight_id ordering.
        # ---------------------------------------------------------

        train_group = group[
            group["split"] == "TRAIN"
        ]

        val_group = group[
            group["split"] == "VALIDATION"
        ]

        test_group = group[
            group["split"] == "TEST"
        ]

        # ---------------------------------------------------------
        # TRAIN -> VALIDATION
        # ---------------------------------------------------------

        if (
            not train_group.empty
            and not val_group.empty
        ):

            train_last = (
                train_group.iloc[-1]
            )

            val_first = (
                val_group.iloc[0]
            )

            train_key = (
                pd.Timestamp(
                    train_last[
                        "first_timestamp"
                    ]
                ),
                str(
                    train_last[
                        "flight_id"
                    ]
                ),
            )

            val_key = (
                pd.Timestamp(
                    val_first[
                        "first_timestamp"
                    ]
                ),
                str(
                    val_first[
                        "flight_id"
                    ]
                ),
            )

            if train_key >= val_key:

                problems.append(
                    f"{fault_mode}: "
                    "TRAIN/VALIDATION chronology"
                )

        # ---------------------------------------------------------
        # VALIDATION -> TEST
        # ---------------------------------------------------------

        if (
            not val_group.empty
            and not test_group.empty
        ):

            val_last = (
                val_group.iloc[-1]
            )

            test_first = (
                test_group.iloc[0]
            )

            val_key = (
                pd.Timestamp(
                    val_last[
                        "first_timestamp"
                    ]
                ),
                str(
                    val_last[
                        "flight_id"
                    ]
                ),
            )

            test_key = (
                pd.Timestamp(
                    test_first[
                        "first_timestamp"
                    ]
                ),
                str(
                    test_first[
                        "flight_id"
                    ]
                ),
            )

            if val_key >= test_key:

                problems.append(
                    f"{fault_mode}: "
                    "VALIDATION/TEST chronology"
                )

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    if problems:

        fail(
            "Chronology validation failed:\n"
            f"{problems}"
        )

    print(
        "PASS: Split order is chronological "
        "within every fault class."
    )

    print(
        "Ordering rule: first_timestamp + flight_id"
    )


# =====================================================================
# PRINT SUMMARY
# =====================================================================

def print_summary(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("FINAL SPLIT SUMMARY")
    print("=" * 78)

    print()

    flight_counts = (
        split_df[
            "split"
        ]
        .value_counts()
        .reindex(
            [
                "TRAIN",
                "VALIDATION",
                "TEST",
            ],
            fill_value=0,
        )
    )

    print(
        "Flights:"
    )

    print(
        flight_counts.to_string()
    )

    print()

    row_counts = (
        split_df
        .groupby(
            "split"
        )[
            "active_rows"
        ]
        .sum()
        .reindex(
            [
                "TRAIN",
                "VALIDATION",
                "TEST",
            ]
        )
    )

    print(
        "Active telemetry rows:"
    )

    print(
        row_counts.to_string()
    )

    print()

    print(
        "Per-class split:"
    )

    class_table = (
        pd.crosstab(
            split_df[
                "fault_mode"
            ],
            split_df[
                "split"
            ],
        )
        .reindex(
            columns=[
                "TRAIN",
                "VALIDATION",
                "TEST",
            ],
            fill_value=0,
        )
    )

    print(
        class_table.to_string()
    )

    print()

    # -------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------

    print(
        "Split time ranges:"
    )

    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        part = split_df[
            split_df["split"]
            == split_name
        ]

        print(
            f"  {split_name:10s}: "
            f"{part['first_timestamp'].min()} "
            f"→ "
            f"{part['last_timestamp'].max()}"
        )


# =====================================================================
# SAVE
# =====================================================================

def save_split(
    split_df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "engine_id",
        "flight_id",
        "fault_mode",
        "first_timestamp",
        "last_timestamp",
        "active_rows",
        "flight_order_within_fault_class",
        "total_active_flights_fault_class",
        "split",
    ]

    split_df[
        columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=" * 78
    )

    print(
        f"Saved final split:\n"
        f"{OUTPUT_PATH}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 78)
    print(
        "BUILD FINAL FAULT CLASSIFICATION FLIGHT SPLIT"
    )
    print("=" * 78)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    df = load_raw_data()

    # -------------------------------------------------------------
    # Active population
    # -------------------------------------------------------------

    active = build_active_population(
        df
    )

    print()
    print(
        f"Active-fault rows: "
        f"{len(active):,}"
    )

    # -------------------------------------------------------------
    # Active flights
    # -------------------------------------------------------------

    active_flights = (
        build_active_flight_table(
            active
        )
    )

    print(
        f"Active-fault flights: "
        f"{len(active_flights):,}"
    )

    print(
        f"Fault classes: "
        f"{active_flights['fault_mode'].nunique()}"
    )

    # -------------------------------------------------------------
    # Final split
    # -------------------------------------------------------------

    split_df = build_final_split(
        active_flights
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    validate_class_coverage(
        split_df
    )

    validate_flight_leakage(
        split_df
    )

    validate_chronology(
        split_df
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print_summary(
        split_df
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    save_split(
        split_df
    )

    # -------------------------------------------------------------
    # Final message
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL SPLIT READY")
    print("=" * 78)

    print(
        """
Next step:

    fault_features.csv
            +
    fault_flight_split_final.csv
            ↓
    leakage-safe feature selection
            ↓
    Fault Classifier training

engine_id will be used only for grouping/auditing,
NOT as a model feature.
"""
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()