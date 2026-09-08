"""
FAULT CLASSIFICATION TRAINING POPULATION ANALYSIS
=================================================

Purpose
-------
Analyze the population that will be used for Fault Classification.

Why?
----
The audit showed:

    - Each fault class is associated with one engine.
    - Therefore engine-wise holdout cannot test unseen fault classes.
    - engine_id must NOT be a model feature.

For the classifier we therefore need to study:

    fault_active_label == 1

and perform a flight-wise evaluation rather than an
unseen-engine class split.

This script DOES NOT train a model.

It produces:

    1. Active-fault row distribution
    2. Active-fault flight distribution
    3. Fault mode -> engine mapping
    4. First/last active fault flight
    5. Per-engine active-fault flight timeline
    6. Candidate chronological flight split:
           TRAIN / VALIDATION / TEST
    7. Checks for leakage and class coverage

Outputs
-------
training/datasets/processed/
    fault_active_training_population.csv

training/datasets/processed/
    fault_flight_split_plan.csv
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

# File:
# ai-backend/training/fault/analyze_fault_training_population.py
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

POPULATION_OUTPUT = (
    OUTPUT_DIR
    / "fault_active_training_population.csv"
)

SPLIT_OUTPUT = (
    OUTPUT_DIR
    / "fault_flight_split_plan.csv"
)


# =====================================================================
# CONFIG
# =====================================================================

ACTIVE_LABEL = 1

# Chronological flight-wise split.
#
# Roughly:
#   70% train
#   15% validation
#   15% test
#
# These are only CANDIDATE split ratios.
# We will inspect the generated split before using it for training.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


EXPECTED_FAULT_MODES = {
    "misfire",
    "injector_abnormality",
    "coking_degradation",
    "lubrication_issue",
    "sensor_drift",
    "combustion_instability",
    "overheating_trend",
    "abnormal_vibration",
}


# =====================================================================
# HELPERS
# =====================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


# =====================================================================
# LOAD DATA
# =====================================================================

def load_data() -> pd.DataFrame:

    if not RAW_PATH.exists():
        fail(
            "Raw dataset not found:\n"
            f"{RAW_PATH}"
        )

    print(
        "Loading dataset..."
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
        .astype(int)
    )

    df["fault_mode_ground_truth"] = (
        df["fault_mode_ground_truth"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# =====================================================================
# ACTIVE-FAULT POPULATION
# =====================================================================

def build_active_population(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only telemetry rows where raw fault_active_label == 1.
    """

    active = (
        df[
            df["fault_active_label"]
            == ACTIVE_LABEL
        ]
        .copy()
    )

    if active.empty:
        fail(
            "No active-fault rows found."
        )

    return active


# =====================================================================
# BASIC SUMMARY
# =====================================================================

def print_basic_summary(
    df: pd.DataFrame,
    active: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("ACTIVE-FAULT TRAINING POPULATION")
    print("=" * 78)

    print(
        f"Total telemetry rows : "
        f"{len(df):,}"
    )

    print(
        f"Active-fault rows    : "
        f"{len(active):,}"
    )

    print(
        f"Active-fault ratio   : "
        f"{len(active) / len(df):.2%}"
    )

    print(
        f"Active-fault engines : "
        f"{active['engine_id'].nunique()}"
    )

    print(
        f"Active-fault flights : "
        f"{active['flight_id'].nunique()}"
    )


# =====================================================================
# FAULT MODE DISTRIBUTION
# =====================================================================

def fault_mode_distribution(
    active: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 78)
    print("ACTIVE-FAULT DISTRIBUTION BY FAULT MODE")
    print("=" * 78)

    distribution = (
        active[
            "fault_mode_ground_truth"
        ]
        .value_counts()
        .rename_axis(
            "fault_mode"
        )
        .reset_index(
            name="active_rows"
        )
    )

    distribution[
        "percentage"
    ] = (
        distribution[
            "active_rows"
        ]
        / len(active)
        * 100
    )

    print()

    print(
        f"{'FAULT MODE':35s}"
        f"{'ROWS':>12s}"
        f"{'PERCENT':>12s}"
    )

    print("-" * 59)

    for _, row in distribution.iterrows():

        print(
            f"{row['fault_mode']:35s}"
            f"{int(row['active_rows']):12,d}"
            f"{row['percentage']:11.2f}%"
        )

    # -------------------------------------------------------------
    # Validate expected classes
    # -------------------------------------------------------------

    observed = set(
        distribution[
            "fault_mode"
        ]
    )

    missing_classes = (
        EXPECTED_FAULT_MODES
        - observed
    )

    unexpected_classes = (
        observed
        - EXPECTED_FAULT_MODES
    )

    print()

    if missing_classes:

        print(
            "WARNING - Missing expected fault classes:"
        )

        for mode in sorted(
            missing_classes
        ):

            print(
                f"  - {mode}"
            )

    else:

        print(
            "All 8 expected fault classes are present."
        )

    if unexpected_classes:

        print(
            "WARNING - Unexpected fault classes:"
        )

        for mode in sorted(
            unexpected_classes
        ):

            print(
                f"  - {mode}"
            )

    return distribution


# =====================================================================
# FAULT MODE -> ENGINE MAPPING
# =====================================================================

def fault_engine_mapping(
    active: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 78)
    print("FAULT MODE -> ENGINE MAPPING")
    print("=" * 78)

    mapping = (
        active[
            [
                "fault_mode_ground_truth",
                "engine_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "fault_mode_ground_truth",
                "engine_id",
            ]
        )
    )

    counts = (
        mapping
        .groupby(
            "fault_mode_ground_truth"
        )
        .size()
    )

    print()

    for fault_mode, group in (
        mapping.groupby(
            "fault_mode_ground_truth"
        )
    ):

        engines = (
            sorted(
                group[
                    "engine_id"
                ]
                .unique()
                .tolist()
            )
        )

        print(
            f"{fault_mode:35s}"
            f"→ {engines}"
        )

    # -------------------------------------------------------------
    # Strong shortcut warning
    # -------------------------------------------------------------

    print()

    one_engine_classes = int(
        (
            counts
            == 1
        ).sum()
    )

    total_classes = len(
        counts
    )

    print(
        f"Fault classes mapped to exactly "
        f"one engine: "
        f"{one_engine_classes}/{total_classes}"
    )

    if (
        total_classes > 0
        and one_engine_classes == total_classes
    ):

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Every fault class is associated with exactly "
            "one engine."
        )

        print(
            "Therefore engine_id MUST NOT be used as "
            "a classifier feature."
        )

        print(
            "Evaluation should hold out FLIGHTS, "
            "not engines, otherwise some fault classes "
            "would disappear from training."
        )

    return mapping


# =====================================================================
# ACTIVE FLIGHT TABLE
# =====================================================================

def build_active_flight_table(
    active: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse active telemetry into one row per active flight.

    Because the earlier audit showed zero mixed fault modes per flight,
    each flight should have one fault mode.
    """

    flight_modes = (
        active
        .groupby(
            [
                "engine_id",
                "flight_id",
            ]
        )
    )

    rows = []

    for (
        engine_id,
        flight_id,
    ), group in flight_modes:

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
                "A flight contains multiple "
                "fault modes in active data:\n"
                f"{engine_id} / {flight_id} "
                f"-> {modes}"
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

    flights = (
        flights
        .sort_values(
            [
                "engine_id",
                "first_timestamp",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    return flights


# =====================================================================
# FIRST / LAST ACTIVE FLIGHT
# =====================================================================

def print_fault_timeline(
    active_flights: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("FAULT TIMELINE")
    print("=" * 78)

    summary = (
        active_flights
        .groupby(
            "fault_mode"
        )
        .agg(
            first_active=(
                "first_timestamp",
                "min",
            ),
            last_active=(
                "last_timestamp",
                "max",
            ),
            flights=(
                "flight_id",
                "nunique",
            ),
            engines=(
                "engine_id",
                "nunique",
            ),
            rows=(
                "active_rows",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "first_active"
        )
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )


# =====================================================================
# PER ENGINE TIMELINE
# =====================================================================

def print_engine_timelines(
    active_flights: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("PER-ENGINE ACTIVE-FAULT FLIGHT TIMELINE")
    print("=" * 78)

    for engine_id, group in (
        active_flights
        .groupby(
            "engine_id",
            sort=False,
        )
    ):

        group = (
            group
            .sort_values(
                "first_timestamp"
            )
        )

        print()
        print(
            f"{engine_id}"
        )

        print(
            f"  Fault mode: "
            f"{group['fault_mode'].iloc[0]}"
        )

        print(
            f"  Active flights: "
            f"{len(group)}"
        )

        print(
            "  Flight sequence:"
        )

        for _, row in group.iterrows():

            print(
                f"    {row['flight_id']} "
                f"| {row['first_timestamp']} "
                f"| {row['active_rows']:,} rows"
            )


# =====================================================================
# CHRONOLOGICAL SPLIT
# =====================================================================

def create_candidate_split(
    active_flights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create chronological TRAIN / VAL / TEST assignments per engine.

    The split is performed at FLIGHT level.

    Each engine keeps its own fault class, so all fault classes remain
    represented in train/validation/test as long as the engine has
    enough active flights.
    """

    split_rows = []

    for engine_id, group in (
        active_flights
        .groupby(
            "engine_id",
            sort=False,
        )
    ):

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
                f"Engine {engine_id} has only "
                f"{n} active flights. "
                "At least 3 are needed for a "
                "train/val/test candidate split."
            )

        # ---------------------------------------------------------
        # Initial boundaries
        # ---------------------------------------------------------

        train_end = int(
            np.floor(
                n
                * TRAIN_RATIO
            )
        )

        val_end = int(
            np.floor(
                n
                * (
                    TRAIN_RATIO
                    + VAL_RATIO
                )
            )
        )

        # ---------------------------------------------------------
        # Ensure at least one flight in each split
        # ---------------------------------------------------------

        train_end = max(
            1,
            train_end,
        )

        val_end = max(
            train_end + 1,
            val_end,
        )

        val_end = min(
            val_end,
            n - 1,
        )

        for position, row in (
            group.iterrows()
        ):

            if position < train_end:

                split = "TRAIN"

            elif position < val_end:

                split = "VALIDATION"

            else:

                split = "TEST"

            split_rows.append(
                {
                    "engine_id":
                        engine_id,

                    "flight_id":
                        row["flight_id"],

                    "fault_mode":
                        row["fault_mode"],

                    "first_timestamp":
                        row["first_timestamp"],

                    "last_timestamp":
                        row["last_timestamp"],

                    "active_rows":
                        row["active_rows"],

                    "flight_order_within_engine":
                        position + 1,

                    "total_active_flights_engine":
                        n,

                    "split":
                        split,
                }
            )

    split_df = pd.DataFrame(
        split_rows
    )

    return split_df


# =====================================================================
# SPLIT SUMMARY
# =====================================================================

def print_split_summary(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("CANDIDATE FLIGHT-WISE SPLIT")
    print("=" * 78)

    print()

    split_counts = (
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
        split_counts.to_string()
    )

    print()

    print(
        "Rows by split:"
    )

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
        row_counts.to_string()
    )

    # -------------------------------------------------------------
    # Fault classes per split
    # -------------------------------------------------------------

    print()
    print(
        "Fault classes per split:"
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

    # -------------------------------------------------------------
    # Missing class check
    # -------------------------------------------------------------

    missing_in_train = (
        class_table
        .index[
            class_table[
                "TRAIN"
            ]
            == 0
        ]
        .tolist()
    )

    missing_in_val = (
        class_table
        .index[
            class_table[
                "VALIDATION"
            ]
            == 0
        ]
        .tolist()
    )

    missing_in_test = (
        class_table
        .index[
            class_table[
                "TEST"
            ]
            == 0
        ]
        .tolist()
    )

    print()

    if missing_in_train:

        print(
            "WARNING: Fault classes missing from TRAIN:"
        )

        print(
            missing_in_train
        )

    else:

        print(
            "TRAIN contains all fault classes."
        )

    if missing_in_val:

        print(
            "WARNING: Fault classes missing from VALIDATION:"
        )

        print(
            missing_in_val
        )

    else:

        print(
            "VALIDATION contains all fault classes."
        )

    if missing_in_test:

        print(
            "WARNING: Fault classes missing from TEST:"
        )

        print(
            missing_in_test
        )

    else:

        print(
            "TEST contains all fault classes."
        )


# =====================================================================
# LEAKAGE CHECK
# =====================================================================

def check_split_leakage(
    split_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("FLIGHT SPLIT LEAKAGE CHECK")
    print("=" * 78)

    train_flights = set(
        split_df[
            split_df["split"]
            == "TRAIN"
        ]["flight_id"]
    )

    val_flights = set(
        split_df[
            split_df["split"]
            == "VALIDATION"
        ]["flight_id"]
    )

    test_flights = set(
        split_df[
            split_df["split"]
            == "TEST"
        ]["flight_id"]
    )

    train_val_overlap = (
        train_flights
        & val_flights
    )

    train_test_overlap = (
        train_flights
        & test_flights
    )

    val_test_overlap = (
        val_flights
        & test_flights
    )

    print(
        f"TRAIN ∩ VALIDATION: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"TRAIN ∩ TEST      : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"VALIDATION ∩ TEST : "
        f"{len(val_test_overlap)}"
    )

    if any(
        [
            train_val_overlap,
            train_test_overlap,
            val_test_overlap,
        ]
    ):

        fail(
            "Flight leakage detected."
        )

    print()
    print(
        "PASS: No flight appears in more than one split."
    )


# =====================================================================
# SAVE
# =====================================================================

def save_outputs(
    active: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Population
    # -------------------------------------------------------------

    population_columns = [
        "timestamp",
        "engine_id",
        "uav_id",
        "flight_id",
        "fault_active_label",
        "fault_mode_ground_truth",
    ]

    active[
        population_columns
    ].to_csv(
        POPULATION_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # Flight split
    # -------------------------------------------------------------

    split_df.to_csv(
        SPLIT_OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    print()
    print(
        f"Active population:\n"
        f"{POPULATION_OUTPUT}"
    )

    print()
    print(
        f"Flight split plan:\n"
        f"{SPLIT_OUTPUT}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 78)
    print(
        "FAULT CLASSIFICATION TRAINING POPULATION ANALYSIS"
    )
    print("=" * 78)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------
    # Active population
    # -------------------------------------------------------------

    active = build_active_population(
        df
    )

    # -------------------------------------------------------------
    # Basic
    # -------------------------------------------------------------

    print_basic_summary(
        df,
        active,
    )

    # -------------------------------------------------------------
    # Fault distribution
    # -------------------------------------------------------------

    fault_mode_distribution(
        active
    )

    # -------------------------------------------------------------
    # Engine mapping
    # -------------------------------------------------------------

    fault_engine_mapping(
        active
    )

    # -------------------------------------------------------------
    # Flight table
    # -------------------------------------------------------------

    active_flights = (
        build_active_flight_table(
            active
        )
    )

    # -------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------

    print_fault_timeline(
        active_flights
    )

    # -------------------------------------------------------------
    # Engine timelines
    # -------------------------------------------------------------

    print_engine_timelines(
        active_flights
    )

    # -------------------------------------------------------------
    # Candidate split
    # -------------------------------------------------------------

    split_df = (
        create_candidate_split(
            active_flights
        )
    )

    # -------------------------------------------------------------
    # Split summary
    # -------------------------------------------------------------

    print_split_summary(
        split_df
    )

    # -------------------------------------------------------------
    # Leakage check
    # -------------------------------------------------------------

    check_split_leakage(
        split_df
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    save_outputs(
        active,
        split_df,
    )

    # -------------------------------------------------------------
    # Final message
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        """
Important:

This script did NOT train a classifier.

The result of this analysis will determine:
    1. exact training population
    2. exact flight-wise split
    3. target classes
    4. whether class balancing is necessary
    5. final feature-selection strategy

Only after this will we train the Fault Classifier.
"""
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()