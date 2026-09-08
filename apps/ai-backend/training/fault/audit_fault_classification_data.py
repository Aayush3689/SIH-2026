"""
FAULT CLASSIFICATION DATA AUDIT
================================

Purpose:
    Audit the fault-mode target before training any classifier.

Target:
    fault_mode_ground_truth

Expected fault classes:
    none
    misfire
    injector_abnormality
    coking_degradation
    lubrication_issue
    sensor_drift
    combustion_instability
    overheating_trend
    abnormal_vibration

Checks:
    1. Dataset size
    2. Target distribution
    3. Missing target values
    4. Per-engine fault distribution
    5. Per-flight fault distribution
    6. Fault-active vs fault-mode consistency
    7. Engine ID shortcut / leakage
    8. Number of unique fault modes per engine
    9. First/last occurrence of every fault mode
    10. Class imbalance

IMPORTANT:
    This script does NOT train a model.
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

# File:
# ai-backend/training/fault/audit_fault_classification_data.py
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

OUT_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
)

SUMMARY_PATH = (
    OUT_DIR
    / "fault_classification_audit.csv"
)


# =====================================================================
# EXPECTED VALUES
# =====================================================================

EXPECTED_FAULT_MODES = {
    "none",
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


def validate_columns(
    df: pd.DataFrame,
) -> None:

    required = {
        "engine_id",
        "uav_id",
        "flight_id",
        "timestamp",
        "fault_mode_ground_truth",
        "fault_active_label",
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


# =====================================================================
# LOAD
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

    validate_columns(
        df
    )

    return df


# =====================================================================
# BASIC AUDIT
# =====================================================================

def basic_audit(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("BASIC DATASET AUDIT")
    print("=" * 78)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"Engines    : {df['engine_id'].nunique()}"
    )

    print(
        f"UAVs       : {df['uav_id'].nunique()}"
    )

    print(
        f"Flights    : {df['flight_id'].nunique()}"
    )

    print(
        f"Date range : "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )


# =====================================================================
# TARGET AUDIT
# =====================================================================

def target_audit(
    df: pd.DataFrame,
) -> pd.Series:

    print()
    print("=" * 78)
    print("FAULT TARGET DISTRIBUTION")
    print("=" * 78)

    target = (
        df[
            "fault_mode_ground_truth"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    missing = (
        df[
            "fault_mode_ground_truth"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: {missing}"
    )

    counts = (
        target
        .value_counts()
    )

    percentages = (
        target
        .value_counts(
            normalize=True
        )
        * 100
    )

    print()

    print(
        f"{'FAULT MODE':35s}"
        f"{'ROWS':>12s}"
        f"{'PERCENT':>12s}"
    )

    print("-" * 59)

    for fault_mode, count in counts.items():

        print(
            f"{fault_mode:35s}"
            f"{count:12,d}"
            f"{percentages[fault_mode]:11.2f}%"
        )

    unknown_modes = (
        set(target.unique())
        - EXPECTED_FAULT_MODES
    )

    if unknown_modes:

        print()
        print(
            "WARNING: Unexpected fault modes found:"
        )

        for mode in sorted(
            unknown_modes
        ):

            print(
                f"  - {mode}"
            )

    return target


# =====================================================================
# NONE VS FAULTY
# =====================================================================

def faulty_vs_normal_audit(
    target: pd.Series,
) -> None:

    print()
    print("=" * 78)
    print("NORMAL VS FAULT SAMPLES")
    print("=" * 78)

    none_count = int(
        (
            target
            == "none"
        ).sum()
    )

    faulty_count = int(
        (
            target
            != "none"
        ).sum()
    )

    total = len(target)

    print(
        f"NONE / NORMAL : "
        f"{none_count:,} "
        f"({none_count / total * 100:.2f}%)"
    )

    print(
        f"FAULT         : "
        f"{faulty_count:,} "
        f"({faulty_count / total * 100:.2f}%)"
    )


# =====================================================================
# PER ENGINE TARGET DISTRIBUTION
# =====================================================================

def engine_fault_audit(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:

    print()
    print("=" * 78)
    print("PER-ENGINE FAULT DISTRIBUTION")
    print("=" * 78)

    work = df[
        [
            "engine_id",
            "flight_id",
        ]
    ].copy()

    work[
        "fault_mode"
    ] = target.values

    # Count unique fault modes per engine
    engine_modes = (
        work
        .groupby(
            "engine_id"
        )[
            "fault_mode"
        ]
        .unique()
    )

    print()

    for engine_id in engine_modes.index:

        modes = sorted(
            set(
                engine_modes[
                    engine_id
                ]
            )
        )

        print(
            f"{engine_id}: "
            f"{modes}"
        )

    # -------------------------------------------------------------
    # Fault-only engine mapping
    # -------------------------------------------------------------

    fault_only = work[
        work["fault_mode"]
        != "none"
    ]

    engine_fault_mode = (
        fault_only
        .groupby(
            "engine_id"
        )[
            "fault_mode"
        ]
        .nunique()
    )

    print()
    print(
        "Fault-mode uniqueness per engine:"
    )

    for engine_id, count in (
        engine_fault_mode.items()
    ):

        print(
            f"  {engine_id}: "
            f"{count} fault mode(s)"
        )

    return engine_modes.reset_index(
        name="fault_modes"
    )


# =====================================================================
# ENGINE SHORTCUT / LEAKAGE
# =====================================================================

def engine_leakage_audit(
    df: pd.DataFrame,
    target: pd.Series,
) -> None:

    print()
    print("=" * 78)
    print("ENGINE ID SHORTCUT / LEAKAGE AUDIT")
    print("=" * 78)

    work = pd.DataFrame(
        {
            "engine_id":
                df["engine_id"].values,

            "fault_mode":
                target.values,
        }
    )

    fault_only = work[
        work["fault_mode"]
        != "none"
    ]

    # -------------------------------------------------------------
    # Dominant fault per engine
    # -------------------------------------------------------------

    dominant_rows = []

    for engine_id, group in (
        fault_only
        .groupby("engine_id")
    ):

        counts = (
            group[
                "fault_mode"
            ]
            .value_counts()
        )

        dominant_fault = (
            counts.index[0]
        )

        dominant_count = (
            counts.iloc[0]
        )

        total_fault_rows = len(
            group
        )

        dominant_ratio = (
            dominant_count
            / total_fault_rows
        )

        dominant_rows.append(
            {
                "engine_id":
                    engine_id,

                "dominant_fault":
                    dominant_fault,

                "fault_rows":
                    total_fault_rows,

                "dominant_fault_rows":
                    int(dominant_count),

                "dominant_ratio":
                    dominant_ratio,
            }
        )

    dominant_df = pd.DataFrame(
        dominant_rows
    )

    if dominant_df.empty:

        print(
            "No faulty samples found."
        )

        return

    print()

    print(
        dominant_df.to_string(
            index=False,
            formatters={
                "dominant_ratio":
                    lambda x:
                    f"{x:.2%}"
            },
        )
    )

    # -------------------------------------------------------------
    # Strong one-to-one mapping
    # -------------------------------------------------------------

    unique_faults = (
        dominant_df[
            "dominant_fault"
        ]
        .nunique()
    )

    unique_engines = (
        dominant_df[
            "engine_id"
        ]
        .nunique()
    )

    print()

    print(
        f"Faulty engines : "
        f"{unique_engines}"
    )

    print(
        f"Dominant fault classes : "
        f"{unique_faults}"
    )

    if unique_engines > 0:

        perfect_engine_mapping = (
            dominant_df[
                "dominant_ratio"
            ]
            >= 0.99
        ).mean()
        
        print(
            "Engines with >=99% "
            "single dominant fault: "
            f"{perfect_engine_mapping:.2%}"
        )

        if perfect_engine_mapping >= 0.75:

            print()
            print(
                "WARNING:"
            )

            print(
                "Fault mode is strongly associated "
                "with engine_id."
            )

            print(
                "DO NOT use engine_id as a model feature."
            )


# =====================================================================
# FAULT ACTIVE CONSISTENCY
# =====================================================================

def fault_active_consistency(
    df: pd.DataFrame,
    target: pd.Series,
) -> None:

    print()
    print("=" * 78)
    print("FAULT MODE vs FAULT ACTIVE CONSISTENCY")
    print("=" * 78)

    active = (
        pd.to_numeric(
            df[
                "fault_active_label"
            ],
            errors="coerce",
        )
        .astype(int)
    )

    mode_is_fault = (
        target
        != "none"
    )

    # -------------------------------------------------------------
    # Mode says fault but active says 0
    # -------------------------------------------------------------

    mismatch_1 = int(
        (
            mode_is_fault
            & (active == 0)
        ).sum()
    )

    # -------------------------------------------------------------
    # Active says 1 but mode says none
    # -------------------------------------------------------------

    mismatch_2 = int(
        (
            (~mode_is_fault)
            & (active == 1)
        ).sum()
    )

    total = len(df)

    print(
        f"Fault mode != none "
        f"but fault_active_label = 0: "
        f"{mismatch_1:,}"
    )

    print(
        f"Fault mode = none "
        f"but fault_active_label = 1: "
        f"{mismatch_2:,}"
    )

    print(
        f"Total consistency mismatches: "
        f"{mismatch_1 + mismatch_2:,}"
    )

    print(
        f"Mismatch rate: "
        f"{(mismatch_1 + mismatch_2) / total:.2%}"
    )


# =====================================================================
# FLIGHT LEVEL AUDIT
# =====================================================================

def flight_level_audit(
    df: pd.DataFrame,
    target: pd.Series,
) -> None:

    print()
    print("=" * 78)
    print("FLIGHT-LEVEL FAULT AUDIT")
    print("=" * 78)

    work = pd.DataFrame(
        {
            "engine_id":
                df["engine_id"].values,

            "flight_id":
                df["flight_id"].values,

            "fault_mode":
                target.values,
        }
    )

    # One flight should ideally have one dominant fault mode.
    flight_modes = (
        work
        .groupby(
            [
                "engine_id",
                "flight_id",
            ]
        )[
            "fault_mode"
        ]
        .nunique()
    )

    mixed_flights = (
        flight_modes[
            flight_modes > 1
        ]
    )

    print(
        f"Total flights: "
        f"{len(flight_modes):,}"
    )

    print(
        f"Flights containing "
        f"multiple fault modes: "
        f"{len(mixed_flights):,}"
    )

    if not mixed_flights.empty:

        print()
        print(
            "WARNING: Some flights contain "
            "multiple fault-mode labels."
        )

        print(
            mixed_flights.head(20)
        )

    # -------------------------------------------------------------
    # Dominant fault per flight
    # -------------------------------------------------------------

    dominant = (
        work
        .groupby(
            [
                "engine_id",
                "flight_id",
            ]
        )[
            "fault_mode"
        ]
        .agg(
            lambda x:
            x.value_counts()
            .index[0]
        )
        .reset_index(
            name="dominant_fault_mode"
        )
    )

    print()
    print(
        "Dominant fault per flight:"
    )

    print(
        dominant[
            "dominant_fault_mode"
        ]
        .value_counts()
        .to_string()
    )


# =====================================================================
# FAULT TIMELINE
# =====================================================================

def fault_timeline_audit(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:

    print()
    print("=" * 78)
    print("FAULT MODE TIMELINE")
    print("=" * 78)

    work = df[
        [
            "engine_id",
            "flight_id",
            "timestamp",
        ]
    ].copy()

    work[
        "fault_mode"
    ] = target.values

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    fault_only = work[
        work["fault_mode"]
        != "none"
    ]

    summary_rows = []

    for fault_mode, group in (
        fault_only
        .groupby("fault_mode")
    ):

        summary_rows.append(
            {
                "fault_mode":
                    fault_mode,

                "first_timestamp":
                    group[
                        "timestamp"
                    ].min(),

                "last_timestamp":
                    group[
                        "timestamp"
                    ].max(),

                "rows":
                    len(group),

                "engines":
                    group[
                        "engine_id"
                    ].nunique(),

                "flights":
                    group[
                        "flight_id"
                    ].nunique(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if not summary.empty:

        print(
            summary
            .sort_values(
                "first_timestamp"
            )
            .to_string(
                index=False
            )
        )

    return summary


# =====================================================================
# CLASS IMBALANCE
# =====================================================================

def class_imbalance_audit(
    target: pd.Series,
) -> None:

    print()
    print("=" * 78)
    print("CLASS IMBALANCE")
    print("=" * 78)

    counts = (
        target
        .value_counts()
    )

    if counts.empty:
        return

    largest = (
        counts.max()
    )

    smallest = (
        counts.min()
    )

    ratio = (
        largest
        / smallest
        if smallest > 0
        else np.inf
    )

    print(
        f"Largest class : "
        f"{largest:,}"
    )

    print(
        f"Smallest class: "
        f"{smallest:,}"
    )

    print(
        f"Max / Min ratio: "
        f"{ratio:.2f}x"
    )

    if ratio >= 5:

        print()
        print(
            "NOTE: Significant class imbalance detected."
        )

        print(
            "Training should use class balancing "
            "or another imbalance strategy."
        )


# =====================================================================
# SAVE SUMMARY
# =====================================================================

def save_summary(
    target: pd.Series,
    timeline: pd.DataFrame,
) -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = (
        target
        .value_counts()
        .rename_axis(
            "fault_mode"
        )
        .reset_index(
            name="row_count"
        )
    )

    counts[
        "percentage"
    ] = (
        counts["row_count"]
        / len(target)
        * 100
    )

    if timeline.empty:

        counts.to_csv(
            SUMMARY_PATH,
            index=False,
        )

    else:

        counts.to_csv(
            SUMMARY_PATH,
            index=False,
        )

    print()
    print(
        f"Saved target summary:\n"
        f"{SUMMARY_PATH}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 78)
    print(
        "FAULT CLASSIFICATION DATA AUDIT"
    )
    print("=" * 78)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------
    # Basic audit
    # -------------------------------------------------------------

    basic_audit(
        df
    )

    # -------------------------------------------------------------
    # Target
    # -------------------------------------------------------------

    target = target_audit(
        df
    )

    # -------------------------------------------------------------
    # Normal vs faulty
    # -------------------------------------------------------------

    faulty_vs_normal_audit(
        target
    )

    # -------------------------------------------------------------
    # Engine-level
    # -------------------------------------------------------------

    engine_fault_audit(
        df,
        target
    )

    # -------------------------------------------------------------
    # Leakage
    # -------------------------------------------------------------

    engine_leakage_audit(
        df,
        target
    )

    # -------------------------------------------------------------
    # Consistency
    # -------------------------------------------------------------

    fault_active_consistency(
        df,
        target
    )

    # -------------------------------------------------------------
    # Flight level
    # -------------------------------------------------------------

    flight_level_audit(
        df,
        target
    )

    # -------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------

    timeline = fault_timeline_audit(
        df,
        target
    )

    # -------------------------------------------------------------
    # Class imbalance
    # -------------------------------------------------------------

    class_imbalance_audit(
        target
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    save_summary(
        target,
        timeline
    )

    # -------------------------------------------------------------
    # Final guidance
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    print(
        """
Next decisions depend on the audit:

1. Whether 'none' should be excluded from the fault
   classifier or handled as a separate class.

2. Whether fault labels are strongly tied to engine_id.
   If yes, engine_id must NOT be a model feature.

3. Whether fault_active_label and fault_mode_ground_truth
   are consistent.

4. Which features should be used for the classifier.

5. Final engine-wise train / validation / test split.

No model was trained in this script.
"""
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()