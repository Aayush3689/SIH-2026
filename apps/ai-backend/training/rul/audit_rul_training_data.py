from __future__ import annotations

import json
import sys
from pathlib import Path

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
    / "rul_features.csv"
)

OUTPUT_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "rul_audit"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rul_audit_summary.json"
)

ENGINE_FILE = (
    OUTPUT_DIR
    / "rul_engine_summary.csv"
)

TARGET_FILE = (
    OUTPUT_DIR
    / "rul_target_summary.csv"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "rul_feature_correlations.csv"
)

FEATURE_AUDIT_FILE = (
    OUTPUT_DIR
    / "rul_feature_audit.csv"
)

FLIGHT_FILE = (
    OUTPUT_DIR
    / "rul_flight_summary.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

TARGET = "rul_hours_remaining"

ENGINE = "engine_id"
FLIGHT = "flight_id"
TIMESTAMP = "timestamp"

EXPECTED_ROWS = 94808

# Columns that must NOT be used directly as model features.
IDENTIFIER_COLUMNS = {
    "index",
    TIMESTAMP,
    ENGINE,
    "uav_id",
    FLIGHT,
    "flight_index",
}

TARGET_COLUMNS = {
    TARGET,
}

# Potential leakage / target-derived columns.
LEAKAGE_COLUMNS = {
    "health_index",
    "fault_severity_score",
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

# Operating-condition duplicates / metadata.
NON_FEATURE_COLUMNS = {
    "flight_phase",
    "mission_type",
    "environmental_profile",
}


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


def safe_float(value):
    if pd.isna(value):
        return None

    return float(value)


# =============================================================================
# LOAD
# =============================================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        fail(
            f"RUL feature file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        TARGET,
        ENGINE,
        FLIGHT,
        TIMESTAMP,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        fail(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    return df


# =============================================================================
# BASIC AUDIT
# =============================================================================

def audit_basic(df: pd.DataFrame) -> dict:

    section("BASIC DATASET AUDIT")

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns):,}"
    )

    print(
        f"Engines    : {df[ENGINE].nunique():,}"
    )

    print(
        f"Flights    : {df[FLIGHT].nunique():,}"
    )

    print(
        f"Target     : {TARGET}"
    )

    parsed_time = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    invalid_timestamps = int(
        parsed_time.isna().sum()
    )

    print(
        f"Invalid timestamps: "
        f"{invalid_timestamps:,}"
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    duplicate_flights = int(
        df[FLIGHT].duplicated().sum()
    )

    print(
        f"Duplicate flight references: "
        f"{duplicate_flights:,}"
    )

    if invalid_timestamps > 0:
        fail(
            "Invalid timestamps detected."
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "engines": int(df[ENGINE].nunique()),
        "flights": int(df[FLIGHT].nunique()),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_rows": duplicate_rows,
    }


# =============================================================================
# TARGET AUDIT
# =============================================================================

def audit_target(
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:

    section("RUL TARGET AUDIT")

    target = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if target.isna().any():
        fail(
            f"Missing/non-numeric values found in {TARGET}."
        )

    negative_count = int(
        (target < 0).sum()
    )

    zero_count = int(
        (target == 0).sum()
    )

    positive_count = int(
        (target > 0).sum()
    )

    print(
        f"Minimum : {target.min():.4f} h"
    )

    print(
        f"Maximum : {target.max():.4f} h"
    )

    print(
        f"Mean    : {target.mean():.4f} h"
    )

    print(
        f"Median  : {target.median():.4f} h"
    )

    print(
        f"Zeros   : {zero_count:,} "
        f"({zero_count / len(df) * 100:.2f}%)"
    )

    print(
        f"Positive: {positive_count:,} "
        f"({positive_count / len(df) * 100:.2f}%)"
    )

    print(
        f"Negative: {negative_count:,}"
    )

    if negative_count > 0:
        fail(
            "Negative RUL values detected."
        )

    quantiles = target.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print("\nTarget quantiles:")

    for q, value in quantiles.items():
        print(
            f"  {q:.2f}: {value:.4f}"
        )

    target_summary = pd.DataFrame(
        {
            "statistic": [
                "count",
                "minimum",
                "maximum",
                "mean",
                "median",
                "q01",
                "q05",
                "q25",
                "q50",
                "q75",
                "q95",
                "q99",
                "zero_count",
                "positive_count",
                "negative_count",
            ],
            "value": [
                len(target),
                target.min(),
                target.max(),
                target.mean(),
                target.median(),
                quantiles.loc[0.01],
                quantiles.loc[0.05],
                quantiles.loc[0.25],
                quantiles.loc[0.50],
                quantiles.loc[0.75],
                quantiles.loc[0.95],
                quantiles.loc[0.99],
                zero_count,
                positive_count,
                negative_count,
            ],
        }
    )

    return {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "zero_count": zero_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }, target_summary


# =============================================================================
# ENGINE LEVEL AUDIT
# =============================================================================

def audit_engines(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("ENGINE-WISE RUL AUDIT")

    working = df.copy()

    working["_time"] = pd.to_datetime(
        working[TIMESTAMP],
        errors="coerce",
    )

    rows = []

    for engine_id, group in working.groupby(
        ENGINE,
        sort=True,
    ):

        group = group.sort_values(
            "_time"
        )

        rul = pd.to_numeric(
            group[TARGET],
            errors="coerce",
        )

        first_rul = float(
            rul.iloc[0]
        )

        last_rul = float(
            rul.iloc[-1]
        )

        minimum_rul = float(
            rul.min()
        )

        maximum_rul = float(
            rul.max()
        )

        differences = (
            rul.diff()
            .dropna()
        )

        decreases = int(
            (differences < 0).sum()
        )

        increases = int(
            (differences > 0).sum()
        )

        unchanged = int(
            (differences == 0).sum()
        )

        rows.append(
            {
                "engine_id": engine_id,
                "rows": len(group),
                "flights": group[FLIGHT].nunique(),
                "first_timestamp": (
                    group["_time"].min()
                ),
                "last_timestamp": (
                    group["_time"].max()
                ),
                "first_rul_h": first_rul,
                "last_rul_h": last_rul,
                "minimum_rul_h": minimum_rul,
                "maximum_rul_h": maximum_rul,
                "rul_decreases": decreases,
                "rul_increases": increases,
                "rul_unchanged": unchanged,
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        ENGINE_FILE,
        index=False,
    )

    return result


# =============================================================================
# FLIGHT LEVEL AUDIT
# =============================================================================

def audit_flights(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("FLIGHT-WISE RUL AUDIT")

    working = df.copy()

    working["_time"] = pd.to_datetime(
        working[TIMESTAMP],
        errors="coerce",
    )

    rows = []

    for flight_id, group in working.groupby(
        FLIGHT,
        sort=False,
    ):

        group = group.sort_values(
            "_time"
        )

        rul = pd.to_numeric(
            group[TARGET],
            errors="coerce",
        )

        first_rul = float(
            rul.iloc[0]
        )

        last_rul = float(
            rul.iloc[-1]
        )

        unique_rul = int(
            rul.nunique()
        )

        rows.append(
            {
                "flight_id": flight_id,
                "engine_id": group[
                    ENGINE
                ].iloc[0],
                "rows": len(group),
                "first_timestamp": group[
                    "_time"
                ].min(),
                "last_timestamp": group[
                    "_time"
                ].max(),
                "start_rul_h": first_rul,
                "end_rul_h": last_rul,
                "rul_change_h": (
                    last_rul
                    - first_rul
                ),
                "unique_rul_values": unique_rul,
                "constant_rul_within_flight": (
                    unique_rul == 1
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        f"Flights with constant RUL: "
        f"{int(result['constant_rul_within_flight'].sum())} / "
        f"{len(result)}"
    )

    print(
        "\nLargest RUL increases within a flight:"
    )

    print(
        result.sort_values(
            "rul_change_h",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nLargest RUL decreases within a flight:"
    )

    print(
        result.sort_values(
            "rul_change_h"
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    result.to_csv(
        FLIGHT_FILE,
        index=False,
    )

    return result


# =============================================================================
# FEATURE AUDIT
# =============================================================================

def audit_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    section("RUL FEATURE AUDIT")

    numeric_candidates = []

    for column in df.columns:

        if column in TARGET_COLUMNS:
            continue

        if column in IDENTIFIER_COLUMNS:
            continue

        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            numeric_candidates.append(
                column
            )

    rows = []

    for column in numeric_candidates:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        missing = int(
            series.isna().sum()
        )

        inf_count = int(
            np.isinf(
                series.to_numpy(
                    dtype=float
                )
            ).sum()
        )

        unique_count = int(
            series.nunique()
        )

        rows.append(
            {
                "feature": column,
                "dtype": str(
                    df[column].dtype
                ),
                "missing_count": missing,
                "missing_pct": (
                    missing
                    / len(df)
                    * 100
                ),
                "infinite_count": inf_count,
                "unique_count": unique_count,
                "constant": (
                    unique_count <= 1
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # Detect explicitly suspicious columns.
    result[
        "potential_leakage"
    ] = result["feature"].isin(
        LEAKAGE_COLUMNS
    )

    print(
        f"Numeric candidates: "
        f"{len(result):,}"
    )

    leakage = result[
        result["potential_leakage"]
    ]

    if len(leakage) > 0:

        print(
            "\nPotential leakage columns:"
        )

        print(
            leakage[
                [
                    "feature",
                    "missing_count",
                    "unique_count",
                ]
            ].to_string(
                index=False
            )
        )

    constants = result[
        result["constant"]
    ]

    print(
        f"\nConstant features: "
        f"{len(constants)}"
    )

    result.to_csv(
        FEATURE_AUDIT_FILE,
        index=False,
    )

    return (
        result,
        numeric_candidates,
    )


# =============================================================================
# CORRELATION AUDIT
# =============================================================================

def correlation_audit(
    df: pd.DataFrame,
    numeric_candidates: list[str],
) -> pd.DataFrame:

    section("RUL TARGET CORRELATIONS")

    working = df[
        numeric_candidates
        + [TARGET]
    ].copy()

    for column in working.columns:

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    correlations = (
        working.corr(
            numeric_only=True
        )[TARGET]
        .drop(
            TARGET,
            errors="ignore",
        )
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    result = (
        correlations
        .reset_index()
    )

    result.columns = [
        "feature",
        "correlation_with_rul",
    ]

    result[
        "absolute_correlation"
    ] = result[
        "correlation_with_rul"
    ].abs()

    print(
        result.head(30).to_string(
            index=False,
            formatters={
                "correlation_with_rul":
                    "{:.6f}".format,
                "absolute_correlation":
                    "{:.6f}".format,
            },
        )
    )

    result.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    return result


# =============================================================================
# TEMPORAL TARGET BEHAVIOUR
# =============================================================================

def temporal_target_audit(
    df: pd.DataFrame,
) -> dict:

    section("TEMPORAL TARGET BEHAVIOUR")

    working = df.copy()

    working["_time"] = pd.to_datetime(
        working[TIMESTAMP],
        errors="coerce",
    )

    working[TARGET] = pd.to_numeric(
        working[TARGET],
        errors="coerce",
    )

    suspicious_rows = []

    for engine_id, group in working.groupby(
        ENGINE,
        sort=True,
    ):

        group = group.sort_values(
            "_time"
        )

        differences = (
            group[TARGET]
            .diff()
        )

        increases = differences[
            differences > 0
        ]

        decreases = differences[
            differences < 0
        ]

        suspicious_rows.append(
            {
                "engine_id": engine_id,
                "rows": len(group),
                "increases": int(
                    len(increases)
                ),
                "decreases": int(
                    len(decreases)
                ),
                "max_increase_h": (
                    float(
                        increases.max()
                    )
                    if len(increases)
                    else 0.0
                ),
                "max_decrease_h": (
                    float(
                        decreases.min()
                    )
                    if len(decreases)
                    else 0.0
                ),
            }
        )

    result = pd.DataFrame(
        suspicious_rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    total_increases = int(
        result["increases"].sum()
    )

    total_decreases = int(
        result["decreases"].sum()
    )

    print(
        f"\nTotal chronological RUL increases: "
        f"{total_increases:,}"
    )

    print(
        f"Total chronological RUL decreases: "
        f"{total_decreases:,}"
    )

    return {
        "total_increases": total_increases,
        "total_decreases": total_decreases,
    }


# =============================================================================
# LEAKAGE CHECK
# =============================================================================

def leakage_check(
    df: pd.DataFrame,
    feature_audit: pd.DataFrame,
) -> dict:

    section("LEAKAGE CHECK")

    explicit_leakage = sorted(
        set(df.columns)
        & LEAKAGE_COLUMNS
    )

    print(
        "Explicit suspicious columns:"
    )

    if explicit_leakage:
        for column in explicit_leakage:
            print(
                f"  - {column}"
            )
    else:
        print(
            "  None found."
        )

    target_like = []

    for column in df.columns:

        lower = column.lower()

        if (
            column == TARGET
            or "rul" in lower
        ):
            target_like.append(
                column
            )

    print(
        "\nRUL-like columns:"
    )

    for column in sorted(
        set(target_like)
    ):
        print(
            f"  - {column}"
        )

    return {
        "explicit_leakage_columns": explicit_leakage,
        "rul_like_columns": sorted(
            set(target_like)
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 78)
    print("RUL TRAINING DATA AUDIT")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    basic = audit_basic(
        df
    )

    target_summary, target_df = (
        audit_target(
            df
        )
    )

    engine_df = audit_engines(
        df
    )

    flight_df = audit_flights(
        df
    )

    feature_df, numeric_candidates = (
        audit_features(
            df
        )
    )

    correlation_df = (
        correlation_audit(
            df,
            numeric_candidates,
        )
    )

    temporal = (
        temporal_target_audit(
            df
        )
    )

    leakage = leakage_check(
        df,
        feature_df,
    )

    # -------------------------------------------------------------------------
    # Save target summary
    # -------------------------------------------------------------------------

    target_df.to_csv(
        TARGET_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Strongly suspicious correlations
    # -------------------------------------------------------------------------

    suspicious_correlations = (
        correlation_df[
            correlation_df[
                "absolute_correlation"
            ] >= 0.95
        ]
        .head(50)
        .to_dict(
            orient="records"
        )
    )

    # -------------------------------------------------------------------------
    # Potential feature list
    # -------------------------------------------------------------------------

    usable_candidates = (
        feature_df[
            (~feature_df["potential_leakage"])
            & (~feature_df["constant"])
        ]["feature"]
        .tolist()
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "input_file": str(
            INPUT_FILE
        ),
        "rows": basic["rows"],
        "columns": basic["columns"],
        "engines": basic["engines"],
        "flights": basic["flights"],
        "target": TARGET,
        "target_statistics": target_summary,
        "temporal_target_behaviour": temporal,
        "leakage_check": leakage,
        "numeric_candidate_count": len(
            numeric_candidates
        ),
        "usable_candidate_count": len(
            usable_candidates
        ),
        "usable_candidates": sorted(
            usable_candidates
        ),
        "high_correlation_features": (
            suspicious_correlations
        ),
        "constant_features": (
            feature_df[
                feature_df["constant"]
            ]["feature"]
            .tolist()
        ),
        "missing_features": (
            feature_df[
                feature_df[
                    "missing_count"
                ] > 0
            ][
                [
                    "feature",
                    "missing_count",
                    "missing_pct",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    section("AUDIT SUMMARY")

    print(
        f"Rows              : "
        f"{basic['rows']:,}"
    )

    print(
        f"Engines           : "
        f"{basic['engines']}"
    )

    print(
        f"Flights           : "
        f"{basic['flights']}"
    )

    print(
        f"Numeric candidates: "
        f"{len(numeric_candidates)}"
    )

    print(
        f"Usable candidates : "
        f"{len(usable_candidates)}"
    )

    print(
        f"RUL increases     : "
        f"{temporal['total_increases']:,}"
    )

    print(
        f"RUL decreases     : "
        f"{temporal['total_decreases']:,}"
    )

    print(
        "\nAudit files:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {ENGINE_FILE}"
    )

    print(
        f"  {TARGET_FILE}"
    )

    print(
        f"  {CORRELATION_FILE}"
    )

    print(
        f"  {FEATURE_AUDIT_FILE}"
    )

    print(
        f"  {FLIGHT_FILE}"
    )

    print(
        "\nRUL AUDIT COMPLETE."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("RUL AUDIT FAILED")
        print("=" * 78)
        print(str(exc))
        sys.exit(1)