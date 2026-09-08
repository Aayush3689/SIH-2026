from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "processed"
    / "final_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "processed"
    / "anomaly"
)

OUTPUT_PATH = OUTPUT_DIR / "anomaly_labeled.csv"
SUMMARY_PATH = OUTPUT_DIR / "anomaly_label_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "anomaly_label_manifest.json"


# ============================================================
# CONFIGURATION
# ============================================================

# Immediately preceding flights before the first active-fault
# flight are marked as PRE_FAULT.
PREFAULT_FLIGHTS = 3

REQUIRED_COLUMNS = [
    "timestamp",
    "engine_id",
    "flight_id",
    "fault_active_label",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Validate required source columns."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


# ============================================================
# BUILD FLIGHT TABLE
# ============================================================

def build_flight_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per engine-flight.

    A flight is considered ACTIVE_FAULT if any telemetry row
    inside that flight has fault_active_label == 1.
    """

    flight_table = (
        df.groupby(
            ["engine_id", "flight_id"],
            sort=False,
        )
        .agg(
            flight_start=("timestamp", "min"),
            flight_end=("timestamp", "max"),
            fault_active=(
                "fault_active_label",
                "max",
            ),
            telemetry_rows=(
                "fault_active_label",
                "size",
            ),
        )
        .reset_index()
    )

    flight_table["fault_active"] = (
        flight_table["fault_active"]
        .astype(int)
    )

    # IMPORTANT:
    # Sorting by actual timestamp defines the chronological
    # order of flights. We do NOT trust F001/F002/F003 naming.
    flight_table = (
        flight_table
        .sort_values(
            [
                "engine_id",
                "flight_start",
            ]
        )
        .reset_index(drop=True)
    )

    return flight_table


# ============================================================
# FIND FIRST FAULT FLIGHT
# ============================================================

def find_first_fault_positions(
    flight_table: pd.DataFrame,
) -> dict[str, int]:
    """
    Find the LOCAL positional index of the first active-fault
    flight for every faulty engine.

    Example for one engine:

        position 0 -> F001
        position 1 -> F002
        position 2 -> F004
        position 3 -> F003
        ...

    This function returns the position inside that engine's
    own sorted flight table.
    """

    first_fault_positions: dict[str, int] = {}

    for engine_id, engine_df in flight_table.groupby(
        "engine_id",
        sort=True,
    ):

        # VERY IMPORTANT:
        # Reset the index after filtering one engine.
        # This prevents global dataframe indices from being
        # confused with local engine positions.
        engine_df = (
            engine_df
            .sort_values("flight_start")
            .reset_index(drop=True)
        )

        active_positions = engine_df.index[
            engine_df["fault_active"] == 1
        ].tolist()

        if not active_positions:
            continue

        first_fault_positions[engine_id] = (
            active_positions[0]
        )

    return first_fault_positions


# ============================================================
# ASSIGN STATES
# ============================================================

def assign_anomaly_states(
    flight_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign flight-level anomaly states.

        0 = NORMAL
        1 = PRE_FAULT
        2 = ACTIVE_FAULT

    Healthy engines:
        all flights -> NORMAL

    Faulty engines:
        previous PREFAULT_FLIGHTS before first fault
            -> PRE_FAULT

        first active-fault flight onward
            -> ACTIVE_FAULT

        earlier flights
            -> NORMAL
    """

    result = flight_table.copy()

    # Default state for every flight.
    result["anomaly_state"] = 0
    result["anomaly_state_name"] = "NORMAL"

    first_fault_positions = find_first_fault_positions(
        result
    )

    # --------------------------------------------------------
    # Process every faulty engine independently
    # --------------------------------------------------------

    for engine_id, first_fault_position in (
        first_fault_positions.items()
    ):

        # Extract this engine's rows and RESET INDEX.
        engine_df = (
            result[
                result["engine_id"] == engine_id
            ]
            .sort_values("flight_start")
            .reset_index(drop=True)
        )

        if engine_df.empty:
            continue

        # Safety check.
        if not (
            0 <= first_fault_position < len(engine_df)
        ):
            raise RuntimeError(
                f"Invalid first fault position "
                f"{first_fault_position} for {engine_id}."
            )

        # ----------------------------------------------------
        # PRE_FAULT
        # ----------------------------------------------------

        prefault_start = max(
            0,
            first_fault_position - PREFAULT_FLIGHTS,
        )

        prefault_positions = list(
            range(
                prefault_start,
                first_fault_position,
            )
        )

        # Convert local positions back to actual result
        # dataframe indices.
        prefault_flight_ids = engine_df.iloc[
            prefault_positions
        ]["flight_id"].tolist()

        if prefault_flight_ids:

            prefault_mask = (
                (result["engine_id"] == engine_id)
                & result["flight_id"].isin(
                    prefault_flight_ids
                )
            )

            result.loc[
                prefault_mask,
                "anomaly_state",
            ] = 1

            result.loc[
                prefault_mask,
                "anomaly_state_name",
            ] = "PRE_FAULT"

        # ----------------------------------------------------
        # ACTIVE_FAULT
        # ----------------------------------------------------

        active_positions = list(
            range(
                first_fault_position,
                len(engine_df),
            )
        )

        active_flight_ids = engine_df.iloc[
            active_positions
        ]["flight_id"].tolist()

        if active_flight_ids:

            active_mask = (
                (result["engine_id"] == engine_id)
                & result["flight_id"].isin(
                    active_flight_ids
                )
            )

            result.loc[
                active_mask,
                "anomaly_state",
            ] = 2

            result.loc[
                active_mask,
                "anomaly_state_name",
            ] = "ACTIVE_FAULT"

    return result


# ============================================================
# VALIDATE GENERATED STATES
# ============================================================

def validate_generated_states(
    flight_states: pd.DataFrame,
) -> None:
    """
    Check that generated flight states obey the intended rules.
    """

    expected_mapping = {
        0: "NORMAL",
        1: "PRE_FAULT",
        2: "ACTIVE_FAULT",
    }

    # --------------------------------------------------------
    # State number/name consistency
    # --------------------------------------------------------

    mapping = (
        flight_states[
            [
                "anomaly_state",
                "anomaly_state_name",
            ]
        ]
        .drop_duplicates()
    )

    for row in mapping.itertuples(index=False):

        expected_name = expected_mapping.get(
            int(row.anomaly_state)
        )

        if expected_name != row.anomaly_state_name:
            raise RuntimeError(
                "Invalid anomaly-state mapping:\n"
                f"state={row.anomaly_state}, "
                f"name={row.anomaly_state_name}"
            )

    # --------------------------------------------------------
    # Validate each engine chronologically
    # --------------------------------------------------------

    for engine_id, engine_df in flight_states.groupby(
        "engine_id",
        sort=True,
    ):

        engine_df = (
            engine_df
            .sort_values("flight_start")
            .reset_index(drop=True)
        )

        active_positions = engine_df.index[
            engine_df["fault_active"] == 1
        ].tolist()

        # Healthy engine
        if not active_positions:

            if not (
                engine_df["anomaly_state"] == 0
            ).all():
                raise RuntimeError(
                    f"Healthy engine {engine_id} "
                    "contains non-NORMAL states."
                )

            continue

        first_fault_position = active_positions[0]

        # Expected local positions
        expected_prefault = set(
            range(
                max(
                    0,
                    first_fault_position
                    - PREFAULT_FLIGHTS,
                ),
                first_fault_position,
            )
        )

        expected_active = set(
            range(
                first_fault_position,
                len(engine_df),
            )
        )

        for position, row in engine_df.iterrows():

            expected_state = 0

            if position in expected_prefault:
                expected_state = 1

            if position in expected_active:
                expected_state = 2

            actual_state = int(
                row["anomaly_state"]
            )

            if actual_state != expected_state:
                raise RuntimeError(
                    f"State validation failed for "
                    f"{engine_id}, position={position}, "
                    f"flight={row['flight_id']}.\n"
                    f"Expected={expected_state}, "
                    f"Actual={actual_state}"
                )


# ============================================================
# ATTACH FLIGHT STATES TO TELEMETRY
# ============================================================

def attach_states_to_telemetry(
    df: pd.DataFrame,
    flight_states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map one flight-level anomaly state to all telemetry rows
    belonging to that flight.
    """

    state_columns = [
        "engine_id",
        "flight_id",
        "anomaly_state",
        "anomaly_state_name",
    ]

    state_table = flight_states[
        state_columns
    ].copy()

    labeled_df = df.merge(
        state_table,
        on=[
            "engine_id",
            "flight_id",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_count = int(
        labeled_df["anomaly_state"]
        .isna()
        .sum()
    )

    if missing_count:
        raise RuntimeError(
            f"{missing_count:,} telemetry rows could not "
            "be assigned an anomaly state."
        )

    labeled_df["anomaly_state"] = (
        labeled_df["anomaly_state"]
        .astype(int)
    )

    return labeled_df


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    labeled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Telemetry-level summary by engine and anomaly state.
    """

    summary = (
        labeled_df.groupby(
            [
                "engine_id",
                "anomaly_state",
                "anomaly_state_name",
            ],
            sort=True,
        )
        .agg(
            telemetry_rows=(
                "timestamp",
                "size",
            ),
            flights=(
                "flight_id",
                "nunique",
            ),
            first_timestamp=(
                "timestamp",
                "min",
            ),
            last_timestamp=(
                "timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    summary["row_percentage"] = (
        summary["telemetry_rows"]
        / len(labeled_df)
        * 100
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ANOMALY STATE LABEL GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load source dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run first:\n"
            "python training/features/build_features.py"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
    )

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    validate_columns(df)

    # --------------------------------------------------------
    # Build flight-level table
    # --------------------------------------------------------

    print(
        "\nBuilding flight-level table..."
    )

    flight_table = build_flight_table(
        df
    )

    print(
        f"Flights found: {len(flight_table):,}"
    )

    # --------------------------------------------------------
    # Print source fault distribution per engine
    # --------------------------------------------------------

    print(
        "\nSource fault-active distribution:"
    )

    for engine_id, engine_df in flight_table.groupby(
        "engine_id",
        sort=True,
    ):

        active_flights = int(
            engine_df["fault_active"].sum()
        )

        print(
            f"  {engine_id} → "
            f"{active_flights} active-fault flights / "
            f"{len(engine_df)} total flights"
        )

    # --------------------------------------------------------
    # Assign states
    # --------------------------------------------------------

    print(
        "\nAssigning anomaly states..."
    )

    flight_states = assign_anomaly_states(
        flight_table
    )

    # --------------------------------------------------------
    # Validate before mapping back to telemetry
    # --------------------------------------------------------

    validate_generated_states(
        flight_states
    )

    # --------------------------------------------------------
    # First fault flights
    # --------------------------------------------------------

    print(
        "\nFirst fault-active flight per faulty engine:"
    )

    faulty_engine_found = False

    for engine_id, engine_df in flight_states.groupby(
        "engine_id",
        sort=True,
    ):

        engine_df = (
            engine_df
            .sort_values("flight_start")
            .reset_index(drop=True)
        )

        active_rows = engine_df[
            engine_df["fault_active"] == 1
        ]

        if active_rows.empty:
            continue

        faulty_engine_found = True

        first_fault = active_rows.iloc[0]

        print(
            f"  {engine_id} → "
            f"{first_fault['flight_id']}"
        )

    if not faulty_engine_found:
        print(
            "  No faulty engines found."
        )

    # --------------------------------------------------------
    # Map states back to telemetry
    # --------------------------------------------------------

    labeled_df = attach_states_to_telemetry(
        df,
        flight_states,
    )

    # --------------------------------------------------------
    # State distribution
    # --------------------------------------------------------

    print(
        "\nState distribution:"
    )

    state_counts = (
        labeled_df[
            "anomaly_state_name"
        ]
        .value_counts()
    )

    for state_name in [
        "NORMAL",
        "PRE_FAULT",
        "ACTIVE_FAULT",
    ]:

        count = int(
            state_counts.get(
                state_name,
                0,
            )
        )

        percentage = (
            count
            / len(labeled_df)
            * 100
        )

        print(
            f"  {state_name:15} → "
            f"{count:8,} rows "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        labeled_df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save labeled telemetry
    # --------------------------------------------------------

    labeled_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "prefault_flights": PREFAULT_FLIGHTS,
        "label_definition": {
            "0": "NORMAL",
            "1": "PRE_FAULT",
            "2": "ACTIVE_FAULT",
        },
        "label_construction": [
            (
                "Healthy engines remain NORMAL."
            ),
            (
                "For faulty engines, the immediately "
                f"preceding {PREFAULT_FLIGHTS} flights "
                "before the first active-fault flight "
                "are labelled PRE_FAULT."
            ),
            (
                "The first active-fault flight and every "
                "subsequent flight are labelled ACTIVE_FAULT."
            ),
            (
                "Earlier flights of a faulty engine remain "
                "NORMAL."
            ),
            (
                "Labels are derived only from "
                "fault_active_label and flight chronology."
            ),
        ],
        "rows": int(len(labeled_df)),
        "flights": int(len(flight_states)),
        "engines": int(
            labeled_df["engine_id"].nunique()
        ),
        "state_distribution": {
            str(key): int(value)
            for key, value in (
                labeled_df["anomaly_state"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY LABEL GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nGenerated:"
        f"\n  {OUTPUT_PATH}"
        f"\n  {SUMMARY_PATH}"
        f"\n  {MANIFEST_PATH}"
    )

    print(
        "\nLabel definition:"
        "\n  0 → NORMAL"
        "\n  1 → PRE_FAULT"
        "\n  2 → ACTIVE_FAULT"
    )

    print(
        "\n✅ All engines processed independently."
    )

    print(
        "✅ Labels are based on flight chronology."
    )

    print(
        "✅ No sensor/health/severity value was used "
        "to create labels."
    )

    print(
        "✅ State consistency validation passed."
    )


if __name__ == "__main__":
    main()