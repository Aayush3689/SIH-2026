import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.dataset_loader import load_dataset


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_result(check: str, passed: bool, details: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    message = f"[{status}] {check}"

    if details:
        message += f" -> {details}"

    print(message)


# ---------------------------------------------------------
# Validation functions
# ---------------------------------------------------------
def validate_engine_mapping(
    metadata: pd.DataFrame,
    telemetry: pd.DataFrame,
    missions: pd.DataFrame,
) -> bool:
    all_passed = True

    metadata_engines = set(metadata["engine_id"].dropna().unique())
    telemetry_engines = set(telemetry["engine_id"].dropna().unique())
    mission_engines = set(missions["engine_id"].dropna().unique())

    # Telemetry engine IDs must exist in metadata
    unknown_telemetry_engines = telemetry_engines - metadata_engines

    passed = len(unknown_telemetry_engines) == 0
    print_result(
        "Telemetry engine IDs exist in metadata",
        passed,
        f"Unknown engines: {unknown_telemetry_engines}"
        if not passed
        else f"{len(telemetry_engines)} engines verified",
    )

    all_passed &= passed

    # Mission engine IDs must exist in metadata
    unknown_mission_engines = mission_engines - metadata_engines

    passed = len(unknown_mission_engines) == 0
    print_result(
        "Mission engine IDs exist in metadata",
        passed,
        f"Unknown engines: {unknown_mission_engines}"
        if not passed
        else f"{len(mission_engines)} engines verified",
    )

    all_passed &= passed

    return all_passed


def validate_uav_mapping(
    metadata: pd.DataFrame,
    telemetry: pd.DataFrame,
    missions: pd.DataFrame,
) -> bool:
    all_passed = True

    metadata_uavs = set(metadata["uav_id"].dropna().unique())
    telemetry_uavs = set(telemetry["uav_id"].dropna().unique())
    mission_uavs = set(missions["uav_id"].dropna().unique())

    unknown_telemetry_uavs = telemetry_uavs - metadata_uavs

    passed = len(unknown_telemetry_uavs) == 0
    print_result(
        "Telemetry UAV IDs exist in metadata",
        passed,
        f"Unknown UAVs: {unknown_telemetry_uavs}"
        if not passed
        else f"{len(telemetry_uavs)} UAVs verified",
    )

    all_passed &= passed

    unknown_mission_uavs = mission_uavs - metadata_uavs

    passed = len(unknown_mission_uavs) == 0
    print_result(
        "Mission UAV IDs exist in metadata",
        passed,
        f"Unknown UAVs: {unknown_mission_uavs}"
        if not passed
        else f"{len(mission_uavs)} UAVs verified",
    )

    all_passed &= passed

    return all_passed


def validate_engine_uav_pairs(
    metadata: pd.DataFrame,
    telemetry: pd.DataFrame,
    missions: pd.DataFrame,
) -> bool:
    all_passed = True

    metadata_pairs = set(
        zip(metadata["engine_id"], metadata["uav_id"])
    )

    telemetry_pairs = set(
        zip(telemetry["engine_id"], telemetry["uav_id"])
    )

    mission_pairs = set(
        zip(missions["engine_id"], missions["uav_id"])
    )

    unknown_telemetry_pairs = telemetry_pairs - metadata_pairs

    passed = len(unknown_telemetry_pairs) == 0
    print_result(
        "Telemetry engine-UAV pairs match metadata",
        passed,
        f"Invalid pairs: {unknown_telemetry_pairs}"
        if not passed
        else f"{len(telemetry_pairs)} pairs verified",
    )

    all_passed &= passed

    unknown_mission_pairs = mission_pairs - metadata_pairs

    passed = len(unknown_mission_pairs) == 0
    print_result(
        "Mission engine-UAV pairs match metadata",
        passed,
        f"Invalid pairs: {unknown_mission_pairs}"
        if not passed
        else f"{len(mission_pairs)} pairs verified",
    )

    all_passed &= passed

    return all_passed


def validate_flights(
    telemetry: pd.DataFrame,
    missions: pd.DataFrame,
) -> bool:
    all_passed = True

    telemetry_flights = set(
        telemetry["flight_id"].dropna().unique()
    )

    mission_flights = set(
        missions["flight_id"].dropna().unique()
    )

    # Telemetry flights should exist in mission log
    unknown_telemetry_flights = telemetry_flights - mission_flights

    passed = len(unknown_telemetry_flights) == 0

    print_result(
        "Telemetry flight IDs exist in mission log",
        passed,
        f"Unknown flights: {len(unknown_telemetry_flights)}"
        if not passed
        else f"{len(telemetry_flights)} flights verified",
    )

    all_passed &= passed

    # Mission flights should exist in telemetry
    missing_telemetry_flights = mission_flights - telemetry_flights

    passed = len(missing_telemetry_flights) == 0

    print_result(
        "All mission flights have telemetry",
        passed,
        f"Missing telemetry flights: {len(missing_telemetry_flights)}"
        if not passed
        else f"{len(mission_flights)} flights covered",
    )

    all_passed &= passed

    return all_passed


def validate_timestamps(telemetry: pd.DataFrame) -> bool:
    if "timestamp" not in telemetry.columns:
        print_result(
            "Telemetry timestamp column exists",
            False,
            "Column not found",
        )
        return False

    timestamps = pd.to_datetime(
        telemetry["timestamp"],
        errors="coerce",
    )

    invalid_count = int(timestamps.isna().sum())

    passed = invalid_count == 0

    print_result(
        "Telemetry timestamps are valid",
        passed,
        f"Invalid timestamps: {invalid_count}",
    )

    return passed


def validate_leakage_columns(telemetry: pd.DataFrame) -> bool:
    leakage_columns = {
        "fault_mode_ground_truth",
        "fault_severity_score",
        "fault_active_label",
        "health_index",
        "rul_hours_remaining",
        "label_misfire",
        "label_injector_abnormality",
        "label_coking_degradation",
        "label_lubrication_issue",
        "label_sensor_drift",
        "label_combustion_instability",
        "label_overheating_trend",
        "label_abnormal_vibration",
    }

    found_columns = leakage_columns.intersection(telemetry.columns)

    # This is intentionally NOT a failure.
    # These columns exist as targets/reference labels.
    print_result(
        "Ground-truth / target columns identified",
        True,
        f"{len(found_columns)} target/reference columns found",
    )

    print("\nTarget/reference columns:")
    for column in sorted(found_columns):
        print(f"  - {column}")

    return True


def validate_numeric_ranges(telemetry: pd.DataFrame) -> bool:
    all_passed = True

    range_checks = {
        "throttle_position_pct": (0, 100),
        "engine_load_pct": (0, 100),
        "fault_severity_score": (0, 1),
        "health_index": (0, 100),
        "rul_hours_remaining": (0, None),
    }

    for column, (minimum, maximum) in range_checks.items():
        if column not in telemetry.columns:
            continue

        values = pd.to_numeric(
            telemetry[column],
            errors="coerce",
        )

        invalid_mask = pd.Series(False, index=telemetry.index)

        if minimum is not None:
            invalid_mask |= values < minimum

        if maximum is not None:
            invalid_mask |= values > maximum

        invalid_count = int(invalid_mask.sum())

        passed = invalid_count == 0

        details = (
            f"Invalid values: {invalid_count}"
            if not passed
            else "Range OK"
        )

        print_result(
            f"{column} sanity check",
            passed,
            details,
        )

        all_passed &= passed

    return all_passed


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print_section("DRDO UAV ENGINE DATASET VALIDATION")

    dataset = load_dataset()

    metadata = dataset["metadata"]
    telemetry = dataset["telemetry"]
    missions = dataset["missions"]

    validation_results = []

    print_section("ENGINE RELATIONSHIPS")

    validation_results.append(
        validate_engine_mapping(
            metadata,
            telemetry,
            missions,
        )
    )

    print_section("UAV RELATIONSHIPS")

    validation_results.append(
        validate_uav_mapping(
            metadata,
            telemetry,
            missions,
        )
    )

    print_section("ENGINE ↔ UAV CONSISTENCY")

    validation_results.append(
        validate_engine_uav_pairs(
            metadata,
            telemetry,
            missions,
        )
    )

    print_section("FLIGHT RELATIONSHIPS")

    validation_results.append(
        validate_flights(
            telemetry,
            missions,
        )
    )

    print_section("TIMESTAMP VALIDATION")

    validation_results.append(
        validate_timestamps(telemetry)
    )

    print_section("DATA LEAKAGE / TARGET IDENTIFICATION")

    validation_results.append(
        validate_leakage_columns(telemetry)
    )

    print_section("NUMERIC SANITY CHECKS")

    validation_results.append(
        validate_numeric_ranges(telemetry)
    )

    print_section("FINAL RESULT")

    if all(validation_results):
        print("✅ DATASET VALIDATION PASSED")
    else:
        print("❌ DATASET VALIDATION FAILED")


if __name__ == "__main__":
    main()