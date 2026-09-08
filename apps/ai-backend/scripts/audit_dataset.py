import sys
from pathlib import Path

import pandas as pd


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.dataset_loader import load_dataset


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def audit_dataframe(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(f"Rows        : {len(df):,}")
    print(f"Columns     : {len(df.columns)}")
    print(f"Duplicates  : {df.duplicated().sum():,}")

    missing = int(df.isna().sum().sum())
    print(f"Missing     : {missing:,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


def main() -> None:
    print_section("DRDO UAV ENGINE DATASET AUDIT")

    dataset = load_dataset()

    metadata = dataset["metadata"]
    telemetry = dataset["telemetry"]
    missions = dataset["missions"]

    audit_dataframe("ENGINE METADATA", metadata)
    audit_dataframe("ENGINE TELEMETRY", telemetry)
    audit_dataframe("MISSION LOG", missions)

    print_section("BASIC IDENTIFIERS")

    if "engine_id" in metadata.columns:
        print(f"Unique engines : {metadata['engine_id'].nunique()}")

    if "uav_id" in metadata.columns:
        print(f"Unique UAVs    : {metadata['uav_id'].nunique()}")

    if "flight_id" in telemetry.columns:
        print(f"Telemetry flights : {telemetry['flight_id'].nunique()}")

    if "mission_id" in missions.columns:
        print(f"Unique missions : {missions['mission_id'].nunique()}")

    print_section("AUDIT COMPLETE")


if __name__ == "__main__":
    main()