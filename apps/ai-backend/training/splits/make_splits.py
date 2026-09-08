from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "training" / "datasets" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "training" / "datasets" / "splits"


# Current fixed engine-wise split.
# Same engine must never appear in more than one split.
TRAIN_ENGINES = {
    "ENG-2023000",
    "ENG-2023001",
    "ENG-2023002",
    "ENG-2023003",
    "ENG-2023004",
    "ENG-2023005",
    "ENG-2023006",
}

VAL_ENGINES = {
    "ENG-2023007",
}

TEST_ENGINES = {
    "ENG-2023008",
    "ENG-2023009",
}


DATASETS = {
    "anomaly": "anomaly_features.csv",
    "fault": "fault_features.csv",
    "rul": "rul_features.csv",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_engine_split(engine_ids: set[str]) -> None:
    """Ensure no engine is present in multiple splits."""

    overlap_train_val = TRAIN_ENGINES & VAL_ENGINES
    overlap_train_test = TRAIN_ENGINES & TEST_ENGINES
    overlap_val_test = VAL_ENGINES & TEST_ENGINES

    if overlap_train_val:
        raise ValueError(
            f"Engine overlap between TRAIN and VAL: {overlap_train_val}"
        )

    if overlap_train_test:
        raise ValueError(
            f"Engine overlap between TRAIN and TEST: {overlap_train_test}"
        )

    if overlap_val_test:
        raise ValueError(
            f"Engine overlap between VAL and TEST: {overlap_val_test}"
        )

    assigned_engines = (
        TRAIN_ENGINES
        | VAL_ENGINES
        | TEST_ENGINES
    )

    missing_engines = engine_ids - assigned_engines
    unknown_assigned_engines = assigned_engines - engine_ids

    if missing_engines:
        raise ValueError(
            f"These engines are not assigned to any split: {missing_engines}"
        )

    if unknown_assigned_engines:
        raise ValueError(
            f"These assigned engines do not exist in dataset: "
            f"{unknown_assigned_engines}"
        )


def split_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split rows using engine_id, never by individual rows."""

    if "engine_id" not in df.columns:
        raise ValueError(
            "Required column 'engine_id' not found in dataset."
        )

    engine_ids = set(df["engine_id"].dropna().unique())

    validate_engine_split(engine_ids)

    train_df = df[df["engine_id"].isin(TRAIN_ENGINES)].copy()
    val_df = df[df["engine_id"].isin(VAL_ENGINES)].copy()
    test_df = df[df["engine_id"].isin(TEST_ENGINES)].copy()

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def save_splits(
    dataset_name: str,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Save train/val/test CSV files for one dataset."""

    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        output_path = dataset_output_dir / f"{split_name}.csv"

        split_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"  {split_name.upper():5} → "
            f"{len(split_df):,} rows → {output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("ENGINE-WISE DATASET SPLITTING")
    print("=" * 70)

    print("\nEngine assignment:")
    print(f"  TRAIN: {sorted(TRAIN_ENGINES)}")
    print(f"  VAL  : {sorted(VAL_ENGINES)}")
    print(f"  TEST : {sorted(TEST_ENGINES)}")

    for dataset_name, filename in DATASETS.items():

        input_path = INPUT_DIR / filename

        print("\n" + "-" * 70)
        print(f"Processing: {dataset_name}")
        print(f"Input: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(
                f"\nInput file not found:\n{input_path}\n\n"
                "First run build_features.py so that the processed "
                "feature datasets are generated."
            )

        df = pd.read_csv(input_path)

        print(f"Original rows: {len(df):,}")

        splits = split_dataframe(df)

        total_split_rows = sum(
            len(split_df)
            for split_df in splits.values()
        )

        if total_split_rows != len(df):
            raise RuntimeError(
                f"Row count mismatch for {dataset_name}: "
                f"original={len(df)}, split_total={total_split_rows}"
            )

        save_splits(
            dataset_name=dataset_name,
            splits=splits,
        )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    print("\nTrain engines:")
    print(f"  {sorted(TRAIN_ENGINES)}")

    print("\nValidation engines:")
    print(f"  {sorted(VAL_ENGINES)}")

    print("\nTest engines:")
    print(f"  {sorted(TEST_ENGINES)}")

    print("\n✅ Engine-wise split completed successfully.")
    print("✅ No engine appears in multiple splits.")
    print(f"✅ Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()