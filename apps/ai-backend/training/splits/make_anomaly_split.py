from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

# apps/ai-backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "processed"
    / "anomaly_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "splits"
    / "anomaly"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Healthy engines identified from the DRDO dataset.
#
# One healthy engine is used for training and the other
# healthy engine is used for validation.
NORMAL_TRAIN_ENGINE = "ENG-2023008"
NORMAL_VAL_ENGINE = "ENG-2023009"


# ============================================================
# VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame) -> None:
    """Validate the minimum structure required."""

    required_columns = [
        "engine_id",
        "fault_active_label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def validate_normal_engine(
    df: pd.DataFrame,
    engine_id: str,
) -> None:
    """
    Verify that the selected engine contains only normal samples.

    For anomaly training/validation, fault_active_label must be 0.
    """

    engine_df = df[df["engine_id"] == engine_id]

    if engine_df.empty:
        raise ValueError(
            f"Engine not found in dataset: {engine_id}"
        )

    fault_values = (
        engine_df["fault_active_label"]
        .dropna()
        .unique()
    )

    if not set(fault_values).issubset({0}):
        raise ValueError(
            f"{engine_id} contains fault-active samples. "
            f"Found fault_active_label values: {fault_values}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ANOMALY DETECTION DATASET SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    print("\nInput:")
    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n{INPUT_PATH}\n\n"
            "Run first:\n"
            "python training/features/build_features.py"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    validate_input(df)

    # --------------------------------------------------------
    # Engine discovery
    # --------------------------------------------------------

    engine_ids = set(
        df["engine_id"]
        .dropna()
        .unique()
    )

    print("\nEngines found:")
    for engine_id in sorted(engine_ids):
        engine_rows = df[
            df["engine_id"] == engine_id
        ]

        fault_count = int(
            engine_rows["fault_active_label"].sum()
        )

        print(
            f"  {engine_id}"
            f" → rows={len(engine_rows):,}"
            f", fault_samples={fault_count:,}"
        )

    # --------------------------------------------------------
    # Validate healthy engines
    # --------------------------------------------------------

    print("\nValidating healthy engines...")

    validate_normal_engine(
        df,
        NORMAL_TRAIN_ENGINE,
    )

    validate_normal_engine(
        df,
        NORMAL_VAL_ENGINE,
    )

    print(
        f"  {NORMAL_TRAIN_ENGINE} → healthy ✅"
    )

    print(
        f"  {NORMAL_VAL_ENGINE} → healthy ✅"
    )

    # --------------------------------------------------------
    # Build train / validation / test
    # --------------------------------------------------------

    train_df = df[
        df["engine_id"] == NORMAL_TRAIN_ENGINE
    ].copy()

    val_df = df[
        df["engine_id"] == NORMAL_VAL_ENGINE
    ].copy()

    faulty_engines = engine_ids - {
        NORMAL_TRAIN_ENGINE,
        NORMAL_VAL_ENGINE,
    }

    test_df = df[
        df["engine_id"].isin(faulty_engines)
    ].copy()

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if train_df.empty:
        raise RuntimeError(
            "Anomaly training dataset is empty."
        )

    if val_df.empty:
        raise RuntimeError(
            "Anomaly validation dataset is empty."
        )

    if test_df.empty:
        raise RuntimeError(
            "Anomaly test dataset is empty."
        )

    # Train must be completely normal.
    train_fault_count = int(
        train_df["fault_active_label"].sum()
    )

    if train_fault_count != 0:
        raise RuntimeError(
            "Training dataset contains fault-active samples."
        )

    # Validation must be completely normal.
    val_fault_count = int(
        val_df["fault_active_label"].sum()
    )

    if val_fault_count != 0:
        raise RuntimeError(
            "Validation dataset contains fault-active samples."
        )

    # Test must contain at least some anomalous samples.
    test_fault_count = int(
        test_df["fault_active_label"].sum()
    )

    if test_fault_count == 0:
        raise RuntimeError(
            "Test dataset contains no anomalous samples."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Engine isolation check
    # --------------------------------------------------------

    train_engines = set(
        train_df["engine_id"].unique()
    )

    val_engines = set(
        val_df["engine_id"].unique()
    )

    test_engines = set(
        test_df["engine_id"].unique()
    )

    if train_engines & val_engines:
        raise RuntimeError(
            "Engine overlap detected between TRAIN and VAL."
        )

    if train_engines & test_engines:
        raise RuntimeError(
            "Engine overlap detected between TRAIN and TEST."
        )

    if val_engines & test_engines:
        raise RuntimeError(
            "Engine overlap detected between VAL and TEST."
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ANOMALY SPLIT CREATED")
    print("-" * 70)

    print(
        f"\nTRAIN"
        f"\n  Engine: {NORMAL_TRAIN_ENGINE}"
        f"\n  Rows: {len(train_df):,}"
        f"\n  Fault samples: {train_fault_count:,}"
        f"\n  Output: {train_path}"
    )

    print(
        f"\nVAL"
        f"\n  Engine: {NORMAL_VAL_ENGINE}"
        f"\n  Rows: {len(val_df):,}"
        f"\n  Fault samples: {val_fault_count:,}"
        f"\n  Output: {val_path}"
    )

    print(
        f"\nTEST"
        f"\n  Engines: {sorted(test_engines)}"
        f"\n  Rows: {len(test_df):,}"
        f"\n  Fault samples: {test_fault_count:,}"
        f"\n  Output: {test_path}"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    print(
        f"\nTRAIN engines: {sorted(train_engines)}"
    )

    print(
        f"VAL engines:   {sorted(val_engines)}"
    )

    print(
        f"TEST engines:  {sorted(test_engines)}"
    )

    print("\n✅ TRAIN contains only healthy samples.")
    print("✅ VAL contains only healthy samples.")
    print("✅ TEST contains faulty-engine samples.")
    print("✅ No engine appears in multiple splits.")
    print(f"✅ Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()