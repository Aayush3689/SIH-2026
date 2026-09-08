from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "training/datasets/processed/rul_v1/rul_flight_level_v1.csv"
)

OUTPUT_DIR = Path(
    "training/datasets/processed/rul_v1/diagnosis"
)

TARGET = "rul_hours_remaining"


# ============================================================
# HELPERS
# ============================================================

def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_corr(a: pd.Series, b: pd.Series):
    if a.nunique(dropna=True) <= 1:
        return np.nan

    if b.nunique(dropna=True) <= 1:
        return np.nan

    return a.corr(b)


# ============================================================
# MAIN
# ============================================================

def main():

    section("RUL V2 DIAGNOSTIC ANALYSIS")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

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
    # BASIC
    # ========================================================

    section("BASIC DATASET")

    print(f"Rows       : {len(df):,}")
    print(f"Engines    : {df['engine_id'].nunique()}")
    print(f"Flights    : {df['flight_id'].nunique()}")
    print(f"RUL min    : {df[TARGET].min():.2f} h")
    print(f"RUL max    : {df[TARGET].max():.2f} h")
    print(f"RUL mean   : {df[TARGET].mean():.2f} h")
    print(f"RUL median : {df[TARGET].median():.2f} h")

    # ========================================================
    # ENGINE FLIGHT NUMBER
    # ========================================================

    section("RUL VS ENGINE FLIGHT NUMBER")

    df["engine_flight_number"] = (
        df.groupby("engine_id")
        .cumcount()
        + 1
    )

    flight_number_summary = (
        df.groupby("engine_flight_number")[TARGET]
        .agg(
            samples="count",
            mean_rul="mean",
            median_rul="median",
            min_rul="min",
            max_rul="max",
        )
        .reset_index()
    )

    print(
        flight_number_summary.to_string(
            index=False
        )
    )

    corr_flight_number = safe_corr(
        df["engine_flight_number"],
        df[TARGET]
    )

    print(
        f"\nCorrelation(engine_flight_number, RUL): "
        f"{corr_flight_number:.4f}"
    )

    # ========================================================
    # GLOBAL TIMELINE
    # ========================================================

    section("RUL VS GLOBAL TIME")

    first_time = df["flight_start"].min()

    df["elapsed_days_global"] = (
        df["flight_start"] - first_time
    ).dt.total_seconds() / 86400.0

    corr_global_time = safe_corr(
        df["elapsed_days_global"],
        df[TARGET]
    )

    print(
        f"Correlation(global elapsed days, RUL): "
        f"{corr_global_time:.4f}"
    )

    # ========================================================
    # ENGINE-WISE TRAJECTORIES
    # ========================================================

    section("ENGINE-WISE RUL TRAJECTORIES")

    trajectory_rows = []

    for engine_id, group in df.groupby(
        "engine_id",
        sort=True
    ):

        group = group.sort_values(
            ["flight_start", "flight_id"]
        ).reset_index(drop=True)

        rul = group[TARGET].to_numpy()

        deltas = np.diff(rul)

        increases = int(
            (deltas > 0).sum()
        )

        decreases = int(
            (deltas < 0).sum()
        )

        unchanged = int(
            (deltas == 0).sum()
        )

        trajectory_rows.append(
            {
                "engine_id": engine_id,
                "flights": len(group),
                "first_rul": rul[0],
                "last_rul": rul[-1],
                "min_rul": rul.min(),
                "max_rul": rul.max(),
                "mean_rul": rul.mean(),
                "increases": increases,
                "decreases": decreases,
                "unchanged": unchanged,
                "net_change": rul[-1] - rul[0],
            }
        )

        print(
            f"{engine_id}: "
            f"first={rul[0]:6.2f}  "
            f"last={rul[-1]:6.2f}  "
            f"min={rul.min():6.2f}  "
            f"max={rul.max():6.2f}  "
            f"inc={increases:2d}  "
            f"dec={decreases:2d}"
        )

    trajectory_df = pd.DataFrame(
        trajectory_rows
    )

    # ========================================================
    # FLIGHT-TO-FLIGHT RUL CHANGE
    # ========================================================

    section("RUL CHANGE BETWEEN FLIGHTS")

    df["rul_delta"] = (
        df.groupby("engine_id")[TARGET]
        .diff()
    )

    delta = df["rul_delta"].dropna()

    print(
        f"Flight transitions : {len(delta)}"
    )

    print(
        f"Mean RUL delta     : {delta.mean():.2f} h"
    )

    print(
        f"Median RUL delta   : {delta.median():.2f} h"
    )

    print(
        f"Increase count     : {(delta > 0).sum()}"
    )

    print(
        f"Decrease count     : {(delta < 0).sum()}"
    )

    print(
        f"Unchanged count    : {(delta == 0).sum()}"
    )

    # ========================================================
    # RUL BINS
    # ========================================================

    section("RUL BINS")

    bins = [
        -0.01,
        5,
        10,
        20,
        40,
        60,
        np.inf,
    ]

    labels = [
        "0-5h",
        "5-10h",
        "10-20h",
        "20-40h",
        "40-60h",
        "60h+",
    ]

    df["rul_bin"] = pd.cut(
        df[TARGET],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rul_bin_summary = (
        df.groupby(
            "rul_bin",
            observed=False
        )
        .agg(
            samples=(TARGET, "count"),
            mean_rul=(TARGET, "mean"),
        )
        .reset_index()
    )

    rul_bin_summary["percentage"] = (
        rul_bin_summary["samples"]
        / len(df)
        * 100
    )

    print(
        rul_bin_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # LOW-RUL VS HIGH-RUL FEATURE ANALYSIS
    # ========================================================

    section("LOW-RUL VS HIGH-RUL FEATURE SEPARATION")

    excluded = {
        TARGET,
        "engine_id",
        "uav_id",
        "flight_id",
        "flight_start",
        "flight_end",
        "rul_delta",
        "rul_bin",
        "engine_flight_number",
        "elapsed_days_global",
    }

    numeric_features = [
        col
        for col in df.select_dtypes(
            include=[np.number]
        ).columns
        if col not in excluded
    ]

    low_mask = df[TARGET] <= 10
    high_mask = df[TARGET] >= 30

    print(
        f"Low-RUL samples (<=10 h): "
        f"{low_mask.sum()}"
    )

    print(
        f"High-RUL samples (>=30 h): "
        f"{high_mask.sum()}"
    )

    separation_rows = []

    for feature in numeric_features:

        low = df.loc[
            low_mask,
            feature
        ].dropna()

        high = df.loc[
            high_mask,
            feature
        ].dropna()

        if len(low) == 0 or len(high) == 0:
            continue

        low_mean = low.mean()
        high_mean = high.mean()

        pooled_std = np.sqrt(
            (
                low.var(ddof=1)
                + high.var(ddof=1)
            ) / 2
        )

        if pooled_std > 0:
            effect = (
                high_mean - low_mean
            ) / pooled_std
        else:
            effect = 0.0

        separation_rows.append(
            {
                "feature": feature,
                "low_rul_mean": low_mean,
                "high_rul_mean": high_mean,
                "effect_size": effect,
                "abs_effect_size": abs(effect),
            }
        )

    separation_df = (
        pd.DataFrame(separation_rows)
        .sort_values(
            "abs_effect_size",
            ascending=False
        )
    )

    print(
        separation_df.head(25).to_string(
            index=False
        )
    )

    # ========================================================
    # FEATURE ↔ RUL CORRELATION
    # ========================================================

    section("FEATURE CORRELATION WITH RUL")

    correlation_rows = []

    for feature in numeric_features:

        corr = safe_corr(
            df[feature],
            df[TARGET]
        )

        if pd.isna(corr):
            continue

        correlation_rows.append(
            {
                "feature": feature,
                "correlation": corr,
                "abs_correlation": abs(corr),
            }
        )

    correlation_df = (
        pd.DataFrame(correlation_rows)
        .sort_values(
            "abs_correlation",
            ascending=False
        )
    )

    print(
        correlation_df.head(30).to_string(
            index=False
        )
    )

    # ========================================================
    # TRAIN / VAL / TEST DISTRIBUTION
    # ========================================================

    section("V1 SPLIT DISTRIBUTION")

    split_dir = Path(
        "training/datasets/processed/rul_v1/split"
    )

    split_files = {
        "TRAIN": split_dir / "rul_train_v1.csv",
        "VALIDATION": split_dir / "rul_validation_v1.csv",
        "TEST": split_dir / "rul_test_v1.csv",
    }

    split_rows = []

    for split_name, path in split_files.items():

        if not path.exists():
            print(
                f"{split_name}: file not found"
            )
            continue

        split_df = pd.read_csv(
            path
        )

        values = split_df[TARGET].astype(float)

        split_rows.append(
            {
                "split": split_name,
                "samples": len(split_df),
                "min_rul": values.min(),
                "max_rul": values.max(),
                "mean_rul": values.mean(),
                "median_rul": values.median(),
                "zero_rul": int(
                    (values == 0).sum()
                ),
                "rul_le_10": int(
                    (values <= 10).sum()
                ),
                "rul_ge_30": int(
                    (values >= 30).sum()
                ),
            }
        )

    split_distribution_df = pd.DataFrame(
        split_rows
    )

    print(
        split_distribution_df.to_string(
            index=False
        )
    )

    # ========================================================
    # TEST COVERAGE VS TRAIN RANGE
    # ========================================================

    section("TRAIN → TEST RANGE CHECK")

    train = pd.read_csv(
        split_files["TRAIN"]
    )

    test = pd.read_csv(
        split_files["TEST"]
    )

    train_min = train[TARGET].min()
    train_max = train[TARGET].max()

    test_min = test[TARGET].min()
    test_max = test[TARGET].max()

    print(
        f"Train RUL range : "
        f"{train_min:.2f} → {train_max:.2f} h"
    )

    print(
        f"Test RUL range  : "
        f"{test_min:.2f} → {test_max:.2f} h"
    )

    test_below_train_median = (
        test[TARGET]
        < train[TARGET].median()
    ).sum()

    print(
        f"Test samples below train median: "
        f"{test_below_train_median}/{len(test)}"
    )

    # ========================================================
    # BASELINE 1 — TRAIN-MEAN PREDICTOR
    # ========================================================

    section("NAIVE BASELINES")

    train_mean = train[TARGET].mean()
    train_median = train[TARGET].median()

    y_test = test[TARGET].to_numpy()

    mean_mae = np.mean(
        np.abs(
            y_test - train_mean
        )
    )

    median_mae = np.mean(
        np.abs(
            y_test - train_median
        )
    )

    print(
        f"Train-mean predictor MAE   : "
        f"{mean_mae:.4f} h"
    )

    print(
        f"Train-median predictor MAE : "
        f"{median_mae:.4f} h"
    )

    # ========================================================
    # SIMPLE FLIGHT-SEQUENCE BASELINE
    # ========================================================

    section(
        "FLIGHT-SEQUENCE BASELINE"
    )

    train_engine = (
        train.groupby("engine_id")[TARGET]
        .agg(
            first="first",
            last="last",
        )
    )

    sequence_rows = []

    for engine_id, test_group in (
        test.groupby("engine_id")
    ):

        train_engine_rows = train[
            train["engine_id"] == engine_id
        ].sort_values(
            ["flight_start", "flight_id"]
        )

        if len(train_engine_rows) == 0:
            continue

        last_train_rul = (
            train_engine_rows[TARGET].iloc[-1]
        )

        actual = (
            test_group[TARGET].to_numpy()
        )

        prediction = np.full(
            len(actual),
            last_train_rul
        )

        mae = np.mean(
            np.abs(
                actual - prediction
            )
        )

        sequence_rows.append(
            {
                "engine_id": engine_id,
                "last_train_rul": last_train_rul,
                "test_samples": len(actual),
                "mae": mae,
            }
        )

        print(
            f"{engine_id}: "
            f"last_train_RUL={last_train_rul:.2f} h  "
            f"MAE={mae:.2f} h"
        )

    sequence_df = pd.DataFrame(
        sequence_rows
    )

    if len(sequence_df) > 0:

        print(
            f"\nAverage engine-sequence "
            f"baseline MAE: "
            f"{sequence_df['mae'].mean():.4f} h"
        )

    # ========================================================
    # DIAGNOSTIC CONCLUSION
    # ========================================================

    section("AUTOMATIC DIAGNOSTIC SUMMARY")

    corr_abs_sorted = (
        correlation_df
        .head(10)
    )

    print(
        "\nTop RUL-related features:"
    )

    for _, row in corr_abs_sorted.iterrows():

        print(
            f"  {row['feature']:<45} "
            f"corr={row['correlation']:+.4f}"
        )

    print(
        "\nKey observations:"
    )

    print(
        "1. Check how strongly RUL follows engine flight sequence."
    )

    print(
        "2. Check whether low-RUL and high-RUL states have "
        "clear health-feature separation."
    )

    print(
        "3. Compare the naive baselines against the V1 model."
    )

    print(
        "4. If health features poorly separate low/high RUL, "
        "the target may not be sufficiently health-driven."
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trajectory_path = (
        OUTPUT_DIR
        / "engine_rul_trajectories.csv"
    )

    flight_number_path = (
        OUTPUT_DIR
        / "rul_by_engine_flight_number.csv"
    )

    separation_path = (
        OUTPUT_DIR
        / "low_high_rul_feature_separation.csv"
    )

    correlation_path = (
        OUTPUT_DIR
        / "feature_rul_correlation.csv"
    )

    split_distribution_path = (
        OUTPUT_DIR
        / "rul_split_distribution.csv"
    )

    sequence_baseline_path = (
        OUTPUT_DIR
        / "flight_sequence_baseline.csv"
    )

    trajectory_df.to_csv(
        trajectory_path,
        index=False
    )

    flight_number_summary.to_csv(
        flight_number_path,
        index=False
    )

    separation_df.to_csv(
        separation_path,
        index=False
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    split_distribution_df.to_csv(
        split_distribution_path,
        index=False
    )

    sequence_df.to_csv(
        sequence_baseline_path,
        index=False
    )

    print("\nSaved diagnosis files:")
    print(f"  {trajectory_path}")
    print(f"  {flight_number_path}")
    print(f"  {separation_path}")
    print(f"  {correlation_path}")
    print(f"  {split_distribution_path}")
    print(f"  {sequence_baseline_path}")

    print("\nRUL V2 DIAGNOSTIC COMPLETE.")


if __name__ == "__main__":
    main()