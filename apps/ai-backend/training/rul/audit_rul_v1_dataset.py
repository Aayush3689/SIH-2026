from pathlib import Path
import pandas as pd
import numpy as np


INPUT_PATH = Path(
    "training/datasets/processed/rul_v1/rul_flight_level_v1.csv"
)

OUTPUT_DIR = Path(
    "training/datasets/processed/rul_v1/audit"
)

TARGET = "rul_hours_remaining"

EXCLUDED_FEATURES = {
    TARGET,
    "engine_id",
    "uav_id",
    "flight_id",
    "first_timestamp",
    "last_timestamp",
    "flight_phase",
    "mission_type",
    "environmental_profile",
    "engine_flight_number",
    "engine_total_flights",
}


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():

    section("RUL V1 DATASET AUDIT")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df):,}")

    # ------------------------------------------------------------------
    # BASIC VALIDATION
    # ------------------------------------------------------------------

    section("BASIC VALIDATION")

    print(f"Engines : {df['engine_id'].nunique()}")
    print(f"Flights : {df['flight_id'].nunique()}")

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    print(f"Target missing : {df[TARGET].isna().sum():,}")
    print(f"Target min     : {df[TARGET].min():.2f}")
    print(f"Target max     : {df[TARGET].max():.2f}")
    print(f"Target mean    : {df[TARGET].mean():.2f}")
    print(f"Target median  : {df[TARGET].median():.2f}")

    # ------------------------------------------------------------------
    # DUPLICATES
    # ------------------------------------------------------------------

    section("DUPLICATES")

    print(f"Duplicate full rows : {df.duplicated().sum():,}")

    duplicate_flights = df["flight_id"].duplicated().sum()

    print(
        f"Duplicate flight IDs : {duplicate_flights:,}"
    )

    if duplicate_flights > 0:
        raise ValueError("Each flight must appear exactly once.")

    # ------------------------------------------------------------------
    # TARGET DISTRIBUTION
    # ------------------------------------------------------------------

    section("TARGET DISTRIBUTION")

    quantiles = df[TARGET].quantile(
        [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    )

    for q, value in quantiles.items():
        print(f"Q{q * 100:05.1f} : {value:.2f}")

    print(
        f"\nZero RUL flights : {(df[TARGET] == 0).sum():,}"
    )

    # ------------------------------------------------------------------
    # ENGINE-WISE ANALYSIS
    # ------------------------------------------------------------------

    section("ENGINE-WISE RUL")

    engine_summary = (
        df.groupby("engine_id")[TARGET]
        .agg(
            flights="count",
            first_rul="first",
            last_rul="last",
            min_rul="min",
            max_rul="max",
            mean_rul="mean",
        )
        .reset_index()
    )

    print(engine_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # FEATURE TYPES
    # ------------------------------------------------------------------

    section("FEATURE TYPES")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    print(f"Numeric columns     : {len(numeric_columns)}")
    print(f"Non-numeric columns : {len(categorical_columns)}")

    print("\nNon-numeric columns:")
    for col in categorical_columns:
        print(f"  - {col}")

    # ------------------------------------------------------------------
    # MODEL CANDIDATES
    # ------------------------------------------------------------------

    feature_columns = [
        col
        for col in numeric_columns
        if col not in EXCLUDED_FEATURES
    ]

    section("MODEL FEATURE CANDIDATES")

    print(f"Candidate features : {len(feature_columns)}")

    for col in feature_columns:
        print(f"  - {col}")

    # ------------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------------

    section("MISSING VALUES")

    missing = (
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values in candidate features.")
    else:
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"{col:<45} {count:>6,} ({pct:6.2f}%)")

    # ------------------------------------------------------------------
    # CONSTANT FEATURES
    # ------------------------------------------------------------------

    section("CONSTANT FEATURES")

    constant_features = []

    for col in feature_columns:
        if df[col].nunique(dropna=False) <= 1:
            constant_features.append(col)

    if constant_features:
        for col in constant_features:
            print(f"  - {col}")
    else:
        print("No constant features found.")

    # ------------------------------------------------------------------
    # CORRELATION WITH TARGET
    # ------------------------------------------------------------------

    section("FEATURE ↔ RUL CORRELATION")

    correlations = []

    for col in feature_columns:

        series = df[col]

        if series.nunique(dropna=True) <= 1:
            continue

        corr = series.corr(df[TARGET])

        correlations.append(
            {
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
            }
        )

    corr_df = (
        pd.DataFrame(correlations)
        .sort_values(
            "abs_correlation",
            ascending=False
        )
    )

    print(
        corr_df[
            ["feature", "correlation"]
        ]
        .head(25)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # HIGHLY CORRELATED FEATURES
    # ------------------------------------------------------------------

    section("HIGH FEATURE CORRELATION")

    feature_corr = df[feature_columns].corr().abs()

    pairs = []

    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):

            a = feature_columns[i]
            b = feature_columns[j]

            value = feature_corr.loc[a, b]

            if value >= 0.98:
                pairs.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "abs_correlation": value,
                    }
                )

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values(
            "abs_correlation",
            ascending=False
        )
    )

    if len(pairs_df) == 0:
        print("No feature pairs with |corr| >= 0.98")
    else:
        print(
            pairs_df.head(30)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # ENGINE × FEATURE LEAKAGE CHECK
    # ------------------------------------------------------------------

    section("ENGINE-WISE FEATURE STABILITY")

    stability_rows = []

    for col in feature_columns:

        grouped = (
            df.groupby("engine_id")[col]
            .mean()
        )

        stability_rows.append(
            {
                "feature": col,
                "engine_mean_min": grouped.min(),
                "engine_mean_max": grouped.max(),
                "engine_mean_std": grouped.std(),
            }
        )

    stability_df = (
        pd.DataFrame(stability_rows)
        .sort_values(
            "engine_mean_std",
            ascending=False
        )
    )

    print(
        stability_df.head(20)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    corr_path = OUTPUT_DIR / "feature_target_correlation.csv"
    stability_path = OUTPUT_DIR / "engine_feature_stability.csv"
    pairs_path = OUTPUT_DIR / "high_feature_correlations.csv"
    summary_path = OUTPUT_DIR / "audit_summary.txt"

    corr_df.to_csv(
        corr_path,
        index=False
    )

    stability_df.to_csv(
        stability_path,
        index=False
    )

    pairs_df.to_csv(
        pairs_path,
        index=False
    )

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("RUL V1 DATASET AUDIT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Engines: {df['engine_id'].nunique()}\n")
        f.write(f"Flights: {df['flight_id'].nunique()}\n")
        f.write(f"Candidate features: {len(feature_columns)}\n")
        f.write(f"Missing target: {df[TARGET].isna().sum()}\n")
        f.write(
            f"Duplicate rows: {df.duplicated().sum()}\n"
        )
        f.write(
            f"Zero RUL flights: {(df[TARGET] == 0).sum()}\n"
        )

        f.write("\nExcluded features:\n")

        for col in sorted(EXCLUDED_FEATURES):
            f.write(f"- {col}\n")

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    section("OUTPUT")

    print(f"Saved:")
    print(f"  {corr_path}")
    print(f"  {stability_path}")
    print(f"  {pairs_path}")
    print(f"  {summary_path}")

    print("\nRUL V1 AUDIT COMPLETE.")


if __name__ == "__main__":
    main()