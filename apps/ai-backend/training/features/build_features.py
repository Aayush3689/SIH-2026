from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

# apps/ai-backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "training" / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "training" / "datasets" / "processed"

RAW_TELEMETRY = RAW_DIR / "dt_engine_telemetry_full.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

ID_COLUMNS = [
    "timestamp",
    "engine_id",
    "uav_id",
    "flight_id",
    "flight_index",
]

RAW_FEATURES = [
    "rpm",
    "throttle_position_pct",
    "cht_c",
    "egt_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
    "fuel_flow_lph",
    "fuel_pressure_kpa",
    "intake_manifold_pressure_kpa",
    "engine_vibration_mm_s",
    "battery_voltage_v",
    "alternator_output_a",
    "injection_timing_deg_btdc",
    "engine_load_pct",
    "ambient_temp_c",
    "planned_altitude_ft",
    "airspeed_kts",
    "elapsed_s",
]

TEMPORAL_FEATURES = [
    "vibration_mean_30s",
    "vibration_std_30s",
    "rpm_mean_30s",
    "rpm_std_30s",
    "egt_mean_60s",
    "egt_std_60s",
    "cht_mean_60s",
    "cht_std_60s",
    "oil_pressure_mean_60s",
    "oil_pressure_std_60s",
    "oil_temperature_mean_60s",
    "oil_temperature_std_60s",
    "egt_delta",
    "cht_delta",
    "oil_pressure_delta",
    "vibration_delta",
    "egt_rate",
    "cht_rate",
    "oil_pressure_rate",
    "vibration_rate",
]

PHYSICS_FEATURES = [
    "egt_expected",
    "egt_residual",
    "cht_expected",
    "cht_residual",
    "oil_temperature_expected",
    "oil_temperature_residual",
]

CONTEXT_FEATURES = [
    "mission_type",
    "environmental_profile",
    "flight_phase",
    "throttle_profile",
    "mission_priority",
    "takeoff_duration_min",
    "cruise_duration_min",
    "loiter_duration_min",
    "return_to_base_duration_min",
    "engine_model_variant",
    "total_cycles_at_start",
    "last_overhaul_days_ago",
    "last_major_maintenance_days_ago",
]

FAULT_TARGETS = [
    "fault_mode_ground_truth",
    "fault_active_label",
    "fault_severity_score",
    "label_misfire",
    "label_injector_abnormality",
    "label_coking_degradation",
    "label_lubrication_issue",
    "label_sensor_drift",
    "label_combustion_instability",
    "label_overheating_trend",
    "label_abnormal_vibration",
]

RUL_TARGETS = [
    "rul_hours_remaining",
]

REFERENCE_COLUMNS = [
    "health_index",
]


# ============================================================
# UTILITY
# ============================================================

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling, delta and rate-of-change features.

    Temporal calculations reset for every engine + flight,
    so one flight never leaks temporal information into another.
    """

    require_columns(
        df,
        [
            "timestamp",
            "engine_id",
            "flight_id",
            "elapsed_s",
            "rpm",
            "egt_c",
            "cht_c",
            "oil_pressure_kpa",
            "oil_temperature_c",
            "engine_vibration_mm_s",
        ],
    )

    df = df.sort_values(
        ["engine_id", "flight_id", "timestamp"]
    ).reset_index(drop=True)

    result = df.copy()

    # --------------------------------------------------------
    # Rolling windows
    # --------------------------------------------------------

    grouped = (
        result.set_index("timestamp")
        .groupby(["engine_id", "flight_id"], group_keys=False)
    )

    rolling_specs = [
        (
            "engine_vibration_mm_s",
            "vibration_mean_30s",
            "mean",
            "30s",
            1,
        ),
        (
            "engine_vibration_mm_s",
            "vibration_std_30s",
            "std",
            "30s",
            2,
        ),
        (
            "rpm",
            "rpm_mean_30s",
            "mean",
            "30s",
            1,
        ),
        (
            "rpm",
            "rpm_std_30s",
            "std",
            "30s",
            2,
        ),
        (
            "egt_c",
            "egt_mean_60s",
            "mean",
            "60s",
            1,
        ),
        (
            "egt_c",
            "egt_std_60s",
            "std",
            "60s",
            2,
        ),
        (
            "cht_c",
            "cht_mean_60s",
            "mean",
            "60s",
            1,
        ),
        (
            "cht_c",
            "cht_std_60s",
            "std",
            "60s",
            2,
        ),
        (
            "oil_pressure_kpa",
            "oil_pressure_mean_60s",
            "mean",
            "60s",
            1,
        ),
        (
            "oil_pressure_kpa",
            "oil_pressure_std_60s",
            "std",
            "60s",
            2,
        ),
        (
            "oil_temperature_c",
            "oil_temperature_mean_60s",
            "mean",
            "60s",
            1,
        ),
        (
            "oil_temperature_c",
            "oil_temperature_std_60s",
            "std",
            "60s",
            2,
        ),
    ]

    indexed = result.set_index("timestamp")

    for source_col, output_col, statistic, window, min_periods in rolling_specs:

        rolling = grouped[source_col].rolling(
            window,
            min_periods=min_periods,
        )

        if statistic == "mean":
            values = rolling.mean()
        else:
            values = rolling.std()

        values = values.reset_index(level=[0, 1], drop=True)

        indexed[output_col] = values

    result = indexed.reset_index()

    # --------------------------------------------------------
    # Sample-to-sample deltas
    # --------------------------------------------------------

    group = result.groupby(
        ["engine_id", "flight_id"],
        sort=False,
    )

    result["egt_delta"] = group["egt_c"].diff()
    result["cht_delta"] = group["cht_c"].diff()
    result["oil_pressure_delta"] = group["oil_pressure_kpa"].diff()
    result["vibration_delta"] = group["engine_vibration_mm_s"].diff()

    # --------------------------------------------------------
    # Rates per second
    # --------------------------------------------------------

    delta_time = group["elapsed_s"].diff()

    safe_delta_time = delta_time.replace(0, np.nan)

    result["egt_rate"] = (
        result["egt_delta"] / safe_delta_time
    )

    result["cht_rate"] = (
        result["cht_delta"] / safe_delta_time
    )

    result["oil_pressure_rate"] = (
        result["oil_pressure_delta"] / safe_delta_time
    )

    result["vibration_rate"] = (
        result["vibration_delta"] / safe_delta_time
    )

    return result


# ============================================================
# PHYSICS-INFORMED FEATURES
# ============================================================

def add_physics_features(
    df: pd.DataFrame,
    development_engines: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Build statistical expected-behaviour baselines for:

    - EGT
    - CHT
    - Oil temperature

    Residual = actual - expected

    This is a physics-informed baseline, not a full
    thermodynamic engine simulation.
    """

    physics_inputs = {
        "egt": [
            "rpm",
            "throttle_position_pct",
            "engine_load_pct",
            "fuel_flow_lph",
            "planned_altitude_ft",
            "intake_manifold_pressure_kpa",
            "ambient_temp_c",
            "airspeed_kts",
            "fuel_pressure_kpa",
            "injection_timing_deg_btdc",
        ],
        "cht": [
            "rpm",
            "throttle_position_pct",
            "engine_load_pct",
            "fuel_flow_lph",
            "planned_altitude_ft",
            "intake_manifold_pressure_kpa",
            "ambient_temp_c",
            "airspeed_kts",
            "fuel_pressure_kpa",
            "injection_timing_deg_btdc",
        ],
        "oil_temperature": [
            "rpm",
            "throttle_position_pct",
            "engine_load_pct",
            "fuel_flow_lph",
            "planned_altitude_ft",
            "intake_manifold_pressure_kpa",
            "ambient_temp_c",
            "airspeed_kts",
            "fuel_pressure_kpa",
            "injection_timing_deg_btdc",
        ],
    }

    targets = {
        "egt": "egt_c",
        "cht": "cht_c",
        "oil_temperature": "oil_temperature_c",
    }

    require_columns(
        df,
        list(
            set(
                column
                for columns in physics_inputs.values()
                for column in columns
            )
            | set(targets.values())
            | {"engine_id"}
        ),
    )

    train_mask = df["engine_id"].isin(development_engines)

    if train_mask.sum() == 0:
        raise ValueError(
            "No rows available for the physics development engines."
        )

    configs = {}

    for name, input_columns in physics_inputs.items():

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "regressor",
                    LinearRegression(),
                ),
            ]
        )

        model.fit(
            df.loc[train_mask, input_columns],
            df.loc[train_mask, targets[name]],
        )

        expected_column = f"{name}_expected"
        residual_column = f"{name}_residual"

        df[expected_column] = model.predict(
            df[input_columns]
        )

        df[residual_column] = (
            df[targets[name]]
            - df[expected_column]
        )

        artifact_path = (
            PROCESSED_DIR
            / f"physics_{name}_baseline.joblib"
        )

        joblib.dump(
            model,
            artifact_path,
        )

        configs[name] = {
            "inputs": input_columns,
            "target": targets[name],
            "expected_output": expected_column,
            "residual_output": residual_column,
            "artifact": str(artifact_path),
        }

    return df, configs


# ============================================================
# MODEL-SPECIFIC VIEWS
# ============================================================

def save_model_views(df: pd.DataFrame) -> None:
    """
    Generate the datasets used by the three ML tasks.

    These files are generated artifacts.
    They should NOT be manually edited.
    """

    # --------------------------------------------------------
    # Shared model input groups
    # --------------------------------------------------------

    common_sensor_features = RAW_FEATURES.copy()
    common_temporal_features = TEMPORAL_FEATURES.copy()
    common_physics_features = PHYSICS_FEATURES.copy()

    common_context = [
        "ambient_temp_c",
        "planned_altitude_ft",
        "airspeed_kts",
        "flight_phase",
        "mission_type",
        "environmental_profile",
        "engine_load_pct",
    ]

    # --------------------------------------------------------
    # Anomaly model
    # --------------------------------------------------------

    anomaly_features = (
        common_sensor_features
        + common_temporal_features
        + common_physics_features
        + common_context
    )

    # --------------------------------------------------------
    # Fault model
    # --------------------------------------------------------

    fault_features = (
        common_sensor_features
        + common_temporal_features
        + common_physics_features
        + common_context
    )

    # --------------------------------------------------------
    # RUL model
    # --------------------------------------------------------

    rul_features = (
        common_sensor_features
        + common_temporal_features
        + common_physics_features
        + common_context
        + [
            "total_cycles_at_start",
            "last_overhaul_days_ago",
            "last_major_maintenance_days_ago",
        ]
    )

    # Remove duplicates while preserving order.
    anomaly_features = list(dict.fromkeys(anomaly_features))
    fault_features = list(dict.fromkeys(fault_features))
    rul_features = list(dict.fromkeys(rul_features))

    # --------------------------------------------------------
    # Helper for saving
    # --------------------------------------------------------

    def save_view(
        filename: str,
        feature_columns: list[str],
        target_columns: list[str],
    ) -> None:

        columns = (
            ID_COLUMNS
            + feature_columns
            + target_columns
        )

        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        output_path = PROCESSED_DIR / filename

        df[available_columns].to_csv(
            output_path,
            index=False,
        )

        print(
            f"Generated: {output_path}"
            f"  ({len(df):,} rows, {len(available_columns)} columns)"
        )

    # --------------------------------------------------------
    # Save model datasets
    # --------------------------------------------------------

    save_view(
        "anomaly_features.csv",
        anomaly_features,
        [
            "fault_active_label",
        ],
    )

    save_view(
        "fault_features.csv",
        fault_features,
        FAULT_TARGETS,
    )

    save_view(
        "rul_features.csv",
        rul_features,
        RUL_TARGETS,
    )

    # --------------------------------------------------------
    # Save complete consolidated feature dataset
    # --------------------------------------------------------

    all_features = list(
        dict.fromkeys(
            RAW_FEATURES
            + TEMPORAL_FEATURES
            + PHYSICS_FEATURES
            + CONTEXT_FEATURES
        )
    )

    final_columns = (
        ID_COLUMNS
        + all_features
        + FAULT_TARGETS
        + RUL_TARGETS
        + REFERENCE_COLUMNS
    )

    final_columns = [
        column
        for column in final_columns
        if column in df.columns
    ]

    final_path = PROCESSED_DIR / "final_features.csv"

    df[final_columns].to_csv(
        final_path,
        index=False,
    )

    print(
        f"Generated: {final_path}"
        f"  ({len(df):,} rows, {len(final_columns)} columns)"
    )

    return {
        "anomaly": anomaly_features,
        "fault": fault_features,
        "rul": rul_features,
        "all_features": all_features,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw telemetry
    # --------------------------------------------------------

    if not RAW_TELEMETRY.exists():
        raise FileNotFoundError(
            f"\nRaw telemetry file not found:\n"
            f"{RAW_TELEMETRY}\n\n"
            "Make sure the DRDO CSV is inside:\n"
            "training/datasets/raw/"
        )

    print(f"\nLoading:\n{RAW_TELEMETRY}")

    df = pd.read_csv(
        RAW_TELEMETRY,
        parse_dates=["timestamp"],
    )

    if df.empty:
        raise ValueError("Telemetry dataset is empty.")

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # Validate basic structure
    # --------------------------------------------------------

    require_columns(
        df,
        [
            "timestamp",
            "engine_id",
            "flight_id",
        ],
    )

    engines = sorted(
        df["engine_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(engines) < 3:
        raise ValueError(
            "At least 3 engines are required."
        )

    print(f"Engines found: {len(engines)}")
    print(f"Engine IDs: {engines}")

    # --------------------------------------------------------
    # Development engines for physics baseline
    # --------------------------------------------------------
    #
    # For the initial feature-generation baseline we keep the
    # same development-engine policy used in our analysis.
    #
    # IMPORTANT:
    # Final model evaluation should refit the physics baseline
    # inside each train fold instead of using this fixed model.
    # --------------------------------------------------------

    development_engines = engines[: max(1, len(engines) - 3)]

    print(
        "\nPhysics development engines:"
        f"\n{development_engines}"
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    print("\nAdding temporal features...")

    df = add_temporal_features(df)

    print("Temporal features added.")

    # --------------------------------------------------------
    # Physics features
    # --------------------------------------------------------

    print("\nBuilding physics-informed features...")

    df, physics_config = add_physics_features(
        df,
        development_engines,
    )

    print("Physics-informed features added.")

    # --------------------------------------------------------
    # Model-specific datasets
    # --------------------------------------------------------

    print("\nGenerating model-specific datasets...")

    model_views = save_model_views(df)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "rows": int(len(df)),
        "engines": engines,
        "development_engines_for_initial_physics_baseline": (
            development_engines
        ),
        "temporal_reset_keys": [
            "engine_id",
            "flight_id",
        ],
        "physics_features_kept": [
            "egt_residual",
            "cht_residual",
            "oil_temperature_residual",
        ],
        "physics_feature_rejected": [
            "oil_pressure_residual",
        ],
        "model_views": {
            "anomaly": {
                "file": "anomaly_features.csv",
                "target": [
                    "fault_active_label",
                ],
            },
            "fault": {
                "file": "fault_features.csv",
                "target": FAULT_TARGETS,
            },
            "rul": {
                "file": "rul_features.csv",
                "target": RUL_TARGETS,
            },
        },
        "reference_columns_excluded_from_inputs": [
            "health_index",
        ],
        "notes": [
            (
                "Feature CSVs are generated artifacts and "
                "should be recreated by this script."
            ),
            (
                "Physics baseline is a statistical "
                "physics-informed expected-behaviour baseline, "
                "not a full thermodynamic engine simulation."
            ),
            (
                "Final model evaluation should refit the "
                "physics baseline inside each train fold."
            ),
        ],
        "physics_models": physics_config,
        "feature_counts": {
            "anomaly": len(model_views["anomaly"]),
            "fault": len(model_views["fault"]),
            "rul": len(model_views["rul"]),
            "all_features": len(model_views["all_features"]),
        },
    }

    manifest_path = (
        PROCESSED_DIR / "feature_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"\nOutput directory:")
    print(PROCESSED_DIR)

    print("\nGenerated files:")
    print("  - final_features.csv")
    print("  - anomaly_features.csv")
    print("  - fault_features.csv")
    print("  - rul_features.csv")
    print("  - feature_manifest.json")
    print("  - physics_*.joblib")

    print("\n✅ Raw dataset was not modified.")
    print("✅ Feature datasets were generated automatically.")
    print("✅ Ready for make_splits.py")


if __name__ == "__main__":
    main()