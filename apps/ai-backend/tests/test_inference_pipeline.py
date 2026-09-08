from __future__ import annotations

import pandas as pd

from app.features.telemetry import (
    normalize_telemetry,
    validate_sample_count,
)

from app.features.runtime_features import (
    build_temporal_features,
)

from app.features.physics import (
    calculate_physics_features,
)

from app.features.fault_features import (
    add_fault_v4_features,
)

from app.features.rul_features import (
    RULFeatureBuilder,
)


DATASET = (
    "training/datasets/raw/dt_engine_telemetry_full.csv"
)


def main() -> None:

    print("=" * 70)
    print("AI BACKEND FEATURE PIPELINE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load telemetry
    # ------------------------------------------------------------------

    print("\n[1] Loading telemetry...")

    raw = pd.read_csv(
        DATASET
    )

    print(
        f"Rows    : {len(raw):,}"
    )

    print(
        f"Columns : {len(raw.columns)}"
    )

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------

    print("\n[2] Normalizing telemetry...")

    df = normalize_telemetry(
        raw
    )

    print(
        f"Normalized rows: {len(df):,}"
    )

    # ------------------------------------------------------------------
    # 3. Runtime-required fields
    # ------------------------------------------------------------------

    required = [
        "engine_id",
        "flight_id",
        "elapsed_s",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing runtime fields: {missing}"
        )

    print(
        "Required runtime fields: OK"
    )

    # ------------------------------------------------------------------
    # 4. Temporal features
    # ------------------------------------------------------------------

    print("\n[3] Building temporal features...")

    df = build_temporal_features(
        df
    )

    temporal_features = [
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

    missing = [
        col
        for col in temporal_features
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing temporal features: {missing}"
        )

    print(
        f"Temporal features: {len(temporal_features)} -> OK"
    )

    # ------------------------------------------------------------------
    # 5. Physics features
    # ------------------------------------------------------------------

    print("\n[4] Building physics features...")

    df = calculate_physics_features(
        df
    )

    physics_features = [
        "egt_expected",
        "egt_residual",
        "cht_expected",
        "cht_residual",
        "oil_temperature_expected",
        "oil_temperature_residual",
    ]

    missing = [
        col
        for col in physics_features
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing physics features: {missing}"
        )

    print(
        f"Physics features: {len(physics_features)} -> OK"
    )

    # ------------------------------------------------------------------
    # 6. Fault V4 features
    # ------------------------------------------------------------------

    print("\n[5] Building Fault V4 features...")

    df = add_fault_v4_features(
        df
    )

    fault_v4_features = [
        "egt_residual_mean_5",
        "cht_residual_mean_5",
        "oil_temperature_residual_mean_5",
        "egt_signed_residual_mean_5",
        "cht_signed_residual_mean_5",
        "oil_temperature_signed_residual_mean_5",
        "thermal_abs_residual_mean_5",
        "thermal_residual_product",
        "thermal_residual_same_sign",
        "fuel_pressure_mean_5",
        "fuel_pressure_std_5",
        "vibration_rate_mean_5",
        "vibration_rate_std_5",
    ]

    missing = [
        col
        for col in fault_v4_features
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing Fault V4 features: {missing}"
        )

    print(
        f"Fault V4 derived features: "
        f"{len(fault_v4_features)} -> OK"
    )

    # ------------------------------------------------------------------
    # 7. Current engine / flight
    # ------------------------------------------------------------------

    print("\n[6] Checking current flight...")

    latest_engine = (
        df["engine_id"].iloc[-1]
    )

    latest_flight = (
        df["flight_id"].iloc[-1]
    )

    current_flight = df[
        (df["engine_id"] == latest_engine)
        & (df["flight_id"] == latest_flight)
    ].copy()

    print(
        f"Engine : {latest_engine}"
    )

    print(
        f"Flight : {latest_flight}"
    )

    print(
        f"Samples: {len(current_flight)}"
    )

    # ------------------------------------------------------------------
    # 8. RUL window
    # ------------------------------------------------------------------

    print("\n[7] Checking RUL 8-sample window...")

    validate_sample_count(
        current_flight,
        minimum_samples=8,
    )

    rul_builder = RULFeatureBuilder(
        window_samples=8
    )

    print(
        "RUL window: 8 samples -> OK"
    )

    # ------------------------------------------------------------------
    # 9. Basic numeric sanity
    # ------------------------------------------------------------------

    print("\n[8] Numeric sanity check...")

    check_columns = (
        temporal_features
        + physics_features
        + fault_v4_features
    )

    infinite_values = 0

    for column in check_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        infinite_values += int(
            values.isin(
                [float("inf"), float("-inf")]
            ).sum()
        )

    if infinite_values:
        raise RuntimeError(
            f"Infinite values detected: "
            f"{infinite_values}"
        )

    print(
        "Infinite values: 0 -> OK"
    )

    # ------------------------------------------------------------------
    # 10. Latest feature snapshot
    # ------------------------------------------------------------------

    latest = df.tail(
        1
    )

    print("\n[9] Latest feature snapshot")

    print(
        latest[
            [
                "engine_id",
                "flight_id",
                "rpm",
                "egt_c",
                "cht_c",
                "oil_pressure_kpa",
                "oil_temperature_c",
                "egt_expected",
                "egt_residual",
                "cht_expected",
                "cht_residual",
                "oil_temperature_expected",
                "oil_temperature_residual",
                "egt_residual_mean_5",
                "cht_residual_mean_5",
                "fuel_pressure_mean_5",
                "vibration_rate_mean_5",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE PIPELINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()